import argparse
import json
import os
import re
import sys
from glob import glob
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.llm_integration.online_judge_llm import OnlineJudgeLLM
from src.llm_integration.prompt_templates import build_judge_scoring_prompt


def _extract_json_object(text: str) -> Dict[str, Any]:
    """从评委模型输出文本中提取JSON对象。"""
    text = (text or "").strip()
    if not text:
        raise ValueError("Judge model returned empty response.")

    # 1) 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) 兼容 ```json ... ``` 包裹
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    # 3) 提取第一个大括号对象
    brace = re.search(r"(\{[\s\S]*\})", text)
    if brace:
        return json.loads(brace.group(1))

    raise ValueError(f"Cannot parse JSON from judge response: {text[:200]}")

def _validate_scores(judge_json: Dict[str, Any]) -> None:
    """校验评分JSON结构。"""
    if "scores" not in judge_json or "winner" not in judge_json or "reason" not in judge_json:
        raise ValueError("Judge JSON missing required keys: scores/winner/reason")

    scores = judge_json["scores"]
    if "A" not in scores or "B" not in scores:
        raise ValueError("Judge JSON scores must contain A and B")

    required_dims = ["accuracy", "completeness", "clarity", "relevance", "total_score"]
    for side in ("A", "B"):
        for key in required_dims:
            if key not in scores[side]:
                raise ValueError(f"Judge JSON missing {side}.{key}")


def evaluate_single(
    judge_llm: OnlineJudgeLLM,
    question: str,
    answer_a: str,
    answer_b: str,
) -> Dict[str, Any]:
    """评测单题对比并返回结构化结果。"""
    prompt = build_judge_scoring_prompt(
        question=question,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    raw_response = judge_llm.generate(prompt)
    judge_json = _extract_json_object(raw_response)
    _validate_scores(judge_json)
    return judge_json


def evaluate_file(
    input_json_path: str,
    output_json_path: Optional[str] = None,
    judge_llm: Optional[OnlineJudgeLLM] = None,
) -> Dict[str, Any]:
    """评测一整个对比结果文件。"""
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(
            f"Input JSON not found: {input_json_path}. "
            "Please pass --input explicitly or generate rag_vs_non_rag JSON first."
        )

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of records.")

    judge_llm = judge_llm or OnlineJudgeLLM()

    evaluations: List[Dict[str, Any]] = []
    winner_count = {"A": 0, "B": 0, "Tie": 0, "Invalid": 0}

    for idx, qa in enumerate(data, start=1):
        question = qa["question"]
        answer_a = qa["A"]
        answer_b = qa["B"]

        item: Dict[str, Any] = {
            "index": idx,
            "question": question,
            "A": answer_a,
            "B": answer_b,
        }

        if not question or (not answer_a and not answer_b):
            item["judge_result"] = None
            item["error"] = "Invalid record: missing question or both answers are empty."
            winner_count["Invalid"] += 1
            evaluations.append(item)
            continue

        try:
            judge_result = evaluate_single(
                judge_llm=judge_llm,
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
            )
            winner = judge_result.get("winner", "Invalid")
            if winner not in winner_count:
                winner = "Invalid"
            winner_count[winner] += 1

            item["judge_result"] = judge_result
            item["error"] = None
        except Exception as exc:  # pragma: no cover - 外部LLM调用相关
            winner_count["Invalid"] += 1
            item["judge_result"] = None
            item["error"] = str(exc)

        evaluations.append(item)

    summary = {
        "total_questions": len(data),
        "winner_count": winner_count,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": os.path.abspath(input_json_path),
    }

    result = {
        "summary": summary,
        "evaluations": evaluations,
    }

    if output_json_path is None:
        stem = os.path.splitext(os.path.basename(input_json_path))[0]
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_json_path = os.path.join(
            os.path.dirname(input_json_path),
            f"{stem}_judged_{ts}.json",
        )

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return {
        "output_json_path": os.path.abspath(output_json_path),
        "summary": summary,
    }


def main() -> None:
    config_path = os.path.join(PROJECT_ROOT, "config", "configs.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    default_output_dir = (
        config.get("paths", {}).get("output_dir")
        or os.path.join(PROJECT_ROOT, "outputs")
    )
    default_input_json_path = os.path.join(default_output_dir, "evaluation_results", "result.json")
    default_output_path = os.path.join(default_output_dir, "evaluation_results", "scoring.json")

    # 若默认 result.json 不存在，则自动回退到 evaluation_results 中最新的对比文件
    if not os.path.exists(default_input_json_path):
        candidate_pattern = os.path.join(
            default_output_dir, "evaluation_results", "rag_vs_non_rag_*.json"
        )
        candidates = glob(candidate_pattern)
        if candidates:
            candidates.sort(key=os.path.getmtime, reverse=True)
            default_input_json_path = candidates[0]
            default_output_path = os.path.join(
                os.path.dirname(default_input_json_path),
                f"{os.path.splitext(os.path.basename(default_input_json_path))[0]}_scoring.json",
            )

    parser = argparse.ArgumentParser(description="Evaluate A/B answers with a judge LLM.")
    parser.add_argument(
        "--input",
        default=default_input_json_path,
        help=f"Path to A/B comparison JSON file. Default: {default_input_json_path}",
    )
    parser.add_argument(
        "--output",
        default=default_output_path,
        help=f"Path to output judged JSON file. Default: {default_output_path}",
    )
    args = parser.parse_args()

    result = evaluate_file(
        input_json_path=args.input,
        output_json_path=args.output,
    )
    print("✅ 评测完成")
    print(f"输出文件: {result['output_json_path']}")
    print(f"统计摘要: {json.dumps(result['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()