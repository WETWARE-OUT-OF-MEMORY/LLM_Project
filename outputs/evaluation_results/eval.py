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

from src.llm_integration.prompt_templates import build_judge_scoring_prompt


def _safe_percent(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def _mean_number(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def analyze_topk_result(
    topk: int,
    evaluation_results_dir: str = os.path.dirname(os.path.abspath(__file__)),
) -> Dict[str, Any]:
    """
    分析指定 topk 的 result.json / scoring.json。

    参数:
        topk: 例如 1/3/5/8
        evaluation_results_dir: outputs/evaluation_results 目录路径
    """
    topk_dir = os.path.join(evaluation_results_dir, f"topk_{topk}")
    result_path = os.path.join(topk_dir, "result.json")
    scoring_path = os.path.join(topk_dir, "scoring.json")

    if not os.path.isdir(topk_dir):
        raise FileNotFoundError(f"Topk directory not found: {topk_dir}")
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"result.json not found: {result_path}")
    if not os.path.exists(scoring_path):
        raise FileNotFoundError(f"scoring.json not found: {scoring_path}")

    with open(result_path, "r", encoding="utf-8") as f:
        result_data = json.load(f)
    with open(scoring_path, "r", encoding="utf-8") as f:
        scoring_data = json.load(f)

    if not isinstance(result_data, list):
        raise ValueError(f"Invalid result.json format in {result_path}: expect list.")
    evaluations = scoring_data.get("evaluations", [])
    summary = scoring_data.get("summary", {})
    winner_count = summary.get("winner_count", {})

    # 基础统计
    result_total = len(result_data)
    scoring_total = int(summary.get("total_questions", len(evaluations) if isinstance(evaluations, list) else 0))
    count_a = int(winner_count.get("A", 0))
    count_b = int(winner_count.get("B", 0))
    count_tie = int(winner_count.get("Tie", 0))
    count_invalid = int(winner_count.get("Invalid", 0))

    valid_total = max(scoring_total - count_invalid, 0)

    # 评分统计（基于 scoring.json 中的 judge_result）
    a_total_scores: List[float] = []
    b_total_scores: List[float] = []
    failed_count = 0
    if isinstance(evaluations, list):
        for item in evaluations:
            if not isinstance(item, dict):
                continue
            if item.get("error"):
                failed_count += 1
            judge_result = item.get("judge_result")
            if not isinstance(judge_result, dict):
                continue
            scores = judge_result.get("scores", {})
            if not isinstance(scores, dict):
                continue
            score_a = (scores.get("A") or {}).get("total_score")
            score_b = (scores.get("B") or {}).get("total_score")
            if isinstance(score_a, (int, float)):
                a_total_scores.append(float(score_a))
            if isinstance(score_b, (int, float)):
                b_total_scores.append(float(score_b))

    avg_a_score = _mean_number(a_total_scores)
    avg_b_score = _mean_number(b_total_scores)
    avg_score_gap_a_minus_b = (
        round(avg_a_score - avg_b_score, 4)
        if isinstance(avg_a_score, (int, float)) and isinstance(avg_b_score, (int, float))
        else None
    )

    return {
        "topk": int(topk),
        "paths": {
            "topk_dir": os.path.abspath(topk_dir),
            "result_json": os.path.abspath(result_path),
            "scoring_json": os.path.abspath(scoring_path),
        },
        "consistency_check": {
            "result_total_questions": result_total,
            "scoring_total_questions": scoring_total,
            "is_consistent": (result_total == scoring_total),
        },
        "winner_count": {
            "A": count_a,
            "B": count_b,
            "Tie": count_tie,
            "Invalid": count_invalid,
        },
        "winner_rate_percent": {
            "A": _safe_percent(count_a, valid_total),
            "B": _safe_percent(count_b, valid_total),
            "Tie": _safe_percent(count_tie, valid_total),
            "Invalid": _safe_percent(count_invalid, scoring_total if scoring_total > 0 else 1),
        },
        "score_stats": {
            "avg_total_score_A": avg_a_score,
            "avg_total_score_B": avg_b_score,
            "avg_score_gap_A_minus_B": avg_score_gap_a_minus_b,
        },
        "error_count": failed_count,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def analyze_all_topk_results(
    evaluation_results_dir: str = os.path.dirname(os.path.abspath(__file__)),
) -> Dict[str, Any]:
    """
    分析 outputs/evaluation_results 下所有 topk_* 目录的结果。
    """
    pattern = os.path.join(evaluation_results_dir, "topk_*")
    candidates = [p for p in glob(pattern) if os.path.isdir(p)]

    topk_values: List[int] = []
    for path in candidates:
        name = os.path.basename(path)
        m = re.fullmatch(r"topk_(\d+)", name)
        if m:
            topk_values.append(int(m.group(1)))
    topk_values = sorted(set(topk_values))

    reports: List[Dict[str, Any]] = []
    for topk in topk_values:
        try:
            reports.append(analyze_topk_result(topk=topk, evaluation_results_dir=evaluation_results_dir))
        except Exception as exc:
            reports.append(
                {
                    "topk": topk,
                    "error": str(exc),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                }
            )

    valid_reports = [r for r in reports if "error" not in r]
    best_by_a_win_count = None
    best_by_a_minus_b_gap = None
    if valid_reports:
        best_by_a_win_count = max(valid_reports, key=lambda r: r["winner_count"]["A"])["topk"]
        # 如果 gap 缺失，则按极小值处理
        best_by_a_minus_b_gap = max(
            valid_reports,
            key=lambda r: r["score_stats"]["avg_score_gap_A_minus_B"]
            if isinstance(r["score_stats"]["avg_score_gap_A_minus_B"], (int, float))
            else float("-inf"),
        )["topk"]

    return {
        "base_dir": os.path.abspath(evaluation_results_dir),
        "topk_list": topk_values,
        "report_count": len(reports),
        "reports": reports,
        "best_topk": {
            "by_a_win_count": best_by_a_win_count,
            "by_avg_score_gap_a_minus_b": best_by_a_minus_b_gap,
        },
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }


def _normalize_text(text: str) -> str:
    """将文本归一化为便于匹配的形式。"""
    if text is None:
        return ""
    normalized = str(text).lower()
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _resolve_default_keypoint_path(input_json_path: str) -> Optional[str]:
    """自动寻找统一命名的关键点文件 keypoint.json。"""
    base_dir = os.path.dirname(os.path.abspath(input_json_path))
    path = os.path.join(base_dir, "keypoint.json")
    if os.path.exists(path):
        return path
    return None


def _load_keypoints(keypoint_json_path: Optional[str]) -> Dict[int, List[Dict[str, Any]]]:
    """加载关键点文件，返回 index -> keypoints 的映射。"""
    if not keypoint_json_path:
        return {}
    if not os.path.exists(keypoint_json_path):
        raise FileNotFoundError(f"Keypoint JSON not found: {keypoint_json_path}")

    with open(keypoint_json_path, "r", encoding="utf-8") as f:
        keypoint_data = json.load(f)

    items = keypoint_data.get("items", [])
    if not isinstance(items, list):
        raise ValueError("Keypoint JSON must contain an 'items' list.")

    mapping: Dict[int, List[Dict[str, Any]]] = {}
    for item in items:
        idx = item.get("index")
        kps = item.get("keypoints", [])
        if isinstance(idx, int) and isinstance(kps, list):
            mapping[idx] = kps
    return mapping


def _build_match_candidates(kp: Dict[str, Any]) -> List[str]:
    """从关键点构造可用于字符串召回匹配的候选短语。"""
    raw_text = str(kp.get("text", "")).strip()
    aliases = kp.get("aliases") or []

    candidates: List[str] = []
    if raw_text:
        candidates.append(raw_text)
        # 去掉“说明：/给出定义：”这类前缀，保留语义主体。
        if "：" in raw_text:
            candidates.append(raw_text.split("：", 1)[1].strip())
        if ":" in raw_text:
            candidates.append(raw_text.split(":", 1)[1].strip())
        cleaned = re.sub(r"^(说明|给出定义|列举|解释原因|说明区别|说明联系)\s*[:：]?", "", raw_text).strip()
        if cleaned:
            candidates.append(cleaned)

    if isinstance(aliases, list):
        for alias in aliases:
            if alias is None:
                continue
            alias_text = str(alias).strip()
            if alias_text:
                candidates.append(alias_text)

    # 去重并过滤过短/过泛候选，降低误判。
    generic_terms = {"概念", "定义", "含义", "特点", "特征", "步骤", "流程", "过程", "作用", "用途"}
    uniq: List[str] = []
    seen = set()
    for text in candidates:
        norm = _normalize_text(text)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        if len(norm) <= 1:
            continue
        if norm in generic_terms:
            continue
        uniq.append(norm)
    return uniq


def _compute_answer_recall(answer: str, keypoints: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算单个回答针对关键点集合的召回率。"""
    if not keypoints:
        return {
            "point_recall": None,
            "weighted_recall": None,
            "must_have_hit_rate": None,
            "matched_keypoint_ids": [],
            "missed_keypoint_ids": [],
        }

    answer_norm = _normalize_text(answer or "")
    total_count = len(keypoints)
    total_weight = 0.0
    hit_count = 0
    hit_weight = 0.0
    must_have_total = 0
    must_have_hit = 0
    matched_ids: List[str] = []
    missed_ids: List[str] = []

    for i, kp in enumerate(keypoints, start=1):
        kp_id = kp.get("id") or f"KP{i}"
        weight = kp.get("weight", 0.0)
        try:
            weight_f = float(weight)
        except (TypeError, ValueError):
            weight_f = 0.0
        total_weight += max(weight_f, 0.0)

        must_have = bool(kp.get("must_have", False))
        if must_have:
            must_have_total += 1

        candidates = _build_match_candidates(kp)
        matched = any(c in answer_norm for c in candidates) if candidates else False

        if matched:
            hit_count += 1
            hit_weight += max(weight_f, 0.0)
            matched_ids.append(kp_id)
            if must_have:
                must_have_hit += 1
        else:
            missed_ids.append(kp_id)

    point_recall = round(hit_count / total_count, 4) if total_count > 0 else None
    weighted_recall = round(hit_weight / total_weight, 4) if total_weight > 0 else None
    must_have_hit_rate = (
        round(must_have_hit / must_have_total, 4) if must_have_total > 0 else None
    )

    return {
        "point_recall": point_recall,
        "weighted_recall": weighted_recall,
        "must_have_hit_rate": must_have_hit_rate,
        "matched_keypoint_ids": matched_ids,
        "missed_keypoint_ids": missed_ids,
    }


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


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
    judge_llm: Any,
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
    judge_llm: Optional[Any] = None,
    keypoint_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """评测一整个对比结果文件。"""
    if not os.path.exists(input_json_path):
        raise FileNotFoundError(
            f"Input JSON not found: {input_json_path}. "
            "Please pass --input explicitly or generate rag_vs_non_rag JSON first."
        )

    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    is_raw_ab_input = isinstance(data, list)
    is_existing_scoring_input = isinstance(data, dict) and isinstance(data.get("evaluations"), list)
    if not is_raw_ab_input and not is_existing_scoring_input:
        raise ValueError("Input JSON must be a list (result.json) or a scoring dict with 'evaluations'.")

    # result.json 场景下才需要调用评委模型；scoring.json 回填场景只计算召回率。
    if is_raw_ab_input:
        if judge_llm is None:
            from src.llm_integration.online_judge_llm import OnlineJudgeLLM
            judge_llm = OnlineJudgeLLM()
        records = data
    else:
        records = data.get("evaluations", [])
    resolved_keypoint_path = keypoint_json_path or _resolve_default_keypoint_path(input_json_path)
    keypoint_map = _load_keypoints(resolved_keypoint_path)

    evaluations: List[Dict[str, Any]] = []
    winner_count = {"A": 0, "B": 0, "Tie": 0, "Invalid": 0}

    for idx, qa in enumerate(records, start=1):
        record_index = (
            qa.get("index")
            if isinstance(qa, dict) and isinstance(qa.get("index"), int)
            else idx
        )
        question = qa["question"]
        answer_a = qa["A"]
        answer_b = qa["B"]

        item: Dict[str, Any] = {
            "index": record_index,
            "question": question,
            "A": answer_a,
            "B": answer_b,
        }

        # 召回率计算（不依赖评委模型结果）
        item_keypoints = keypoint_map.get(record_index, [])
        recall_a = _compute_answer_recall(answer_a, item_keypoints)
        recall_b = _compute_answer_recall(answer_b, item_keypoints)
        weighted_a = recall_a.get("weighted_recall")
        weighted_b = recall_b.get("weighted_recall")
        if isinstance(weighted_a, (int, float)) and isinstance(weighted_b, (int, float)):
            if weighted_a > weighted_b:
                recall_winner = "A"
            elif weighted_b > weighted_a:
                recall_winner = "B"
            else:
                recall_winner = "Tie"
        else:
            recall_winner = None
        item["recall"] = {
            "A": recall_a,
            "B": recall_b,
            "winner_by_weighted_recall": recall_winner,
            "keypoint_count": len(item_keypoints),
        }

        if not question or (not answer_a and not answer_b):
            item["judge_result"] = None
            item["error"] = "Invalid record: missing question or both answers are empty."
            winner_count["Invalid"] += 1
            evaluations.append(item)
            continue

        if is_existing_scoring_input:
            existing_judge_result = qa.get("judge_result")
            existing_error = qa.get("error")
            item["judge_result"] = existing_judge_result
            item["error"] = existing_error
            winner = (
                (existing_judge_result or {}).get("winner", "Invalid")
                if isinstance(existing_judge_result, dict)
                else "Invalid"
            )
            if winner not in winner_count:
                winner = "Invalid"
            winner_count[winner] += 1
        else:
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
        "total_questions": len(records),
        "winner_count": winner_count,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_file": os.path.abspath(input_json_path),
        "input_mode": "scoring_json" if is_existing_scoring_input else "ab_result_json",
    }

    # 汇总召回率指标
    summary["recall_summary"] = {
        "enabled": bool(keypoint_map),
        "keypoint_file": os.path.abspath(resolved_keypoint_path) if resolved_keypoint_path else None,
        "A": {
            "avg_point_recall": _safe_mean([x["recall"]["A"]["point_recall"] for x in evaluations]),
            "avg_weighted_recall": _safe_mean([x["recall"]["A"]["weighted_recall"] for x in evaluations]),
            "avg_must_have_hit_rate": _safe_mean([x["recall"]["A"]["must_have_hit_rate"] for x in evaluations]),
        },
        "B": {
            "avg_point_recall": _safe_mean([x["recall"]["B"]["point_recall"] for x in evaluations]),
            "avg_weighted_recall": _safe_mean([x["recall"]["B"]["weighted_recall"] for x in evaluations]),
            "avg_must_have_hit_rate": _safe_mean([x["recall"]["B"]["must_have_hit_rate"] for x in evaluations]),
        },
        "winner_by_weighted_recall_count": {
            "A": sum(1 for x in evaluations if x["recall"]["winner_by_weighted_recall"] == "A"),
            "B": sum(1 for x in evaluations if x["recall"]["winner_by_weighted_recall"] == "B"),
            "Tie": sum(1 for x in evaluations if x["recall"]["winner_by_weighted_recall"] == "Tie"),
        },
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
    config: Dict[str, Any] = {}
    if yaml is not None and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    default_output_dir = config['paths']['output_dir']
    default_input_json_path = os.path.join(default_output_dir, "evaluation_results", f"topk_{config['retrieval']['top_k']}", "result.json")
    default_output_path = os.path.join(default_output_dir, "evaluation_results", f"topk_{config['retrieval']['top_k']}", "scoring.json")

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
    parser.add_argument(
        "--keypoints",
        default=None,
        help="Path to keypoint JSON file (optional). If omitted, auto-detect keypoint.json near input file.",
    )
    args = parser.parse_args()

    result = evaluate_file(
        input_json_path=args.input,
        output_json_path=args.output,
        keypoint_json_path=args.keypoints,
    )
    print("[OK] 评测完成")
    print(f"输出文件: {result['output_json_path']}")
    print(f"统计摘要: {json.dumps(result['summary'], ensure_ascii=False)}")


if __name__ == "__main__":
    main()