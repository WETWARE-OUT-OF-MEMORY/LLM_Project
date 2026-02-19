"""提示词模板"""
from typing import Dict


def build_rag_prompt(question: str, context: str) -> str:
    """构建RAG提示词"""
    prompt = f"""【参考内容】
{context}

【问题】
{question}

【回答】
请根据上述参考内容回答问题。如果参考内容中没有相关信息，请说明。"""
    return prompt


def build_non_rag_prompt(question: str) -> str:
    """构建非RAG提示词"""
    prompt = f"""【问题】
{question}

【回答】
"""
    return prompt


def build_judge_scoring_prompt(question: str, answer_a: str, answer_b: str) -> Dict[str, str]:
    """构建评委模型打分提示词（Responses API 的 instructions + input 结构）。"""
    instructions = """你是严谨、公正的专业问答评委。请比较两个回答，并给出结构化评分。

【评分维度（每项0-10分）】
1) accuracy: 事实正确性与概念严谨性
2) completeness: 覆盖关键要点的完整程度
3) clarity: 表达清晰度与逻辑性
4) relevance: 与问题直接相关程度

【总分】
- total_score_A = 四项之和（0-40）
- total_score_B = 四项之和（0-40）

【裁决要求】
- winner 只能是 "A"、"B"、"Tie"
- 给出简明理由，指出关键优缺点

【输出格式要求（必须严格遵守）】
只输出一个JSON对象，不要输出任何额外文本、解释、Markdown、代码块标记。
JSON字段必须包含：
{
  "scores": {
    "A": {
      "accuracy": 0,
      "completeness": 0,
      "clarity": 0,
      "relevance": 0,
      "total_score": 0
    },
    "B": {
      "accuracy": 0,
      "completeness": 0,
      "clarity": 0,
      "relevance": 0,
      "total_score": 0
    }
  },
  "winner": "A",
  "reason": "一句话到三句话，简述判定依据"
}"""

    input_text = f"""【问题】
{question}

【回答A】
{answer_a}

【回答B】
{answer_b}
"""
    return {"instructions": instructions, "input": input_text}
