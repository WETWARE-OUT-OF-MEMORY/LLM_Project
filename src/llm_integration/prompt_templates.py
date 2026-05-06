"""提示词模板"""
from typing import Dict, Optional


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
    instructions = """你是严谨、公正的专业计算机学科问答评委。请严格依据问题本身，对两个回答分别进行独立评分，然后再做比较。

【评分原则】
- 不因篇幅长短直接加分
- 不因语言华丽程度加分
- 只根据问题要求判断是否准确、完整
- 若存在事实错误，应显著扣减 accuracy 分
- 若出现与问题无关扩展，应降低 relevance 分

【评分维度（每项0-10分）】
1) accuracy: 概念是否准确，是否存在错误或混淆
2) completeness: 是否覆盖问题的关键知识点
3) clarity: 逻辑是否清晰，结构是否合理
4) relevance: 是否紧扣题目要求，无明显跑题

【评分流程（必须遵守）】
1. 分别独立评估回答A
2. 分别独立评估回答B
3. 再根据总分判断 winner

【总分计算】
- total_score = 四项之和（0-40）

【裁决规则】
- 若总分差 ≥ 3 分，则高分者为 winner
- 若差值 < 3 分，则 winner = "Tie"

【输出格式要求（必须严格遵守）】
只输出一个JSON对象，不得输出任何额外文本。

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
  "reason": "1-3句话，简述关键差异"
}"""

    input_text = f"""【问题】
{question}

【回答A】
{answer_a}

【回答B】
{answer_b}
"""
    return {"instructions": instructions, "input": input_text}

def build_image_description_prompt(section_title: Optional[str] = None, nearby_text: Optional[str] = None,)->str:
    """构造图片描述 prompt（VLM 的 user 消息）。"""
    parts = [
        "请用一段简明的中文描述这张图片的关键内容。",
        "要求：",
        "- 100-200 字",
        "- 突出图中的概念、流程、关系，便于知识检索",
        "- 不要描述颜色、字体等无关样式",
    ]
    if section_title:
        parts.append(f"\n图片所在章节：{section_title}")
    if nearby_text:
        parts.append(f"\n相邻文本：{nearby_text}")
    return "\n".join(parts)