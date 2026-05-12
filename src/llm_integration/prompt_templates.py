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
- 只根据问题要求判断是否准确、完整、是否紧扣题意
- 若存在事实错误，应显著扣减 accuracy 分
- 若出现与问题无关扩展，应降低 relevance 分
- 在无充分依据或材料明显不匹配时：若某回答明确说明「无法根据已有信息作答」「参考不足以支持结论」或仅作保守的常识性说明并标明不确定性，且未编造事实、未把无关内容强行当作依据，则其 accuracy 应明显高于另一回答在同类情形下臆造、串题、张冠李戴或严重跑题的情况
- completeness 不要求「写满」；在依据不足时，简短但诚实、不胡编的回答，其 completeness 可高于冗长但关键事实错误或与题无关的回答
- 当两名回答总分接近难以区分时（差值处于裁决边界附近），优先将更高分判给更诚实、更少事实错误、更少误导的一方，而不是更长但更易误导的一方
【评分维度（每项0-10分）】
1) accuracy: 概念是否准确，是否存在错误或混淆；是否编造、误用或与公认事实/题意明显矛盾
2) completeness: 是否覆盖问题的关键知识点；在题意清晰时是否答到点；依据不足时不按「越长越完整」给高分
3) clarity: 逻辑是否清晰，结构是否合理，表述是否自洽
4) relevance: 是否紧扣题目要求，无明显跑题；是否大段粘贴与本题无关的问答、习题版式或其它题目
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

def build_question_rewrite_prompt(question: str) -> Dict[str, str]:
    instruction = """
    你负责对用户提出的问题用更严谨、更准确的语言进行同义改写。
    不要回答这个问题，而是将用户提问中模糊的用词转换为专业的、精炼的词汇，再做同义改写的润色。
    如果用户的用词表义宽泛，改写时可以用多个专业词汇尽可能全面的覆盖问题内容，必要时可以改写为多个子问题。
    """

    input_text = f"【问题】: {question}"

    return {'instructions': instruction, 'input': input_text}