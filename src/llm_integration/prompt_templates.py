"""提示词模板"""


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
