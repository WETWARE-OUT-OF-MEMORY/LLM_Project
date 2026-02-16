from typing import List, Dict
from .vector_store import VectorStore
from ..llm_integration.local_llm import LocalLLM
from ..llm_integration.prompt_templates import build_rag_prompt, build_non_rag_prompt


class RAGCore:
    """RAG核心类：整合检索和生成"""

    def __init__(self, config_path: str = "config/configs.yaml"):
        print("🔄 初始化RAG系统...")
        self.vector_store = VectorStore(config_path)
        self.llm = LocalLLM(config_path)
        print("✅ RAG系统初始化完成")

    def answer_with_rag(self, question: str, top_k: int = 3) -> Dict:
        """使用RAG回答问题"""
        # 1. 检索相关文档
        retrieved_docs = self.vector_store.search(question, top_k=top_k)

        if not retrieved_docs:
            context = "未找到相关参考内容"
        else:
            # 2. 构建上下文
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"[文档{i}] {doc['text']}")
            context = "\n\n".join(context_parts)

        # 3. 构建提示词
        prompt = build_rag_prompt(question, context)

        # 4. 生成回答
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "retrieved_docs": retrieved_docs
        }

    def answer_without_rag(self, question: str) -> Dict:
        """不使用RAG直接回答问题"""
        prompt = build_non_rag_prompt(question)
        answer = self.llm.generate(prompt)

        return {
            "question": question,
            "answer": answer
        }
