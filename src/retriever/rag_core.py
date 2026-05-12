import yaml
from typing import List, Dict

from src.retriever.vector_store import VectorStore
from src.llm_integration.local_llm import LocalLLM
from src.llm_integration.prompt_templates import build_rag_prompt, build_non_rag_prompt, build_question_rewrite_prompt
from src.utils.chunk_renderer import render_chunk_for_llm
from src.llm_integration.online_rewrite_llm import OnlineRewriteLLM

class RAGCore:
    """RAG核心类：整合检索和生成"""

    def __init__(self, config_path: str = "D:/Learn/machine_learning/LLM_Project/config/configs.yaml"):
        print("🔄 初始化RAG系统...")
        with open(config_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.vector_store = VectorStore(config_path)
        self.llm = LocalLLM(config_path)
        self.rewrite_llm = OnlineRewriteLLM(config_path)
        print("✅ RAG系统初始化完成")

    def answer_with_rag(self, question: str, top_k: int = 8, top_m: int = 20, threshold: float = 0.5, rewrite: bool = False) -> Dict:
        """使用RAG回答问题"""

        # 0. 改写用户问题
        if rewrite:
            prompt = build_question_rewrite_prompt(question)
            rewritten_question = self.rewrite_llm.generate(prompt)
            print(f"原问题: {question} \n改写后: {rewritten_question}")
            question = rewritten_question

        # 1. 检索相关文档
        retrieved_docs = self.vector_store.search(question, top_k=top_k, top_m=top_m, threshold=threshold)

        if not retrieved_docs:
            context = "未找到相关参考内容"
        else:
            # 2. 构建上下文
            context_parts = []
            for i, doc in enumerate(retrieved_docs, 1):
                dct = {'text': doc['text'], **doc['metadata']}
                context_parts.append(render_chunk_for_llm(dct))
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

if __name__ == "__main__":
    core = RAGCore()
    question = input("问题：")
    print(core.answer_with_rag(question, rewrite=True))
    """
    查询改写示例运行结果
    问题：
        栈怎么使用？在哪些地方使用？
    改写运行结果：
        1. 栈（Stack）这一抽象数据类型的核心操作（如入栈、出栈、判空、获取栈顶元素等）在不同编程语言（如C/C++、Java、Python）中的具体实现方式与API使用规范是怎样的？  
        2. 栈在计算机科学与软件工程中的典型应用场景有哪些？例如：函数调用过程中的运行时栈管理、表达式求值（中缀/后缀转换与计算）、括号匹配验证、深度优先搜索（DFS）的迭代实现、回溯算法的状态保存与撤销、编译器的语法分析（如递归下降解析）、内存管理中的局部变量分配等，其底层原理与设计动因分别是什么？  
        3. 在系统级或算法级设计中，何时应选择栈结构而非其他线性结构（如队列、链表、数组）？其时间复杂度（O(1) 均摊入/出栈）、空间局部性、LIFO语义约束等特性如何影响架构决策？
    """