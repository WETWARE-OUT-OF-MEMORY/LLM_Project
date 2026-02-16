"""
使用Faiss作为向量数据库（支持Python 3.14，CPU版本）
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import yaml
import json
import pickle


class VectorStore:
    """使用Faiss构建和管理向量数据库（CPU版本）"""

    def __init__(self, config_path: str = "config/configs.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化向量数据库路径
        db_path = self.config['paths']['vector_db_path']
        os.makedirs(db_path, exist_ok=True)
        
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.pkl")
        self.texts_file = os.path.join(db_path, "texts.json")
        self.metadatas_file = os.path.join(db_path, "metadatas.json")

        # 初始化嵌入模型
        embedding_model_name = self.config['retrieval']['embedding_model']
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 存储文本和元数据
        self.texts: List[str] = []
        self.metadatas: List[Dict] = []
        self.embeddings: np.ndarray = None

        # 尝试加载已有的数据
        self._load_if_exists()

        print(f"✅ 向量存储初始化完成（Faiss-CPU后端）。模型: {embedding_model_name}, 维度: {self.embedding_dim}")

    def _load_if_exists(self):
        """如果存在已保存的数据，则加载"""
        if os.path.exists(self.texts_file) and os.path.exists(self.metadatas_file):
            with open(self.texts_file, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
            with open(self.metadatas_file, 'r', encoding='utf-8') as f:
                self.metadatas = json.load(f)
            
            if os.path.exists(self.index_file):
                with open(self.index_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"📂 从缓存加载了 {len(self.texts)} 个文本块")

    def build_from_chunks(self, chunks: List[Dict]):
        """将文本块添加到向量数据库"""
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [{
            'source_page': chunk.get('source_page'),
            'source_doc': chunk.get('source_doc')
        } for chunk in chunks]

        # 生成嵌入向量
        print("🔄 正在生成文本嵌入向量...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)

        # 存储数据
        self.texts = texts
        self.metadatas = metadatas
        self.embeddings = embeddings

        # 保存到磁盘
        with open(self.texts_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        with open(self.metadatas_file, 'w', encoding='utf-8') as f:
            json.dump(metadatas, f, ensure_ascii=False, indent=2)
        with open(self.index_file, 'wb') as f:
            pickle.dump(embeddings, f)

        print(f"✅ 成功添加 {len(texts)} 个文本块到向量数据库。")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索与查询最相关的文本块"""
        if not self.texts or self.embeddings is None:
            return []

        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])[0]

        # 计算余弦相似度
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # 获取top-k结果
        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved_chunks = []
        for idx in top_indices:
            retrieved_chunks.append({
                'text': self.texts[idx],
                'metadata': self.metadatas[idx],
                'score': float(similarities[idx])
            })

        return retrieved_chunks

    def clear(self) -> bool:
        """清空向量数据库"""
        try:
            # 清空内存中的数据
            self.texts = []
            self.metadatas = []
            self.embeddings = None
            
            # 删除磁盘上的文件
            files_to_delete = [self.index_file, self.texts_file, self.metadatas_file]
            deleted_count = 0
            
            for file_path in files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
            
            print(f"✅ 向量数据库已清空（删除了 {deleted_count} 个文件）")
            return True
            
        except Exception as e:
            print(f"❌ 清空向量数据库失败: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取向量数据库统计信息"""
        stats = {
            'total_chunks': len(self.texts),
            'embedding_dim': self.embedding_dim,
            'db_path': self.db_path,
            'files_exist': {
                'index': os.path.exists(self.index_file),
                'texts': os.path.exists(self.texts_file),
                'metadatas': os.path.exists(self.metadatas_file)
            }
        }
        
        # 计算文件大小
        if os.path.exists(self.index_file):
            stats['index_size_mb'] = os.path.getsize(self.index_file) / (1024 * 1024)
        if os.path.exists(self.texts_file):
            stats['texts_size_mb'] = os.path.getsize(self.texts_file) / (1024 * 1024)
        if os.path.exists(self.metadatas_file):
            stats['metadatas_size_mb'] = os.path.getsize(self.metadatas_file) / (1024 * 1024)
        
        return stats


# 快速测试函数
def test_vector_store():
    """测试向量存储功能"""
    # 模拟一些文本块
    test_chunks = [
        {'text': '进程是程序的一次执行过程，是系统进行资源分配和调度的基本单位。',
         'source_page': 1, 'source_doc': '操作系统'},
        {'text': '线程是进程内的一个执行单元，是CPU调度和分派的基本单位。',
         'source_page': 2, 'source_doc': '操作系统'},
        {'text': '虚拟内存通过页面置换算法，使得程序可以使用比物理内存更大的地址空间。',
         'source_page': 3, 'source_doc': '操作系统'}
    ]

    # 构建向量存储
    vs = VectorStore()
    vs.build_from_chunks(test_chunks)

    # 测试检索
    query = "进程和线程有什么区别？"
    results = vs.search(query, top_k=2)

    print(f"\n🔍 测试查询: '{query}'")
    print("检索结果:")
    for i, r in enumerate(results):
        print(f"[{i + 1}] 相似度: {r['score']:.3f}, 页码: {r['metadata']['source_page']}")
        print(f"    文本: {r['text'][:100]}...")
