"""
使用使用Numpy+余弦检索构建数据库
"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os
import yaml
import json
import pickle
# from langchain_community.retrievers import BM25Retriever
from rank_bm25 import BM25Okapi
import jieba

class VectorStore:
    """使用Numpy+余弦检索构建数据库"""

    def __init__(self, config_path: str = "config/configs.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        # 初始化向量数据库路径
        db_path = self.config['paths']['vector_db_path']
        os.makedirs(db_path, exist_ok=True)
        
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.pkl")
        self.prefixed_index_file = os.path.join(db_path, "prefixed_faiss_index.pkl")
        self.texts_file = os.path.join(db_path, "texts.json")
        self.prefixed_texts_file = os.path.join(db_path, "prefixed_texts.json")
        self.metadatas_file = os.path.join(db_path, "metadatas.json")

        # 初始化嵌入模型
        embedding_model_name = self.config['retrieval']['embedding_model']
        trc = self.config['retrieval']['trust_remote_code']
        self.embedding_model = SentenceTransformer(embedding_model_name,trust_remote_code=trc)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # 存储文本和元数据
        self.texts: List[str] = []
        self.prefixed_texts: List[str] = []
        self.metadata: List[Dict] = []
        self.embeddings: np.ndarray | None = None
        self.prefixed_embeddings: np.ndarray | None = None

        # 缓存BM25
        self.bm25_cache_path = os.path.join(self.db_path, "bm25_tokens.json")
        self._tokenized_corpus: List[List[str]] | None = None
        self._bm25: BM25Okapi | None = None

        # 尝试加载已有的数据
        self._load_if_exists()

        print(f"✅ 向量存储初始化完成（Faiss-CPU后端）。模型: {embedding_model_name}, 维度: {self.embedding_dim}")

    def _load_if_exists(self):
        """如果存在已保存的数据，则加载"""
        if (self.texts is not [] and self.prefixed_texts is not [] and self.prefixed_embeddings is not None) and \
                len(self.texts) == len(self.prefixed_texts) == len(self.prefixed_embeddings):
            return

        if os.path.exists(self.texts_file):
            with open(self.texts_file, 'r', encoding='utf-8') as f:
                self.texts = json.load(f)
            if os.path.exists(self.index_file):
                with open(self.index_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                print(f"📂 从缓存加载了 {len(self.texts)} 个文本块")

        if os.path.exists(self.prefixed_texts_file):
            with open(self.prefixed_texts_file, 'r', encoding='utf-8') as f:
                self.prefixed_texts = json.load(f)
            if os.path.exists(self.prefixed_index_file):
                with open(self.prefixed_index_file, 'rb') as f:
                    self.prefixed_embeddings = pickle.load(f)
                print(f"📂 从缓存加载了 {len(self.prefixed_texts)} 个文本块")

        if os.path.exists(self.metadatas_file):
            with open(self.metadatas_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)

    def build_from_chunks(self, chunks: List[Dict]):
        """将文本块添加到向量数据库"""
        doc_prefix = self.config['retrieval']['embedding_doc_prefix']
        texts = [chunk['text'] for chunk in chunks]
        prefixed_texts = [doc_prefix + chunk['text'] for chunk in chunks]
        # metadata = [{
        #     'source_page': chunk.get('source_page'),
        #     'source_doc': chunk.get('source_doc')
        # } for chunk in chunks]
        metadata = [{k:v for k,v in chunk.items() if k != 'text'} for chunk in chunks]

        # 生成嵌入向量
        print("🔄 正在生成文本嵌入向量...")
        # 暂时不计算无前缀
        # embeddings = self.embedding_model.encode(texts, batch_size=16, show_progress_bar=True)
        embeddings = None
        prefixed_embeddings = self.embedding_model.encode(prefixed_texts, batch_size=16, show_progress_bar=True)

        # 存储数据
        self.texts = texts
        self.prefixed_texts = prefixed_texts
        self.metadata = metadata
        self.embeddings = embeddings
        self.prefixed_embeddings = prefixed_embeddings

        # 保存到磁盘
        with open(self.texts_file, 'w', encoding='utf-8') as f:
            json.dump(texts, f, ensure_ascii=False, indent=2)
        with open(self.prefixed_texts_file, 'w', encoding='utf-8') as f:
            json.dump(prefixed_texts, f, ensure_ascii=False, indent=2)

        with open(self.metadatas_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        # with open(self.index_file, 'wb') as f:
        #     pickle.dump(embeddings, f)
        with open(self.prefixed_index_file, 'wb') as f:
            pickle.dump(prefixed_embeddings, f)

        print(f"✅ 成功添加 {len(texts)} 个文本块到向量数据库。")

    def RRF(self, rank_list: List[int], k=60)->float:
        # reciprocal_rank_fusion, 倒数排名融合
        # rank为-1表示未能成功录入排名，该项排名的贡献为0
        return sum([1.0 / (rank + 60) for rank in rank_list if rank >= 0])

    def rewrite(self, query:str):
        """查询改写"""
        pass

    def search(self, query: str, top_k: int = 8, top_m: int = 20, threshold: float=0) -> List[Dict]:
        """检索与查询最相关的文本块"""
        query_prefix = self.config['retrieval']['embedding_query_prefix']
        search_query = query_prefix + query
        # 两路各取top_m,在RRF中融合
        # BM25检索
        bm25_idx, scores = self.search_by_BM25(query, top_m)
        # 向量检索（余弦相似度）
        # 只在向量检索侧添加搜索前缀
        vc_idx, similarities = self.search_by_vector(search_query, top_m, threshold)

        idx2ranks = {}
        for i in range(len(bm25_idx)):
            if idx2ranks.get(bm25_idx[i]) is None:
                idx2ranks[bm25_idx[i]] = [-1, -1]
            idx2ranks[bm25_idx[i]][0] = i

        for i in range(len(vc_idx)):
            if idx2ranks.get(vc_idx[i]) is None:
                idx2ranks[vc_idx[i]] = [-1, -1]
            idx2ranks[vc_idx[i]][1] = i

        compositive_scores = []
        indices = []
        for idx, rank_list in idx2ranks.items():
            indices.append(idx)
            compositive_scores.append(self.RRF(rank_list))

        top_indexes = np.argsort(compositive_scores)[::-1][:top_k].tolist()
        top_indices = [indices[i] for i in top_indexes]

        retrieved_chunks = []
        for idx in top_indices:
            retrieved_chunks.append({
                'text': self.texts[idx],
                'metadata': self.metadata[idx],
                'similarity': float(similarities[idx]) if len(similarities) > 0 else None,
                'score': float(scores[idx]) if scores else None
            })

        return retrieved_chunks

    def tokenize_zh(self, text: str):
        return [w for w in jieba.lcut(text.strip()) if w.strip()]

    def search_by_vector(self, query: str, top_m: int, threshold: float) -> tuple[List[int], List[float]]:
        """向量检索"""
        if not self.texts or self.prefixed_embeddings is None:
            return [],[]

        # 生成查询向量
        query_embedding = self.embedding_model.encode([query])[0]

        # 计算余弦相似度
        similarities = np.dot(self.prefixed_embeddings, query_embedding) / (
                np.linalg.norm(self.prefixed_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        vector_idx = [i for i in np.argsort(similarities)[::-1][:top_m] if similarities[i] >= threshold]

        return vector_idx, similarities

    def search_by_BM25(self, query: str, top_m: int) -> tuple[List[int], List[float]]:
        """BM25检索"""

        if not self.texts:
            return [],[]

        # BM25缓存
        if self._tokenized_corpus is not None and len(self.texts) == len(self._tokenized_corpus):
            tokenized_corpus = self._tokenized_corpus
            bm25 = self._bm25
        else:
            if os.path.exists(self.bm25_cache_path):
                with open(self.bm25_cache_path, "r", encoding="utf-8") as f:
                    tokenized_corpus: List[List[str]] = json.load(f)
            else:
                tokenized_corpus = [self.tokenize_zh(t) for t in self.texts]
                with open(self.bm25_cache_path, "w", encoding="utf-8") as f:
                    json.dump(tokenized_corpus, f, ensure_ascii=False, indent=0)
            bm25 = BM25Okapi(tokenized_corpus)
            self._tokenized_corpus = tokenized_corpus
            self._bm25 = bm25

        query_token = self.tokenize_zh(query)
        scores = bm25.get_scores(query_token)

        bm25_idx = np.argsort(scores)[::-1][:top_m].tolist()

        scores = [float(s) for s in scores]
        return bm25_idx, scores

    def clear(self) -> bool:
        """清空向量数据库"""
        try:
            # 清空内存中的数据
            self.texts = []
            self.prefixed_texts = []
            self.metadata = []
            self.embeddings = None
            self.prefixed_embeddings = None
            self._tokenized_corpus = None
            self._bm25 = None

            # 删除磁盘上的文件
            files_to_delete = [self.index_file, self.prefixed_index_file,
                               self.texts_file, self.prefixed_texts_file,
                               self.metadatas_file, self.bm25_cache_path]
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
    results = vs.search(query)

    print(f"\n🔍 测试查询: '{query}'")
    print("检索结果:")
    for i, r in enumerate(results):
        print(f"[{i + 1}] 余弦相似度: {r['similarity']:.3f}, BM25分数: {r['score']:.3f}, 页码: {r['metadata']['source_page']}")
        print(f"    文本: {r['text']}...")
