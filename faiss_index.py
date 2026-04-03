import faiss
import numpy as np

class FAISSIndex:
    def __init__(self, dimension):
        """
        初始化FAISS索引
        
        Args:
            dimension: 嵌入向量的维度
        """
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
    
    def add_embeddings(self, embeddings, texts):
        """
        添加嵌入向量到索引
        
        Args:
            embeddings: 嵌入向量数组
            texts: 对应的文本列表
        """
        self.index.add(embeddings)
        self.texts.extend(texts)
    
    def search(self, query_embedding, k=3):
        """
        搜索相似的嵌入向量
        
        Args:
            query_embedding: 查询嵌入向量
            k: 返回的结果数量
        
        Returns:
            相似文本列表
        """
        distances, indices = self.index.search(np.array([query_embedding]), k)
        results = []
        for i in range(k):
            if indices[0][i] < len(self.texts):
                results.append(self.texts[indices[0][i]])
        return results
