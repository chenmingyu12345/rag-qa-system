from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        初始化重排序模型
        
        Args:
            model_name: 模型名称
        """
        self.model = CrossEncoder(model_name)
    
    def rerank(self, query, documents, top_k=3):
        """
        对检索结果进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回的结果数量
        
        Returns:
            重排序后的文档列表
        """
        # 构建查询-文档对
        pairs = [[query, doc] for doc in documents]
        # 计算相关性分数
        scores = self.model.predict(pairs)
        # 按分数排序
        sorted_indices = scores.argsort()[::-1]
        # 返回排序后的文档
        return [documents[i] for i in sorted_indices[:top_k]]
