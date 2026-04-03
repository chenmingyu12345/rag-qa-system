from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingModel:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2'):
        """
        初始化嵌入模型
        
        Args:
            model_name: 模型名称
        """
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts):
        """
        生成文本嵌入
        
        Args:
            texts: 文本列表
        
        Returns:
            嵌入向量数组
        """
        embeddings = self.model.encode(texts)
        return np.array(embeddings)
    
    def embed_query(self, query):
        """
        生成查询嵌入
        
        Args:
            query: 查询文本
        
        Returns:
            嵌入向量
        """
        embedding = self.model.encode([query])[0]
        return np.array(embedding)
