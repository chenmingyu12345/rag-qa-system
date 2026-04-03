from chunk import load_and_chunk
from embedding import EmbeddingModel
from faiss_index import FAISSIndex
from rerank import Reranker
from llm import build_prompt, generate_answer

class RAGPipeline:
    def __init__(self, docs_path='docs.txt'):
        """
        初始化RAG pipeline
        
        Args:
            docs_path: 文档路径
        """
        # 加载并分块文档
        self.chunks = load_and_chunk(docs_path)
        
        # 初始化嵌入模型
        self.embedding_model = EmbeddingModel()
        
        # 生成嵌入
        embeddings = self.embedding_model.embed_texts(self.chunks)
        
        # 初始化FAISS索引
        dimension = embeddings.shape[1]
        self.index = FAISSIndex(dimension)
        self.index.add_embeddings(embeddings, self.chunks)
        
        # 初始化重排序模型
        self.reranker = Reranker()
    
    def retrieve(self, query):
        """
        检索相关文档
        
        Args:
            query: 查询文本
        
        Returns:
            相关文档列表
        """
        # 生成查询嵌入
        query_embedding = self.embedding_model.embed_query(query)
        
        # 检索相似文档
        documents = self.index.search(query_embedding, k=5)
        
        # 重排序
        reranked_docs = self.reranker.rerank(query, documents)
        
        return reranked_docs
    
    def rag_qa(self, query):
        """
        RAG问答
        
        Args:
            query: 查询文本
        
        Returns:
            回答
        """
        # 检索相关文档
        contexts = self.retrieve(query)
        
        # 构建prompt
        prompt = build_prompt(query, contexts)
        
        # 生成回答
        answer = generate_answer(prompt)
        
        return answer

# 测试
if __name__ == "__main__":
    pipeline = RAGPipeline()
    
    # 测试问题
    test_queries = [
        "什么是人工智能？",
        "机器学习和深度学习有什么关系？",
        "自然语言处理有哪些应用？"
    ]
    
    for query in test_queries:
        print(f"问题：{query}")
        print(f"回答：{pipeline.rag_qa(query)}")
        print("=" * 50)
    
    # CLI测试
    print("\nCLI测试开始：")
    print("输入 'exit' 退出")
    while True:
        q = input("Question:")
        if q.lower() == 'exit':
            break
        print(pipeline.rag_qa(q))
