from rag_pipeline import RAGPipeline

def evaluate_rag_system():
    """
    评估RAG系统性能
    """
    # 初始化RAG pipeline
    pipeline = RAGPipeline()
    
    # 测试集
    test_cases = [
        {
            "query": "什么是人工智能？",
            "expected_keywords": ["机器", "计算机系统", "人类智能", "模拟"]
        },
        {
            "query": "机器学习和深度学习有什么关系？",
            "expected_keywords": ["分支", "子集", "神经网络"]
        },
        {
            "query": "自然语言处理有哪些应用？",
            "expected_keywords": ["机器翻译", "情感分析", "问答系统"]
        },
        {
            "query": "计算机视觉的应用有哪些？",
            "expected_keywords": ["人脸识别", "物体检测", "图像分类"]
        },
        {
            "query": "语音识别的应用有哪些？",
            "expected_keywords": ["语音助手", "语音搜索", "语音控制"]
        }
    ]
    
    # 评估结果
    results = []
    
    for test_case in test_cases:
        query = test_case["query"]
        expected_keywords = test_case["expected_keywords"]
        
        # 生成回答
        answer = pipeline.rag_qa(query)
        
        # 计算关键词匹配率
        matched_keywords = 0
        for keyword in expected_keywords:
            if keyword in answer:
                matched_keywords += 1
        
        match_rate = matched_keywords / len(expected_keywords)
        
        results.append({
            "query": query,
            "answer": answer,
            "match_rate": match_rate
        })
    
    # 计算平均匹配率
    average_match_rate = sum([r["match_rate"] for r in results]) / len(results)
    
    # 打印评估结果
    print("RAG系统评估结果：")
    print(f"平均关键词匹配率: {average_match_rate:.2f}")
    print("=" * 80)
    
    for result in results:
        print(f"问题: {result['query']}")
        print(f"回答: {result['answer']}")
        print(f"匹配率: {result['match_rate']:.2f}")
        print("-" * 80)

if __name__ == "__main__":
    evaluate_rag_system()
