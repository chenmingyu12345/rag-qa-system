import streamlit as st
from rag_pipeline import RAGPipeline

# 初始化RAG pipeline
@st.cache_resource

def init_pipeline():
    return RAGPipeline()

pipeline = init_pipeline()

# 设置页面标题
st.title("RAG-QA 系统")
st.subheader("基于检索增强生成的智能问答系统")

# 输入框
query = st.text_input("请输入你的问题：", placeholder="例如：什么是人工智能？")

# 提交按钮
if st.button("提交"):
    if query:
        with st.spinner("正在生成回答..."):
            # 获取回答
            answer = pipeline.rag_qa(query)
            
            # 显示回答
            st.success("回答：")
            st.write(answer)
    else:
        st.warning("请输入问题")

# 示例问题
st.sidebar.title("示例问题")
example_queries = [
    "什么是人工智能？",
    "机器学习和深度学习有什么关系？",
    "自然语言处理有哪些应用？",
    "计算机视觉的应用有哪些？",
    "语音识别的应用有哪些？"
]

for example in example_queries:
    if st.sidebar.button(example):
        query = example
        with st.spinner("正在生成回答..."):
            answer = pipeline.rag_qa(query)
            st.success("回答：")
            st.write(answer)

# 关于部分
st.sidebar.title("关于")
st.sidebar.info(
    "这是一个基于检索增强生成（RAG）的问答系统，使用Qwen模型进行文本生成，结合FAISS索引进行高效检索。\n\n"+
    "技术栈：\n"+
    "- Transformers\n"+
    "- Sentence-Transformers\n"+
    "- FAISS\n"+
    "- PyTorch\n"+
    "- Streamlit"
)
