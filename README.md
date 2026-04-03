# RAG-QA 系统

一个基于检索增强生成（RAG）的问答系统，使用Qwen模型进行文本生成，结合FAISS索引进行高效检索。

## 项目结构

```
rag_qa_system
│
├── docs.txt          # 示例文档
├── chunk.py          # 文档分块功能
├── embedding.py      # 文本嵌入生成
├── faiss_index.py    # FAISS索引构建和搜索
├── rerank.py         # 检索结果重排序
├── llm.py            # 语言模型加载和文本生成
├── rag_pipeline.py   # RAG pipeline整合
├── evaluate.py       # 系统评估
├── requirements.txt  # 依赖包列表
└── README.md         # 项目说明
```

## 功能特点

- **文档分块**：将长文档分割成小块，提高检索效率
- **文本嵌入**：使用多语言模型生成文本嵌入
- **FAISS索引**：高效的向量检索
- **结果重排序**：使用交叉编码器对检索结果进行重排序
- **中文支持**：支持中文问答
- **完整的RAG流程**：检索 → 重排序 → 生成

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备文档

在 `docs.txt` 文件中添加你的文档内容，每行一个段落。

### 2. 运行RAG系统

```bash
python rag_pipeline.py
```

系统会自动：
- 加载并分块文档
- 生成文本嵌入
- 构建FAISS索引
- 进入CLI交互模式，等待用户输入问题

### 3. 评估系统

```bash
python evaluate.py
```

系统会使用预设的测试案例评估性能，并输出评估结果。

## 示例

```
Question: 什么是人工智能？

回答：
人工智能（AI）是机器，特别是计算机系统对人类智能过程的模拟。AI指的是开发能够执行通常需要人类智能的任务的计算机系统。
```

## 技术栈

- **Transformers**：用于加载和使用语言模型
- **Sentence-Transformers**：用于生成文本嵌入和重排序
- **FAISS**：用于高效的向量检索
- **PyTorch**：深度学习框架
- **NumPy**：数值计算

## 注意事项

- 首次运行时会自动下载所需的模型，可能需要一些时间
- 建议在具有足够内存的环境中运行
- 可以根据需要修改 `docs.txt` 文件，添加更多领域相关的文档
