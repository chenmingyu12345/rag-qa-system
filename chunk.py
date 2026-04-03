def chunk_text(text, chunk_size=100, overlap=20):
    """
    将文本分块
    
    Args:
        text: 原始文本
        chunk_size: 每个块的大小
        overlap: 块之间的重叠大小
    
    Returns:
        分块后的文本列表
    """
    chunks = []
    text_length = len(text)
    start = 0
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    
    return chunks

def load_and_chunk(file_path):
    """
    加载文档并分块
    
    Args:
        file_path: 文档路径
    
    Returns:
        分块后的文本列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 按段落分割
    paragraphs = text.split('\n\n')
    chunks = []
    
    for para in paragraphs:
        if para.strip():
            para_chunks = chunk_text(para)
            chunks.extend(para_chunks)
    
    return chunks
