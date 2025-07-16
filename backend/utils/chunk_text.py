import re
from typing import List, Optional

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 100, 
    sentence_boundary: bool = True
) -> List[str]:
    """
    将文本分割成指定大小的块，尽量不切断单词和句子
    
    参数:
        text: 要分割的文本
        chunk_size: 每个块的目标大小(字符数)
        overlap: 块之间的重叠大小(字符数)
        sentence_boundary: 是否尽量在句子边界处分割
    
    返回:
        文本块列表
    """
    # 参数校验
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap必须在0和chunk_size之间")
    
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # 计算块的结束位置
        end_pos = min(current_pos + chunk_size, text_length)
        
        # 如果还有剩余文本且不是最后一块
        if end_pos < text_length:
            # 尝试在句子边界处分割
            if sentence_boundary:
                # 查找最近的句子结束标点
                sentence_end = _find_sentence_boundary(text, end_pos)
                if sentence_end is not None and sentence_end > current_pos:
                    end_pos = sentence_end
            
            # 如果没有找到句子边界或不允许在句子边界分割，则在单词边界分割
            if end_pos == current_pos + chunk_size:
                # 查找最近的单词边界
                word_boundary = _find_word_boundary(text, end_pos)
                if word_boundary is not None and word_boundary > current_pos:
                    end_pos = word_boundary
        
        # 添加当前块
        chunks.append(text[current_pos:end_pos].strip())
        
        # 更新位置，考虑重叠
        current_pos = end_pos - overlap if end_pos - overlap > current_pos else end_pos
    
    return chunks

def _find_sentence_boundary(text: str, position: int) -> Optional[int]:
    """
    查找最近的句子边界位置
    
    参数:
        text: 文本
        position: 开始查找的位置
    
    返回:
        最近的句子结束位置，如果没有找到则返回None
    """
    # 查找position之后最近的句子结束标点
    match = re.search(r'[.!?]\s+', text[position:])
    if match:
        return position + match.end()
    
    # 如果没有找到，尝试向前查找
    match = re.search(r'[.!?]\s+', text[:position][::-1])
    if match:
        return position - match.start()
    
    return None

def _find_word_boundary(text: str, position: int) -> Optional[int]:
    """
    查找最近的单词边界位置
    
    参数:
        text: 文本
        position: 开始查找的位置
    
    返回:
        最近的单词边界位置，如果没有找到则返回None
    """
    # 查找position之后最近的空白字符
    match = re.search(r'\s', text[position:])
    if match:
        return position + match.start()
    
    # 如果没有找到，尝试向前查找
    match = re.search(r'\s', text[:position][::-1])
    if match:
        return position - match.start()
    
    return None