# import re

# def chunk_text(
#     text: str, 
#     chunk_size: int = 1000, 
#     overlap: int = 100, 
#     sentence_boundary: bool = True
# ) -> list[str]:
#     """
#     将文本分割成指定大小的块，尽量不切断单词和句子
    
#     参数:
#         text: 要分割的文本
#         chunk_size: 每个块的目标大小(字符数)
#         overlap: 块之间的重叠大小(字符数)
#         sentence_boundary: 是否尽量在句子边界处分割
    
#     返回:
#         文本块列表
#     """
#     # 参数校验
#     if chunk_size <= 0:
#         raise ValueError("chunk_size必须大于0")
#     if overlap < 0 or overlap >= chunk_size:
#         raise ValueError("overlap必须在0和chunk_size之间")
    
#     chunks = []
#     current_pos = 0
#     text_length = len(text)
    
#     while current_pos < text_length:
#         # 计算块的结束位置
#         end_pos = min(current_pos + chunk_size, text_length)
        
#         # 如果还有剩余文本且不是最后一块
#         if end_pos < text_length:
#             # 尝试在句子边界处分割
#             if sentence_boundary:
#                 # 查找最近的句子结束标点
#                 sentence_end = _find_sentence_boundary(text, end_pos)
#                 if sentence_end is not None and sentence_end > current_pos:
#                     end_pos = sentence_end
            
#             # 如果没有找到句子边界或不允许在句子边界分割，则在单词边界分割
#             if end_pos == current_pos + chunk_size:
#                 # 查找最近的单词边界
#                 word_boundary = _find_word_boundary(text, end_pos)
#                 if word_boundary is not None and word_boundary > current_pos:
#                     end_pos = word_boundary
        
#         # 添加当前块
#         chunks.append(text[current_pos:end_pos].strip())
        
#         # 更新位置，考虑重叠
#         current_pos = end_pos - overlap if end_pos - overlap > current_pos else end_pos
    
#     return chunks

# def _find_sentence_boundary(text: str, position: int) -> int | None:
#     """
#     查找最近的句子边界位置
    
#     参数:
#         text: 文本
#         position: 开始查找的位置
    
#     返回:
#         最近的句子结束位置，如果没有找到则返回None
#     """
#     # 查找position之后最近的句子结束标点
#     match = re.search(r'[.!?]\s+', text[position:])
#     if match:
#         return position + match.end()
    
#     # 如果没有找到，尝试向前查找
#     match = re.search(r'[.!?]\s+', text[:position][::-1])
#     if match:
#         return position - match.start()
    
#     return None

# def _find_word_boundary(text: str, position: int) -> int | None:
#     """
#     查找最近的单词边界位置
    
#     参数:
#         text: 文本
#         position: 开始查找的位置
    
#     返回:
#         最近的单词边界位置，如果没有找到则返回None
#     """
#     # 查找position之后最近的空白字符
#     match = re.search(r'\s', text[position:])
#     if match:
#         return position + match.start()
    
#     # 如果没有找到，尝试向前查找
#     match = re.search(r'\s', text[:position][::-1])
#     if match:
#         return position - match.start()
    
#     return None

import re

def chunk_text(
    text: str, 
    chunk_size: int = 1000, 
    overlap: int = 100, 
    sentence_boundary: bool = True
) -> list[str]:
    """
    将文本分割成指定大小的块，尽量不切断单词和句子
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size必须大于0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap必须在0和chunk_size之间")
    
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        end_pos = min(current_pos + chunk_size, text_length)
        
        # 如果是最后一块，直接添加并退出
        if end_pos == text_length:
            chunk = text[current_pos:end_pos].strip()
            if chunk:  # 避免添加空字符串
                chunks.append(chunk)
            break
        
        # 尝试在合适的位置分割
        if sentence_boundary:
            # 优先在句子边界分割
            boundary_pos = _find_sentence_boundary(text, current_pos, end_pos)
            if boundary_pos is not None and boundary_pos > current_pos:
                end_pos = boundary_pos
            else:
                # 退回到单词边界
                boundary_pos = _find_word_boundary(text, current_pos, end_pos)
                if boundary_pos is not None and boundary_pos > current_pos:
                    end_pos = boundary_pos
        
        chunk = text[current_pos:end_pos].strip()
        if chunk:  # 避免添加空字符串
            chunks.append(chunk)
        
        # 更新位置，确保不会倒退或产生负值
        current_pos = max(current_pos + 1, end_pos - overlap)
    
    return chunks

def _find_sentence_boundary(text: str, start_pos: int, end_pos: int) -> int | None:
    """
    在指定范围内查找句子边界
    """
    # 在当前位置向后查找句子结束符
    segment = text[start_pos:end_pos]
    match = re.search(r'[.!?][\s\n]+', segment)
    if match:
        return start_pos + match.end()
    
    # 向前查找（从当前位置向前搜索）
    for i in range(end_pos - 1, start_pos, -1):
        if re.match(r'[.!?][\s\n]*$', text[i-1:i+1]):
            # 找到句子结束符后，继续找到空白字符结束的位置
            j = i
            while j < len(text) and text[j].isspace():
                j += 1
            return j
    
    return None

def _find_word_boundary(text: str, start_pos: int, end_pos: int) -> int | None:
    """
    在指定范围内查找单词边界
    """
    # 向后查找空白字符
    for i in range(end_pos - 1, start_pos, -1):
        if text[i].isspace():
            return i + 1  # 返回空白字符后的位置
    
    # 如果找不到，就在当前位置分割（避免无限循环）
    return end_pos