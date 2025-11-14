import re
import numpy as np
import hashlib
from typing import Any
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

@dataclass
class Chunk:
    """块数据类"""
    content: str
    metadata: dict[str, Any]
    chunk_id: str
    parent_id: str | None = None
    chunk_type: str = "base"  # 'small' for retrieval, 'large' for generation

class AdvancedTextSplitter:
    """
    高级文本切分器
    采用分层分块 + 语义分块策略最大化 RAG 召回率
    """
    
    def __init__(self, 
                 embedding_model_name: str = "BAAI/bge-small-zh-v1.5",
                 small_chunk_size: int = 256,
                 large_chunk_size: int = 1024,
                 chunk_overlap: int = 50,
                 semantic_threshold: float = 0.75,
                 enable_semantic_split: bool = True):
        """
        初始化
        
        Args:
            embedding_model_name: 语义嵌入模型
            small_chunk_size: 小块大小（用于检索）
            large_chunk_size: 大块大小（用于生成）
            chunk_overlap: 块重叠大小
            semantic_threshold: 语义分割阈值
            enable_semantic_split: 是否启用语义分割
        """
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.enable_semantic_split = enable_semantic_split
        
        # 中文分隔符优先级
        self.separators = [
            "\n\n", "\n", "。", "！", "？", "；", "……", "…", "．", "．", 
            "．", "．", "．", "．", "．", "．", "．", "．", "．", "．", 
            "，", "、", " ", ""
        ]
        
        # 章节标题正则模式
        self.heading_patterns = [
            r'^第[零一二三四五六七八九十百千\d]+[章节条].*?$',
            r'^[一二三四五六七八九十]、.*?$',
            r'^\(\d+\)\s.*?$',
            r'^\d+\.\d+\s.*?$',
            r'^\d+\.\s.*?$',
            r'^[A-Z][A-Z\s]+\s*$'
        ]
        
        # 加载嵌入模型（用于语义分块）
        if self.enable_semantic_split:
            logger.info(f"Loading embedding model: {embedding_model_name}")
            self.embedding_model = SentenceTransformer(embedding_model_name)
        else:
            self.embedding_model = None
    
    def _generate_chunk_id(self, content: str) -> str:
        """生成块ID"""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _detect_document_structure(self, text: str) -> list[tuple[str, int, int]]:
        """
        检测文档结构，识别章节标题
        
        Returns:
            list of (heading, start_pos, end_pos)
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # 检查是否是标题
            is_heading = False
            for pattern in self.heading_patterns:
                if re.match(pattern, line_stripped):
                    is_heading = True
                    break
            
            # 启发式规则：短行且包含特定关键词
            if len(line_stripped) < 50 and any(keyword in line_stripped for keyword in 
                                            ['摘要', '引言', '背景', '方法', '结果', '讨论', '结论', '参考文献']):
                is_heading = True
            
            if is_heading:
                # 计算在原始文本中的位置
                start_pos = text.find(line)
                if start_pos != -1:
                    headings.append((line_stripped, start_pos, start_pos + len(line)))
        
        return headings
    
    def _semantic_split(self, text: str, is_large_chunk: bool = False) -> list[str]:
        """
        语义分块
        
        Args:
            text: 输入文本
            is_large_chunk: 是否是大块（用于生成）
        """
        if not self.enable_semantic_split or not self.embedding_model:
            return [text]
        
        try:
            # 先按句子分割
            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                return [text]
            
            # 计算句子嵌入
            sentence_embeddings = self.embedding_model.encode(sentences)
            
            # 计算相邻句子相似度
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(
                    [sentence_embeddings[i]], 
                    [sentence_embeddings[i + 1]]
                )[0][0]
                similarities.append(sim)
            
            # 根据相似度确定分割点
            split_points = []
            for i, sim in enumerate(similarities):
                if sim < self.semantic_threshold:
                    split_points.append(i + 1)  # 在i和i+1之间分割
            
            # 根据是否是大型块调整分割策略
            if is_large_chunk:
                # 对于大块，减少分割点，保持更大的上下文
                avg_sim = np.mean(similarities)
                split_points = [p for i, p in enumerate(split_points) 
                              if similarities[p-1] < avg_sim * 0.8]
            
            # 构建块
            chunks = []
            start = 0
            for point in split_points:
                chunk = "".join(sentences[start:point])
                if chunk.strip():
                    chunks.append(chunk)
                start = point
            
            # 添加最后一个块
            if start < len(sentences):
                chunk = "".join(sentences[start:])
                if chunk.strip():
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            logger.warning(f"Semantic split failed: {e}, fallback to recursive split")
            return [text]
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """将文本分割成句子"""
        # 中文句子分割
        sentences = re.split(r'([。！？；\.!?;])', text)
        result = []
        buffer = ""
        
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            buffer += sentence
            
            # 如果缓冲区达到最小块大小，或者遇到明显的分割点
            if len(buffer) >= self.small_chunk_size // 3 or i == len(sentences) - 2:
                result.append(buffer)
                buffer = ""
        
        if buffer:
            result.append(buffer)
        
        return result
    
    def _recursive_split(self, text: str, chunk_size: int) -> list[str]:
        """递归分块"""
        if len(text) <= chunk_size:
            return [text]
        
        # 尝试在不同分隔符处分割
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                if len(splits) > 1:
                    # 重建分割，保留分隔符
                    reconstructed_splits = []
                    for i, split in enumerate(splits):
                        if i < len(splits) - 1:
                            reconstructed_splits.append(split + separator)
                        else:
                            reconstructed_splits.append(split)
                    
                    # 递归处理每个部分
                    chunks = []
                    current_chunk = ""
                    
                    for split in reconstructed_splits:
                        if len(current_chunk + split) <= chunk_size:
                            current_chunk += split
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = split
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # 如果分割结果合理，返回
                    if len(chunks) > 1 and all(len(chunk) <= chunk_size for chunk in chunks):
                        return chunks
        
        # 如果没有合适的分隔符，强制分割
        chunks = []
        for i in range(0, len(text), chunk_size - self.chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _create_hierarchical_chunks(self, large_chunks: list[Chunk]) -> list[Chunk]:
        """创建分层块结构"""
        all_chunks = []
        
        for large_chunk in large_chunks:
            # 为大块创建小块（用于检索）
            small_chunks_text = self._recursive_split(
                large_chunk.content, 
                self.small_chunk_size
            )
            
            # 对小块进行语义优化
            optimized_small_chunks = []
            for small_chunk_text in small_chunks_text:
                semantic_chunks = self._semantic_split(small_chunk_text, is_large_chunk=False)
                optimized_small_chunks.extend(semantic_chunks)
            
            # 创建小块对象
            for i, chunk_text in enumerate(optimized_small_chunks):
                small_chunk = Chunk(
                    content=chunk_text,
                    metadata=large_chunk.metadata.copy(),
                    chunk_id=self._generate_chunk_id(f"small_{large_chunk.chunk_id}_{i}"),
                    parent_id=large_chunk.chunk_id,
                    chunk_type="small"
                )
                all_chunks.append(small_chunk)
            
            # 保留大块（用于生成）
            large_chunk.chunk_type = "large"
            all_chunks.append(large_chunk)
        
        return all_chunks
    
    def split_text(self, text: str, metadata: dict[str, Any] = None) -> list[Chunk]:
        """
        主分割方法
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            list of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        logger.info(f"Splitting text of length: {len(text)}")
        
        # 1. 检测文档结构
        headings = self._detect_document_structure(text)
        logger.info(f"Detected {len(headings)} headings in document")
        
        # 2. 基于结构进行初始分割
        if headings:
            large_chunks = self._split_by_headings(text, headings, metadata)
        else:
            # 没有检测到结构，使用递归分割创建大块
            large_chunk_texts = self._recursive_split(text, self.large_chunk_size)
            large_chunks = [
                Chunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    chunk_id=self._generate_chunk_id(chunk_text)
                )
                for chunk_text in large_chunk_texts
            ]
        
        # 3. 对每个大块进行语义优化
        optimized_large_chunks = []
        for large_chunk in large_chunks:
            semantic_chunks = self._semantic_split(large_chunk.content, is_large_chunk=True)
            for semantic_chunk in semantic_chunks:
                optimized_chunk = Chunk(
                    content=semantic_chunk,
                    metadata=large_chunk.metadata.copy(),
                    chunk_id=self._generate_chunk_id(semantic_chunk)
                )
                optimized_large_chunks.append(optimized_chunk)
        
        # 4. 创建分层块结构
        hierarchical_chunks = self._create_hierarchical_chunks(optimized_large_chunks)
        
        logger.info(f"Created {len([c for c in hierarchical_chunks if c.chunk_type == 'small'])} small chunks")
        logger.info(f"Created {len([c for c in hierarchical_chunks if c.chunk_type == 'large'])} large chunks")
        
        return hierarchical_chunks
    
    def _split_by_headings(self, text: str, headings: list[tuple[str, int, int]], 
                          metadata: dict[str, Any]) -> list[Chunk]:
        """基于标题分割文本"""
        chunks = []
        
        # 添加开始到第一个标题
        if headings:
            first_heading_start = headings[0][1]
            if first_heading_start > 0:
                initial_content = text[:first_heading_start].strip()
                if initial_content:
                    chunks.append(Chunk(
                        content=initial_content,
                        metadata=metadata.copy(),
                        chunk_id=self._generate_chunk_id(initial_content)
                    ))
        
        # 基于标题分割
        for i in range(len(headings)):
            heading, start_pos, end_pos = headings[i]
            
            # 确定当前块的结束位置
            if i < len(headings) - 1:
                next_start = headings[i + 1][1]
                content = text[end_pos:next_start].strip()
            else:
                content = text[end_pos:].strip()
            
            # 包含标题的完整内容
            full_content = text[start_pos:end_pos] + "\n" + content
            
            if full_content.strip():
                chunk_metadata = metadata.copy()
                chunk_metadata['heading'] = heading
                chunk_metadata['heading_level'] = self._estimate_heading_level(heading)
                
                chunks.append(Chunk(
                    content=full_content,
                    metadata=chunk_metadata,
                    chunk_id=self._generate_chunk_id(full_content)
                ))
        
        return chunks
    
    def _estimate_heading_level(self, heading: str) -> int:
        """估计标题级别"""
        if re.match(r'^第[零一二三四五六七八九十百千\d]+章', heading):
            return 1
        elif re.match(r'^第[零一二三四五六七八九十百千\d]+节', heading):
            return 2
        elif re.match(r'^[一二三四五六七八九十]、', heading):
            return 3
        elif re.match(r'^\d+\.', heading):
            return 4
        else:
            return 5

# 使用示例
def example_usage():
    """使用示例"""
    splitter = AdvancedTextSplitter(
        small_chunk_size=256,
        large_chunk_size=1024,
        chunk_overlap=50,
        semantic_threshold=0.75,
        enable_semantic_split=True  # 根据硬件条件调整
    )
    
    # 读取 TXT 文件
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分割文本
    chunks = splitter.split_text(text, {
        'source': 'example.txt',
        'document_type': 'technical'
    })
    
    # 分别处理小块和大块
    small_chunks = [chunk for chunk in chunks if chunk.chunk_type == 'small']
    large_chunks = {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_type == 'large'}
    
    print(f"生成 {len(small_chunks)} 个小块用于检索")
    print(f"生成 {len(large_chunks)} 个大块用于生成")
    
    return chunks

if __name__ == "__main__":
    chunks = example_usage()