import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
import hashlib
import logging
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Chunk:
    """块数据类"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_type: str = "base"  # 'small' for retrieval, 'large' for generation
    embedding: Optional[np.ndarray] = None

class QwenEmbeddingAdapter:
    """Qwen3 Embedding 服务适配器（假设已实现）"""
    
    def __init__(self, base_url: str, api_key: str = None, timeout: int = 30):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        # 这里假设已经实现了客户端初始化
        self._initialized = True
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        获取文本嵌入向量
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        # 这里调用你实际实现的 Qwen3 Embedding 服务
        # 返回格式: [[v1, v2, ...], [v1, v2, ...], ...]
        try:
            # 示例实现 - 请替换为你的实际调用代码
            embeddings = []
            for text in texts:
                # 这里应该是调用你的 Qwen3 Embedding API
                # embedding = self._call_qwen_embedding_api(text)
                # 暂时用随机向量模拟
                dummy_embedding = np.random.randn(1024).tolist()
                embeddings.append(dummy_embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to get embeddings from Qwen: {e}")
            raise
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 文本1
            text2: 文本2
            
        Returns:
            相似度分数 (0-1)
        """
        try:
            embeddings = self.get_embeddings([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            # 将相似度归一化到 0-1 范围
            return max(0.0, min(1.0, (similarity + 1) / 2))
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0

class AdvancedTextSplitter:
    """
    高级文本切分器 - 集成 Qwen3 Embedding
    采用分层分块 + 语义分块策略最大化 RAG 召回率
    """
    
    def __init__(self, 
                 embedding_client: Optional[QwenEmbeddingAdapter] = None,
                 small_chunk_size: int = 256,
                 large_chunk_size: int = 1024,
                 chunk_overlap: int = 50,
                 semantic_threshold: float = 0.75,
                 max_sentences_per_chunk: int = 20,
                 enable_semantic_split: bool = True,
                 enable_hierarchical: bool = True):
        """
        初始化
        
        Args:
            embedding_client: Qwen3 Embedding 客户端适配器
            small_chunk_size: 小块大小（用于检索，字符数）
            large_chunk_size: 大块大小（用于生成，字符数）
            chunk_overlap: 块重叠大小（字符数）
            semantic_threshold: 语义分割阈值
            max_sentences_per_chunk: 每个块最大句子数
            enable_semantic_split: 是否启用语义分割
            enable_hierarchical: 是否启用分层分块
        """
        self.small_chunk_size = small_chunk_size
        self.large_chunk_size = large_chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_threshold = semantic_threshold
        self.max_sentences_per_chunk = max_sentences_per_chunk
        self.enable_semantic_split = enable_semantic_split
        self.enable_hierarchical = enable_hierarchical
        
        # 中文分隔符优先级
        self.separators = [
            "\n\n", "\n", "。", "！", "？", "；", "……", "…", "．", 
            "，", "、", "\t", " ", ""
        ]
        
        # 章节标题正则模式
        self.heading_patterns = [
            r'^第[零一二三四五六七八九十百千\d]+[章节条].*?$',
            r'^[一二三四五六七八九十]、.*?$',
            r'^\(\d+\)\s.*?$',
            r'^\d+\.\d+\s.*?$',
            r'^\d+\.\s.*?$',
            r'^[A-Z][A-Z\s]+\s*$',
            r'^##\s.*?$',
            r'^#\s.*?$'
        ]
        
        # 特殊段落类型识别
        self.special_paragraph_patterns = {
            'code_block': r'```[\s\S]*?```',
            'table': r'\|.*?\|.*?\n',
            'list_item': r'^[\s]*[-*•]\s+',
            'number_list': r'^[\s]*\d+\.\s+'
        }
        
        # 使用 Qwen3 Embedding 客户端
        self.embedding_client = embedding_client
        
        # 验证配置
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数"""
        if self.enable_semantic_split and self.embedding_client is None:
            logger.warning("Semantic split enabled but no embedding client provided. Disabling semantic split.")
            self.enable_semantic_split = False
        
        if self.chunk_overlap >= min(self.small_chunk_size, self.large_chunk_size) // 2:
            logger.warning("Chunk overlap is too large, may cause excessive duplication")
    
    def _generate_chunk_id(self, content: str, suffix: str = "") -> str:
        """生成块ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
        return f"{content_hash}_{suffix}" if suffix else content_hash
    
    def _detect_document_structure(self, text: str) -> List[Tuple[str, int, int, int]]:
        """
        检测文档结构，识别章节标题
        
        Returns:
            List of (heading, start_pos, end_pos, level)
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            # 检查是否是标题
            level = 0
            is_heading = False
            
            for pattern in self.heading_patterns:
                if re.match(pattern, line_stripped):
                    is_heading = True
                    level = self._estimate_heading_level(line_stripped)
                    break
            
            # 启发式规则：短行且包含特定关键词
            if (not is_heading and len(line_stripped) < 50 and 
                any(keyword in line_stripped for keyword in 
                    ['摘要', '引言', '背景', '方法', '结果', '讨论', '结论', 
                     '参考文献', '附录', '致谢', '目录', '前言'])):
                is_heading = True
                level = 2
            
            if is_heading:
                # 计算在原始文本中的位置
                start_pos = text.find(line)
                if start_pos != -1:
                    headings.append((line_stripped, start_pos, start_pos + len(line), level))
        
        return headings
    
    def _estimate_heading_level(self, heading: str) -> int:
        """估计标题级别"""
        if re.match(r'^第[零一二三四五六七八九十百千\d]+章', heading):
            return 1
        elif re.match(r'^第[零一二三四五六七八九十百千\d]+节', heading):
            return 2
        elif re.match(r'^[一二三四五六七八九十]、', heading):
            return 3
        elif re.match(r'^\d+\.\d+', heading):
            return 4
        elif re.match(r'^\d+\.', heading):
            return 5
        else:
            return 6
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 保护特殊段落（代码块、表格等）
        protected_segments = {}
        
        # 标记代码块
        code_blocks = list(re.finditer(self.special_paragraph_patterns['code_block'], text, re.DOTALL))
        for i, match in enumerate(code_blocks):
            placeholder = f"__CODE_BLOCK_{i}__"
            protected_segments[placeholder] = match.group()
            text = text.replace(match.group(), placeholder)
        
        # 中文句子分割
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in {'。', '！', '？', '；', '\n', '.', '!', '?', ';'}:
                # 检查是否在保护段内
                if any(placeholder in current_sentence for placeholder in protected_segments):
                    continue
                
                sentences.append(current_sentence)
                current_sentence = ""
        
        if current_sentence:
            sentences.append(current_sentence)
        
        # 恢复保护段
        restored_sentences = []
        for sentence in sentences:
            restored_sentence = sentence
            for placeholder, original in protected_segments.items():
                restored_sentence = restored_sentence.replace(placeholder, original)
            restored_sentences.append(restored_sentence)
        
        # 合并过短的句子
        merged_sentences = []
        buffer = ""
        for sentence in restored_sentences:
            if len(buffer + sentence) < self.small_chunk_size // 2:
                buffer += sentence
            else:
                if buffer:
                    merged_sentences.append(buffer)
                buffer = sentence
        
        if buffer:
            merged_sentences.append(buffer)
        
        return merged_sentences
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        调用 Qwen3 Embedding 服务获取嵌入向量
        """
        if not self.embedding_client:
            raise ValueError("Embedding client is not initialized")
        
        try:
            embeddings_list = self.embedding_client.get_embeddings(texts)
            return np.array(embeddings_list)
        except Exception as e:
            logger.error(f"Failed to get embeddings from Qwen: {e}")
            raise
    
    def _semantic_split(self, text: str, is_large_chunk: bool = False) -> List[str]:
        """
        语义分块 - 使用 Qwen3 Embedding
        """
        if not self.enable_semantic_split or not self.embedding_client:
            return [text]
        
        try:
            # 先按句子分割
            sentences = self._split_into_sentences(text)
            if len(sentences) <= 1:
                return [text]
            
            # 限制句子数量以避免过多API调用
            if len(sentences) > self.max_sentences_per_chunk:
                logger.info(f"Too many sentences ({len(sentences)}), using recursive split instead")
                return self._recursive_split(text, self.large_chunk_size if is_large_chunk else self.small_chunk_size)
            
            # 计算句子嵌入
            sentence_embeddings = self._get_embeddings(sentences)
            
            # 计算相邻句子相似度
            similarities = []
            for i in range(len(sentences) - 1):
                sim = cosine_similarity(
                    [sentence_embeddings[i]], 
                    [sentence_embeddings[i + 1]]
                )[0][0]
                # 归一化到 0-1
                similarities.append(max(0.0, min(1.0, (sim + 1) / 2)))
            
            # 根据相似度确定分割点
            split_points = []
            for i, sim in enumerate(similarities):
                if sim < self.semantic_threshold:
                    split_points.append(i + 1)
            
            # 根据块类型调整分割策略
            if is_large_chunk:
                # 对于大块，减少分割点，保持更大的上下文
                if similarities:
                    avg_sim = np.mean(similarities)
                    split_points = [p for p in split_points if similarities[p-1] < avg_sim * 0.8]
            
            # 构建块
            chunks = []
            start = 0
            for point in split_points:
                chunk = "".join(sentences[start:point])
                if len(chunk.strip()) >= 10:  # 最小长度限制
                    chunks.append(chunk)
                start = point
            
            # 添加最后一个块
            if start < len(sentences):
                chunk = "".join(sentences[start:])
                if len(chunk.strip()) >= 10:
                    chunks.append(chunk)
            
            return chunks if chunks else [text]
            
        except Exception as e:
            logger.warning(f"Semantic split failed: {e}, fallback to recursive split")
            return self._recursive_split(text, self.large_chunk_size if is_large_chunk else self.small_chunk_size)
    
    def _recursive_split(self, text: str, chunk_size: int) -> List[str]:
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
                        if split.strip():  # 忽略空字符串
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
                            # 如果单个split就超过chunk_size，需要进一步分割
                            if len(split) > chunk_size:
                                sub_chunks = self._recursive_split(split, chunk_size)
                                chunks.extend(sub_chunks[:-1])
                                current_chunk = sub_chunks[-1] if sub_chunks else ""
                            else:
                                current_chunk = split
                    
                    if current_chunk:
                        chunks.append(current_chunk)
                    
                    # 如果分割结果合理，返回
                    if len(chunks) > 1 and all(len(chunk) <= chunk_size * 1.1 for chunk in chunks):
                        return chunks
        
        # 如果没有合适的分隔符，按字符分割
        chunks = []
        for i in range(0, len(text), chunk_size - self.chunk_overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    def _split_by_headings(self, text: str, headings: List[Tuple[str, int, int, int]], 
                          metadata: Dict[str, Any]) -> List[Chunk]:
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
                        chunk_id=self._generate_chunk_id(initial_content, "intro")
                    ))
        
        # 基于标题分割
        for i in range(len(headings)):
            heading, start_pos, end_pos, level = headings[i]
            
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
                chunk_metadata['heading_level'] = level
                chunk_metadata['section_index'] = i
                
                chunks.append(Chunk(
                    content=full_content,
                    metadata=chunk_metadata,
                    chunk_id=self._generate_chunk_id(full_content, f"sec_{i}")
                ))
        
        return chunks
    
    def _create_hierarchical_chunks(self, large_chunks: List[Chunk]) -> List[Chunk]:
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
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        主分割方法
        
        Args:
            text: 输入文本
            metadata: 元数据
            
        Returns:
            List of Chunk objects
        """
        if metadata is None:
            metadata = {}
        
        logger.info(f"Splitting text of length: {len(text)}")
        
        # 预处理文本
        text = self._preprocess_text(text)
        
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
                    chunk_id=self._generate_chunk_id(chunk_text, "recursive")
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
                    chunk_id=self._generate_chunk_id(semantic_chunk, "semantic")
                )
                optimized_large_chunks.append(optimized_chunk)
        
        # 4. 创建分层块结构
        if self.enable_hierarchical:
            final_chunks = self._create_hierarchical_chunks(optimized_large_chunks)
        else:
            final_chunks = optimized_large_chunks
        
        # 5. 后处理：过滤空块，添加统计信息
        final_chunks = [chunk for chunk in final_chunks if chunk.content.strip()]
        self._add_chunk_statistics(final_chunks)
        
        logger.info(f"Created {len([c for c in final_chunks if c.chunk_type == 'small'])} small chunks")
        logger.info(f"Created {len([c for c in final_chunks if c.chunk_type == 'large'])} large chunks")
        
        return final_chunks
    
    def _preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 合并多个空行
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # 去除首尾空白
        text = text.strip()
        
        return text
    
    def _add_chunk_statistics(self, chunks: List[Chunk]):
        """为块添加统计信息"""
        for chunk in chunks:
            chunk.metadata['char_count'] = len(chunk.content)
            chunk.metadata['word_count'] = len(chunk.content.split())
            chunk.metadata['line_count'] = chunk.content.count('\n') + 1

# 使用示例
def example_usage():
    """使用示例"""
    # 初始化 Qwen3 Embedding 客户端
    qwen_client = QwenEmbeddingAdapter(
        base_url="http://localhost:8080",
        api_key="your-api-key"
    )
    
    # 创建分块器
    splitter = AdvancedTextSplitter(
        embedding_client=qwen_client,
        small_chunk_size=256,
        large_chunk_size=1024,
        chunk_overlap=50,
        semantic_threshold=0.75,
        enable_semantic_split=True,
        enable_hierarchical=True
    )
    
    # 读取 TXT 文件
    with open('example.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 分割文本
    chunks = splitter.split_text(text, {
        'source': 'example.txt',
        'document_type': 'technical',
        'language': 'zh'
    })
    
    # 分别处理小块和大块
    small_chunks = [chunk for chunk in chunks if chunk.chunk_type == 'small']
    large_chunks = {chunk.chunk_id: chunk for chunk in chunks if chunk.chunk_type == 'large'}
    
    print(f"生成 {len(small_chunks)} 个小块用于检索")
    print(f"生成 {len(large_chunks)} 个大块用于生成")
    
    # 示例：展示第一个小块的信息
    if small_chunks:
        first_small = small_chunks[0]
        print(f"\n第一个小块:")
        print(f"内容: {first_small.content[:100]}...")
        print(f"元数据: {first_small.metadata}")
        print(f"父块ID: {first_small.parent_id}")
    
    return chunks

if __name__ == "__main__":
    chunks = example_usage()