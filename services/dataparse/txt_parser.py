import uuid
import re
import jieba
import langid
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger
from .common import *
from .summary_parser import SummaryParser
from .file_params import FileParams
from dao.entities.kbot_biz_txt_embedding import KbotBizTxtEmbedding
from core.dictionary import FileStatus, ChunkType, SplitStrategy
from utils.call_models import CallModel


class EnhancedTextSplitter:
    """
    增强版文本切割器
    优化语义完整性，避免重复切片和句子切断问题
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 strategy: str = "semantic",
                 min_chunk_size: int = 100,
                 semantic_threshold: float = 0.65,
                 max_chunk_size: int = 800):
        """
        初始化增强版文本切割器
        
        Args:
            chunk_size: 目标块大小（字符数）
            chunk_overlap: 块之间重叠大小
            strategy: 切割策略 (semantic, structural, hybrid)
            min_chunk_size: 最小块大小
            semantic_threshold: 语义分割的相似度阈值
            max_chunk_size: 最大块大小硬限制
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.min_chunk_size = min_chunk_size
        self.semantic_threshold = semantic_threshold
        self.max_chunk_size = max_chunk_size
        
        # 初始化中文分词器
        try:
            jieba.initialize()
        except:
            pass
        
        # 缓存相似度计算
        self.similarity_cache = {}
    
    def split_text(self, text: str) -> list[str]:
        """
        主切割方法：确保语义完整性的智能分割
        """
        if not text or not text.strip():
            return []
        
        # 检测文本语言
        lang = self._detect_language(text)
        
        # 预处理文本
        cleaned_text = self._preprocess_text(text, lang)
        
        # 根据策略选择切割方法
        if self.strategy == "semantic":
            chunks = self._enhanced_semantic_split(cleaned_text, lang)
        elif self.strategy == "structural":
            chunks = self._enhanced_structural_split(cleaned_text, lang)
        else:  # hybrid
            chunks = self._enhanced_hybrid_split(cleaned_text, lang)
        
        # 后处理：移除空块和过小块，确保质量
        return self._postprocess_chunks(chunks, lang)
    
    def _detect_language(self, text: str) -> str:
        """检测文本语言"""
        sample = text[:500]
        lang, _ = langid.classify(sample)
        return lang
    
    def _preprocess_text(self, text: str, lang: str) -> str:
        """增强文本预处理"""
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 合并过多空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 中文文本特定预处理
        if lang == 'zh':
            # 保留原始格式，仅在必要处添加空格
            text = re.sub(r'([\u4e00-\u9fff])([A-Za-z0-9])', r'\1 \2', text)
            text = re.sub(r'([A-Za-z0-9])([\u4e00-\u9fff])', r'\1 \2', text)
        
        return text.strip()
    
    def _enhanced_semantic_split(self, text: str, lang: str) -> list[str]:
        """
        增强版语义分割
        使用动态窗口和边界检测确保语义完整性
        """
        # 首先按句子分割
        sentences = self._split_into_sentences(text, lang)
        
        if len(sentences) <= 1:
            return [text] if self.min_chunk_size <= len(text) <= self.max_chunk_size else []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i].strip()
            if not sentence:
                i += 1
                continue
                
            sent_length = len(sentence)
            
            # 如果单句就超过最大限制，需要特殊处理
            if sent_length > self.max_chunk_size:
                if current_chunk:
                    # 先保存当前块
                    chunk_text = self._join_sentences(current_chunk, lang)
                    if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
                
                # 对长句子进行强制分割
                sub_chunks = self._split_long_sentence(sentence, lang)
                for sub_chunk in sub_chunks:
                    if self.min_chunk_size <= len(sub_chunk) <= self.max_chunk_size:
                        chunks.append(sub_chunk)
                i += 1
                continue
            
            # 检查添加该句子是否会超过限制
            if current_length + sent_length > self.chunk_size:
                # 寻找最佳分割点
                if current_chunk:
                    best_split_index = self._find_best_split_point(current_chunk, sentences, i, lang)
                    
                    if best_split_index > 0:
                        # 在最佳点分割
                        chunk_text = self._join_sentences(current_chunk[:best_split_index], lang)
                        if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                            chunks.append(chunk_text)
                        
                        # 保留重叠部分
                        overlap_start = max(0, best_split_index - self._get_overlap_sentence_count(current_chunk))
                        current_chunk = current_chunk[overlap_start:]
                        current_length = sum(len(s) for s in current_chunk)
                    else:
                        # 没有找到好的分割点，强制分割
                        chunk_text = self._join_sentences(current_chunk, lang)
                        if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                            chunks.append(chunk_text)
                        current_chunk = []
                        current_length = 0
            
            # 添加当前句子到块中
            current_chunk.append(sentence)
            current_length += sent_length
            i += 1
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = self._join_sentences(current_chunk, lang)
            if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _enhanced_structural_split(self, text: str, lang: str) -> list[str]:
        """
        增强版结构分割
        更智能的段落和标题识别
        """
        # 按段落分割
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for i, paragraph in enumerate(paragraphs):
            para_length = len(paragraph)
            
            # 判断段落属性
            is_heading = self._is_heading(paragraph)
            is_important = self._is_important_paragraph(paragraph, i, len(paragraphs))
            
            # 如果当前段落是标题或重要段落，考虑分割
            should_split = False
            if is_heading and current_length >= self.min_chunk_size:
                should_split = True
            elif is_important and current_length + para_length > self.chunk_size and current_length >= self.min_chunk_size:
                should_split = True
            elif current_length + para_length > self.max_chunk_size:
                should_split = True
            
            if should_split and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                    chunks.append(chunk_text)
                
                # 开始新块，重要段落单独成块
                if is_heading or is_important:
                    current_chunk = [paragraph]
                    current_length = para_length
                else:
                    current_chunk = [paragraph]
                    current_length = para_length
            else:
                current_chunk.append(paragraph)
                current_length += para_length + 2
        
        # 处理最后一个块
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if self.min_chunk_size <= len(chunk_text) <= self.max_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def _enhanced_hybrid_split(self, text: str, lang: str) -> list[str]:
        """
        增强版混合分割
        结合结构和语义信息，动态调整分割策略
        """
        # 首先进行结构分割
        structural_chunks = self._enhanced_structural_split(text, lang)
        
        final_chunks = []
        
        for chunk in structural_chunks:
            chunk_length = len(chunk)
            
            # 根据长度决定是否需要进行语义细分
            if chunk_length > self.chunk_size * 1.2:
                # 对较大的结构块进行语义细分
                semantic_subchunks = self._enhanced_semantic_split(chunk, lang)
                final_chunks.extend(semantic_subchunks)
            elif chunk_length < self.min_chunk_size and final_chunks:
                # 小段落合并到前一个块（如果语义相关）
                if self._should_merge_with_previous(final_chunks[-1], chunk, lang):
                    final_chunks[-1] = final_chunks[-1] + "\n\n" + chunk
                else:
                    final_chunks.append(chunk)
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_into_sentences(self, text: str, lang: str) -> list[str]:
        """
        增强版句子分割
        更准确的句子边界检测
        """
        if lang == 'zh':
            # 中文句子分割，考虑更多结束符和特殊情况
            pattern = r'([。！？；\.!?;]+\s*)'
        else:
            # 英文句子分割，避免在缩写处错误分割
            pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
        
        # 使用更复杂的分割逻辑
        parts = re.split(pattern, text)
        
        # 重新组合句子和标点
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if re.match(pattern, part) and current_sentence:
                # 当前部分是结束符，添加到句子末尾
                current_sentence += part
                sentences.append(current_sentence.strip())
                current_sentence = ""
            else:
                # 检查是否需要开始新句子
                if not current_sentence and part.strip():
                    current_sentence = part
                elif current_sentence:
                    current_sentence += part
        
        # 添加最后一个句子
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        return [s for s in sentences if s.strip()]
    
    def _find_best_split_point(self, current_chunk: list[str], all_sentences: list[str], next_index: int, lang: str) -> int:
        """
        在当前位置寻找最佳分割点
        返回在current_chunk中的索引位置
        """
        if len(current_chunk) <= 1:
            return 0
        
        # 策略1：在语义相似度最低的地方分割
        min_similarity = 1.0
        best_split = 0
        
        for i in range(1, len(current_chunk)):
            similarity = self._calculate_sentence_similarity(
                current_chunk[i-1], current_chunk[i], lang
            )
            if similarity < min_similarity:
                min_similarity = similarity
                best_split = i
        
        # 如果找到明显的语义边界，使用该点
        if min_similarity < self.semantic_threshold:
            return best_split
        
        # 策略2：在长度接近目标值的位置分割
        target_length = self.chunk_size - self.chunk_overlap
        current_length = 0
        for i, sentence in enumerate(current_chunk):
            current_length += len(sentence)
            if current_length >= target_length:
                return max(1, i)  # 确保至少有一个句子
        
        return 0  # 不分割
    
    def _split_long_sentence(self, sentence: str, lang: str) -> list[str]:
        """分割过长的句子"""
        if len(sentence) <= self.max_chunk_size:
            return [sentence]
        
        # 按标点符号尝试分割
        if lang == 'zh':
            split_pattern = r'([，,。！？；])'
        else:
            split_pattern = r'([,\.!?;])'
        
        parts = re.split(split_pattern, sentence)
        sub_sentences = []
        current_sub = ""
        
        for i, part in enumerate(parts):
            if re.match(split_pattern, part) and current_sub:
                current_sub += part
                if len(current_sub) >= self.min_chunk_size:
                    sub_sentences.append(current_sub.strip())
                    current_sub = ""
            else:
                if not current_sub and part.strip():
                    current_sub = part
                elif current_sub:
                    current_sub += part
        
        if current_sub.strip():
            sub_sentences.append(current_sub.strip())
        
        # 如果仍然过长，强制按长度分割
        final_chunks = []
        for sub in sub_sentences:
            if len(sub) > self.max_chunk_size:
                # 按字符数均匀分割，但尽量在词语边界
                chunks = self._split_by_length_with_boundary(sub, lang)
                final_chunks.extend(chunks)
            else:
                final_chunks.append(sub)
        
        return final_chunks
    
    def _split_by_length_with_boundary(self, text: str, lang: str) -> list[str]:
        """按长度分割，但尽量在词语边界处分割"""
        chunk_size = self.max_chunk_size
        chunks = []
        
        while len(text) > chunk_size:
            # 寻找最佳分割点
            split_pos = chunk_size
            
            # 向后寻找边界
            for i in range(min(100, len(text) - chunk_size)):
                candidate_pos = chunk_size + i
                if candidate_pos >= len(text):
                    break
                if self._is_good_boundary(text, candidate_pos, lang):
                    split_pos = candidate_pos
                    break
            
            # 向前寻找边界
            if split_pos == chunk_size:
                for i in range(min(50, chunk_size)):
                    candidate_pos = chunk_size - i
                    if candidate_pos <= self.min_chunk_size:
                        break
                    if self._is_good_boundary(text, candidate_pos, lang):
                        split_pos = candidate_pos
                        break
            
            chunk = text[:split_pos].strip()
            if chunk:
                chunks.append(chunk)
            text = text[split_pos:].strip()
        
        if text:
            chunks.append(text)
        
        return chunks
    
    def _is_good_boundary(self, text: str, position: int, lang: str) -> bool:
        """判断位置是否是好的分割边界"""
        if position <= 0 or position >= len(text):
            return False
        
        # 检查标点符号
        prev_char = text[position-1]
        curr_char = text[position] if position < len(text) else ''
        
        boundary_chars = {'。', '！', '？', '；', '.', '!', '?', ';', '，', ','}
        
        if prev_char in boundary_chars:
            return True
        
        # 检查空格（英文）
        if lang != 'zh' and curr_char.isspace():
            return True
        
        # 检查词语边界（中文）
        if lang == 'zh' and position > 0 and position < len(text):
            # 简单的词语边界检查
            return not (text[position-1].isalnum() and text[position].isalnum())
        
        return False
    
    def _calculate_sentence_similarity(self, sent1: str, sent2: str, lang: str) -> float:
        """计算句子相似度，带缓存"""
        cache_key = (sent1, sent2, lang)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        try:
            # 使用TF-IDF
            vectorizer = TfidfVectorizer(min_df=1, max_df=0.8)
            
            if lang == 'zh':
                sent1_cut = ' '.join(jieba.cut(sent1))
                sent2_cut = ' '.join(jieba.cut(sent2))
                vectors = vectorizer.fit_transform([sent1_cut, sent2_cut])
            else:
                vectors = vectorizer.fit_transform([sent1, sent2])
            
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] # type: ignore
            
            self.similarity_cache[cache_key] = similarity
            return similarity
            
        except Exception:
            # 退回基于词汇重叠的相似度
            if lang == 'zh':
                words1 = set(jieba.cut(sent1))
                words2 = set(jieba.cut(sent2))
            else:
                words1 = set(sent1.lower().split())
                words2 = set(sent2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            similarity = intersection / union if union > 0 else 0.0
            self.similarity_cache[cache_key] = similarity
            return similarity
    
    def _is_heading(self, text: str) -> bool:
        """判断是否是标题"""
        lines = text.split('\n')
        first_line = lines[0].strip()
        
        if len(first_line) < 50:
            heading_indicators = [
                r'^#+\s+',  # Markdown标题
                r'^第[一二三四五六七八九十\d]+[章节条]\s*',  # 中文编号
                r'^\d+\.\s+',  # 数字编号
                r'^[A-Z][A-Z\s]{1,30}$',  # 全大写短文本
                r'^(摘要|目录|前言|引言|结论|参考文献|附录)',  # 常见章节标题
            ]
            
            for pattern in heading_indicators:
                if re.match(pattern, first_line):
                    return True
        
        return False
    
    def _is_important_paragraph(self, paragraph: str, index: int, total: int) -> bool:
        """判断段落是否重要"""
        # 开头和结尾的段落通常更重要
        if index == 0 or index == total - 1:
            return True
        
        # 包含关键信息的段落
        important_keywords = ['总结', '结论', '重要', '注意', '关键', '主要', '核心']
        if any(keyword in paragraph for keyword in important_keywords):
            return True
        
        return False
    
    def _should_merge_with_previous(self, previous_chunk: str, current_chunk: str, lang: str) -> bool:
        """判断是否应该与前一块合并"""
        # 如果合并后不会太大，且语义相关，则合并
        combined_length = len(previous_chunk) + len(current_chunk)
        if combined_length > self.max_chunk_size:
            return False
        
        # 计算语义相似度
        similarity = self._calculate_sentence_similarity(
            previous_chunk[-200:],  # 取前一块的结尾部分
            current_chunk[:200],    # 取当前块的开头部分
            lang
        )
        
        return similarity > self.semantic_threshold
    
    def _join_sentences(self, sentences: list[str], lang: str) -> str:
        """根据语言连接句子"""
        if lang == 'zh':
            return ''.join(sentences)
        else:
            return ' '.join(sentences)
    
    def _get_overlap_sentence_count(self, sentences: list[str]) -> int:
        """计算应该重叠的句子数量"""
        if len(sentences) <= 2:
            return 1
        return min(2, max(1, int(len(sentences) * 0.3)))
    
    def _postprocess_chunks(self, chunks: list[str], lang: str) -> list[str]:
        """后处理块，确保质量"""
        result = []
        
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
                
            chunk_length = len(chunk)
            
            # 长度检查
            if chunk_length < self.min_chunk_size:
                # 过小块尝试与相邻块合并
                continue
            elif chunk_length > self.max_chunk_size:
                # 过大的块强制分割
                sub_chunks = self._split_by_length_with_boundary(chunk, lang)
                result.extend([sc for sc in sub_chunks if len(sc) >= self.min_chunk_size])
            else:
                result.append(chunk)
        
        # 处理过小的块（尝试合并）
        return self._merge_small_chunks(result, lang)
    
    def _merge_small_chunks(self, chunks: list[str], lang: str) -> list[str]:
        """合并过小的块"""
        if len(chunks) <= 1:
            return chunks
        
        result = []
        i = 0
        
        while i < len(chunks):
            current_chunk = chunks[i]
            
            if len(current_chunk) < self.min_chunk_size and i < len(chunks) - 1:
                # 尝试与下一块合并
                next_chunk = chunks[i + 1]
                combined = current_chunk + ("\n\n" if lang != 'zh' else "") + next_chunk
                
                if len(combined) <= self.max_chunk_size:
                    result.append(combined)
                    i += 2  # 跳过下一块
                else:
                    result.append(current_chunk)
                    i += 1
            else:
                result.append(current_chunk)
                i += 1
        
        return result


async def process_txt(file_params: FileParams) -> bool:
    """
    处理文本文件，使用增强版文本切割器
    """
    if not await check_text_file(file_params):
        return False
    
    try:
        logger.debug(f"正在处理文本文件: {file_params.file_path}")

        # 1. 读取文本文件
        file_encoding = detect_file_encoding(file_params.file_path)
        logger.debug(f"将使用编码 [{file_encoding}] 读取文件: {file_params.file_path}")

        with open(file_params.file_path, 'r', encoding=file_encoding) as f:
            text = f.read()
        
        # 2. 文本分割
        text_length = len(text)

        if text_length == 0:
            msg = f"解析文件为空，无法处理文件: {file_params.file_path}"
            logger.info(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSED, msg)
            return True
            
        # 参数安全处理
        split_strategy = int(file_params.parser.get("split_strategy", 1))
        chunk_size = int(file_params.parser.get("chunk_size", 500))
        overlap = int(file_params.parser.get("chunk_overlap", 50))

        logger.debug(f"分割策略：{SplitStrategy(split_strategy).name}，分块大小: {chunk_size}, 重叠大小: {overlap}")

        # 使用增强版文本切割器
        strategy_map = {
            SplitStrategy.FIXED_SIZE.value: "hybrid",
            SplitStrategy.DOC_STRUCTURE.value: "structural", 
            SplitStrategy.SEMANTIC.value: "semantic"
        }
        
        strategy_name = strategy_map.get(split_strategy, "hybrid")
        
        splitter = EnhancedTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            strategy=strategy_name,
            min_chunk_size=50,
            max_chunk_size=chunk_size * 2  # 最大不超过目标大小的2倍
        )

        chunks = splitter.split_text(text)
        logger.info(f"成功分割 {len(chunks)} 个文本块")

        # 去重检查
        unique_chunks = []
        seen_chunks = set()
        
        for chunk in chunks:
            chunk_hash = hash(chunk[:100])  # 使用前100字符的哈希作为去重依据
            if chunk_hash not in seen_chunks:
                seen_chunks.add(chunk_hash)
                unique_chunks.append(chunk)
        
        if len(unique_chunks) != len(chunks):
            logger.info(f"移除 {len(chunks) - len(unique_chunks)} 个重复块")
            chunks = unique_chunks

        # 3. 调用 embedding 微服务获取文本 chunk 的向量
        logger.info(f"正在调用 embedding 微服务获取文本 chunk 的向量")
        if file_params.txt_embed_model is None:
            msg = f"文件 {file_params.file_path} 未配置 embedding 模型"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        
        response_data = await CallModel().call_embedding_model(file_params.txt_embed_model, chunks)
        if response_data is None:
            msg = f"获取文件 {file_params.file_path} 的 embedding 向量失败"
            logger.error(msg)
            await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
            return False
        else:
            logger.info(f"成功获取 {len(response_data)} 个 embedding 向量")

            embeddings = [item.embedding for item in response_data]
            embed_entities = []
            summary_result = True
            chunk_num = 1

            for chunk, embedding in zip(chunks, embeddings):
                embed_entity = KbotBizTxtEmbedding(
                    embed_id=str(uuid.uuid4()),
                    chunk_doc=chunk,
                    chunk_metadata={
                        "chunk_type": ChunkType.TEXT, 
                        "chunk_num": chunk_num,
                        "chunk_size": len(chunk),
                        "strategy": strategy_name
                    },
                    biz_metadata=file_params.biz_metadata,
                    file_id=file_params.file_id,
                    kb_id=file_params.kb_id,
                    embedding=embedding,
                    security_level=file_params.security_level,
                    status=1
                )
                embed_entities.append(embed_entity)
                chunk_num += 1
            
            # 保存 embedding 向量到向量数据库
            save_result = await save_embeddings(file_params, embed_entities)
            
            if file_params.enable_summary:
                logger.debug("启用摘要处理")
                summary_result = await SummaryParser.process_summary(file_params=file_params, embed_entities=embed_entities)
            
            return save_result and summary_result
        
    except Exception as e:
        msg = f"处理文本文件 {file_params.file_path} 时发生错误: {str(e)}"
        logger.exception(msg)  
        await update_file_status(file_params.file_id, FileStatus.PARSE_FAILED, msg)
        return False