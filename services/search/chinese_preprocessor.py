import re
import jieba
import jieba.posseg as pseg
from pathlib import Path
from loguru import logger
from configuration import ConfigManager
from utils.call_models import CallModel

class ChinesePreprocessor:
    """
    查询预处理类
    主要针对中文环境，兼顾英文专业词汇
    """
    
    def __init__(self):
        """
        初始化预处理器
        """

        model_config = ConfigManager.get_model_config()
        self.stopwords_file = model_config.tokenizer.stop_words_path
        self.custom_dict_file = model_config.tokenizer.custom_dict_path
        self.stopwords: set[str] = self._load_stopwords(self.stopwords_file)
        self._setup_jieba(self.custom_dict_file)

        
    def _load_stopwords(self, file_path: str) -> set[str]:
        """加载停用词表"""
        stopwords = set()
        try:
            path = Path(file_path)
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    stopwords = {line.strip() for line in f if line.strip()}
                logger.info(f"成功从 {file_path} 加载 {len(stopwords)} 个停用词")
            else:
                logger.warning(f"停用词文件 {file_path} 不存在，将使用空停用词表")
        except Exception as e:
            logger.error(f"加载停用词文件失败: {e}")
        return stopwords
    
    def _setup_jieba(self, custom_dict_file: str):
        """配置jieba分词器"""
        try:
            # 加载自定义词典（用于专业词汇）
            jieba.load_userdict(custom_dict_file)
            logger.info(f"成功加载自定义词典: {custom_dict_file}")
            
            # 调整分词精度，更好地识别英文词汇
            jieba.suggest_freq(('python', 'x'), True)
            jieba.suggest_freq(('java', 'x'), True)
            jieba.suggest_freq(('mysql', 'x'), True)
            jieba.suggest_freq(('docker', 'x'), True)
            jieba.suggest_freq(('kubernetes', 'x'), True)
            
        except Exception as e:
            logger.warning(f"加载自定义词典失败: {e}")
    
    async def clean_and_normalize(self, text: str) -> str:
        """
        清理和归一化文本
        
        Args:
            text: 原始输入文本
            
        Returns:
            清理后的文本
        """
        if not text or not isinstance(text, str):
            return ""
        
        # 1. 转换为小写（保留英文专业词汇可能的大写，后面特殊处理）
        cleaned = text.lower()
        
        # 2. 移除特殊字符，但保留中英文、数字、空格和常用标点
        # 保留的字符：中文、英文、数字、空格、常见标点（.,!?;:()[]{}）
        cleaned = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\s\.\,\!\?\;\\:\(\)\[\]\{\}]', '', cleaned)
        
        # 3. 处理多余的空格和换行
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # 4. 全角转半角
        cleaned = await self._full_to_half_width(cleaned)
        
        logger.debug(f"清理后: '{text}' -> '{cleaned}'")
        return cleaned
    
    async def _full_to_half_width(self, text: str) -> str:
        """全角字符转半角字符"""
        result = []
        for char in text:
            code = ord(char)
            # 全角字母、数字、空格、标点转半角
            if 0xFF01 <= code <= 0xFF5E:
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:  # 全角空格
                result.append(' ')
            else:
                result.append(char)
        return ''.join(result)
    
    async def tokenize_with_jieba(self, text: str) -> list[str]:
        """
        使用jieba进行分词，保留英文专业词汇
        
        Args:
            text: 清理后的文本
            
        Returns:
            分词后的词元列表
        """
        if not text:
            return []
        
        # 使用搜索引擎模式，更适合检索场景
        words = jieba.cut_for_search(text)
        
        # 处理英文专业词汇：将连续英文字母和数字组合在一起
        tokens = []
        current_english = []
        
        for word in words:
            # 如果是纯英文单词或包含数字（可能是专业词汇）
            if re.match(r'^[a-zA-Z0-9_\-\.]+$', word):
                current_english.append(word)
            else:
                # 如果当前有积累的英文词汇，先处理它们
                if current_english:
                    english_token = ''.join(current_english)
                    tokens.append(english_token.lower())
                    current_english = []
                tokens.append(word)
        
        # 处理最后可能剩余的英文词汇
        if current_english:
            english_token = ''.join(current_english)
            tokens.append(english_token.lower())
        
        logger.debug(f"分词结果: {tokens}")
        return tokens
    
    async def filter_stopwords(self, tokens: list[str]) -> list[str]:
        """
        过滤停用词
        
        Args:
            tokens: 分词后的词元列表
            
        Returns:
            过滤后的词元列表
        """
        if not tokens:
            return []
        
        # 不过滤英文专业词汇和长度较长的词汇
        filtered_tokens = []
        for token in tokens:
            # 保留条件：
            # 1. 不是停用词
            # 2. 长度大于1（单字除非是专业词汇）
            # 3. 或者是英文专业词汇（包含字母）
            if (token not in self.stopwords and 
                (len(token) > 1 or re.search(r'[a-zA-Z]', token))):
                filtered_tokens.append(token)
        
        logger.debug(f"停用词过滤后: {filtered_tokens}")
        return filtered_tokens
    
    async def pos_filter(self, tokens: list[str]) -> list[str]:
        """
        基于词性过滤（可选）
        保留名词、动词、形容词等实词，过滤虚词
        """
        if not tokens:
            return tokens
        
        # 使用jieba进行词性标注
        text = ''.join(tokens)
        words_with_pos = pseg.cut(text)
        
        # 要保留的词性（名词、动词、形容词、英文、数字等）
        keep_pos = {'n', 'v', 'a', 'eng', 'l', 'nr', 'ns', 'nt', 'nz', 'vn', 'an'}
        filtered_tokens = []
        
        for word, pos in words_with_pos:
            if pos.lower() in keep_pos or re.match(r'^[a-zA-Z0-9_\-\.]+$', word):
                filtered_tokens.append(word)
        
        logger.debug(f"词性过滤后: {filtered_tokens}")
        return filtered_tokens
    
    async def synonym_expansion(self, tokens: list[str],
                                synonym_similarity_threshold: float | None = 0.65,  # 较低的阈值获取更多同义词
                                max_synonyms_per_word: int | None = 2            # 每个词最多扩展2个同义词
                            ) -> list[str]:
        """
        同义词扩展
        
        Args:
            tokens: 分词后的词元列表
            synonym_similarity_threshold: 同义词相似度阈值
            max_synonyms_per_word: 每个词最多扩展的同义词数量
            
        Returns:
            扩展后的词元列表
        """

        # 同义词扩展
        synonyms = await CallModel().call_synonym_model(words=tokens, top_k=max_synonyms_per_word, threshold=synonym_similarity_threshold)
        
        if not synonyms:
            logger.warning(f"同义词扩展失败: {tokens}")
            return tokens
        
        # 合并原始词元和同义词
        expanded_tokens = tokens + [synonym for token in tokens for synonym in synonyms.get(token, [])]
        
        logger.debug(f"同义词扩展后: {expanded_tokens}")
        return expanded_tokens


    async def preprocess(self, query: str,
                         enable_pos_filtering: bool | None = True,
                         enable_synonym_expansion: bool | None = None,
                         synonym_similarity_threshold: float | None = 0.65,  # 较低的阈值获取更多同义词
                         max_synonyms_per_word: int | None = 2            # 每个词最多扩展2个同义词
                 ) -> dict[str, str|list[str]] | None:
        """
        完整的预处理流程
        
        Args:
            query: 用户原始查询
            enable_pos_filtering: 是否启用词性过滤
            enable_synonym_expansion: 是否启用同义词扩展
            synonym_similarity_threshold: 同义词相似度阈值
            max_synonyms_per_word: 每个词最多扩展的同义词数量
            
        Returns:
            处理后的查询字符串或词元列表
        """
        try:
            # 1. 清理与归一化
            cleaned = await self.clean_and_normalize(query)
            
            if not cleaned:
                return None
            
            # 2. 分词
            tokens = await self.tokenize_with_jieba(cleaned)
            
            # 3. 停用词过滤
            tokens = await self.filter_stopwords(tokens)
            
            # 4. （可选）词性过滤
            if enable_pos_filtering:
                tokens = await self.pos_filter(tokens)

            # 去重
            tokens = list(set(tokens))
            
            # 构建返回结果
            results = {}

            # 用于语义检索：用空格连接
            result = " ".join(tokens)
            logger.debug(f"语义检索预处理完成: '{query}' -> '{result}'")
            results["semantic"] = result

            # 用于全文检索：返回词元列表
            logger.debug(f"全文检索预处理完成: '{query}' -> {tokens}")
            # 5. （可选）同义词扩展
            if enable_synonym_expansion:
                logger.debug(f"全文检索开始同义词扩展...")
                expanded_tokens = await self.synonym_expansion(tokens, synonym_similarity_threshold, max_synonyms_per_word)
                results["fulltext"] = expanded_tokens
            else:
                results["fulltext"] = tokens
            
            return results
                
        except Exception as e:
            logger.error(f"预处理失败: {e}", exc_info=True)
            return None

# 单例模式，方便全局使用
_preprocessor_instance = None

def get_preprocessor() -> ChinesePreprocessor:
    """获取预处理器单例"""
    global _preprocessor_instance
    if _preprocessor_instance is None:
        _preprocessor_instance = ChinesePreprocessor()
    return _preprocessor_instance

async def preprocess_cn_query(
        query: str,
        enable_pos_filtering: bool = True,
        enable_synonym_expansion: bool | None = None,
        synonym_similarity_threshold: float | None = 0.65,  # 较低的阈值获取更多同义词
        max_synonyms_per_word: int | None = 2            # 每个词最多扩展2个同义词
        ) -> dict[str, str|list[str]] | None:
    """
    便捷函数：预处理查询
    
    Args:
        query: 用户原始查询
        enable_pos_filtering: 是否启用词性过滤
        enable_synonym_expansion: 是否启用同义词扩展
        synonym_similarity_threshold: 同义词相似度阈值
        max_synonyms_per_word: 每个词最多扩展的同义词数量
        
    Returns:
        处理后的结果字典，包含语义检索和全文检索的预处理结果
    """
    preprocessor = get_preprocessor()
    return await preprocessor.preprocess(
        query,
        enable_pos_filtering,
        enable_synonym_expansion,
        synonym_similarity_threshold,
        max_synonyms_per_word
        )