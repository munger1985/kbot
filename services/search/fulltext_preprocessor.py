import re
import json
import jieba
import jieba.analyse
from pathlib import Path
from loguru import logger
from core.config.settings import get_jieba_config
from utils.model_client import CallModel


class LLMFullTextPreprocessor:
    """
    基于LLM的全文检索查询预处理类，带Jieba降级方案
    """
    
    def __init__(self):
        self.llm_client = CallModel()
        
        # 初始化Jieba配置（用于降级方案）
        jieba_config = get_jieba_config()
        self.stopwords_file = jieba_config.stop_words_path
        self.custom_dict_file = jieba_config.custom_dict_path
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
            if Path(custom_dict_file).exists():
                jieba.load_userdict(custom_dict_file)
                logger.info(f"成功加载自定义词典: {custom_dict_file}")
        except Exception as e:
            logger.warning(f"加载自定义词典失败: {e}")
    
    async def _clean_text(self, text: str) -> str:
        """简化文本清理"""
        if not text or not isinstance(text, str):
            return ""
        
        # 基础清理：去除非中英文字符，保留空格
        cleaned = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\s]', ' ', text)
        # 合并多余空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
        
    async def optimize_for_fulltext_search(self, query: str, model_id: int) -> str:
        """
        使用LLM优化全文检索查询
        
        Args:
            query: 原始查询文本
            model_id: LLM模型ID
            
        Returns:
            优化后的查询字符串
        """
        prompt = self._build_fulltext_optimization_prompt(query)
        
        try:
            async for chunk in self.llm_client.call_llm_model(
                model_id=model_id,
                prompt=prompt,
                tools=None,  # 不需要工具调用
                tool_choice=None,
                stream=False,
                temperature=0.1  # 低温度保证稳定性
            ):
                logger.debug(f"LLM全文检索优化响应: {chunk}")
                response = await self._extract_optimized_query_from_response(chunk, query)
                
                if not response:
                    logger.warning(f"LLM全文检索优化失败: 优化后的查询为空")
                    return await self._fallback_to_jieba(query)
                
                json_response = json.loads(response)
                optimized_query = json_response.get("choices")[0].get("message").get("content", "").strip()

                logger.info(f"LLM全文检索优化完成: '{query}' -> '{optimized_query}'")
            
            return optimized_query
                
        except Exception as e:
            logger.error(f"LLM全文检索优化失败: {e}")
            return await self._fallback_to_jieba(query)
    
    def _build_fulltext_optimization_prompt(self, query: str) -> str:
        """构建全文检索优化提示词"""
        return f"""你是一个搜索优化专家，请将用户问题改写成最适合全文检索的查询格式。

原始问题：{query}

全文检索要求：
1. 提取3-5个最核心的关键词或短语
2. 保留重要的实体、术语、时间、地点等具体信息
3. 去除疑问词、修饰词和泛化词汇
4. 用空格分隔关键词，保持简洁
5. 确保改写后的查询能准确匹配相关文档内容

改写原则：
- 优先保留专业术语和具体概念
- 去除"如何"、"什么"、"为什么"等疑问词
- 去除"的"、"了"、"在"等无意义虚词
- 保持核心语义不变

改写示例：
- "文艺复兴是什么时候发生的？" → "文艺复兴 发生时间 历史时期"
- "如何配置Redis集群？" → "Redis 集群 配置 方法"
- "Python中的装饰器有什么作用？" → "Python 装饰器 作用 功能"
- "电脑经常卡顿怎么办？" → "电脑 卡顿 解决方案 性能优化"
- "学习机器学习需要哪些数学基础？" → "机器学习 数学基础 线性代数 概率论"

请直接输出优化后的查询，不要添加任何解释："""
    
    async def _extract_optimized_query_from_response(self, llm_response: str, original_query: str) -> str:
        """从LLM响应中提取优化后的查询"""
        try:
            if isinstance(llm_response, str):
                content = llm_response
            else:
                content = str(llm_response)
            
            # 清理响应内容
            optimized_query = content.strip()
            
            # 移除可能的引号和其他标记
            optimized_query = re.sub(r'^["\']|["\']$', '', optimized_query)
            
            # 如果响应为空或不合理，使用降级方案
            if not optimized_query or len(optimized_query) < 2:
                logger.warning(f"LLM返回的查询为空或不合理: '{optimized_query}'")
                return await self._fallback_to_jieba(original_query)
            
            # 确保查询不会太长（防止LLM输出过多内容）
            if len(optimized_query.split()) > 8:
                words = optimized_query.split()[:6]  # 限制最多6个词
                optimized_query = ' '.join(words)
                logger.info(f"优化查询过长，截断为: {optimized_query}")
            
            return optimized_query
            
        except Exception as e:
            logger.error(f"解析LLM响应失败: {e}")
            return await self._fallback_to_jieba(original_query)
    
    async def _fallback_to_jieba(self, query: str, topk: int = 5) -> str:
        """降级方案：使用Jieba分词提取关键词（不需要同义词扩展）"""
        try:
            # 先清理文本
            cleaned_text = await self._clean_text(query)
            if not cleaned_text:
                return query  # 降级返回原文本
            
            # 使用TF-IDF提取最重要的关键词
            keywords = jieba.analyse.extract_tags(
                cleaned_text, 
                topK=topk,
                withWeight=False,
                allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'vn', 'eng')  # 名词、动词、英文
            )
            
            # 过滤停用词
            filtered_keywords = [word for word in keywords if word not in self.stopwords]
            
            # 如果过滤后为空，使用精确分词作为fallback
            if not filtered_keywords:
                words = jieba.lcut(cleaned_text)
                filtered_keywords = [word for word in words if len(word) > 1 and word not in self.stopwords]
                filtered_keywords = filtered_keywords[:topk]
            
            result = ' '.join(filtered_keywords) if filtered_keywords else cleaned_text
            logger.info(f"使用Jieba降级方案处理: '{query}' -> '{result}'")
            return result
            
        except Exception as e:
            logger.error(f"Jieba降级方案也失败: {e}")
            return query
    
    async def preprocess_with_synonym_expansion(self, query: str, model_id: int | None = None, 
                                              enable_synonym: bool = False) -> str:
        """
        带同义词扩展的全文检索预处理
        
        Args:
            query: 原始查询
            model_id: LLM模型ID，如果为None，则使用jieba分词且不进行同义词扩展
            enable_synonym: 是否启用同义词扩展
            
        Returns:
            预处理后的查询字符串
        """
        # 如果没有指定模型ID，则使用jieba分词且不进行同义词扩展
        if not model_id:
            return await self._fallback_to_jieba(query)
        
        # 第一步：基础优化
        optimized_query = await self.optimize_for_fulltext_search(query, model_id)
        
        # 第二步：同义词扩展（可选）
        if enable_synonym:
            expanded_query = await self._expand_synonyms_with_llm(optimized_query, model_id)
            return expanded_query
        
        return optimized_query
    
    async def _expand_synonyms_with_llm(self, query: str, model_id: int) -> str:
        """使用LLM进行同义词扩展"""
        prompt = f"""请为以下搜索查询扩展同义词和相关术语，用于改善检索效果：

原始查询：{query}

要求：
1. 为每个核心概念添加1-2个同义词或相关术语
2. 保持查询简洁，总词数不超过8个
3. 用空格分隔所有词汇
4. 优先添加最相关和常用的同义词

示例：
- "电脑 卡顿" → "计算机 电脑 运行缓慢 卡顿 性能问题"
- "Python 学习" → "Python 编程 学习 教程 入门"

请直接输出扩展后的查询："""
        
        try:
            async for chunk in self.llm_client.call_llm_model(
                model_id=model_id,
                prompt=prompt,
                tools=None,
                tool_choice=None,
                stream=False,
                temperature=0.2
            ):
                response = await self._extract_optimized_query_from_response(chunk, query)

                if not response:
                    logger.warning(f"LLM同义词扩展失败: 扩展后的查询为空")
                    return query

                json_response = json.loads(response)
                expanded_query = json_response.get("choices")[0].get("message").get("content", "").strip()

                logger.info(f"同义词扩展完成: '{query}' -> '{expanded_query}'")
            
            return expanded_query
                
        except Exception as e:
            logger.warning(f"LLM同义词扩展失败: {e}")
            return query  # 失败时返回原查询

# 单例模式
_llm_preprocessor_instance = None

def get_llm_preprocessor() -> LLMFullTextPreprocessor:
    global _llm_preprocessor_instance
    if _llm_preprocessor_instance is None:
        _llm_preprocessor_instance = LLMFullTextPreprocessor()
    return _llm_preprocessor_instance

async def preprocess_for_fulltext(
        query: str,
        model_id: int | None = None,
        enable_synonym_expansion: bool = False
    ) -> str:
    """
    基于LLM的全文检索预处理函数
    
    Args:
        query: 查询文本
        model_id: LLM模型ID，如果未提供则直接使用jieba分词且不进行同义词扩展
        enable_synonym_expansion: 是否启用同义词扩展
        
    Returns:
        预处理后的查询字符串
    """
    preprocessor = get_llm_preprocessor()
    
    return await preprocessor.preprocess_with_synonym_expansion(
        query, 
        model_id, 
        enable_synonym_expansion
    )