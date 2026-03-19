import re
import json
import jieba
import jieba.analyse
from pathlib import Path
from loguru import logger
from core.database.oracle import get_session
from core.config.settings import get_jieba_config, get_prompt_config
from utils.clients import AIModelClient
from dao.repositories import PromptRepository


class LLMFullTextPreprocessor:
    """
    基于LLM的全文检索查询预处理类，支持 JSON 结构化提取与 Jieba 降级方案
    """

    def __init__(self):
        self.llm_client = AIModelClient()
        
        # 初始化Jieba配置（用于降级方案）
        jieba_config = get_jieba_config()
        self.stopwords_file = jieba_config.stop_words_path
        self.custom_dict_file = jieba_config.custom_dict_path
        self.stopwords: set[str] = self._load_stopwords(self.stopwords_file)
        self._setup_jieba(self.custom_dict_file)

    @property
    def oracle_session(self):
        return get_session()
        
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
        """基础文本清理"""
        if not text or not isinstance(text, str):
            return ""
        # 保留中英文字符、数字及部分对搜索有意义的符号（如点号、横杠）
        cleaned = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\s\.\-]', ' ', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    async def _build_fulltext_optimization_prompt(self, query: str) -> str:
        """构建针对混合检索优化的 JSON 格式提示词"""
        prompt_name = get_prompt_config().fulltext_optimization
        system_prompt = await self._get_system_prompt(prompt_name)

        # 只要没有获取到系统提示词，就使用硬编码的默认值，不在此处触发降级
        if not system_prompt:
            return f"""你是一个搜索指令专家。请将用户问题改写为适用于“全文检索”的结构化 JSON。

### 处理规则：
1. **must**: 提取最核心、不可缺失的实体或术语（如产品名、错误码、版本号）。
2. **expansion**: 对缩写、别名或中英文对照进行扩展（如 K8s -> Kubernetes, 响应时间 -> Response Time）。
3. **synonyms**: 语义相近的同义词。
4. **exclude**: 过滤掉所有语气词、疑问词和虚词。

### 输出格式：
仅输出一个合法的 JSON 对象，不要包含任何解释或 Markdown 标签：
{{
  "must": ["核心词1"],
  "expansion": ["扩展词1"],
  "synonyms": ["同义词1"]
}}

### 原始问题：
{query}
"""
        # 如果从 DB 获取到了提示词，进行变量替换
        return system_prompt.replace("{query}", query) if "{query}" in system_prompt else f"{system_prompt}\n\n问题：{query}"

    async def _extract_optimized_query_from_response(self, llm_response: str, original_query: str) -> str:
        """
        从 LLM 响应中解析 JSON 并生成关键词字符串
        """
        try:
            content = llm_response.strip()
            
            # 移除可能存在的 Markdown 代码块包裹
            if content.startswith("```"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end+1]
            
            # 解析 JSON
            data = {}
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    data = json.loads(match.group(0))
                else:
                    raise ValueError("No valid JSON found in LLM response")

            # 按权重顺序汇总关键词，并显式确保所有元素都是字符串
            keywords: list[str] = []
            for key in ["must", "expansion", "synonyms"]:
                vals = data.get(key, [])
                if isinstance(vals, list):
                    for v in vals:
                        # 核心修复：确保 v 是字符串且不为空
                        if isinstance(v, str):
                            cleaned_v = v.strip()
                            if cleaned_v and cleaned_v not in keywords:
                                keywords.append(cleaned_v)

            if not keywords:
                return await self._fallback_to_jieba(original_query)

            # 最终合并为以空格分隔的字符串
            final_query = " ".join(keywords[:12])
            logger.info(f"LLM 预处理成功: {final_query}")
            return final_query

        except Exception as e:
            logger.error(f"解析响应失败: {e}")
            return await self._fallback_to_jieba(original_query)

    async def optimize_for_fulltext_search(self, query: str, model_name: str) -> str:
        """调用 LLM 并获取优化结果"""
        prompt = await self._build_fulltext_optimization_prompt(query)
        try:
            full_text = ""
            async for chunk in self.llm_client.call_llm_model(
                model_name=model_name,
                prompt=prompt,
                stream=True,
                temperature=0.1
            ):
                full_text += chunk
            
            if not full_text:
                raise ValueError("LLM returned empty response")
            
            return await self._extract_optimized_query_from_response(full_text, query)
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return await self._fallback_to_jieba(query)

    async def _fallback_to_jieba(self, query: str, topk: int = 5) -> str:
        """降级方案：Jieba 关键词提取"""
        try:
            cleaned = await self._clean_text(query)
            if not cleaned: return query
            
            tags = jieba.analyse.extract_tags(cleaned, topK=topk, allowPOS=('n', 'nr', 'ns', 'nt', 'nz', 'vn', 'eng'))
            result = [t for t in tags if t not in self.stopwords]
            
            if not result:
                words = jieba.lcut(cleaned)
                result = [w for w in words if len(w) > 1 and w not in self.stopwords][:topk]
            
            final_res = " ".join(result) if result else cleaned # type: ignore
            logger.info(f"Jieba 降级结果: {final_res}")
            return final_res
        except Exception as e:
            logger.error(f"Jieba 异常: {e}")
            return query

    async def preprocess(self, query: str, model_name: str | None = None) -> str:
        """主入口"""
        if not model_name:
            return await self._fallback_to_jieba(query)
        return await self.optimize_for_fulltext_search(query, model_name)

    async def _get_system_prompt(self, prompt_name: str) -> str | None:
        """从 DB 获取 Prompt"""
        async with self.oracle_session as session:
            repo = PromptRepository(session)
            try:
                # 这里改为 warning，因为有内置默认 Prompt 作为兜底
                prompt = await repo.get_prompt_by_unique_name(prompt_name)
                if not prompt:
                    logger.warning(f"未在数据库中找到名为 {prompt_name} 的提示词，将使用默认配置")
                return prompt
            except Exception as e:
                logger.warning(f"DB Prompt 获取异常 (将使用默认配置): {e}")
                return None

_llm_preprocessor_instance = None

def get_llm_preprocessor() -> LLMFullTextPreprocessor:
    global _llm_preprocessor_instance
    if _llm_preprocessor_instance is None:
        _llm_preprocessor_instance = LLMFullTextPreprocessor()
    return _llm_preprocessor_instance

async def preprocess_for_fulltext(query: str, model_name: str | None = None) -> str:
    preprocessor = get_llm_preprocessor()
    return await preprocessor.preprocess(query, model_name)