from core.config.settings import get_embed_config
from core.dictionary import ModelCategory, EmbeddingProvider
from loguru import logger
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_embedding_model


class EmbeddingModelPool(BaseModelPool[BaseEmbedding[Any]]):
    """
    Embedding 模型池实现
    负责具体 Embedding 模型的生命周期管理与配置映射
    """

    def _get_model_category(self) -> int:
        """定义此池管理的数据库模型类别"""
        return ModelCategory.TXT_EMBEDDING.value

    async def _shutdown_model_instance(self, model: BaseEmbedding[Any]) -> None:
        """调用具体模型的卸载逻辑"""
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseEmbedding[Any]) -> None:
        """
        执行轻量级推理检查
        注意：使用 try-except 包裹已在父类处理，这里只需关注检查动作
        """
        # 使用一个极短的文本进行探测
        await model.embed(["ping"], batch_size=1)
        logger.debug(f"🔍 模型 {model_name} 心跳检查正常")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseEmbedding[Any]:
        """
        构造配置并启动模型实例
        优化点：提取配置构造逻辑，确保职责单一
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {model_name} 缺少必要参数: provider")

        # 1. 获取基础全局配置（作为兜底）
        global_config = get_embed_config()
        
        # 2. 构造特定 Provider 的 Config 对象
        model_config = self._build_config(model_name, provider, model_data, global_config)

        # 3. 通过工厂创建模型
        model = create_embedding_model(model_config)
        
        # 4. 初始化模型资源
        try:
            await model.startup()
            # 注意：不需要手动 self._models[model_name] = model，父类 load_model 会统一处理
            logger.success(f"🚀 Embedding 模型 {model_name} ({provider}) 启动成功")
            return model
        except Exception as e:
            logger.error(f"❌ 模型 {model_name} 启动失败: {str(e)}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> EmbeddingConfig:
        """
        将数据库数据映射为具体的 Pydantic Config 对象
        """
        params = data.get("model_params", {})
        path = data.get("model_path")
        
        # 基础参数提取（复用频率高）
        common_kwargs = {
            "model_name": name,
            "provider": provider,
            "max_tokens": params.get("max_tokens", global_cfg.max_tokens),
            "batch_size": params.get("batch_size", 2),
        }

        # 根据 Provider 映射
        if provider == EmbeddingProvider.LOCAL_QWEN.value:
            if not path: raise ValueError(f"{name} 缺少 model_path")
            return Qwen3EmbeddingConfig(
                **common_kwargs,
                model_path=path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", True),
                instruction=params.get("instruction")
            )

        if provider == EmbeddingProvider.LOCAL_BGE.value:
            if not path: raise ValueError(f"{name} 缺少 model_path")
            return BGEEmbeddingConfig(
                **common_kwargs,
                model_path=path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", False),
                query_instruction=params.get("query_instruction"),
                pooling_strategy=params.get("pooling_strategy", "cls")
            )

        if provider in [EmbeddingProvider.API_QWEN.value, EmbeddingProvider.CHATGPT.value]:
            api_key = data.get("api_key")
            api_base = data.get("api_endpoint")
            if not api_key: raise ValueError(f"{name} 缺少 api_key")
            # 移除 api_base 末尾的 /embeddings，避免路径重复
            if api_base and api_base.endswith("/embeddings"):
                api_base = api_base[:-11]
            return OpenAIEmbeddingConfig(
                **common_kwargs,
                api_key=api_key,
                api_base=api_base,
                dimensions=params.get("dimensions", 1536),
                timeout=params.get("timeout", global_cfg.timeout),
                max_retries=params.get("max_retries", 3)
            )

        if provider == EmbeddingProvider.COHERE.value:
            api_key = data.get("api_key")
            if not api_key: raise ValueError(f"{name} 缺少 api_key")
            return CohereEmbeddingConfig(
                **common_kwargs,
                api_key=api_key,
                timeout=params.get("timeout", global_cfg.timeout),
                input_type_doc=params.get("input_type_doc", "search_document"),
                input_type_query=params.get("input_type_query", "search_query")
            )

        raise ValueError(f"尚未实现或不支持的 Provider: {provider}")