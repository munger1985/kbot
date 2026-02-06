from core.config.settings import get_reranker_config
from core.dictionary import ModelCategory, RerankerProvider
from loguru import logger
from typing import Any

from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_reranker_model

class RerankerModelPool(BaseModelPool[BaseReranker[Any]]):
    """
    Reranker 模型池
    优化点：
    1. 统一配置映射逻辑
    2. 移除子类中的重复 self._models 赋值（由父类 load_model 统一管理）
    3. 增强类型提示，适配泛型基类
    """
    
    def _get_model_category(self) -> int:
        """返回 Reranker 类别枚举"""
        return ModelCategory.RERANKER.value

    async def _shutdown_model_instance(self, model: BaseReranker[Any]):
        """关闭重排序模型实例"""
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseReranker[Any]):
        """
        执行模型健康检查
        通过一次极简的重排序任务验证模型存活性
        """
        await model.rerank(query="ping", documents=["pong"], top_k=1)
        logger.debug(f"🔍 Reranker 模型 {model_name} 健康检查通过")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseReranker[Any]:
        """
        构造配置并启动 Reranker
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {model_name} 缺少必要参数: provider")

        # 1. 获取基础全局配置
        global_config = get_reranker_config()
        
        # 2. 映射具体的 Config 对象
        model_config = self._build_config(model_name, provider, model_data, global_config)

        # 3. 通过工厂创建并启动
        model = create_reranker_model(model_config)
        try:
            await model.startup()
            # 注意：此处不再手动赋值 self._models[model_name]，交由基类处理以保证状态一致性
            logger.success(f"🚀 Reranker 模型 {model_name} ({provider}) 已就绪")
            return model
        except Exception as e:
            logger.error(f"❌ Reranker 模型 {model_name} 启动异常: {e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> RerankerConfig:
        """
        将数据库原始数据转换为强类型配置类
        """
        params = data.get("model_params", {})
        path = data.get("model_path")
        api_key = data.get("api_key")
        api_endpoint = data.get("api_endpoint")

        # 公共参数快捷映射
        common_kwargs = {
            "provider": provider,
            "model_name": name,
            "max_tokens": params.get("max_tokens", 8192),
            "batch_size": params.get("batch_size", 16),
        }

        # 1. 本地模型处理 (Qwen/BGE)
        if provider in [RerankerProvider.LOCAL_QWEN.value, RerankerProvider.LOCAL_BGE.value]:
            if not path:
                raise ValueError(f"本地模型 {name} 缺失 model_path")
            
            config_cls: Any = Qwen3RerankerConfig if provider == RerankerProvider.LOCAL_QWEN.value else BGERerankerConfig
            return config_cls(
                **common_kwargs,
                model_path=path,
                device=params.get("device"),
                use_fp16=params.get("use_fp16", provider == RerankerProvider.LOCAL_QWEN.value),
                # score_threshold=params.get("score_threshold", 0.0),
                # Qwen 可能需要的特定指令
                instruction=params.get("instruction") if provider == RerankerProvider.LOCAL_QWEN.value else None
            )

        # 2. Cohere API 处理
        if provider == RerankerProvider.COHERE.value:
            if not api_key:
                raise ValueError(f"Cohere 模型 {name} 缺失 api_key")
            return CohereRerankerConfig(
                **common_kwargs,
                api_key=api_key,
                batch_size=params.get("batch_size", 1000), # Cohere 支持极大 batch
                timeout=params.get("timeout", global_cfg.timeout)
            )

        # 3. 通用 OpenAI 兼容接口处理
        if provider in [RerankerProvider.API_QWEN.value, RerankerProvider.CHATGPT.value]:
            if not api_key or not api_endpoint:
                raise ValueError(f"API 模型 {name} 缺失 api_key 或 api_endpoint")
            return OpenAIRerankerConfig(
                **common_kwargs,
                api_key=api_key,
                api_endpoint=api_endpoint,
                timeout=params.get("timeout", global_cfg.timeout)
            )

        raise ValueError(f"未知的 Reranker Provider: {provider}")