import asyncio
from typing import Any
from loguru import logger
from core.config.settings import get_vlm_config
from core.dictionary import ModelCategory, VLMProvider
from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_vlm_model


class VLMModelPool(BaseModelPool[BaseVLM[Any]]):
    """
    VLM 模型池
    管理视觉语言模型（如 Qwen-VL API 等）的生命周期
    """

    def _get_model_category(self) -> int:
        """返回 VLM 类别枚举"""
        return ModelCategory.VLM.value

    async def _shutdown_model_instance(self, model: BaseVLM[Any]):
        """执行 VLM 资源释放"""
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseVLM[Any]):
        """
        执行模型健康检查
        优化点：简化状态判断逻辑，假设 BaseVLM 已统一 health_check 接口
        """
        try:
            # 兼容异步/同步健康检查（如果 BaseVLM 定义得当，通常直接 await 即可）
            if asyncio.iscoroutinefunction(model.health_check):
                status = await model.health_check()
            else:
                status = await asyncio.to_thread(model.health_check)

            # 统一判断逻辑：支持字典返回或对象属性返回
            is_ready = False
            if isinstance(status, dict):
                is_ready = status.get('initialized', False)
            else:
                is_ready = getattr(status, 'initialized', False)

            if not is_ready:
                raise RuntimeError(f"模型 {model_name} 未就绪 (initialized=False)")
                
            logger.debug(f"🔍 VLM 模型 {model_name} 健康检查通过")
            
        except Exception as e:
            logger.warning(f"❌ 模型 {model_name} 健康检查异常: {e}")
            raise  # 抛出异常由基类触发 reload_model

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseVLM[Any]:
        """
        创建并启动 VLM 实例
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {model_name} 缺少必要参数: provider")

        # 1. 获取全局 VLM 默认配置
        global_vlm_config = get_vlm_config()

        # 2. 构建特定 Provider 的配置对象
        model_config = self._build_config(model_name, provider, model_data, global_vlm_config)

        # 3. 创建并启动模型
        model = create_vlm_model(model_config)
        try:
            await model.startup()
            # 状态管理交由基类 BaseModelPool 处理
            logger.success(f"🚀 VLM 模型 {model_name} ({provider}) 启动成功")
            return model
        except Exception as e:
            logger.error(f"❌ VLM 模型 {model_name} 启动失败: {e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> VLMConfig:
        """
        配置转换映射器
        """
        params = data.get("model_params", {})
        api_endpoint = data.get("api_endpoint")
        api_key = data.get("api_key")

        # 提取公共参数
        common_kwargs = {
            "model_name": name,
            "provider": provider,
            "max_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.1),
        }

        # 1. OpenAI 协议兼容的 VLM API (如 Qwen-VL-Plus/Max)
        if provider == VLMProvider.API_QWEN.value:
            if not api_endpoint or not api_key:
                raise ValueError(f"模型 {name} 缺少 API 参数 (endpoint/key)")

            return OpenAIVLMConfig(
                **common_kwargs,
                api_key=api_key,
                api_endpoint=api_endpoint,
                api_version=params.get("api_version", ""),
                timeout=params.get("timeout", global_cfg.timeout),
                max_retries=params.get("max_retries", 3)
            )

        # 此处可扩展其他 Provider，如 LocalVLMConfig (Llava/Qwen-VL 本地部署)
        
        raise NotImplementedError(f"不支持的 VLM 提供者: {provider}")