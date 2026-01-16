from core.config.settings import get_llm_config
from core.dictionary import ModelCategory, LLMProvider
from loguru import logger
from typing import Any
from microservices.common.model_pool import BaseModelPool
from .model import *
from .model_factory import create_llm_model


class LLMModelPool(BaseModelPool[BaseLLM[Any]]):
    """
    LLM 模型池
    负责大语言模型（OpenAI 兼容接口、OCI 等）的实例管理
    """

    def _get_model_category(self) -> int:
        """返回 LLM 类别枚举"""
        return ModelCategory.LLM.value

    async def _shutdown_model_instance(self, model: BaseLLM[Any]):
        """执行 LLM 资源释放（如关闭 HTTP 客户端 Session）"""
        await model.shutdown()

    async def _perform_model_health_check(self, model_name: str, model: BaseLLM[Any]):
        """
        通过一次极简对话检查 API 连通性
        使用 stream=False 和极小的 max_tokens 以节省 Token 并降低延迟
        """
        # 探测调用
        await model.chat(
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
            max_tokens=2
        )
        logger.debug(f"🔍 LLM 模型 {model_name} 响应正常")

    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> BaseLLM[Any]:
        """
        创建并启动 LLM 实例
        """
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {model_name} 缺少必要参数: provider")

        # 1. 获取全局默认配置
        global_config = get_llm_config()
        
        # 2. 构造特定 Provider 的 Config 对象
        model_config = self._build_config(model_name, provider, model_data, global_config)

        # 3. 通过工厂创建实例
        model = create_llm_model(model_config)
        
        try:
            await model.startup()
            # 状态记录由父类 load_model 统一处理，此处不再手动操作 self._models
            logger.success(f"🚀 LLM 模型 {model_name} ({provider}) 已接入模型池")
            return model
        except Exception as e:
            logger.error(f"❌ LLM 模型 {model_name} 启动失败: {e}")
            raise

    def _build_config(self, name: str, provider: str, data: dict[str, Any], global_cfg: Any) -> LLMConfig:
        """
        将数据库配置映射为强类型 Config 对象
        """
        params = data.get("model_params", {})
        api_key = data.get("api_key")
        api_endpoint = data.get("api_endpoint")

        # 公共参数快捷提取
        common_kwargs = {
            "model_name": name,
            "provider": provider,
            "temperature": params.get("temperature", global_cfg.temperature),
            "max_tokens": params.get("max_tokens", global_cfg.max_tokens),
            "top_p": params.get("top_p", global_cfg.top_p),
            "frequency_penalty": params.get("frequency_penalty", global_cfg.frequency_penalty),
            "presence_penalty": params.get("presence_penalty", global_cfg.presence_penalty),
        }

        # 1. OpenAI 兼容接口 (DeepSeek, Qwen API, GPT)
        openai_providers = [
            LLMProvider.API_DEEPSEEK.value, 
            LLMProvider.API_QWEN.value, 
            LLMProvider.CHATGPT.value
        ]
        if provider in openai_providers:
            if not api_key or not api_endpoint:
                raise ValueError(f"API 模型 {name} 缺失 api_key 或 api_endpoint")
            
            return OpenaiLLMConfig(
                **common_kwargs,
                api_key=api_key,
                api_endpoint=api_endpoint,
                timeout=params.get("timeout", global_cfg.timeout)
            )

        # 2. Oracle Cloud Infrastructure (OCI) 接口
        if provider == LLMProvider.OCI.value:
            compartment_id = params.get("compartment_id")
            config_file = params.get("config_file")
            
            if not all([api_endpoint, compartment_id, config_file]):
                raise ValueError(f"OCI 模型 {name} 缺少必要参数 (compartment_id/config_file/endpoint)")

            return OCILLMConfig(
                **common_kwargs,
                api_endpoint=api_endpoint, # type: ignore
                compartment_id=compartment_id,
                config_file=config_file,
                top_k=params.get("top_k", global_cfg.top_k)
            )

        raise ValueError(f"不支持的 LLM Provider: {provider}")

    def get_provider_in_pool(self, model_name: str) -> str | None:
        """
        获取已加载模型的 Provider
        直接从模型实例的 config 中读取，确保数据源唯一
        """
        model = self._models.get(model_name)
        return model.config.provider if model else None