import asyncio
from typing import Dict, Union
from .base import BaseLLM, LocalLLMConfig, CloudLLMConfig
from .local import LocalLLM
from .cloud import CloudLLM

class LLMProvider:
    def __init__(self):
        self.models: Dict[str, BaseLLM] = {}
        self._initialized = False

    async def initialize(self):
        """异步初始化所有模型"""
        if not self._initialized:
            await asyncio.gather(*[model.startup() for model in self.models.values()])
            self._initialized = True

    async def add_model(self, config: Union[LocalLLMConfig, CloudLLMConfig]):
        """添加模型配置"""
        if config.model_name in self.models:
            raise ValueError(f"Model with name '{config.model_name}' already exists")
        
        if isinstance(config, LocalLLMConfig):
            model = LocalLLM(config)
        else:
            model = CloudLLM(config)
            
        await model.startup()
        self.models[config.model_name] = model
        return model

    async def shutdown(self):
        """关闭所有模型"""
        await asyncio.gather(*[model.shutdown() for model in self.models.values()])
        self._initialized = False