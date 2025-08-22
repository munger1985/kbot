"""LLM model pool implementation."""

import os
import sys
import asyncio
import configparser
from loguru import logger
from typing import Optional
from datetime import datetime, timedelta
from model import(
    BaseLLM, 
    OpenaiLLMConfig, 
    LLMProvider,
    OCILLMConfig,
    create_llm_model
)
from ms_core import load_config, ModelConfig

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository


class ModelPool:
    """Model pool class for managing LLM models."""

    def __init__(self, health_check_interval: int = 600) -> None:
        """Initialize model pool.

        Args:
            health_check_interval: Interval in seconds between health checks
        """
        
        # 用于按提供商管理模型的池
        self._models: dict[str, BaseLLM] = {}
        self._last_used: dict[str, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None

    async def initialize(self):
        """Initialize the model pool and start health check task"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def shutdown(self):
        """Shutdown the model pool and all models"""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            
        for model_unique_name, model in self._models.items():
            try:
                await model.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down model {model_unique_name}: {e}")
                
        self._models.clear()
        self._last_used.clear()

    async def load_model(self, model_unique_name: str) -> BaseLLM:
        """Load a model instance by model_unique_name
        
        Args:
            model_unique_name: unique name of the model to get
            
        Returns:
            The model instance
            
        Raises:
            ValueError: If model_unique_name is not found in database
            RuntimeError: If model creation fails
        """
        # Check if model is already loaded
        if model_unique_name in self._models:
            self._last_used[model_unique_name] = datetime.now()
            return self._models[model_unique_name]
            
        # Load model config from database
        md_repo = KbotMdModelsRepository()
        model_entity = await md_repo.get_by_unique_name(model_unique_name)
        if not model_entity:
            raise ValueError(f"Model {model_unique_name} not found in database")
        
        if model_entity.model_params is None:
            raise ValueError(f"Model {model_unique_name} has no model_params")      
        
        if model_entity.model_params.get("temperature") is not None and not (0 <= model_entity.model_params["temperature"] <= 2):
            raise ValueError(f"Model {model_unique_name} has invalid temperature {model_entity.model_params['temperature']}. Temperature must be between 0 and 2.")
        
        # 从 Nacos 获取 llm 默认参数
        try:
            config = load_config("model_config")
            if not isinstance(config, ModelConfig):
                raise ValueError
            max_tokens = config.llm.max_tokens or 8192
            timeout = config.llm.timeout or 30
            temperature = config.llm.temperature or 0.7
            top_p = config.llm.top_p or 1.0
            top_k = config.llm.top_k or 0
            frequency_penalty = config.llm.frequency_penalty or 0.0
            presence_penalty = config.llm.presence_penalty or 0.0
            
        except Exception as e:
            logger.error(f"Failed to get llm config from Nacos: {e}")
            # 设置默认值
            max_tokens = 819
            timeout = 30
            temperature = 0.7
            top_p = 1.0
            top_k = 0
            frequency_penalty = 0.0
            presence_penalty = 0.0

        # 根据模型类型创建相应的配置
        if model_entity.provider == LLMProvider.OPENAI.value:
            if model_entity.api_key is None or model_entity.api_endpoint is None:
                raise ValueError(f"Model {model_unique_name} has no api_key or api_endpoint")
            
            model_config = OpenaiLLMConfig(
                provider=model_entity.provider,
                api_key=model_entity.api_key,
                api_endpoint=model_entity.api_endpoint,
                model_name=model_entity.model_name,
                temperature=model_entity.model_params.get("temperature", temperature),
                max_tokens=model_entity.model_params.get("max_tokens", max_tokens),
                top_p=model_entity.model_params.get("top_p", top_p),
                frequency_penalty=model_entity.model_params.get("frequency_penalty", frequency_penalty),
                presence_penalty=model_entity.model_params.get("presence_penalty", presence_penalty),
                timeout=model_entity.model_params.get("timeout", timeout)
            )
        elif model_entity.provider in [LLMProvider.OCI_COHERE.value, LLMProvider.OCI_GROK.value, LLMProvider.OCI_LLAMA.value]:
            if model_entity.model_name is None or model_entity.api_endpoint is None or model_entity.model_params.get("compartment_id") is None:
                raise ValueError(f"Model {model_unique_name} has no model_name, api_endpoint or compartment_id")
            model_config = OCILLMConfig(
                provider=model_entity.provider,
                api_endpoint=model_entity.api_endpoint,
                model_name=model_entity.model_name,
                temperature=model_entity.model_params.get("temperature", temperature),
                compartment_id=str(model_entity.model_params.get("compartment_id")),
                max_tokens=model_entity.model_params.get("max_tokens", max_tokens),
                top_p=model_entity.model_params.get("top_p", top_p),
                top_k=model_entity.model_params.get("top_k", top_k),
                frequency_penalty=model_entity.model_params.get("frequency_penalty", frequency_penalty),
                presence_penalty=model_entity.model_params.get("presence_penalty", presence_penalty)
            )
        else:
            # TODO: support other providers
            logger.error(f"Provider {model_entity.provider} is not supported yet")
            raise ValueError(f"Model {model_unique_name} has unsupported provider {model_entity.provider}")
        
        # Create and initialize model //创建和初始化模型
        try:
            model = create_llm_model(model_config)
            await model.startup()
            self._models[model_unique_name] = model
            self._last_used[model_unique_name] = datetime.now()
            return model
        except Exception as e:
            logger.exception(f"Failed to create model {model_unique_name}: {e}")
            raise RuntimeError(f"Failed to create model {model_unique_name}: {str(e)}")



    async def unload_model(self, model_unique_name: str):
        """Unload a model from the pool
        
        Args:
            model_unique_name: unique name of the model to unload
        """
        if model_unique_name in self._models:
            model = self._models.pop(model_unique_name)
            self._last_used.pop(model_unique_name, None)
            try:
                await model.shutdown()
            except Exception as e:
                logger.exception(f"Error unloading model {model_unique_name}: {e}")
                
    async def reload_model(self, model_unique_name: str):
        """Reload a model in the pool
        
        Args:
            model_unique_name: unique name of the model to reload
        """
        if model_unique_name in self._models:
            await self.unload_model(model_unique_name)
        return await self.load_model(model_unique_name)
        
    async def _health_check_loop(self):
        """Background task to periodically check model health"""
        while True:
            await asyncio.sleep(self._health_check_interval)
            await self._perform_health_checks()
            
    async def _perform_health_checks(self):
        """Check health of all models and unload inactive ones"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)  # Unload after 1 hour of inactivity
        
        for model_unique_name in list(self._models.keys()):
            try:
                # Check if model is inactive
                if self._last_used.get(model_unique_name, now) < inactive_threshold:
                    logger.info(f"Unloading inactive model {model_unique_name}")
                    await self.unload_model(model_unique_name)
                    continue
                    
                # Simple health check by calling model with a test text
                model = self._models[model_unique_name]
                await model.chat([{'role': 'user', 'content': 'Hi'}], False, **{"max_tokens": 2})
                logger.debug(f"Health check for model {model_unique_name} succeeded.")
                
            except Exception as e:
                logger.error(f"Health check failed for model {model_unique_name}: {e}")
                # Try to restart the model
                try:
                    logger.info(f"Attempting to restart model {model_unique_name}")
                    await self.reload_model(model_unique_name)
                except Exception as restart_error:
                    logger.exception(f"Failed to restart model {model_unique_name}: {restart_error}")
                    await self.unload_model(model_unique_name)