"""LLM model pool implementation."""

import asyncio
import os
import sys
from loguru import logger
from typing import Dict, List, Optional
from datetime import datetime, timedelta

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from models.llm import(
    BaseLLM, 
    OpenaiLLMConfig, 
    LLMProvider,
    HuggingFaceLLMConfig, 
    AnthropicLLMConfig, 
    create_llm_model
)
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository
from core.config import settings


class ModelPool:
    """Model pool class for managing LLM models."""

    def __init__(self, health_check_interval: int = 300) -> None:
        """Initialize model pool.

        Args:
            health_check_interval: Interval in seconds between health checks
        """
        
        # 用于按提供商管理模型的池
        self._models: Dict[str, BaseLLM] = {}
        self._last_used: Dict[str, datetime] = {}
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

        # 根据模型类型创建相应的配置
        
        if model_entity.provider == LLMProvider.OPENAI.value:
            model_config = OpenaiLLMConfig(
                api_key=model_entity.api_key, # type: ignore
                api_endpoint=model_entity.api_endpoint,
                model_name=model_entity.model_name,
                temperature=model_entity.model_params.get("temperature", settings['llm']['temperature']),
                max_tokens=model_entity.model_params.get("max_tokens", settings['llm']['max_tokens']),
                top_p=model_entity.model_params.get("top_p", settings['llm']['top_p']),
                frequency_penalty=model_entity.model_params.get("frequency_penalty", settings['llm'].get('frequency_penalty', 0)), # type: ignore
                presence_penalty=model_entity.model_params.get("presence_penalty", settings['llm'].get('presence_penalty', 0)), # type: ignore
                timeout=model_entity.model_params.get("timeout", settings['llm']['timeout'])
            )
        if model_entity.provider == LLMProvider.ANTHROPIC.value:
            model_config = AnthropicLLMConfig(
                model_name=model_entity.model_name,
                api_key=model_entity.api_key # type: ignore
            )
        if model_entity.provider == LLMProvider.HUGGINGFACE.value:
            model_config = HuggingFaceLLMConfig(
                model_name=model_entity.model_name,
                api_key=model_entity.api_key # type: ignore
            )
        # Create and initialize model //创建和初始化模型
        try:
            model = create_llm_model(model_config)
            await model.startup()
            self._models[model_unique_name] = model
            self._last_used[model_unique_name] = datetime.now()
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_unique_name}: {e}")
            raise RuntimeError(f"Failed to create model {model_unique_name}: {e}")


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
                logger.error(f"Error unloading model {model_unique_name}: {e}")
                
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
                    
                # Simple health check by calling embed with a test text
                model = self._models[model_unique_name]
                await model.chat("Hi")
                
            except Exception as e:
                logger.error(f"Health check failed for model {model_unique_name}: {e}")
                # Try to restart the model
                try:
                    logger.info(f"Attempting to restart model {model_unique_name}")
                    await self.reload_model(model_unique_name)
                except Exception as restart_error:
                    logger.error(f"Failed to restart model {model_unique_name}: {restart_error}")
                    await self.unload_model(model_unique_name)