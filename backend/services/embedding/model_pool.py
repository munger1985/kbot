import asyncio
import torch
from loguru import logger
from typing import Dict, Optional
from datetime import datetime, timedelta

from models.embedding.factory import create_embedding_model
from models.embedding.base import BaseEmbedding, LocalEmbeddingConfig, RemoteEmbeddingConfig
from dao.repositories.kbot_md_models_repo import KbotMdModelsRepository



class ModelPool:
    """Manage a pool of embedding models with health checking and lifecycle management"""
    
    def __init__(self, health_check_interval: int = 300):
        """
        Args:
            health_check_interval: Interval in seconds between health checks
        """
        self._models: Dict[int, BaseEmbedding] = {}
        self._last_used: Dict[int, datetime] = {}
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
            
        for model_id, model in self._models.items():
            try:
                await model.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down model {model_id}: {e}")
                
        self._models.clear()
        self._last_used.clear()
        
    async def load_model(self, model_id: int) -> BaseEmbedding:
        """Load a model instance by model_id
        
        Args:
            model_id: ID of the model to get
            
        Returns:
            The model instance
            
        Raises:
            ValueError: If model_id is not found in database
            RuntimeError: If model creation fails
        """
        # Check if model is already loaded
        if model_id in self._models:
            self._last_used[model_id] = datetime.now()
            return self._models[model_id]
            
        # Load model config from database
        md_repo = KbotMdModelsRepository()
        model_entity = await md_repo.get_by_id(model_id)
        if not model_entity:
            raise ValueError(f"Model {model_id} not found in database")
        
        if model_entity.model_params is None:
            raise ValueError(f"Model {model_id} has no model_params")

        # 根据模型类型创建相应的配置
        
        if model_entity.provider == "local":
            model_config = LocalEmbeddingConfig(
                model_name=model_entity.model_name,
                provider=model_entity.provider,
                max_tokens=model_entity.model_params.get("max_tokens", 512),
                model_path=model_entity.model_params.get("model_path", None),
                device=model_entity.model_params.get("device", None),
                device_map=model_entity.model_params.get("device_map", None),
                max_memory=model_entity.model_params.get("max_memory", None),
                trust_remote_code=model_entity.model_params.get("trust_remote_code", False),
                use_fp16=model_entity.model_params.get("use_fp16", False),
                local_files_only=model_entity.model_params.get("local_files_only", False),
                compile_model=model_entity.model_params.get("compile_model", True)
            )
        else:  # 远程模型
            model_config = RemoteEmbeddingConfig(
                model_name=model_entity.model_name,
                provider=model_entity.provider,
                max_tokens=model_entity.model_params.get("max_tokens", 512),
                api_key=model_entity.api_key, # type: ignore
                endpoint=model_entity.api_endpoint, # type: ignore
                timeout=model_entity.model_params.get("timeout", 30),        
                max_retries=model_entity.model_params.get("max_retries", 3),
                organization=model_entity.model_params.get("organization", ""),
                deployment_name=model_entity.model_params.get("deployment_name", ""),
                api_version=model_entity.model_params.get("api_version", "2023-05-15")
            )

        # Create and initialize model //创建和初始化模型
        try:
            model = create_embedding_model(model_config)
            await model.startup()
            self._models[model_id] = model
            self._last_used[model_id] = datetime.now()
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_id}: {e}")
            raise RuntimeError(f"Failed to create model {model_id}: {e}")
                    
    async def unload_model(self, model_id: int):
        """Unload a model from the pool
        
        Args:
            model_id: ID of the model to unload
        """
        if model_id in self._models:
            model = self._models.pop(model_id)
            self._last_used.pop(model_id, None)
            try:
                await model.shutdown()
            except Exception as e:
                logger.error(f"Error unloading model {model_id}: {e}")
                
    async def reload_model(self, model_id: int):
        """Reload a model in the pool
        
        Args:
            model_id: ID of the model to reload
        """
        if model_id in self._models:
            await self.unload_model(model_id)
        return await self.load_model(model_id)
        
    async def _health_check_loop(self):
        """Background task to periodically check model health"""
        while True:
            await asyncio.sleep(self._health_check_interval)
            await self._perform_health_checks()
            
    async def _perform_health_checks(self):
        """Check health of all models and unload inactive ones"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)  # Unload after 1 hour of inactivity
        
        for model_id in list(self._models.keys()):
            try:
                # Check if model is inactive
                if self._last_used.get(model_id, now) < inactive_threshold:
                    logger.info(f"Unloading inactive model {model_id}")
                    await self.unload_model(model_id)
                    continue
                    
                # Simple health check by calling embed with empty list
                model = self._models[model_id]
                await model.embed([])
                
            except Exception as e:
                logger.error(f"Health check failed for model {model_id}: {e}")
                # Try to restart the model
                try:
                    logger.info(f"Attempting to restart model {model_id}")
                    await self.reload_model(model_id)
                except Exception as restart_error:
                    logger.error(f"Failed to restart model {model_id}: {restart_error}")
                    await self.unload_model(model_id)