import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict

from loguru import logger
from ms_core import load_config, ModelConfig, ModelCategory, AsyncRedisPool
from model import (
    BaseEmbedding,
    EmbeddingProvider,
    LocalEmbeddingConfig,
    OCIEmbeddingConfig,
    create_embedding_model
)


class ModelPool:
    """Manage a pool of embedding models with health checking and lifecycle management"""

    def __init__(self, health_check_interval: int = 600):
        """Initialize model pool.
        Args:
            health_check_interval: Interval in seconds between health checks
        """
        self._models: Dict[str, BaseEmbedding] = {}
        self._last_used: Dict[str, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: Optional[asyncio.Task] = None
        self.redis = AsyncRedisPool(db=1)

    async def initialize(self):
        """Initialize the model pool and start health check task"""
        try:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            logger.info("Model pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model pool: {e}")
            raise

    async def shutdown(self):
        """Gracefully shutdown the model pool and all models"""
        # Cancel health check task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                logger.info("Health check task cancelled")
            except Exception as e:
                logger.error(f"Error cancelling health check task: {e}")

        # Shutdown all models
        shutdown_tasks = []
        for model_unique_name, model in self._models.items():
            shutdown_tasks.append(asyncio.create_task(
                self._safe_shutdown_model(model_unique_name, model)
            ))

        # Wait for all shutdown tasks to complete
        if shutdown_tasks:
            await asyncio.wait(shutdown_tasks)

        self._models.clear()
        self._last_used.clear()

        # Close Redis connection pool
        try:
            await self.redis.close()
            logger.info("Redis connection pool closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection pool: {e}")

    async def _safe_shutdown_model(self, model_unique_name: str, model: BaseEmbedding):
        """Safely shutdown a single model with error handling"""
        try:
            await model.shutdown()
            logger.info(f"Model {model_unique_name} shutdown successfully")
        except Exception as e:
            logger.error(f"Error shutting down model {model_unique_name}: {e}")

    async def load_model(self, model_unique_name: str) -> BaseEmbedding:
        """Load a model instance by model_unique_name"""
        if not model_unique_name:
            raise ValueError("model_unique_name cannot be empty")

        # Check if model is already loaded
        if model_unique_name in self._models:
            model = self._models[model_unique_name]
            # Verify model is still healthy
            try:
                await model.embed(["test"], batch_size=1)  # Simple health check
                self._last_used[model_unique_name] = datetime.now()
                return model
            except Exception:
                logger.warning(f"Model {model_unique_name} found but failed health check, reloading...")
                await self.unload_model(model_unique_name)

        try:
            async with self.redis as redis:
                # 1. Get model_id from unique_name
                model_id = await redis.get(f"index:unique_name:{model_unique_name}")
                if not model_id:
                    raise ValueError(f"Model {model_unique_name} not found in redis")

                # 2. Get all model data
                model_data = await redis.hgetall(f"model:{model_id}")
                if not model_data:
                    raise ValueError(f"Model {model_unique_name} not found in redis")

                # Parse JSON fields
                model_params = json.loads(model_data["model_params"]) if model_data.get("model_params") else {}

            provider = model_data.get("provider")
            if not provider:
                raise ValueError(f"Model {model_unique_name} has no provider")

            model_name = model_data.get("model_name")
            if not model_name:
                raise ValueError(f"Model {model_unique_name} has no model_name")

            # Load config from Nacos or use defaults
            try:
                config = load_config("model_config")
                if not isinstance(config, ModelConfig):
                    raise ValueError("Invalid model config")
                max_tokens = config.embed.max_tokens or 8192
                timeout = config.embed.timeout or 300
                max_retries = config.embed.max_retries or 0
                cache_dir = config.embed.cache_dir
            except Exception as e:
                logger.warning(f"Failed to get embedding config from Nacos: {e}")
                max_tokens = 8192
                timeout = 30
                max_retries = 0
                cache_dir = "./cached_models"

            # Create appropriate config
            if provider == EmbeddingProvider.LOCAL.value:
                model_config = LocalEmbeddingConfig(
                    model_name=model_name,
                    provider=provider,
                    max_tokens=model_params.get("max_tokens", max_tokens),
                    batch_size=model_params.get("batch_size", 2),
                    model_path=model_params.get("model_path"),
                    device=model_params.get("device"),
                    device_map=model_params.get("device_map"),
                    max_memory=model_params.get("max_memory"),
                    trust_remote_code=model_params.get("trust_remote_code", False),
                    use_fp16=model_params.get("use_fp16", False),
                    local_files_only=model_params.get("local_files_only", False),
                    compile_model=model_params.get("compile_model", True),
                    cache_dir=cache_dir
                )
            elif provider == EmbeddingProvider.OCI.value:
                compartment_id = model_params.get("compartment_id")
                config_file = model_params.get("config_file")
                api_endpoint = model_data.get("api_endpoint")
                if not all([model_name, api_endpoint, compartment_id, config_file]):
                    raise ValueError(f"Model {model_unique_name} missing required parameters")
                model_config = OCIEmbeddingConfig(
                    model_name=model_name,
                    provider=provider,
                    max_tokens=model_params.get("max_tokens", max_tokens),
                    batch_size=model_params.get("batch_size", 2),
                    api_endpoint=api_endpoint,
                    compartment_id=compartment_id, # type: ignore
                    config_file=config_file # type: ignore
                )
            else:
                raise ValueError(f"Unsupported provider {provider}")

            # Create and initialize model
            model = create_embedding_model(model_config)
            await model.startup()
            self._models[model_unique_name] = model
            self._last_used[model_unique_name] = datetime.now()
            logger.success(f"Model {model_unique_name} loaded successfully")
            return model

        except Exception as e:
            logger.exception(f"Failed to load model {model_unique_name}")
            raise RuntimeError(f"Failed to load model {model_unique_name}: {e}")

    async def unload_model(self, model_unique_name: str):
        """Unload a model from the pool"""
        if model_unique_name in self._models:
            model = self._models.pop(model_unique_name)
            self._last_used.pop(model_unique_name, None)
            try:
                await model.shutdown()
                logger.info(f"Model {model_unique_name} unloaded successfully")
            except Exception as e:
                logger.error(f"Error unloading model {model_unique_name}: {e}")

    async def reload_model(self, model_unique_name: str) -> BaseEmbedding:
        """Reload a model in the pool"""
        await self.unload_model(model_unique_name)
        return await self.load_model(model_unique_name)

    async def _health_check_loop(self):
        """Background task to periodically check model health"""
        try:
            while True:
                try:
                    await asyncio.sleep(self._health_check_interval)
                    await self._perform_health_checks()
                except asyncio.CancelledError:
                    logger.info("Health check loop cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in health check loop: {e}")
                    await asyncio.sleep(5)  # Add delay before retry
        finally:
            logger.info("Health check loop stopped")

    async def _perform_health_checks(self):
        """Check health of all models and unload inactive ones"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)  # Unload after 1 hour of inactivity

        for model_unique_name in list(self._models.keys()):
            try:
                # Check if model is inactive
                if self._last_used.get(model_unique_name, now) < inactive_threshold:
                    logger.warning(f"Model {model_unique_name} inactive for >1 hour, unloading")
                    # await self.unload_model(model_unique_name)
                    continue

                # Perform health check
                model = self._models[model_unique_name]
                try:
                    await model.embed(["health check"], batch_size=1)
                    logger.debug(f"Model {model_unique_name} health check passed")
                except Exception as e:
                    logger.error(f"Health check failed for {model_unique_name}: {e}")
                    await self.reload_model(model_unique_name)

            except Exception as e:
                logger.error(f"Error during health check for {model_unique_name}: {e}")
                try:
                    await self.reload_model(model_unique_name)
                except Exception as reload_error:
                    logger.error(f"Failed to reload {model_unique_name}: {reload_error}")
                    await self.unload_model(model_unique_name)

    async def warmup(self) -> None:
        """Warm up all models in the pool"""
        try:
            async with self.redis as redis:
                model_unique_names = await redis.execute_command(
                    'SMEMBERS', f'index:category:{ModelCategory.EMBEDDING.value}'
                )
            
            for unique_name in model_unique_names:
                try:
                    await self.load_model(unique_name)
                    logger.success(f"Model {unique_name} warmed up successfully")
                except Exception as e:
                    logger.warning(f"Failed to warm up model {unique_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to warm up models: {e}")

    def get_pool_status(self) -> Dict:
        """Get current status of the model pool"""
        return {
            "loaded_models": list(self._models.keys()),
            "last_used": {k: v.isoformat() for k, v in self._last_used.items()},
            "health_check_active": self._health_check_task is not None and not self._health_check_task.done(),
            "health_check_interval": self._health_check_interval
        }