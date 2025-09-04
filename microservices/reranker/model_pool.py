import json
import asyncio
from loguru import logger
from datetime import datetime, timedelta
from model import *
from ms_core import load_config, ModelConfig, ModelCategory, AsyncRedisPool


class ModelPool:
    """Manage a pool of reranker models with health checking and lifecycle management"""
    
    def __init__(self, health_check_interval: int = 300):
        """Initialize model pool.
        Args:
            health_check_interval: Interval in seconds between health checks
        """
        self._models: dict[str, BaseReranker] = {}
        self._last_used: dict[str, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        self.redis = AsyncRedisPool(db=1)
        
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
        
    async def load_model(self, model_unique_name: str) -> BaseReranker:
        """Load a model instance by model_unique_name
        
        Args:
            model_unique_name: ID of the model to get
            
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
            
        # 根据 model_unique_name 从 Redis 获取模型信息
        async with self.redis as redis:
            # 1. 先通过 unique_name 获取 model_id
            model_id = await redis.get(f"index:unique_name:{model_unique_name}")
            if not model_id:
                raise ValueError(f"Model {model_unique_name} not found in redis")
            
            # 2. 通过 model_id 获取所有字段
            model_data = await redis.hgetall(f"model:{model_id}")
            if not model_data:
                raise ValueError(f"Model {model_unique_name} not found in redis")
            
            # 处理 JSON 字符串字段
            if model_data.get("model_params"):
                model_params = json.loads(model_data["model_params"])

        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"Model {model_unique_name} has no provider")
        
        # 从模型数据中获取参数
        model_name = model_data.get("model_name")
        if not model_name:
            raise ValueError(f"Model {model_unique_name} has no model_name")
        
        # 从 Nacos 获取配置信息
        config = load_config("model_config")
        if not isinstance(config, ModelConfig):
            raise ValueError
  

        # 根据模型类型创建相应的配置
        if provider == RerankerProvider.LOCAL.value:
            if "jina" in model_name.lower():
                model_config = JinaRerankerConfig(
                    provider = provider,
                    model_name = model_name,
                    model_path = model_params.get("model_path", None),
                    device = model_params.get("device", None),
                    device_map = model_params.get("device_map", None),
                    max_tokens = model_params.get("max_tokens", 512),
                    batch_size = model_params.get("batch_size", 16),
                    compile_model = model_params.get("compile_model", True),
                    use_fp16 = model_params.get("use_fp16", True),
                    trust_remote_code = model_params.get("trust_remote_code", True),
                    local_files_only = model_params.get("local_files_only", False),
                    max_memory = model_params.get("max_memory", None),
                    cache_dir = config.reranker.cache_dir or "./cached_models"
                )
            else:
                model_config = LocalRerankerConfig(
                    provider = provider,
                    model_name = model_name,
                    model_path = model_params.get("model_path", None),
                    device = model_params.get("device", None),
                    device_map = model_params.get("device_map", None),
                    max_tokens = model_params.get("max_tokens", 8192),
                    batch_size = model_params.get("batch_size", 16),
                    compile_model = model_params.get("compile_model", True),
                    use_fp16 = model_params.get("use_fp16", False),
                    trust_remote_code = model_params.get("trust_remote_code", True),
                    local_files_only = model_params.get("local_files_only", False),
                    max_memory = model_params.get("max_memory", None),
                    cache_dir = config.reranker.cache_dir or "./cached_models"
                )
            
        elif provider == RerankerProvider.COHERE.value:
            api_endpoint = model_data.get("api_endpoint")
            api_key = model_data.get("api_key")
            if not api_endpoint or not api_key:
                raise ValueError(f"Model {model_unique_name} has no api_endpoint or api_key")
            model_config = CohereRerankerConfig(
                provider = provider,
                model_name = model_name,
                max_tokens = model_params.get("max_tokens", 8192),
                api_key = api_key,
                api_endpoint = api_endpoint,
                timeout  = model_params.get("timeout", 10)
            )
        else:
            raise ValueError(f"Unsupported reranker model: {provider}")
        
        # Create and initialize model //创建和初始化模型
        try:
            model = create_reranker_model(model_config)
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
            model_unique_name: ID of the model to unload
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
            model_unique_name: ID of the model to reload
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
                    logger.warning(f"Model {model_unique_name} is inactive for more than 1 hour")
                    # await self.unload_model(model_unique_name)
                    continue
                    
                # Simple health check by calling embed with a test text
                model = self._models[model_unique_name]

                await model.rerank(query="test", documents=["test"], top_k=1)
                logger.success(f"Health check passed for model {model_unique_name}")

            except Exception as e:
                logger.error(f"Health check failed for model {model_unique_name}: {e}")
                # Try to restart the model
                try:
                    logger.info(f"Attempting to restart model {model_unique_name}")
                    await self.reload_model(model_unique_name)
                except Exception as restart_error:
                    logger.error(f"Failed to restart model {model_unique_name}: {restart_error}")
                    await self.unload_model(model_unique_name)

    async def warmup(self) -> None:
        """Warm up all models in the pool"""
        async with self.redis as redis:
            # 直接获取对应 category 集合中的所有 model_unique_names
            model_unique_names = await redis.execute_command('SMEMBERS', f'index:category:{ModelCategory.RERANKER.value}')
        for unique_name in model_unique_names:
            try:
                await self.load_model(unique_name)
                logger.success(f"Model {unique_name} warmed up successfully")
            except Exception as e:
                logger.warning(f"Failed to warm up models: {e}")
                continue