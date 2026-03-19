import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, TypeVar, Generic
from datetime import datetime, timedelta
from dao.repositories import AIModelRepository as ModelRepository
from dao.entities import AIModelEntity as Model
from core.database.oracle import get_session

# Define generic type T representing specific model instance types (e.g., BaseReranker, BaseEmbedding)
T = TypeVar('T')

class BaseModelPool(ABC, Generic[T]):
    """
    Base model pool class providing universal model management functionality.
    Improvements: Enhanced generic support, async lifecycle management, and static type checking compatibility.
    """
    
    def __init__(self, health_check_interval: int = 3600):
        """
        Initialize model pool.
        
        Args:
            health_check_interval: Health check interval in seconds
        """
        self._models: dict[str, T] = {}
        self._last_used: dict[str, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        # Protection lock to prevent concurrent loading of the same model
        self._load_lock = asyncio.Lock()

    @property
    def oracle_session(self):
        """Get Oracle database session"""
        return get_session()

    async def initialize(self) -> None:
        """Initialize model pool and start health check background task"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning(f"[{self.__class__.__name__}] Health check task is already running")
            return

        coro = self._health_check_loop()
        self._health_check_task = asyncio.create_task(coro)
        
        # Set task name for debugging purposes
        if hasattr(self._health_check_task, "set_name"):
            self._health_check_task.set_name(f"HealthCheck-{self.__class__.__name__}")
            
        logger.success(f"✅ {self.__class__.__name__} initialized successfully, health check interval: {self._health_check_interval}s")

    async def shutdown(self) -> None:
        """Shutdown model pool and release all model resources"""
        logger.info(f"🔄 Shutting down {self.__class__.__name__}...")

        # 1. Stop health check background task
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error stopping health check task: {e}")

        # 2. Shutdown all loaded models in parallel
        if self._models:
            tasks = [
                self._safe_shutdown_model(name, model) 
                for name, model in self._models.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        self._models.clear()
        self._last_used.clear()
        logger.success(f"✅ {self.__class__.__name__} shutdown completed successfully")

    async def _safe_shutdown_model(self, model_name: str, model: T) -> None:
        """Safely shutdown a single model with error handling"""
        try:
            await self._shutdown_model_instance(model)
            logger.info(f"Model {model_name} resources released")
        except Exception as e:
            logger.error(f"Failed to release model {model_name} resources: {e}")

    # --- Abstract Methods ---

    @abstractmethod
    async def _shutdown_model_instance(self, model: T) -> None:
        """Must be implemented by subclass: Call specific model's shutdown method"""
        pass

    @abstractmethod
    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> T:
        """Must be implemented by subclass: Instantiate and start model from configuration"""
        pass

    @abstractmethod
    def _get_model_category(self) -> int:
        """Must be implemented by subclass: Return corresponding category enum value from database"""
        pass

    @abstractmethod
    async def _perform_model_health_check(self, model_name: str, model: T) -> None:
        """Must be implemented by subclass: Execute model-specific health check logic (e.g., Ping or dummy inference)"""
        pass

    # --- Public Interface ---

    async def load_model(self, model_name: str) -> T:
        """Load model instance or retrieve from cache"""
        async with self._load_lock:  # Prevent concurrent loading of the same model
            if model_name in self._models:
                self._last_used[model_name] = datetime.now()
                return self._models[model_name]
            
            logger.info(f"🚀 Loading model from database: {model_name}")
            model_data = await self._fetch_model_data(model_name)
            if not model_data:
                raise ValueError(f"Model configuration not found in database: {model_name}")

            model = await self._start_model(model_name, model_data)
            self._models[model_name] = model
            self._last_used[model_name] = datetime.now()
            return model

    async def unload_model(self, model_name: str) -> bool:
        """Explicitly unload model"""
        if model_name not in self._models:
            return True
            
        model = self._models.pop(model_name)
        self._last_used.pop(model_name, None)
        
        await self._safe_shutdown_model(model_name, model)
        return True

    async def reload_model(self, model_name: str) -> bool:
        """Force restart model"""
        await self.unload_model(model_name)
        try:
            await self.load_model(model_name)
            return True
        except Exception as e:
            logger.error(f"Failed to restart model {model_name}: {e}")
            return False

    # --- Internal Logic ---

    async def _fetch_model_data(self, model_name: str) -> dict[str, Any]:
        """Retrieve configuration information from database"""
        async with self.oracle_session as session:
            repo = ModelRepository(session)
            model = await repo.get_by_name(model_name)
            return self._map_entity_to_dict(model) if model else {}

    async def _fetch_available_models(self) -> list[dict[str, Any]]:
        """Get all available model configurations for the category"""
        async with self.oracle_session as session:
            repo = ModelRepository(session=session)
            entities = await repo.get_available_by_category(self._get_model_category())
            return [self._map_entity_to_dict(m) for m in entities]

    @staticmethod
    def _map_entity_to_dict(model: Model) -> dict[str, Any]:
        """Convert model entity to configuration dictionary"""
        return {
            "model_name": model.model_name,
            "category": model.category,
            "provider": model.provider,
            "api_endpoint": model.api_endpoint,
            "api_key": model.api_key,
            "model_path": model.model_params.get("model_path") if model.model_params else None,
            "model_params": model.model_params
        }

    async def _health_check_loop(self) -> None:
        """Background health check loop"""
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
        except asyncio.CancelledError:
            logger.info(f"[{self.__class__.__name__}] Health check task received shutdown signal")
        except Exception as e:
            logger.exception(f"[{self.__class__.__name__}] Background loop crashed with exception: {e}")
        finally:
            logger.info(f"[{self.__class__.__name__}] Background task stopped completely")

    async def _perform_health_checks(self) -> None:
        """Actual health check execution logic"""
        now = datetime.now()
        # Automatically unload models unused for 2 hours (slightly longer than check interval)
        inactive_threshold = now - timedelta(hours=2)

        for model_name in list(self._models.keys()):
            # 1. Auto-cleanup long-inactive models (cold start strategy)
            last_time = self._last_used.get(model_name, now)
            if last_time < inactive_threshold:
                logger.info(f"♻️ Unloading idle model: {model_name}")
                await self.unload_model(model_name)
                continue

            # 2. Call subclass-implemented specific checks
            try:
                model = self._models[model_name]
                await self._perform_model_health_check(model_name, model)
            except Exception as e:
                logger.error(f"🚨 Model {model_name} status abnormal: {e}, attempting restart...")
                await self.reload_model(model_name)

    async def warmup(self) -> None:
        """Warmup: Start all models of this category from database"""
        models_data = await self._fetch_available_models()
        if not models_data:
            return

        logger.info(f"🔥 Warming up {len(models_data)} models...")
        # Serial execution or concurrency limiting recommended for warmup to prevent OOM
        for data in models_data:
            try:
                await self.load_model(data["model_name"])
            except Exception as e:
                logger.error(f"Failed to warmup model {data['model_name']}: {e}")

    def get_pool_status(self) -> dict[str, Any]:
        """Get pool status overview"""
        return {
            "pool_type": self.__class__.__name__,
            "loaded_count": len(self._models),
            "loaded_models": list(self._models.keys()),
            "health_check_running": self._health_check_task is not None and not self._health_check_task.done()
        }