import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from loguru import logger
from typing import Any, Callable, TypeVar, Generic
from time import monotonic

from platform_core.dictionary import Status
from platform_core.config.settings import get_app_config

from .model_repository import AIModelRepository as ModelRepository
from .entities.ai_model import AIModelEntity as Model

T = TypeVar('T')


@dataclass
class _ModelLockState:
    lock: asyncio.Lock
    users: int = 0


class BaseModelPool(ABC, Generic[T]):
    """提供并发安全加载、空闲回收和健康检查的通用模型池。"""
    
    def __init__(self, health_check_interval: int = 3600, session_factory: Callable | None = None):
        """初始化模型池；健康检查周期单位为秒。"""
        self._models: dict[str, T] = {}
        self._last_used: dict[str, float] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        # 每个服务名独立串行化生命周期操作，不让一个慢模型阻塞其他模型加载。
        self._model_locks: dict[str, _ModelLockState] = {}
        self._session_factory = session_factory

    def set_session_factory(self, session_factory: Callable) -> None:
        """在首次加载模型前绑定所属 App 的数据库 Session Factory。"""
        if self._models:
            raise RuntimeError("模型已加载，不能替换数据库 Session Factory")
        self._session_factory = session_factory

    @property
    def oracle_session(self):
        """从所属 App 运行时创建数据库 Session。"""
        if self._session_factory is None:
            raise RuntimeError("模型池未配置数据库 Session Factory")
        return self._session_factory()

    async def initialize(self) -> None:
        """启动模型池健康检查任务。"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning(f"{self.__class__.__name__} 健康检查任务已经运行")
            return

        coro = self._health_check_loop()
        self._health_check_task = asyncio.create_task(coro)
        
        # 为异步任务设置可观测名称。
        if hasattr(self._health_check_task, "set_name"):
            self._health_check_task.set_name(f"HealthCheck-{self.__class__.__name__}")
            
        logger.success(
            f"{self.__class__.__name__} 初始化完成，健康检查周期："
            f"{self._health_check_interval} 秒"
        )

    async def shutdown(self) -> None:
        """停止模型池并释放所有模型资源。"""
        logger.info(f"正在停止 {self.__class__.__name__}")

        # 先停止后台健康检查，避免关闭期间触发重载。
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"停止健康检查任务失败：{e}")

        # 并行释放已加载模型，缩短进程退出时间。
        if self._models:
            tasks = [
                self._safe_shutdown_model(name, model) 
                for name, model in self._models.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        self._models.clear()
        self._last_used.clear()
        self._model_locks.clear()
        logger.success(f"{self.__class__.__name__} 已停止")

    async def _safe_shutdown_model(self, served_model_name: str, model: T) -> None:
        """安全释放单个模型，避免一个实例阻塞整体关闭。"""
        try:
            await self._shutdown_model_instance(model)
            logger.info(f"模型 {served_model_name} 的资源已释放")
        except Exception as e:
            logger.error(f"释放模型 {served_model_name} 的资源失败：{e}")

    # --- 子类扩展点 ---

    @abstractmethod
    async def _shutdown_model_instance(self, model: T) -> None:
        """调用具体模型实例的资源释放逻辑。"""
        pass

    @abstractmethod
    async def _start_model(
        self, served_model_name: str, model_data: dict[str, Any],
    ) -> T:
        """根据目录配置创建并启动具体模型实例。"""
        pass

    @abstractmethod
    def _get_model_category(self) -> int:
        """返回当前进程负责的模型类别。"""
        pass

    @abstractmethod
    async def _perform_model_health_check(
        self, served_model_name: str, model: T,
    ) -> None:
        """执行模型专属健康检查。"""
        pass

    # --- 对外接口 ---

    @staticmethod
    def _normalize_served_name(served_model_name: str) -> str:
        normalized = served_model_name.strip()
        if not normalized:
            raise ValueError("served_model_name 不能为空")
        return normalized

    @asynccontextmanager
    async def _model_lifecycle_lock(self, served_model_name: str):
        """串行化同一模型的生命周期操作，并回收失败请求创建的空锁。"""
        state = self._model_locks.get(served_model_name)
        if state is None:
            state = _ModelLockState(lock=asyncio.Lock())
            self._model_locks[served_model_name] = state
        state.users += 1
        try:
            async with state.lock:
                yield
        finally:
            state.users -= 1
            if (
                state.users == 0
                and served_model_name not in self._models
                and self._model_locks.get(served_model_name) is state
            ):
                self._model_locks.pop(served_model_name, None)

    async def load_model(self, served_model_name: str) -> T:
        """按公开服务名加载模型；相同模型的并发冷启动只执行一次。"""
        served_model_name = self._normalize_served_name(served_model_name)
        async with self._model_lifecycle_lock(served_model_name):
            if served_model_name in self._models:
                self._last_used[served_model_name] = monotonic()
                return self._models[served_model_name]

            logger.info(f"正在加载模型配置：{served_model_name}")
            model_data = await self._fetch_model_data(served_model_name)
            model = await self._start_model(served_model_name, model_data)
            self._models[served_model_name] = model
            self._last_used[served_model_name] = monotonic()
            return model

    async def unload_model(self, served_model_name: str) -> bool:
        """卸载一个已加载模型；不存在时保持幂等。"""
        served_model_name = self._normalize_served_name(served_model_name)
        async with self._model_lifecycle_lock(served_model_name):
            model = self._models.pop(served_model_name, None)
            self._last_used.pop(served_model_name, None)
            if model is None:
                return True
            await self._safe_shutdown_model(served_model_name, model)
        return True

    async def reload_model(self, served_model_name: str) -> bool:
        """强制重启指定模型。"""
        await self.unload_model(served_model_name)
        try:
            await self.load_model(served_model_name)
            return True
        except Exception as e:
            logger.error(f"重启模型 {served_model_name} 失败：{e}")
            return False

    # --- 内部逻辑 ---

    async def _fetch_model_data(self, served_model_name: str) -> dict[str, Any]:
        """读取并校验当前进程可加载的模型定义。"""
        async with self.oracle_session as session:
            repo = ModelRepository(session)
            model = await repo.get_by_served_name(
                app_id=get_app_config().app_id,
                served_model_name=served_model_name,
            )
            if int(model.category) != int(self._get_model_category()):
                raise ValueError(
                    f"模型 {served_model_name} 不属于当前模型进程"
                )
            if int(model.status) != int(Status.ENABLED.value):
                raise ValueError(f"模型 {served_model_name} 未启用")
            return self._map_entity_to_dict(model)

    async def _fetch_available_models(self) -> list[dict[str, Any]]:
        """读取当前类别下所有已启用模型。"""
        async with self.oracle_session as session:
            repo = ModelRepository(session=session)
            entities = await repo.list_by_scope(
                app_id=get_app_config().app_id,
                category=self._get_model_category(),
            )
            entities = [
                model for model in entities
                if int(model.status) == int(Status.ENABLED.value)
            ]
            return [self._map_entity_to_dict(m) for m in entities]

    @staticmethod
    def _map_entity_to_dict(model: Model) -> dict[str, Any]:
        """将目录实体映射为模型驱动配置。"""
        return {
            "model_id": str(model.model_id),
            "served_model_name": model.served_model_name,
            "provider_model_name": model.provider_model_name,
            "category": model.category,
            "provider": model.provider,
            "api_endpoint": model.api_endpoint,
            "api_key": model.api_key,
            "model_path": model.model_params.get("model_path") if model.model_params else None,
            "model_params": model.model_params
        }

    async def _health_check_loop(self) -> None:
        """运行后台健康检查循环。"""
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
        except asyncio.CancelledError:
            logger.info(f"{self.__class__.__name__} 健康检查任务收到停止信号")
        except Exception as e:
            logger.exception(f"{self.__class__.__name__} 健康检查循环异常退出：{e}")
        finally:
            logger.info(f"{self.__class__.__name__} 健康检查任务已停止")

    async def _perform_health_checks(self) -> None:
        """检查空闲时间和具体模型健康状态。"""
        now = monotonic()
        inactive_threshold = now - 7200

        for served_model_name in list(self._models.keys()):
            # 自动回收长时间未调用的模型。
            last_time = self._last_used.get(served_model_name, now)
            if last_time < inactive_threshold:
                logger.info(f"正在卸载长时间未使用的模型：{served_model_name}")
                await self.unload_model(served_model_name)
                continue

            # 执行模型实现提供的健康检查。
            try:
                model = self._models.get(served_model_name)
                if model is not None:
                    await self._perform_model_health_check(served_model_name, model)
            except Exception as e:
                logger.error(
                    f"模型 {served_model_name} 健康检查失败，准备重启：{e}"
                )
                await self.reload_model(served_model_name)

    async def warmup(self) -> None:
        """串行预热当前类别下的全部启用模型，避免瞬时资源峰值。"""
        models_data = await self._fetch_available_models()
        if not models_data:
            return

        logger.info(f"准备预热 {len(models_data)} 个模型")
        for data in models_data:
            try:
                await self.load_model(data["served_model_name"])
            except Exception as e:
                logger.error(f"预热模型 {data['served_model_name']} 失败：{e}")

    def get_pool_status(self) -> dict[str, Any]:
        """返回不含凭据的模型池运行摘要。"""
        return {
            "pool_type": self.__class__.__name__,
            "loaded_count": len(self._models),
            "loaded_models": list(self._models.keys()),
            "health_check_running": self._health_check_task is not None and not self._health_check_task.done()
        }
