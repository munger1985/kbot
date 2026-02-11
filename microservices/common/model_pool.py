import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, TypeVar, Generic
from datetime import datetime, timedelta
from .model_repo import KbotMdModelsRepository as ModelRepository
from .model_entity import KbotMdModels as Model

# 定义泛型 T，代表具体的模型实例类型（如 BaseReranker, BaseEmbedding）
T = TypeVar('T')

class BaseModelPool(ABC, Generic[T]):
    """
    模型池基类，提供通用的模型管理功能
    优化项：完善了泛型支持、异步生命周期管理及静态类型检查适配
    """
    
    def __init__(self, health_check_interval: int = 3600):
        """
        初始化模型池
        
        Args:
            health_check_interval: 健康检查间隔时间（秒）
        """
        self._models: dict[str, T] = {}
        self._last_used: dict[str, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        # 保护锁，防止并发加载同一个模型
        self._load_lock = asyncio.Lock()

    async def initialize(self) -> None:
        """初始化模型池并启动健康检查任务"""
        if self._health_check_task and not self._health_check_task.done():
            logger.warning(f"[{self.__class__.__name__}] 健康检查任务已在运行中")
            return

        coro = self._health_check_loop()
        self._health_check_task = asyncio.create_task(coro)
        
        # 设置任务名称便于调试
        if hasattr(self._health_check_task, "set_name"):
            self._health_check_task.set_name(f"HealthCheck-{self.__class__.__name__}")
            
        logger.success(f"✅ {self.__class__.__name__} 初始化完成，健康检查间隔: {self._health_check_interval}s")

    async def shutdown(self) -> None:
        """关闭模型池并释放所有模型资源"""
        logger.info(f"🔄 正在关闭 {self.__class__.__name__}...")

        # 1. 停止健康检查后台任务
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"停止健康检查任务时出错: {e}")

        # 2. 并行关闭所有已加载的模型
        if self._models:
            tasks = [
                self._safe_shutdown_model(name, model) 
                for name, model in self._models.items()
            ]
            await asyncio.gather(*tasks, return_exceptions=True)

        self._models.clear()
        self._last_used.clear()
        logger.success(f"✅ 成功关闭 {self.__class__.__name__}")

    async def _safe_shutdown_model(self, model_name: str, model: T) -> None:
        """安全关闭单个模型并进行错误处理"""
        try:
            await self._shutdown_model_instance(model)
            logger.info(f"模型 {model_name} 资源已释放")
        except Exception as e:
            logger.error(f"释放模型 {model_name} 资源失败: {e}")

    # --- 抽象方法 ---

    @abstractmethod
    async def _shutdown_model_instance(self, model: T) -> None:
        """子类需实现：调用具体模型的 shutdown 方法"""
        pass

    @abstractmethod
    async def _start_model(self, model_name: str, model_data: dict[str, Any]) -> T:
        """子类需实现：根据配置实例化并启动模型"""
        pass

    @abstractmethod
    def _get_model_category(self) -> int:
        """子类需实现：返回对应数据库中的 category 枚举值"""
        pass

    @abstractmethod
    async def _perform_model_health_check(self, model_name: str, model: T) -> None:
        """子类需实现：执行特定模型的健康检查逻辑（如 Ping 或 dummy 推理）"""
        pass

    # --- 公共接口 ---

    async def load_model(self, model_name: str) -> T:
        """加载或从缓存获取模型实例"""
        async with self._load_lock:  # 防止并发加载同一模型
            if model_name in self._models:
                self._last_used[model_name] = datetime.now()
                return self._models[model_name]
            
            logger.info(f"🚀 正在从数据库加载模型: {model_name}")
            model_data = await self._fetch_model_data(model_name)
            if not model_data:
                raise ValueError(f"数据库中未找到模型配置: {model_name}")

            model = await self._start_model(model_name, model_data)
            self._models[model_name] = model
            self._last_used[model_name] = datetime.now()
            return model

    async def unload_model(self, model_name: str) -> bool:
        """显式卸载模型"""
        if model_name not in self._models:
            return True
            
        model = self._models.pop(model_name)
        self._last_used.pop(model_name, None)
        
        await self._safe_shutdown_model(model_name, model)
        return True

    async def reload_model(self, model_name: str) -> bool:
        """强制重启模型"""
        await self.unload_model(model_name)
        try:
            await self.load_model(model_name)
            return True
        except Exception as e:
            logger.error(f"模型 {model_name} 重启失败: {e}")
            return False

    # --- 内部逻辑 ---

    async def _fetch_model_data(self, model_name: str) -> dict[str, Any]:
        """从数据库读取配置信息"""
        repo = ModelRepository()
        model = await repo.get_by_name(model_name)
        return self._map_entity_to_dict(model) if model else {}

    async def _fetch_available_models(self) -> list[dict[str, Any]]:
        """获取所属类别的所有可用模型配置"""
        repo = ModelRepository()
        entities = await repo.get_available_by_category(self._get_model_category())
        return [self._map_entity_to_dict(m) for m in entities]

    @staticmethod
    def _map_entity_to_dict(model: Model) -> dict[str, Any]:
        """将模型实体转换为配置字典"""
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
        """后台健康检查循环"""
        try:
            while True:
                await asyncio.sleep(self._health_check_interval)
                await self._perform_health_checks()
        except asyncio.CancelledError:
            logger.info(f"[{self.__class__.__name__}] 健康检查任务收到了退出信号")
        except Exception as e:
            logger.exception(f"[{self.__class__.__name__}] 后台循环异常崩溃: {e}")
        finally:
            logger.info(f"[{self.__class__.__name__}] 后台任务已彻底停止")

    async def _perform_health_checks(self) -> None:
        """具体的健康检查执行逻辑"""
        now = datetime.now()
        # 默认 2 小时未使用的模型自动卸载（比检查间隔稍长）
        inactive_threshold = now - timedelta(hours=2)

        for model_name in list(self._models.keys()):
            # 1. 自动清理长时间不使用的模型（冷启动策略）
            last_time = self._last_used.get(model_name, now)
            if last_time < inactive_threshold:
                logger.info(f"♻️ 卸载长期闲置模型: {model_name}")
                await self.unload_model(model_name)
                continue

            # 2. 调用子类实现的具体检查
            try:
                model = self._models[model_name]
                await self._perform_model_health_check(model_name, model)
            except Exception as e:
                logger.error(f"🚨 模型 {model_name} 状态异常: {e}，尝试重启...")
                await self.reload_model(model_name)

    async def warmup(self) -> None:
        """预热：启动数据库中该类别的所有模型"""
        models_data = await self._fetch_available_models()
        if not models_data:
            return

        logger.info(f"🔥 正在预热 {len(models_data)} 个模型...")
        # 预热过程建议串行或限制并发，防止显存 OOM
        for data in models_data:
            try:
                await self.load_model(data["model_name"])
            except Exception as e:
                logger.error(f"预热模型 {data['model_name']} 失败: {e}")

    def get_pool_status(self) -> dict[str, Any]:
        """获取状态概览"""
        return {
            "pool_type": self.__class__.__name__,
            "loaded_count": len(self._models),
            "loaded_models": list(self._models.keys()),
            "health_check_running": self._health_check_task is not None and not self._health_check_task.done()
        }