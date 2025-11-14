import os
import aiohttp
import asyncio
from abc import ABC, abstractmethod
from loguru import logger
from typing import Any, TypeVar, Generic
from datetime import datetime, timedelta
from .model_entity import KbotMdModels
from .model_repo import KbotMdModelsRepository

T = TypeVar('T')  # 模型类型


class BaseModelPool(ABC, Generic[T]):
    """模型池基类，提供通用的模型管理功能"""
    
    def __init__(self, health_check_interval: int = 600):
        """初始化模型池
        
        Args:
            health_check_interval: 健康检查间隔时间（秒）
        """
        self._models: dict[int, T] = {}
        self._model_names: dict[int, str] = {}
        self._last_used: dict[int, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None

    async def initialize(self):
        """初始化模型池并启动健康检查任务"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())

    async def shutdown(self):
        """关闭模型池和所有模型资源"""
        # 取消健康检查任务
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                logger.info("健康检查任务已取消")
            except Exception as e:
                logger.error(f"取消健康检查任务时发生错误: {e}")

        # 关闭所有模型
        shutdown_tasks = []
        for model_id, model in self._models.items():
            shutdown_tasks.append(asyncio.create_task(
                self._safe_shutdown_model(model_id, model)
            ))

        # 等待所有关闭任务完成
        if shutdown_tasks:
            await asyncio.wait(shutdown_tasks)

        self._models.clear()
        self._last_used.clear()
        logger.info("模型池已关闭")

    async def _safe_shutdown_model(self, model_id: int, model: T):
        """安全关闭单个模型并进行错误处理"""
        try:
            await self._shutdown_model_instance(model)
            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 关闭成功")
        except Exception as e:
            logger.error(f"关闭模型 {self._model_names.get(model_id, str(model_id))} 时发生错误: {e}")

    @abstractmethod
    async def _shutdown_model_instance(self, model: T):
        """关闭模型实例的具体实现"""
        pass

    @abstractmethod
    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> T:
        """根据模型数据创建和启动模型实例"""
        pass

    @abstractmethod
    def _get_model_category(self) -> str:
        """获取模型类别"""
        pass

    @abstractmethod
    async def _perform_model_health_check(self, model_id: int, model: T):
        """执行模型健康检查的具体实现"""
        pass

    async def _fetch_model_data(self, model_id: int) -> dict[str, Any]:
        """从主服务获取模型数据"""
        
        repo = KbotMdModelsRepository()
        model = await repo.get_by_id(model_id)
        if model:
            return {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "display_name": model.display_name,
                "category": model.category,
                "provider": model.provider,
                "model_params": model.model_params,
                "api_endpoint": model.api_endpoint,
                "api_key": model.api_key
            }
        else:
            return None

    async def _fetch_available_models(self) -> list:
        """获取可用的模型列表"""
            
        repo = KbotMdModelsRepository()
        model_category = self._get_model_category()
        models = await repo.get_available_by_category(model_category)
        return [
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "display_name": model.display_name,
                "category": model.category,
                "provider": model.provider,
                "model_params": model.model_params,
                "api_endpoint": model.api_endpoint,
                "api_key": model.api_key
            }
            for model in models
        ]

    async def load_model(self, model_id: int) -> T:
        """根据模型ID加载模型实例"""
        if model_id in self._models:
            logger.debug(f"模型 {model_id} 已缓存，直接返回")
            self._last_used[model_id] = datetime.now()
            return self._models[model_id]
        
        logger.debug(f"模型 {model_id} 未缓存，尝试从数据库加载。当前缓存模型: {list(self._models.keys())}")
        
        model_data = await self._fetch_model_data(model_id)
        model = await self._start_model(model_id, model_data)
        return model

    async def unload_model(self, model_id: int) -> bool:
        """从模型池中卸载指定模型"""
        if model_id not in self._models:
            logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 未加载，无法卸载")
            return True
            
        model = self._models.pop(model_id)
        self._last_used.pop(model_id, None)
        
        try:
            await self._shutdown_model_instance(model)
            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 卸载成功")
            return True
        except Exception as e:
            logger.error(f"卸载模型 {self._model_names.get(model_id, str(model_id))} 时发生错误: {e}")
            return False

    async def reload_model(self, model_id: int) -> bool:
        """重新加载模型池中的指定模型"""
        if model_id in self._models:
            await self.unload_model(model_id)

        try:
            await self.load_model(model_id)
            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 重新加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型 {self._model_names.get(model_id, str(model_id))} 时发生错误: {e}")
            return False

    async def _health_check_loop(self):
        """后台任务：定期检查模型健康状态"""
        try:
            while True:
                try:
                    await asyncio.sleep(self._health_check_interval)
                    await self._perform_health_checks()
                except asyncio.CancelledError:
                    logger.info("健康检查循环已取消")
                    break
                except Exception as e:
                    logger.error(f"健康检查循环中发生错误: {e}")
                    await asyncio.sleep(5)
        finally:
            logger.info("健康检查循环已停止")

    async def _perform_health_checks(self):
        """检查所有模型的健康状态并卸载不活跃的模型"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)

        for model_id in list(self._models.keys()):
            try:
                # 检查模型是否不活跃
                if self._last_used.get(model_id, now) < inactive_threshold:
                    logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 已超过1小时未使用")
                    continue

                # 执行健康检查
                await self._perform_model_health_check(model_id, self._models[model_id])

            except Exception as e:
                logger.error(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查过程中发生错误: {e}")
                try:
                    logger.info(f"正在尝试重启模型 {self._model_names.get(model_id, str(model_id))}")
                    await self.reload_model(model_id)
                except Exception as reload_error:
                    logger.error(f"重新加载模型 {self._model_names.get(model_id, str(model_id))} 失败: {reload_error}")
                    await self.unload_model(model_id)

    async def warmup(self) -> None:
        """预热模型池中的所有模型"""
        try:
            models = await self._fetch_available_models()
            for model_data in models:
                model_id = int(model_data["model_id"])
                logger.debug(f"正在预热模型 {model_id}，模型名称: {model_data.get('display_name', 'N/A')}")
                await self._start_model(model_id, model_data)
                logger.debug(f"模型 {model_id} 预热完成，已缓存: {model_id in self._models}")

        except Exception as e:
            logger.exception(f"模型预热失败: {e}")

    def get_pool_status(self) -> dict:
        """获取模型池的当前状态信息"""
        return {
            "loaded_models": list(self._models.keys()),
            "last_used": {k: v.isoformat() for k, v in self._last_used.items()},
            "health_check_active": self._health_check_task is not None and not self._health_check_task.done(),
            "health_check_interval": self._health_check_interval
        }