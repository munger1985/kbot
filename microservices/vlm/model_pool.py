import os
import aiohttp
import asyncio
from loguru import logger
from typing import Any
from datetime import datetime, timedelta
from model import *
from ms_core import ModelCategory


class ModelPool:
    """管理 VLM 模型池，包含健康检查和生命周期管理"""
    
    def __init__(self, health_check_interval: int = 600):
        """初始化模型池
        
        参数:
            health_check_interval: 健康检查间隔时间（秒）
        """
        self._models: dict[int, BaseVLM] = {}
        self._model_names: dict[int, str] = {}
        self._last_used: dict[int, datetime] = {}
        self._health_check_interval = health_check_interval
        self._health_check_task: asyncio.Task | None = None
        
    async def initialize(self):
        """初始化模型池并启动健康检查任务。"""
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
    async def shutdown(self):
        """关闭模型池和所有模型资源。"""
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

    async def _safe_shutdown_model(self, model_id: int, model: BaseVLM):
        """安全关闭单个模型并进行错误处理。
        
        Args:
            model_id: 模型ID
            model: 模型实例
        """
        try:
            await model.shutdown()
            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 关闭成功")
        except Exception as e:
            logger.error(f"关闭模型 {self._model_names.get(model_id, str(model_id))} 时发生错误: {e}")
        
    async def _start_model(self, model_id: int, model_data: dict[str, Any]) -> BaseVLM:
        """根据模型数据创建和启动模型实例。"""
        
        # 从模型数据中获取模型显示名称
        display_name = model_data.get("display_name")
        if not display_name:
            logger.warning(f"模型 {model_id} 缺少模型显示名称")

        # 从模型数据中获取模型名称
        model_name = model_data.get("model_name")
        if not model_name:
            raise ValueError(f"模型 {display_name or model_id} 没有模型名称")
        
        # 从模型数据中获取提供商
        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {display_name or model_name} 没有提供商")
        
        # 从模型数据中提取参数
        model_params = model_data["model_params"] if model_data.get("model_params") else {}
        
        # 根据模型类型创建相应的配置
        if provider == VLMProvider.OPENAI.value:
            api_endpoint = model_data.get("api_endpoint")
            api_key = model_data.get("api_key")

            if not api_endpoint or not api_key:
                raise ValueError(f"模型 {display_name or model_name} 没有 API 端点或 API 密钥")
            
            model_config = OpenAIVLMConfig(
                model_name=model_name,
                provider=provider,
                max_tokens=model_params.get("max_tokens", 512),
                api_key=api_key,
                api_endpoint=api_endpoint,
                api_version=model_params.get("api_version", ""),
                timeout=model_params.get("timeout", 30),
                max_retries=model_params.get("max_retries", 3),
                temperature=model_params.get("temperature", 0.1)
            )
        else:
            raise NotImplementedError(f"不支持的模型提供者: {provider}")

        # 创建和初始化模型
        try:
            model = create_vlm_model(model_config)
            await model.startup()
            self._models[model_id] = model
            self._model_names[model_id] = display_name or model_name
            self._last_used[model_id] = datetime.now()
            logger.success(f"模型 {display_name or model_name} 加载成功")
            return model
        except Exception as e:
            logger.error(f"创建模型 {display_name or model_name} 失败: {e}")
            raise RuntimeError(f"创建模型 {display_name or model_name} 失败: {e}")


    async def load_model(self, model_id: int) -> BaseVLM:
        """根据 model_id 加载模型实例
        
        参数:
            model_id: 要获取的模型的唯一名称
            
        返回:
            模型实例
            
        异常:
            ValueError: 如果在数据库中找不到 model_id
            RuntimeError: 如果模型创建失败
        """
        # 检查模型是否已加载
        if model_id in self._models:
            self._last_used[model_id] = datetime.now()
            return self._models[model_id]
        
        # 调用 main 服务从数据库获取模型信息
        try:
            # 从环境变量获取 main 服务的地址和端口
            main_host = os.getenv("KBOT_HOST") or "0.0.0.0"
            main_port = int(os.getenv("KBOT_PORT") or 8000)
            
            # 构建请求 URL
            url = f"http://{main_host}:{main_port}/api/model/params"
            headers = {"Content-Type": "application/json"}
            payload = {"model_id": model_id}
            timeout = aiohttp.ClientTimeout(total=30)
            
            # 发送请求
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logger.error(f"获取模型参数失败：HTTP {response.status} - {error_msg}")
                        raise ValueError(f"获取模型参数失败：HTTP {response.status} - {error_msg}")
                    
                    model_data = await response.json()
                  
            # 启动模型
            model = await self._start_model(model_id, model_data)
            return model

        except Exception as e:
            logger.exception(f"获取模型参数失败: {e}")
            raise ValueError(f"获取模型参数失败: {e}")

        
                    
    async def unload_model(self, model_id: int) -> bool:
        """从模型池中卸载指定模型。
        
        Args:
            model_id: 要卸载的模型ID
            
        Returns:
            bool: 卸载成功返回True，否则返回False
        """
        if model_id not in self._models:
            logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 未加载，无法卸载")
            return True
            
        model = self._models.pop(model_id)
        self._last_used.pop(model_id, None)
        
        try:
            await model.shutdown()
            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 卸载成功")
            return True
        except Exception as e:
            logger.error(f"卸载模型 {self._model_names.get(model_id, str(model_id))} 时发生错误: {e}")
            return False

    async def reload_model(self, model_id: int) -> bool:
        """重新加载模型池中的指定模型。
        
        Args:
            model_id: 要重新加载的模型ID
            
        Returns:
            bool: 加载是否成功
        """
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
        """后台任务，定期检查模型健康状态"""
        while True:
            await asyncio.sleep(self._health_check_interval)
            await self._perform_health_checks()
            
    async def _perform_health_checks(self):
        """检查所有模型的健康状况并卸载不活跃的模型"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)  # 1小时不活动后卸载
        
        for model_id in list(self._models.keys()):
            try:
                # 检查模型是否不活跃
                if self._last_used.get(model_id, now) < inactive_threshold:
                    logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 已超过1小时未活动")
                    # await self.unload_model(model_id)
                    continue
                    
                # 通过使用测试文本调用 embed 进行简单的健康检查
                model = self._models[model_id]
                try:
                    # 确保 health_check 返回的是可等待对象
                    if asyncio.iscoroutinefunction(model.health_check):
                        status = await model.health_check()
                    else:
                        # 如果 health_check 是同步方法，包装为异步结果
                        status = await asyncio.to_thread(model.health_check)
                    
                    if isinstance(status, dict):
                        if status.get('initialized', False):
                            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")
                        else:
                            logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败")
                            # 尝试重启模型
                            await self.reload_model(model_id)
                    else:
                        if getattr(status, 'initialized', False):
                            logger.info(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查通过")
                        else:
                            logger.warning(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败")
                            # 尝试重启模型
                            await self.reload_model(model_id)
                except Exception as e:
                    logger.error(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败: {e}")
                    # 尝试重启模型
                    await self.reload_model(model_id)
                
            except Exception as e:
                logger.error(f"模型 {self._model_names.get(model_id, str(model_id))} 健康检查失败: {e}")
                # 尝试重启模型
                try:
                    logger.info(f"正在尝试重启模型 {self._model_names.get(model_id, str(model_id))}")
                    await self.reload_model(model_id)
                except Exception as e:
                    logger.error(f"重启模型 {self._model_names.get(model_id, str(model_id))} 失败: {e}")
                    await self.unload_model(model_id)

    async def warmup(self) -> None:
        """预热模型池中的所有模型。
        
        Raises:
            Exception: 预热过程中发生错误时抛出
        """
        # 调用 main 服务从数据库获取模型信息
        try:
            # 从环境变量获取 main 服务的地址和端口
            main_host = os.getenv("KBOT_HOST") or "0.0.0.0"
            main_port = int(os.getenv("KBOT_PORT") or 8000)
            
            # 构建请求 URL
            url = f"http://{main_host}:{main_port}/api/model/available?model_category={ModelCategory.VLM.value}"
            timeout = aiohttp.ClientTimeout(total=30)
            
            # 发送请求
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        error_msg = await response.text()
                        logger.error(f"获取模型参数失败：HTTP {response.status} - {error_msg}")
                        raise ValueError(f"获取模型参数失败：HTTP {response.status} - {error_msg}")
                    
                    models = await response.json()

        except Exception as e:
            logger.exception(f"获取模型参数失败: {e}")
            raise ValueError(f"获取模型参数失败: {e}")

        try: 
            for model in models:
                await self._start_model(model["model_id"], model)

        except Exception as e:
            logger.exception(f"模型预热失败: {e}")

    def get_pool_status(self) -> dict:
        """获取模型池的当前状态信息。
        
        Returns:
            dict: 包含模型池状态信息的字典
        """
        return {
            "loaded_models": list(self._models.keys()),
            "last_used": {k: v.isoformat() for k, v in self._last_used.items()},
            "health_check_active": self._health_check_task is not None and not self._health_check_task.done(),
            "health_check_interval": self._health_check_interval
        }