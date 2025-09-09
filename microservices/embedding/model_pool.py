import asyncio
import aiohttp
import os
from typing import Any
from datetime import datetime, timedelta
from loguru import logger
from ms_core import ConfigManager, ModelCategory
from model import *


class ModelPool:
    """管理模型池，提供健康检查和生命周期管理功能"""

    def __init__(self, health_check_interval: int = 600):
        """初始化模型池实例。
        
        Args:
            health_check_interval: 健康检查间隔时间（秒）
        """
        self._models: dict[str, BaseEmbedding] = {}
        self._last_used: dict[str, datetime] = {}
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

    async def _safe_shutdown_model(self, model_id: str, model: BaseEmbedding):
        """安全关闭单个模型并进行错误处理。
        
        Args:
            model_id: 模型ID
            model: 模型实例
        """
        try:
            await model.shutdown()
            logger.info(f"模型 {model_id} 关闭成功")
        except Exception as e:
            logger.error(f"关闭模型 {model_id} 时发生错误: {e}")

    async def _start_model(self, model_id: str, model_data: dict[str, Any]) -> BaseEmbedding:
        """根据模型参数启动模型。"""
        
        # 从模型数据中提取参数
        model_params = model_data["model_params"] if model_data.get("model_params") else {}

        provider = model_data.get("provider")
        if not provider:
            raise ValueError(f"模型 {model_id} 缺少提供者信息")

        model_name = model_data.get("model_name")
        if not model_name:
            raise ValueError(f"模型 {model_id} 缺少模型名称")


        # 从Nacos加载配置或使用默认值
        config = ConfigManager.get_model_config()
        max_tokens = config.embed.max_tokens
        timeout = config.embed.timeout
        max_retries = config.embed.max_retries
        cache_dir = config.embed.cache_dir

        # 根据配置创建模型实例 
        try:
            # 不同的提供者需要不同的配置
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
                    raise ValueError(f"模型 {model_id} 缺少必要参数")
                model_config = OCIEmbeddingConfig(
                    model_name=model_name,
                    provider=provider,
                    max_tokens=model_params.get("max_tokens", max_tokens),
                    batch_size=model_params.get("batch_size", 2),
                    api_endpoint=api_endpoint, # type: ignore
                    compartment_id=compartment_id, # type: ignore
                    config_file=config_file # type: ignore
                )
            else:
                raise ValueError(f"不支持的提供者: {provider}")

            # 创建并初始化模型
            model = create_embedding_model(model_config)
            await model.startup()
            self._models[model_id] = model
            self._last_used[model_id] = datetime.now()
            logger.success(f"模型 {model_name} 加载成功")
            return model

        except Exception as e:
            logger.exception(f"加载模型 {model_id} 失败")
            raise RuntimeError(f"加载模型 {model_id} 失败: {e}")

    async def load_model(self, model_id: str) -> BaseEmbedding:
        """通过模型ID加载模型实例。
        
        Args:
            model_id: 模型唯一标识符
            
        Returns:
            BaseEmbedding: 加载的模型实例
            
        Raises:
            ValueError: 当模型未找到或缺少必要参数时抛出
            RuntimeError: 当模型加载失败时抛出
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
            payload = {"model_unique_name": model_id}
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

        

    async def unload_model(self, model_id: str) -> bool:
        """从模型池中卸载指定模型。
        
        Args:
            model_id: 要卸载的模型ID
            
        Returns:
            bool: 卸载成功返回True，否则返回False
        """
        if model_id not in self._models:
            logger.warning(f"模型 {model_id} 未加载，无法卸载")
            return True
            
        model = self._models.pop(model_id)
        self._last_used.pop(model_id, None)
        
        try:
            await model.shutdown()
            logger.info(f"模型 {model_id} 卸载成功")
            return True
        except Exception as e:
            logger.error(f"卸载模型 {model_id} 时发生错误: {e}")
            return False

    async def reload_model(self, model_id: str) -> bool:
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
            logger.info(f"模型 {model_id} 加载成功")
            return True
        except Exception as e:
            logger.error(f"加载模型 {model_id} 失败: {e}")
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
                    await asyncio.sleep(5)  # 重试前添加延迟
        finally:
            logger.info("健康检查循环已停止")

    async def _perform_health_checks(self):
        """检查所有模型的健康状态并卸载不活跃的模型"""
        now = datetime.now()
        inactive_threshold = now - timedelta(hours=1)  # 1小时不活动后卸载

        for model_id in list(self._models.keys()):
            try:
                # 检查模型是否不活跃
                if self._last_used.get(model_id, now) < inactive_threshold:
                    logger.warning(f"模型 {model_id} 已超过1小时未使用，准备卸载")
                    # await self.unload_model(model_id)
                    continue

                # 执行健康检查
                model = self._models[model_id]
                try:
                    # 如果模型不支持则跳过健康检查
                    await model.embed(["健康检查"], batch_size=1)
                    logger.debug(f"模型 {model_id} 健康检查通过")
                except Exception as e:
                    logger.error(f"模型 {model_id} 健康检查失败: {e}")
                    await self.reload_model(model_id)

            except Exception as e:
                logger.error(f"模型 {model_id} 健康检查过程中发生错误: {e}")
                try:
                    await self.reload_model(model_id)
                except Exception as e:
                    logger.error(f"重新加载模型 {model_id} 失败: {e}")
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
            url = f"http://{main_host}:{main_port}/api/model/available"
            headers = {"Content-Type": "application/json"}
            payload = {"model_category": ModelCategory.EMBEDDING.value}
            timeout = aiohttp.ClientTimeout(total=30)
            
            # 发送请求
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, headers=headers, json=payload) as response:
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
                await self._start_model(model["model_unique_name"], model)

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