import aiohttp
from loguru import logger
from dao.repositories.kbot_md_models_repo import *
from core.dictionary import ModelCategory
from configuration import ConfigManager


class ModelController:
    """模型控制器，负责模型的同步、启用、禁用等操作"""
    
    def __init__(self):
        self.repo = KbotMdModelsRepository()
    
    async def toggle(self, model_unique_name: str, enable: bool) -> bool:
        """
        启用/禁用指定模型
        
        步骤：
        1. 修改数据库状态
        2. 调用对应微服务接口加载/卸载模型
        
        Args:
            model_unique_name: 模型唯一名称
            enable: 是否启用
            
        Returns:
            bool: 启用/禁用是否成功
            
        Raises:
            ValueError: 未知的模型类型时抛出
        """
        # 1. 启用/禁用模型（修改数据库状态）
        try:
            if enable:
                await self.repo.enable_model(model_unique_name)
            else:
                await self.repo.disable_model(model_unique_name)
            logger.info(f"模型 {model_unique_name}: 数据库操作成功")

        except Exception as e:
            logger.error(f"模型 {model_unique_name}: 数据库操作失败: {e}")
            return False

        # 2. 调用对应微服务的接口，加载模型到内存中
        model_type = await self.repo.get_category_by_uname(model_unique_name)
        model_config = ConfigManager.get_model_config()

        if model_type == ModelCategory.EMBEDDING.value:
            service_host = model_config.embed.service_host
            service_port = model_config.embed.service_port
            total = model_config.embed.timeout
        elif model_type == ModelCategory.LLM.value:
            service_host = model_config.llm.service_host
            service_port = model_config.llm.service_port
            total = model_config.llm.timeout
        elif model_type == ModelCategory.RERANKER.value:
            service_host = model_config.reranker.service_host
            service_port = model_config.reranker.service_port
            total = model_config.reranker.timeout
        elif model_type == ModelCategory.VLM.value:
            service_host = model_config.vlm.service_host
            service_port = model_config.vlm.service_port
            total = model_config.vlm.timeout
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
            
        timeout = aiohttp.ClientTimeout(total=total)
        url = f"http://{service_host}:{service_port}/load"
        headers = {"Content-Type": "application/json"}
        payload = {"model_unique_name": model_unique_name,
                   "operation": "load" if enable else "unload"}
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url=url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"模型 {model_unique_name} 微服务操作成功")
                        return True
                    else:
                        logger.error(f"模型 {model_unique_name} 微服务操作失败，状态码: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"调用微服务修改模型失败: {e}")
            return False
    
    
    async def get_model_params_by_uname(self, model_unique_name: str) -> dict | None:
        """
        通过模型唯一名称获取模型参数
        
        Args:
            model_unique_name: 模型唯一名称
            
        Returns:
            dict | None: 模型参数，如果不存在则返回None
        """
        result = await self.repo.get_by_uname(model_unique_name)
        if result:
            return {
                "model_name": result.model_name,
                "category": result.category,
                "provider": result.provider,
                "model_params": result.model_params,
                "model_unique_name": result.model_unique_name,
                "api_endpoint": result.api_endpoint,
                "api_key": result.api_key
            }
        else:
            return None

    async def get_all_available_models(self, model_category: int) -> Sequence[dict]:
        """
        根据类别获取所有可用模型
        
        Args:
            model_category: 模型类别
        
        Returns:
            Sequence[dict]: 模型列表（字典格式）
        """
        models = await self.repo.get_available_by_category(model_category)
        return [
            {
                "model_name": model.model_name,
                "category": model.category,
                "provider": model.provider,
                "model_params": model.model_params,
                "model_unique_name": model.model_unique_name,
                "api_endpoint": model.api_endpoint,
                "api_key": model.api_key
            }
            for model in models
        ]
    