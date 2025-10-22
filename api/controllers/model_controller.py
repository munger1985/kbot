import aiohttp
from loguru import logger
from PIL import Image
from dao.repositories.kbot_md_models_repo import *
from core.dictionary import ModelCategory
from configuration import ConfigManager
from utils.call_models import CallModel


class ModelController:
    """模型控制器，负责模型的同步、启用、禁用等操作"""
    
    def __init__(self):
        self.repo = KbotMdModelsRepository()
    
    async def toggle(self, model_id: int, enable: bool) -> bool:
        """
        启用/禁用指定模型
        
        步骤：
        1. 修改数据库状态
        2. 调用对应微服务接口加载/卸载模型
        
        Args:
            model_id: 模型唯一标识
            enable: 是否启用
            
        Returns:
            bool: 启用/禁用是否成功
            
        Raises:
            ValueError: 未知的模型类型时抛出
        """
        # 1. 启用/禁用模型（修改数据库状态）
        try:
            if enable:
                await self.repo.enable_model(model_id)
            else:
                await self.repo.disable_model(model_id)
            logger.info(f"模型 {model_id}: 数据库操作成功")

        except Exception as e:
            logger.error(f"模型 {model_id}: 数据库操作失败: {e}")
            return False

        # 2. 调用对应微服务的接口，加载模型到内存中
        model_type = await self.repo.get_category_by_id(model_id)
        model_config = ConfigManager.get_model_config()

        if model_type == ModelCategory.TXT_EMBEDDING.value:
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
        payload = {"model_id": model_id,
                   "operation": "load" if enable else "unload"}
        
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url=url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        logger.info(f"模型 {model_id} 微服务操作成功")
                        return True
                    else:
                        logger.error(f"模型 {model_id} 微服务操作失败，状态码: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"调用微服务修改模型失败: {e}")
            return False
    
    
    async def get_model_by_id(self, model_id: int) -> dict | None:
        """
        通过模型唯一标识获取模型参数
        
        Args:
            model_id: 模型唯一标识
            
        Returns:
            dict | None: 模型参数，如果不存在则返回None
        """
        model = await self.repo.get_by_id(model_id)
        if model:
            return {
                "model_id": model_id,
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
    
    async def verify_model(self, model_id: int, model_type: int) -> bool:
        """
        验证指定模型
        
        Args:
            model_id: 模型唯一标识
            model_type: 模型类型
            
        Returns:
            bool: 模型验证是否成功
            
        Raises:
            ValueError: 未知的模型类型时抛出
        """

        
        model_config = ConfigManager.get_model_config()

        # 测试Embedding模型
        if model_type == ModelCategory.TXT_EMBEDDING.value:
            input_texts = ["test"]
            result = await CallModel().call_embedding_model(
                model_id, 
                input_texts
            )
            
        # 测试LLM模型
        elif model_type == ModelCategory.LLM.value:
            input_text = "test"
            async for chunk in CallModel().call_llm_model(
                model_id,
                input_text,
                stream=False,
                max_tokens=5
            ):
                result = chunk

        # 测试Reranker模型
        elif model_type == ModelCategory.RERANKER.value:
            question = "test"
            inputs_list = [
                "test1",
                "test2"
            ]
            result = await CallModel().call_reranker_model(
                model_id,
                question,
                inputs_list,
                1
            )

        # 测试VLM模型
        elif model_type == ModelCategory.VLM.value:
            prompt_unique_name = "KBOT1/pdf_parsing"
            # 创建纯色图片的最简代码
            image = Image.new('RGB', (100, 100), 'lightblue')
            result = await CallModel().call_vlm_model_for_parsing_picture(
                model_id, 
                image
            )
            
        else:
            raise ValueError(f"未知的模型类型: {model_type}")
        
        if result:
            return True
        else:
            return False
        
model_controller = ModelController()      