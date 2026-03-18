import aiohttp
from loguru import logger
from PIL import Image
from dao.repositories.ai_model_repo import *
from core.dictionary import ModelCategory
from core.config.settings import get_settings
from utils.clients import AIModelClient
from services.ai_model import AIModelService
from core.exceptions import ParamValueError, InternalServerError, NotFoundError


class ModelController:
    """模型控制器，负责模型的同步、启用、禁用等操作"""
    def __init__(self):
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
    
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

        # 测试Embedding模型
        model_name = await self.model_service.get_model_name_by_id(model_id)
        if model_type == ModelCategory.TXT_EMBEDDING.value:
            input_texts = ["test"]
            result = await self.model_client.call_embedding_model(
                model_name, 
                input_texts
            )
            
        # 测试LLM模型
        elif model_type == ModelCategory.LLM.value:
            input_text = "test"
            async for chunk in self.model_client.call_llm_model(
                model_name,
                input_text,
                stream=False,
                max_tokens=16
            ):
                result = chunk

        # 测试Reranker模型
        elif model_type == ModelCategory.RERANKER.value:
            question = "test"
            inputs_list = [
                "test1",
                "test2"
            ]
            result = await self.model_client.call_reranker_model(
                model_name,
                question,
                inputs_list,
                1
            )

        # 测试VLM模型
        elif model_type == ModelCategory.VLM.value:
            prompt_unique_name = "KBOT1/pdf_parsing"
            # 创建纯色图片的最简代码
            image = Image.new('RGB', (100, 100), 'lightblue')
            result = await self.model_client.call_vlm_model(
                model_name, 
                image,
                prompt="描述该图片"
            )
            
        else:
            raise ParamValueError(f"未知的模型类型: {model_type}")
        
        if result:
            return True
        else:
            raise InternalServerError(message=f"模型 {model_id} 验证失败")
        
model_controller = ModelController()      