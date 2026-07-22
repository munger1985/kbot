from loguru import logger
from PIL import Image
from platform_core.dictionary import ModelCategory
from platform_core.config.settings import get_settings
from platform_clients import AIModelClient
from services.basic import AIModelService
from platform_core.exceptions import ParamValueError, InternalServerError


class ModelController:
    """Model controller, responsible for model synchronization, activation, deactivation and other operations"""
    def __init__(self):
        self.model_client = AIModelClient()
        self.model_service = AIModelService()
    
    async def verify_model(self, model_id: int, model_type: int) -> bool:
        """
        Verify the specified model
        
        Args:
            model_id: Unique identifier of the model
            model_type: Type of the model
            
        Returns:
            bool: Whether the model verification is successful
            
        Raises:
            ValueError: Raised when the model type is unknown
        """

        # Test Embedding model
        model_name = await self.model_service.get_display_name_by_id(model_id)
        if model_type == ModelCategory.TXT_EMBEDDING.value:
            input_texts = ["test"]
            result = await self.model_client.call_embedding_model(
                model_name, 
                input_texts
            )
            
        # Test LLM model
        elif model_type == ModelCategory.LLM.value:
            input_text = "test"
            async for chunk in self.model_client.call_llm_model(
                model_name,
                input_text,
                stream=False,
                max_tokens=16
            ):
                result = chunk

        # Test Reranker model
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

        # Test VLM model
        elif model_type == ModelCategory.VLM.value:
            prompt_unique_name = "KBOT1/pdf_parsing"
            # Simplest code to create a solid color image
            image = Image.new('RGB', (100, 100), 'lightblue')
            result = await self.model_client.get_vlm_answer(
                model_name, 
                image,
                prompt="Describe this image"
            )
            
        else:
            raise ParamValueError(f"Unknown model type: {model_type}")
        
        if result:
            return True
        else:
            raise InternalServerError(message=f"Model {model_id} verification failed")
        
model_controller = ModelController()      