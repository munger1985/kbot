import os
import sys
from loguru import logger
from PIL import Image

# 添加项目根目录到 Python 路径，确保可以导入项目模块
current_file = os.path.abspath(__file__)
backend_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)
    
from microservices.vlm.model_pool import ModelPool
from models.vlm.base import BaseVLM


class VLMService:
    """
    统一的VLM服务，用于管理和使用不同的VLM模型
    """
    
    def __init__(self):
        """
        Initialize VLM service // 初始化VLM服务
        """
        self._model_pool = ModelPool()
        self._initialized = False
        
    async def initialize(self):
        """
        Initialize VLM service and model pool // 初始化VLM服务和模型池
        """
        if not self._initialized:
            await self._model_pool.initialize()
            self._initialized = True
            logger.info("VLM service initialized")
        
    async def shutdown(self):
        """
        Shutdown VLM service and all models // 关闭VLM服务和所有模型
        """
        if self._initialized:
            await self._model_pool.shutdown()
            self._initialized = False
            logger.info("VLM service has been shutdown")
    
    async def get_vlm_model(self, model_unique_name: str) -> BaseVLM:
        """
        Get a VLM model by unique name // 获取指定唯一名的VLM模型

        Args:
            model_unique_name: The unique name of the model to get // 要获取的模型ID

        Returns:
            VLM model instance // VLM模型实例

        Raises:
            ValueError: If model_unique_name is not found in database // 如果模型ID在数据库中不存在
            RuntimeError: If model creation fails // 如果模型创建失败
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.load_model(model_unique_name)
    
    async def unload_model(self, model_unique_name: str):
        """
        从模型池中卸载模型

        Args:
            model_unique_name: 要卸载的模型ID
        """
        if self._initialized:
            await self._model_pool.unload_model(model_unique_name)
            logger.info(f"Model {model_unique_name} has been unloaded.")
    
    async def reload_model(self, model_unique_name: str) -> BaseVLM:
        """
        Reload a model from the pool // 重新加载模型

        Args:
            model_unique_name: The unique name of the model to reload // 要重新加载的模型ID

        Returns:
            The reloaded VLM model instance // 重新加载的VLM模型实例
        """
        if not self._initialized:
            await self.initialize()
        
        return await self._model_pool.reload_model(model_unique_name)
    
    async def inference(self, 
                        model_unique_name: str, 
                        text: str, 
                        image: str | Image.Image, 
                        **kwargs) -> str:
        """
        Generate response for a given query using a loaded model. //使用VLM模型生成回复
        
        Args:
            model_unique_name: unique name of the model to use. //模型唯一名称
            text: input text query. //输入查询文本
            image: input image. //输入图像，可以是文件路径或 PIL Image 对象
            **kwargs: optional keyword arguments for model. //可选参数
            
        Returns:
            The prediction result. //返回结果
        """        
        if not text:
            raise ValueError("Text must not be None.")
        if not image:
            raise ValueError("Image must not be None.")
        
        try:
            vlm = await self.get_vlm_model(model_unique_name)
            return await vlm.inference(text=text, image=image, **kwargs)

        except Exception as e:
            logger.error(f"VLM inference failed: {e}")
            raise RuntimeError("VLM inference failed") from e
