from typing import Dict, Union, Optional
from loguru import logger
from .base import BaseEmbedding, LocalEmbeddingConfig, CloudEmbeddingConfig
from .local import LocalEmbedding
from .cloud import CloudEmbedding

class EmbeddingProvider:
    """Factory class for text embedding models"""
    
    def __init__(self):
        self.models: Dict[str, BaseEmbedding] = {}
    
    async def initialize(self, config: Union[LocalEmbeddingConfig, CloudEmbeddingConfig]) -> None:
        """Initialize a new embedding model based on the provided configuration"""
        model_name = config.model_name
        
        # Check if model already exists
        if model_name in self.models:
            logger.warning(f"Model {model_name} already initialized")
            return
        
        # Create the appropriate model based on config type
        if isinstance(config, LocalEmbeddingConfig):
            logger.info(f"Initializing local embedding model: {model_name}")
            model = LocalEmbedding(config)
        elif isinstance(config, CloudEmbeddingConfig):
            logger.info(f"Initializing cloud embedding model: {model_name}")
            model = CloudEmbedding(config)
        else:
            raise ValueError(f"Unsupported embedding config type: {type(config)}")
        
        # Start the model
        await model.startup()
        
        # Add to models dictionary
        self.models[model_name] = model
        logger.info(f"Model {model_name} initialized successfully")
    
    def get_model(self, model_name: str) -> Optional[BaseEmbedding]:
        """Get a model by name"""
        return self.models.get(model_name)
    
    async def close_model(self, model_name: str) -> None:
        """Shutdown and remove a model"""
        if model_name in self.models:
            model = self.models[model_name]
            await model.shutdown()
            del self.models[model_name]
            logger.info(f"Model {model_name} closed")
    
    async def close_all(self) -> None:
        """Shutdown and remove all models"""
        for model_name in list(self.models.keys()):
            await self.close_model(model_name)