from loguru import logger
from dao.repositories import AIModelRepository
from core.database.oracle import get_session
from core.exceptions import *

class AIModelService:
    def __init__(self):
        pass

    @property
    def oracle_session(self):
        return get_session()
    
    async def get_display_name_by_id(self, model_id: int) -> str:
        try:
            async with self.oracle_session as session:
                repo = AIModelRepository(session)
                model = await repo.get_by_id(model_id)
                return model.display_name
        except Exception as e:
            logger.error(f"Failed to get model by model id: {model_id}", exc_info=e)
            raise e
        
    async def get_name_and_max_token_by_display_name(self, display_name: str) -> tuple[str, int]:
        try:
            async with self.oracle_session as session:
                repo = AIModelRepository(session)
                
                model = await repo.get_by_display_name(display_name)
                if not model.model_params:
                    max_tokens = 4096
                else:
                    max_tokens = model.model_params.get("max_tokens", 4096)
                return model.model_name, max_tokens
        except Exception as e:
            logger.error(f"Failed to get model by display name: {display_name}", exc_info=e)
            raise e
        
    async def get_embedding_batch_size(self, embedding_model_name: str) -> int | None:
        """Get batch size params for embedding model."""
        async with self.oracle_session as session:
            repo = AIModelRepository(session)
            try:
                embed_model = await repo.get_by_name(embedding_model_name)
                if not embed_model:
                    raise NotFoundError(f"Model {embedding_model_name} not found.")
                params = embed_model.model_params
                if not params:
                    raise NotFoundError(f"Model {embedding_model_name} params not found.")
                batch_size = params.get("batch_size", None)
                # Ensure batch_size is an integer if it exists
                if batch_size is not None:
                    batch_size = int(batch_size)
                return batch_size
            except Exception as e:
                handle_exception(e, f"Retrieve failed when getting batch size for embedding model {embedding_model_name}：{e}")