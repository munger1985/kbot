from loguru import logger
from typing import Any
from core.config.settings import get_prompt_config
from core.dictionary import ChunkType
from dao.repositories import KBRepository, PromptRepository, TxtChunkRepository
from utils.clients.model_client import AIModelClient
from core.database.oracle import get_session
from services.ai_model import AIModelService
from core.exceptions import NotFoundError, handle_exception, InternalServerError, ParamValueError

class ChunkService:
    """Knowledge Base Chunk Operation Handler.
    
    Manages core operations for knowledge base text chunks, including editing,
    deletion, retrieval, description update, and tag management. Integrates with
    embedding models to regenerate vectors when chunk content/metadata changes.
    """
    
    def __init__(self) -> None:
        self.model_client = AIModelClient()
        self.model_service = AIModelService()

    @property
    def oracle_session(self):
        """Get async Oracle database session context manager.
        
        Returns:
            AsyncContextManager[AsyncSession]: Async database session manager
        """
        return get_session()

    async def edit_file_chunk(self, chunk_id: str, file_id: str, kb_id: int, new_chunk: str) -> bool:
        """Edit content of a specific text chunk and regenerate embedding vector.
        
        Updates chunk content in database and re-calculates embedding vector using
        the knowledge base's configured embedding model.
        
        Args:
            chunk_id: Unique ID of the target chunk
            file_id: ID of the associated file
            kb_id: ID of the knowledge base
            new_chunk: New content for the text chunk
            
        Returns:
            bool: True if update succeeds (always True if no exception raised)
            
        Raises:
            NotFoundError: If knowledge base has no embedding model configured
            InternalServerError: If chunk update or embedding generation fails
        """
        async with self.oracle_session as session:
            kb_repo = KBRepository(session)
            chunk_repo = TxtChunkRepository(session)

            # Get embedding model configured for the knowledge base
            kb = await kb_repo.get_by_id(kb_id)
            model_id = kb.txt_embed_model_id
            
            if not model_id:
                error_msg = f"Knowledge base {kb_id} has no embedding model configured - cannot update chunk"
                logger.error(error_msg)
                raise NotFoundError(error_msg)
                
            embed_model = await self.model_service.get_model_name_by_id(model_id)

            # Generate new embedding vector for updated chunk content
            try:
                response_data = await self.model_client.call_embedding_model(embed_model, [new_chunk])
                logger.info(f"Successfully generated embedding for chunk {chunk_id}")
                embeddings = [item.embedding for item in response_data]
            except Exception as e:
                error_msg = f"Failed to generate embedding for chunk {chunk_id}: {str(e)}"
                logger.error(error_msg)
                raise InternalServerError(error_msg) from e

            # Update chunk content and embedding in database
            try:
                await chunk_repo.update_chunk(chunk_id=chunk_id, new_content=new_chunk, new_embedding=embeddings[0])
                logger.info(f"Successfully updated chunk {chunk_id} for file {file_id} (KB: {kb_id})")
                return True
            except Exception as e:
                error_msg = f"Failed to update chunk {chunk_id} for file {file_id}: {str(e)}"
                logger.error(error_msg)
                raise InternalServerError(error_msg) from e

    async def delete_file_chunk(self, chunk_id: str, file_id: str, kb_id: int) -> bool:
        """Delete a specific text chunk from the knowledge base.
        
        Args:
            chunk_id: Unique ID of the target chunk
            file_id: ID of the associated file
            kb_id: ID of the knowledge base
            
        Returns:
            bool: True if deletion succeeds (always True if no exception raised)
            
        Raises:
            InternalServerError: If chunk deletion fails (wrapped by handle_exception)
        """
        async with self.oracle_session as session:
            # kb_repo = KBRepository(session)  # Unused - removed to reduce overhead
            chunk_repo = TxtChunkRepository(session)

            try:
                await chunk_repo.delete(chunk_id)
                logger.info(f"Successfully deleted chunk {chunk_id} for file {file_id} (KB: {kb_id})")
                return True
            except Exception as e:
                error_msg = f"Failed to delete chunk {chunk_id} for file {file_id}: {str(e)}"
                handle_exception(e, error_msg)
                raise InternalServerError(error_msg) from e

    async def get_chunks_by_file_id(self, file_id: str) -> list[dict[str, Any]]:
        """Retrieve all text chunks for a specific file.
        
        Args:
            file_id: ID of the target file
            
        Returns:
            List[Dict[str, Any]]: List of chunk dictionaries with core metadata
                                  (excludes embedding vector for performance)
        """
        async with self.oracle_session as session:
            chunk_repo = TxtChunkRepository(session)
            chunks = await chunk_repo.get_by_file_id(file_id)
            
            # Convert ORM objects to serializable dictionaries
            chunk_list = [
                {
                    "chunk_id": chunk.chunk_id,
                    "kb_id": chunk.kb_id,
                    "file_id": chunk.file_id,
                    "content": chunk.content,
                    "structure_level": chunk.structure_level,
                    "path_names": chunk.path_names,
                    "chunk_type": chunk.chunk_type,
                    "chunk_metadata": chunk.chunk_metadata,
                    "security_level": chunk.security_level,
                    "is_active": chunk.is_active,
                    "biz_metadata": chunk.biz_metadata
                }
                for chunk in chunks
            ]
            
            logger.info(f"Retrieved {len(chunk_list)} chunks for file {file_id}")
            return chunk_list

    async def update_chunk_description(self, chunk_id: str, kb_id: int, description: str) -> bool:
        """Update chunk description and regenerate embedding with combined content+description.
        
        Combines the original chunk content with new description to generate a new
        embedding vector, then updates the chunk's description and embedding.
        
        Args:
            chunk_id: Unique ID of the target chunk
            kb_id: ID of the knowledge base
            description: New description to add to the chunk
            
        Returns:
            bool: True if update succeeds, False if embedding generation fails
            
        Raises:
            ParamValueError: If knowledge base has no embedding model configured
            InternalServerError: If chunk update fails (wrapped by handle_exception)
        """
        async with self.oracle_session as session:
            chunk_repo = TxtChunkRepository(session)
            kb_repo = KBRepository(session)

            try:
                # Validate knowledge base embedding model configuration
                kb = await kb_repo.get_by_id(kb_id)
                embed_model_id = kb.txt_embed_model_id
                
                if not embed_model_id:
                    error_msg = f"Knowledge base {kb_id} has no embedding model configured - cannot update chunk description"
                    logger.error(error_msg)
                    raise ParamValueError(error_msg)
                
                embed_model = await self.model_service.get_model_name_by_id(embed_model_id)

                # Step 1: Get original chunk content
                content = await chunk_repo.get_content(chunk_id)
                if not content:
                    error_msg = f"Chunk {chunk_id} not found or has empty content"
                    logger.error(error_msg)
                    raise NotFoundError(error_msg)

                # Step 2: Combine description with original content
                content_with_desc = f"Text description: {description}\nOriginal content: {content}"

                # Step 3: Generate new embedding for combined content
                response_data = await self.model_client.call_embedding_model(embed_model, [content_with_desc])
                if not response_data:
                    error_msg = f"Failed to generate embedding for chunk {chunk_id} with new description"
                    logger.error(error_msg)
                    return False
                
                logger.info(f"Successfully generated embedding for chunk {chunk_id} with new description")
                embeddings = [item.embedding for item in response_data]

                # Step 4: Update chunk description and embedding
                await chunk_repo.update_description(chunk_id=chunk_id, description=description, new_embedding=embeddings[0])
                logger.info(f"Successfully updated description for chunk {chunk_id} (KB: {kb_id})")
                return True

            except Exception as e:
                error_msg = f"Failed to update description for chunk {chunk_id}: {str(e)}"
                handle_exception(e, error_msg)
                raise InternalServerError(error_msg) from e
            
    async def toggle_chunk_active_status(self, chunk_id: str, is_active: bool) -> None:
        """Toggle active status of a specific text chunk.
        
        Args:
            chunk_id: Unique ID of the target chunk
            kb_id: ID of the knowledge base
            is_active: New active status to set (True/False)
            
        Returns:
            None
            
        Raises:
            InternalServerError: If chunk update fails (wrapped by handle_exception)
        """
        async with self.oracle_session as session:
            chunk_repo = TxtChunkRepository(session)
            try:
                await chunk_repo.toggle_active_status(chunk_id, is_active)
                logger.info(f"Successfully toggled active status for chunk {chunk_id}")
            except Exception as e:
                error_msg = f"Failed to toggle active status for chunk {chunk_id}: {str(e)}"
                handle_exception(e, error_msg)
