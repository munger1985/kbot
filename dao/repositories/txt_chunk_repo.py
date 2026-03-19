import math
from loguru import logger
from typing import List, Dict, Any, Optional, Sequence
from sqlalchemy import select, update, delete, and_, or_, func, text, literal_column, Boolean
from dao.entities import TxtChunkEntity
from core.exceptions import DatabaseException, DataNotFoundException
from .base_repo import BaseRepository
from utils.oracle_vec_handler import OracleVecHandler
from utils.common import safe_read_content


class TxtChunkRepository(BaseRepository[TxtChunkEntity]):
    """Oracle SQLAlchemy 2.0 implementation for TxtChunkEntity operations
    Fully compatible with Elasticsearch version interface
    """
    

    async def create(self, chunks: List[TxtChunkEntity]):
        """
        Batch create text chunk records
        Adapt to structured fields: path_names, structure_level, chunk_type

        :param chunks: List of TxtChunkEntity instances to create
        """
        if not chunks:
            logger.warning("Empty chunk list provided for creation, skipping execution")
            return

        try:
            # Convert embeddings to Oracle-compatible format
            vec_handler = OracleVecHandler()

            # Batch processing (Oracle bulk insert best practice: 100 records per batch)
            batch_size = 100
            total_success = 0

            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Convert embeddings to Oracle array format
                for chunk in batch_chunks:
                    chunk.embedding = vec_handler.convert(chunk.embedding)
                    logger.debug(f"Converted embedding for chunk {chunk.chunk_id}, type: {type(chunk.embedding)}")

                # Add batch entities to session
                self.session.add_all(batch_chunks)
                await self.session.flush()  # Execute insert without commit for better performance

                total_success += len(batch_chunks)
                logger.info(
                    f"Successfully batch inserted {len(batch_chunks)} text chunk records to Oracle, "
                    f"progress: {total_success}/{len(chunks)}"
                )

            logger.info(f"Completed all batch insertions, total successful records: {total_success}")

        except Exception as e:
            logger.error(f"Oracle batch insert text chunks failed: {str(e)}", exc_info=True)
            raise DatabaseException("Oracle batch insert text chunks failed", original_error=e)
        
    async def vector_search(
        self,
        kb_id: int,
        query_vec: List[float],
        security: int,
        similarity_threshold: float = 0.5,
        search_top_k: int = 10,
        tags: List[str] = [],
        path_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Vector similarity search (Oracle version)
        Note: Oracle 21c+ or vector extensions required for vector operations
        
        :param query_vec: Query embedding vector
        :param security: Security level filter (records with level <= security will be returned)
        :param similarity_threshold: Minimum similarity score threshold (0.0-1.0)
        :param search_top_k: Maximum number of results to return
        :param tags: List of tags to filter results
        :param path_filter: Path name filter (match any in path_names array)
        :return: List of search results with similarity scores
        """
        try:
            # 1. Build base filter conditions
            filter_conditions = [
                TxtChunkEntity.kb_id == kb_id,
                TxtChunkEntity.is_active == 1,
                TxtChunkEntity.security_level <= security
            ]

            # 2. Add path filter
            if path_filter:
                # Use text() to directly specify the SQL expression with bind parameters
                filter_conditions.append(text("JSON_EXISTS(path_names, '$[*]?(@ == :path)').bindparams(path=path_filter)"))

            # 3. Add tag filter
            if tags:
                tag_conditions = []
                for tag in tags:
                    tag_conditions.append(text(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == :tag_{i}]')").bindparams(**{f"tag_{i}": tag}))
                if tag_conditions:
                    filter_conditions.append(or_(*tag_conditions))

            # 4. Build vector similarity query
            # Oracle vector cosine similarity calculation: DOT_PRODUCT / (NORM(vec1) * NORM(vec2))
            oracle_embedding = OracleVecHandler().convert(vec=query_vec)
            similarity_expr = text(f"""
                UTL_VECTOR.DOT_PRODUCT(
                    {oracle_embedding},
                    embedding
                ) / (
                    UTL_VECTOR.NORM({oracle_embedding}) *
                    UTL_VECTOR.NORM(embedding)
                )
            """)

            # 5. Execute query
            stmt = (
                select(
                    TxtChunkEntity.chunk_id,
                    TxtChunkEntity.file_id,
                    TxtChunkEntity.content,
                    TxtChunkEntity.path_names,
                    TxtChunkEntity.structure_level,
                    TxtChunkEntity.chunk_metadata,
                    similarity_expr.label("similarity_score")
                )
                .where(and_(*filter_conditions))
                .where(text("similarity_score >= :threshold"))
                .order_by(text("similarity_score DESC"))
                .limit(search_top_k)
            )
            
            # Bind parameters separately
            result = await self.session.execute(stmt, {"threshold": similarity_threshold})
            chunks = result.fetchall()

            # 6. Format results
            results = []
            for chunk in chunks:
                path_names = chunk.path_names or []
                results.append({
                    "id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "content": chunk.content,
                    "path": " > ".join(path_names),
                    "structure_level": chunk.structure_level,
                    "metadata": chunk.chunk_metadata,
                    "score": float(chunk.similarity_score) if chunk.similarity_score else 0.0
                })

            return results

        except Exception as e:
            raise DatabaseException("Exception occurred during vector search execution", original_error=e)

    async def full_text_search(
        self,
        kb_id: int,
        keyword: str,
        security: int,
        search_top_k: int = 10,
        tags: List[str] = [],
        path_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Full text search (Oracle version)
        Uses Oracle TEXT full-text index or LIKE pattern matching
        
        :param keyword: Search keyword/phrase
        :param security: Security level filter (records with level <= security will be returned)
        :param search_top_k: Maximum number of results to return
        :param tags: List of tags to filter results
        :param path_filter: Path name filter (match any in path_names array)
        :return: List of search results with simulated scores
        """
        try:
            # 1. Build base filter conditions
            filter_conditions = [
                TxtChunkEntity.kb_id == kb_id,
                TxtChunkEntity.is_active == 1,
                TxtChunkEntity.security_level <= security
            ]

            # 2. Add path filter
            if path_filter:
                # Use text() with bind parameter
                filter_conditions.append(text("JSON_EXISTS(path_names, '$[*]?(@ == :path_filter)').bindparams(path_filter=path_filter)"))

            # 3. Add tag filter
            if tags:
                tag_conditions = []
                for i, tag in enumerate(tags):
                    tag_conditions.append(text(f"JSON_EXISTS(biz_metadata, '$.tags[*]?(@ == :tag_{i}]').bindparams(**{{f'tag_{i}': tag}})"))
                
                if tag_conditions:
                    filter_conditions.append(or_(*tag_conditions))

            # 4. Add keyword search
            if keyword.strip():
                # Use text() to specify the SQL expression
                filter_conditions.append(text("CONTAINS(content, :keyword) > 0"))

            # 5. Execute query
            stmt = (
                select(
                    TxtChunkEntity.chunk_id,
                    TxtChunkEntity.file_id,
                    TxtChunkEntity.content,
                    TxtChunkEntity.path_names,
                    TxtChunkEntity.structure_level,
                    TxtChunkEntity.chunk_metadata
                )
                .where(and_(*filter_conditions))
                .order_by(TxtChunkEntity.structure_level)
                .limit(search_top_k)
            )

            result = await self.session.execute(stmt)
            chunks = result.fetchall()

            # 6. Format results
            results = []
            for idx, chunk in enumerate(chunks):
                path_names = chunk.path_names or []
                results.append({
                    "id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "content": chunk.content,
                    "path_names": path_names,
                    "full_path": " / ".join(path_names),
                    "structure_level": chunk.structure_level,
                    "metadata": chunk.chunk_metadata,
                    "score": (search_top_k - idx) / search_top_k  # Simulated relevance score
                })

            return results

        except Exception as e:
            logger.error(f"Oracle full text search failed: {str(e)}")
            raise DatabaseException("Oracle full text search execution failed", original_error=e)

    async def delete_by_file_ids(self, file_ids: List[str]):
        """
        Delete text chunk records by file IDs
        
        :param kb_id: Knowledge base unique identifier
        :param file_ids: List of file IDs to delete chunks for
        :return: Number of deleted records
        :raises DataNotFoundException: If no records found for the given file IDs
        """
        try:
            # Build delete statement
            stmt = (
                delete(TxtChunkEntity)
                .where(TxtChunkEntity.file_id.in_(file_ids))
            )

            # Execute deletion and return affected rows
            await self.session.execute(stmt)
            logger.info(f"Successfully deleted text chunk records, file IDs: {file_ids[:5]}...")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Oracle delete text chunks by file IDs failed: {str(e)}")
            raise DatabaseException("Oracle delete text chunks by file IDs failed", original_error=e)

    async def delete_by_kb_id(self, kb_id: int):
        """
        Delete all text chunk records by knowledge base ID
        
        :param kb_id: Knowledge base unique identifier
        :return: Number of deleted records
        :raises DataNotFoundException: If no records found for the given KB ID
        """
        try:
            # Build delete statement
            stmt = delete(TxtChunkEntity).where(TxtChunkEntity.kb_id == kb_id)
            
            # Execute deletion
            await self.session.execute(stmt)            
            logger.info(f"Successfully deleted text chunk records for KB ID: {kb_id}")

        except Exception as e:
            logger.error(f"Oracle delete text chunks by KB ID failed: {str(e)}")
            raise DatabaseException("Oracle delete text chunks by KB ID failed", original_error=e)

    async def get_by_file_id(self, file_id: str) -> Sequence[TxtChunkEntity]:
        """
        Get all text chunks for a file (exclude embedding to save memory)
        
        :param file_id: File unique identifier
        :return: List of TxtChunkEntity instances (without embedding data)
        """
        try:
            # Build query
            stmt = (
                select(
                    TxtChunkEntity
                )
                .where(TxtChunkEntity.file_id == file_id)
                .order_by(
                    text("JSON_VALUE(chunk_metadata, '$.chunk_num') ASC"),
                    text("JSON_VALUE(chunk_metadata, '$.sub_index') ASC")
                )
            )

            result = await self.session.execute(stmt)
            return result.scalars().all()

        except Exception as e:
            logger.error(f"Oracle get file chunks failed for file ID {file_id}: {str(e)}")
            raise DatabaseException(f"Oracle get text chunks for file {file_id} failed", original_error=e)

    async def update_chunk(self, chunk_id: str, new_content: str, new_embedding: list[float]) -> bool:
        """
        Update text chunk content by ID
        
        :param chunk_id: Text chunk unique identifier
        :param new_content: New content for the text chunk
        :return: True if update successful
        :raises DataNotFoundException: If chunk ID not found
        """
        oracle_embedding = OracleVecHandler().convert(new_embedding)
        try:
            stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(content=new_content)
                .values(embedding=oracle_embedding)
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            rowcount = await self.session.execute(stmt)
            
            if rowcount == 0:
                raise DataNotFoundException(f"Chunk ID not found: {chunk_id}")

            logger.info(f"Successfully updated content for chunk {chunk_id}")
            return True

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Oracle update chunk failed: {str(e)}")
            raise DatabaseException("Oracle update text chunk content failed", original_error=e)

    async def delete(self, chunk_id: str):
        """
        Delete text chunk record by ID
        
        :param chunk_id: Text chunk unique identifier
        :return: True if deletion successful
        :raises DataNotFoundException: If chunk ID not found
        """
        try:
            stmt = delete(TxtChunkEntity).where(TxtChunkEntity.chunk_id == chunk_id)
            await self.session.execute(stmt)
            logger.info(f"Successfully deleted chunk {chunk_id}")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Oracle delete chunk failed: {str(e)}")
            raise DatabaseException("Oracle delete text chunk failed", original_error=e)
        
    async def get_content(self, chunk_id: str) -> str:
        """
        Get text chunk content by ID
        
        :param chunk_id: Text chunk unique identifier
        :return: Text chunk content
        :raises DataNotFoundException: If chunk ID not found
        """
        try:
            stmt = select(TxtChunkEntity.content).where(TxtChunkEntity.chunk_id == chunk_id)
            result = await self.session.execute(stmt)
            content = result.scalar_one()
            if content is None:
                raise DataNotFoundException(f"Chunk ID not found: {chunk_id}")
            return safe_read_content(content)
        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Oracle get chunk content failed: {str(e)}")
            raise DatabaseException("Oracle get text chunk content failed", original_error=e)

    async def update_tag(self, file_id: str, tags: list[str]):
        """
        Update tags for text chunks by file ID
        
        :param file_id: File unique identifier
        :param tags: List of tags to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.file_id == file_id)
                .values(
                    biz_metadata=func.json_mergepatch(
                        func.nvl(TxtChunkEntity.biz_metadata, func.json_object()),  # Handle null biz_metadata
                        func.json_object('tags', tags)  # Create JSON object with tags array
                    )
                )
                .execution_options(synchronize_session="fetch")
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            # Execute update using SQLAlchemy ORM
            await self.session.execute(update_stmt)
            logger.info(f"Successfully updated tags for file {file_id}: {tags}")

        except Exception as e:
            logger.error(f"Oracle update tags failed: {str(e)}")
            raise DatabaseException("Oracle update text chunk tags failed", original_error=e)
          
    async def update_description(self, chunk_id: str, description: str, new_embedding: list[float]):
        """
        Update description for text chunk by chunk ID
        
        :param chunk_id: Chunk unique identifier
        :param description: Chunk description to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            oracle_embedding = OracleVecHandler().convert(new_embedding)
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(
                    biz_metadata=func.json_mergepatch(
                        func.nvl(TxtChunkEntity.biz_metadata, func.json_object()),  # Handle null biz_metadata
                        func.json_object('description', description)  # Create JSON object with description
                    )
                )
                .values(embedding=oracle_embedding)
                .execution_options(synchronize_session="fetch")
                .returning(func.count(TxtChunkEntity.chunk_id))
            )

            # Execute update using SQLAlchemy ORM
            updated_count = await self.session.execute(update_stmt)
            
            if updated_count == 0:
                raise DataNotFoundException(f"No records found for chunk ID: {chunk_id}")

            logger.info(f"Successfully updated description for chunk {chunk_id}, affected {updated_count} records")

        except DataNotFoundException as e:
            raise e
        except Exception as e:
            logger.error(f"Oracle update description failed: {str(e)}")
            raise DatabaseException("Oracle update text chunk description failed", original_error=e)
        
    async def toggle_active_status(self, chunk_id: str, is_active: bool):
        """
        Toggle active status for text chunk by chunk ID
        
        :param chunk_id: Chunk unique identifier
        :param is_active: Active status to update
        :raises DataNotFoundException: If no records found for the given file ID
        """
        try:
            # Build update statement using pure SQLAlchemy 2.0 ORM syntax
            # Oracle JSON_MERGEPATCH equivalent in SQLAlchemy
            update_stmt = (
                update(TxtChunkEntity)
                .where(TxtChunkEntity.chunk_id == chunk_id)
                .values(is_active=is_active)
            )
            
            # Execute update using SQLAlchemy ORM
            await self.session.execute(update_stmt)
            logger.info(f"Successfully updated active status for chunk {chunk_id}, is_active: {is_active}")
        
        except Exception as e:
            logger.error(f"Oracle update active status failed: {str(e)}")
            raise DatabaseException("Oracle update text chunk active status failed", original_error=e)