from loguru import logger
from typing import Any, Sequence
from datetime import datetime

from sqlalchemy import select, update, delete, func, text
from sqlalchemy.types import ARRAY, Float


from core.exceptions import DatabaseException
from .base_repo import BaseRepository
from dao.entities import ChatRecordEntity
from utils.oracle_vec_handler import OracleVecHandler


class ChatRecordRepository(BaseRepository[ChatRecordEntity]):
    """
    User conversation record repository - responsible for physical maintenance and data access of chat record entries.
    """

    async def add_record(self, chat_record: ChatRecordEntity):
        """Inserts a single chat record.

        Args:
            doc: Chat record document dictionary containing session_id, question, answer, etc.

        Returns:
            Created record ID.
        """
        try:
            # Insert the record into the database.
            self.session.add(chat_record)
            
        except Exception as e:
            raise DatabaseException("Failed to add chat record.", original_error=e)

    async def search_short_term(self, session_id: str, limit: int = 5) -> list[dict]:
        """Gets context for the current session in chronological order.

        Args:
            session_id: Target session ID.
            limit: Maximum number of records to return. Defaults to 5.

        Returns:
            list of chat records in ascending time order.
        """
        try:
            stmt = select(ChatRecordEntity).where(
                ChatRecordEntity.session_id == session_id
            ).order_by(
                ChatRecordEntity.created_at.desc()
            ).limit(limit)
            
            result = await self.session.execute(stmt)
            records = result.scalars().all()
            
            # Convert to dictionary and reverse to get ascending order.
            record_dicts = [self._entity_to_dict(record) for record in records]
            return list(reversed(record_dicts))
            
        except Exception as e:
            raise DatabaseException("Failed to search short-term chat records.", original_error=e)

    async def search_long_term(self, vector: list[float], exclude_session: str, limit: int = 2) -> list[dict]:
        """
        Performs cross-session semantic search using the Oracle 26ai VECTOR_DISTANCE function.

        Args:
            vector: Query vector (list of floats).
            exclude_session: Session ID to exclude.
            limit: Maximum number of results. Defaults to 2.

        Returns:
            list of semantically similar records sorted by cosine similarity.
        """
        try:
            # 1. Convert the query vector to Oracle VECTOR format string.
            oracle_query_vector = OracleVecHandler().convert(vec=vector, to_string=False)
            
            # 2. Oracle 26ai vector similarity query (cosine distance).
            # VECTOR_DISTANCE syntax: VECTOR_DISTANCE(vector1, vector2, 'COSINE')
            stmt = select(
                ChatRecordEntity,
                # Calculate cosine distance (smaller value means more similar).
                func.VECTOR_DISTANCE(
                    oracle_query_vector,                     # Query vector.
                    ChatRecordEntity.question_vector,         # Stored vector (vector in JSON field).
                    text("'COSINE'")                         # Distance type.
                ).label("distance")
            ).where(
                ChatRecordEntity.session_id != exclude_session
            ).order_by(
                # Sort by cosine distance in ascending order (most similar first).
                text("distance ASC")
            ).limit(limit)
            
            # Execute the query.
            result = await self.session.execute(stmt)
            rows = result.all()
            
            # Convert the result format.
            records = []
            for row in rows:
                record = row[0]  # ChatRecordEntity object.
                distance = row[1] # Cosine distance value.
                record_dict = self._entity_to_dict(record)
                record_dict["similarity_score"] = 1 - float(distance)  # Convert to similarity (1 - distance).
                records.append(record_dict)
            
            return records
            
        except Exception as e:
            raise DatabaseException(f"Oracle 26ai vector search failed: {str(e)}", original_error=e)

    async def get_session_history(self, session_id: str) -> list[dict]:
        """
        Gets the complete history of the specified session for frontend rendering.

        Args:
            session_id: Target session ID.

        Returns:
            list of session history records in ascending time order.
        """
        try:
            stmt = select(ChatRecordEntity).where(
                ChatRecordEntity.session_id == session_id
            ).order_by(
                ChatRecordEntity.created_at.asc()  # Ascending order for rendering.
            )
            
            result = await self.session.execute(stmt)
            rows = result.all()
            
            records = []
            for row in rows:
                record_dict = self._entity_to_dict(row[0])
                records.append(record_dict)
            return records

        except Exception as e:
            logger.error(f"Failed to get session history, session id: {session_id}: {e}")
            raise DatabaseException("Failed to get session history", original_error=e)

    async def delete_session_records(self, session_id: str) -> None:
        """Deletes all chat records for a specific session.

        Args:
            session_id: Session ID to delete.
        """
        try:
            stmt = delete(ChatRecordEntity).where(
                ChatRecordEntity.session_id == session_id
            )
            await self.session.execute(stmt)
            logger.info(f"Deleted all chat records for session {session_id}.")
            
        except Exception as e:
            raise DatabaseException("Failed to delete session chat records.", original_error=e)
        
    async def delete_by_ids(self, session_ids: list[str]):
        """
        Deletes chat records by session IDs.
        
        Args:
            session_ids: List of session IDs to delete.
        """
        try:
            stmt = delete(ChatRecordEntity).where(
                ChatRecordEntity.session_id.in_(session_ids)
            )
            await self.session.execute(stmt)
            logger.info(f"Deleted chat records for sessions: {', '.join(session_ids)}.")
            
        except Exception as e:
            raise DatabaseException("Failed to delete chat records by session IDs.", original_error=e)
    
    async def feedback(self, record_id: int, feedback: int):
        """
        Submits feedback for a chat record.

        Args:
            record_id: Record ID.
            feedback: Feedback result. 1: positive, 0: no feedback, -1: negative.
        """
        try:
            stmt = update(ChatRecordEntity).where(
                ChatRecordEntity.record_id == record_id
            ).values(
                feedback=feedback
            )
            await self.session.execute(stmt)
            logger.info(f"Submitted feedback for record {record_id} with result {feedback}.")
            
        except Exception as e:
            raise DatabaseException("Failed to submit feedback for chat record.", original_error=e)
        
    def _entity_to_dict(self, entity: ChatRecordEntity) -> dict:
        """Converts ORM entity to a dictionary.

        Args:
            entity: ChatRecordEntity instance.

        Returns:
            dictionary representation of the entity.
        """
        return {
            "record_id": entity.record_id,
            "session_id": entity.session_id,
            "question": entity.question,
            "answer": entity.answer,
            "question_vector": entity.question_vector,
            "references": entity.references,
            "feedback": entity.feedback,
            "request_time": entity.request_time.isoformat() if entity.request_time else None,
            "response_time": entity.response_time.isoformat() if entity.response_time else None,
            "created_at": entity.created_at.isoformat() if entity.created_at else None
        }