import json
from loguru import logger
from sqlalchemy import select, update, delete, func, text, desc
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import DatabaseException
from .base_repo import BaseRepository
from utils.oracle_vec_handler import OracleVecHandler
from dao.entities import MemoryEntryEntity, ConversationContextEntity


class MemoryEntryRepository(BaseRepository[MemoryEntryEntity]):
    """
    User conversation record repository - responsible for physical maintenance and data access of chat record entries.
    """
    async def get_context_by_id(self, session_id: str) -> ConversationContextEntity | None:
        """Get conversation context by session id"""
        try:
            stmt = select(ConversationContextEntity).where(
                ConversationContextEntity.session_id == session_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get context by id: {session_id}", exc_info=e)
            raise DatabaseException("Failed to get conversation context", original_error=e)

    async def update_context_state(self, session_id: str, new_state: dict, increment_count: bool = True):
        """Update conversation context state"""
        try:
            context = await self.get_context_by_id(session_id)
            if context:
                context.session_state = new_state
                context.last_active_at = func.now()
                if increment_count:
                    context.interaction_count += 1
        except Exception as e:
            logger.error(f"Failed to update context state for session {session_id}", exc_info=e)
            raise DatabaseException("Failed to update context state", original_error=e)

    async def add_memory_entry(self, entry: MemoryEntryEntity):
        """Persist memory entry to storage"""
        try:
            self.session.add(entry)
        except Exception as e:
            logger.error(f"Failed to add memory entry", exc_info=e)
            raise DatabaseException("Failed to add memory entry", original_error=e)
        
    async def get_recent_entries(self, session_id: str, limit: int = 10) -> list[MemoryEntryEntity]:
        """
        获取最近 N 轮的对话历史，用于生成摘要
        过滤负反馈，优先展示正反馈
        逻辑：
        1. feedback = -1: 过滤掉（用户点踩，表示回答质量差，不作为参考）
        2. feedback = 1: 优先（用户点赞，高质量参考）
        3. feedback = 0: 正常包含（默认状态）
        """
        stmt = (
        select(MemoryEntryEntity)
        .where(
            MemoryEntryEntity.session_id == session_id,
            MemoryEntryEntity.feedback >= 0  # 核心过滤：排除 -1
            )
            .order_by(
                desc(MemoryEntryEntity.feedback),     # 1 优先于 0
                desc(MemoryEntryEntity.request_time)  # 同等反馈下，按时间倒序
            )
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        # 返回按时间正序排列的记录
        return list(result.scalars().all())

    async def update_context_summary(self, session_id: str, new_summary: str):
        """更新会话的滚动摘要"""
        context = await self.get_context_by_id(session_id)
        if context:
            context.context_summary = new_summary

    async def upsert_user_profile(self, user_id: str, profile_updates: dict):
        """
        更新用户画像
        """
        # 1. 转换字典为 JSON 字符串
        updates_json = json.dumps(profile_updates, ensure_ascii=False)
        
        # 2. 使用 text() 包装 SQL 语句
        # 注意：Oracle 的 MERGE 语句在 SQLAlchemy 中必须用 text()
        sql = text("""
            MERGE INTO KBOT_MD_USER_PROFILE p
            USING (SELECT :uid as user_id FROM dual) s
            ON (p.user_id = s.user_id)
            WHEN MATCHED THEN
                UPDATE SET 
                    profile_data = JSON_MERGEPATCH(profile_data, :updates),
                    update_time = CURRENT_TIMESTAMP
            WHEN NOT MATCHED THEN
                INSERT (user_id, profile_data, update_time, create_time)
                VALUES (:uid, :updates, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """)
        
        # 3. 传入参数字典
        await self.session.execute(sql, {
            "uid": user_id, 
            "updates": updates_json
        })

    async def search_vector_memory(self, user_id: str, query_vector: list[float], limit: int = 3):
        """
        在 Oracle 23ai 中执行原生向量搜索
        """
        # 使用 Oracle 23ai 的原生向量距离函数
        # 假设使用的是 Cosine 距离
        sql = text("""
            SELECT entry_id, raw_question, answer, 
                   VECTOR_DISTANCE(memory_vector, :vec, COSINE) as distance
            FROM kbot_md_memory_entry m
            JOIN kbot_md_conv_context c ON m.session_id = c.session_id
            WHERE c.user_id = :user_id
            ORDER BY distance
            FETCH FIRST :limit ROWS ONLY
        """)
        
        # 将 list[float] 转换为 Oracle 向量格式（通常是 array.array）
        vec_handler = OracleVecHandler()
        oracle_vec = vec_handler.convert(query_vector)
        try:
            result = await self.session.execute(
                sql, {"vec": oracle_vec, "user_id": user_id, "limit": limit}
            )
            return result.fetchall()
        except Exception as e:
            logger.error(f"Failed to search vector memory", exc_info=e)
            raise DatabaseException("Failed to search vector memory", original_error=e)
        
    async def update_feedback(self, entry_id: int, score: int):
        """
        更新特定交互的反馈分数
        score: -1 (差), 0 (无), 1 (好)
        """
        stmt = (
            update(MemoryEntryEntity)
            .where(MemoryEntryEntity.entry_id == entry_id)
            .values(feedback=score)
        )
        await self.session.execute(stmt)