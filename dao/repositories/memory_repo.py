import json
from datetime import datetime, timezone
from typing import Any
from loguru import logger
from sqlalchemy import select, update, delete, func, text, desc, and_
from core.exceptions import DatabaseException
from .base_repo import BaseRepository
from utils.oracle_vec_handler import OracleVecHandler
from dao.entities import MemoryEntryEntity, ConversationContextEntity, UserProfileEntity, ConversationContextModel
from core.config.settings import get_app_config
from utils.common import safe_read_content


class MemoryRepository(BaseRepository[MemoryEntryEntity]):
    """
    User conversation record repository - responsible for physical maintenance and data access of chat record entries.
    """
    async def ensure_session(self, session_id: str, user_id: str, agent_id: int, question: str | None = None):
        """
        通用会话环境确保逻辑：
        只要 session_id 或 user_id 在 DB 中不存在，就自动初始化。
        """
        # 1. 处理用户 (UserProfile) - 保证 L1 级联可用
        user = await self.session.get(UserProfileEntity, user_id)
        if not user:
            logger.info(f"Initializing new user profile: {user_id}")
            user = UserProfileEntity(
                user_id=user_id,
                global_preferences={"confirmed": {}, "inferred": {}},
                frequent_entities={},
                correction_history=[],
                profile_summary="Automatically initialized user profile",
                last_update_time=datetime.now(timezone.utc)
            )
            self.session.add(user)
            await self.session.flush()

        # 2. 处理会话 (ConversationContext) - 保证 L2 级联可用
        ctx = await self.session.get(ConversationContextEntity, session_id)
        if not ctx:
            logger.info(f"Initializing new conversation context: {session_id}")
            session_title = question[:100] if question else f"New Chat Session"
            ctx = ConversationContextEntity(
                session_id=session_id,
                user_id=user_id,
                app_id=get_app_config().app_id,
                agent_id=agent_id,
                session_title=session_title,
                # 新增字段初始化
                session_state={},           # 槽位参数池
                current_plan=None,          # 初始无执行计划
                step_outputs={},            # 初始无 Skill 产物
                last_relevance_score=1.0,   # 初始相关性设为最高
                active_topic=None,          # 初始无活跃话题
                context_summary="对话刚刚开始",
                interaction_count=0
            )
            self.session.add(ctx)
    
    async def update_feedback(self, entry_id: str, score: int):
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
            .order_by(desc(MemoryEntryEntity.request_time))
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        # 返回按时间正序排列的记录
        entries = list(result.scalars().all())
        return entries[::-1]
    
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

    async def save_context(self, context: ConversationContextEntity):
        """
        持久化或更新会话状态 (Oracle UPSERT 实现)
        Args:
            context: 会话上下文实体对象 (ConversationContextEntity)
        """
        
        try:
            app_id = get_app_config().app_id
            
            # 2. 构建 Oracle 标准的 MERGE INTO 语句
            sql = text("""
                MERGE INTO kbot_md_conv_context ent
                USING (SELECT :sid as session_id FROM dual) src
                ON (ent.session_id = src.session_id)
                WHEN MATCHED THEN
                    UPDATE SET 
                        user_id             = :user_id,
                        agent_id            = :agent_id,
                        app_id              = :app_id,
                        session_title       = :session_title,
                        session_state       = :session_state,
                        current_plan        = :current_plan,
                        step_outputs        = :step_outputs,
                        context_summary     = :context_summary,
                        last_relevance_score= :last_relevance_score,
                        active_topic        = :active_topic,
                        interaction_count   = :interaction_count,
                        is_deleted          = :is_deleted,
                        last_active_at      = SYSTIMESTAMP
                WHEN NOT MATCHED THEN
                    INSERT (
                        session_id, user_id, agent_id, app_id, session_title,
                        session_state, current_plan, step_outputs,
                        context_summary, last_relevance_score, active_topic,
                        interaction_count, is_deleted, created_at, last_active_at
                    )
                    VALUES (
                        :sid, :user_id, :agent_id, :app_id, :session_title,
                        :session_state, :current_plan, :step_outputs,
                        :context_summary, :last_relevance_score, :active_topic,
                        :interaction_count, :is_deleted, SYSTIMESTAMP, SYSTIMESTAMP
                    )
            """)
            
            # 3. 准备参数
            # 直接从 Entity 对象中获取属性
            params = {
                "sid": context.session_id,
                "user_id": context.user_id,
                "agent_id": context.agent_id,
                "app_id": app_id,
                "session_title": context.session_title,
                "session_state": context.session_state,
                "current_plan": context.current_plan,
                "step_outputs": context.step_outputs,
                "context_summary": context.context_summary,
                "last_relevance_score": context.last_relevance_score,
                "active_topic": context.active_topic,
                "interaction_count": context.interaction_count,
                "is_deleted": context.is_deleted,
            }
            
            # 4. 执行
            await self.session.execute(sql, params)
            
        except Exception as e:
            raise DatabaseException(f"保存会话上下文失败: {str(e)}")

    async def get_user_profile(self, user_id: str) -> UserProfileEntity | None:
        """Get user profile"""
        try:
            stmt = select(UserProfileEntity).where(
                UserProfileEntity.user_id == user_id
            )
            result = await self.session.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Failed to get user profile for user {user_id}", exc_info=e)
            raise DatabaseException("Failed to get user profile", original_error=e)

    async def update_context_state(
        self, 
        session_id: str, 
        new_state: dict, 
        current_plan: dict | None = None,
        step_outputs: dict | None = None,
        active_topic: str | None = None,  # 新增入参
        relevance_score: float | None = None,
        increment_count: bool = True
    ):
        """Update conversation context state"""
        try:
            context = await self.get_context_by_id(session_id)
            if context:
                context.session_state = new_state
                if current_plan:
                    context.current_plan = current_plan
                if step_outputs:
                    context.step_outputs = step_outputs
                if active_topic:
                    context.active_topic = active_topic
                if relevance_score:
                    context.last_relevance_score = relevance_score
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
        
    async def get_sessions(self, session_id: str) -> list[dict[str, Any]]:
        """Get all entries for a session"""
        try:
            stmt = select(MemoryEntryEntity.entry_id, 
                          MemoryEntryEntity.raw_question,
                          MemoryEntryEntity.answer,
                          MemoryEntryEntity.blocks,
                          MemoryEntryEntity.feedback,
                          MemoryEntryEntity.request_time,
                          MemoryEntryEntity.response_time) \
                .where(MemoryEntryEntity.session_id == session_id) \
                .order_by(MemoryEntryEntity.request_time)
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            return [{
                "entry_id": row[0],
                "raw_question": safe_read_content(row[1]),
                "answer": safe_read_content(row[2]),
                "blocks": row[3],
                "feedback": row[4],
                "request_time": row[5],
                "response_time": row[6]
            } for row in rows]
            
        except Exception as e:
            logger.error(f"Failed to get entries for session {session_id}", exc_info=e)
            raise DatabaseException("Failed to get entries", original_error=e)
        
    

    async def update_context_summary(self, session_id: str, new_summary: str):
        """更新会话的滚动摘要"""
        context = await self.get_context_by_id(session_id)
        if context:
            context.context_summary = new_summary
            context.last_update_at = datetime.now(timezone.utc)

    async def upsert_user_profile(self, user_id: str, profile_updates: dict):
        """
        更新用户画像
        """
        # 1. 转换字典为 JSON 字符串
        updates_json = json.dumps(profile_updates, ensure_ascii=False)
        
        # 2. 使用 text() 包装 SQL 语句
        # 注意：Oracle 的 MERGE 语句在 SQLAlchemy 中必须用 text()
        sql = text("""
            MERGE INTO kbot_md_user_profile p
            USING (SELECT :uid as user_id FROM dual) s
            ON (p.user_id = s.user_id)
            WHEN MATCHED THEN
                UPDATE SET 
                    global_preferences = JSON_PARSE(:updates), 
                    frequent_entities  = JSON_PARSE(:updates),
                    entity_relations   = JSON_PARSE(:updates),
                    correction_history = JSON_PARSE(:updates),
                    profile_summary    = COALESCE(p.profile_summary, ''),
                    last_update_time   = SYSTIMESTAMP
            WHEN NOT MATCHED THEN
                INSERT (user_id, global_preferences, frequent_entities, entity_relations, correction_history, last_update_time)
                VALUES (:uid, JSON_PARSE(:updates), JSON_PARSE(:updates), JSON_PARSE(:updates), JSON_PARSE(:updates), SYSTIMESTAMP)
        """)
        
        # 3. 传入参数字典
        await self.session.execute(sql, {"uid": user_id, "updates": updates_json})

    async def search_vector_memory(self, user_id: str, query_vector: list[float], limit: int = 3):
        """
        在 Oracle 26ai 中执行原生向量搜索
        重构点：
        1. 显式过滤掉 feedback = -1 (点踩) 的记录
        2. 返回 feedback 字段用于业务层打标签
        3. 确保跨 session 搜索该用户的所有历史
        """
        # 注意：这里增加了 m.feedback >= 0 的过滤条件
        sql = text("""
            SELECT m.entry_id, m.raw_question, m.answer, m.feedback,
                m.thought, m.current_plan, m.reasoning_path, m.blocks, 
                m.turn_type, m.memory_summary, m.turn_entities,
                m.request_time, m.response_time,
                VECTOR_DISTANCE(m.memory_vector, :vec, COSINE) as distance
            FROM kbot_md_memory_entry m
            JOIN kbot_md_conv_context c ON m.session_id = c.session_id
            WHERE c.user_id = :user_id 
              AND m.feedback >= 0
            ORDER BY distance
            FETCH FIRST :limit ROWS ONLY
        """)
        
        vec_handler = OracleVecHandler()
        oracle_vec = vec_handler.convert(query_vector)
        
        try:
            result = await self.session.execute(
                sql, {
                    "vec": oracle_vec, 
                    "user_id": user_id, 
                    "limit": limit
                }
            )
            # 使用 fetchall() 后，返回的是 Row 对象列表
            return result.fetchall()
        except Exception as e:
            logger.error(f"Failed to search vector memory for user {user_id}", exc_info=e)
            raise DatabaseException("Failed to search vector memory", original_error=e)
        
    async def update_user_profile_summary(self, user_id: str, profile_summary: str):
        """
        更新用户的长期画像描述 (定性总结)
        """
        try:
            # 使用 update 语句而非 ORM 属性赋值，可以避免不必要的 JSON 字段干扰
            stmt = (
                update(UserProfileEntity)
                .where(UserProfileEntity.user_id == user_id)
                .values(
                    profile_summary=profile_summary,
                    last_update_time=func.now()
                )
                .returning(UserProfileEntity.user_id)
            )
            result = await self.session.execute(stmt)
            
            # 如果影响行数为 0，说明用户还没初始化，执行插入
            if result.scalar_one_or_none() is None:
                new_user = UserProfileEntity(
                    user_id=user_id,
                    global_preferences={"confirmed": {}, "inferred": {}},
                    frequent_entities={},
                    correction_history=[],
                    profile_summary="Automatically initialized user profile",
                    last_update_time=datetime.now(timezone.utc)
                )
                self.session.add(new_user)
                
            await self.session.flush() 
        except Exception as e:
            logger.error(f"Failed to update profile summary for user {user_id}", exc_info=e)
            raise DatabaseException("Failed to update user profile summary", original_error=e)
        
    async def update_entry_vector(self, entry_id: str, summary: str, vector: list[float] | None = None):
        """
        物理写入向量到 Oracle 26ai 向量字段
        """
        try:
            update_data: dict[str, Any] = {"memory_summary": summary}
            if vector:
                vec_handler = OracleVecHandler()
                update_data["memory_vector"] = vec_handler.convert(vector)
            
            stmt = (
                update(MemoryEntryEntity)
                .where(MemoryEntryEntity.entry_id == entry_id)
                .values(**update_data)
            )
            await self.session.execute(stmt)
            logger.debug(f"Vector updated for memory entry: {entry_id}")
        except Exception as e:
            logger.error(f"Failed to update vector for entry {entry_id}", exc_info=e)
            raise DatabaseException("Failed to persist vector to Oracle", original_error=e)
        
    async def remove_context_by_agent(self, agent_id: int) -> None:
        """Remove context by agent id"""
        try:
            await self.session.execute(
                update(ConversationContextEntity)
                .where(ConversationContextEntity.agent_id == agent_id)
                .values(is_deleted=True)
            )
        except Exception as e:
            raise DatabaseException("Failed to remove session", original_error=e)
        
    async def remove_context_by_id(self, session_id: str) -> None:
        """Remove session by context id"""
        try:
            await self.session.execute(
                update(ConversationContextEntity)
                .where(ConversationContextEntity.session_id == session_id)
                .values(is_deleted=True)
            )
        except Exception as e:
            raise DatabaseException("Failed to remove session", original_error=e)
    
    async def get_conversation_detail(self, session_id: str) -> list[dict[str, Any]]:
        """
        获取指定会话的所有历史记录：完整展示推理链路 (Oracle 实现)
        
        逻辑：
        1. 使用 row_to_json(t) 将整行数据转为 JSON 字符串（适配 Oracle 23ai）
        2. 筛选 session_id 并按时间升序排列
        3. 将 JSON 字符串解析为 Python dict 列表，保持与 ES 返回结构一致
        """
        
        try:
            # 1. 构建原生 SQL
            # 使用 row_to_json(t) 是最高效的方法，它会自动处理 CLOB/JSON/Vector 到 JSON 的转换
            # 注意：需要给子查询起别名 't'，以便 row_to_json 识别
            sql = text("""
                SELECT json_serialize(row_to_json(t)) as json_data
                FROM kbot_md_memory_entry t
                WHERE t.session_id = :sid
                ORDER BY t.request_time ASC
            """)
            
            result = await self.session.execute(sql, {"sid": session_id})
            
            # 2. 解析结果
            rows = result.fetchall()
            details = []
            
            for row in rows:
                # row.json_data 是从数据库返回的 JSON 字符串
                if row.json_data:
                    # 将 JSON 字符串反序列化为 Python 字典
                    entry_dict = json.loads(row.json_data)
                    details.append(entry_dict)
                    
            return details
            
        except Exception as e:
            raise DatabaseException(f"获取会话详情失败: {session_id}", original_error=e)

    async def get_contexts_by_agent(self, agent_id: int, user_id: str, limit: int = 50) -> list[ConversationContextModel]:
        """
        获取指定 Agent 下的所有会话上下文（元数据）。
        用于前端侧边栏渲染会话列表。
        
        逻辑：
        1. 筛选 agent_id (注意类型转换)，user_id 和 is_deleted
        2. 按 last_active_at 倒序排列
        3. 仅查询轻量级字段，避免加载 session_state 等大字段
        4. 映射回 Pydantic Model
        """
        
        try:
            # 1. 构建查询语句
            # 显式选择需要的列，对应 ES 的 _source 过滤
            stmt = (
                select(
                    ConversationContextEntity.session_id,
                    ConversationContextEntity.user_id,
                    ConversationContextEntity.agent_id,
                    ConversationContextEntity.session_title,
                    ConversationContextEntity.interaction_count,
                    ConversationContextEntity.last_active_at,
                    ConversationContextEntity.created_at,
                    ConversationContextEntity.active_topic
                )
                .where(ConversationContextEntity.agent_id == agent_id)
                .where(ConversationContextEntity.user_id == user_id)
                .where(ConversationContextEntity.is_deleted == False)
                .order_by(desc(ConversationContextEntity.last_active_at))
                .limit(limit)
            )
            
            result = await self.session.execute(stmt)
            rows = result.fetchall()
            
            # 2. 将 ORM 结果映射为 Pydantic Model
            contexts = []
            for row in rows:
                # 使用 model_validate 或 **dict() 转换
                # 注意：row._mapping 提供了类似字典的访问方式
                contexts.append(ConversationContextModel.model_validate(row._mapping))
                
            logger.debug(f"已获取 Agent {agent_id} 下的 {len(contexts)} 个活跃会话")
            return contexts

        except Exception as e:
            logger.error(f"获取 Agent {agent_id} 的会话列表失败: {e}")
            return []
        
    async def get_conversation_list_by_user_id(self, user_id: str) -> list[dict[str, Any]]:
        """Retrieves a list of all chat records associated with a specific `user_id`."""
        try:
            async with self.session as session:
                stmt = (
                    select(ConversationContextEntity.session_id, 
                        ConversationContextEntity.session_title, 
                        ConversationContextEntity.last_active_at)
                    .where(and_(ConversationContextEntity.user_id == user_id, 
                                ConversationContextEntity.is_deleted == False))
                    .order_by(ConversationContextEntity.last_active_at.desc())
                )
                result = await session.execute(stmt)
                rows = result.all()
                return [
                    {
                        "session_id": row[0],
                        "session_title": row[1],
                        "last_active_at": row[2].isoformat() if row[2] else None
                    }
                    for row in rows
                ]
        except Exception as e:
            raise DatabaseException("Failed to get conversation list", original_error=e)
        
    async def rename_conversation(self, session_id: str, new_title: str) -> None:
        """Renamesames a chat session title in the database."""
        try:
            await self.session.execute(
                update(ConversationContextEntity)
                .where(ConversationContextEntity.session_id == session_id)
                .values(session_title=new_title)
            )
        except Exception as e:
            raise DatabaseException("Failed to rename session", original_error=e)