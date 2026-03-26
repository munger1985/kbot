from datetime import datetime
from loguru import logger
from core.database.oracle import get_session
from core.exceptions import InternalServerError
from dao.entities import MemoryEntryEntity
from dao.repositories import MemoryEntryRepository
from .state_manager import SessionStateManager
from .context_manager import ContextManager
from utils.clients.model_client import AIModelClient

class MemoryService:
    def __init__(self):
        self.manager = ContextManager()
        self.model_client = AIModelClient()
    
    @property
    def oracle_session(self):
        return get_session()

    async def prepare_context_and_rewrite(
        self, 
        session_id: str, 
        raw_question: str,
        llm_model: str
    ) -> dict:
        """
        功能 1：加载 -> 改写 -> 合并状态（用于 RAG 检索前）
        """
        # 1. 从 Repo 获取原始数据
        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            context = await context_repo.get_context_by_id(session_id)
            old_state = context.session_state if context else {}
            history_summary = context.context_summary if context else ""

            # 2. 调用 LLM 改写模块
            rewrite_data = await self.manager.process_query_with_memory(
                query=raw_question,
                context_summary=history_summary,
                session_state=old_state,
                model_name=llm_model
            )

            # 3. 内存合并新状态（不立即写入 DB，供本轮 RAG 使用）
            new_state = SessionStateManager.merge_state(
                old_state, 
                rewrite_data.get('turn_entities')
            )

            return {
                "standalone_query": rewrite_data['standalone_query'],
                "search_keywords": rewrite_data['search_keywords'],
                "intent_category": rewrite_data['intent_category'],
                "turn_entities": rewrite_data['turn_entities'],
                "new_state": new_state,
                "old_context": context # 保留引用
            }

    async def finalize_and_persist(
        self,
        session_id: str,
        user_id: str,
        raw_question: str,
        answer: str,
        prepared_data: dict,
        request_time: datetime,
        retrieved_chunks: list | None = None
    ):
        """
        持久化逻辑：
        1. 同步 Session 状态 (短期)
        2. 保存对话记录 (带向量与改写信息)
        3. 更新用户画像 (长期)
        """
        # 从 prepared_data 中提取由 ContextManager 识别的画像更新
        profile_updates = prepared_data.get("user_profile_updates", {})

        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)

            try:
                # 1. 更新 Context 状态 (Session 级别的结构化上下文)
                await context_repo.update_context_state(
                    session_id=session_id,
                    new_state=prepared_data.get('new_state', {})
                )

                # 2. 创建并保存 Memory Entry (对话流水账)
                new_entry = MemoryEntryEntity(
                    session_id=session_id,
                    raw_question=raw_question,
                    answer=answer,
                    standalone_query=prepared_data.get('standalone_query', raw_question),
                    search_keywords=prepared_data.get('search_keywords', ""),
                    turn_entities=prepared_data.get('turn_entities', {}), 
                    intent_category=prepared_data.get('intent_category', "general"),
                    retrieved_chunks=retrieved_chunks,
                    request_time=request_time,
                    response_time=datetime.now()
                )
                await context_repo.add_memory_entry(new_entry)
                logger.info(f"Interaction persisted for session: {session_id}")

                # 3. 核心新增：用户画像持久化 (跨会话的长期特征)
                if profile_updates and len(profile_updates) > 0:
                    logger.info(f"Updating long-term profile for user {user_id}: {profile_updates}")
                    # 调用 repository 进行 JSON 增量合并
                    await context_repo.upsert_user_profile(
                        user_id=user_id,
                        profile_updates=profile_updates
                    )
                    logger.info(f"Full persistence cycle completed for session: {session_id}")

            except Exception as e:
                logger.error(f"Persistence cycle failed, rolled back: {e}")
                raise InternalServerError(f"Failed to persist memory: {e}")

    async def get_relevant_memories(self, user_id: str, query_vector: list[float]) -> str:
        """
        获取相关的历史记忆片段，转为提示词背景
        """
        # 检索最近似的历史 Q&A
        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            hits = await context_repo.search_vector_memory(user_id, query_vector)
        
        if not hits:
            return ""

        # 格式化为 Context 字符串
        memory_context = "\n".join([
            f"历史相关问题: {h.raw_question}\n历史解答: {h.answer}" 
            for h in hits if h.distance < 0.5 # 仅保留相似度高的
        ])
        return f"--- 相关的历史记忆 ---\n{memory_context}\n"
    
    async def record_user_feedback(self, entry_id: int, score: int):
        """
        记录用户对某一轮回答的满意度
        """
        if score not in [-1, 0, 1]:
            raise ValueError("Feedback score must be -1, 0, or 1")

        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            try:
                await context_repo.update_feedback(entry_id, score)
                await context_repo.session.commit()
                
                if score == -1:
                    # 策略：如果是差评，可以在此处记录日志或推送到飞书/钉钉告警，
                    # 方便开发人员后续针对性调优 RAG
                    logger.warning(f"User negative feedback received for entry: {entry_id}")
                
                return True
            except Exception as e:
                logger.error(f"Failed to record feedback for {entry_id}: {e}")
                await context_repo.session.rollback()
                return False