from datetime import datetime
from loguru import logger
from core.database.oracle import get_session
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
        raw_question: str,
        answer: str,
        prepared_data: dict,
        request_time: datetime,
        retrieved_chunks: list | None = None
    ):
        """
        功能 2：持久化（用于 AI 回答后）
        """
        # 1. 更新 Context 状态
        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            await context_repo.update_context_state(
                session_id=session_id,
                new_state=prepared_data['new_state']
            )

            # 2. 创建并保存 Memory Entry
            new_entry = MemoryEntryEntity(
                session_id=session_id,
                raw_question=raw_question,
                answer=answer,
                standalone_query=prepared_data['standalone_query'],
                search_keywords=prepared_data['search_keywords'],
                turn_entities=prepared_data['turn_entities'], # 仅记录本轮增量
                intent_category=prepared_data['intent_category'],
                retrieved_chunks=retrieved_chunks,
                request_time=request_time,
                response_time=datetime.now()
            )
            
            await context_repo.add_memory_entry(new_entry)
            logger.info(f"Interaction persisted for session: {session_id}")

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