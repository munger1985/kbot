from datetime import datetime
from loguru import logger
from core.database.oracle import get_session
from core.exceptions import InternalServerError
from dao.entities import MemoryEntryEntity
from dao.repositories import MemoryEntryRepository
from .state_manager import SessionStateManager
from .context_manager import ContextManager
from utils.clients.model_client import AIModelClient
from utils.common import safe_read_content


class MemoryService:
    def __init__(self):
        self.manager = ContextManager()
        self.model_client = AIModelClient()
    
    @property
    def oracle_session(self):
        return get_session()
    
    async def get_user_profile(self, user_id: str) -> dict:
        """获取用户画像并转换为用于 Context 的字典格式"""
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            profile = await repo.get_user_profile(user_id)
            
            if not profile:
                return {}

            # 将实体中的多个字段合并为一个扁平化的画像字典
            combined_profile = {
                "profile_summary": safe_read_content(profile.profile_summary) or "",
                **(profile.global_preferences or {}),
                **(profile.frequent_entities or {})
            }
            return combined_profile

    async def prepare_context_and_rewrite(
        self, 
        session_id: str, 
        raw_question: str,
        llm_model: str,
        user_profile: dict | None = None  # 1. 接收从 Orchestrator 传来的画像
    ) -> dict:
        """
        功能 1：加载画像与上下文 -> 注入重写 -> 状态合并
        """
        async with self.oracle_session as session:
            context_repo = MemoryEntryRepository(session)
            context = await context_repo.get_context_by_id(session_id)
            
            # 2. 状态提取
            old_state = context.session_state if context else {}
            history_summary = context.context_summary if context else ""

            # 3. 关键改动：将画像与会话状态合并作为重写的依据
            # 优先级：Session State (当前报错/临时路径) 覆盖 User Profile (职业/偏好)
            # 这样 LLM 就能感知到用户是 "DBA" 且正在处理 "RHEL 8" 环境
            rewrite_context_state = {**(user_profile or {}), **old_state}

            # 4. 调用 LLM 改写模块 (传入合并后的认知信息)
            rewrite_data = await self.manager.process_query_with_memory(
                query=raw_question,
                context_summary=history_summary,
                session_state=rewrite_context_state, 
                model_name=llm_model
            )

            # 5. 内存合并新状态
            # 注意：这里我们保留 combined 结果，并混入本轮新提取的 turn_entities
            new_state = SessionStateManager.merge_state(
                rewrite_context_state, 
                rewrite_data.get('turn_entities')
            )

            # 6. 如果重写器提取了新的画像更新（例如用户说：我换成 Ubuntu 22 了）
            # 我们将这些更新也合并到 new_state 中，以便在持久化阶段写入 Profile 表
            profile_updates = rewrite_data.get('user_profile_updates', {})
            if profile_updates:
                new_state = {**new_state, **profile_updates}

            return {
                "standalone_query": rewrite_data['standalone_query'],
                "search_keywords": rewrite_data['search_keywords'],
                "intent_category": rewrite_data['intent_category'],
                "turn_entities": rewrite_data['turn_entities'],
                "user_profile_updates": profile_updates, # 显式返回，方便后续写入
                "new_state": new_state,
                "old_context": context 
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

    async def get_relevant_memories(self, user_id: str, query_vector: list) -> str:
        """
        召回长期记忆并格式化为带权重的字符串
        """
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            # 1. 向量检索 (假设你的 repo 已经支持向量搜索并过滤了 feedback >= 0)
            entries = await repo.search_vector_memory(
                user_id=user_id, 
                query_vector=query_vector, 
                limit=5
            )
            
            if not entries:
                return ""

            # 2. 格式化逻辑：将对象转为带权重的字符串
            memory_blocks = []
            for e in entries:
                # 这里的 e 是 MemoryEntryEntity 对象
                # 再次确保过滤掉用户点踩的内容（如果 repo 层没过滤干净）
                if e.feedback == -1:
                    continue
                    
                # 根据反馈强度添加视觉引导标签
                prefix = "⭐ [高价值历史方案]" if e.feedback == 1 else "[历史参考]"
                
                block = f"{prefix}\n问题: {e.raw_question}\n方案: {e.answer}"
                memory_blocks.append(block)

            return "\n\n".join(memory_blocks)
    
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