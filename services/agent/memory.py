# services/memory/memory_service.py

from datetime import datetime, timezone
from dao.entities import ChatMemoryEntity, ChatSessionEntity
from dao.repositories import ChatMemoryRepository, ChatSessionRepository
from core.database.oracle import get_session

class MemoryService:

    @property
    def oracle_session(self):
        return get_session()


    async def get_context_parts(self, session_id: str, question: str, query_vec: list[float]) -> dict[str, str]:
        """
        获取记忆增强片段
        """
        async with self.oracle_session as session:
            repo = ChatMemoryRepository(session)
            # 1. 召回短期
            st_records = await repo.search_short_term(session_id)
            short_term_txt = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in st_records])

            # 2. 召回长期 (语义检索)
            lt_records = await repo.search_long_term(query_vec, exclude_session=session_id)
            long_term_txt = "\n".join([f"(历史经验) Q: {r['question']} -> A: {r['answer']}" for r in lt_records])

            return {
                "short_term": short_term_txt,
                "long_term": long_term_txt
            }

    async def save_memory(self, session_id: str, user_id: str, question: str, answer: str, references: list, request_time: datetime, query_vec: list[float]):
        """持久化本次问答"""
        async with self.oracle_session as session:
            # 确保会话存在
            session_repo = ChatSessionRepository(session)
            existing_sessions = await session_repo.get_by_session_id(session_id)
            if not existing_sessions:
                # 创建新会话
                new_session = ChatSessionEntity(
                    session_id=session_id,
                    session_title=None,
                    app_id=None,
                    agent_id=None,
                    appuser=user_id,
                    updated_by=user_id
                )
                await session_repo.create(new_session)

            # 保存聊天记忆
            repo = ChatMemoryRepository(session)
            doc = ChatMemoryEntity(
                session_id=session_id,
                user_id=user_id,
                question=question,
                answer=answer,
                question_vector=query_vec,
                references=references,
                feedback=0,
                request_time=request_time,
                response_time=datetime.now(timezone.utc)
            )
            await repo.create(doc)

    async def get_memory(self, session_id: str) -> list[dict]:
        """
        获取指定会话的记忆明细
        返回格式适配前端：[{role: "user", content: "..."}, {role: "assistant", content: "...", references: [...]}]
        """
        async with self.oracle_session as session:
            repo = ChatMemoryRepository(session)
            
            records = await repo.get_session_history(session_id)
        
            formatted_history = []
            for rec in records:
                # 用户问题
                formatted_history.append({
                    "role": "user",
                    "content": rec.get("question", ""),
                    "time": rec.get("request_time")
                })
                # 助手回答及引用
                formatted_history.append({
                    "role": "assistant",
                    "content": rec.get("answer", ""),
                    "references": rec.get("references", []),
                    "time": rec.get("response_time")
                })
                
            return formatted_history