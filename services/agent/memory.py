# services/memory/memory_service.py

from datetime import datetime, timezone
from dao.repositories import ChatMemoryRepository

class MemoryService:
    def __init__(self, user_id: str):
        self.repo = ChatMemoryRepository(user_id)

    async def get_context_parts(self, session_id: str, question: str, query_vec: list[float]) -> dict[str, str]:
        """
        获取记忆增强片段
        """
        # 1. 召回短期
        st_records = await self.repo.search_short_term(session_id)
        short_term_txt = "\n".join([f"Q: {r['question']}\nA: {r['answer']}" for r in st_records])

        # 2. 召回长期 (语义检索)
        lt_records = await self.repo.search_long_term(query_vec, exclude_session=session_id)
        long_term_txt = "\n".join([f"(历史经验) Q: {r['question']} -> A: {r['answer']}" for r in lt_records])

        return {
            "short_term": short_term_txt,
            "long_term": long_term_txt
        }

    async def save_memory(self, session_id: str, question: str, answer: str, references: list, request_time: datetime, query_vec: list[float]):
        """持久化本次问答"""            
        doc = {
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "question_vector": query_vec,
            "references": references,
            "request_time": request_time,
            "created_at": datetime.now(timezone.utc)
        }
        await self.repo.add_record(doc)

    async def get_memory(self, session_id: str) -> list[dict]:
        """
        获取指定会话的记忆明细
        返回格式适配前端：[{role: "user", content: "..."}, {role: "assistant", content: "...", references: [...]}]
        """
        records = await self.repo.get_session_history(session_id)
        
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
                "time": rec.get("created_at")
            })
            
        return formatted_history