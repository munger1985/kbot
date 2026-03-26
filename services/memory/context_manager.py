import json
from loguru import logger
from typing import Any
from services.search.result import TxtBaseSearchResult
from services.prompt_service import PromptService
from utils.clients import AIModelClient
from core.database.oracle import get_session
from dao.repositories import MemoryEntryRepository

class ContextManager:
    def __init__(self):
        self.llm_client = AIModelClient()

    @property
    def oracle_session(self):
        return get_session()

    async def process_query_with_memory(
        self, 
        query: str, 
        context_summary: str | None, 
        session_state: dict | None,
        model_name: str
    ) -> dict:
        """
        重构后的核心入口：
        输入原始问题 + 记忆，输出改写后的全套参数。
        """
        # 1. 组装 Prompt (包含记忆与状态)
        prompt = self._build_rewrite_prompt(query, context_summary, session_state)
        
        try:
            # 2. 获取结构化结果
            # 设置 low temperature 以获得稳定的 JSON
            result = await self.llm_client.call_llm_json(
                model_name=model_name, 
                prompt=prompt, 
                temperature=0.1 
            )
            
            # 3. 结果处理与字段校验 (确保字段存在)
            processed = {
                "standalone_query": result.get("standalone_query", query),
                "search_keywords": " ".join(result.get("search_keywords", [])),
                "turn_entities": result.get("turn_entities", {}),
                "intent_category": result.get("intent", "technical_inquiry")
            }
            
            logger.info(f"Rewrite: {query} -> {processed['standalone_query']}")
            return processed

        except Exception as e:
            logger.warning(f"Query rewrite failed, using raw query. Error: {e}")
            # 彻底故障时的兜底方案：返回原样，不阻塞后续 RAG
            return {
                "standalone_query": query,
                "search_keywords": query,
                "turn_entities": {},
                "intent_category": "fallback"
            }

    def _build_rewrite_prompt(self, query: str, summary: str | None, state: dict | None) -> str:
        return f"""
You are the Context & Identity Engine for the NexusCube RAG system.
Your goal is to transform the user's raw input into a structured execution plan while maintaining a persistent User Profile.

### Context Knowledge
- **Historical Summary**: {summary or 'None'} (General progress of the conversation)
- **Active Session State**: {json.dumps(state or {}, ensure_ascii=False)} (Volatile data: current errors, temporary IPs, active file paths)

### Task 1: Query Rewriting
1. **Pronoun Resolution**: Replace 'it', 'the error', 'there' with specific entities from History/State.
2. **Context Injection**: If the user asks "How to install?", rewrite it as "How to install [Software] on [OS from Session State]?"
3. **Keyword Extraction**: Provide 3-5 high-quality technical keywords for Full-Text Search.

### Task 2: Information Categorization (Crucial)
Distinguish between "Turn Entities" and "User Profile Updates":
- **Turn Entities**: Transient data for the CURRENT turn only (e.g., a specific process ID, a one-time error log snippet).
- **User Profile Updates**: Long-term traits that define WHO the user is or their PERMANENT environment. 
* Examples: Professional role (DBA, Developer), preferred OS (RHEL 8), Hardware specs (Xeon Gold 5520+), habitual coding language (Python), or level of expertise.

### Output Format (Strict JSON)
{{
"standalone_query": "The fully independent and contextualized question",
"search_keywords": ["keyword1", "keyword2", "keyword3"],
"turn_entities": {{
    "temp_file": "/tmp/test.log",
    "current_error_code": "ORA-00600"
}},
"user_profile_updates": {{
    "job_role": "System Architect",
    "primary_os": "Oracle Linux 8.8",
    "cpu_arch": "x86_64",
    "expertise_level": "Senior"
}},
"intent": "troubleshooting | installation | optimization | general_inquiry"
}}

User Input: {query}
"""
    
    async def refresh_summary(self, session_id: str, model_name: str):
        """
        核心逻辑：读取最近对话 -> LLM 压缩 -> 写入 context_summary
        """
        # 1. 获取最近 10 轮交互
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            entries = await repo.get_recent_entries(session_id, limit=10)
            if not entries: return

            # 在格式化历史时，给 feedback=1 的记录加权重标识
            history_list = []
            for e in entries:
                # 显式告诉 LLM 哪些是“点赞”的答案，让其在总结摘要时重点保留
                tag = "[用户认可的正确方案] " if e.feedback == 1 else ""
                history_list.append(f"User: {e.raw_question}\nAssistant: {tag}{e.answer}")

            history_text = "\n".join(history_list)

            # 3. 构建摘要 Prompt
            prompt = f"""
请根据以下对话历史，更新并生成一个精炼的“上下文摘要”。
要求：
1. 保留关键的技术决策、正在讨论的问题、已解决的错误以及用户提到的特定环境。
2. 长度控制在 300 字以内。
3. 采用客观陈述句。
注意：带有“[用户认可的正确方案]”标记的内容是用户反馈有效的关键信息，请务必准确保留其技术细节。
历史对话：
{history_text}
"""
            try:
                # 4. 调用 LLM 生成摘要（此处建议用 call_llm_model 聚合结果）
                summary_content = ""
                async for chunk in self.llm_client.call_llm_model(model_name=model_name, prompt=prompt, stream=False):
                    # 假设返回的是 SSE 格式，解析逻辑同之前
                    summary_content += chunk # 简化示意，实际需按之前解析逻辑处理
                
                # 5. 持久化到 Oracle
                if summary_content:
                    await repo.update_context_summary(session_id, summary_content.strip())
                    await repo.session.commit()
                    logger.info(f"Summary updated for session {session_id}")

            except Exception as e:
                logger.error(f"Failed to refresh summary: {e}")

    def build_final_prompt(
        self,
        system_prompt: str,
        user_question: str,
        kb_results: list[TxtBaseSearchResult],
        session_state: dict[str, Any] | None = None,
        context_summary: str | None = "",
        long_term_memory: str | None = ""
    ) -> str:
        # 1. 格式化环境约束
        env_str = " | ".join([f"{k}: {v}" for k, v in session_state.items() if v]) if session_state else "通用环境"
        
        # 2. 格式化知识库
        kb_segments = []
        for i, res in enumerate(kb_results):
            path_str = " > ".join(res.path_names) if hasattr(res, 'path_names') and res.path_names else "知识库资料"
            kb_segments.append(f"[参考资料 {i+1} | 来源: {path_str}]\n{res.content}")
        kb_context = "\n\n".join(kb_segments) if kb_segments else "未找到直接相关的核心知识。"

        # 3. 构造分层模板
        final_prompt = f"""{system_prompt}

### 当前环境约束 (Session State)
**[必须遵守]** 以下是当前用户的运行环境：
- {env_str}

### 对话背景摘要 (Context Summary)
**[重要上下文]** 本次会话之前的进展：
{context_summary if context_summary else "对话开始阶段。"}

### 历史相关经验 (Long-term Memory)
**[仅供参考]** 以下是过往类似场景的经验（带 ⭐ 为用户认可的方案）：
{long_term_memory if long_term_memory else "暂无相关跨会话历史。"}

### 核心知识库依据 (Knowledge Base)
**[主要依据]** 请根据以下权威文档回答问题：
{kb_context}

---
请综合上述背景，优先依据【核心知识库】和【环境约束】。

用户当前的问题：{user_question}
助手回答："""

        return final_prompt