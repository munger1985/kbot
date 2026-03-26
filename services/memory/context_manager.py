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
        chat_history: str,
        context_summary: str | None, 
        session_state: dict | None,
        model_name: str
    ) -> dict:
        """
        重构后的核心入口：
        输入原始问题 + 记忆，输出改写后的全套参数。
        """
        # 1. 组装 Prompt (包含记忆与状态)
        prompt = self._build_rewrite_prompt(query, chat_history, context_summary, session_state)
        
        try:
            # 2. 获取结构化结果
            # 设置 low temperature 以获得稳定的 JSON
            result = await self.llm_client.call_llm_json(
                model_name=model_name, 
                prompt=prompt, 
                temperature=0.0
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

    def _build_rewrite_prompt(self, query: str, chat_history: str, summary: str | None, state: dict | None) -> str:
        return f"""
You are the Context & Identity Engine for the NexusCube RAG system.
Your goal is to transform the user's raw input into a structured execution plan while maintaining a persistent User Profile.

### Recent Dialogue (Short-term Memory)
{chat_history if chat_history else 'No previous turns in this session.'}

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
    
    async def refresh_summary(self, session_id: str, user_id: str, model_name: str):
        """
        双重更新：1. 会话摘要 (Context Summary) 2. 用户画像描述 (Profile Summary)
        """
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            entries = await repo.get_recent_entries(session_id, limit=10)
            if not entries: return

            # 1. 组装对话事实
            history_list = []
            for e in entries:
                # 显式告诉 LLM 哪些是“点赞”的答案，让其在总结摘要时重点保留
                tag = "[用户认可的正确方案] " if e.feedback == 1 else ""
                history_list.append(f"User: {e.raw_question}\nAssistant: {tag}{e.answer}")
            history_text = "\n".join(history_list)

            # 2. 构造“双任务”Prompt
            # 强制 LLM 分开输出：一个是当前聊的事，一个是关于这个人的描述
            prompt = f"""
请分析以下对话，产出两段总结。用 '---' 分隔。

任务 1：会话摘要 (Context Summary)
要求：记录当前正在处理的技术问题、环境（如 Ubuntu 24.04）及已验证的方案。

任务 2：用户画像描述 (User Profile Summary)
要求：基于对话定性描述用户。例如：职业身份、技术水平、沟通风格。

对话历史：
{history_text}
"""
            try:
                # 3. 调用 LLM 生成摘要
                full_content = ""
                async for chunk in self.llm_client.call_llm_model(model_name=model_name, prompt=prompt, stream=False):
                    # 假设返回的是 SSE 格式，解析逻辑同之前
                    full_content += chunk # 简化示意，实际需按之前解析逻辑处理
                
                if "---" in full_content:
                    parts = full_content.split("---")
                    context_summary = parts[0].strip()
                    profile_summary = parts[1].strip()

                    # 4. 同时持久化会话上下文总结和用户画像
                    # 更新当前会话的上下文摘要
                    await repo.update_context_summary(session_id, context_summary)
                    
                    # 更新用户的长期画像总结（回答你之前的问题：在这里触发更新！）
                    await repo.update_user_profile_summary(user_id, profile_summary)

                    logger.info(f"Memory Capsule synced for user {user_id} in session {session_id}")

            except Exception as e:
                logger.error(f"Failed to sync memory capsule: {e}")

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