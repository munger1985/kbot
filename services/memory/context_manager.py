import json
from loguru import logger
from typing import Any
from services.search.result import TxtBaseSearchResult
from services.prompt_service import PromptService
from utils.clients import AIModelClient
from core.database.oracle import get_session
from core.config.settings import get_prompt_config
from dao.repositories import MemoryEntryRepository
from .default_prompt import *

class ContextManager:
    def __init__(self):
        self.llm_client = AIModelClient()
        self.default_prompt = DefaultPrompt()

        conf = get_prompt_config()
        self.rewrite_prompt = conf.rewrite_question
        self.summary_prompt = conf.refresh_summary
        self.rag_final_render = conf.rag_final_render
        self.formatter = LazyFormatter()

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
        template = await self.default_prompt.get_prompt_content(self.rewrite_prompt, DEFAULT_REWRITE_PROMPT)

        # 填充模板
        prompt = self.formatter.format(
            template,
            chat_history=chat_history if chat_history else 'No previous turns.',
            summary=context_summary or 'None',
            session_state=json.dumps(session_state or {}, ensure_ascii=False),
            query=query
        )
        
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
            template = await self.default_prompt.get_prompt_content(self.summary_prompt, DEFAULT_SUMMARY_PROMPT)
            prompt = self.formatter.format(template, history_text=history_text)

            try:
                # 3. 调用结构化 JSON 接口
                # 使用 low temperature (0.0) 确保 JSON 结构的稳定性
                result = await self.llm_client.call_llm_json(
                    model_name=model_name,
                    prompt=prompt,
                    temperature=0.0
                )
                
                # 4. 字段校验与提取
                context_summary = result.get("context_summary", "").strip()
                profile_summary = result.get("profile_summary", "").strip()
                if not context_summary or not profile_summary:
                    logger.warning(f"LLM returned incomplete JSON for session {session_id}")
                    return

                # 5. 持久化到数据库
                # 更新当前会话的上下文摘要 (短期记忆)
                await repo.update_context_summary(session_id, context_summary)
                
                # 更新用户的长期画像总结 (长期记忆)
                await repo.update_user_profile_summary(user_id, profile_summary)

                logger.info(f"Memory Capsule synced for user {user_id} in session {session_id}")

            except Exception as e:
                logger.error(f"Failed to sync memory capsule: {e}")

    async def build_final_prompt(
        self,
        system_prompt: str,
        user_question: str,
        kb_results: list[TxtBaseSearchResult],
        session_state: dict[str, Any] | None = None,
        context_summary: str | None = "",
        long_term_memory: str | None = ""
    ) -> str:
        # 1. 获取 RAG 最终渲染模板
        template = await self.default_prompt.get_prompt_content(self.rag_final_render, DEFAULT_FINAL_RAG_PROMPT)

        # 2. 格式化环境约束
        env_str = " | ".join([f"{k}: {v}" for k, v in session_state.items() if v]) if session_state else "通用环境"
        
        # 3. 格式化知识库
        kb_segments = []
        for i, res in enumerate(kb_results):
            path_str = " > ".join(res.path_names) if hasattr(res, 'path_names') and res.path_names else "知识库资料"
            kb_segments.append(f"[参考资料 {i+1} | 来源: {path_str}]\n{res.content}")
        kb_context = "\n\n".join(kb_segments) if kb_segments else "未找到直接相关的核心知识。"

        # 4. 构造分层模板
        return self.formatter.format(
            template,
            system_prompt=system_prompt,
            env_str=env_str,
            context_summary=context_summary or "对话开始阶段。",
            long_term_memory=long_term_memory or "暂无相关跨会话历史。",
            kb_context=kb_context,
            user_question=user_question
        )