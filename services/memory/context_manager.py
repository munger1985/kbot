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
        # 这里建议将 Prompt 存储在 DB 或配置中，下面是逻辑展示
        return f"""
You are a context-aware query rewriter for the NexusCube RAG system.

### Context
- Summary of History: {summary or 'None'}
- Active Session State: {json.dumps(state or {}, ensure_ascii=False)}

### Task
Analyze the User's current question and the context.
1. Resolve pronouns (e.g., 'it', 'there', 'that error').
2. Inject technical context from 'Active Session State' if relevant.
3. Extract core keywords for full-text search.
4. Identify any new parameters (IPs, Paths, Codes) as 'turn_entities'.

### Output Format (Strict JSON)
{{
  "standalone_query": "Rewritten independent question",
  "search_keywords": ["keyword1", "keyword2"],
  "turn_entities": {{"key": "value"}},
  "intent": "e.g., install, troubleshooting"
}}

User Question: {query}
"""
    
    async def refresh_summary(self, session_id: str, model_name: str):
        """
        核心逻辑：读取最近对话 -> LLM 压缩 -> 写入 context_summary
        """
        # 1. 获取最近 10 轮交互
        async with self.oracle_session as session:
            repo = MemoryEntryRepository(session)
            entries = await repo.get_recent_entries(session_id, limit=10)
            if not entries:
                return

            # 2. 格式化对话历史给 LLM
            history_text = "\n".join([
                f"User: {e.raw_question}\nAssistant: {e.answer}" 
                for e in entries if e.answer
            ])

            # 3. 构建摘要 Prompt
            prompt = f"""
请根据以下对话历史，更新并生成一个精炼的“上下文摘要”。
要求：
1. 保留关键的技术决策、正在讨论的问题、已解决的错误以及用户提到的特定环境。
2. 长度控制在 300 字以内。
3. 采用客观陈述句。

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
        """
        组装最终的 Prompt。
        
        Args:
            system_prompt: 系统全局角色定义
            user_question: 改写后的 standalone_query
            kb_results: HybridRetriever 返回的文档对象列表
            session_state: 当前会话的结构化实体状态 (如 OS, Version)
            context_summary: ContextManager 生成的滚动摘要 (代替原始对话流)
            long_term_memory: VectorSearchService 召回的历史 Q&A 经验
        """

        # 1. 格式化环境约束 (Session State) - 优先级最高，作为回答的底色
        env_str = " | ".join([f"{k}: {v}" for k, v in session_state.items() if v]) if session_state else "通用环境"
        
        # 2. 格式化知识库检索结果 (带有路径基因)
        kb_segments = []
        for i, res in enumerate(kb_results):
            # res.path_names 是你在 Docling/OpenViking 解析时保存的路径信息
            path_str = " > ".join(res.path_names) if hasattr(res, 'path_names') and res.path_names else "知识库资料"
            segment = f"[参考资料 {i+1} | 来源: {path_str}]\n{res.content}"
            kb_segments.append(segment)
        
        kb_context = "\n\n".join(kb_segments) if kb_segments else "未找到直接相关的核心知识，请基于通用知识并告知用户。"

        # 3. 构造分层上下文模板 (按照优先级排列)
        final_prompt = f"""{system_prompt}

### 当前环境约束 (Session State)
**[必须遵守]** 以下是当前用户的运行环境，请确保回答与其兼容：
- {env_str if env_str else "标准环境"}

### 对话背景摘要 (Context Summary)
**[辅助理解]** 本次会话之前的进展概述：
{context_summary if context_summary else "对话开始阶段。"}

### 历史相关经验 (Long-term Memory)
**[仅供参考]** 以下是用户在过去其他会话中讨论过的类似情况（注意甄别时效性）：
{long_term_memory if long_term_memory else "暂无相关跨会话历史。"}

### 核心知识库依据 (Knowledge Base)
**[主要依据]** 请根据以下权威文档回答问题，并引用对应的 [参考资料 X]：
{kb_context}

---
请综合上述背景，优先依据【核心知识库】和【环境约束】给出准确、专业的回答。

用户当前的问题：{user_question}
助手回答："""

        return final_prompt