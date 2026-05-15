import json
from loguru import logger
from typing import Any
from services.search.result import TxtBaseSearchResult
from agent.prompt import default_prompt
from utils.clients import AIModelClient
from core.config.settings import get_prompt_config


class ContextManager:
    def __init__(self):
        self.llm_client = AIModelClient()
        conf = get_prompt_config()
        self.rewrite_prompt = conf.rewrite_question
        self.summary_prompt = conf.refresh_summary # 用于反思和精炼
        self.rag_final_render = conf.rag_final_render

    async def process_query_with_memory(
        self, 
        query: str,
        chat_history: str,
        context_summary: str | None, 
        session_state: dict | None,
        model_name: str,
        active_topic: str | None = None # 当前感知的话题
    ) -> dict:
        """
        核心入口：
        输出包含意图、改写、话题、实体和状态更新的全套决策包。
        """
        # 1. 组装 Prompt (包含记忆与状态)
        prompt = await default_prompt.generate(
            self.rewrite_prompt,
            chat_history=chat_history if chat_history else 'No previous turns.',
            summary=context_summary or 'None',
            session_state=json.dumps(session_state or {}, ensure_ascii=False),
            active_topic=active_topic or "None",
            query=query
        )
        
        try:
            # 2. 获取结构化结果
            # 设置 low temperature 以获得稳定的 JSON
            result = await self.llm_client.get_llm_json(
                model_name=model_name, 
                prompt=prompt, 
                temperature=0.0
            )
            
            # 3. 话题相关性判定 (核心逻辑)
            relevance = result.get("context_relevance", 1.0)
            standalone_query = result.get("standalone_query", query)

            # 话题判定：如果 LLM 返回了新的 topic，则更新
            # 如果 relevance 过低，通常意味着 active_topic 发生了切换
            detected_topic = result.get("active_topic", active_topic)
            
            # 4. 状态合并 (Slot Filling)
            # 使用增量更新逻辑，防止丢失旧状态
            current_state = session_state or {}
            turn_entities = result.get("turn_entities", {})
            new_state = {**current_state, **turn_entities}

            return {
                "standalone_query": standalone_query,
                "turn_type": result.get("turn_type", "new_topic"),
                "active_topic": detected_topic,      # 传递给下游持久化
                "context_relevance": relevance,
                "search_keywords": " ".join(result.get("search_keywords", [])),
                "turn_entities": turn_entities,
                "new_state": new_state,              # 合并后的完整状态
                "user_profile_updates": result.get("user_profile_updates", {}), # 画像增量
                "thought": result.get("thought", "") # 引导后续 Planner 的思考
            }

        except Exception as e:
            logger.error(f"查询处理失败，降级为默认话题切换: {e}")
            return {
                "standalone_query": query,
                "turn_type": "new_topic",
                "active_topic": active_topic,
                "context_relevance": 1.0,
                "search_keywords": query,
                "turn_entities": {},
                "new_state": session_state or {},
                "user_profile_updates": {}
            }
    
    async def reflect_and_summarize(
        self, 
        user_id: str, 
        old_summary: str, 
        query: str, 
        answer: str, 
        model_name: str
    ) -> tuple[str, str]:
        """
        反思并精炼对话记忆
        用于更新长期画像和精炼当前对话摘要
        返回：(new_profile_summary, memory_snapshot)
        """
        prompt = await default_prompt.generate(
            self.summary_prompt,
            old_summary=old_summary,
            query=query,
            answer=answer
        )

        try:
            result = await self.llm_client.get_llm_json(
                model_name=model_name,
                prompt=prompt,
                temperature=0.0
            )
            # profile_summary: 长期画像的描述
            # memory_snapshot: 本次对话的精炼摘要（用于向量检索）
            return (
                result.get("profile_summary", old_summary), 
                result.get("memory_snapshot", f"Q: {query} A: {answer}")
            )
        except Exception as e:
            logger.error(f"LLM 反思生成失败: {e}")
            return old_summary, f"Q: {query} A: {answer}"

    async def build_final_prompt(
        self,
        system_prompt: str,
        user_question: str,
        kb_results: list[TxtBaseSearchResult],
        session_state: dict[str, Any] | None = None,
        context_summary: str | None = "",
        long_term_memory: str | None = "",
        reasoning_path: list[str] | None = None # 新增：展示推理路径
    ) -> str:
        """
        构建最终 RAG 提示词
        """
        # 1. 环境信息
        env_str = " | ".join([f"{k}: {v}" for k, v in session_state.items() if v]) if session_state else "通用环境"
        
        # 2. 知识库引用
        kb_segments = []
        for i, res in enumerate(kb_results):
            kb_segments.append(f"[[参考资料 {i+1}]]\n{res.content}")
        kb_context = "\n\n".join(kb_segments) if kb_segments else "未找到直接相关的核心知识。"

        # 3. 推理路径格式化 (让 LLM 知道刚才查了什么)
        path_str = " -> ".join(reasoning_path) if reasoning_path else "直接回答"

        return await default_prompt.generate(
            self.rag_final_render,
            system_prompt=system_prompt,
            env_str=env_str,
            reasoning_path=path_str, # 传递给模板
            context_summary=context_summary or "新对话。",
            long_term_memory=long_term_memory or "暂无相关跨会话记忆。",
            kb_context=kb_context,
            user_question=user_question
        )