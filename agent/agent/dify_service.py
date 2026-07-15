from datetime import datetime, timezone
from typing import Any
from loguru import logger
import importlib

# 动态加载带有中划线的模块
module_path = "skills.skill_libs.ask-doc-skill"
ask_doc_module = importlib.import_module(module_path)
AskDocSkill = ask_doc_module.AskDocSkill
from agent.common import ContextMemory # 确保这里包含了基础定义
from agent.memory import MemoryService
from core.dictionary import PacketType
from services.basic.agent_service import AgentService


class DifyService:
    """Dify 接口的检索服务适配器：复用系统原生 Skill 链。"""

    def __init__(self):
        # 技能与内存实例
        self.doc_skill = AskDocSkill()
        self.memory_service = MemoryService()
        self.agent_service = AgentService()

    async def search(
        self,
        agent_id: int,
        question: str,
        session_id: str,
        security_level: int,
        user_id: str = "dify_system",
        tags: list[str] | None = None
    ) -> dict[str, list[dict]]:
        """
        Dify 代理交互接口：调用 AskDocSkill 检索知识。
        """
        question = str(question) if question else ""
        logger.info(f"Dify RAG 请求 | 会话: {session_id} | 长度: {len(question)}")
        model_params = await self.agent_service.get_agent_model_params(agent_id)
        llm_model = model_params.llm_model
        # 1. 持久化层准备
        await self.memory_service.ensure_session_exists(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            question=question
        )
        user_profile = await self.memory_service.get_user_profile(user_id)
        prepared = await self.memory_service.prepare_context_and_rewrite(
            user_id=user_id,
            session_id=session_id,
            raw_question=question,
            llm_model=llm_model,
            user_profile=user_profile
        )

        # 2. 构造上下文 (只填充差异化部分，其余使用默认值)
        # 建议在 ContextMemory 类定义里加一个类方法 from_defaults
        context = self._build_context(agent_id, session_id, question, user_id, security_level, tags)

        records = []
        try:
            # 3. 异步流式处理 (即便 Dify 是同步返回，我们内部依然走流式以复用逻辑)
            async for packet in self.doc_skill.run_stream(context=context):
                p_type = packet.get("type")
                
                # 命中结果包 (兼容 DOC_RESULTS 和可能的 data 载荷)
                if p_type in [PacketType.DOC_RESULTS, "data"]:
                    content = packet.get("content", [])
                    
                    # 如果数据存在于 data.raw_value 中也进行兼容
                    raw_docs = content if isinstance(content, list) else content.get("raw_value", [])
                    
                    if raw_docs:
                        records = self._convert_to_dify_format(raw_docs)
                        # 一旦拿到结果，对于 Dify 这种一次性请求可以提前结束循环
                        break
                
                elif p_type == PacketType.ERROR:
                    logger.error(f"AskDocSkill 内部执行报错: {packet.get('content')}")

        except Exception as e:
            logger.exception(f"Dify 复用 AskDocSkill 深度异常: {e}")
            return {"records": []}

        return {"records": records}

    def _build_context(self, agent_id: int, session_id: str, question: str, user_id: str, security_level: int, tags: list[str] | None) -> ContextMemory:
        """构建标准化的执行上下文"""
        return {
            "user_id": user_id,
            "session_id": session_id,
            "agent_id": agent_id,
            "question": question,
            "standalone_query": question,
            "search_keywords": "",
            "llm_model": "",
            "embedding_model": "",
            "security_level": security_level,
            "tags": tags or [],
            "intent_context": {},
            "runtime_plan": None,
            "current_step_index": 0,
            "current_execution": None,
            "execution_history": [],
            "variables": {
                "extracted_entities": []
            },
            "doc_results": [],
            "sql_results": [],
            "graph_results": [],

            "session_state": {},
            "blocks": [],
            "temp": {}    
        }


    def _convert_to_dify_format(self, skill_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """标准化 Dify 外部数据源格式"""
        return [
            {
                "metadata": {
                    "source": doc.get("file_name", "NexusCube_KB"),
                    "page": doc.get("page_num"),
                    "score": round(doc.get("score", 0), 4),
                    "chunk_id": doc.get("chunk_id")
                },
                "title": doc.get("file_name", "Untitled"),
                "content": doc.get("content", "")
            }
            for doc in skill_docs if isinstance(doc, dict)
        ]