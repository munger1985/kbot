from datetime import datetime, timezone
from loguru import logger
import uuid
from typing import Any
from fastapi import BackgroundTasks
from mcp_tools import MCPTool
from core.dictionary import MCPToolType

from services.agent.orchestrator import ChatOrchestrator
from services.agent.chat_service import ChatService
from services.memory import MemoryService
from services.agent.agent_params import ModelParams

class KBSearchTool(MCPTool):
    """知识库搜索工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.KB_SEARCH,
            tool_name="kbot_search",
            description="搜索知识库获取相关信息"
        )
        self.orchestrator = ChatOrchestrator()
        self.security_level = 9  # Level 9 bypasses security checks
        self.chat_service = ChatService()
        self.memory_service = MemoryService()
    
    async def execute(
        self,
        agent_id: int,
        question: str,
        session_id: str | None = None,
        tags: list | None = None,
        user_id: str = "mcp_call"
    ) -> list[dict]:
        """
        Agent interaction for MCP (Search + Record Persistence).
        """
        # Validate and normalize question parameter
        if not isinstance(question, str):
            logger.warning(f"Question is not a string, got type: {type(question).__name__}, converting to string")
            try:
                question = str(question)
            except:
                raise ValueError(f"Question is not a string and cannot be converted to string")

        if not session_id:
            session_id = uuid.uuid4().hex

        request_time = datetime.now(tz=timezone.utc)
        logger.info(f"Processing mcp request for session {session_id}, agent {agent_id}")
        internal_tasks = BackgroundTasks()

        # 确保会话存在
        await self.memory_service.ensure_session_exists(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            question=question
        )
        
        pipe_out = await self.orchestrator.run_pipeline(
            background_tasks=internal_tasks, user_id=user_id, 
            session_id=session_id, agent_id=agent_id, question=question, 
            security_level=self.security_level, tags=tags
        )
        
        enriched_refs = await self.chat_service._enrich_results_with_metadata(pipe_out['kb_results'])
        records = self._build_records(enriched_refs)

        # 获取模型名
        params = pipe_out.get('model_params')
        if params:
            model_params: ModelParams = params
            # 持久化：使用 MemoryService 的新闭环方法
            entry_id = uuid.uuid4().hex
            response_time = datetime.now(tz=timezone.utc)
            internal_tasks.add_task(
                self.memory_service.persist_and_reflect_memory,
                session_id=session_id,
                entry_id=entry_id,
                user_id=user_id,
                raw_question=question,
                answer="",
                model_params=model_params,
                prepared_data=pipe_out['prepared_data'],
                retrieved_chunks=enriched_refs,
                request_time=request_time,
                response_time=response_time
            )
        else:
            logger.warning(f"Model params are missing for session {session_id}, Agent {agent_id}, skip memory persist.")

        return records
        

    def _build_records(self, enriched_references: list[dict]) -> list[dict]:
        """
        Formats enriched references into record structure.
        Uses the output from _enrich_results_with_metadata.
        """
        records = []
        for idx, ref in enumerate(enriched_references):
            try:
                # Handle content that might be a list or other type
                content = ref.get("content")
                if not isinstance(content, str):
                    content = str(content) if content else ""

                record = {
                    "title": ref.get("file_name", "Unknown File"),
                    "chunk_num": ref.get("chunk_num"),
                    "content": content
                }
                records.append(record)
            except Exception as e:
                logger.error(f"Error building record at index {idx}: {e}, type: {type(e).__name__}, ref type: {type(ref).__name__}")
                raise
        return records

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询语句"
                },
                "agent_id": {
                    "type": "int",
                    "description": "执行搜索知识库任务的Agent ID"
                }
            },
            "required": ["query", "agent_id"]
        }