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

class KBAskTool(MCPTool):
    """知识库问答工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.KB_SEARCH,
            tool_name="kbot_ask",
            description="搜索知识库获取相关信息并回答问题的工具"
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
        tags: list = [],
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
        result = await self.chat_service.non_stream_chat(
            background_tasks=internal_tasks, 
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            question=question,
            security_level=self.security_level, 
            tags=tags
        )
        answer = result.get("answer")
        if not answer:
            raise ValueError(f"Answer is empty, please check the input parameters")

        return answer
        

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "问题"
                },
                "agent_id": {
                    "type": "integer",
                    "description": "执行搜索知识库的Agent ID"
                }
            },
            "required": ["query", "agent_id"]
        }