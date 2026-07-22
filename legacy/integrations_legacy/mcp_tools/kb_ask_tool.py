from datetime import datetime, timezone
from loguru import logger
import uuid
from typing import Any
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse
from .base import MCPTool
from platform_core.dictionary import MCPToolType

from agent.orchestrator import RootOrchestrator
from agent.agent import RootAgent
from agent.memory import MemoryService
from services.kb import ModelParams

class KBAskTool(MCPTool):
    """知识库问答工具"""
    
    def __init__(self):
        super().__init__(
            tool_type=MCPToolType.AGENT_CALL,
            tool_name="kbot_ask",
            description="知识库深度问答工具（RAG）。 专门用于回答涉及特定知识库的问题。它会自动检索相关文档，并结合检索到的背景信息生成准确、完整的答案。当你需要直接回答用户的咨询（如询问政策、操作流程或技术细节）而不仅仅是列出资料时，请优先使用此工具。该工具能有效减少模型幻觉，确保回答基于事实。"
        )
        self.orchestrator = RootOrchestrator()
        self.security_level = 9  # Level 9 bypasses security checks
        self.agent = RootAgent()
        self.memory_service = MemoryService()
    
    async def execute(
        self,
        agent_id: int,
        question: str,
        session_id: str | None = None,
        tags: list = [],
        user_id: str = "mcp_call"
    ) -> StreamingResponse:
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
        return await self.agent.chat(
            background_tasks=internal_tasks, 
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            query=question,
            security_level=self.security_level, 
            tags=tags
        )
        

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "完整的用户问题。该工具将基于此问题进行语义检索并直接生成最终回答。"
                },
                "agent_id": {
                    "type": "integer",
                    "description": "知识库关联的 Agent ID。请根据当前的对话场景或默认配置提供。"
                }
            },
            "required": ["query", "agent_id"]
        }