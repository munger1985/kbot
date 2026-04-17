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
            description="知识库深度问答工具（RAG）。 专门用于回答涉及特定知识库的问题。它会自动检索相关文档，并结合检索到的背景信息生成准确、完整的答案。当你需要直接回答用户的咨询（如询问政策、操作流程或技术细节）而不仅仅是列出资料时，请优先使用此工具。该工具能有效减少模型幻觉，确保回答基于事实。"
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
                    "description": "针对知识库的搜索查询语句，建议使用完整的疑问句或核心术语以获得更好的检索效果。"
                },
                "agent_id": {
                    "type": "integer",
                    "description": "知识库关联的 Agent ID。请根据当前的对话场景或默认配置提供。"
                }
            },
            "required": ["query", "agent_id"]
        }