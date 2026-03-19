from datetime import datetime
import uuid
from typing import Any
from fastapi import Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from services.agent.agent_service import AgentService
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm, DifySearchForm
from api.schemas.base_response import SuccessResponse
from services.agent.memory import MemoryService
from services.agent.dify_service import DifyService
from core.exceptions import *


class AgentController:
    def __init__(self):
        self.agent_service = AgentService()
        self.dify_service = DifyService()

    async def feedback(self, form: AgentChatFeedbackForm) -> SuccessResponse:
        """Submits user feedback for a chat record."""
        await self.agent_service.feedback(form.memory_id, form.feedback)
        return SuccessResponse(message="Feedback submitted successfully")
        
    async def get_session_chat_records(self, session_id: str) -> SuccessResponse:
        """Retrieves history for a specific session."""
        records = await self.agent_service.get_session_history(session_id)
        return SuccessResponse(data=records, message="Session history retrieved")

    async def remove_session(self, session_id: str) -> SuccessResponse:
        """Deletes a chat session."""
        await self.agent_service.remove_session(session_id)
        return SuccessResponse(message="Session deleted")
        
    async def remove_agent(self, agent_id: int, del_prompt: bool = False) -> SuccessResponse:
        """Removes an agent and its configurations."""
        await self.agent_service.remove_agent(agent_id, del_prompt)
        return SuccessResponse(message=f"Agent {agent_id} deleted")

    async def agent_chat_nonstream(self, form: AgentChatForm) -> SuccessResponse:
        """
        Agent interaction (Non-streaming).
        Returns the formatted dictionary with answer, embedding, and timestamps.
        """
        result = await self.agent_service.non_stream_chat(
            session_id=form.session_id,
            user_id = form.by,
            agent_id=form.agent_id,
            question=form.question,
            security_level=form.security_level,
            tags=form.tags or []
        )
        return SuccessResponse(data=result, message="Agent chat successful")
    
    async def agent_chat_stream(self, form: AgentChatForm, background_tasks: BackgroundTasks) -> StreamingResponse:
        """
        Agent interaction (Streaming).
        Uses BackgroundTasks to handle database persistence after the stream starts.
        """
        return await self.agent_service.stream_chat(
            background_tasks=background_tasks,
            session_id=form.session_id,
            user_id = form.by,
            agent_id=form.agent_id,
            question=form.question,
            security_level=form.security_level,
            tags=form.tags or []
        )
    
    async def dify_search(self, form: DifySearchForm, background_tasks: BackgroundTasks) -> dict:
        """
        Dify search.
        Uses BackgroundTasks to handle database persistence after the stream starts.
        """
        agent_id = int(form.knowledge_id)
        session_id = uuid.uuid4().hex
        
        return await self.dify_service.search(
                    agent_id=agent_id, 
                    question=form.query, 
                    session_id=session_id,
                    background_tasks=background_tasks
                )

# initialize the controller
agent_controller = AgentController()