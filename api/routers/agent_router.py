import uuid
import json
from loguru import logger
from datetime import datetime, timezone
from typing import Any
from fastapi import APIRouter, status, Depends, HTTPException
from fastapi import Request, BackgroundTasks
from fastapi.responses import StreamingResponse

from api.schemas.agent_schema import *
from core.auth.shortcuts import *
from api.controllers.agent_controller import agent_controller
from api.schemas.base_response import *

router = APIRouter(prefix="/agent", tags=["Agent Chat"])

@router.post(
    "/chat",
    summary="Agent Chat (Streaming)",
    response_class=StreamingResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_chat(form: AgentChatForm, auth: UserAuth, background_tasks: BackgroundTasks):
    """
    Asynchronous streaming chat interface for the Agent.
    
    Args:
    - **form**: Agent chat request form
        - session_id: Unique session identifier
        - by: User ID making the request
        - agent_id: ID of the targeted agent
        - security_level: Access clearance level
        - question: User input text
        - tags: Optional list of categories for filtering
    - **auth**: User authentication context
    - **background_tasks**: FastAPI background task manager for persistence
    
    Returns:
    - **StreamingResponse**: SSE stream containing LLM chunks and references.
    """
    logger.info(f"Received streaming chat request for agent {form.agent_id} by user {form.by}")
    return await agent_controller.agent_chat_stream(form, background_tasks)


@router.post(
    "/feedback",
    summary="Submit Agent Response Feedback",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_feedback(form: AgentChatFeedbackForm, auth: UserAuth):
    """
    Submit user feedback (like/dislike) for a specific agent response.
    
    Args:
    - **form**: Feedback data including record ID and feedback value (1, 0, -1).
    """
    logger.info(f"Processing feedback for chat record ID: {form.chat_record_id}")
    return await agent_controller.feedback(form)


@router.get(
    "/session/get",
    summary="Retrieve Chat Session History",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_get_session(session_id: str, auth: UserAuth):
    """
    Fetches historical chat records for a specific session.
    """
    logger.info(f"Fetching chat history for session: {session_id}")
    return await agent_controller.get_session_chat_records(session_id)


@router.delete(
    "/session/remove",
    summary="Delete Chat Session",
    response_model=SuccessResponse
)
async def handle_agent_del_session(session_id: str, auth: UserAuth):
    """
    Removes an entire chat session and its associated history.
    """
    logger.warning(f"Request to delete session: {session_id}")
    return await agent_controller.remove_session(session_id)


@router.delete(
    "/remove",
    summary="Delete Agent",
    response_model=SuccessResponse
)
async def handle_del_agent(auth: UserAuth, agent_id: int, del_prompt: int = 0):
    """
    Deletes an agent configuration and optionally its prompt templates.
    """
    logger.warning(f"Request to remove agent {agent_id}. Delete prompt: {del_prompt == 1}")
    return await agent_controller.remove_agent(agent_id, del_prompt == 1)


@router.post(
    "/dify/retrieval",
    summary="Dify Retrieval Adapter",
    response_model=dict
)
async def handle_agent_retrieval(auth: ServiceAuth, form: DifySearchForm, background_tasks: BackgroundTasks):
    """
    Adapter interface for Dify external knowledge retrieval.
    Compatible with Dify's external knowledge base API.
    """
    logger.info(f"Dify retrieval request received for knowledge_id: {form.knowledge_id}")
    return await agent_controller.dify_search(form, background_tasks)


@router.post(
    "/nonstream",
    summary="Agent Chat (Non-Streaming)",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK
)
async def handle_non_stream_chat(auth: ServiceAuth, form: AgentChatForm):
    """
    Synchronous chat interface that returns the full response once completed.
    
    Returns:
    - **SuccessResponse**: Data includes 'answer', 'qa_embedding', and 'references'.
    """
    logger.info(f"Processing non-stream chat for agent {form.agent_id}")
    result = await agent_controller.agent_chat_nonstream(form)
    return SuccessResponse(data=result, message="Chat completed successfully")