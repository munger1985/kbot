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
    ### Description
    Asynchronous streaming interface for AI Agent interactions using Server-Sent Events (SSE).

    ---
    ### Request Body (`AgentChatForm`)
    - **session_id** (`str`): Unique identifier for the chat session.
    - **agent_id** (`int`): The ID of the specific agent configuration to use.
    - **question** (`str`): The user's input text.
    - **security_level** (`int`): Data access clearance level for RAG.
    - **tags** (`list[str]`, optional): Categories used to filter knowledge base retrieval.

    ### Returns
    - **StreamingResponse**: A continuous stream of data chunks. Each chunk is typically a JSON object containing `answer_piece` or `references`.

    > **Note**: Chat history persistence is handled via `background_tasks` to ensure low latency.
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
    ### Description
    Submit user sentiment (Like/Dislike) regarding a specific agent response to improve model performance.

    ---
    ### Parameters
    - **chat_record_id** (`str`): The unique ID of the specific chat message.
    - **feedback_value** (`int`): 
        - `1`: Positive (Like)
        - `0`: Neutral/Reset
        - `-1`: Negative (Dislike)
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
    ### Description
    Retrieves a chronological list of all chat records associated with a specific `session_id`.

    ### Use Case
    Useful for restoring chat UI state when a user reloads the page or switches conversations.
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
    ### Description
    Permanently deletes a chat session and all its nested message history from the database.
    
    > **Warning**: This action is irreversible.
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
    ### Description
    Deletes an agent's settings and metadata.

    ---
    ### Parameters
    - **agent_id** (`int`): ID of the agent to remove.
    - **del_prompt** (`int`): Set to `1` to also delete the associated prompt templates, otherwise `0`.
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
    ### Description
    A specialized adapter endpoint designed to bridge this system with **Dify's External Knowledge Base** API.

    ---
    ### Protocol
    - **Auth**: Requires `ServiceAuth` (typically an API Key).
    - **Compatibility**: Adheres to the standard Dify retrieval request/response schema.
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
    ### Description
    A standard synchronous chat interface. The HTTP connection remains open until the full LLM response is generated.

    ---
    ### Returns
    A `SuccessResponse` where `data` contains:
    - `answer`: The full text response.
    - `references`: Source documents used for the answer.
    - `qa_embedding`: The vector representation of the interaction.

    ### Recommendation
    Use this for service-to-service calls or small responses where streaming UI is not required.
    """
    logger.info(f"Processing non-stream chat for agent {form.agent_id}")
    result = await agent_controller.agent_chat_nonstream(form)
    return SuccessResponse(data=result, message="Chat completed successfully")