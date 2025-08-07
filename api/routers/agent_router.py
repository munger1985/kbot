import json
from loguru import logger
from fastapi import APIRouter, status
from fastapi.responses import StreamingResponse
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm
from api.controllers.agent_controller import (
    agent_chat, 
    agent_feedback, 
    agent_stream_chat, 
    agent_get_session,
    agent_del_session
)
from api.schemas.agent_response import (
    SuccessResponse,
    ErrorResponse,
    SuccessQueryResponse
)

router = APIRouter(
    prefix="/agent",
    tags=["Agent Chat"]
)

@router.post(
    "/chat",
    summary="Chat with the agent. 和智能体聊天",
    description="Chat with the agent. 和智能体聊天",
    response_model=SuccessQueryResponse | ErrorResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_chat(request: AgentChatForm) -> SuccessQueryResponse | ErrorResponse:
    # Parse and validate as form model
    form = AgentChatForm(
        session_id=request.session_id,
        by=request.by,
        agent_id=request.agent_id,
        security_level=request.security_level,
        request_time=request.request_time,
        question_index=request.question_index,
        question=request.question,
    )
    try:
        r = await agent_chat(form)

        logger.debug(f"Chat result: {r}")

        return SuccessQueryResponse(
            code=200,
            success=True,
            message="Chat successfully.",
            data=r
        )
        
    except Exception as e:
        return ErrorResponse(
            code=400,
            success=False,
            message=f"Chat failed. {str(e)}"
        )

@router.get(
    "/stream",
    summary="Get the stream response. 获取流式响应",
    description="Get the stream response. 获取流式响应",
    status_code=status.HTTP_200_OK,
    response_model=None,
    response_class=StreamingResponse
)
async def handle_agent_stream_chat(session_id: str) -> StreamingResponse | ErrorResponse:
    generator = agent_stream_chat(session_id)
    if generator is None:
        return ErrorResponse(
            code=400,
            success=False,
            message="Agent chat has no response."
        )
    
    async def convert_to_bytes():
        async for chunk in generator: # type: ignore
            if isinstance(chunk, dict):
                data = json.dumps(chunk)
            else:
                data = str(chunk)
            # Format as proper SSE with data: prefix and double newline
            yield data
    
    return StreamingResponse(
        convert_to_bytes(),
        media_type="text/event-stream"
    )

@router.post(
    "/feedback",
    summary="Feedback the agent. 反馈智能体",
    description="Feedback the agent. 反馈智能体",
    response_model=SuccessResponse | ErrorResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_feedback(form: AgentChatFeedbackForm) -> SuccessResponse | ErrorResponse:
    r = await agent_feedback(form)
    if r:
        return SuccessResponse(
            code=200,
            success=True,
            message="Feedback successfully."
        )
    else:
        return ErrorResponse(
            code=400,
            success=False,
            message="Feedback failed."
        )

@router.get(
    "/session/get",
    summary="Get the session when login. 在登录智能体时获取session",
    description="Get the session when login. 在登录智能体时获取session",
    response_model=SuccessQueryResponse | ErrorResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_get_session(session_id: str) -> SuccessQueryResponse | ErrorResponse:
    try:
        r = await agent_get_session(session_id)
        if r:
            return SuccessQueryResponse(
                code=200,
                success=True,
                message="Session get successfully.",
                data=r
                )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message="Session get failed."
            )
            
    except Exception as e:
        return ErrorResponse(
            code=400,
            success=False,
            message=f"Session get failed. {str(e)}"
        )
    
@router.get(
    "/session/remove",
    summary="Remove the session. 删除session",
    description="Remove the session. 删除session",
    response_model=SuccessResponse | ErrorResponse,
    status_code=status.HTTP_200_OK
)
async def handle_agent_del_session(session_id: str) -> SuccessResponse | ErrorResponse:
    try:
        if await agent_del_session(session_id):
            return SuccessResponse(
                code=200,
                success=True,
                message="Session successfully removed."
                )
        else:
            return ErrorResponse(
                code=400,
                success=False,
                message="Session failed to remove."
            )
            
    except Exception as e:
        return ErrorResponse(
            code=400,
            success=False,
            message=f"Session get failed. {str(e)}"
        )