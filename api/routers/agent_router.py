import json
from loguru import logger
from fastapi import APIRouter, status, Depends
from fastapi.responses import StreamingResponse
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm
from api.controllers.security_controller import AuthController
from api.controllers.agent_controller import *
from api.schemas.base_response import *

router = APIRouter(
    prefix="/agent",
    tags=["Agent Chat"]
)

@router.post(
    "/chat",
    description="Chat with the agent. 和智能体聊天",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_chat(request: AgentChatForm) -> SuccessQueryResponse | ErrorResponse:
    try:
        r = await agent_chat(request)

        logger.debug(f"Chat result: {r}")

        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="Chat successfully.",
            data=r
        )
        
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"Chat failed. {str(e)}"
        )

@router.get(
    "/stream",
    description="Get the stream response. 获取流式响应",
    dependencies=[Depends(AuthController.get_current_accessor)],
    response_class=StreamingResponse,
    response_model=None
)
async def handle_agent_stream_chat(session_id: str) -> StreamingResponse | ErrorResponse:
    generator = agent_stream_chat(session_id)
    if generator is None:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
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
    description="Feedback the agent. 反馈智能体",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_feedback(form: AgentChatFeedbackForm):
    r = await agent_feedback(form)
    if r:
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="Feedback successfully."
        )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="Feedback failed."
        )

@router.get(
    "/session/get",
    description="Get the session when login. 在登录智能体时获取session",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_get_session(session_id: str):
    try:
        r = await agent_get_session(session_id)

        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="Session get successfully.",
            data=r
            )
   
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"Session get failed. {str(e)}"
        )
    
@router.delete(
    "/session/remove",
    description="Remove the session. 删除session",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_del_session(session_id: str):
    try:
        if await agent_del_session(session_id):
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="Session successfully removed."
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="Session failed to remove."
            )
            
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"Session failed to remove. {str(e)}"
        )
    

@router.delete(
    "/remove",
    description="Remove the agent. 删除智能体",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_del_agent(agent_id: int, del_prompt: bool = False) -> SuccessResponse | ErrorResponse:
    try:
        if await del_agent(agent_id=agent_id, del_prompt=del_prompt):
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="Agent successfully removed."
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="Agent failed to remove."
            )
            
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"Agent failed to remove. {str(e)}"
        )