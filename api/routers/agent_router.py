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
    description="智能体聊天接口",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_chat(request: AgentChatForm) -> SuccessQueryResponse | ErrorResponse:
    try:
        r = await agent_chat(request)

        logger.debug(f"聊天结果: {r}")

        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="",
            data=r
        )
        
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"聊天失败：{str(e)}"
        )

@router.get(
    "/stream",
    description="获取流式响应",
    response_class=StreamingResponse,
    response_model=None
)
async def handle_agent_stream_chat(session_id: str) -> StreamingResponse | ErrorResponse:
    generator = agent_stream_chat(session_id)
    if generator is None:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="智能体无响应"
        )
    
    async def convert_to_bytes():
        async for chunk in generator: # type: ignore
            if isinstance(chunk, dict):
                data = json.dumps(chunk)
            else:
                data = str(chunk)
            # 标准 SSE 事件流格式
            yield data
    
    return StreamingResponse(
        convert_to_bytes(),
        media_type="text/event-stream"
    )

@router.post(
    "/feedback",
    description="反馈智能体",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_feedback(form: AgentChatFeedbackForm):
    r = await agent_feedback(form)
    if r:
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="反馈成功"
        )
    else:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message="反馈失败"
        )

@router.get(
    "/session/get",
    description="在登录智能体时获取会话信息",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_get_session(session_id: str):
    try:
        r = await agent_get_session(session_id)

        return SuccessQueryResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="会话信息获取成功",
            data=r
            )
   
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"会话信息获取失败: {str(e)}"
        )
    
@router.delete(
    "/session/remove",
    description="删除会话信息",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_del_session(session_id: str):
    try:
        if await agent_del_session(session_id):
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="会话信息删除成功"
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="会话信息删除失败"
            )
            
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"会话信息删除失败: {str(e)}"
        )
    

@router.delete(
    "/remove",
    description="删除智能体",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_del_agent(agent_id: int, del_prompt: int = 0) -> SuccessResponse | ErrorResponse:
    delprompt = True if del_prompt == 1 else False
    try:
        if await del_agent(agent_id=agent_id, del_prompt=delprompt):
            return SuccessResponse(
                code=status.HTTP_200_OK,
                success=True,
                message="智能体删除成功"
                )
        else:
            return ErrorResponse(
                code=status.HTTP_400_BAD_REQUEST,
                success=False,
                message="智能体删除失败"
            )
            
    except Exception as e:
        return ErrorResponse(
            code=status.HTTP_400_BAD_REQUEST,
            success=False,
            message=f"智能体删除失败: {str(e)}"
        )