import json
from loguru import logger
from fastapi import APIRouter, status, Depends
from fastapi.responses import StreamingResponse
from api.schemas.agent_schema import AgentChatForm, AgentChatFeedbackForm
from api.controllers.security_controller import AuthController
from api.controllers.agent_controller import *
from api.schemas.base_response import *

router = APIRouter(prefix="/agent", tags=["Agent Chat"])

@router.post(
    "/chat",
    summary="智能体聊天",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_chat(form: AgentChatForm) -> SuccessQueryResponse | ErrorResponse:
    """
    智能体聊天接口
    
    Args:
    - **form**: 智能体聊天表单
    ```
        session_id: str = Field(..., description="会话ID")
        by: str = Field(..., description="请求用户ID")
        agent_id: int = Field(..., description="智能体ID")
        security_level: int = Field(0, description="安全级别")
        request_time: str = Field(..., description="请求时间")
        question: str = Field(..., description="问题")
        tags: list[str] | None = Field(None, description="标签")
    ```
    
    Returns:
    - **SuccessQueryResponse**: 成功查询响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
        data: dict | list[dict] = Field(..., description="响应返回的数据")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```

    """
    try:
        r = await agent_chat(form)

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
    summary="智能体聊天流式响应",
    response_class=StreamingResponse,
    response_model=None
)
async def handle_agent_stream_chat(session_id: str) -> StreamingResponse | ErrorResponse:
    """
    智能体聊天流式响应接口
    
    Args:
    - **session_id**: 会话ID
    
    Returns:
    - **StreamingResponse**: 流式响应
    ```
    data: {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{
            "delta": {"content": content},
            "index": 0,
            "finish_reason": None
        }]
    }
    data: [DONE]
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
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
    summary="智能体回答结果反馈接口",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_feedback(form: AgentChatFeedbackForm):
    """
    智能体回答结果反馈接口
    
    Args:
    - **form**: 智能体聊天获取反馈表单模型
    ```
        session_id: str = Field(..., description="会话ID")
        question_index: int = Field(..., description="问题索引")
        feedback: int = Field(..., description="问题反馈，0：不反馈，1：赞同，-1：不赞同")
    ```
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
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
    summary="登录智能体时获取会话信息",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_get_session(session_id: str):
    """
    登录智能体时获取会话信息
    
    Args:
    - **session_id**: 会话ID
    
    Returns:
    - **SuccessQueryResponse**: 成功查询响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
        data: dict | list[dict] = Field(..., description="响应返回的数据")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
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
    summary="删除聊天会话信息",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_agent_del_session(session_id: str) -> SuccessResponse | ErrorResponse:
    """
    删除聊天会话信息
    
    Args:
    - **session_id**: 会话ID
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
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
    summary="删除智能体",
    dependencies=[Depends(AuthController.get_current_accessor)]
)
async def handle_del_agent(agent_id: int, del_prompt: int = 0) -> SuccessResponse | ErrorResponse:
    """
    删除智能体
    
    Args:
    - **agent_id**: 智能体ID
    - **del_prompt**: 是否删除提示词，0：不删除，1：删除
    
    Returns:
    - **SuccessResponse**: 成功响应
    ```
        code: int = Field(status.HTTP_200_OK, description="响应状态码")
        message: str = Field("Success", description="返回的响应信息")
        success: bool = Field(True, description="请求响应状态")
    ```
    - **ErrorResponse**: 失败响应
    ```
        code: int = Field(status.HTTP_400_BAD_REQUEST, description="响应状态码")
        message: str = Field("Error", description="返回的响应信息")
        success: bool = Field(False, description="请求响应状态")
    ```
    """
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