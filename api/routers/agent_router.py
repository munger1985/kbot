import uuid
from loguru import logger
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
    summary="智能体聊天"
)
async def handle_agent_chat(form: AgentChatForm, auth: UserAuth) -> SuccessQueryResponse:
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
        deep_mind: int = Field(0, description="是否使用深度思考, 0：不使用，1：使用")
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

    r = await agent_controller.agent_search(form)

    logger.debug(f"聊天结果: {r}")

    return SuccessQueryResponse(
        code=status.HTTP_200_OK,
        success=True,
        message="",
        data=r
    )
   

@router.get(
    "/stream",
    summary="智能体聊天流式响应",
    response_class=StreamingResponse,
    response_model=None
)
async def handle_agent_stream_chat(
    request: Request,           # FastAPI 自动注入
    background_tasks: BackgroundTasks,  # FastAPI 自动注入
    session_id: str             # 前端传入的查询参数
) -> StreamingResponse:
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

    return await agent_controller.agent_chat_stream(
        request=request,
        background_tasks=background_tasks,
        session_id=session_id
    )


@router.post(
    "/feedback",
    summary="智能体回答结果反馈接口"
)
async def handle_agent_feedback(form: AgentChatFeedbackForm, auth: UserAuth) -> SuccessResponse:
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
    r = await agent_controller.agent_feedback(form)
    if r:
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="反馈成功"
        )
    else:
        logger.error("反馈失败")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="反馈失败"
        )

@router.get(
    "/session/get",
    summary="登录智能体时获取会话信息"
)
async def handle_agent_get_session(session_id: str, auth: UserAuth):
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

    r = await agent_controller.agent_get_session(session_id)
    
    return SuccessQueryResponse(
        code=status.HTTP_200_OK,
        success=True,
        message="会话信息获取成功",
        data=r or {}
        )
    
    
@router.delete(
    "/session/remove",
    summary="删除聊天会话信息"
)
async def handle_agent_del_session(session_id: str, auth: UserAuth) -> SuccessResponse:
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

    if await agent_controller.agent_del_session(session_id):
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="会话信息删除成功"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="会话信息删除失败"
        )

    

@router.delete(
    "/remove",
    summary="删除智能体"
)
async def handle_del_agent(auth: UserAuth, agent_id: int, del_prompt: int = 0) -> SuccessResponse:
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

    if await agent_controller.del_agent(agent_id=agent_id, del_prompt=delprompt):
        return SuccessResponse(
            code=status.HTTP_200_OK,
            success=True,
            message="智能体删除成功"
            )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="智能体删除失败"
        )
        
    
@router.post(
    "/dify/retrieval",
    summary="智能体Dify检索"
)
async def handle_agent_retrieval(auth: ServiceAuth, form: AgentChatDifyForm) -> dict:
    """
    智能体 Dify 检索接口
    参考：https://docs.dify.ai/en/guides/knowledge-base/external-knowledge-api
    
    Args:
    - **knowledge_id**: Kbot 智能体 ID
    - **query**: 查询文本
    - **retrieval_setting**: 检索设置
    - **metadata_condition**: 元数据条件
    
    Returns:
    - **records**: 检索结果
    """

    agent_id = int(form.knowledge_id)
    session_id = uuid.uuid4().hex
    return await agent_controller.agent_search_dify(
        agent_id=agent_id, 
        question=form.query, 
        session_id=session_id,
        override_question=form.retrieval_setting.get("override_question", False)
    )


@router.post(
    "/nonstream",
    summary="智能体聊天非流式响应",
    response_model=None
)
async def handle_non_stream_chat(auth: ServiceAuth, form: AgentChatForm) -> SuccessQueryResponse:
    """
    智能体聊天接口(非流式)
    
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
        deep_mind: int = Field(0, description="是否使用深度思考, 0：不使用，1：使用")
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

    r = await agent_controller.agent_chat_nonstream(form)

    return SuccessQueryResponse(
        code=status.HTTP_200_OK,
        success=True,
        message="",
        data=r
    )
    