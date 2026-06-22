# api/routers/ops_router.py

from fastapi import APIRouter, Depends, status, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from api.controllers.ops_controller import ops_controller
from api.schemas.ops_schema import CreateInstanceRequest, UpdateInstanceRequest, OpsChatRequest
from api.schemas.base_response import SuccessResponse
from core.auth.shortcuts import UserAuth

router = APIRouter(prefix="/ops", tags=["OPS - 数据库智能运维管理"])


@router.post(
    "/instance/register",
    response_model=SuccessResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册新数据库实例",
    description="DBA 控制面或自动化脚本调用, 将一个新的数据库实例纳入运维监控体系。"
)
async def register_instance(auth: UserAuth, payload: CreateInstanceRequest):
    """注册新数据库实例"""
    return await ops_controller.create_instance(payload)


@router.get(
    "/instances",
    summary="【UI联动】获取当前 Agent 托管的所有物理实例列表",
    description="用于前端界面左侧树或顶部下拉框的动态渲染, 让用户先选定实例再提问。",
    response_model=SuccessResponse
)
async def get_agent_instances(
    auth: UserAuth,
    agent_id: int = Query(..., description="智能体 ID (int)")
):
    """资产拉取接口"""
    return await ops_controller.get_accessible_instances(agent_id=agent_id)


@router.get(
    "/instances/all",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="获取所有运维数据库实例列表",
    description="返回CMDB中所有活跃的数据库实例, 用于Agent表单中的实例选择器。"
)
async def get_all_instances(auth: UserAuth):
    """获取所有活跃的运维实例（不区分Agent绑定关系）"""
    return await ops_controller.get_all_instances()


# ==============================================================================
#  单个实例 CRUD
# ==============================================================================

@router.get(
    "/instances/{instance_id}",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="获取单个运维实例详情"
)
async def get_instance_detail(auth: UserAuth, instance_id: str):
    """获取指定运维实例的完整信息（不含密码）。"""
    return await ops_controller.get_instance_detail(instance_id)


@router.put(
    "/instances/{instance_id}",
    response_model=SuccessResponse,
    status_code=status.HTTP_200_OK,
    summary="更新运维实例信息"
)
async def update_instance(auth: UserAuth, instance_id: str, payload: UpdateInstanceRequest):
    """更新指定运维实例的配置信息。所有字段均为可选, 仅更新传入的非空字段。"""
    return await ops_controller.update_instance(instance_id, payload)


@router.delete(
    "/instances/{instance_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除运维实例"
)
async def delete_instance(auth: UserAuth,instance_id: str):
    """删除指定的运维实例。此操作不可逆。"""
    await ops_controller.delete_instance(instance_id)


@router.post(
    "/chat",
    summary="【流式控制面】发送定靶自愈/探针指令 (UI强管控版)",
    description="用户在前端页面必须先选中一个具体实例, 然后输入自然语言, 点击发送。",
    response_class=StreamingResponse
)
async def ops_agent_chat_stream(
    auth: UserAuth,
    request: OpsChatRequest,
    background_tasks: BackgroundTasks
):
    """流式自愈接口端点"""
    return await ops_controller.stream_chat(
        request=request,
        background_tasks=background_tasks
    )
