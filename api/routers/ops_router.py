# api/routers/ops_router.py

from fastapi import APIRouter, Depends, status, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from api.controllers.ops_controller import ops_controller
from api.schemas.ops_schema import CreateInstanceRequest, UpdateInstanceRequest, OpsChatRequest, OpsResumeRequest, OpsApproveRequest, AlertWebhookRequest
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


@router.post(
    "/chat/resume",
    summary="【HITL】提交用户采集的数据并恢复诊断",
    description="用户在收到 WAIT_FOR_USER 事件后，执行 SQL 并将结果通过此接口提交，Agent 从断点恢复分析。",
    response_class=StreamingResponse
)
async def resume_ops_chat(
    auth: UserAuth,
    request: OpsResumeRequest,
    background_tasks: BackgroundTasks
):
    """HITL 恢复执行接口"""
    return await ops_controller.resume_chat(
        request=request,
        background_tasks=background_tasks
    )


@router.post(
    "/chat/cancel-pending",
    summary="【HITL】取消当前挂起的诊断请求",
    description="用户在等待 Agent 回复期间主动放弃等待，取消挂起的诊断。"
)
async def cancel_pending_request(
    auth: UserAuth,
    request_id: str = Query(..., description="挂起请求 ID")
):
    """取消挂起的 HITL 请求"""
    return await ops_controller.cancel_pending(request_id)


@router.post(
    "/chat/alert-webhook",
    summary="【告警驱动】接收监控系统告警回调并自动触发 AIOps 诊断",
    description="Prometheus AlertManager / Zabbix Action 回调入口。解析告警→自动诊断→流式输出。",
    response_class=StreamingResponse,
)
async def ops_alert_webhook_chat(
    request: AlertWebhookRequest,
    background_tasks: BackgroundTasks,
):
    """告警 Webhook 接口"""
    return await ops_controller.alert_webhook_chat(
        request=request,
        background_tasks=background_tasks,
    )


@router.post(
    "/chat/confirm-action",
    summary="【逐命令确认】用户确认/取消单条变更命令",
    description="用户在收到 CONFIRM_ACTION 事件后，对 SQL 逐条确认或跳过。",
    response_class=StreamingResponse,
)
async def confirm_ops_action(
    request: OpsConfirmActionRequest,
    background_tasks: BackgroundTasks,
):
    """逐命令确认接口"""
    return await ops_controller.confirm_action(request, background_tasks)


@router.post(
    "/chat/approve",
    summary="【HITL 审批】用户对高危变更操作进行审批",
    description="用户在收到 REQUIRE_APPROVAL 事件后，确认风险并提交审批决定（批准或拒绝）。审批通过后 Agent 从断点恢复执行变更 SQL。",
    response_class=StreamingResponse
)
async def approve_ops_action(
    auth: UserAuth,
    request: OpsApproveRequest,
    background_tasks: BackgroundTasks
):
    """HITL 审批接口"""
    return await ops_controller.approve_action(
        request=request,
        background_tasks=background_tasks
    )


@router.delete(
    "/instances/{instance_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除运维实例"
)
async def delete_instance(auth: UserAuth, instance_id: str):
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
