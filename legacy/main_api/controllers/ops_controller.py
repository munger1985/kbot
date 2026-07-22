# api/controllers/ops_controller.py

import uuid
from loguru import logger
from fastapi import BackgroundTasks
from fastapi.responses import StreamingResponse

from api.schemas.ops_schema import CreateInstanceRequest, UpdateInstanceRequest, OpsChatRequest, OpsResumeRequest, OpsApproveRequest, OpsConfirmActionRequest, AlertWebhookRequest
from api.schemas.base_response import SuccessResponse
from services.basic import OpsDBInstanceService, OpsAgentConfService


class OpsController:
    """
    AIOps 智能运维控制器
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str
    """

    def __init__(self):
        self.instance_service = OpsDBInstanceService()
        self.agent_conf_service = OpsAgentConfService()

    async def create_instance(self, payload: CreateInstanceRequest) -> SuccessResponse:
        """注册新数据库实例"""
        asset_data = payload.model_dump()
        # 如果未提供 instance_id, 自动生成 UUID v7 格式
        if not asset_data.get("instance_id"):
            asset_data["instance_id"] = str(uuid.uuid4())
        instance_id = await self.instance_service.register_new_instance(asset_data)
        return SuccessResponse(message="实例注册成功", data={"instance_id": instance_id})

    async def get_instance_detail(self, instance_id: str) -> SuccessResponse:
        """获取单个实例详情（不含密码）"""
        detail = await self.instance_service.get_instance_for_ui(instance_id)
        if not detail:
            return SuccessResponse(message="实例不存在", data=None)
        return SuccessResponse(message="查询成功", data=detail)

    async def update_instance(self, instance_id: str, payload: UpdateInstanceRequest) -> SuccessResponse:
        """更新运维实例信息"""
        data = payload.model_dump(exclude_none=True)
        await self.instance_service.update_instance(instance_id, data)
        return SuccessResponse(message="实例信息已更新", data={"instance_id": instance_id})

    async def delete_instance(self, instance_id: str) -> None:
        """删除运维实例"""
        await self.instance_service.delete_instance(instance_id)

    async def get_accessible_instances(self, agent_id: int) -> SuccessResponse:
        """获取指定 Agent 绑定的所有运维实例列表"""
        instances = await self.agent_conf_service.get_bound_instances_by_agent(agent_id)
        return SuccessResponse(message="查询成功", data=instances)

    async def get_all_instances(self) -> SuccessResponse:
        """获取所有活跃的运维实例列表"""
        instances = await self.instance_service.get_all_instances()
        return SuccessResponse(message="查询成功", data=instances)

    async def stream_chat(
        self,
        request: OpsChatRequest,
        background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        """流式自愈接口端点"""
        from agent.agent.ops_agent import OpsAgent
        ops_agent = OpsAgent()

        return await ops_agent.chat(
            background_tasks=background_tasks,
            user_id=request.user_id,
            agent_id=request.agent_id,
            instance_id=request.instance_id,
            query=request.query,
            session_id=request.session_id
        )

    async def alert_webhook_chat(
        self,
        request: AlertWebhookRequest,
        background_tasks: BackgroundTasks,
    ) -> StreamingResponse:
        """告警 Webhook 驱动自愈 — 解析告警并自动触发诊断"""
        from agent.common.alert_parser import AlertParser
        from agent.agent.ops_agent import OpsAgent

        alert_context = AlertParser.parse(request.payload)
        if request.source:
            alert_context["source"] = request.source

        # 自动匹配 instance_id (如果调用方未提供)
        instance_id = request.instance_id
        if not instance_id:
            instance_id = await self._match_instance_from_alert(
                request.payload, alert_context.get("source", "generic"),
            )
            logger.info(f"[OpsController] 告警自动匹配实例: {instance_id}")

        ops_agent = OpsAgent()
        return await ops_agent.chat_from_alert(
            background_tasks=background_tasks,
            user_id="alert-bot",
            agent_id=request.agent_id,
            instance_id=instance_id,
            alert_context=alert_context,
        )

    async def _match_instance_from_alert(
        self, payload: dict, source: str,
    ) -> str:
        """从告警 payload 中提取标识，反查 CMDB 匹配 instance_id。

        匹配策略 (按优先级):
        1. Prometheus: 取 alerts[0].labels.instance → CMDB.prometheus_instance_label
        2. Zabbix:     取 host_name → CMDB.zabbix_host_name
        """
        from services.basic import OpsDBInstanceService
        svc = OpsDBInstanceService()

        alerts_list = payload.get("alerts", [])
        if alerts_list:
            labels = alerts_list[0].get("labels", {})
            instance_label = labels.get("instance", "")
            if instance_label:
                inst = await svc.find_by_prometheus_label(instance_label)
                if inst:
                    return inst["instance_id"]

        host_name = payload.get("host_name", "")
        if host_name:
            inst = await svc.find_by_zabbix_host(host_name)
            if inst:
                return inst["instance_id"]

        trigger_host = payload.get("trigger_host") or payload.get("hosts", [None])[0]
        if isinstance(trigger_host, str) and trigger_host:
            inst = await svc.find_by_zabbix_host(trigger_host)
            if inst:
                return inst["instance_id"]

        raise NotFoundError(
            f"无法从告警匹配到 CMDB 实例。请确保 CMDB 中已配置 prometheus_instance_label "
            f"或 zabbix_host_name。告警来源: {source}"
        )

    async def resume_chat(
        self,
        request: OpsResumeRequest,
        background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        """HITL 恢复执行接口端点"""
        from agent.agent.ops_agent import OpsAgent
        ops_agent = OpsAgent()

        return await ops_agent.resume(
            background_tasks=background_tasks,
            request_id=request.request_id,
            user_data=request.user_data,
            user_note=request.user_note,
            user_error=request.user_error,
        )

    async def confirm_action(
        self,
        request: OpsConfirmActionRequest,
        background_tasks: BackgroundTasks,
    ) -> StreamingResponse:
        """逐命令确认接口"""
        from agent.agent.ops_agent import OpsAgent
        ops_agent = OpsAgent()
        return await ops_agent.confirm_action(
            background_tasks=background_tasks,
            request_id=request.request_id,
            confirmed=request.confirmed,
        )

    async def approve_action(
        self,
        request: OpsApproveRequest,
        background_tasks: BackgroundTasks
    ) -> StreamingResponse:
        """HITL 审批接口端点"""
        from agent.agent.ops_agent import OpsAgent
        ops_agent = OpsAgent()

        return await ops_agent.approve(
            background_tasks=background_tasks,
            request_id=request.request_id,
            approved=request.approved,
            approver_note=request.approver_note,
        )

    async def cancel_pending(self, request_id: str) -> SuccessResponse:
        """取消挂起的 HITL 请求"""
        from platform_core.database.oracle import get_session
        from dao.repositories import PendingRequestRepository, MemoryRepository
        from sqlalchemy import update as sql_update
        from dao.entities import ConversationContextEntity

        async with get_session() as session:
            repo = PendingRequestRepository(session)
            pending = await repo.get_by_request_id(request_id)

            if not pending:
                return SuccessResponse(message="挂起请求不存在", data=None)

            if pending["status"] != "pending":
                return SuccessResponse(
                    message=f"挂起请求状态为 {pending['status']}，无需取消",
                    data=None
                )

            # 标记为已取消
            await repo.mark_cancelled(request_id)

            # 清除会话挂起标记
            mem_repo = MemoryRepository(session)
            stmt = (
                sql_update(ConversationContextEntity)
                .where(ConversationContextEntity.session_id == pending["session_id"])
                .values(pending_request_id=None, is_suspended=0)
            )
            await session.execute(stmt)
            await session.commit()

        logger.info(f"[HITL Cancel] request_id={request_id} 已被用户取消")
        return SuccessResponse(
            message="挂起请求已取消",
            data={"request_id": request_id}
        )


# 单例导出
ops_controller = OpsController()
