"""使用真实 SERVICE 身份读取任务，再以当前平台事实核权；不伪造用户登录。"""

from datetime import datetime, timezone

from platform_core.contracts.aiops.output_authority import output_required_permissions
from platform_core.contracts.aiops.output_projection import OutputProjectionRequest, OutputProjectionSnapshot
from platform_core.security import create_service_auth_context


def projection_presentation(snapshot):
    labels = {"ACCEPTED": "已排队", "VALIDATING": "核验来源", "ASSEMBLING": "编排正文",
        "REDACTING": "脱敏处理", "INGESTING": "知识入库", "INDEXING": "等待索引",
        "FINALIZING": "保存成果", "COMPLETED": "已完成", "FAILED": "输出未完成"}
    summaries = {"QUEUED": "输出任务已排队。", "RUNNING": "输出任务处理中。",
        "SUCCEEDED": ("案例草稿已生成，人工发布后才能用于后续诊断。"
            if snapshot.request.job_type == "CASE_EXTRACTION" else "报告已生成，内容完整度与缺口请查看报告详情。"),
        "FAILED": "输出任务未完成，请查看生成记录。"}
    return dict(stage_code=snapshot.stage.lower(), stage_label=labels[snapshot.stage],
        summary=summaries[snapshot.status], safe_error_code=snapshot.error_code)


class AIOpsOutputReconciliation:
    def __init__(self, *, client, authority, caller_service: str):
        self._client, self._authority, self._caller = client, authority, caller_service

    async def resolve(self, request: OutputProjectionRequest) -> OutputProjectionSnapshot:
        context = create_service_auth_context(caller_service=self._caller,
            trace_id=f"notification-output-{request.job_id}")
        payload = await self._client.get_output_projection(
            payload=request.model_dump(mode="json"), auth_context=context)
        snapshot = OutputProjectionSnapshot.model_validate(payload)
        if snapshot.request != request:
            raise ValueError("通知任务权威响应的作用域不匹配")
        authority = await self._authority.resolve(snapshot.authority)
        age = (datetime.now(timezone.utc) - authority.checked_at).total_seconds()
        if (authority.request != snapshot.authority
            or not -5 <= age <= 30
            or not output_required_permissions(snapshot.authority) <= set(authority.permissions)
            or (authority.max_security_level is not None
                and snapshot.source_security_level > authority.max_security_level)):
            raise PermissionError("当前发起人已无权读取该输出任务")
        return snapshot
