"""仅用于报告内容回归的阶段组合；公开命令必须经过持久化任务，另有完整 Worker 专项。"""

from types import SimpleNamespace

from aiops_agent.application.output_jobs import OutputJobScope
from aiops_agent.application.report_output import publish_prepared_report
from aiops_agent.application.report_preparation import prepare_conversation_report, prepare_user_report
from platform_core.contracts.aiops.output_jobs import OutputJobSource
from platform_core.identity import uuid7


async def render_report_material(runtime, *, domain_id, actor_id, template, trace_id,
    conversation_id=None, ops_run_id=None, period_kind="AD_HOC"):
    """直接覆盖报告阶段语义，不用合成数据库对象冒充完整异步或权限验收。"""
    async with runtime._uow_factory() as uow:
        kwargs = dict(uow=uow, domain_id=domain_id, actor_id=actor_id, template=template, trace_id=trace_id)
        prepared = (await prepare_conversation_report(runtime, conversation_id=conversation_id, **kwargs)
            if conversation_id is not None else
            await prepare_user_report(runtime, ops_run_id=ops_run_id, period_kind=period_kind, **kwargs))
        scope = OutputJobScope(domain_id=domain_id, actor_id=actor_id,
            tenant_id=getattr(prepared.run, "tenant_id", None), trace_id=trace_id)
        job = SimpleNamespace(job_id=uuid7(), job_type="REPORT_GENERATION", status="RUNNING",
            actor_id=actor_id, domain_id=domain_id, tenant_id=scope.tenant_id, row_version=1, trace_id=trace_id)
        source = OutputJobSource(kind="CONVERSATION" if conversation_id is not None else "RUN",
            source_id=conversation_id or ops_run_id)
        report = await publish_prepared_report(runtime, uow=uow, prepared=prepared, source=source,
            scope=scope, job=job, now=await uow.runs.database_now(),
            minimum_security_level=max(int(getattr(artifact, "security_level", 1))
                for artifact in [prepared.source_artifact, *(row[1] for row in prepared.extra_artifacts)]))
        result = SimpleNamespace(report_id=report.report_id, status=report.status,
            template_id=report.template_id, report_version=int(report.report_version))
        if hasattr(uow, "commit"):
            await uow.commit()
        return result
