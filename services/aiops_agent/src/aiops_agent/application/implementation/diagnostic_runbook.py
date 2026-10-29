"""从同一调查记录编译三库处置审阅稿；不把草案送入 Executor。"""

from aiops_agent.contracts.implementation import (
    ImplementationRunbook, RunbookApplicability, RunbookCommand, RunbookCommandType,
    RunbookExecutor, RunbookPhase, RunbookStatus, RunbookStep, RunbookMissingFact,
    RunbookFactSource,
)
from platform_core.contracts.aiops import ImplementationProfile


def compile_diagnostic_runbook(record, *, database_type, database_version):
    if database_type not in {"ORACLE", "POSTGRESQL", "MYSQL"} or not any(
        item.draft_sql or item.intent in {"MITIGATE", "REPAIR"} for item in record.recommendations
    ):
        return None
    executors = {"ORACLE": RunbookExecutor.SQLPLUS, "POSTGRESQL": RunbookExecutor.PSQL, "MYSQL": RunbookExecutor.MYSQL}
    steps, missing = [], []
    for action in record.recommendations:
        if not action.draft_sql and action.intent not in {"MITIGATE", "REPAIR"}:
            continue
        step_id = "diagnosis." + action.recommendation_id
        unresolved = tuple(action.missing_parameters) + tuple(
            item.name for item in action.parameters if item.source != "OBSERVATION"
        )
        for index, name in enumerate(unresolved, start=1):
            missing.append(RunbookMissingFact(
                fact_key=f"PARAMETER.{action.recommendation_id.upper()}.{index}",
                resolution_source=RunbookFactSource.EXPLICIT_USER_DECISION,
                reason=f"待核验参数：{name}", blocking_steps=(step_id,),
            ))
        commands = ()
        if action.draft_sql:
            commands = (RunbookCommand(
                command_id=step_id + ".draft", command_type=RunbookCommandType.MANUAL,
                executor=executors[database_type], run_as="待审批确定，不继承诊断凭据",
                node_scope=(record.target_id,), title="待验证 SQL 草案（不是执行授权）",
                content=action.draft_sql, notes=action.preconditions,
                expected_result=action.validation,
            ),)
        steps.append(RunbookStep(
            step_id=step_id, title=action.action,
            applicability=RunbookApplicability.BLOCKED if unresolved else RunbookApplicability.CONDITIONAL,
            rationale="关联判断：" + "、".join(action.conclusion_ids) + "；适用条件：" + "；".join(action.preconditions),
            commands=commands,
            verification_commands=(RunbookCommand(
                command_id=step_id + ".verify", command_type=RunbookCommandType.MANUAL,
                executor=RunbookExecutor.MANUAL, run_as="诊断操作者", node_scope=(record.target_id,),
                title="同口径验证与护栏", content="；".join(action.validation),
            ),),
            rollback=(RunbookCommand(
                command_id=step_id + ".stop", command_type=RunbookCommandType.MANUAL,
                executor=RunbookExecutor.MANUAL, run_as="已审批的实施人员", node_scope=(record.target_id,),
                title="停止或回退条件", content=action.rollback_or_stop,
            ),),
            risks=action.risks, required_inputs=unresolved,
        ))
    return ImplementationRunbook(
        profile=ImplementationProfile.NONE, title=f"{database_type} 诊断处置审阅方案",
        database_type=database_type, database_version=database_version,
        status=RunbookStatus.BLOCKED_BY_REQUIRED_INPUTS if missing else RunbookStatus.PARTIAL_EVIDENCE,
        execution_policy="仅是同一调查的条件化审阅稿，不是已审批变更；未知参数不得猜测。目录不支持的动作只能按人工变更流程执行。",
        current_state=({"label": "来源调查", "value": f"{record.record_id}@{record.revision}", "status": "FROZEN"},
                       {"label": "目标", "value": record.target_id, "status": "LOGICAL_TARGET"}),
        missing_facts=tuple(missing),
        phases=(RunbookPhase(phase_id="diagnosis.remediation", title="按优先顺序评审与验证",
            objective="区分补证、缓解和修复；执行、效果与根因状态分别记录", steps=tuple(steps)),),
        stop_conditions=("目标身份、配置或授权改变时停止，并重新核验。", "草案不能直接进入 Executor；必须重新冻结目录动作和审批。"),
        evidence_refs=tuple(dict.fromkeys(ref.evidence_ref for claim in record.claims for ref in claim.evidence_fields)),
    )
