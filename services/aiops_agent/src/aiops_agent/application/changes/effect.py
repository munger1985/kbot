"""处置前冻结、处置后复算和正式报告发布；不签发执行权限。"""

from datetime import datetime
from uuid import UUID

from aiops_agent.application.configuration.common import canonical_json, sha256_json
from aiops_agent.application.effect_measurement import apply_action_verification, evaluate_effect, measurement, validate_window_report
from aiops_agent.application.errors import AIOpsApplicationError, resource_not_found, state_conflict
from aiops_agent.application.source_authorization import source_scope_is_current
from aiops_agent.application.reporting import resolve_system_template, template_summary
from aiops_agent.contracts.change import ActionVerification, ExecutionResultArtifact
from aiops_agent.contracts.report import ComparisonPlan, ComparisonResult, ReportContent
from aiops_agent.entities import OpsArtifactEntity, OutboxEntity, ReportEntity, ReportSourceEntity
from platform_core.contracts.aiops.effect import EffectVerificationView
from platform_core.identity import uuid7


def _key(proposal_id, suffix):
    return f"comparison:proposal:{proposal_id}:{suffix}:business:v1"


def effect_report_facts(result):
    """报告发布和案例来源核验共用结构化测量投影，不从展示正文反推效果。"""
    facts = []
    for name, value in result.measurements.items():
        outcome = "满足" if value.get("passed") is True else "不满足" if value.get("passed") is False else "证据不足"
        facts.append({
            "kind": "effect_measurement", "key": name, **value,
            "summary": f"{name}（{value['metric']}）：基线 {value.get('baseline', '未知')}，"
            f"处置后 {value.get('after', '未知')}，单位 {value.get('unit', '未知')}；"
            f"冻结条件：{outcome}；公式：{value.get('formula', '不可计算')}。",
        })
    for name, value in result.control_analysis.get("signals", {}).items():
        facts.append({
            "kind": "control_measurement", "key": name, **value,
            "summary": f"对照 {name}：前值 {value['control_before']}，后值 {value['control_after']}，"
            f"单位 {value['unit']}；校正相对变化 {value['adjusted_relative_change']}；"
            f"校正条件是否满足 {value['adjusted_passed']}。",
        })
    return tuple(facts)


async def effect_source(*, uow, artifact_id, run, schema=None, target_id=None):
    """重读来源及当前授权；调用方不能把上传内容伪装成库内观测。"""
    source = await uow.runs.get_artifact(artifact_id=artifact_id)
    source_run = await uow.runs.get_run_scoped(ops_run_id=source.ops_run_id, domain_id=run.domain_id) if source is not None else None
    expected_target = target_id or run.target_id
    if source is None or source_run is None or str(source_run.target_id) != str(expected_target):
        raise resource_not_found("效果验证证据")
    if str(expected_target) != str(run.target_id):
        if source.schema_version != "AIOPS_WORKLOAD_REPORT.v2" or source_run.workflow_kind != "WORKLOAD_REPORT" or str(source_run.actor_id) != str(run.actor_id):
            raise resource_not_found("对照窗口证据")
        active = await uow.agents.get(domain_id=run.domain_id, agent_id=run.agent_id)
        if active is None or active.current_version_id is None or not await uow.agents.version_has_target(
            agent_version_id=active.current_version_id, target_id=source_run.target_id,
        ):
            raise state_conflict("当前 Agent 已无对照目标权限")
    if (str(source_run.agent_id) != str(run.agent_id)
        or getattr(source_run, "workflow_kind", None) == "WORKLOAD_REPORT"
        or (source.schema_version == "AIOPS_WORKLOAD_REPORT.v2" and str(source_run.ops_run_id) != str(run.ops_run_id))):
        # 周期快照报告属于 Target；只允许本人生成、当前 Agent 仍获授权的同目标窗口复用。
        if source.schema_version != "AIOPS_WORKLOAD_REPORT.v2" or source_run.workflow_kind != "WORKLOAD_REPORT" or str(source_run.actor_id) != str(run.actor_id):
            raise resource_not_found("效果验证证据")
        target = await uow.targets.get_scoped(target_id=expected_target, domain_id=run.domain_id)
        frozen_target = dict((source_run.plan_snapshot_json or {}).get("target") or {})
        if target is None or target.status != "ENABLED" or str(target.row_version) != str(frozen_target.get("row_version")):
            raise state_conflict("窗口报告来源的 Target 配置已变化")
    if source.trust_level != "SOURCE_VERIFIED" or not isinstance(source.payload_json, dict) or sha256_json(source.payload_json) != source.content_hash or (schema and source.schema_version != schema):
        raise state_conflict("效果验证证据完整性或来源不满足要求")
    if not await source_scope_is_current(uow=uow, run=source_run, domain_id=run.domain_id):
        raise state_conflict("效果验证来源权限或绑定已变化")
    bindings = dict((source_run.plan_snapshot_json or {}).get("monitoring") or {}).get("bindings") or ()
    if bindings:
        active = await uow.agents.get(domain_id=run.domain_id, agent_id=run.agent_id)
        if active is None or active.current_version_id is None:
            raise state_conflict("效果验证 Agent 已不可用")
        allowed_sources = await uow.agents.version_source_ids(
            agent_version_id=active.current_version_id, target_id=expected_target,
        )
        used_sources = {str(item["source"]["source_id"]) for item in bindings}
        if not used_sources <= {str(value) for value in allowed_sources}:
            raise state_conflict("当前 Agent 已无效果证据所使用的来源权限")
    return source


class ChangeEffectVerification:
    async def _effect_scope(self, uow, proposal_id, domain_id, actor_id, lock=False, agent_ids=()):
        preliminary = await uow.changes.get_proposal_scoped(proposal_id=proposal_id, domain_id=domain_id)
        if preliminary is None:
            raise resource_not_found("Proposal")
        run = await uow.runs.get_run(ops_run_id=preliminary.ops_run_id, lock=lock)
        proposal = await uow.changes.get_proposal(proposal_id=proposal_id, lock=lock)
        if run is None or proposal is None or str(run.actor_id) != str(actor_id):
            raise resource_not_found("Proposal")
        if agent_ids and str(run.agent_id) not in {str(value) for value in agent_ids}:
            raise resource_not_found("Proposal")
        active = await uow.agents.get(domain_id=domain_id, agent_id=run.agent_id)
        if active is None or active.current_version_id is None or not await uow.agents.version_has_target(agent_version_id=active.current_version_id, target_id=run.target_id):
            raise state_conflict("Agent 当前已无此 Target 的验证权限")
        target = await uow.targets.get_scoped(target_id=run.target_id, domain_id=domain_id)
        snapshot = dict((run.plan_snapshot_json or {}).get("target") or {})
        if target is None or target.status != "ENABLED" or str(snapshot.get("row_version")) != str(target.row_version):
            raise state_conflict("Target 版本或状态已经改变，需重新调查")
        return run, proposal

    async def get_effect_verification(self, *, proposal_id, domain_id, actor_id, agent_ids=(), control_target_id=None):
        async with self._uow_factory() as uow:
            run, proposal = await self._effect_scope(uow, proposal_id, domain_id, actor_id, agent_ids=agent_ids)
            return await self._effect_view(uow, run, proposal, candidates=True, control_target_id=control_target_id)

    async def freeze_effect_plan(self, *, proposal_id, domain_id, actor_id, command, trace_id, agent_ids=()):
        async with self._uow_factory() as uow:
            run, proposal = await self._effect_scope(uow, proposal_id, domain_id, actor_id, lock=True, agent_ids=agent_ids)
            prior = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=_key(proposal_id, "plan"))
            request_hash = sha256_json(command.model_dump(mode="json"))
            if prior is not None:
                if dict(prior.provenance_json or {}).get("request_hash") != request_hash:
                    raise state_conflict("效果计划已冻结，不允许事后调整成功条件")
                return await self._effect_view(uow, run, proposal)
            if proposal.status not in {"ADVISORY_READY", "PENDING_APPROVAL"} or int(proposal.row_version) != command.expected_row_version:
                raise state_conflict("只能在执行或审批前冻结当前版本的效果计划")
            baseline = await effect_source(uow=uow, artifact_id=command.baseline_artifact_id, run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
            now = await uow.runs.database_now()
            if proposal.expires_at is not None and proposal.expires_at <= now:
                raise state_conflict("Proposal 已过期，不能冻结效果计划")
            try:
                start, end = validate_window_report(baseline.payload_json)
                if end > now or (now - end).total_seconds() > command.max_baseline_age_seconds:
                    raise ValueError("基线不是近期已结束的窗口")
                for rule in command.signals:
                    measurement(baseline.payload_json, rule)
            except (KeyError, TypeError, ValueError) as exc:
                raise state_conflict(str(exc)) from None
            snapshot = await self._snapshot(uow, proposal)
            body = command.model_dump(mode="json")
            body.update({"baseline_hash": baseline.content_hash, "frozen_at": now.isoformat()})
            from aiops_agent.application.changes.effect_controls import freeze_control_reports
            control_freeze, control_sources = await freeze_control_reports(
                uow=uow, run=run, command=command, baseline=baseline, now=now,
            )
            body.update(control_freeze)
            if command.auto_evaluate:
                target = await uow.targets.get_scoped(
                    target_id=run.target_id, domain_id=domain_id,
                )
                policy = dict(target.workload_snapshot_policy_json or {})
                if not policy.get("enabled"):
                    raise state_conflict("自动效果验证需要已启用的周期快照；不会自动开启采集")
                body["collection_interval_seconds"] = int(policy["interval_minutes"]) * 60
            plan = ComparisonPlan(
                proposal_id=str(proposal.proposal_id), source_run_id=str(run.ops_run_id),
                solution_group_key=proposal.solution_group_key,
                action_template_id=proposal.action_template_id,
                action_template_version=proposal.action_template_version,
                baseline_start=start, baseline_end=end,
                settle_delay_seconds=command.settle_delay_seconds,
                after_window_seconds=command.after_window_seconds,
                primary_signals=tuple(item.key for item in command.signals if item.role == "PRIMARY"),
                guardrail_signals=tuple(item.key for item in command.signals if item.role == "GUARDRAIL"),
                required_tool_refs=tuple(snapshot.verification_plan),
                baseline_evidence_refs=(f"artifact:{baseline.artifact_id}",),
                business_effect=body, result_rule_version="business-effect.v1",
            )
            security = max(int(item.security_level) for item in (baseline, *control_sources))
            await self._effect_artifact(uow, run, proposal, _key(proposal_id, "plan"), "COMPARISON_PLAN.v1", plan.model_dump(mode="json"), {"request_hash": request_hash}, security)
            await uow.runs.append_event(ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id, event_type="comparison.business_plan.frozen", event_key=_key(proposal_id, "plan"), visibility="USER", payload_json={"proposal_id": str(proposal_id), "trace_id": trace_id})
            await uow.commit()
            return await self._effect_view(uow, run, proposal)

    async def evaluate_effect_plan(self, *, proposal_id, domain_id, actor_id, command, trace_id, agent_ids=()):
        async with self._uow_factory() as uow:
            run, proposal = await self._effect_scope(uow, proposal_id, domain_id, actor_id, lock=True, agent_ids=agent_ids)
            request_hash = sha256_json(command.model_dump(mode="json"))
            result_key = _key(proposal_id, f"result:{request_hash}")
            existing = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=result_key)
            if existing is not None:
                if dict(existing.provenance_json or {}).get("request_hash") != request_hash:
                    raise state_conflict("效果验证幂等记录不完整")
                return await self._effect_view(uow, run, proposal)
            if int(proposal.row_version) != command.expected_row_version:
                raise state_conflict("Proposal 版本已变化")
            plan_artifact = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=_key(proposal_id, "plan"))
            if plan_artifact is None or sha256_json(plan_artifact.payload_json) != plan_artifact.content_hash:
                raise state_conflict("执行前未冻结有效的业务效果计划")
            plan = ComparisonPlan.model_validate(plan_artifact.payload_json)
            if plan.business_effect is None or plan.result_rule_version != "business-effect.v1":
                raise state_conflict("没有冻结业务效果规则，动作复核不能代替业务效果")
            frozen = plan.business_effect.model_dump(mode="json")
            baseline = await effect_source(uow=uow, artifact_id=UUID(frozen["baseline_artifact_id"]), run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
            after = await effect_source(uow=uow, artifact_id=command.after_artifact_id, run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
            control_sources = ()
            if plan.business_effect.controlled_comparison is not None:
                from aiops_agent.application.changes.effect_controls import load_control_reports
                control_sources = await load_control_reports(
                    uow=uow, run=run, control=plan.business_effect.controlled_comparison,
                    frozen=frozen, after_artifact_id=command.control_after_artifact_id,
                )
            elif command.control_after_artifact_id is not None:
                raise state_conflict("执行前未冻结对照，不能事后选择对照组")
            verification_artifact = await effect_source(uow=uow, artifact_id=command.verification_artifact_id, run=run, schema="ACTION_VERIFICATION.v1")
            verification = ActionVerification.model_validate(verification_artifact.payload_json)
            if str(verification.proposal_id) != str(proposal_id) or str(verification.source_run_id) != str(run.ops_run_id) or baseline.content_hash != frozen["baseline_hash"] or baseline.artifact_id == after.artifact_id:
                raise state_conflict("验证与冻结计划的 Proposal、基线或来源不一致")
            receipt = await uow.runs.get_artifact(artifact_id=verification.result_artifact_id)
            if receipt is None or sha256_json(receipt.payload_json) != receipt.content_hash or str(receipt.ops_run_id) != str(run.ops_run_id) or str(receipt.payload_json.get("proposal_id")) != str(proposal_id):
                raise state_conflict("缺少对应动作的完整执行来源链")
            controlled = receipt.schema_version == "EXECUTION_RESULT.v1" and receipt.trust_level == "SOURCE_VERIFIED"
            manual = receipt.schema_version == "USER_PROVIDED_ACTION_RESULT.v1" and receipt.trust_level == "USER_PROVIDED"
            if not ((controlled and receipt.payload_json.get("status") == "SUCCEEDED") or (manual and receipt.payload_json.get("status") == "EXECUTED")):
                raise state_conflict("执行来源不明或动作未执行成功")
            if controlled:
                execution_result = ExecutionResultArtifact.model_validate(receipt.payload_json)
                execution = await uow.changes.get_execution(execution_id=execution_result.execution_id)
                if execution is None or str(execution.proposal_id) != str(proposal_id) or str(execution.result_artifact_id) != str(receipt.artifact_id) or execution.status != "SUCCEEDED" or execution.proposal_hash != execution_result.proposal_hash or execution.command_hash != execution_result.command_hash:
                    raise state_conflict("执行回执与权威 Execution 不匹配")
            occurred_at = datetime.fromisoformat(str(receipt.payload_json["occurred_at"]).replace("Z", "+00:00"))
            now = await uow.runs.database_now()
            if occurred_at < datetime.fromisoformat(frozen["frozen_at"]) or occurred_at > now:
                raise state_conflict("动作时间早于计划冻结或晚于当前时间")
            try:
                evaluated = evaluate_effect(
                    frozen=frozen, baseline=baseline.payload_json, after=after.payload_json,
                    occurred_at=occurred_at, direct_status=verification.status,
                    controlled_execution=controlled,
                    baseline_ref=f"artifact:{baseline.artifact_id}",
                    after_ref=f"artifact:{after.artifact_id}",
                    control_reports={str(item.artifact_id): item.payload_json for item in control_sources},
                    control_after_ref=str(command.control_after_artifact_id) if command.control_after_artifact_id else None,
                )
                _, after_end = validate_window_report(after.payload_json)
                if after_end > now:
                    raise ValueError("验证窗口尚未结束")
                for source in control_sources:
                    if validate_window_report(source.payload_json)[1] > now:
                        raise ValueError("对照窗口尚未结束")
            except (KeyError, TypeError, ValueError) as exc:
                raise state_conflict(str(exc)) from None
            artifacts = (plan_artifact, baseline, after, verification_artifact, receipt, *control_sources)
            refs = tuple({"artifact_id": str(item.artifact_id), "content_hash": item.content_hash, "schema_version": item.schema_version} for item in artifacts)
            evaluated = apply_action_verification(evaluated, verification)
            after_start, after_end = validate_window_report(after.payload_json)
            result = ComparisonResult(comparison_plan_artifact_id=str(plan_artifact.artifact_id), verification_artifact_id=str(verification_artifact.artifact_id), proposal_id=str(proposal_id), source_run_id=str(run.ops_run_id), source_result_artifact_id=str(receipt.artifact_id), baseline_start=plan.baseline_start, baseline_end=plan.baseline_end, after_start=after_start, after_end=after_end, primary_signals=evaluated["primary_signals"], guardrail_signals=evaluated["guardrail_signals"], load_signals=evaluated["load_signals"], measurements=evaluated["measurements"], measurement_artifact_refs=refs, execution_source=evaluated["execution_source"], gap_codes=evaluated["gap_codes"], evidence_hashes=tuple(item.content_hash for item in artifacts), result=evaluated["result"], rationale_codes=("FROZEN_BUSINESS_EFFECT_RECOMPUTED", "MATCHED_CONTROL_SUPPORT" if evaluated["attribution_strength"] == "SUPPORTED" else "OBSERVATIONAL_ATTRIBUTION_LIMITED"), direct_effect={"VERIFIED": "ACHIEVED", "NOT_ACHIEVED": "NOT_ACHIEVED", "ADVERSE": "ADVERSE"}.get(verification.status, "UNKNOWN"), measured_effect=evaluated["measured_effect"], symptom_recovered=evaluated["symptom_recovered"], guardrails_passed=evaluated["guardrails_passed"], attribution_strength=evaluated["attribution_strength"], causal_limitations=evaluated["causal_limitations"], control_analysis=evaluated.get("control_analysis", {}))
            security = max(int(item.security_level) for item in artifacts)
            result_artifact = await self._effect_artifact(uow, run, proposal, result_key, "COMPARISON_RESULT.v1", result.model_dump(mode="json"), {"request_hash": request_hash}, security)
            report_id = await self._publish_effect_report(uow, run, proposal, result_artifact, refs, security, trace_id, now)
            await uow.runs.append_event(ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id, event_type="comparison.business_effect.ready", event_key=result_key, visibility="USER", payload_json={"proposal_id": str(proposal_id), "report_id": str(report_id), "result": result.result, "trace_id": trace_id})
            await uow.commit()
            return (await self._effect_view(uow, run, proposal)).model_copy(update={"report_id": report_id})

    async def _effect_view(self, uow, run, proposal, candidates=False, control_target_id=None):
        plan = await uow.runs.get_artifact_by_key(ops_run_id=run.ops_run_id, artifact_key=_key(proposal.proposal_id, "plan"))
        result = await uow.runs.get_latest_effect_result(ops_run_id=run.ops_run_id, proposal_id=proposal.proposal_id)
        if plan:
            await effect_source(uow=uow, artifact_id=plan.artifact_id, run=run, schema="COMPARISON_PLAN.v1")
            checked_plan = ComparisonPlan.model_validate(plan.payload_json)
            if checked_plan.business_effect is None or checked_plan.result_rule_version != "business-effect.v1":
                raise state_conflict("业务效果计划合同不完整")
            frozen = checked_plan.business_effect.model_dump(mode="json")
            baseline = await effect_source(uow=uow, artifact_id=UUID(frozen["baseline_artifact_id"]), run=run, schema="AIOPS_WORKLOAD_REPORT.v2")
            if baseline.content_hash != frozen["baseline_hash"]:
                raise state_conflict("冻结的基线哈希已经改变")
            if checked_plan.business_effect.controlled_comparison is not None:
                from aiops_agent.application.changes.effect_controls import load_control_reports
                await load_control_reports(
                    uow=uow, run=run, control=checked_plan.business_effect.controlled_comparison,
                    frozen=frozen,
                )
        if result:
            await effect_source(uow=uow, artifact_id=result.artifact_id, run=run, schema="COMPARISON_RESULT.v1")
            for ref in ComparisonResult.model_validate(result.payload_json).measurement_artifact_refs:
                source = await uow.runs.get_artifact(artifact_id=UUID(ref["artifact_id"]))
                if source is None or source.content_hash != ref["content_hash"] or sha256_json(source.payload_json) != source.content_hash:
                    raise state_conflict("效果来源链已经改变")
                if source.trust_level == "SOURCE_VERIFIED":
                    source_target_id = None
                    if plan and checked_plan.business_effect.controlled_comparison is not None:
                        control = checked_plan.business_effect.controlled_comparison
                        from aiops_agent.application.control_measurement import control_source_ids
                        control_ids = {identity for identity, target in control_source_ids(control) if target}
                        after_id = dict(result.payload_json.get("control_analysis") or {}).get("after_artifact_id")
                        if after_id:
                            control_ids.add(str(after_id))
                        if str(source.artifact_id) in control_ids:
                            source_target_id = control.target_id
                    await effect_source(uow=uow, artifact_id=source.artifact_id, run=run, target_id=source_target_id)
                elif source.schema_version != "USER_PROVIDED_ACTION_RESULT.v1" or source.trust_level != "USER_PROVIDED" or str(source.ops_run_id) != str(run.ops_run_id):
                    raise state_conflict("人工执行来源链不匹配")
        items = []
        if candidates:
            frozen_control = checked_plan.business_effect.controlled_comparison if plan else None
            if frozen_control is not None:
                if control_target_id is not None and str(control_target_id) != str(frozen_control.target_id):
                    raise state_conflict("不能在冻结后替换对照目标")
                control_target_id = frozen_control.target_id
            candidate_targets = [run.target_id]
            if control_target_id is not None:
                from aiops_agent.application.changes.effect_controls import control_target
                await control_target(uow=uow, run=run, target_id=control_target_id)
                candidate_targets.append(control_target_id)
            for candidate_target_id in candidate_targets:
                for item in await uow.runs.list_effect_artifacts(domain_id=run.domain_id, target_id=candidate_target_id, agent_id=run.agent_id, actor_id=run.actor_id, ops_run_id=run.ops_run_id, proposal_id=proposal.proposal_id, limit=20):
                    try:
                        await effect_source(uow=uow, artifact_id=item.artifact_id, run=run, target_id=candidate_target_id)
                    except AIOpsApplicationError:
                        continue
                    payload = item.payload_json
                    if item.schema_version == "ACTION_VERIFICATION.v1" and str(payload.get("proposal_id")) != str(proposal.proposal_id):
                        continue
                    options = []
                    if item.schema_version == "AIOPS_WORKLOAD_REPORT.v2":
                        options = [{"metric": f"load.{key}", "statement_identity": {}} for key in (payload.get("load_dimension_semantics") or {})]
                        options.extend({"metric": f"metric.{key}", "statement_identity": {}, "label": f"{value['metric_code']} · {value['dimension_key']} · {value['unit']}/s"} for key, value in (payload.get("native_window_metrics") or {}).items())
                        for row in payload.get("top_statements", ()):
                            identity = {key: row[key] for key in ("statement_identity_type", "statement_identity_value", "database_name", "user_identifier", "top_level") if key in row}
                            if len(identity) == 5:
                                if row.get("plan_hash_value"):
                                    identity["plan_hash_value"] = row["plan_hash_value"]
                                options.extend({"metric": metric, "statement_identity": identity} for metric in ("statement.mean_duration_microseconds", "statement.rows_per_call", "statement.calls_per_second"))
                    items.append({"target_id": str(candidate_target_id), "artifact_id": str(item.artifact_id), "schema_version": item.schema_version, "period_start": payload.get("period_start"), "period_end": payload.get("period_end"), "status": payload.get("status"), "metrics": options[:128], "metrics_truncated": len(options) > 128})
        report = await uow.inspections.get_current_effect_report(ops_run_id=run.ops_run_id, report_key=f"comparison.business.{proposal.proposal_id}") if result else None
        automatic = await uow.runs.get_artifact_by_key(
            ops_run_id=run.ops_run_id,
            artifact_key=_key(proposal.proposal_id, "automatic-result"),
        )
        if automatic is None:
            automatic = await uow.runs.get_artifact_by_key(
                ops_run_id=run.ops_run_id,
                artifact_key=_key(proposal.proposal_id, "automatic"),
            )
        if automatic is not None:
            await effect_source(uow=uow, artifact_id=automatic.artifact_id, run=run)
        return EffectVerificationView(
            proposal_id=proposal.proposal_id,
            plan_artifact=self._artifact_ref(plan) if plan else None,
            frozen_plan=plan.payload_json if plan else {},
            comparison_artifact=self._artifact_ref(result) if result else None,
            result=result.payload_json if result else {},
            report_id=report.report_id if report else None,
            evidence_candidates=tuple(items),
            followup=automatic.payload_json if automatic else {},
        )

    @staticmethod
    async def _effect_artifact(uow, run, proposal, key, schema, payload, provenance, security):
        return await uow.runs.add_artifact(OpsArtifactEntity(artifact_id=uuid7(), ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id, artifact_key=key, artifact_type=schema.split(".")[0], schema_version=schema, payload_json=payload, content_hash=sha256_json(payload), byte_size=len(canonical_json(payload).encode()), provenance_json={"producer": "aiops.business-effect", "producer_version": "1", **provenance}, trust_level="SOURCE_VERIFIED", security_level=int(security)))

    async def _publish_effect_report(self, uow, run, proposal, artifact, refs, security, trace_id, now):
        result = ComparisonResult.model_validate(artifact.payload_json)
        # 报告及案例沿同一调查与执行提案溯源，不从效果反推根因或重新生成动作。
        for identity in (getattr(run, "final_artifact_id", None), getattr(proposal, "snapshot_artifact_id", None)):
            if identity is not None and str(identity) not in {ref["artifact_id"] for ref in refs}:
                source = await effect_source(uow=uow, artifact_id=identity, run=run)
                if str(source.ops_run_id) != str(run.ops_run_id):
                    raise state_conflict("效果报告的调查或执行提案不属于当前 Run")
                refs = (*refs, {"artifact_id": str(source.artifact_id), "content_hash": source.content_hash, "schema_version": source.schema_version})
                security = max(security, int(source.security_level))
        report_id = uuid7()
        key = f"comparison.business.{proposal.proposal_id}"
        prior = await uow.inspections.get_current_effect_report(
            ops_run_id=run.ops_run_id, report_key=key,
        )
        report_version = int(prior.report_version) + 1 if prior is not None else 1
        summary = {
            "IMPROVED": "可比业务负载下，冻结的主要效果条件已达到；归因仍为观察性支持",
            "RESOLVED": "实测效果及冻结的症状条件满足；不自动关闭工作项",
            "UNCHANGED": "未达到预先冻结的改善条件",
            "DEGRADED": "效果或护栏出现退化，请按停止条件处理",
            "INCONCLUSIVE": "当前窗口或负载不可比，不能确认业务效果",
        }[result.result]
        if result.attribution_strength == "SUPPORTED":
            summary = "预登记匹配对照支持本次措施效果：主要效果达到、负载可比、护栏通过；不等于根因已确认"
        elif result.control_analysis:
            summary += "；匹配对照未达到支持归因的条件"
        template = resolve_system_template("system:comparison.business")
        facts = effect_report_facts(result)
        content = ReportContent(
            report_key=key, report_type="COMPARISON", ops_run_id=str(run.ops_run_id),
            target_id=str(run.target_id), title="业务效果验证报告",
            status="PARTIAL" if result.result == "INCONCLUSIVE" or result.gap_codes else "READY",
            summary=summary, period_start=result.baseline_start, period_end=result.after_end,
            scope={
                "proposal_id": str(proposal.proposal_id),
                "attribution_strength": result.attribution_strength,
                "execution_source": result.execution_source,
                "measured_effect": result.measured_effect,
                "symptom_recovered": result.symptom_recovered,
                "guardrails_passed": result.guardrails_passed,
                "comparison_result_artifact_id": str(artifact.artifact_id),
                "comparison_method": result.control_analysis.get("method", "OBSERVATIONAL"),
                "verification": {
                    "method": result.control_analysis.get("method", "OBSERVATIONAL"), "result": result.result,
                    "direct_effect": result.direct_effect, "measured_effect": result.measured_effect,
                    "symptom_recovered": result.symptom_recovered, "guardrails_passed": result.guardrails_passed,
                    "attribution_strength": result.attribution_strength,
                },
            },
            facts=tuple(facts), gaps=tuple({"code": code} for code in result.gap_codes),
            evidence_refs=(*refs, {
                "artifact_id": str(artifact.artifact_id), "content_hash": artifact.content_hash,
                "schema_version": artifact.schema_version,
            }),
            recommendations=result.causal_limitations,
            provenance={
                "deterministic": True, "execution_does_not_imply_effect": True,
                "template": {**template_summary(template), "definition": template.definition},
            },
        )
        report_artifact = await self._effect_artifact(
            uow, run, proposal, f"report:{key}:v{report_version}", "REPORT_CONTENT.v1",
            content.model_dump(mode="json"), {"comparison_result_artifact_id": str(artifact.artifact_id)}, security,
        )
        report = await uow.inspections.publish_report(ReportEntity(
            report_id=report_id, ops_run_id=run.ops_run_id, target_id=run.target_id,
            report_key=key, report_version=report_version,
            supersedes_report_id=prior.report_id if prior is not None else None,
            is_current=False, report_type="COMPARISON", title=content.title, status=content.status,
            period_start=result.baseline_start, period_end=result.after_end,
            baseline_start=result.baseline_start, baseline_end=result.baseline_end,
            after_start=result.after_start, after_end=result.after_end, result=result.result,
            template_id=template.template_ref, template_version=template.version,
            generated_by_task_id=proposal.ops_task_id, content_artifact_id=report_artifact.artifact_id,
            content_hash=report_artifact.content_hash, summary=summary,
            security_level=security, schema_version="REPORT_CONTENT.v1",
        ))
        sources = []
        for ref in (*refs, {"artifact_id": str(artifact.artifact_id)}):
            source = await uow.runs.get_artifact(artifact_id=UUID(ref["artifact_id"]))
            sources.append(ReportSourceEntity(
                report_id=report_id, ops_run_id=source.ops_run_id, source_artifact_id=source.artifact_id,
                source_kind="COMPARISON", content_hash=source.content_hash, observed_at=result.after_end,
            ))
        await uow.inspections.add_report_sources(sources)
        event_key = f"report:{report_id}:ready"
        payload = {
            "report_id": str(report_id), "ops_run_id": str(run.ops_run_id), "report_key": key,
            "report_type": "COMPARISON", "report_version": report_version, "status": content.status,
        }
        await uow.runs.append_event(
            ops_run_id=run.ops_run_id, ops_task_id=proposal.ops_task_id,
            event_type="report.ready", event_key=event_key, visibility="USER",
            payload_json={**payload, "summary": summary, "trace_id": trace_id},
        )
        await uow.outbox.add(OutboxEntity(
            aggregate_type="OPS_RUN", aggregate_id=report_id, event_type="OPS_REPORT_READY",
            idempotency_key=event_key, payload_json=payload, payload_hash=sha256_json(payload),
            status="PENDING", available_at=now, max_attempts=5, trace_id=trace_id,
        ))
        await uow.platform_notifications.emit_report_ready(run=run, report=report, actor_id=run.actor_id)
        return report_id
