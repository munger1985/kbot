"""SQL 调查与回答共享一次诊断，不再先评估后丢弃推理结果。"""

from datetime import datetime, timedelta
from pydantic import create_model, model_validator

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, validate_sql_diagnosis
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft, SqlDiagnosisReview, SqlDiagnosticCheckpoint
from aiops_agent.contracts.sql_diagnostic_response import (
    SqlDiagnosticResponse, SingleSqlDiagnosticResponse, SingleSqlPlanDiagnosticResponse, SqlPlanDiagnosticResponse,
)
from aiops_agent.application.diagnosis.sql_plan_presentation import matching_plan_refs
from aiops_agent.playbooks import canonical_hash


def scope_hash(task_frame, database_type):
    # 补证进展不是新任务；主体、目标、窗口、授权与交付义务等范围变化仍拒绝复用。
    scope = {key: value for key, value in task_frame.items() if key not in {"known_facts", "unknowns"}}
    return canonical_hash({"database_type": database_type, "task_frame": scope})


def evidence_hash(evidence):
    return canonical_hash([fact.model_dump(mode="json") for fact in sorted(evidence, key=lambda fact: fact.evidence_ref)])


def review_context(bundle):
    """只去除重复投影和执行器内部血缘；不按固定指标白名单删去调查资源。"""
    result = {key: value for key, value in bundle.items() if key != "object_graph"}
    observations = []
    for original in bundle["observations"]:
        item = dict(original)
        lineage = dict(item.get("lineage") or {})
        provenance = dict(lineage.get("provenance") or {})
        native = dict(provenance.pop("native_lineage", None) or {})
        provenance.pop("content_hash", None)
        lineage["provenance"] = provenance
        # 字段换算已在记录中逐项投影，重置范围和原生视图语义仍须保留。
        lineage["native_statistics_scope"] = {key: native[key] for key in (
            "relation", "reset_scope", "reset_context_columns", "series_truncated",
            "comparable_window_verified", "exporter_identity_verified",
        ) if key in native}
        item["lineage"] = lineage
        observations.append(item)
    result["observations"] = observations
    # 各主体保留对象关系，但计划匹配只传一次，避免复制每份计划核验记录。
    if "subjects" in result:
        result["subjects"] = [{key: value for key, value in item.items() if key != "plan_assessments"}
                              for item in result["subjects"]]
    return result


def _reference_projection(value, aliases):
    """仅缩短引用字段，不改写 SQL、对象名、原始指标值或声明正文。"""
    if isinstance(value, list):
        return [_reference_projection(item, aliases) for item in value]
    if isinstance(value, dict):
        return {key: aliases.get(item, item) if key in {"evidence_ref", "anchor_ref", "plan_evidence_ref"} and isinstance(item, str)
                else item if key == "fields" else _reference_projection(item, aliases) for key, item in value.items()}
    return value


def _plan_rows(observation):
    records = observation["records"]
    if records and "node_id" in records[0]["fields"]:
        return {int(row["fields"]["node_id"]): row for row in records}
    # Oracle 原生行号可稀疏且顺序不保证是深度优先；必须复现计划投影遍历，不按行号猜节点。
    ordered = []
    visited = set()
    def visit(parent):
        for row in records:
            fields = row["fields"]
            if fields.get("parent_id") == parent and row["row_index"] not in visited:
                visited.add(row["row_index"])
                ordered.append(row)
                visit(fields.get("plan_line_id"))
    if records and all("plan_line_id" in row["fields"] for row in records):
        visit(None)
    return dict(enumerate(ordered))


def plan_review_tasks(bundle):
    matched = matching_plan_refs(bundle)
    return [{"evidence_ref": item["evidence_ref"], "origin": item["sql_plan"]["origin"],
             "runtime_statistics": item["sql_plan"]["runtime_statistics"],
             "nodes": item["sql_plan"]["nodes"]}
            for item in bundle["observations"] if item["evidence_ref"] in matched and item.get("sql_plan")]


def diagnostic_response_model(bundle, aliases=None):
    single = len(bundle.get("subjects") or ()) <= 1 and not bundle.get("requests")
    if plan_review_tasks(bundle):
        base = SingleSqlPlanDiagnosticResponse if single else SqlPlanDiagnosticResponse
    else:
        base = SingleSqlDiagnosticResponse if single else SqlDiagnosticResponse
    if aliases is None:
        return base
    def validate_delivery(self):
        # 在同一次模型合同校验内检查正式引用与目标一致性，错误进入原有有限修正，
        # 不能等回执显示成功后才丢弃整份已生成的诊断。
        from aiops_agent.application.diagnosis.investigation_record import sql_investigation_draft
        draft = expand_diagnostic_response(self, bundle, aliases)
        sql_investigation_draft(draft, "当前选定 SQL", bundle.get("subjects", ()))
        return self
    return create_model(base.__name__, __base__=base,
        __validators__={"validate_delivery": model_validator(mode="after")(validate_delivery)})


def expand_diagnostic_response(response, bundle, aliases):
    """从短引用恢复正式字段；计划解释必须绑定真实节点并进入同一报告/案例事实链。"""
    payload = _reference_projection(response.model_dump(mode="json"), {short: full for full, short in aliases.items()})
    findings = {item["conclusion_id"]: item for item in payload.pop("findings")}
    if len(findings) != len(response.findings):
        raise ValueError("诊断结论标识不能重复")
    observations = {item["evidence_ref"]: item for item in bundle["observations"]}
    matched = matching_plan_refs(bundle)
    reviewed = set()
    for review in payload.pop("plan_reviews"):
        ref = review["evidence_ref"]
        if ref not in matched or review["conclusion_id"] not in findings:
            raise ValueError("计划解释必须引用已匹配的计划及本轮结论")
        observation = observations[ref]
        nodes = {node["node_id"] for node in observation["sql_plan"]["nodes"]}
        if len(set(review["node_ids"])) != len(review["node_ids"]) or not set(review["node_ids"]) <= nodes:
            raise ValueError("计划解释引用了重复或不存在的节点")
        finding = findings[review["conclusion_id"]]
        rows = _plan_rows(observation)
        for node_id in review["node_ids"]:
            if node_id not in rows:
                raise ValueError("计划节点缺少可重放的原始行")
            row = rows[node_id]
            finding["evidence_fields"].extend(
                {"evidence_ref": ref, "row_index": row["row_index"], "field": field}
                for field in ("operation", "options") if field in row["fields"])
        finding["reasoning"] += f"\n\n计划节点 {','.join(map(str, review['node_ids']))}：{review['mechanism']}\n{review['impact']}"
        reviewed.add(ref)
    if matched and not reviewed:
        raise ValueError("已有匹配计划时必须交付关键节点的机制与影响分析")
    # 完成多主体分析不能只解释其中一个主体的计划。
    if payload["goal_satisfied"]:
        for subject in bundle.get("subjects", ()):
            available = {item["evidence_ref"] for item in subject.get("plan_assessments", ())} & matched
            if available and not available & reviewed:
                raise ValueError("诊断完成必须解释每个已有匹配计划的统计主体")
    for finding in findings.values():
        refs = finding["evidence_fields"]
        finding["evidence_fields"] = list({(ref["evidence_ref"], ref["row_index"], ref["field"]): ref for ref in refs}.values())
        times = list(dict.fromkeys(observations[ref["evidence_ref"]]["captured_at"] for ref in refs if ref["evidence_ref"] in observations))
        finding["applicable_time"] = "证据采集于 " + "；".join(times) if times else "未取得直接观测，属于待验证判断"
    payload["conclusions"] = list(findings.values())
    draft = SqlDiagnosisReview.model_validate(payload)
    validate_sql_diagnosis(draft, bundle)
    return draft


def render_sql_review(draft, record):
    """交互回答保留机制与关键边界；报告仍使用完整冻结记录。"""
    from aiops_agent.application.diagnosis.investigation_record import render_investigation
    _, solution, refs = render_investigation(record)
    parts = ["结论：" + draft.summary]
    boundaries = []
    for claim in record.claims:
        if claim.revision_state != "ACTIVE":
            continue
        parts.append(claim.statement + "\n\n" + claim.reasoning)
        boundaries.extend(claim.limitations)
    if boundaries:
        parts.append("适用边界：" + "；".join(dict.fromkeys(boundaries)))
    return "\n\n".join(parts), solution, refs


def reusable_checkpoint(checkpoint, *, task_frame, database_type, evidence, current_only=True):
    if checkpoint is None or checkpoint.scope_hash != scope_hash(task_frame, database_type):
        return None
    if current_only and checkpoint.evidence_hash != evidence_hash(evidence):
        return None
    bundle = build_sql_evidence_bundle(task_frame=task_frame, database_type=database_type, evidence=evidence)
    if bundle is None:
        return None
    try:
        draft = SqlDiagnosisDraft.model_validate(checkpoint.draft.model_dump(mode="python"))
        validate_sql_diagnosis(draft, bundle)
        if current_only and matching_plan_refs(bundle) and not any(
            reference.evidence_ref in matching_plan_refs(bundle)
            for conclusion in draft.conclusions for reference in conclusion.evidence_fields
        ):
            return None
    except (ValueError, TypeError):
        return None
    return draft


async def generate_sql_review(*, context, task_frame, database_type, evidence, gaps, model, prompts, previous=None):
    answer = dict(context.plan_snapshot.get("answer_context") or {})
    bundle = build_sql_evidence_bundle(task_frame=task_frame, database_type=database_type, evidence=evidence)
    prompt = await prompts.resolve("sql_diagnosis", frozen_prompts=dict(answer.get("prompts") or {}))
    aliases = {item["evidence_ref"]: f"e{index}" for index, item in enumerate(bundle["observations"], 1)}
    projected = review_context(bundle)
    output_model = diagnostic_response_model(bundle, aliases)
    from aiops_agent.application.diagnosis.oracle_sql_evidence import (
        authorized_oracle_sql_tools, oracle_evidence_policy,
    )
    oracle_policy = oracle_evidence_policy(
        task_frame=task_frame, evidence=evidence, gaps=gaps,
        available_tool_ids=authorized_oracle_sql_tools(dict(context.plan_snapshot.get("capability_snapshot") or {})),
        prior_actions=tuple(dict(answer.get("investigation_plan") or {}).get("actions") or ()),
    ) if database_type == "ORACLE" else None
    result = await model.generate_structured(
        purpose="aiops.sql-diagnosis", output_model=output_model,
        model_snapshot=dict(answer.get("model") or {}), prompt_ref={**prompt.ref(), "content": prompt.content},
        input_payload={
            "question": answer.get("question", ""), "evidence_bundle": _reference_projection(projected, aliases),
            "oracle_evidence_policy": oracle_policy,
            "plan_review_tasks": _reference_projection([
                {**{key: value for key, value in task.items() if key != "nodes"},
                 "node_ids": [node["node_id"] for node in task["nodes"]]}
                for task in plan_review_tasks(bundle)
            ], aliases),
            "parent_investigation": answer.get("parent_investigation"),
            "previous_diagnosis": previous.draft.model_dump(mode="json") if previous else None,
            "goal_catalog": [{"goal_id": f"g{index}", "description": value} for index, value in enumerate(task_frame.get("success_criteria") or (), 1)]
                or [{"goal_id": "answer", "description": "回答当前问题"}],
            "gaps": [gap.model_dump(mode="json") for gap in gaps],
            "delivery_requirements": {
                "format": "返回精简 SqlDiagnosticResponse，不是完整报告。单主体用1至4项 findings、1至3项建议；省略空的可选字段，不复制 SQL、统计表、计划树、JSON Schema、输入材料或报告模板。每项只写一次机制与针对性措施。",
                "verdict": "必须区分 PROBLEM_FOUND、NO_OBVIOUS_ISSUE、INSUFFICIENT_EVIDENCE，先正面回答；没有发现问题时明确检查范围及暂不变更，不能凭均值低或缺证判断健康。",
                "execution_plan": "plan_review_tasks 非空时必须填写 plan_reviews，引用真实 evidence_ref、node_ids 和关联 conclusion_id；完整计划在 evidence_bundle 对应观察的 sql_plan.nodes，不重复传送。分别说明机制和对该 SQL 的影响：关键访问/连接/聚合节点如何影响开销，估算与实际有何差异；仅列节点或说已取得计划不算分析。FIXED TABLE FULL 是固定表访问，不等于业务大表全扫。保留技术术语原文。",
                "evidence": "累计值可用于单次开销和机制分析；仅窗口排名、退化和效果验证要求对应窗口。缺基线不能抹去已有计划、锁等待或错误证据。",
                "next_evidence": "只有会改变当前问题结论、当前授权可能补齐的关键缺口才填写 next_evidence（每项说明要验证什么及如何改变判断）。可使用数据库、监控、日志等任意适用来源；一般性限制只写结论边界，不触发补证。",
                "oracle": "Oracle 不降级为三库最低共同能力。已有XPLAN时解释相关谓词和Outline，但不把Outline当作调优建议或某次生产执行。已有Monitor时分析精确单次执行的CPU/等待、I/O及本实例进程行源/工作区，结合匹配计划行解释；不能相加父子或PX计数。ASH用于按计划行、ON CPU/等待及阻塞者的采样归因，样本数/比例不是精确耗时，不能用没有样本证明健康。需要且尚未尝试的Oracle证据用oracle_evidence_requests声明kind、真实subject_id和改变判断的reason，不在next_evidence重复；只申请oracle_evidence_policy中许可允许的工具，ASH显式历史窗口必须带时区且不超过24小时，否则使用已冻结默认窗口；不能冒充未覆盖的历史故障。原生HTML报告按需下载，不作为所有SQL诊断的前置条件。",
                "recommendations": "没有证据支持问题时不强造优化方案；OBSERVE/VERIFY无需虚构变更风险和回滚。变更建议仍须具备条件、风险、验证、回退和审批。",
                "continuity": "之前的阶段判断不是新事实；新证据反驳时明确修正，补证失败不抹去仍成立的局部判断。",
            },
        },
        deadline=datetime.fromisoformat(context.deadline_at) - timedelta(seconds=3) if getattr(context, "deadline_at", None) else None,
        idempotency_key=f"turn:{context.run_id}:sql-diagnosis:{context.task_key}:{context.attempt}",
    )
    draft = expand_diagnostic_response(output_model.model_validate(result.output.model_dump(mode="python")), bundle, aliases)
    return SqlDiagnosticCheckpoint(draft=draft, scope_hash=scope_hash(task_frame, database_type),
        evidence_hash=evidence_hash(evidence), model_receipt=result.receipt.model_dump(mode="json") if result.receipt else None)
