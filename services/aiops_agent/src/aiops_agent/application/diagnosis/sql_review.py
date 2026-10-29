"""SQL 调查与回答共享一次诊断，不再先评估后丢弃推理结果。"""

from datetime import datetime, timedelta

from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle, validate_sql_diagnosis
from aiops_agent.contracts.sql_diagnosis import SqlDiagnosisDraft, SqlDiagnosisReview, SqlDiagnosticCheckpoint
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
    return result


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
    except (ValueError, TypeError):
        return None
    return draft


async def generate_sql_review(*, context, task_frame, database_type, evidence, gaps, model, prompts, previous=None):
    answer = dict(context.plan_snapshot.get("answer_context") or {})
    bundle = build_sql_evidence_bundle(task_frame=task_frame, database_type=database_type, evidence=evidence)
    prompt = await prompts.resolve("sql_diagnosis", frozen_prompts=dict(answer.get("prompts") or {}))
    result = await model.generate_structured(
        purpose="aiops.sql-diagnosis", output_model=SqlDiagnosisReview,
        model_snapshot=dict(answer.get("model") or {}), prompt_ref={**prompt.ref(), "content": prompt.content},
        input_payload={
            "question": answer.get("question", ""), "evidence_bundle": review_context(bundle),
            "parent_investigation": answer.get("parent_investigation"),
            "previous_diagnosis": previous.draft.model_dump(mode="json") if previous else None,
            "goal_catalog": [{"goal_id": f"g{index}", "description": value} for index, value in enumerate(task_frame.get("success_criteria") or (), 1)]
                or [{"goal_id": "answer", "description": "回答当前问题"}],
            "gaps": [gap.model_dump(mode="json") for gap in gaps],
            "delivery_requirements": {
                "format": "这是交互诊断，不是完整报告。保留回答问题所需的最少结论、引用和措施；不重复原始表格、每条限制或全套运维流程。",
                "verdict": "必须区分 PROBLEM_FOUND、NO_OBVIOUS_ISSUE、INSUFFICIENT_EVIDENCE，先正面回答；没有发现问题时明确检查范围及暂不变更，不能凭均值低或缺证判断健康。",
                "execution_plan": "解释已有匹配计划的关键节点和机制，而非仅串联节点；FIXED TABLE FULL 是固定表访问，不等于业务大表全扫。保留技术术语原文。",
                "evidence": "累计值可用于单次开销和机制分析；仅窗口排名、退化和效果验证要求对应窗口。缺基线不能抹去已有计划、锁等待或错误证据。",
                "next_evidence": "只有会改变当前问题结论、当前授权可能补齐的关键缺口才填写 next_evidence（每项说明要验证什么及如何改变判断）。可使用数据库、监控、日志等任意适用来源；一般性限制只写结论边界，不触发补证。",
                "recommendations": "没有证据支持问题时不强造优化方案；OBSERVE/VERIFY无需虚构变更风险和回滚。变更建议仍须具备条件、风险、验证、回退和审批。",
                "continuity": "之前的阶段判断不是新事实；新证据反驳时明确修正，补证失败不抹去仍成立的局部判断。",
            },
        },
        deadline=datetime.fromisoformat(context.deadline_at) - timedelta(seconds=3) if getattr(context, "deadline_at", None) else None,
        idempotency_key=f"turn:{context.run_id}:sql-diagnosis:{context.task_key}:{context.attempt}",
    )
    draft = SqlDiagnosisReview.model_validate(result.output.model_dump(mode="python"))
    validate_sql_diagnosis(draft, bundle)
    return SqlDiagnosticCheckpoint(draft=draft, scope_hash=scope_hash(task_frame, database_type),
        evidence_hash=evidence_hash(evidence), model_receipt=result.receipt.model_dump(mode="json") if result.receipt else None)
