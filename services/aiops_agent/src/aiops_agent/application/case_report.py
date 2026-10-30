"""正式报告提炼案例的公共来源入口；不把建议或人工反馈变成已验证效果。"""

from dataclasses import dataclass
from uuid import UUID

from aiops_agent.application.case_source_validation import validate_case_source_rows
from aiops_agent.application.errors import AIOpsApplicationError
from aiops_agent.contracts.knowledge import DiagnosisCaseProfile
from aiops_agent.playbooks import canonical_hash


def rejected_source():
    return AIOpsApplicationError(
        code="AIOPS_KNOWLEDGE_SOURCE_REJECTED",
        message="报告或原始证据已变化、不可核验或授权已撤销，不能提炼或复用案例",
        status_code=422,
    )


@dataclass(frozen=True)
class ReportCaseSource:
    source_report_id: object
    source_run_id: object
    source_artifact_id: object
    content_hash: str


async def load_case_report(*, uow, domain_id, report_id, expected_hash=None, plan_capture_reader=None):
    """在当前事务重新读取源；同一冻结内容的模型前、索引前和落库后复核共用。"""
    report = await uow.inspections.get_report_scoped(report_id=report_id, domain_id=domain_id)
    if (report is None or not report.is_current or report.status not in {"READY", "PARTIAL"}
        or not report.content_hash or report.content_artifact_id is None
        or expected_hash is not None and report.content_hash != expected_hash):
        raise rejected_source()
    artifact = await uow.runs.get_artifact(artifact_id=report.content_artifact_id)
    if (artifact is None or artifact.schema_version != "REPORT_CONTENT.v1"
        or str(artifact.ops_run_id) != str(report.ops_run_id)
        or artifact.content_hash != report.content_hash
        or canonical_hash(artifact.payload_json) != artifact.content_hash):
        raise rejected_source()
    content = dict(artifact.payload_json or {})
    if (str(content.get("ops_run_id")) != str(report.ops_run_id)
        or str(content.get("target_id")) != str(report.target_id)):
        raise rejected_source()
    sources = await uow.inspections.list_report_sources(report_id=report_id)
    if not sources:
        raise rejected_source()
    closure_id = dict(content.get("scope") or {}).get("closure_artifact_id")
    if closure_id:
        try:
            closure_uuid = UUID(str(closure_id))
        except (TypeError, ValueError):
            raise rejected_source() from None
        if str(closure_uuid) not in {str(row.source_artifact_id) for row in sources}:
            raise rejected_source()
        closure = await uow.runs.get_artifact(artifact_id=closure_uuid)
        if closure is None or closure.schema_version != "DIAGNOSIS_CLOSURE.v1":
            raise rejected_source()
    rows = [
        ReportCaseSource(report.report_id, report.ops_run_id, report.content_artifact_id, report.content_hash),
        *(ReportCaseSource(report.report_id, row.ops_run_id, row.source_artifact_id, row.content_hash) for row in sources),
    ]
    if not await validate_case_source_rows(
        uow=uow, domain_id=domain_id, sources=rows, plan_capture_reader=plan_capture_reader,
    ):
        raise rejected_source()
    # 报告中的判断必须能逐项回到实际冻结记录，不能只检查几个非空字段。
    from aiops_agent.application.diagnosis.investigation_record import investigation_report_source
    from aiops_agent.contracts.investigation import InvestigationRecord
    originals = {}
    for row in sources:
        source = await uow.runs.get_artifact(artifact_id=row.source_artifact_id)
        record = dict(source.payload_json or {}).get("investigation_record")
        if record:
            projected = investigation_report_source(InvestigationRecord.model_validate(record))
            for fact in projected["facts"]:
                key = (fact["source_run_id"], fact["record_id"], fact["record_revision"], fact["claim_id"])
                originals[key] = fact
    for fact in content.get("facts") or ():
        if not isinstance(fact, dict) or not fact.get("claim_id"):
            continue
        key = (fact.get("source_run_id"), fact.get("record_id"), fact.get("record_revision"), fact.get("claim_id"))
        original = originals.get(key)
        if original is None or any(fact.get(name) != value for name, value in original.items()):
            raise rejected_source()
    return report, content, sources


async def profile_from_diagnosis_report(*, uow, report, content, sources, target):
    """已执行动作只继承原收口记录，不把整份报告引用机械附到每项动作。"""
    scope = dict(content.get("scope") or {})
    verification = dict(scope.get("verification") or {})
    actions = ()
    closure_id = scope.get("closure_artifact_id")
    if closure_id:
        try:
            identity = UUID(str(closure_id))
        except (TypeError, ValueError):
            raise rejected_source() from None
        if str(identity) not in {str(row.source_artifact_id) for row in sources}:
            raise rejected_source()
        closure = await uow.runs.get_artifact(artifact_id=identity)
        if closure is None or closure.schema_version != "DIAGNOSIS_CLOSURE.v1":
            raise rejected_source()
        frozen = dict(closure.payload_json.get("profile") or {})
        verification = dict(frozen.get("verification") or {})
        # 人工登记即使引用回执也仍是人工叙述；引用不能替代受控效果重放。
        verification.setdefault("method", "USER_REPORTED")
        actions = tuple({
            **dict(item), "source_kind": "USER_PROVIDED",
            "registration_ref": f"artifact:{identity}",
            "evidence_refs": list(item.get("evidence_refs") or ()),
        } for item in frozen.get("actions") or () if isinstance(item, dict) and item.get("summary"))
    elif scope.get("executed_actions"):
        actions = tuple({
            "summary": str(item), "source_kind": scope.get("action_source") or "REPORT_DECLARED",
            "evidence_refs": [],
        } for item in scope["executed_actions"] if str(item).strip())
    verification.setdefault("result", "NOT_VERIFIED")
    # 失败经验有确切尝试和失败反馈；有无受控测量继续由 method 等原字段表明。
    negative = (actions and all(item.get("evidence_refs") for item in actions)
        and verification["result"] in {"NOT_ACHIEVED", "ADVERSE", "DEGRADED", "UNCHANGED"})
    facts = [dict(item) for item in content.get("facts") or ()
        if isinstance(item, dict) and item.get("claim_id") and item.get("level") in {"OBSERVED", "SUPPORTED"}]
    claim_refs = [{key: item[key] for key in ("source_run_id", "record_id", "record_revision", "claim_id", "evidence_refs")}
        for item in facts]
    refs = list(dict.fromkeys(ref for item in facts for ref in item.get("evidence_refs") or ()))
    conclusions = [item["summary"] for item in facts if item.get("kind") == "CAUSE"] or [
        item["summary"] for item in facts
    ]
    return DiagnosisCaseProfile(
        case_kind="NEGATIVE_EXPERIENCE" if negative else "DIAGNOSTIC_REFERENCE",
        problem_signature={
            "problem_class": report.report_type, "component": None, "error_codes": [],
            "normalized_symptoms": [str(item["summary"]) for item in facts] or [report.title], "signal_names": [],
        },
        environment={
            "database_type": target.db_type, "database_version": target.version_code,
            "platform": target.environment, "topology": (target.capabilities_json or {}).get("topology"),
        },
        root_cause={
            "summary": "\n".join(conclusions), "confidence": "UNKNOWN",
            "effective_level": str(scope.get("root_cause_grade") or "INCONCLUSIVE"), "evidence_refs": refs,
        },
        actions=actions, verification=verification,
        source={
            "report_id": str(report.report_id), "content_hash": report.content_hash,
            "run_ids": sorted({str(row.ops_run_id) for row in sources}), "claim_refs": claim_refs,
        },
    )
