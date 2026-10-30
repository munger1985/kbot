"""区分报告的展示修订与实质更正；不以新版本标签替代原件及授权复核。"""

from aiops_agent.playbooks import canonical_hash


MAX_REPORT_REVISIONS = 64


def report_diagnosis_identity(payload):
    """仅排除确定的展示信息；章节人工改写、判断、建议及未知字段均参与比较。"""
    content = dict(payload)
    content.pop("title", None)
    provenance = dict(content.get("provenance") or {})
    for key in ("template", "human_presentation_edit", "report_input_hash"):
        provenance.pop(key, None)
    content["provenance"] = provenance
    return canonical_hash(content)


def same_report_diagnosis(before, after):
    """展示编辑不获得事实豁免；文本章节变化保守地视为实质更正。"""
    return report_diagnosis_identity(before) == report_diagnosis_identity(after)


async def _content(uow, report):
    if (report.status not in {"READY", "PARTIAL"} or not report.content_hash
        or report.content_artifact_id is None):
        return None
    artifact = await uow.runs.get_artifact(artifact_id=report.content_artifact_id)
    if (artifact is None or artifact.schema_version != "REPORT_CONTENT.v1"
        or not isinstance(artifact.payload_json, dict)
        or not isinstance(artifact.payload_json.get("provenance", {}), dict)
        or str(artifact.ops_run_id) != str(report.ops_run_id)
        or artifact.security_level != report.security_level
        or artifact.content_hash != report.content_hash
        or canonical_hash(artifact.payload_json) != report.content_hash):
        return None
    content = dict(artifact.payload_json or {})
    if (str(content.get("ops_run_id")) != str(report.ops_run_id)
        or str(content.get("target_id")) != str(report.target_id)
        or content.get("report_key") != report.report_key
        or content.get("report_type") != report.report_type
        or content.get("status") != report.status):
        return None
    return content


async def equivalent_report_revisions(*, uow, before, after):
    """原始证据登记、范围与安全级必须相同，不能只比较渲染文字。"""
    fields = ("ops_run_id", "target_id", "source_conversation_id", "report_key", "report_type",
        "status", "period_start", "period_end", "baseline_start", "baseline_end",
        "after_start", "after_end", "result", "security_level")
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        return False
    old, new = await _content(uow, before), await _content(uow, after)
    if old is None or new is None or not same_report_diagnosis(old, new):
        return False
    def identity(rows):
        return sorted((str(row.ops_run_id), str(row.source_artifact_id), row.content_hash,
            row.source_kind, str(row.observed_at)) for row in rows)
    old_sources = await uow.inspections.list_report_sources(report_id=before.report_id)
    new_sources = await uow.inspections.list_report_sources(report_id=after.report_id)
    return bool(old_sources) and identity(old_sources) == identity(new_sources)


async def report_source_is_usable(*, uow, domain_id, report):
    """旧报告只沿每条确切的纯展示修订到当前版本；中途技术更正后再改回也不放行。"""
    if report is None:
        return False
    if report.is_current:
        return report.status in {"READY", "PARTIAL"}
    seen = {str(report.report_id)}
    for _ in range(MAX_REPORT_REVISIONS):
        following = await uow.inspections.list_report_successors_scoped(
            report_id=report.report_id, domain_id=domain_id,
        )
        if len(following) != 1:
            return False
        current = following[0]
        if (str(current.report_id) in seen
            or str(current.supersedes_report_id) != str(report.report_id)
            or int(current.report_version) != int(report.report_version) + 1
            or not await equivalent_report_revisions(uow=uow, before=report, after=current)):
            return False
        seen.add(str(current.report_id))
        if current.is_current:
            return True
        report = current
    return False


async def report_ancestors(*, uow, domain_id, report):
    """沿确切父版本寻找旧案例和受影响源，不跨 Target、逻辑报告或授权域。"""
    ancestors, seen = [], {str(report.report_id)}
    for _ in range(MAX_REPORT_REVISIONS):
        if report.supersedes_report_id is None:
            break
        previous = await uow.inspections.get_report_scoped(
            report_id=report.supersedes_report_id, domain_id=domain_id,
        )
        if (previous is None or str(previous.report_id) in seen
            or any(getattr(previous, field) != getattr(report, field)
                for field in ("target_id", "report_key", "source_conversation_id"))
            or int(previous.report_version) + 1 != int(report.report_version)):
            break
        ancestors.append(previous)
        seen.add(str(previous.report_id))
        report = previous
    return ancestors
