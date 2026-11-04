"""从可信 AIOps 原信封恢复白名单定位，不公开任意 safe_data。"""

from uuid import UUID

from pydantic import ValidationError

from platform_core.notifications import NotificationEnvelope


AIOPS_OUTPUT_EVENTS = frozenset({
    "aiops.output_job.started", "aiops.output_job.failed", "aiops.case.draft_ready", "aiops.report.ready",
})
AIOPS_RESOURCE_EVENTS = {
    "aiops.situation.detected": ("aiops_situation", "aiops.situation.detail", "situation_id"),
    "aiops.situation.recovered": ("aiops_situation", "aiops.situation.detail", "situation_id"),
    "aiops.diagnosis.started": ("aiops_run", "aiops.run.detail", "run_id"),
    "aiops.diagnostic.input_required": ("aiops_run", "aiops.run.detail", "run_id"),
    "aiops.run.failed": ("aiops_run", "aiops.run.detail", "run_id"),
    "aiops.knowledge.review_required": ("aiops_run", "aiops.run.detail", "run_id"),
    "aiops.proposal.review_required": ("aiops_proposal", "aiops.run.detail", "proposal_id"),
    "aiops.proposal.approved": ("aiops_proposal", "aiops.run.detail", "proposal_id"),
    "aiops.proposal.rejected": ("aiops_proposal", "aiops.run.detail", "proposal_id"),
}


def aiops_notification_route(row, payload, *, domain_id: int):
    """资源与收件页身份必须一致；缺失／错误原件没有可点击结果。"""
    try:
        envelope = NotificationEnvelope.model_validate(payload)
        if (envelope.domain_id != domain_id or envelope.domain_id != int(row.domain_id)
            or envelope.event_type != row.event_type
            or row.event_type not in AIOPS_OUTPUT_EVENTS | AIOPS_RESOURCE_EVENTS.keys()
            or envelope.resource_type != row.resource_type or envelope.resource_id != row.resource_id):
            return {}
        safe = envelope.safe_data
        params = {"domain_id": str(domain_id)}
        if row.event_type in AIOPS_RESOURCE_EVENTS:
            resource_type, route_key, id_key = AIOPS_RESOURCE_EVENTS[row.event_type]
            if row.resource_type != resource_type:
                return {}
            params[id_key] = str(UUID(row.resource_id))
            if resource_type == "aiops_proposal":
                params["run_id"] = str(UUID(safe["run_id"]))
            return {"route_key": route_key, "route_params": params}
        if row.event_type == "aiops.report.ready":
            if row.resource_type != "aiops_report":
                return {}
            params["report_id"] = str(UUID(row.resource_id))
            if safe.get("job_id") is not None:
                if safe.get("job_type") != "REPORT_GENERATION":
                    return {}
                params["job_id"] = str(UUID(safe["job_id"]))
                version = safe.get("report_version")
                if isinstance(version, bool) or not str(version).isdigit() or int(version) < 1:
                    return {}
                params["report_version"] = str(version)
            return {"route_key": "aiops.report.detail", "route_params": params}
        if row.resource_type != "aiops_output_job":
            return {}
        params["job_id"] = str(UUID(row.resource_id))
        if params["job_id"] != str(UUID(safe["job_id"])):
            return {}
        params["job_type"] = safe["job_type"]
        if params["job_type"] not in {"CASE_EXTRACTION", "REPORT_GENERATION"}:
            return {}
        if row.event_type == "aiops.case.draft_ready":
            if params["job_type"] != "CASE_EXTRACTION":
                return {}
            params.update(asset_id=str(UUID(safe["asset_id"])),
                asset_version_id=str(UUID(safe["asset_version_id"])))
        return {"route_key": "aiops.output_job.detail", "route_params": params}
    except (ValidationError, ValueError, TypeError, KeyError, AttributeError):
        return {}
