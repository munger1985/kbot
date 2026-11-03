"""业务来源版本的顺序规则；接收时间不构成任务状态的先后依据。"""

from typing import Literal


def source_version_decision(*, incoming: int, current: int | None,
    incoming_status: str, current_status: str | None) -> Literal["ADVANCE", "DELIVER", "IGNORE"]:
    if type(incoming) is not int or incoming < 1:
        raise ValueError("通知事件缺少合法业务来源版本")
    if current is None or incoming > current:
        return "ADVANCE"
    if incoming == current and incoming_status == current_status:
        # 对账可能先于终态 Outbox 消费；仍须投递尚未消费的同版本通知。
        return "DELIVER"
    return "IGNORE"


def output_event_version(*, producer: str, event_type: str, params: dict, data: dict) -> int | None:
    is_output = producer == "aiops" and (
        event_type in {"aiops.output_job.started", "aiops.output_job.failed", "aiops.case.draft_ready"}
        or (event_type == "aiops.report.ready" and params.get("job_id") is not None)
    )
    if not is_output:
        return None
    version = data.get("source_version")
    if type(version) is not int or version < 1:
        raise ValueError("输出任务事件缺少合法业务来源版本")
    return version
