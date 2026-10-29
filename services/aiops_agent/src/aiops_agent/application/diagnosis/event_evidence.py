"""把活动事件投影为独立证据，并在报告/案例复核时重放相同事实。"""

from aiops_agent.contracts.measurements import EvidenceProvenance
from aiops_agent.contracts.turn_answer import TurnEvidenceFact
from aiops_agent.playbooks import canonical_hash


EVENT_COLUMNS = (
    "event_id", "source_type", "name", "severity", "status", "is_open", "active_at", "updated_at",
)


def _text(value, maximum=4000):
    return value[:maximum] if isinstance(value, str) else ""


def source_event_fact(*, artifact_id, result):
    """原生事件是供应商观测，不是SQL因果、全部历史告警或独立来源增信。"""
    query = result.event_query
    if query is None and not result.active_alerts:
        return None
    rows = tuple((
        _text(item.get("event_id") or item.get("incident_id"), 256),
        _text(item.get("source_type"), 64),
        _text(item.get("name")), _text(item.get("severity"), 64),
        _text(item.get("status"), 64),
        item.get("is_open") if type(item.get("is_open")) is bool else None,
        _text(item.get("active_at"), 64), _text(item.get("updated_at"), 64),
    ) for item in result.active_alerts)
    scopes = {
        "CURRENT_OPEN_INCIDENTS": (
            "OEM仅返回采集时仍未关闭、且创建时间落在请求半开窗口内的Incident；"
            "不覆盖已关闭或更早创建但窗口内仍活跃的事件，零行不等于历史无告警。"),
        "CURRENT_ALERTS": (
            "Prometheus仅返回当前规则告警，未按请求窗口还原历史状态；"
            "包含可能尚未触发的pending规则，零行不等于历史无告警。"),
        "RECENT_PROBLEMS": (
            "Zabbix recent问题包含当前及供应商留存的近期已恢复问题；"
            "恢复状态依来源恢复事件标识判断，缺失时保持未知；未覆盖完整请求历史窗口，零行不等于历史无告警。"),
    }
    warnings = (
        scopes[query.query_scope] if query is not None else
        "事件为供应商当前观测；未取得完整历史覆盖证明，不能推断窗口内无其他告警。",
        "监控告警本身不是数据库根因，须结合数据库、SQL、等待及其他独立证据核验。",
        *(("事件单页达到上限或来源仍有后页；结果可能不完整，未跟随来源URL。",)
          if query is not None and query.truncated else ()),
    )
    provenance = EvidenceProvenance(
        target_id=str(result.target_id), native_origin_ids=(f"database:{result.target_id}",),
        source_version=query.source_version if query is not None else "EVENT_SOURCE_VERSION_UNKNOWN",
        content_hash=canonical_hash(result.model_dump(mode="json")),
        window_start=query.window_start if query is not None else None,
        window_end=query.window_end if query is not None else None,
        quality_flags=(
            "PHYSICAL_IDENTITY_UNKNOWN", "NATIVE_ORIGIN_COARSE", "EVENT_HISTORY_INCOMPLETE",
            *((query.query_scope,) if query is not None else ("EVENT_QUERY_SCOPE_UNKNOWN",)),
            *(("TRUNCATED",) if query is not None and query.truncated else ()),
        ),
    )
    return TurnEvidenceFact(
        evidence_ref=f"artifact:{artifact_id}#source-events", artifact_id=str(artifact_id),
        source_id=str(result.source_id) if result.source_id is not None else "monitoring.events",
        step_id="events", tool_id="monitor.event.query", trust_level="SOURCE_VERIFIED",
        measurement_semantics="CURRENT_ACTIVITY", presentation_kind="TABLE",
        captured_at=(query.captured_at if query is not None else result.collected_at).isoformat(),
        columns=tuple({"name": name} for name in EVENT_COLUMNS), rows=rows, row_count=len(rows),
        truncated=query.truncated if query is not None else False, warnings=warnings, provenance=provenance,
    )
