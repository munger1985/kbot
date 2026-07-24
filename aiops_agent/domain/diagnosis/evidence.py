"""把监控和数据库 Artifact 归一为可引用事实。"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from aiops_agent.contracts.diagnosis import EvidenceFact, EvidenceIndex


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _fact(
    *,
    artifact_id: str,
    pointer: str,
    source_type: str,
    source_group_id: str,
    target_id: str,
    fact_type: str,
    value: Any,
    summary: str,
    unit: str | None = None,
    dimensions: dict[str, str] | None = None,
    window_start=None,
    window_end=None,
    captured_at=None,
    quality_flags: tuple[str, ...] = (),
) -> EvidenceFact:
    basis = {
        "artifact_id": artifact_id,
        "pointer": pointer,
        "value": value,
        "fact_type": fact_type,
    }
    return EvidenceFact(
        fact_id=_digest(basis),
        source_artifact_id=artifact_id,
        source_json_pointer=pointer,
        source_type=source_type,
        source_group_id=source_group_id,
        trust_level="SOURCE_VERIFIED",
        target_id=target_id,
        observed_subject=target_id,
        metric_or_fact_type=fact_type,
        value=value,
        unit=unit,
        dimensions=dimensions or {},
        window_start=window_start,
        window_end=window_end,
        captured_at=captured_at,
        quality_flags=quality_flags,
        fact_summary=summary[:2000],
    )


def normalize_evidence_artifacts(
    artifacts: tuple[dict[str, Any], ...],
    *,
    target_id: str,
    max_facts: int | None = None,
) -> EvidenceIndex:
    facts: list[EvidenceFact] = []
    gaps: list[dict[str, Any]] = []
    expanded: list[dict[str, Any]] = []
    for artifact in artifacts:
        if (
            artifact["schema_version"]
            == "DIAGNOSIS_EVIDENCE_COLLECTION.v1"
        ):
            for index, result in enumerate(
                artifact["payload"].get("results", [])
            ):
                expanded.append(
                    {
                        "artifact_id": artifact["artifact_id"],
                        "schema_version": "DATABASE_DIAGNOSTIC_RESULT.v1",
                        "payload": result,
                        "_pointer_prefix": f"/results/{index}",
                    }
                )
        elif artifact["schema_version"] == "EVIDENCE_INDEX.v1":
            expanded.append(artifact)
        else:
            expanded.append(artifact)
    for artifact in expanded:
        schema = artifact["schema_version"]
        payload = artifact["payload"]
        artifact_id = artifact["artifact_id"]
        if schema == "KNOWLEDGE_CITATION_PACK.v1":
            if payload.get("gap_code"):
                gaps.append(
                    {
                        "code": payload["gap_code"],
                        "detail": "Knowledge Core SOP 证据本次不可用",
                        "retryable": True,
                    }
                )
            for citation_index, citation in enumerate(
                payload.get("citations", [])
            ):
                items = citation.get("items", [])
                summary_parts = []
                for item in items[:5]:
                    evidence = item.get("evidence", {})
                    text = (
                        evidence.get("retrieval_text")
                        or evidence.get("content_text")
                        or ""
                    )
                    if text:
                        summary_parts.append(str(text)[:400])
                summary = "；".join(summary_parts) or "知识库引用"
                pointer = f"/citations/{citation_index}"
                facts.append(
                    EvidenceFact(
                        fact_id=_digest(
                            {
                                "artifact_id": artifact_id,
                                "pointer": pointer,
                                "citation": citation,
                            }
                        ),
                        source_artifact_id=artifact_id,
                        source_json_pointer=pointer,
                        source_type="KNOWLEDGE_CITATION",
                        source_group_id=f"knowledge:{artifact_id}",
                        trust_level="KNOWLEDGE_CITATION",
                        target_id=target_id,
                        observed_subject="product-knowledge",
                        metric_or_fact_type="knowledge.sop",
                        value={
                            "citation_label": citation.get(
                                "citation_label"
                            ),
                            "bundle_id": citation.get("bundle_id"),
                        },
                        fact_summary=summary[:2000],
                    )
                )
        elif schema == "EVIDENCE_INDEX.v1":
            facts.extend(
                EvidenceFact.model_validate(item)
                for item in payload.get("facts", [])
            )
            gaps.extend(payload.get("gaps", []))
        elif schema == "OBSERVATION_SET.v1":
            source_group = f"monitor:{artifact_id}"
            for metric_index, metric in enumerate(
                payload.get("observations", [])
            ):
                summary = dict(metric.get("summary", {}))
                flags = list(metric.get("warnings", []))
                if metric.get("truncated"):
                    flags.append("TRUNCATED")
                if float(metric.get("coverage_ratio", 1)) < 0.8:
                    flags.append("LOW_COVERAGE")
                for name, value in sorted(summary.items()):
                    pointer = (
                        f"/observations/{metric_index}/summary/{name}"
                    )
                    facts.append(
                        _fact(
                            artifact_id=artifact_id,
                            pointer=pointer,
                            source_type="MONITOR_METRIC",
                            source_group_id=source_group,
                            target_id=target_id,
                            fact_type=f"{metric['metric_code']}.{name}",
                            value=value,
                            unit=metric.get("unit"),
                            dimensions={
                                "source_id": str(metric["source_id"]),
                                "binding_id": str(metric["binding_id"]),
                            },
                            window_start=metric.get("window_start"),
                            window_end=metric.get("window_end"),
                            quality_flags=tuple(sorted(set(flags))),
                            summary=(
                                f"指标 {metric['metric_code']} 的 {name}="
                                f"{value} {metric.get('unit', '')}".strip()
                            ),
                        )
                    )
            for alert_index, alert in enumerate(
                payload.get("active_alerts", [])
            ):
                facts.append(
                    _fact(
                        artifact_id=artifact_id,
                        pointer=f"/active_alerts/{alert_index}",
                        source_type="MONITOR_ALERT",
                        source_group_id=source_group,
                        target_id=target_id,
                        fact_type="monitor.alert.active",
                        value=alert,
                        summary=(
                            "存在活动监控告警："
                            + str(
                                alert.get("summary")
                                or alert.get("name")
                                or "未命名告警"
                            )
                        ),
                    )
                )
            gaps.extend(payload.get("gaps", []))
        elif schema == "DATABASE_DIAGNOSTIC_RESULT.v1":
            if payload.get("status") != "SUCCEEDED":
                if payload.get("gap"):
                    gaps.append(payload["gap"])
                continue
            observation = payload["observation"]
            prefix = artifact.get("_pointer_prefix", "")
            source_group = f"database:{artifact_id}:{prefix or '/'}"
            columns = [
                item["name"] for item in observation.get("columns", [])
            ]
            for row_index, row in enumerate(observation.get("rows", [])):
                row_value = dict(zip(columns, row, strict=True))
                flags = (
                    ("TRUNCATED",)
                    if observation.get("truncated")
                    else ()
                )
                facts.append(
                    _fact(
                        artifact_id=artifact_id,
                        pointer=(
                            f"{prefix}/observation/rows/{row_index}"
                        ),
                        source_type="DATABASE_OBSERVATION",
                        source_group_id=source_group,
                        target_id=target_id,
                        fact_type=observation["tool_id"],
                        value=row_value,
                        captured_at=observation.get("captured_at"),
                        quality_flags=flags,
                        summary=(
                            f"数据库工具 {observation['tool_id']} 返回"
                            f"记录：{_safe_row_summary(row_value)}"
                        ),
                    )
                )
    facts = list({item.fact_id: item for item in facts}.values())
    facts.sort(key=lambda item: item.fact_id)
    if max_facts is not None and len(facts) > max_facts:
        gaps.append(
            {
                "code": "EVIDENCE_FACT_LIMIT_REACHED",
                "detail": "Evidence Index 已按服务端上限截断",
                "retryable": False,
            }
        )
        facts = facts[:max_facts]
    index_payload = [
        {
            "fact_id": item.fact_id,
            "source_group_id": item.source_group_id,
            "quality_flags": item.quality_flags,
        }
        for item in facts
    ]
    return EvidenceIndex(
        target_id=target_id,
        facts=tuple(facts),
        gaps=tuple(gaps),
        fact_count=len(facts),
        source_group_count=len({item.source_group_id for item in facts}),
        index_hash=_digest(index_payload),
    )


def _safe_row_summary(row: dict[str, Any]) -> str:
    parts = []
    for key, value in list(row.items())[:12]:
        rendered = str(value)
        if len(rendered) > 120:
            rendered = rendered[:117] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)[:1500]
