"""MySQL/PostgreSQL 工作负载报告的确定性计算测试。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from aiops_agent.application.reporting import (
    render_pdf,
    report_presentation,
    resolve_historical_report_template,
)
from aiops_agent.application.workload import (
    _report_content_payload,
    build_activity_report,
    build_workload_diff_report,
    build_workload_report,
    counter_delta,
    sanitize_statement_text,
)
from aiops_agent.contracts.report import ReportContent
from platform_core.contracts.aiops.workload import (
    ActivitySample,
    StatementIdentityType,
    WorkloadReportType,
    WorkloadStatementSnapshot,
)
from platform_core.identity import uuid7


STARTED_AT = datetime(2026, 9, 22, 1, 0, tzinfo=UTC)
TARGET_ID = uuid7()
DIGEST_A = "A" * 64
DIGEST_B = "B" * 64


def _snapshot(*, minute: int, questions: int, server_uuid: str = "server-a"):
    return SimpleNamespace(
        workload_snapshot_id=uuid7(),
        collected_at=STARTED_AT + timedelta(minutes=minute),
        database_type="MYSQL",
        instance_identity_json={
            "identity_type": "MYSQL_SERVER_UUID",
            "identity_value": server_uuid,
        },
        instance_started_at=STARTED_AT,
        database_version="8.4.0",
        catalog_version="catalog-a",
        capability_probe_version="mysql-capabilities.v1",
        instance_metrics_json={
            "questions": questions,
            "commits": questions // 2,
        },
        coverage_json={"STATEMENT": "AVAILABLE", "WAIT": "AVAILABLE"},
    )


def _statement(snapshot, digest: str, *, executions: int, latency: int):
    return SimpleNamespace(
        workload_snapshot_id=snapshot.workload_snapshot_id,
        statement_identity_type="MYSQL_DIGEST",
        statement_identity_value=digest,
        database_name="app",
        user_identifier="",
        top_level=-1,
        normalized_statement=(
            "SELECT * FROM orders WHERE customer_id = 123 "
            "AND token = 'secret'"
        ),
        execution_count=executions,
        total_duration_microseconds=latency,
        mean_duration_microseconds=latency // executions,
        max_duration_microseconds=latency,
        rows_processed=executions,
    )


def _metric(
    snapshot,
    *,
    family: str,
    code: str,
    dimension: str,
    value: int,
    quality: str = "AVAILABLE",
):
    return SimpleNamespace(
        workload_snapshot_id=snapshot.workload_snapshot_id,
        metric_family=family,
        metric_code=code,
        dimension_key=dimension,
        quality_status=quality,
        counter_value=value,
    )


def _activity_sample(*, index: int = 0):
    return SimpleNamespace(
        activity_sample_id=uuid7(),
        sampled_at=STARTED_AT + timedelta(seconds=index * 5),
        sample_weight=1,
        quality_status="AVAILABLE",
        database_type="MYSQL",
        statement_identity_value=DIGEST_A,
        schema_name="app",
        command_name="Query",
        session_state="executing",
        stage_name="stage/sql/executing",
        wait_name=None,
    )


def test_counter_delta_requires_stable_instance_identity() -> None:
    result = counter_delta(
        {"questions": 10},
        {"questions": 20},
        previous_instance_identity={"identity_value": "server-a"},
        current_instance_identity={"identity_value": "server-b"},
        previous_started_at=STARTED_AT,
        current_started_at=STARTED_AT,
    )

    assert result == {
        "status": "DISCONTINUITY",
        "reason": "SERVER_RESTART_OR_IDENTITY_CHANGED",
        "deltas": {},
    }


def test_counter_delta_rejects_counter_reset() -> None:
    result = counter_delta(
        {"questions": 20, "commits": 8},
        {"questions": 10, "commits": 9},
        previous_instance_identity={"identity_value": "server-a"},
        current_instance_identity={"identity_value": "server-a"},
        previous_started_at=STARTED_AT,
        current_started_at=STARTED_AT,
    )

    assert result["status"] == "DISCONTINUITY"
    assert result["reason"] == "COUNTER_RESET"
    assert result["reset_metrics"] == ["questions"]


def test_workload_report_calculates_only_continuous_deltas() -> None:
    first = _snapshot(minute=0, questions=100)
    second = _snapshot(minute=5, questions=160)
    report = build_workload_report(
        target_id=TARGET_ID,
        snapshots=(second, first),
        statements=(
            _statement(first, DIGEST_A, executions=10, latency=1_000),
            _statement(second, DIGEST_A, executions=14, latency=1_800),
        ),
    )

    assert report["status"] == "READY"
    assert report["continuous_seconds"] == 300
    assert report["load_totals"]["questions"] == 60
    assert report["load_profile"]["questions_per_second"] == 0.2
    assert report["top_statements"][0]["execution_count"] == 4
    assert report["top_statements"][0]["mean_duration_microseconds"] == 200
    assert "123" not in report["top_statements"][0]["normalized_statement"]
    assert "secret" not in report["top_statements"][0]["normalized_statement"]


def test_workload_report_marks_new_and_evicted_digests() -> None:
    first = _snapshot(minute=0, questions=100)
    second = _snapshot(minute=5, questions=120)
    report = build_workload_report(
        target_id=TARGET_ID,
        snapshots=(first, second),
        statements=(
            _statement(first, DIGEST_A, executions=10, latency=1_000),
            _statement(second, DIGEST_B, executions=2, latency=400),
        ),
    )

    assert report["top_statements"] == []
    assert report["statement_gaps"] == {
        "EVICTED_OR_INACTIVE": 1,
        "NEW_ACTIVITY": 1,
    }


def test_workload_report_isolates_counter_and_metric_resets() -> None:
    first = _snapshot(minute=0, questions=100)
    second = _snapshot(minute=5, questions=80)
    first.instance_metrics_json["commits"] = 20
    second.instance_metrics_json["commits"] = 35
    report = build_workload_report(
        target_id=TARGET_ID,
        snapshots=(first, second),
        statements=(),
        metrics=(
            _metric(
                first, family="WAL", code="wal.bytes",
                dimension="all", value=100,
            ),
            _metric(
                second, family="WAL", code="wal.bytes",
                dimension="all", value=20,
            ),
            _metric(
                first, family="IO", code="io.read",
                dimension="all", value=10,
            ),
            _metric(
                second, family="IO", code="io.read",
                dimension="all", value=25,
            ),
            _metric(
                first, family="WAIT", code="wait.lock",
                dimension="Lock", value=1,
            ),
            _metric(
                second, family="WAIT", code="wait.lock",
                dimension="Lock", value=2, quality="PARTIAL",
            ),
        ),
    )

    assert "questions" not in report["load_totals"]
    assert report["load_totals"]["commits"] == 15
    assert report["metric_totals_by_family"] == {"IO": 15.0}
    assert report["metric_discontinuities"][0]["metric_family"] == "WAL"
    assert report["metric_gaps"] == {"QUALITY_UNAVAILABLE": 1}


def test_postgresql_statement_scope_keeps_user_dimension_distinct() -> None:
    first = _snapshot(minute=0, questions=100)
    second = _snapshot(minute=5, questions=120)
    first.database_type = second.database_type = "POSTGRESQL"

    def statement(snapshot, user: str, executions: int, duration: int):
        return SimpleNamespace(
            workload_snapshot_id=snapshot.workload_snapshot_id,
            statement_identity_type="POSTGRESQL_QUERY_ID",
            statement_identity_value="9223372036854775807",
            database_name="16384",
            user_identifier=user,
            top_level=1,
            normalized_statement="SELECT ?",
            execution_count=executions,
            total_duration_microseconds=duration,
            mean_duration_microseconds=duration // max(1, executions),
            max_duration_microseconds=duration,
            rows_processed=executions,
        )

    report = build_workload_report(
        target_id=TARGET_ID,
        snapshots=(first, second),
        statements=(
            statement(first, "10", 1, 100),
            statement(second, "10", 2, 250),
            statement(first, "11", 3, 300),
            statement(second, "11", 5, 700),
        ),
    )

    assert [item["user_identifier"] for item in report["top_statements"]] == [
        "11",
        "10",
    ]


def test_diff_marks_zero_baseline_as_new_activity() -> None:
    report = build_workload_diff_report(
        baseline={
            "status": "READY",
            "target_id": str(TARGET_ID),
            "period_start": STARTED_AT.isoformat(),
            "period_end": (STARTED_AT + timedelta(minutes=5)).isoformat(),
            "load_profile": {"questions_per_second": 0},
            "database_type": "MYSQL",
            "top_statements": [],
        },
        after={
            "status": "READY",
            "target_id": str(TARGET_ID),
            "period_start": (STARTED_AT + timedelta(minutes=5)).isoformat(),
            "period_end": (STARTED_AT + timedelta(minutes=10)).isoformat(),
            "load_profile": {"questions_per_second": 2},
            "database_type": "MYSQL",
            "top_statements": [{
                "statement_identity_type": "MYSQL_DIGEST",
                "statement_identity_value": DIGEST_A,
                "database_name": "app",
                "user_identifier": "",
                "top_level": None,
                "total_duration_microseconds": 500,
                "rank_no": 1,
            }],
        },
    )

    assert report["load_profile_changes"]["questions_per_second"] == {
        "baseline": 0.0,
        "after": 2.0,
        "delta": 2.0,
        "change_percent": None,
        "change_status": "NEW_ACTIVITY",
    }
    assert report["statement_changes"][0]["change_status"] == "NEW_ACTIVITY"


def test_activity_report_discloses_sampling_boundary() -> None:
    report = build_activity_report([
        _activity_sample(index=0),
        _activity_sample(index=1),
    ])

    assert report["sample_count"] == 2
    assert report["top_activity"]["statement"][0]["sample_percent"] == 100
    assert "不是数据库原生ASH" in report["disclaimer"]


def test_statement_text_is_bounded_and_literal_redacted() -> None:
    sanitized = sanitize_statement_text(
        "SELECT 'password', 42, 0xAABB " + "x" * 5000
    )

    assert sanitized is not None
    assert "password" not in sanitized
    assert "42" not in sanitized
    assert "AABB" not in sanitized
    assert len(sanitized) == 4000


@pytest.mark.parametrize(
    "query_id",
    (str(-(2**63)), str(2**63 - 1)),
)
def test_postgresql_query_id_signed_bigint_boundaries_are_strings(
    query_id: str,
) -> None:
    statement = WorkloadStatementSnapshot(
        statement_identity_type=StatementIdentityType.POSTGRESQL_QUERY_ID,
        statement_identity_value=query_id,
        execution_count=0,
        total_duration_microseconds=0,
        mean_duration_microseconds=0,
        max_duration_microseconds=0,
        rows_processed=0,
        rank_no=1,
    )
    sample = ActivitySample(
        statement_identity_type=StatementIdentityType.POSTGRESQL_QUERY_ID,
        statement_identity_value=query_id,
    )

    assert statement.statement_identity_value == query_id
    assert sample.statement_identity_value == query_id


@pytest.mark.parametrize(
    "query_id",
    (str(-(2**63) - 1), str(2**63), "01", "+1", "1.0"),
)
def test_postgresql_query_id_rejects_invalid_values(query_id: str) -> None:
    with pytest.raises(ValueError):
        ActivitySample(
            statement_identity_type=StatementIdentityType.POSTGRESQL_QUERY_ID,
            statement_identity_value=query_id,
        )


def test_workload_report_projects_to_formal_report_and_pdf() -> None:
    first = _snapshot(minute=0, questions=100)
    second = _snapshot(minute=5, questions=160)
    raw = build_workload_report(
        target_id=TARGET_ID,
        snapshots=(first, second),
        statements=(
            _statement(first, DIGEST_A, executions=10, latency=1_000),
            _statement(second, DIGEST_A, executions=14, latency=1_800),
        ),
    )
    content = _report_content_payload(
        report_type=WorkloadReportType.MYSQL_WORKLOAD,
        run_id=uuid7(),
        target_id=TARGET_ID,
        title="生产库 MySQL Workload Report",
        payload=raw,
        period_start=first.collected_at,
        period_end=second.collected_at,
        raw_artifact_id=uuid7(),
        raw_content_hash="a" * 64,
        source_ids=(first.workload_snapshot_id, second.workload_snapshot_id),
    )

    validated = ReportContent.model_validate(content)
    assert validated.report_type == "MYSQL_WORKLOAD"
    assert validated.provenance["template"]["template_ref"] == (
        "system:mysql.workload"
    )
    template = resolve_historical_report_template(
        template_ref="system:mysql.workload",
        report_type="MYSQL_WORKLOAD",
    )
    assert template is not None
    presentation = report_presentation(payload=content, template=template)
    assert any(
        section["kind"] == "INSPECTION_COVERAGE"
        for section in presentation["sections"]
    )
    assert render_pdf(presentation).startswith(b"%PDF-")


def test_activity_content_keeps_sampling_boundary() -> None:
    raw = build_activity_report([_activity_sample()])
    content = _report_content_payload(
        report_type=WorkloadReportType.MYSQL_ACTIVITY,
        run_id=uuid7(),
        target_id=TARGET_ID,
        title="MySQL Activity",
        payload=raw,
        period_start=STARTED_AT,
        period_end=STARTED_AT + timedelta(minutes=5),
        raw_artifact_id=uuid7(),
        raw_content_hash="c" * 64,
        source_ids=(),
    )

    activity = ReportContent.model_validate(content)
    assert activity.report_type == "MYSQL_ACTIVITY"
    assert any(gap["code"] == "SAMPLING_BOUNDARY" for gap in activity.gaps)
    assert "不是数据库原生ASH" in activity.gaps[-1]["detail"]
