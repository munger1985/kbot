"""Finding Compiler、诊断四段合成与自动入口冻结测试。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from pydantic import BaseModel

from aiops_agent.application.diagnosis import (
    compile_findings,
    freeze_automatic_entry_intent,
    is_diagnosis_turn,
)
from aiops_agent.application.diagnosis.findings import load_finding_catalog
from aiops_agent.application.runtime.service import proposal_insert_block_no
from aiops_agent.contracts.diagnosis import ModelInvocationReceipt
from aiops_agent.contracts.turn_answer import (
    DbaAnswerDraft,
    DbaSufficiencyAssessment,
    DiagnosisAnswerDraft,
    TurnEvidenceFact,
)
from aiops_agent.ports.model import StructuredModelResult
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.contracts.aiops import (
    ActionIntent,
    AnswerBlockType,
    FindingConfirmation,
    FindingSeverity,
    FindingType,
    MeasurementSemantics,
    SufficiencyStatus,
    TaskFrame,
)
from platform_core.identity import uuid7


TEST_PROMPT_SNAPSHOT = {"test": {"prompt_version_id": str(uuid7())}}
_LOCK_COLUMNS = (
    "waiting_instance_id",
    "waiting_session_id",
    "waiting_serial_number",
    "waiting_username",
    "waiting_sql_id",
    "waiting_prev_sql_id",
    "waiting_status",
    "blocking_instance_id",
    "blocking_session_id",
    "blocking_serial_number",
    "blocking_username",
    "blocking_sql_id",
    "blocking_prev_sql_id",
    "blocking_status",
    "lock_type",
    "lock_mode",
    "lock_ctime_seconds",
    "wait_event",
    "wait_seconds",
    "chain_depth",
    "is_holder",
)
_LOCK_ROW = (
    1,
    88,
    12345,
    "APP",
    "sqlwait01",
    None,
    "WAITING",
    1,
    12,
    999,
    "BATCH",
    "sqlhold01",
    None,
    "ACTIVE",
    "TX",
    6,
    420,
    "enq: TX - row lock contention",
    90,
    1,
    "N",
)


class _Prompt:
    content = "测试诊断 Prompt"

    @staticmethod
    def ref() -> dict[str, str]:
        return {
            "prompt_id": "aiops_agent.test",
            "prompt_version": "1.0.0",
            "prompt_sha256": "d" * 64,
            "prompt_version_id": str(uuid7()),
            "prompt_source": "DATABASE",
        }


class _TestPrompts:
    async def resolve(self, *_args, **_kwargs):
        return _Prompt()


class _AnswerModel:
    def __init__(self, *, evidence_refs: tuple[str, ...] = ()) -> None:
        self.evidence_refs = evidence_refs
        self.calls = []
        self.stream_calls = 0

    async def generate_structured(self, **kwargs) -> StructuredModelResult:
        self.calls.append(kwargs)
        output_model = kwargs.get("output_model") or DbaAnswerDraft
        if output_model is DiagnosisAnswerDraft:
            output = DiagnosisAnswerDraft(
                analysis_markdown="持有会话 12 阻塞等待会话 88。",
                solution_markdown="先确认持有会话事务，再决定是否中断。",
                evidence_refs=self.evidence_refs,
            )
        else:
            output = DbaAnswerDraft(
                markdown="这是解释类回答，不是巡检报告。",
                evidence_refs=self.evidence_refs,
            )
        digest = "a" * 64
        return StructuredModelResult(
            output=output,
            receipt=ModelInvocationReceipt(
                purpose=kwargs["purpose"],
                schema_id="DBA_DIAGNOSIS_ANSWER_DRAFT.v1",
                model_technical_name="test-model",
                model_revision="1",
                prompt_id=kwargs["prompt_ref"]["prompt_id"],
                prompt_version=kwargs["prompt_ref"]["prompt_version"],
                prompt_sha256=kwargs["prompt_ref"]["prompt_sha256"],
                input_sha256=digest,
                output_sha256=digest,
                duration_ms=1,
            ),
        )

    async def stream_text(self, **_kwargs):
        self.stream_calls += 1
        raise AssertionError("诊断 Turn 不得走 stream_text")


class _Planning(BaseModel):
    task_frame: TaskFrame


def _expected_finding_id(finding_type: str, identity: dict) -> str:
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    digest = hashlib.sha256(
        f"{finding_type}:{canonical}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{finding_type}:{digest}"


def _fact(
    *,
    tool_id: str,
    columns: tuple[str, ...],
    rows: tuple[tuple[object, ...], ...],
    step_id: str = "a1",
    evidence_ref: str | None = None,
    trust_level: str = "SOURCE_VERIFIED",
) -> TurnEvidenceFact:
    return TurnEvidenceFact(
        evidence_ref=evidence_ref or f"artifact:{tool_id}#fact",
        artifact_id=str(uuid7()),
        source_id=tool_id,
        step_id=step_id,
        tool_id=tool_id,
        trust_level=trust_level,
        measurement_semantics=MeasurementSemantics.CURRENT_ACTIVITY,
        presentation_kind="TABLE",
        captured_at=datetime.now(UTC).isoformat(),
        columns=tuple(
            {
                "name": name,
                "logical_type": "STRING",
                "sensitivity": "PUBLIC",
            }
            for name in columns
        ),
        rows=rows,
        row_count=len(rows),
    )


def _lock_fact(**overrides) -> TurnEvidenceFact:
    payload = {
        "tool_id": "db.session.blocking_chain",
        "columns": _LOCK_COLUMNS,
        "rows": (_LOCK_ROW,),
        "evidence_ref": "artifact:blocking#fact",
    }
    payload.update(overrides)
    return _fact(**payload)


def _sqlhc_fact(**overrides) -> TurnEvidenceFact:
    payload = {
        "tool_id": "user.sqlhc.report",
        "columns": (
            "sql_id",
            "owner",
            "object_name",
            "object_type",
            "last_analyzed",
            "stale_stats",
            "file_name",
        ),
        "rows": (
            (
                "6tjx7su0q5ttj",
                "APP",
                "ORDERS",
                "TABLE",
                "2024-01-01",
                "YES",
                "sqlhc_6tjx7su0q5ttj.html",
            ),
        ),
        "evidence_ref": "artifact:sqlhc#fact",
        "trust_level": "USER_PROVIDED",
    }
    payload.update(overrides)
    return _fact(**payload)


def _exacheck_fact(**overrides) -> TurnEvidenceFact:
    payload = {
        "tool_id": "user.exacheck.report",
        "columns": (
            "status",
            "check_name",
            "message",
            "host",
            "check_id",
            "file_name",
        ),
        "rows": (
            (
                "FAIL",
                "Hardware",
                "InfiniBand firmware is not current",
                "cel01",
                "IB_SWITCH_FW",
                "exachk_db01.html",
            ),
            (
                "WARNING",
                "OS Check",
                "Database server RAM is below recommended",
                "db01",
                "OS_RAM",
                "exachk_db01.html",
            ),
            (
                "INFO",
                "Software",
                "Clusterware version is 19.21",
                "db01",
                "CRS_VER",
                "exachk_db01.html",
            ),
        ),
        "evidence_ref": "artifact:exacheck#fact",
        "trust_level": "USER_PROVIDED",
    }
    payload.update(overrides)
    return _fact(**payload)


def _tablespace_fact(**overrides) -> TurnEvidenceFact:
    payload = {
        "tool_id": "db.storage.capacity",
        "columns": (
            "tablespace_name",
            "allocated_mb",
            "used_mb",
            "free_mb",
            "used_percent",
            "maximum_mb",
            "maximum_headroom_mb",
            "file_count",
        ),
        "rows": (("USERS", 1000.0, 900.0, 100.0, 90.0, 2000.0, 1100.0, 2),),
        "evidence_ref": "artifact:capacity#fact",
    }
    payload.update(overrides)
    return _fact(**payload)


def _context(
    *,
    artifacts=(),
    trigger_type: str = "API",
    workflow_kind: str = "",
    task_frame_overrides: dict | None = None,
    target_facts=(),
) -> TaskExecutionContext:
    task_frame = {
        "objectives": ["ASSESS"],
        "problem_statement": "分析当前锁等待",
        "success_criteria": ["找出阻塞会话"],
        "action_intent": "NONE",
    }
    task_frame.update(task_frame_overrides or {})
    return TaskExecutionContext(
        run_id=str(uuid7()),
        task_id=str(uuid7()),
        task_key="evidence:assess",
        target_id=str(uuid7()),
        agent_id=str(uuid7()),
        trigger_type=trigger_type,
        trace_id="trace-findings",
        attempt=1,
        deadline_at=None,
        plan_snapshot={
            "answer_context": {
                "question": "当前谁堵住了会话？",
                "workflow_kind": workflow_kind,
                "task_frame": task_frame,
                "model": {"technical_name": "test-model", "revision": "1"},
                "prompts": TEST_PROMPT_SNAPSHOT,
            },
            "target_facts": list(target_facts),
        },
        policy_snapshot={},
        input_artifacts=artifacts,
    )


def _sufficiency_artifact(evidence: tuple[TurnEvidenceFact, ...]) -> dict:
    assessment = DbaSufficiencyAssessment(
        status=SufficiencyStatus.ANSWERABLE,
        evidence=evidence,
        reasons=("已取得本轮只读证据",),
    )
    return {
        "artifact_id": str(uuid7()),
        "schema_version": "DBA_SUFFICIENCY.v1",
        "payload": assessment.model_dump(mode="json"),
    }


def _proposal_artifact(*, mode: str = "ADVISORY") -> dict:
    now = datetime.now(UTC)
    return {
        "artifact_id": str(uuid7()),
        "schema_version": "PROPOSAL_OUTCOME.v1",
        "payload": {
            "schema_version": "PROPOSAL_OUTCOME.v1",
            "status": "CREATED",
            "proposal": {
                "schema_version": "CHANGE_PROPOSAL_SNAPSHOT.v1",
                "proposal_id": str(uuid7()),
                "run_id": str(uuid7()),
                "task_id": str(uuid7()),
                "target_id": str(uuid7()),
                "target_version": 1,
                "solution_group_key": "turn:test:change",
                "command_ordinal": 1,
                "proposal_version": 1,
                "mode": mode,
                "action_family": "SESSION_INTERRUPT",
                "effect_class": "SESSION_INTERRUPT",
                "execution_mode": "EXECUTABLE_AFTER_APPROVAL",
                "executor_kind": "DATABASE",
                "canonical_object_ref": {
                    "schema": "APP",
                    "object_type": "SESSION",
                    "object_name": "88",
                },
                "action_template_id": "db.session.kill",
                "action_template_version": "1.0.0",
                "action_template_variant": "oracle_session",
                "action_template_hash": "a" * 64,
                "renderer_version": "strict-template.v2",
                "canonical_parameters": {"sid": 88, "serial": 12345},
                "parameter_fact_refs": {"sid": "artifact:blocking#fact"},
                "parameters_hash": "b" * 64,
                "rendered_command": "ALTER SYSTEM KILL SESSION '88,12345';",
                "command_hash": "c" * 64,
                "risk_level": "HIGH",
                "lock_impact": "中断阻塞会话",
                "estimated_duration_seconds": 5,
                "impact": "中断持有锁的会话",
                "rationale": "持有会话长时间阻塞等待会话",
                "preconditions": ["db.session.blocking_chain"],
                "rollback_plan": "会话中断后不可回滚，必要时由应用重连",
                "verification_plan": ["db.session.blocking_chain"],
                "evidence_refs": ["artifact:blocking#fact"],
                "policy_decision_hash": "d" * 64,
                "expires_at": (now + timedelta(minutes=15)).isoformat(),
                "proposal_hash": "e" * 64,
            },
        },
    }


class FindingCompilerTest(unittest.TestCase):
    def test_blocking_chain_compiles_lock_wait_fields(self) -> None:
        compilation = compile_findings((_lock_fact(),), target_id="tgt-1")
        self.assertEqual((), compilation.empty_reasons)
        self.assertEqual((), compilation.gaps)
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.LOCK_WAIT, card.finding_type)
        self.assertEqual(FindingConfirmation.CONFIRMED, card.confirmation)
        self.assertEqual(88, card.fields["waiting_session_id"])
        self.assertEqual(12345, card.fields["waiting_serial_number"])
        self.assertEqual("sqlwait01", card.fields["waiting_sql_id"])
        self.assertEqual(12, card.fields["blocking_session_id"])
        self.assertEqual(999, card.fields["blocking_serial_number"])
        self.assertEqual("sqlhold01", card.fields["blocking_sql_id"])
        self.assertEqual("TX", card.fields["lock_type"])
        self.assertEqual(6, card.fields["lock_mode"])
        self.assertEqual(420, card.fields["lock_ctime_seconds"])
        self.assertEqual(
            _expected_finding_id(
                "LOCK_WAIT",
                {
                    "waiting_instance_id": 1,
                    "waiting_session_id": 88,
                    "blocking_instance_id": 1,
                    "blocking_session_id": 12,
                    "lock_type": "TX",
                },
            ),
            card.finding_id,
        )
        self.assertIn("88", card.impact)
        self.assertIn("12", card.impact)

    def test_zero_row_blocking_chain_is_empty_not_missing_evidence(self) -> None:
        compilation = compile_findings(
            (
                _lock_fact(rows=(),),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("当前未观察到会话阻塞链，不等于未取证。",),
            compilation.empty_reasons,
        )
        self.assertEqual((), compilation.gaps)

    def test_tool_not_run_does_not_invent_empty_reason(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.alert.recent",
                    columns=("SQL_ID",),
                    rows=(("abc123",),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual((), compilation.empty_reasons)
        self.assertEqual((), compilation.gaps)

    def test_missing_lock_columns_keep_card_with_nulls_and_gap(self) -> None:
        kept = [
            name
            for name in _LOCK_COLUMNS
            if name not in {"lock_type", "lock_mode", "lock_ctime_seconds"}
        ]
        values = [
            value
            for name, value in zip(_LOCK_COLUMNS, _LOCK_ROW, strict=True)
            if name not in {"lock_type", "lock_mode", "lock_ctime_seconds"}
        ]
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.session.blocking_chain",
                    columns=tuple(kept),
                    rows=(tuple(values),),
                    evidence_ref="artifact:blocking#partial",
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingConfirmation.UNKNOWN, card.confirmation)
        self.assertIsNone(card.fields["lock_type"])
        self.assertIsNone(card.fields["lock_mode"])
        self.assertIsNone(card.fields["lock_ctime_seconds"])
        self.assertEqual(88, card.fields["waiting_session_id"])
        missing = {item.column for item in compilation.gaps}
        self.assertEqual(
            {"lock_type", "lock_mode", "lock_ctime_seconds"},
            missing,
        )
        self.assertTrue(
            all(item.code == "FINDING_COLUMN_MISSING" for item in compilation.gaps)
        )

    def test_null_lock_value_with_column_present_stays_confirmed(self) -> None:
        values = list(_LOCK_ROW)
        values[_LOCK_COLUMNS.index("lock_type")] = None
        compilation = compile_findings(
            (
                _lock_fact(rows=(tuple(values),)),
            )
        )
        card = compilation.findings[0]
        self.assertEqual(FindingConfirmation.CONFIRMED, card.confirmation)
        self.assertIsNone(card.fields["lock_type"])
        self.assertEqual((), compilation.gaps)

    def test_sqlhc_stale_stats_compile_as_likely_user_provided(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings((_sqlhc_fact(),), target_id="tgt-1")
        self.assertEqual((), compilation.empty_reasons)
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.SQL_STATS_STALE, card.finding_type)
        self.assertEqual(FindingConfirmation.LIKELY, card.confirmation)
        self.assertEqual("APP", card.fields["owner"])
        self.assertEqual("ORDERS", card.fields["object_name"])
        self.assertEqual("TABLE", card.fields["object_type"])
        self.assertEqual("YES", card.fields["stale_stats"])
        self.assertEqual("oracle.sql.healthcheck", card.playbook_id)

    def test_sqlhc_fresh_stats_do_not_create_finding(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings(
            (
                _sqlhc_fact(
                    rows=(
                        (
                            "6tjx7su0q5ttj",
                            "APP",
                            "ORDERS",
                            "TABLE",
                            "2024-01-01",
                            "NO",
                            "sqlhc_6tjx7su0q5ttj.html",
                        ),
                    ),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("上传的 SQLHC 报告未发现过期统计，不等于未取证。",),
            compilation.empty_reasons,
        )

    def test_exacheck_fail_and_warning_compile_as_likely_user_provided(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings((_exacheck_fact(),), target_id="tgt-1")
        types = tuple(card.finding_type for card in compilation.findings)
        self.assertEqual(
            (FindingType.EXACHECK_FAIL, FindingType.EXACHECK_WARNING),
            types,
        )
        fail_card = compilation.findings[0]
        warn_card = compilation.findings[1]
        self.assertEqual(FindingConfirmation.LIKELY, fail_card.confirmation)
        self.assertEqual(FindingConfirmation.LIKELY, warn_card.confirmation)
        self.assertEqual(FindingSeverity.HIGH, fail_card.severity)
        self.assertEqual(FindingSeverity.MEDIUM, warn_card.severity)
        self.assertEqual("Hardware", fail_card.fields["check_name"])
        self.assertEqual("OS Check", warn_card.fields["check_name"])
        self.assertIsNone(fail_card.playbook_id)

    def test_exacheck_info_rows_do_not_create_finding(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings(
            (
                _exacheck_fact(
                    rows=(
                        (
                            "INFO",
                            "Software",
                            "Clusterware version is 19.21",
                            "db01",
                            "CRS_VER",
                            "exachk_db01.html",
                        ),
                    ),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertIn(
            "上传的 ExaCheck 报告未发现 FAIL 项，不等于未取证。",
            compilation.empty_reasons,
        )
        self.assertIn(
            "上传的 ExaCheck 报告未发现 WARNING 项，不等于未取证。",
            compilation.empty_reasons,
        )

    def test_long_session_below_threshold_is_not_a_finding(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.session.active",
                    columns=(
                        "instance_id",
                        "session_id",
                        "serial_number",
                        "username",
                        "status",
                        "wait_event",
                        "wait_seconds",
                        "sql_id",
                    ),
                    rows=((1, 10, 20, "APP", "ACTIVE", "db file sequential read", 12, "sql1"),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("当前活动会话均未达到长会话阈值，不等于未取证。",),
            compilation.empty_reasons,
        )

    def test_mysql_replication_lag_compiles_when_threshold_reached(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.mysql.replication.lag",
                    columns=("channel_name", "lag_seconds"),
                    rows=(("", 45),),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.REPLICATION_LAG, card.finding_type)
        self.assertEqual(FindingSeverity.HIGH, card.severity)
        self.assertEqual(45, card.fields["lag_seconds"])
        self.assertEqual("mysql.replication.lag", card.playbook_id)

    def test_postgresql_dead_tuples_below_threshold_are_not_findings(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.storage.dead_tuples",
                    columns=(
                        "schema_name",
                        "table_name",
                        "live_tuples",
                        "dead_tuples",
                        "dead_tuple_percent",
                        "last_autovacuum",
                        "last_vacuum",
                    ),
                    rows=(("public", "orders", 10000, 1200, 10.71, None, None),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertEqual(
            ("当前表死元组占比均未达到 20% 阈值，不等于未取证。",),
            compilation.empty_reasons,
        )

    def test_idle_session_and_connection_usage_compile(self) -> None:
        load_finding_catalog.cache_clear()
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.session.idle",
                    columns=(
                        "session_id",
                        "username",
                        "client_host",
                        "database_name",
                        "session_state",
                        "idle_seconds",
                        "query_text",
                    ),
                    rows=(
                        (
                            8841,
                            "app",
                            "10.0.0.8",
                            "sales",
                            "idle in transaction",
                            420,
                            "SELECT 1",
                        ),
                    ),
                ),
                _fact(
                    tool_id="db.postgresql.connection.utilization",
                    columns=(
                        "resource_name",
                        "current_utilization",
                        "max_utilization",
                        "limit_value",
                        "utilization_percent",
                    ),
                    rows=(("connections", 180, None, "200", 90.0),),
                ),
                _fact(
                    tool_id="db.maintenance.autovacuum",
                    columns=("schema_name", "table_name", "frozen_xid_age"),
                    rows=(("public", "history", 180000000),),
                ),
            )
        )
        types = tuple(card.finding_type for card in compilation.findings)
        self.assertEqual(
            (
                FindingType.AUTOVACUUM,
                FindingType.IDLE_SESSION,
                FindingType.CONNECTION_USAGE,
            ),
            types,
        )
        idle = compilation.findings[1]
        self.assertEqual(420, idle.fields["idle_seconds"])
        self.assertEqual(FindingConfirmation.CONFIRMED, idle.confirmation)


    def test_invalid_object_compiles_from_summary_rows(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.objects.invalid_summary",
                    columns=("owner", "object_type", "status", "object_count"),
                    rows=(("APP", "PACKAGE", "INVALID", 3), ("SYS", "VIEW", "VALID", 2)),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.INVALID_OBJECT, card.finding_type)
        self.assertEqual("APP", card.fields["owner"])
        self.assertEqual(3, card.fields["object_count"])
        self.assertEqual("oracle.maintenance.health", card.playbook_id)
        self.assertIn("3 个无效 PACKAGE", card.impact)

    def test_archive_headroom_uses_fra_ratio_not_generation_volume(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.archive.status",
                    columns=(
                        "database_role",
                        "open_mode",
                        "log_mode",
                        "force_logging",
                        "flashback_on",
                        "recovery_file_dest",
                        "fra_limit_mb",
                        "fra_used_mb",
                        "fra_reclaimable_mb",
                        "fra_file_count",
                    ),
                    rows=(
                        (
                            "PRIMARY",
                            "READ WRITE",
                            "ARCHIVELOG",
                            "YES",
                            "NO",
                            "/u01/fast_recovery_area",
                            1000,
                            900,
                            50,
                            12,
                        ),
                    ),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.ARCHIVE_HEADROOM, card.finding_type)
        self.assertEqual(90.0, card.fields["fra_used_percent"])
        self.assertEqual(900, card.fields["fra_used_mb"])
        self.assertEqual(1000, card.fields["fra_limit_mb"])
        self.assertIn("FRA 使用率", card.impact)
        self.assertNotIn("生成量", card.impact)

    def test_archive_zero_limit_records_gap_without_finding(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.archive.status",
                    columns=("fra_used_mb", "fra_limit_mb", "recovery_file_dest"),
                    rows=((80, 0, "/u01/fra"),),
                ),
            )
        )
        self.assertEqual((), compilation.findings)
        self.assertTrue(compilation.empty_reasons)
        self.assertEqual({"fra_limit_mb"}, {item.column for item in compilation.gaps})

    def test_backup_failed_matches_warning_and_error_status(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.backup.recent_jobs",
                    columns=(
                        "session_key",
                        "input_type",
                        "status",
                        "start_time",
                        "end_time",
                        "elapsed_seconds",
                        "input_mb",
                        "output_mb",
                        "output_device_type",
                    ),
                    rows=(
                        (11, "DB INCR", "COMPLETED", "2026-01-01", "2026-01-01", 10, 1, 1, "DISK"),
                        (12, "ARCHIVELOG", "COMPLETED WITH WARNINGS", "2026-01-02", "2026-01-02", 20, 2, 2, "SBT_TAPE"),
                    ),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.BACKUP_FAILED, card.finding_type)
        self.assertEqual(12, card.fields["session_key"])
        self.assertEqual("COMPLETED WITH WARNINGS", card.fields["status"])

    def test_long_transaction_uses_elapsed_threshold(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.transaction.long_running",
                    columns=(
                        "instance_id",
                        "session_id",
                        "username",
                        "transaction_started_at",
                        "elapsed_seconds",
                        "undo_blocks",
                        "undo_records",
                    ),
                    rows=((1, 44, "APP", "2026-01-01T00:00:00Z", 300, 8, 16),),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.LONG_TRANSACTION, card.finding_type)
        self.assertEqual(44, card.fields["session_id"])
        self.assertEqual(300, card.fields["elapsed_seconds"])
        self.assertEqual("oracle.transaction.long_running", card.playbook_id)

    def test_top_sql_compiles_above_elapsed_threshold(self) -> None:
        compilation = compile_findings(
            (
                _fact(
                    tool_id="db.sql.top_current",
                    columns=(
                        "sql_id",
                        "plan_hash_value",
                        "executions",
                        "elapsed_seconds",
                        "cpu_seconds",
                        "buffer_gets",
                        "disk_reads",
                        "rows_processed",
                        "last_active_time",
                    ),
                    rows=(
                        ("abc123", 99, 3, 9.5, 1.0, 10, 2, 1, "2026-01-01"),
                        ("sqlhot01", 100, 8, 12.5, 4.0, 200, 20, 50, "2026-01-01"),
                    ),
                ),
            )
        )
        self.assertEqual(1, len(compilation.findings))
        card = compilation.findings[0]
        self.assertEqual(FindingType.TOP_SQL, card.finding_type)
        self.assertEqual("sqlhot01", card.fields["sql_id"])
        self.assertEqual(12.5, card.fields["elapsed_seconds"])


class DiagnosisPolicyTest(unittest.TestCase):
    def test_explain_questions_are_not_diagnosis_turns(self) -> None:
        self.assertFalse(is_diagnosis_turn(objectives=["EXPLAIN"]))
        self.assertFalse(is_diagnosis_turn(objectives=["UNDERSTAND"]))
        self.assertFalse(is_diagnosis_turn(objectives=["PLAN"]))
        self.assertTrue(is_diagnosis_turn(objectives=["ASSESS"]))
        self.assertTrue(
            is_diagnosis_turn(
                objectives=["EXPLAIN"],
                trigger_type="ALERT",
            )
        )

    def test_automatic_entry_freezes_execute_intent(self) -> None:
        planning = _Planning(
            task_frame=TaskFrame(
                objectives=["ASSESS"],
                problem_statement="告警诊断锁等待",
                success_criteria=["确认阻塞会话"],
                action_intent=ActionIntent.EXECUTE,
            )
        )
        frozen = freeze_automatic_entry_intent(
            planning,
            workflow_kind="ALERT_DIAGNOSIS",
        )
        self.assertEqual(ActionIntent.NONE, frozen.task_frame.action_intent)
        self.assertFalse(frozen.task_frame.requires_change)
        chat = freeze_automatic_entry_intent(planning, workflow_kind="CHAT")
        self.assertIs(planning, chat)


class DiagnosisComposeTest(unittest.TestCase):
    def test_diagnosis_turn_emits_four_section_blocks(self) -> None:
        fact = _lock_fact()
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
            prompts=_TestPrompts(),
        )
        result = asyncio.run(
            handler.execute(
                _context(artifacts=(_sufficiency_artifact((fact,)),))
            )
        )
        block_types = [item.block_type for item in result.blocks]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
            ],
            block_types[:3],
        )
        cards = result.blocks[0].payload["findings"]
        self.assertEqual(1, len(cards))
        self.assertEqual(88, cards[0]["fields"]["waiting_session_id"])
        self.assertEqual("TX", cards[0]["fields"]["lock_type"])
        self.assertEqual(
            "持有会话 12 阻塞等待会话 88。",
            result.blocks[1].payload["markdown"],
        )
        self.assertEqual(
            "先确认持有会话事务，再决定是否中断。",
            result.blocks[2].payload["markdown"],
        )

    def test_weekly_inspection_analysis_uses_server_trend_fields(self) -> None:
        fact = _fact(
            tool_id="metric.query_range",
            columns=(
                "metric_code",
                "dimensions",
                "first",
                "latest",
                "change",
                "change_per_day",
            ),
            rows=(
                (
                    "db.storage.used_bytes",
                    "tablespace=USERS",
                    100.0,
                    140.0,
                    40.0,
                    5.0,
                ),
            ),
            evidence_ref="artifact:metric#trend",
        )
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        weekly = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=model,
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    workflow_kind="INSPECTION",
                    task_frame_overrides={
                        "objectives": ["DIAGNOSE", "ASSESS"],
                        "subject_ref": {"schedule_type": "WEEKLY"},
                    },
                )
            )
        )
        payload = model.calls[0]["input_payload"]
        self.assertEqual("USE_SERVER_FIELDS", payload["trend_computation_policy"])
        self.assertEqual(
            [
                {
                    "metric_code": "db.storage.used_bytes",
                    "dimensions": "tablespace=USERS",
                    "first": 100.0,
                    "latest": 140.0,
                    "change": 40.0,
                    "change_per_day": 5.0,
                }
            ],
            payload["metric_trends"],
        )
        self.assertEqual(AnswerBlockType.ANALYSIS_MARKDOWN, weekly.blocks[1].block_type)

        daily_model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        asyncio.run(
            DbaAnswerComposeHandler(
                model_client=daily_model,
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    workflow_kind="INSPECTION",
                    task_frame_overrides={
                        "objectives": ["DIAGNOSE", "ASSESS"],
                        "subject_ref": {"schedule_type": "DAILY"},
                    },
                )
            )
        )
        daily_payload = daily_model.calls[0]["input_payload"]
        self.assertNotIn("metric_trends", daily_payload)
        self.assertNotIn("trend_computation_policy", daily_payload)

    def test_explain_question_does_not_emit_empty_finding_section(self) -> None:
        fact = _lock_fact()
        handler = DbaAnswerComposeHandler(
            model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
            prompts=_TestPrompts(),
        )
        result = asyncio.run(
            handler.execute(
                _context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    task_frame_overrides={"objectives": ["EXPLAIN"]},
                )
            )
        )
        self.assertEqual(AnswerBlockType.MARKDOWN, result.blocks[0].block_type)
        self.assertNotIn(
            AnswerBlockType.FINDING_CARDS,
            [item.block_type for item in result.blocks],
        )

    def test_automatic_alert_omits_proposal_summary(self) -> None:
        fact = _lock_fact()
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=model,
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(
                        _sufficiency_artifact((fact,)),
                        _proposal_artifact(),
                    ),
                    trigger_type="ALERT",
                    workflow_kind="ALERT_DIAGNOSIS",
                    task_frame_overrides={"action_intent": "EXECUTE"},
                )
            )
        )
        self.assertIsNone(model.calls[0]["input_payload"]["proposal_summary"])
        self.assertNotIn(
            AnswerBlockType.PROPOSAL_SUMMARY,
            [item.block_type for item in result.blocks],
        )

    def test_follow_up_execute_places_proposal_after_solution(self) -> None:
        fact = _lock_fact()
        awr = _fact(
            tool_id="db.oracle.awr.report",
            columns=("REPORT",),
            rows=(("<html>awr</html>",),),
            step_id="a1",
            evidence_ref="artifact:awr#fact",
        )
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(
                        _sufficiency_artifact((fact, awr)),
                        _proposal_artifact(),
                    ),
                    task_frame_overrides={"action_intent": "EXECUTE"},
                )
            )
        )
        block_types = [item.block_type for item in result.blocks]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
                AnswerBlockType.PROPOSAL_SUMMARY,
                AnswerBlockType.HTML_REPORT_LINKS,
            ],
            block_types[:5],
        )
        self.assertEqual(
            "a1",
            result.blocks[4].payload["reports"][0]["action_id"],
        )

    def test_html_report_without_action_id_keeps_label_only(self) -> None:
        awr = _fact(
            tool_id="db.oracle.awr.report",
            columns=("REPORT",),
            rows=(("<html>awr</html>",),),
            step_id="prometheus",
            evidence_ref="artifact:awr#fact",
        )
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(awr.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(_context(artifacts=(_sufficiency_artifact((awr,)),)))
        )
        html_block = next(
            item
            for item in result.blocks
            if item.block_type == AnswerBlockType.HTML_REPORT_LINKS
        )
        self.assertIsNone(html_block.payload["reports"][0]["action_id"])
        self.assertFalse(
            any(
                item.block_type in {AnswerBlockType.TABLE, AnswerBlockType.CHART}
                for item in result.blocks
            )
        )

    def test_diagnosis_stream_uses_structured_draft_not_heading_split(self) -> None:
        fact = _lock_fact()
        model = _AnswerModel(evidence_refs=(fact.evidence_ref,))
        handler = DbaAnswerComposeHandler(
            model_client=model,
            prompts=_TestPrompts(),
        )

        async def collect():
            return [
                item
                async for item in handler.execute_stream(
                    _context(artifacts=(_sufficiency_artifact((fact,)),))
                )
            ]

        items = asyncio.run(collect())
        self.assertEqual(0, model.stream_calls)
        self.assertEqual(
            "aiops.dba-diagnosis-answer-stream",
            model.calls[0]["purpose"],
        )
        self.assertIs(DiagnosisAnswerDraft, model.calls[0]["output_model"])
        self.assertTrue(
            any(getattr(item, "event_type", "") == "answer.delta" for item in items[:-1])
        )
        final = items[-1]
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
            ],
            [item.block_type for item in final.blocks[:3]],
        )


    def test_tablespace_missing_facts_inserts_confirmation_after_solution(self) -> None:
        fact = _tablespace_fact()
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(_sufficiency_artifact((fact,)),),
                    target_facts=(),
                )
            )
        )
        block_types = [item.block_type for item in result.blocks]
        self.assertEqual("TABLESPACE", result.blocks[0].payload["findings"][0]["finding_type"])
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
                AnswerBlockType.FACT_CONFIRMATION,
            ],
            block_types[:4],
        )
        payload = result.blocks[3].payload
        self.assertEqual(
            ["ASM_DISKGROUP", "DATAFILE_PATH"],
            payload["missing_fact_types"],
        )
        self.assertNotIn("command_preview", payload)
        self.assertNotIn("proposal_id", payload)

    def test_tablespace_with_asm_or_path_skips_confirmation(self) -> None:
        fact = _tablespace_fact()
        for present in (
            {"fact_type": "ASM_DISKGROUP", "status": "ACTIVE"},
            {"fact_type": "DATAFILE_PATH", "status": "ACTIVE"},
        ):
            result = asyncio.run(
                DbaAnswerComposeHandler(
                    model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                    prompts=_TestPrompts(),
                ).execute(
                    _context(
                        artifacts=(_sufficiency_artifact((fact,)),),
                        target_facts=(present,),
                    )
                )
            )
            self.assertNotIn(
                AnswerBlockType.FACT_CONFIRMATION,
                [item.block_type for item in result.blocks],
            )

    def test_tablespace_confirmation_stays_before_proposal(self) -> None:
        fact = _tablespace_fact()
        result = asyncio.run(
            DbaAnswerComposeHandler(
                model_client=_AnswerModel(evidence_refs=(fact.evidence_ref,)),
                prompts=_TestPrompts(),
            ).execute(
                _context(
                    artifacts=(
                        _sufficiency_artifact((fact,)),
                        _proposal_artifact(),
                    ),
                    task_frame_overrides={"action_intent": "EXECUTE"},
                )
            )
        )
        self.assertEqual(
            [
                AnswerBlockType.FINDING_CARDS,
                AnswerBlockType.ANALYSIS_MARKDOWN,
                AnswerBlockType.SOLUTION_MARKDOWN,
                AnswerBlockType.FACT_CONFIRMATION,
                AnswerBlockType.PROPOSAL_SUMMARY,
            ],
            [item.block_type for item in result.blocks[:5]],
        )
        self.assertNotIn(
            "command_preview",
            result.blocks[3].payload,
        )


class ProposalSequencerTest(unittest.TestCase):
    def test_proposal_inserts_after_solution_before_later_blocks(self) -> None:
        blocks = [
            SimpleNamespace(block_no=1, block_type="FINDING_CARDS"),
            SimpleNamespace(block_no=2, block_type="ANALYSIS_MARKDOWN"),
            SimpleNamespace(block_no=3, block_type="SOLUTION_MARKDOWN"),
            SimpleNamespace(block_no=4, block_type="HTML_REPORT_LINKS"),
            SimpleNamespace(block_no=5, block_type="TABLE"),
        ]
        self.assertEqual(4, proposal_insert_block_no(blocks))

    def test_proposal_appends_when_no_later_blocks(self) -> None:
        blocks = [
            SimpleNamespace(block_no=1, block_type="FINDING_CARDS"),
            SimpleNamespace(block_no=2, block_type="ANALYSIS_MARKDOWN"),
            SimpleNamespace(block_no=3, block_type="SOLUTION_MARKDOWN"),
        ]
        self.assertEqual(4, proposal_insert_block_no(blocks))

    def test_fact_confirmation_is_not_a_later_proposal_block(self) -> None:
        blocks = [
            SimpleNamespace(block_no=1, block_type="FINDING_CARDS"),
            SimpleNamespace(block_no=2, block_type="ANALYSIS_MARKDOWN"),
            SimpleNamespace(block_no=3, block_type="SOLUTION_MARKDOWN"),
            SimpleNamespace(block_no=4, block_type="FACT_CONFIRMATION"),
            SimpleNamespace(block_no=5, block_type="HTML_REPORT_LINKS"),
        ]
        self.assertEqual(5, proposal_insert_block_no(blocks))


if __name__ == "__main__":
    unittest.main()
