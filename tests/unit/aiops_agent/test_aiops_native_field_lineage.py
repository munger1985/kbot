"""逐字段原生来源必须接通正式证据、单位换算、报告和案例复核，不能自动升级可比性。"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlglot
from sqlglot import exp

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis.native_measurements import native_mapping_catalog, native_counter_origins
from aiops_agent.application.diagnosis.provenance import database_provenance, monitoring_provenance
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.diagnosis.sql_diagnosis import build_sql_evidence_bundle
from aiops_agent.application.reporting import normalize_report_source
from aiops_agent.contracts.investigation import DiagnosisAnswerDraft
from aiops_agent.contracts.measurements import NativeFieldLineage
from aiops_agent.diagnostics import DiagnosticRegistry
from aiops_agent.domain.evidence.native_measurements import canonical_native_value
from aiops_agent.domain.measurements import independent_origin_groups
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.executor import DatabaseColumn, DatabaseObservation
from platform_core.identity import uuid7


NOW = datetime(2026, 10, 28, tzinfo=timezone.utc)
REGISTRY = DiagnosticRegistry.load()
BINDINGS = tuple(native_mapping_catalog()[1].values())
ENGINES = ("POSTGRESQL", "MYSQL", "ORACLE")


def tool_for(engine, variant=None, tool_id=None):
    tool_id = tool_id or {"POSTGRESQL": "db.postgresql.statement.details", "MYSQL": "db.sql.cursor_details", "ORACLE": "db.sql.cursor_details"}[engine]
    return next(item for item in REGISTRY.tools if item.definition.db_type == engine
        and item.definition.tool_id == tool_id
        and (item.definition.variant == variant if variant else engine != "POSTGRESQL" or not tool_id.startswith("db.postgresql.statement.") or item.definition.variant.endswith("14_17")))


def observation_for(tool, *, values=None, target=None, rows=None, **updates):
    definition = tool.definition
    values = {"database_oid": 16510, "database_name": "aiops_demo", "user_oid": 42,
        "query_id": "1285382359565720997", "top_level": True, "schema_name": "mes", "digest": "a" * 64,
        "dbid": 3017315513, "container_id": 3, "inst_id": 1, "child_number": 0, "plan_hash_value": 7,
        "sql_id": "36apy6dncjggq", "normalized_statement": "SELECT * FROM mes.orders", "username": "aiops_test", **(values or {})}
    physical = {"POSTGRESQL": {"system_identifier": "7694909560417873958", "database_oid": "16510"},
        "MYSQL": {"server_uuid": "aa6b4e59-410b-4b92-aa59-b4f07bff7411"},
        "ORACLE": {"dbid": "3017315513", "instance_number": "1", "container_id": "3"}}[definition.db_type]
    physical["instance_started_at"] = NOW.isoformat()
    row = tuple(values.get(column.name, NOW.isoformat() if column.logical_type == "DATETIME" else 1 if column.logical_type in {"INTEGER", "DECIMAL"} else "sample") for column in definition.output_columns)
    rows = (row,) if rows is None else rows
    return DatabaseObservation(executor_request_id=uuid7(), target_id=target or uuid7(),
        tool_id=definition.tool_id, tool_version=definition.version, variant=definition.variant,
        template_sha256=definition.template_sha256, db_type=definition.db_type, db_version={"POSTGRESQL": "16", "MYSQL": "8.4", "ORACLE": "19"}[definition.db_type],
        capability_snapshot_hash="a" * 64, captured_at=NOW, duration_ms=3,
        columns=tuple(DatabaseColumn(name=column.name, logical_type=column.logical_type, sensitivity=column.sensitivity) for column in definition.output_columns),
        rows=rows, row_count=len(rows), truncated=False, result_sha256=canonical_hash(rows), parameters_sha256="b" * 64,
        provenance={"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": physical}).model_copy(update=updates)


def tool_artifact(observation, run_id):
    body = {"schema_version": "DBA_TOOL_RESULT.v1", "source_type": "TOOL", "source_id": "native.counter.test",
        "source_version": "1.0.0", "definition_hash": "c" * 64, "output_schema": "DBA_TOOL_RESULT.v1",
        "measurement_semantics": "CUMULATIVE_SINCE_LOAD", "presentation_kind": "TABLE", "status": "SUCCEEDED",
        "tool_outcomes": [{"step_id": "a1", "tool_id": observation.tool_id, "tool_version": observation.tool_version,
            "status": "SUCCEEDED", "observation": observation.model_dump(mode="json")}]}
    return SimpleNamespace(artifact_id=uuid7(), ops_run_id=run_id, schema_version="DBA_TOOL_RESULT.v1",
        trust_level="SOURCE_VERIFIED", payload_json=body, content_hash=canonical_hash(body))


def projected_expression(node):
    """只用于目录验收：核对实际模板的投影、输入和常数比例，不校验目标库 Schema。"""
    if isinstance(node, (exp.Alias, exp.Paren)):
        return projected_expression(node.this)
    if isinstance(node, exp.Column):
        return (node.name.lower(),), Decimal(1), False
    if isinstance(node, exp.Round):
        columns, scale, _ = projected_expression(node.this)
        return columns, scale, True
    if isinstance(node, exp.Div) and isinstance(node.expression, exp.Literal):
        columns, scale, rounded = projected_expression(node.this)
        return columns, scale / Decimal(node.expression.this), rounded
    if isinstance(node, exp.Add):
        left, lscale, lround = projected_expression(node.this)
        right, rscale, rround = projected_expression(node.expression)
        assert lscale == rscale
        return (*left, *right), lscale, lround or rround
    raise ValueError("模板存在未声明的数值投影")


@pytest.mark.parametrize("binding,profile", BINDINGS, ids=[binding.variant + ":" + binding.tool_id for binding, _ in BINDINGS])
def test_every_mapping_pins_an_actual_catalog_template_and_its_native_expression(binding, profile):
    tool = REGISTRY.resolve_exact(tool_id=binding.tool_id, tool_version=binding.tool_version,
        db_type=binding.database_type, variant=binding.variant, template_sha256=binding.template_sha256)
    assert tuple((column.name, column.logical_type) for column in tool.definition.output_columns) == binding.output_columns
    tree = sqlglot.parse_one(tool.sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[binding.database_type])
    for field in profile.fields:
        candidates = []
        for select in tree.find_all(exp.Select):
            for projection in select.expressions:
                if projection.alias_or_name.lower() == field.column.lower():
                    try:
                        candidates.append(projected_expression(projection))
                    except ValueError:
                        pass
        expected = tuple(column.lower() for column in field.native_columns)
        matches = [(columns, scale, rounded) for columns, scale, rounded in candidates if columns == expected]
        assert matches, (binding.variant, field.column, candidates)
        _, scale, rounded = matches[-1]
        native_scale = Decimal("0.001") if field.unit == "second" and binding.database_type == "POSTGRESQL" else Decimal("0.000001") if field.unit == "second" and binding.database_type == "ORACLE" else Decimal("0.000000000001") if field.unit == "second" else Decimal(1)
        assert scale * field.scale_to_canonical == native_scale
        if field.unit == "second":
            assert field.native_unit == {"POSTGRESQL": "millisecond", "MYSQL": "picosecond", "ORACLE": "microsecond"}[binding.database_type]
            unit_scales = {"second": Decimal(1), "millisecond": Decimal("0.001"), "microsecond": Decimal("0.000001"), "picosecond": Decimal("0.000000000001")}
            assert unit_scales[field.observed_unit] == field.scale_to_canonical
        else:
            assert field.native_unit == field.observed_unit == field.unit
        assert rounded == (field.projection == "ROUNDED")
        assert (len(field.native_columns) > 1) == (field.projection == "SUM")
    lineage = database_provenance(observation_for(tool)).native_lineage
    assert lineage is not None and lineage.fields == profile.fields
    assert not lineage.comparable_window_verified and not lineage.exporter_identity_verified


@pytest.mark.parametrize("value,expected", [
    ("9007199254740993123456789", Decimal("9007199254740.993123456789")),
    (0, Decimal(0)), (None, None), (True, None), ("NaN", None), ("Infinity", None), ("1e10000", None),
])
def test_picosecond_conversion_is_exact_and_missing_or_invalid_values_are_not_zero(value, expected):
    field = next(item for item in database_provenance(observation_for(tool_for("MYSQL"))).native_lineage.fields if item.column == "total_latency_picoseconds")
    assert canonical_native_value(field, value) == expected


def test_conversion_preserves_more_than_default_decimal_precision_and_semantics():
    field = NativeFieldLineage(column="duration", native_columns=("TOTAL_TIME",), native_unit="microsecond", observed_unit="microsecond", unit="second", scale_to_canonical="0.000001", kind="MAXIMUM")
    number = "1234567890" * 8 + "001"
    assert canonical_native_value(field, number) == Decimal(number[:-6] + "." + number[-6:])
    assert field.kind == "MAXIMUM"


@pytest.mark.parametrize("engine", ENGINES)
def test_average_and_maximum_are_not_implicitly_made_additive_by_a_unit_mapping(engine):
    lineage = database_provenance(observation_for(tool_for(engine))).native_lineage
    fields = {item.column: item for item in lineage.fields}
    if engine == "ORACLE":
        assert lineage.reset_scope == "CURSOR_LIFETIME" and lineage.reset_context_columns == ("first_load_time", "last_load_time")
    else:
        mean = fields["mean_exec_time_ms" if engine == "POSTGRESQL" else "average_latency_picoseconds"]
        maximum = fields["max_exec_time_ms" if engine == "POSTGRESQL" else "maximum_latency_picoseconds"]
        assert mean.kind == "CUMULATIVE_AVERAGE" and maximum.kind == "MAXIMUM"
        assert lineage.reset_scope == "STATEMENT_ENTRY"
    assert not lineage.comparable_window_verified


def test_pg18_io_is_explicitly_two_inputs_and_never_claimed_to_be_temp_io_or_single_old_counter():
    tool = tool_for("POSTGRESQL", variant="postgresql_pg_stat_statements_details_18_plus")
    lineage = database_provenance(observation_for(tool)).native_lineage
    reads = next(field for field in lineage.fields if field.column == "block_read_time_ms")
    assert reads.native_columns == ("shared_blk_read_time", "local_blk_read_time")
    assert reads.projection == "SUM" and reads.unit == "second"
    assert canonical_native_value(reads, "1001.234") == Decimal("1.001234")


def test_blocks_are_not_silently_converted_to_bytes_with_a_guessed_block_size():
    lineage = database_provenance(observation_for(tool_for("POSTGRESQL"))).native_lineage
    field = next(item for item in lineage.fields if item.column == "shared_blks_read")
    assert field.unit == "postgresql_block" and field.scale_to_canonical == 1


@pytest.mark.parametrize("change", [
    {"template_sha256": "f" * 64}, {"tool_version": "2.0.0"}, {"variant": "other"},
    {"db_type": "MYSQL"}, {"tool_id": "db.postgresql.readonly_query"},
    {"row_count": 5}, {"columns": ()}, {"rows": ((1,),)},
])
def test_unpinned_dynamic_mismatched_or_malformed_observations_do_not_inherit_native_semantics(change):
    observed = observation_for(tool_for("POSTGRESQL")).model_copy(update=change)
    assert database_provenance(observed).native_lineage is None


def test_masked_numerical_fields_cannot_be_reinterpreted_as_public_native_counters():
    observed = observation_for(tool_for("POSTGRESQL"))
    columns = tuple(column.model_copy(update={"sensitivity": "HASHED"}) if column.name == "execution_count" else column for column in observed.columns)
    assert database_provenance(observed.model_copy(update={"columns": columns})).native_lineage is None


def test_database_scopes_are_per_row_and_do_not_borrow_the_connection_database_oid():
    tool = tool_for("POSTGRESQL", tool_id="db.postgresql.database.statistics")
    first = observation_for(tool, values={"database_oid": 16510})
    second = observation_for(tool, values={"database_oid": 99999})
    observed = first.model_copy(update={"rows": (first.rows[0], second.rows[0]), "row_count": 2})
    provenance = database_provenance(observed)
    assert provenance.native_lineage.series[1].dimensions == (("datid", "99999"),)
    assert native_counter_origins(provenance, row_index=0, column="xact_commit") != native_counter_origins(provenance, row_index=1, column="xact_commit")
    assert provenance.native_lineage.reset_scope == "DATABASE_STATISTICS"


def test_mysql_native_null_schema_is_not_missing_schema_or_current_database():
    observed = observation_for(tool_for("MYSQL"), values={"schema_name": None})
    provenance = database_provenance(observed)
    scope, = provenance.native_lineage.series
    assert ("SCHEMA_NAME", None) in scope.dimensions and not scope.missing_dimensions
    assert native_counter_origins(provenance, row_index=0, column="execution_count")


@pytest.mark.parametrize("values", [{"query_id": None}, {"user_oid": None}, {"top_level": None}, {"query_id": "\nsecret"}, {"user_oid": 1.1}])
def test_missing_native_scope_is_explicit_and_cannot_produce_a_counter_identity(values):
    provenance = database_provenance(observation_for(tool_for("POSTGRESQL"), values=values))
    assert provenance.native_lineage.series[0].missing_dimensions
    assert "NATIVE_SERIES_SCOPE_INCOMPLETE" in provenance.quality_flags
    assert not native_counter_origins(provenance, row_index=0, column="execution_count")


def test_oracle_unreturned_container_is_not_assumed_from_connection_and_rounding_is_preserved():
    tool = tool_for("ORACLE", tool_id="db.sql.top_current")
    provenance = database_provenance(observation_for(tool))
    assert "con_id" in provenance.native_lineage.series[0].missing_dimensions
    assert not native_counter_origins(provenance, row_index=0, column="executions")
    elapsed = next(field for field in provenance.native_lineage.fields if field.column == "elapsed_seconds")
    assert elapsed.projection == "ROUNDED"


def test_oracle_current_container_filter_is_pinned_before_using_connection_context():
    tool = tool_for("ORACLE", tool_id="db.oracle.workload.statements")
    provenance = database_provenance(observation_for(tool))
    assert ("con_id", "3") in provenance.native_lineage.series[0].dimensions
    assert "WHERE con_id = TO_NUMBER(SYS_CONTEXT('USERENV', 'CON_ID'))" in tool.sql
    assert native_counter_origins(provenance, row_index=0, column="executions")


def test_unknown_physical_identity_and_mismatched_oracle_dbid_do_not_create_counter_identity():
    observed = observation_for(tool_for("ORACLE"))
    unknown = database_provenance(observed.model_copy(update={"provenance": {}}))
    assert unknown.native_lineage is not None and not native_counter_origins(unknown, row_index=0, column="executions")
    wrong = database_provenance(observation_for(tool_for("ORACLE"), values={"dbid": 999}))
    assert not native_counter_origins(wrong, row_index=0, column="executions")


def test_same_counter_through_top_and_details_is_not_a_new_native_origin():
    top = database_provenance(observation_for(tool_for("POSTGRESQL", variant="postgresql_pg_stat_statements_14_17", tool_id="db.postgresql.statement.top")))
    details = database_provenance(observation_for(tool_for("POSTGRESQL")))
    assert native_counter_origins(top, row_index=0, column="execution_count") == native_counter_origins(details, row_index=0, column="execution_count")
    assert native_counter_origins(details, row_index=0, column="execution_count") != native_counter_origins(details, row_index=0, column="rows_processed")
    monitoring = monitoring_provenance(SimpleNamespace(target_id=uuid7(), observations=(), model_dump=lambda **kw: {}))
    assert len(independent_origin_groups([SimpleNamespace(provenance=item) for item in (top, details, monitoring)])) == 1


def test_mapping_limits_are_explicit_without_dropping_raw_rows():
    observed = observation_for(tool_for("POSTGRESQL"))
    observed = observed.model_copy(update={"rows": observed.rows * 100, "row_count": 100})
    provenance = database_provenance(observed)
    assert provenance.native_lineage.series_truncated and len(provenance.native_lineage.series) == 64
    assert "NATIVE_SERIES_TRUNCATED" in provenance.quality_flags
    assert observed.row_count == 100 and not observed.truncated
    assert not native_counter_origins(provenance, row_index=64, column="execution_count")


def test_unrelated_catalog_formatting_or_new_profiles_do_not_revoke_an_unchanged_field_mapping(monkeypatch):
    observed = observation_for(tool_for("POSTGRESQL"))
    original = database_provenance(observed).native_lineage
    _, bindings = native_mapping_catalog()
    monkeypatch.setattr("aiops_agent.application.diagnosis.native_measurements.native_mapping_catalog", lambda: ("f" * 64, bindings))
    assert database_provenance(observed).native_lineage == original


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("tamper", [None, "missing_source", "row", "scope", "trust", "run", "mapping", "unit", "removed_mapping", "columns", "source_hash", "fact_trust"])
def test_formal_assessment_answer_report_and_case_replay_native_mapping_even_without_object_graph_or_sourced_claim(engine, tamper):
    async def run_test():
        run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
        observed = observation_for(tool_for(engine), target=run.target_id)
        artifact = tool_artifact(observed, run.ops_run_id)
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": artifact.artifact_id, "payload": artifact.payload_json},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        evidence, = assessment.evidence
        assert evidence.provenance.native_lineage is not None
        draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "INTERPRETATION", "subject": "统计口径",
            "statement": "目前是累计统计，未核验请求窗口内的增量。", "level": "UNKNOWN", "applicable_time": NOW.isoformat(),
            "reasoning": "字段语义不等于窗口覆盖。", "limitations": ["未取得两个可比快照。"]}], "goal_satisfied": False})
        context = SimpleNamespace(run_id=run.ops_run_id, target_id=run.target_id, task_id=uuid7(),
            plan_snapshot={"answer_context": {"task_frame": {}}})
        record = freeze_investigation(draft=draft, evidence=(evidence,), context=context)
        assert not record.sql_graphs
        assert record.observations[0].provenance == evidence.provenance
        payload = {"status": "PARTIAL", "blocks": [], "investigation_record": record.model_dump(mode="json"), "evidence": [evidence.model_dump(mode="json")]}
        normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
        raw = artifact.payload_json["tool_outcomes"][0]["observation"]
        if tamper == "row":
            index = next(index for index, column in enumerate(raw["columns"]) if column["name"] == evidence.provenance.native_lineage.fields[0].column)
            raw["rows"][0][index] = "99999999"
        elif tamper == "scope":
            raw["provenance"]["instance_identity"] = {}
        elif tamper == "trust":
            artifact.trust_level = "USER_PROVIDED"
        elif tamper == "run":
            artifact.ops_run_id = uuid7()
        elif tamper == "columns":
            raw["columns"][0]["name"] = "invented"
        elif tamper in {"mapping", "unit", "removed_mapping"}:
            for item in (payload["evidence"][0], payload["investigation_record"]["observations"][0]):
                mapping = item["provenance"]["native_lineage"]
                if tamper == "removed_mapping":
                    item["provenance"].pop("native_lineage")
                elif tamper == "mapping":
                    mapping["mapping_sha256"] = "e" * 64
                else:
                    mapping["fields"][0]["unit"] = "invented"
        elif tamper == "fact_trust":
            for item in (payload["evidence"][0], payload["investigation_record"]["observations"][0]):
                item["trust_level"] = "USER_PROVIDED"
        artifact.content_hash = "d" * 64 if tamper == "source_hash" else canonical_hash(artifact.payload_json)
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=None if tamper == "missing_source" else artifact)))
        assert await validate_investigation_sources(uow=uow, run=run, payload=payload) == (tamper is None)
        uow.runs.get_artifact.assert_awaited()
    asyncio.run(run_test())


def test_sql_bundle_keeps_pinned_native_field_units_and_all_statement_dimensions():
    async def run_test():
        observed = observation_for(tool_for("POSTGRESQL"))
        artifact = tool_artifact(observed, uuid7())
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": artifact.artifact_id, "payload": artifact.payload_json},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        bundle = build_sql_evidence_bundle(task_frame={"diagnostic_profile": "SINGLE_SQL_PERFORMANCE", "subject_ref": {
            "query_id": "1285382359565720997", "database_oid": "16510", "user_oid": "42", "top_level": True}},
            database_type="POSTGRESQL", evidence=assessment.evidence)
        lineage = bundle["observations"][0]["lineage"]["provenance"]["native_lineage"]
        assert lineage["relation"] == "pg_stat_statements"
        assert dict(lineage["series"][0]["dimensions"])["toplevel"] == "true"
        assert any(item["column"] == "total_exec_time_ms" and item["unit"] == "second" and item["scale_to_canonical"] == "0.001" for item in lineage["fields"])
        native_values = bundle["observations"][0]["records"][0]["native_measurements"]
        assert native_values["total_exec_time_ms"]["value"] == "0.001"
        assert native_values["total_exec_time_ms"]["unit"] == "second"
        assert native_values["max_exec_time_ms"]["kind"] == "MAXIMUM"
        assert native_values["execution_count"]["native_origin_ids"]
    asyncio.run(run_test())


def test_native_source_replay_preserves_mysql_current_digest_sample_delta_semantics():
    async def run_test():
        run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
        observed = observation_for(tool_for("MYSQL", tool_id="db.sql.top_current", variant="mysql_8_digest_window"), target=run.target_id)
        artifact = tool_artifact(observed, run.ops_run_id)
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": artifact.artifact_id, "payload": artifact.payload_json},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        evidence, = assessment.evidence
        assert evidence.measurement_semantics == "SNAPSHOT_DELTA" and evidence.provenance.native_lineage is None
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)))
        payload = {"evidence": [evidence.model_dump(mode="json")]}
        assert await validate_investigation_sources(uow=uow, run=run, payload=payload)
        payload["evidence"][0]["measurement_semantics"] = "CUMULATIVE_SINCE_LOAD"
        assert not await validate_investigation_sources(uow=uow, run=run, payload=payload)
    asyncio.run(run_test())
