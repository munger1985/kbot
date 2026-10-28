"""逐字段原生来源必须接通正式证据、单位换算、报告和案例复核，不能自动升级可比性。"""

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import sqlglot
from sqlglot import exp

from aiops_agent.application.diagnosis.investigation_record import freeze_investigation
from aiops_agent.application.diagnosis import compile_findings
from aiops_agent.application.display_dictionary import display_field_name
from aiops_agent.application.diagnosis.native_measurements import (
    NativeMappingProfile, native_mapping_catalog, native_counter_origins, native_row_projection, native_table_projection,
)
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
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler, DbaEvidenceAssessmentHandler
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


@dataclass(frozen=True)
class Projection:
    columns: tuple[str, ...]
    scale: Decimal = Decimal(1)
    rounded: bool = False
    operation: str = "IDENTITY"
    aggregation: str = "NONE"
    dimensions: tuple[tuple[str, str], ...] = ()


def projected_expression(node):
    """只用于目录验收：核对实际模板的投影、输入和常数比例，不校验目标库 Schema。"""
    if isinstance(node, (exp.Alias, exp.Paren)):
        return projected_expression(node.this)
    if isinstance(node, exp.Column):
        return Projection((node.name.lower(),))
    if isinstance(node, exp.Cast) and node.args["to"].this in {exp.DataType.Type.BIGINT, exp.DataType.Type.DECIMAL}:
        return projected_expression(node.this)
    if isinstance(node, exp.Max) and isinstance(node.this, exp.Case):
        case = node.this
        if len(case.args.get("ifs") or ()) != 1 or case.args.get("this") is not None or case.args.get("default") is not None:
            raise ValueError("状态项必须是单个无默认值的条件透视")
        branch = case.args["ifs"][0]
        condition = branch.this
        if (not isinstance(condition, exp.EQ) or not isinstance(condition.this, exp.Column)
                or condition.this.name.lower() != "variable_name"
                or not isinstance(condition.expression, exp.Literal) or not condition.expression.is_string
                or not isinstance(branch.args["true"], exp.Column) or branch.args["true"].name.lower() != "variable_value"):
            raise ValueError("状态项选择器或来源列不匹配")
        return Projection(("variable_value",), dimensions=(("variable_name", condition.expression.this),))
    if isinstance(node, (exp.Sum, exp.Max)):
        result = projected_expression(node.this)
        assert result.aggregation == "NONE"
        return replace(result, aggregation="SUM" if isinstance(node, exp.Sum) else "MAX")
    if isinstance(node, exp.Round):
        return replace(projected_expression(node.this), rounded=True)
    if isinstance(node, exp.Div) and isinstance(node.expression, exp.Literal):
        result = projected_expression(node.this)
        return replace(result, scale=result.scale / Decimal(node.expression.this))
    if isinstance(node, (exp.Add, exp.Mul)):
        left, right = projected_expression(node.this), projected_expression(node.expression)
        assert not left.dimensions and not right.dimensions
        assert left.aggregation == right.aggregation == "NONE"
        if isinstance(node, exp.Add):
            assert left.scale == right.scale
        return Projection((*left.columns, *right.columns),
            scale=left.scale if isinstance(node, exp.Add) else left.scale * right.scale,
            rounded=left.rounded or right.rounded, operation="SUM" if isinstance(node, exp.Add) else "PRODUCT")
    raise ValueError("模板存在未声明的数值投影")


@pytest.mark.parametrize("binding,profile", BINDINGS, ids=[binding.variant + ":" + binding.tool_id for binding, _ in BINDINGS])
def test_every_mapping_pins_an_actual_catalog_template_and_its_native_expression(binding, profile):
    tool = REGISTRY.resolve_exact(tool_id=binding.tool_id, tool_version=binding.tool_version,
        db_type=binding.database_type, variant=binding.variant, template_sha256=binding.template_sha256)
    assert tuple((column.name, column.logical_type) for column in tool.definition.output_columns) == binding.output_columns
    tree = sqlglot.parse_one(tool.sql, read={"POSTGRESQL": "postgres", "MYSQL": "mysql", "ORACLE": "oracle"}[binding.database_type])
    relations = {".".join(part.name.lower() for part in table.parts) for table in tree.find_all(exp.Table)}
    def relation_key(name):
        key = name.lower()
        if binding.database_type == "POSTGRESQL":
            return key.removeprefix("pg_catalog.")
        if binding.database_type == "ORACLE":
            return key.removeprefix("sys.").replace("gv_$", "gv$", 1).replace("v_$", "v$", 1)
        return key
    assert relation_key(profile.relation) in {relation_key(name) for name in relations}
    for native, column in profile.dimensions.items():
        if column not in dict(binding.output_columns):
            continue
        expressions = [projection.this if isinstance(projection, exp.Alias) else projection
            for select in tree.find_all(exp.Select) for projection in select.expressions
            if projection.alias_or_name.lower() == column.lower()]
        if native == "EVENT_NAME_PREFIX_2":
            assert any(item == sqlglot.parse_one("SUBSTRING_INDEX(event_name, '/', 2)", read="mysql") for item in expressions)
        else:
            # 身份的文本转换不改变原生列；数值字段的 CAST 则由上面的严格表达式核查。
            assert any((item.this if isinstance(item, exp.Cast) else item).name.lower() == native.lower() for item in expressions)
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
        matches = [item for item in candidates if item.columns == expected]
        assert matches, (binding.variant, field.column, candidates)
        projection = matches[-1]
        if field.unit == "second":
            unit_scales = {"second": Decimal(1), "centisecond": Decimal("0.01"), "millisecond": Decimal("0.001"), "microsecond": Decimal("0.000001"), "picosecond": Decimal("0.000000000001")}
            assert projection.scale * field.scale_to_canonical == unit_scales[field.native_unit]
            assert unit_scales[field.observed_unit] == field.scale_to_canonical
        else:
            assert projection.scale * field.scale_to_canonical == 1
            assert field.native_unit == field.observed_unit == field.unit
        assert projection.rounded == (field.projection == "ROUNDED")
        assert projection.operation == ("IDENTITY" if field.projection == "ROUNDED" else field.projection)
        assert projection.aggregation == field.aggregation
        assert projection.dimensions == tuple((key.lower(), value) for key, value in field.native_dimensions)
        if field.projection == "PRODUCT":
            assert field.native_operand_units == ("io_operation", "byte_per_io_operation")
        if field.aggregation != "NONE":
            assert any(select.args.get("group") is not None for select in tree.find_all(exp.Select))
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


def test_expanding_metadata_does_not_revoke_the_previous_unchanged_statement_mapping():
    lineage = database_provenance(observation_for(tool_for("POSTGRESQL"))).native_lineage
    assert lineage.mapping_sha256 == "04442b350025814de26c7b1f38c2d80de5cd75f2b40ef0758685e8dd4d485896"


def test_mysql_pivot_dimensions_separate_counters_and_join_the_same_status_from_two_tools():
    instance = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.instance.performance")))
    throughput = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.mysql.instance.throughput")))
    fields = {field.column: field for field in instance.native_lineage.fields}
    assert fields["questions"].native_columns == fields["commits"].native_columns == ("VARIABLE_VALUE",)
    assert fields["questions"].native_dimensions == (("VARIABLE_NAME", "Questions"),)
    assert fields["commits"].native_dimensions == (("VARIABLE_NAME", "Com_commit"),)
    identities = [native_counter_origins(instance, row_index=0, column=column) for column in fields]
    assert all(identities) and len(set(identities)) == len(fields)
    assert native_counter_origins(instance, row_index=0, column="questions") == native_counter_origins(throughput, row_index=0, column="questions")
    assert fields["threads_running"].kind == fields["uptime_seconds"].kind == "GAUGE"
    assert "questions_per_second" not in {item.column for item in throughput.native_lineage.fields}
    assert not throughput.native_lineage.comparable_window_verified


@pytest.mark.parametrize("sql", [
    "MAX(CASE WHEN variable_name = 'Com_commit' THEN variable_value END)",
    "MAX(CASE WHEN variable_name = 'Questions' THEN variable_value ELSE '0' END)",
    "MAX(CASE WHEN other_name = 'Questions' THEN variable_value END)",
    "MAX(CASE WHEN variable_name = 'Questions' THEN other_value END)",
    "MAX(CASE WHEN variable_name = 'Questions' THEN variable_value WHEN variable_name = 'Com_commit' THEN variable_value END)",
])
def test_pivot_expression_verifier_does_not_ignore_selectors_or_default_values(sql):
    try:
        result = projected_expression(sqlglot.parse_one(sql, read="mysql"))
    except ValueError:
        return
    assert result.dimensions != (("variable_name", "Questions"),)


@pytest.mark.parametrize("column", ["seq_scan", "seq_tup_read", "idx_scan", "idx_tup_fetch", "n_live_tup", "n_dead_tup", "n_mod_since_analyze"])
def test_pg_object_aliases_keep_the_same_native_table_scope_and_other_databases_do_not(column):
    table = tool_for("POSTGRESQL", tool_id="db.postgresql.table.statistics")
    objects = tool_for("POSTGRESQL", tool_id="db.sql.object_statistics")
    provenance = database_provenance(observation_for(table, values={"relation_oid": 42}))
    alias = database_provenance(observation_for(objects, values={"relation_oid": 42}))
    assert native_counter_origins(provenance, row_index=0, column=column) == native_counter_origins(alias, row_index=0, column=column)
    observed = observation_for(table, values={"relation_oid": 42})
    physical = {**observed.provenance["instance_identity"], "database_oid": "999"}
    other_database = database_provenance(observed.model_copy(update={"provenance": {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": physical}}))
    assert native_counter_origins(provenance, row_index=0, column=column) != native_counter_origins(other_database, row_index=0, column=column)


def test_pg_estimates_stay_estimated_and_io_product_retains_both_native_inputs():
    tables = tool_for("POSTGRESQL", tool_id="db.postgresql.table.statistics")
    provenance = database_provenance(observation_for(tables))
    projected = native_row_projection(provenance, row_index=0, row={"n_live_tup": 0, "n_dead_tup": None, "n_mod_since_analyze": 12, "seq_scan": 1})
    assert projected["n_live_tup"]["value"] == "0" and projected["n_live_tup"]["quality"] == "ESTIMATED"
    assert projected["n_dead_tup"]["value"] is None and projected["n_dead_tup"]["quality"] == "MISSING"
    assert projected["n_mod_since_analyze"]["kind"] == "GAUGE"
    assert projected["seq_scan"]["kind"] == "ADDITIVE_COUNTER"
    io = tool_for("POSTGRESQL", tool_id="db.postgresql.io.summary", variant="postgresql_pg_stat_io_16_17")
    provenance = database_provenance(observation_for(io))
    fields = {field.column: field for field in provenance.native_lineage.fields}
    assert fields["read_bytes"].native_columns == ("reads", "op_bytes")
    assert fields["read_bytes"].native_operand_units == ("io_operation", "byte_per_io_operation")
    assert fields["read_bytes"].projection == "PRODUCT"
    assert set(native_counter_origins(provenance, row_index=0, column="reads")) <= set(native_counter_origins(provenance, row_index=0, column="read_bytes"))
    assert native_row_projection(provenance, row_index=0, row={"read_bytes": 24576})["read_bytes"]["value"] == "24576"


@pytest.mark.parametrize("column", ["temp_files", "temp_bytes"])
def test_pg_temp_and_database_statistics_use_the_same_canonical_native_relation(column):
    database = database_provenance(observation_for(tool_for("POSTGRESQL", tool_id="db.postgresql.database.statistics")))
    temp = database_provenance(observation_for(tool_for("POSTGRESQL", tool_id="db.postgresql.temp.usage")))
    assert database.native_lineage.relation == temp.native_lineage.relation == "pg_catalog.pg_stat_database"
    assert native_counter_origins(database, row_index=0, column=column) == native_counter_origins(temp, row_index=0, column=column)


def test_mysql_null_index_is_a_native_no_index_bucket_and_table_aliases_remain_same_origin():
    tool = tool_for("MYSQL", tool_id="db.index.health")
    no_index = database_provenance(observation_for(tool, values={"index_name": None}))
    index = database_provenance(observation_for(tool, values={"index_name": "primary"}))
    assert ("INDEX_NAME", None) in no_index.native_lineage.series[0].dimensions
    assert not no_index.native_lineage.series[0].missing_dimensions
    assert native_counter_origins(no_index, row_index=0, column="count_read")
    assert native_counter_origins(no_index, row_index=0, column="count_read") != native_counter_origins(index, row_index=0, column="count_read")
    table = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.mysql.table.hotspots")))
    objects = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.sql.object_statistics")))
    assert native_counter_origins(table, row_index=0, column="count_read") == native_counter_origins(objects, row_index=0, column="count_read")


def test_mysql_wait_groups_and_memory_high_watermarks_are_not_window_counters():
    wait = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.wait.class_summary"), values={"wait_class": "wait/io"}))
    fields = {field.column: field for field in wait.native_lineage.fields}
    assert wait.native_lineage.series[0].dimensions == (("EVENT_NAME_PREFIX_2", "wait/io"),)
    assert fields["wait_count"].aggregation == fields["total_wait_picoseconds"].aggregation == "SUM"
    assert fields["max_wait_picoseconds"].aggregation == "MAX" and fields["max_wait_picoseconds"].kind == "MAXIMUM"
    memory = database_provenance(observation_for(tool_for("MYSQL", tool_id="db.memory.summary")))
    fields = {field.column: field for field in memory.native_lineage.fields}
    assert fields["current_number_of_bytes_used"].kind == "GAUGE"
    assert fields["high_number_of_bytes_used"].kind == "MAXIMUM"
    assert not wait.native_lineage.comparable_window_verified and not memory.native_lineage.comparable_window_verified


def test_oracle_centisecond_wait_and_average_workarea_time_have_distinct_semantics_and_containers():
    waits = tool_for("ORACLE", tool_id="db.wait.class_summary")
    provenance = database_provenance(observation_for(waits))
    field = next(field for field in provenance.native_lineage.fields if field.column == "time_waited_seconds")
    assert field.native_unit == "centisecond" and field.observed_unit == "second"
    assert canonical_native_value(field, "1.234") == Decimal("1.234")
    assert "con_id AS container_id" in waits.sql
    other = database_provenance(observation_for(waits, values={"container_id": 4}))
    assert native_counter_origins(provenance, row_index=0, column="total_waits") != native_counter_origins(other, row_index=0, column="total_waits")
    workarea = tool_for("ORACLE", tool_id="db.oracle.performance.heavy_sort_sql")
    field = next(field for field in database_provenance(observation_for(workarea)).native_lineage.fields if field.column == "max_workarea_avg_active_seconds")
    assert field.native_unit == "centisecond" and field.kind == "GAUGE" and field.aggregation == "MAX"
    counts = next(field for field in database_provenance(observation_for(workarea)).native_lineage.fields if field.column == "total_executions")
    assert counts.unit == "workarea_execution" and counts.aggregation == "SUM"
    assert "ROUND(MAX(w.active_time) / 100, 3)" in workarea.sql
    assert "GROUP BY w.inst_id, w.con_id, w.sql_id" in workarea.sql
    assert "SUM(w.active_time)" not in workarea.sql and "active_seconds" not in {column.name for column in workarea.definition.output_columns}


def test_oracle_wait_findings_do_not_merge_the_same_wait_class_from_different_containers():
    async def run_test():
        run_id, target_id = uuid7(), uuid7()
        tool = tool_for("ORACLE", tool_id="db.wait.class_summary")
        first = observation_for(tool, target=target_id, values={"container_id": 3, "wait_class": "User I/O", "foreground_waited_seconds": 4000})
        second = observation_for(tool, target=target_id, values={"container_id": 4, "wait_class": "User I/O", "foreground_waited_seconds": 4000})
        rows = (first.rows[0], second.rows[0])
        observed = first.model_copy(update={"rows": rows, "row_count": 2, "result_sha256": canonical_hash(rows),
            "provenance": {"identity_scope": "SAME_CONNECTION_CURRENT", "instance_identity": {**first.provenance["instance_identity"], "container_id": "1"}}})
        artifact = tool_artifact(observed, run_id)
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": artifact.artifact_id, "payload": artifact.payload_json},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        compilation = compile_findings(assessment.evidence, database_type="ORACLE", target_id=str(target_id))
        assert len(compilation.findings) == 2
        assert len({item.finding_id for item in compilation.findings}) == 2
        assert {item.fields["container_id"] for item in compilation.findings} == {3, 4}
    asyncio.run(run_test())


@pytest.mark.parametrize("column,expected", [
    ("container_id", "容器 ID"),
    ("max_workarea_avg_active_seconds", "工作区平均活跃时间最大值（秒）"),
    ("total_executions", "工作区操作次数"),
    ("last_temp_mb", "各工作区最近临时段最大值（MiB）"),
])
def test_native_workarea_and_container_report_columns_have_accurate_chinese_labels(column, expected):
    assert display_field_name(column) == expected


@pytest.mark.parametrize("updates", [
    {"native_dimensions": (("VARIABLE_NAME", "Questions"), ("VARIABLE_NAME", "Com_commit"))},
    {"native_dimensions": (("VARIABLE_NAME", "Questions\n"),)},
    {"native_dimensions": (("", "Questions"),)},
    {"projection": "PRODUCT", "native_columns": ("reads", "op_bytes")},
    {"projection": "IDENTITY", "native_operand_units": ("byte",)},
    {"aggregation": "MAX"}, {"estimated": True},
])
def test_invalid_or_unsafe_native_scope_and_numeric_semantics_are_rejected(updates):
    field = dict(column="count", native_columns=("count",), native_unit="event", observed_unit="event", unit="event", scale_to_canonical="1", kind="ADDITIVE_COUNTER")
    with pytest.raises(ValueError):
        NativeFieldLineage.model_validate({**field, **updates})


def test_profile_does_not_allow_fixed_field_dimensions_to_override_row_identity():
    _, bindings = native_mapping_catalog()
    _, profile = next((binding, profile) for binding, profile in bindings.values() if binding.tool_id == "db.instance.performance" and binding.database_type == "MYSQL")
    with pytest.raises(ValueError):
        NativeMappingProfile.model_validate({**profile.model_dump(), "dimensions": {"VARIABLE_NAME": "server_uuid"}})


@pytest.mark.parametrize("binding,profile", BINDINGS, ids=[binding.variant + ":" + binding.tool_id for binding, _ in BINDINGS])
@pytest.mark.parametrize("tamper", [False, True])
def test_all_pinned_counter_families_reach_general_diagnosis_and_replay_their_original_source(binding, profile, tamper):
    async def run_test():
        run = SimpleNamespace(ops_run_id=uuid7(), target_id=uuid7())
        tool = REGISTRY.resolve_exact(tool_id=binding.tool_id, tool_version=binding.tool_version,
            db_type=binding.database_type, variant=binding.variant, template_sha256=binding.template_sha256)
        observed = observation_for(tool, target=run.target_id)
        artifact = tool_artifact(observed, run.ops_run_id)
        assessment = await DbaEvidenceAssessmentHandler().execute(SimpleNamespace(
            input_artifacts=({"schema_version": artifact.schema_version, "artifact_id": artifact.artifact_id, "payload": artifact.payload_json},),
            plan_snapshot={"answer_context": {"task_frame": {}}}, trigger_type="CHAT"))
        fact, = assessment.evidence
        model_input = DbaAnswerComposeHandler._compose_evidence_payload(fact)
        projection = model_input["native_measurements"]
        assert projection["measurement_count"] == len(profile.fields) and not projection["truncated"]
        assert len(projection["rows"]) == 1 and projection["rows"][0]["row_index"] == 0
        assert model_input["rows"] == fact.model_dump(mode="json")["rows"]
        assert not projection["comparable_window_verified"]
        assert set(projection["rows"][0]["values"]) == {item.column for item in profile.fields}
        draft = DiagnosisAnswerDraft.model_validate({"claims": [{"claim_id": "c1", "kind": "INTERPRETATION", "subject": "统计口径",
            "statement": "按原生单位和维度判断；当前尚未证明窗口连续性。", "level": "UNKNOWN", "applicable_time": NOW.isoformat(),
            "reasoning": "单位与身份不能替代时间覆盖证据。", "limitations": ["尚未取得可比窗口。"]}], "goal_satisfied": False})
        context = SimpleNamespace(run_id=run.ops_run_id, target_id=run.target_id, task_id=uuid7(), plan_snapshot={"answer_context": {"task_frame": {}}})
        record = freeze_investigation(draft=draft, evidence=(fact,), context=context)
        payload = {"evidence": [fact.model_dump(mode="json")], "investigation_record": record.model_dump(mode="json")}
        normalized = normalize_report_source(schema_version="AIOPS_TURN_RESULT.v1", payload=payload, source_kind="DIAGNOSIS")
        assert normalized["investigation_record"]["observations"][0]["provenance"]["native_lineage"]["fields"] == fact.provenance.native_lineage.model_dump(mode="json")["fields"]
        if tamper:
            for item in (payload["evidence"][0], payload["investigation_record"]["observations"][0]):
                item["provenance"]["native_lineage"]["fields"][0]["native_dimensions"] = [["INVENTED", "value"]]
        uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)))
        assert await validate_investigation_sources(uow=uow, run=run, payload=payload) == (not tamper)
        uow.runs.get_artifact.assert_awaited()
    asyncio.run(run_test())


def test_general_model_native_projection_has_an_explicit_shared_value_budget_and_keeps_raw_evidence():
    tool = tool_for("POSTGRESQL", tool_id="db.postgresql.io.summary", variant="postgresql_pg_stat_io_16_17")
    observed = observation_for(tool)
    observed = observed.model_copy(update={"rows": observed.rows * 100, "row_count": 100})
    provenance = database_provenance(observed)
    columns = tuple(column.model_dump() for column in observed.columns)
    projection = native_table_projection(provenance, columns=columns, rows=observed.rows)
    assert projection["truncated"] and projection["measurement_count"] == 256
    assert sum(len(row["values"]) for row in projection["rows"]) == 256
    assert observed.row_count == len(observed.rows) == 100
    assert not projection["comparable_window_verified"]
    assert native_table_projection(None, columns=columns, rows=observed.rows) is None
