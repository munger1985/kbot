"""结构采集选择、局部重试、人工补录与语义候选生成。"""

from __future__ import annotations

import re
import hashlib
import json
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from data_query.contracts import (
    DatasetDefinition,
    DimensionDefinition,
    MeasureDefinition,
    SemanticModelCandidate,
    SemanticModelCandidateRequest,
    SemanticModelDefinition,
)
from data_query.persistence import DataQueryUnitOfWork
from data_query.entities import DataQueryAuditEntity
from platform_core.identity import uuid7


class SchemaMetadataError(ValueError):
    """结构元数据管理的稳定错误码。"""


async def _append_audit(uow, *, domain_id: int, actor_id: str, action: str, payload: dict[str, object]) -> None:
    assert uow.audits
    content = {"action": action, **payload}
    await uow.audits.append(DataQueryAuditEntity(
        data_query_run_id=None, domain_id=domain_id, actor_id=actor_id,
        trace_id=f"management:{uuid7()}", action=action,
        payload_json=content,
        content_hash=hashlib.sha256(
            json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    ))


def _logical_key(value: str, *, fallback: str) -> str:
    key = re.sub(r"[^a-z0-9._-]+", "_", value.lower()).strip("._-")
    if not key or not key[0].isalpha():
        key = f"{fallback}_{key}" if key else fallback
    return key[:128]


def _split_columns(body: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    for index, char in enumerate(body):
        if quote:
            if char == quote:
                quote = None
            continue
        if char in {'"', "'", "`"}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(body[start:index].strip())
            start = index + 1
    parts.append(body[start:].strip())
    return [item for item in parts if item]


def parse_create_table_ddl(*, ddl: str, expected_schema: str, expected_object: str) -> dict[str, object]:
    """只解析单条 CREATE TABLE，不执行 DDL。"""
    normalized = ddl.strip().rstrip(";").strip()
    if ";" in normalized:
        raise SchemaMetadataError("MANUAL_DDL_MULTIPLE_STATEMENTS_DENIED")
    match = re.fullmatch(
        r"CREATE\s+TABLE\s+(?:(?P<schema>[A-Za-z_][A-Za-z0-9_$#]*)\.)?"
        r"(?P<table>[A-Za-z_][A-Za-z0-9_$#]*)\s*\((?P<body>.*)\)",
        normalized, flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise SchemaMetadataError("MANUAL_DDL_CREATE_TABLE_ONLY")
    schema = match.group("schema") or expected_schema
    table = match.group("table")
    if schema.lower() != expected_schema.lower() or table.lower() != expected_object.lower():
        raise SchemaMetadataError("MANUAL_DDL_OBJECT_MISMATCH")
    columns: list[dict[str, object]] = []
    for part in _split_columns(match.group("body")):
        if re.match(r"^(CONSTRAINT|PRIMARY|FOREIGN|UNIQUE|CHECK)\b", part, re.IGNORECASE):
            continue
        column = re.match(
            r'^["`]?([A-Za-z_][A-Za-z0-9_$#]*)["`]?\s+([A-Za-z][A-Za-z0-9_]*(?:\s*\([^)]*\))?)',
            part,
        )
        if column is None:
            raise SchemaMetadataError("MANUAL_DDL_COLUMN_INVALID")
        columns.append({
            "name": column.group(1), "type": column.group(2).upper(),
            "nullable": not bool(re.search(r"\bNOT\s+NULL\b", part, re.IGNORECASE)),
            "ordinal": len(columns) + 1, "default": None,
        })
    if not columns:
        raise SchemaMetadataError("MANUAL_DDL_HAS_NO_COLUMNS")
    return {
        "schema": expected_schema, "name": expected_object, "object_type": "TABLE",
        "columns": [str(item["name"]) for item in columns], "column_details": columns,
    }


async def confirm_snapshot_selection(
    *, uow_factory, domain_id: int, actor_id: str, snapshot_id: UUID, object_ids: tuple[UUID, ...],
) -> None:
    async with uow_factory() as uow:
        assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
        source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
        if snapshot is None or source is None or source.domain_id != domain_id:
            raise SchemaMetadataError("SCHEMA_SNAPSHOT_NOT_FOUND")
        if snapshot.status != "WAITING_SELECTION":
            raise SchemaMetadataError("SCHEMA_SNAPSHOT_NOT_SELECTABLE")
        rows = await uow.schema_snapshot_objects.list_by_snapshot(schema_snapshot_id=snapshot_id)
        selected_ids = set(object_ids)
        if not selected_ids.issubset({row.schema_snapshot_object_id for row in rows}):
            raise SchemaMetadataError("SCHEMA_OBJECT_NOT_FOUND")
        for row in rows:
            row.selected = row.schema_snapshot_object_id in selected_ids
            row.status = "QUEUED" if row.selected else "EXCLUDED"
        snapshot.status = "CAPTURING"
        snapshot.completed_at = None
        await _append_audit(
            uow, domain_id=domain_id, actor_id=actor_id,
            action="SCHEMA_SNAPSHOT_SELECTION_CONFIRMED",
            payload={"schema_snapshot_id": str(snapshot_id), "selected_count": len(selected_ids)},
        )
        await uow.commit()


async def retry_snapshot_object(
    *, uow_factory, domain_id: int, actor_id: str, snapshot_id: UUID, object_id: UUID,
) -> None:
    async with uow_factory() as uow:
        assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
        source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
        item = await uow.schema_snapshot_objects.get_by_id(schema_snapshot_object_id=object_id, lock=True)
        if snapshot is None or source is None or source.domain_id != domain_id or item is None or item.schema_snapshot_id != snapshot_id:
            raise SchemaMetadataError("SCHEMA_OBJECT_NOT_FOUND")
        if item.status != "FAILED":
            raise SchemaMetadataError("SCHEMA_OBJECT_NOT_RETRYABLE")
        if int(item.attempt_count) >= 3:
            raise SchemaMetadataError("SCHEMA_OBJECT_RETRY_LIMIT_REACHED")
        item.selected = True
        item.status = "QUEUED"
        item.error_code = None
        item.error_message = None
        item.completed_at = None
        snapshot.status = "CAPTURING"
        snapshot.completed_at = None
        snapshot.error_code = None
        await _append_audit(
            uow, domain_id=domain_id, actor_id=actor_id,
            action="SCHEMA_SNAPSHOT_OBJECT_RETRIED",
            payload={"schema_snapshot_id": str(snapshot_id), "schema_snapshot_object_id": str(object_id)},
        )
        await uow.commit()


async def supply_manual_metadata(
    *, uow_factory, domain_id: int, actor_id: str, snapshot_id: UUID, object_id: UUID, ddl: str,
) -> None:
    async with uow_factory() as uow:
        assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
        source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
        item = await uow.schema_snapshot_objects.get_by_id(schema_snapshot_object_id=object_id, lock=True)
        if snapshot is None or source is None or source.domain_id != domain_id or item is None or item.schema_snapshot_id != snapshot_id:
            raise SchemaMetadataError("SCHEMA_OBJECT_NOT_FOUND")
        if item.status != "FAILED":
            raise SchemaMetadataError("SCHEMA_OBJECT_MANUAL_OVERRIDE_DENIED")
        metadata = parse_create_table_ddl(
            ddl=ddl, expected_schema=item.schema_name, expected_object=item.object_name,
        )
        item.selected = True
        item.status = "MANUAL"
        item.metadata_source = "MANUAL"
        item.metadata_json = metadata
        item.error_code = None
        item.error_message = None
        item.completed_at = datetime.now(UTC)
        rows = await uow.schema_snapshot_objects.list_by_snapshot(schema_snapshot_id=snapshot_id)
        selected = [row for row in rows if row.selected]
        ready = [row for row in selected if row.status in {"READY", "MANUAL"}]
        failed = [row for row in selected if row.status == "FAILED"]
        snapshot.objects_json = {"objects": [row.metadata_json for row in ready if row.metadata_json]}
        content_hash = hashlib.sha256(
            json.dumps(snapshot.objects_json, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        snapshot.capabilities_json = {**snapshot.capabilities_json, "content_hash": content_hash}
        snapshot.snapshot_hash = content_hash
        still_running = any(row.status in {"QUEUED", "CAPTURING"} for row in selected)
        snapshot.status = "CAPTURING" if still_running else "PARTIAL_READY" if failed else "READY"
        snapshot.completed_at = None if still_running else datetime.now(UTC)
        await _append_audit(
            uow, domain_id=domain_id, actor_id=actor_id,
            action="SCHEMA_SNAPSHOT_OBJECT_MANUAL_METADATA_SUPPLIED",
            payload={
                "schema_snapshot_id": str(snapshot_id),
                "schema_snapshot_object_id": str(object_id),
                "ddl_hash": hashlib.sha256(ddl.encode("utf-8")).hexdigest(),
            },
        )
        if not still_running:
            await uow.schema_snapshots.supersede_previous(
                data_source_id=source.data_source_id,
                current_snapshot_id=snapshot.schema_snapshot_id,
            )
            event_type = "data_query.schema.capture_partial" if failed else "data_query.schema.capture_completed"
        await uow.commit()


def _value_type(database_type: str) -> str:
    kind = database_type.upper()
    if any(token in kind for token in ("DATE", "TIME")):
        return "DATETIME" if "TIME" in kind else "DATE"
    if any(token in kind for token in ("BOOL", "BIT")):
        return "BOOLEAN"
    if any(token in kind for token in ("INT", "NUMBER", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL")):
        return "DECIMAL" if any(token in kind for token in ("NUMBER", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "REAL")) else "INTEGER"
    return "STRING"


async def generate_semantic_candidate(
    *, uow_factory, domain_id: int, snapshot_id: UUID, command: SemanticModelCandidateRequest,
) -> SemanticModelCandidate:
    async with uow_factory() as uow:
        assert uow.schema_snapshots and uow.schema_snapshot_objects and uow.data_sources
        snapshot = await uow.schema_snapshots.get_by_id(schema_snapshot_id=snapshot_id)
        source = None if snapshot is None else await uow.data_sources.get_by_id(data_source_id=snapshot.data_source_id)
        if snapshot is None or source is None or source.domain_id != domain_id:
            raise SchemaMetadataError("SCHEMA_SNAPSHOT_NOT_FOUND")
        if snapshot.status not in {"READY", "PARTIAL_READY"}:
            raise SchemaMetadataError("SCHEMA_SNAPSHOT_NOT_READY")
        rows = await uow.schema_snapshot_objects.list_by_snapshot(schema_snapshot_id=snapshot_id)
        await uow.commit()
    requested = set(command.object_ids)
    usable = [
        row for row in rows
        if row.status in {"READY", "MANUAL"} and row.metadata_json
        and (not requested or row.schema_snapshot_object_id in requested)
    ]
    if not usable:
        raise SchemaMetadataError("SEMANTIC_CANDIDATE_HAS_NO_OBJECTS")
    datasets: list[DatasetDefinition] = []
    dimensions: list[DimensionDefinition] = []
    measures: list[MeasureDefinition] = []
    warnings: list[str] = []
    used_keys: set[str] = set()
    for index, row in enumerate(usable, start=1):
        metadata = row.metadata_json or {}
        dataset_key = _logical_key(row.object_name, fallback=f"dataset_{index}")
        while dataset_key in used_keys:
            dataset_key = _logical_key(f"{row.schema_name}_{row.object_name}", fallback=f"dataset_{index}")
        used_keys.add(dataset_key)
        column_details = metadata.get("column_details", [])
        primary_time: str | None = None
        local_dimensions: list[DimensionDefinition] = []
        local_measures: list[MeasureDefinition] = []
        for column in column_details if isinstance(column_details, list) else []:
            if not isinstance(column, dict) or not isinstance(column.get("name"), str):
                continue
            physical = str(column["name"])
            logical = _logical_key(physical, fallback=f"field_{len(local_dimensions) + 1}")
            value_type = _value_type(str(column.get("type", "STRING")))
            is_identifier = logical == "id" or logical.endswith(("_id", "_code", "_no"))
            if value_type in {"INTEGER", "DECIMAL"} and not is_identifier:
                local_measures.append(MeasureDefinition(
                    name=_logical_key(f"{dataset_key}_{logical}", fallback="measure"),
                    display_name=(str(column.get("comment")) if column.get("comment") else physical),
                    dataset=dataset_key, physical_column=physical,
                    aggregation="SUM", value_type=value_type, sensitivity="INTERNAL",
                ))
            else:
                local_dimensions.append(DimensionDefinition(
                    name=_logical_key(f"{dataset_key}_{logical}", fallback="dimension"),
                    display_name=(str(column.get("comment")) if column.get("comment") else physical),
                    dataset=dataset_key, physical_column=physical,
                    value_type=value_type, groupable=True, filterable=True,
                    sensitivity="SENSITIVE" if any(token in logical for token in ("phone", "mobile", "email", "id_card")) else "INTERNAL",
                ))
                if primary_time is None and value_type in {"DATE", "DATETIME"}:
                    primary_time = local_dimensions[-1].name
        local_measures.insert(0, MeasureDefinition(
            name=_logical_key(f"{dataset_key}_count", fallback="row_count"),
            display_name=f"{row.object_name}数量",
            dataset=dataset_key, physical_column=None, aggregation="COUNT",
            value_type="INTEGER", sensitivity="INTERNAL",
        ))
        datasets.append(DatasetDefinition(
            name=dataset_key,
            display_name=str(metadata.get("comment") or row.object_name),
            physical_schema=row.schema_name, physical_object=row.object_name,
            primary_time_dimension=primary_time,
        ))
        dimensions.extend(local_dimensions)
        measures.extend(local_measures)
        if row.metadata_source == "MANUAL":
            warnings.append(f"{row.schema_name}.{row.object_name} 使用人工补录结构，需要重点复核。")
    warnings.append("候选模型由字段类型和命名规则生成；金额口径、状态过滤、关联关系和敏感级别必须人工确认。")
    return SemanticModelCandidate(
        data_source_id=source.data_source_id,
        schema_snapshot_id=snapshot_id,
        definition=SemanticModelDefinition(
            datasets=tuple(datasets), dimensions=tuple(dimensions), measures=tuple(measures),
        ),
        warnings=tuple(warnings),
    )


async def enrich_semantic_candidate(
    *, candidate: SemanticModelCandidate, command: SemanticModelCandidateRequest, uow_factory,
    model_config_client: Any, model_client: Any,
) -> SemanticModelCandidate:
    """用 LLM 增强业务标签，但绝不允许模型改变物理映射和计算定义。"""
    if command.ai_model_id is None:
        return candidate
    try:
        model = await model_config_client.get_model(command.ai_model_id)
        if int(model.get("category", 0)) != 1 or model.get("status") != "ACTIVE":
            raise ValueError("SEMANTIC_AI_MODEL_NOT_AVAILABLE")
        served_model_name = str(model["served_model_name"])
        definition = candidate.definition.model_dump(mode="json")
        metadata_context = await _load_ai_metadata_context(
            uow_factory=uow_factory, candidate=candidate,
        )
        response = await model_client.get_llm_json(
            served_model_name=served_model_name,
            prompt=[
                {
                    "role": "system",
                    "content": (
                        "你是企业数据语义建模助手。只能为给定逻辑对象建议中文业务名称、维度同义词、"
                        "敏感级别和审核警告；不得新增删除对象，不得修改 name、dataset、物理表列、类型或聚合方式。"
                        "展示名必须优先依据 column_comment 或 object_comment；将英文/下划线字段翻译为简短、"
                        "明确的中文业务词语，通常 2 至 12 个汉字（可保留必要通用缩写）。注释为空、含义不明确或"
                        "可能误导时，保留 current_display_name 并在 warnings 说明，不得猜测。"
                        "返回 JSON，包含 datasets、dimensions、measures、warnings；每项必须带原 name 和 display_name。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"业务目标：{command.business_context or '未提供'}\n"
                        f"结构注释上下文：{metadata_context}\n"
                        f"规则候选：{definition}"
                    )[:60_000],
                },
            ],
            max_tokens=4096,
        )
        enriched = _merge_ai_labels(definition=definition, suggestions=response)
        warnings = tuple(candidate.warnings) + tuple(
            str(item)[:500] for item in response.get("warnings", [])
            if isinstance(item, str)
        )
        return candidate.model_copy(update={
            "definition": SemanticModelDefinition.model_validate(enriched),
            "warnings": warnings,
        })
    except Exception:
        # AI 只是建议层；不可用时保留确定性候选，不把模型服务变成数据接入单点故障。
        return candidate.model_copy(update={
            "warnings": tuple(candidate.warnings) + ("AI 增强暂时不可用，本次保留规则生成结果。",),
        })


async def _load_ai_metadata_context(*, uow_factory, candidate: SemanticModelCandidate) -> dict[str, list[dict[str, Any]]]:
    """读取已获授权 Snapshot 的注释，供 AI 为展示名称提供明确依据。"""
    async with uow_factory() as uow:
        assert uow.schema_snapshot_objects
        rows = await uow.schema_snapshot_objects.list_by_snapshot(
            schema_snapshot_id=candidate.schema_snapshot_id,
        )
        await uow.commit()

    object_metadata = {
        (row.schema_name, row.object_name): row.metadata_json or {}
        for row in rows
    }
    datasets: list[dict[str, Any]] = []
    fields: list[dict[str, Any]] = []
    for dataset in candidate.definition.datasets:
        metadata = object_metadata.get((dataset.physical_schema, dataset.physical_object), {})
        datasets.append({
            "name": dataset.name,
            "current_display_name": dataset.display_name,
            "physical_object": dataset.physical_object,
            "object_comment": metadata.get("comment"),
        })
    for section, definitions in (("dimensions", candidate.definition.dimensions), ("measures", candidate.definition.measures)):
        for item in definitions:
            dataset = next((value for value in candidate.definition.datasets if value.name == item.dataset), None)
            metadata = object_metadata.get((dataset.physical_schema, dataset.physical_object), {}) if dataset else {}
            column_comment = None
            if item.physical_column:
                column_comment = next((
                    column.get("comment") for column in metadata.get("column_details", [])
                    if isinstance(column, dict) and column.get("name") == item.physical_column
                ), None)
            fields.append({
                "section": section,
                "name": item.name,
                "dataset": item.dataset,
                "physical_column": item.physical_column,
                "current_display_name": item.display_name,
                "column_comment": column_comment,
            })
    return {"datasets": datasets, "fields": fields}


def _merge_ai_labels(*, definition: dict[str, Any], suggestions: dict[str, Any]) -> dict[str, Any]:
    result = {
        "datasets": [dict(item) for item in definition.get("datasets", [])],
        "dimensions": [dict(item) for item in definition.get("dimensions", [])],
        "measures": [dict(item) for item in definition.get("measures", [])],
    }
    for section in ("datasets", "dimensions", "measures"):
        proposed = suggestions.get(section)
        if not isinstance(proposed, list):
            continue
        by_name = {
            str(item.get("name")): item for item in proposed
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        for item in result[section]:
            suggestion = by_name.get(str(item.get("name")))
            if not suggestion:
                continue
            display_name = suggestion.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                # 名称面向业务用户；避免模型回传冗长说明，仍允许必要的业务缩写。
                normalized_display_name = re.sub(r"\s+", " ", display_name).strip()
                item["display_name"] = normalized_display_name[:32]
            if section == "dimensions":
                synonyms = suggestion.get("synonyms")
                if isinstance(synonyms, list):
                    item["synonyms"] = tuple(
                        value.strip()[:128] for value in synonyms[:32]
                        if isinstance(value, str) and value.strip()
                    )
            if section in {"dimensions", "measures"} and suggestion.get("sensitivity") in {"PUBLIC", "INTERNAL", "SENSITIVE"}:
                item["sensitivity"] = suggestion["sensitivity"]
    return result
