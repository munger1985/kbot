"""版本化原生字段映射；只匹配受控目录模板，不猜动态 SQL 或 Exporter 的语义。"""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aiops_agent.contracts.measurements import NativeFieldLineage, NativeMeasurementLineage, NativeSeriesScope
from aiops_agent.playbooks import canonical_hash


MAPPING_PATH = Path(__file__).resolve().parents[2] / "resources/metrics/native_fields.v1.json"
MAX_NATIVE_SERIES = 64
MAX_MODEL_NATIVE_VALUES = 256


class NativeMappingProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_type: str
    relation: str
    dimensions: dict[str, str]
    nullable_dimensions: tuple[str, ...] = ()
    connection_dimensions: dict[str, str] = Field(default_factory=dict)
    reset_scope: str
    reset_context_columns: tuple[str, ...] = ()
    fields: tuple[NativeFieldLineage, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_profile(self):
        if (not set(self.nullable_dimensions) <= self.dimensions.keys()
                or self.dimensions.keys() & self.connection_dimensions.keys()
                or len({item.column for item in self.fields}) != len(self.fields)):
            raise ValueError("原生字段映射的维度或输出重复")
        row_keys = self.dimensions.keys() | self.connection_dimensions.keys()
        if any(row_keys & dict(item.native_dimensions).keys() for item in self.fields):
            raise ValueError("字段固定维度不能覆盖逐行或连接维度")
        return self


class NativeMappingBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    database_type: str
    tool_id: str
    tool_version: str
    variant: str
    template_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    profile: str
    output_columns: tuple[tuple[str, str], ...]


@lru_cache(maxsize=1)
def native_mapping_catalog():
    raw = MAPPING_PATH.read_bytes()
    payload = json.loads(raw)
    if payload.get("schema_version") != "NATIVE_FIELD_MAPPINGS.v1":
        raise ValueError("原生字段映射版本无效")
    profiles = {name: NativeMappingProfile.model_validate(item) for name, item in payload["profiles"].items()}
    bindings = {}
    for item in payload["bindings"]:
        binding = NativeMappingBinding.model_validate(item)
        profile = profiles[binding.profile]
        key = (binding.database_type, binding.tool_id, binding.tool_version, binding.variant, binding.template_sha256)
        if key in bindings or binding.database_type != profile.database_type:
            raise ValueError("原生字段映射的模板身份无效或重复")
        columns = dict(binding.output_columns)
        if len(columns) != len(binding.output_columns) or not all(
            field.column in columns and columns[field.column] in {"INTEGER", "DECIMAL"} for field in profile.fields
        ):
            raise ValueError("原生字段映射未定位模板中的数值输出")
        bindings[key] = (binding, profile)
    return hashlib.sha256(raw).hexdigest(), bindings


@lru_cache(maxsize=1)
def native_mapping_tool_ids():
    return frozenset(binding.tool_id for binding, _ in native_mapping_catalog()[1].values())


def _dimension_value(value, *, nullable=False):
    if value is None:
        return (nullable, None)
    if isinstance(value, bool):
        return True, "true" if value else "false"
    if not isinstance(value, (str, int)):
        from decimal import Decimal
        if not isinstance(value, Decimal) or not value.is_finite() or value != value.to_integral_value():
            return False, None
    result = str(value)
    if not result or len(result) > 256 or any(ord(char) < 32 for char in result):
        return False, None
    return True, result


def _mapping_definition(profile):
    """语义摘要忽略无作用的空注解；扩充目录不能撤销未改变的既有映射。"""
    result = profile.model_dump(mode="json")
    for field in result["fields"]:
        for name, empty in (("native_dimensions", []), ("native_operand_units", []),
                            ("aggregation", "NONE"), ("estimated", False)):
            if field[name] == empty:
                field.pop(name)
    return result


def database_native_lineage(observation, physical_identity):
    """输出列、引擎、工具版本和模板摘要全部匹配后，才给出本地已审计语义。"""
    _, bindings = native_mapping_catalog()
    key = tuple(getattr(observation, name, None) for name in (
        "db_type", "tool_id", "tool_version", "variant", "template_sha256",
    ))
    match = bindings.get(key)
    if match is None:
        return None
    binding, profile = match
    mapping_sha256 = canonical_hash({
        "schema_version": "NATIVE_FIELD_MAPPINGS.v1",
        "binding": binding.model_dump(mode="json"), "profile": _mapping_definition(profile),
    })
    columns = tuple((item.name, item.logical_type) for item in observation.columns)
    if columns != binding.output_columns or observation.row_count != len(observation.rows):
        return None
    visible = {item.name for item in observation.columns if item.sensitivity == "PUBLIC"}
    if any(item.column not in visible for item in profile.fields):
        return None
    scopes = []
    for index, row in enumerate(observation.rows[:MAX_NATIVE_SERIES]):
        if len(row) != len(columns):
            return None
        mapped = dict(zip((name for name, _ in columns), row, strict=True))
        dimensions, missing = [], []
        for native, column in profile.dimensions.items():
            valid, value = _dimension_value(mapped.get(column), nullable=native in profile.nullable_dimensions) if column in mapped and column in visible else (False, None)
            if valid:
                dimensions.append((native, value))
            else:
                missing.append(native)
        for native, column in profile.connection_dimensions.items():
            valid, value = _dimension_value(physical_identity.get(column))
            if valid:
                dimensions.append((native, value))
            else:
                missing.append(native)
        scopes.append(NativeSeriesScope(row_index=index, dimensions=tuple(sorted(dimensions)), missing_dimensions=tuple(sorted(missing))))
    return NativeMeasurementLineage(
        mapping_sha256=mapping_sha256, template_sha256=observation.template_sha256,
        database_type=profile.database_type, relation=profile.relation, fields=profile.fields,
        series=tuple(scopes), reset_scope=profile.reset_scope,
        reset_context_columns=tuple(column for column in profile.reset_context_columns if column in visible),
        series_truncated=observation.row_count > MAX_NATIVE_SERIES,
    )


def native_counter_origins(provenance, *, row_index, column):
    """同一原生字段经不同模板读取仍是同源；不由来源数量推导因果或独立互证。"""
    from aiops_agent.domain.evidence.native_origin import database_origin_anchor

    lineage = provenance.native_lineage
    if lineage is None or "PHYSICAL_IDENTITY_UNKNOWN" in provenance.quality_flags:
        return ()
    series = next((item for item in lineage.series if item.row_index == row_index), None)
    field = next((item for item in lineage.fields if item.column == column), None)
    if series is None or field is None or series.missing_dimensions:
        return ()
    physical = dict(provenance.physical_identity)
    anchor = database_origin_anchor(lineage.database_type, physical)
    if anchor is None:
        return ()
    # RAC 跨实例游标保留原生 INST_ID；连接实例的启动时间不能替另一实例证明连续性。
    row_dbid = dict(series.dimensions).get("dbid")
    if lineage.database_type == "ORACLE" and row_dbid is not None and row_dbid != physical.get("dbid"):
        return ()
    dimensions = tuple(sorted((*series.dimensions, *field.native_dimensions)))
    if len(dict(dimensions)) != len(dimensions):
        return ()
    return tuple("native-counter:" + canonical_hash({
        "database": anchor, "relation": lineage.relation, "column": native,
        "dimensions": dimensions,
    }) for native in field.native_columns)


def native_row_projection(provenance, *, row_index, row):
    """给模型的是程序换算的已返回值；取整、缺失和不可比状态不会随换算消失。"""
    from aiops_agent.domain.evidence.native_measurements import canonical_native_value

    if provenance is None or provenance.native_lineage is None:
        return {}
    result = {}
    for field in provenance.native_lineage.fields:
        if field.column not in row:
            continue
        value = canonical_native_value(field, row[field.column])
        result[field.column] = {
            "value": str(value) if value is not None else None,
            "unit": field.unit, "kind": field.kind,
            "quality": "MISSING" if row[field.column] is None else "INVALID" if value is None else "ESTIMATED" if field.estimated else "ROUNDED" if field.projection == "ROUNDED" else "OBSERVED_VALUE_CONVERSION",
            "projection": field.projection, "aggregation": field.aggregation,
            "native_dimensions": field.native_dimensions,
            "native_origin_ids": native_counter_origins(provenance, row_index=row_index, column=field.column),
            "comparable_window_verified": False,
        }
    return result


def native_table_projection(provenance, *, columns, rows):
    """为任意 DBA 问题提供有界程序换算；未映射字段仍在原证据中，不猜其语义。"""
    if provenance is None or provenance.native_lineage is None:
        return None
    names = tuple(column["name"] for column in columns)
    projected, count, truncated = [], 0, len(rows) > MAX_NATIVE_SERIES
    for index, row in enumerate(rows[:MAX_NATIVE_SERIES]):
        if len(row) != len(names):
            truncated = True
            continue
        values = native_row_projection(provenance, row_index=index, row=dict(zip(names, row, strict=True)))
        remaining = MAX_MODEL_NATIVE_VALUES - count
        if len(values) > remaining:
            values = dict(tuple(values.items())[:remaining])
            truncated = True
        if values:
            projected.append({"row_index": index, "values": values})
            count += len(values)
        if count == MAX_MODEL_NATIVE_VALUES:
            truncated = truncated or index + 1 < len(rows)
            break
    return {"rows": projected, "measurement_count": count, "truncated": truncated,
            "verification_scope": "MAPPED_FIELDS_ONLY", "comparable_window_verified": False}
