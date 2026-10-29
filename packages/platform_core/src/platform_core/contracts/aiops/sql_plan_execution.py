"""历史计划结果的只读数据合同；不提供估算请求、执行授权或参数兑换。"""

from typing import Literal

from pydantic import Field, model_validator, model_serializer

from .sql_plan_verification import SqlPlanNativeVerification

from .types import AIOpsContract, UUIDv7, UtcDatetime

def _physical_identity(database_type, identity):
    import re
    from datetime import datetime
    from uuid import UUID
    keys = {"POSTGRESQL": {"system_identifier", "database_oid", "instance_started_at"},
        "MYSQL": {"server_uuid", "instance_started_at"},
        "ORACLE": {"dbid", "container_id", "instance_number", "instance_started_at"}}[database_type]
    if set(identity) - keys:
        raise ValueError("计划采集物理身份包含未登记字段")
    for key, value in identity.items():
        if len(value) > 64 or any(ord(char) < 32 for char in value):
            raise ValueError("计划采集物理身份格式无效")
        try:
            if key == "instance_started_at":
                datetime.fromisoformat(value)
            elif key == "server_uuid":
                if UUID(value).int == 0 or str(UUID(value)) != value:
                    raise ValueError
            elif not re.fullmatch(r"(?:0|[1-9][0-9]{0,19})", value) or int(value) < (0 if key == "container_id" else 1):
                raise ValueError
        except ValueError:
            raise ValueError("计划采集物理身份格式无效") from None


class SqlPlanEstimateNode(AIOpsContract):
    node_id: int = Field(ge=0, le=127)
    parent_id: int | None = Field(default=None, ge=0, le=127)
    operation: str = Field(min_length=1, max_length=96, pattern=r"^[A-Za-z][A-Za-z0-9_ /()-]*$")
    schema_name: str | None = Field(default=None, max_length=128)
    relation_name: str | None = Field(default=None, max_length=128)
    index_name: str | None = Field(default=None, max_length=128)
    metrics: dict[str, str] = Field(default_factory=dict, max_length=24)

    @model_validator(mode="after")
    def validate_safe_projection(self):
        import re
        from decimal import Decimal, InvalidOperation
        for value in (self.schema_name, self.relation_name, self.index_name):
            if value is not None and not re.fullmatch(r"[\w$#.-]{1,128}", value):
                raise ValueError("估算计划对象名称无法安全投影")
        for key, value in self.metrics.items():
            if key not in {"estimated_rows", "optimizer_total_cost", "optimizer_cost",
                           "estimated_rows_examined_per_scan", "estimated_rows_produced_per_join",
                           "estimated_filtered_percent", "optimizer_query_cost", "optimizer_read_cost",
                           "optimizer_eval_cost", "optimizer_prefix_cost", "workers_planned"}:
                raise ValueError("估算计划只能包含已登记的估算指标")
            try:
                number = Decimal(value)
                if (len(value) > 64 or not number.is_finite() or not 0 <= number <= Decimal("1e24")
                    or number.as_tuple().exponent < -18):
                    raise ValueError
            except (ValueError, InvalidOperation):
                raise ValueError("估算计划数值无效") from None
        return self

class SqlPlanCaptureResult(AIOpsContract):
    schema_version: Literal["SQL_PLAN_CAPTURE_RESULT.v1"] = "SQL_PLAN_CAPTURE_RESULT.v1"
    capture_run_id: UUIDv7
    request_id: UUIDv7
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    status: Literal["SUCCEEDED", "FAILED"]
    error_code: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{0,95}$")
    captured_at: UtcDatetime
    physical_identity: dict[str, str] = Field(default_factory=dict, max_length=8)
    native_verification: SqlPlanNativeVerification | None = None
    nodes: tuple[SqlPlanEstimateNode, ...] = Field(default=(), max_length=128)
    mode: Literal["ESTIMATE_ONLY"] = "ESTIMATE_ONLY"
    original_parse_context_verified: Literal[False] = False
    representative_parameters: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    production_execution_verified: Literal[False] = False
    oracle_parameter_peeking: Literal["NOT_PROVEN", "NOT_APPLICABLE"] = "NOT_APPLICABLE"

    @model_validator(mode="after")
    def validate_result(self):
        _physical_identity(self.database_type, self.physical_identity)
        if self.native_verification is not None:
            if self.database_type != "ORACLE" and any(check.source == "NATIVE_BIND_CAPTURE"
                for check in (self.native_verification.parameter_types, self.native_verification.parameter_values)):
                raise ValueError("不能为其他引擎声明Oracle原生绑定捕获")
            if self.status != "SUCCEEDED" or any(item.node_id >= len(self.nodes)
                for item in self.native_verification.resolved_objects
            ) or any(index >= len(self.nodes) for index in self.native_verification.unresolved_object_nodes):
                raise ValueError("原生核验只能关联成功结果的有效节点")
        if (self.status == "SUCCEEDED") != bool(self.nodes) or (self.status == "SUCCEEDED") == bool(self.error_code):
            raise ValueError("估算结果状态、节点和错误不一致")
        if self.oracle_parameter_peeking != ("NOT_PROVEN" if self.database_type == "ORACLE" else "NOT_APPLICABLE"):
            raise ValueError("引擎与参数窥探边界不一致")
        if {node.node_id for node in self.nodes} != set(range(len(self.nodes))):
            raise ValueError("估算计划节点必须连续且唯一")
        if any(node.parent_id is not None and node.parent_id >= node.node_id for node in self.nodes):
            raise ValueError("估算计划父节点必须先于子节点")
        if self.nodes and sum(node.parent_id is None for node in self.nodes) != 1:
            raise ValueError("估算计划必须只有一个根节点")
        return self

    @model_serializer(mode="wrap")
    def preserve_historical_result(self, handler):
        result = handler(self)
        if self.native_verification is None:
            result.pop("native_verification", None)
        return result
