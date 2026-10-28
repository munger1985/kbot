"""计划采集专用执行协议；参数兑换与普通只读授权严格隔离。"""

from typing import Literal

from pydantic import Field, SecretStr, model_validator

from .executor import DiagnosticConnectionProfile, ResourceLeaseReceipt
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


class SqlPlanCaptureStart(AIOpsContract):
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SqlPlanCaptureGrant(AIOpsContract):
    schema_version: Literal["sql-plan-capture-grant.v1"] = "sql-plan-capture-grant.v1"
    issuer: str = Field(min_length=1, max_length=128)
    audience: str = Field(min_length=1, max_length=128)
    grant_id: UUIDv7
    issued_at: UtcDatetime
    expires_at: UtcDatetime
    run_id: UUIDv7
    task_id: UUIDv7
    source_run_id: UUIDv7
    request_id: UUIDv7
    approval_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    lease_token_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    db_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    trace_id: str = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def validate_lifetime(self):
        if not 0 < (self.expires_at - self.issued_at).total_seconds() <= 45:
            raise ValueError("计划采集Grant有效期必须在45秒以内")
        return self


class SqlPlanCaptureExecute(AIOpsContract):
    grant: str = Field(min_length=64, max_length=16384, repr=False)


class SqlPlanCaptureExchange(AIOpsContract):
    database_type: Literal["POSTGRESQL", "MYSQL", "ORACLE"]
    connection_profile: DiagnosticConnectionProfile
    username: SecretStr
    password: SecretStr
    source_sql: SecretStr
    protected_document: SecretStr
    allowed_objects: tuple[str, ...] = Field(max_length=64)
    statement_timeout_seconds: int = Field(ge=1, le=10, strict=True)
    source_subject: dict[str, str | bool | None] = Field(max_length=16)
    source_identity: dict[str, str] = Field(default_factory=dict, max_length=8)
    resource_lease: ResourceLeaseReceipt = Field(repr=False)

    @model_validator(mode="after")
    def validate_protected_exchange(self):
        _physical_identity(self.database_type, self.source_identity)
        if (not self.allowed_objects or any(len(name) > 256 for name in self.allowed_objects)
            or len(self.source_sql.get_secret_value()) > 20000
            or len(self.protected_document.get_secret_value().encode("utf-8")) > 65536):
            raise ValueError("计划采集保护输入超出有界范围")
        if self.database_type == "ORACLE" and not self.connection_profile.service:
            raise ValueError("Oracle计划采集必须指定服务")
        if self.database_type != "ORACLE" and not self.connection_profile.database:
            raise ValueError("计划采集必须指定数据库")
        return self

    def protected_transport(self):
        # 仅专用反向兑换路由调用；默认repr/JSON仍不泄露值。
        payload = self.model_dump(mode="json")
        for key in ("username", "password", "source_sql", "protected_document"):
            payload[key] = getattr(self, key).get_secret_value()
        return payload


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
    nodes: tuple[SqlPlanEstimateNode, ...] = Field(default=(), max_length=128)
    mode: Literal["ESTIMATE_ONLY"] = "ESTIMATE_ONLY"
    original_parse_context_verified: Literal[False] = False
    representative_parameters: Literal["USER_DECLARED_UNVERIFIED"] = "USER_DECLARED_UNVERIFIED"
    production_execution_verified: Literal[False] = False
    oracle_parameter_peeking: Literal["NOT_PROVEN", "NOT_APPLICABLE"] = "NOT_APPLICABLE"

    @model_validator(mode="after")
    def validate_result(self):
        _physical_identity(self.database_type, self.physical_identity)
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


class SqlPlanCaptureReport(AIOpsContract):
    result: SqlPlanCaptureResult
    resource_lease: ResourceLeaseReceipt = Field(repr=False)
