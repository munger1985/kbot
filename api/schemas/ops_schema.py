# api/schemas/ops_schema.py

from pydantic import BaseModel, Field
from typing import Literal, Any


class CreateInstanceRequest(BaseModel):
    """录入新数据库实例的请求体"""
    instance_name: str = Field(..., min_length=1, max_length=128)
    environment: Literal["dev", "stg", "prod"] = "dev"
    security_level: int = Field(3, ge=1, le=5)
    db_type: str = Field(..., min_length=1, max_length=32)
    version_code: int
    charset: str = "utf8mb4"
    host: str = Field(..., min_length=1, max_length=256)
    port: int = Field(..., gt=0, lt=65536)
    service_name: str | None = None
    database_name: str | None = None
    dsn: str | None = None
    db_role: Literal["primary", "standby", "cluster_node"] = "primary"
    cluster_id: str | None = None
    ops_user: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=512)
    secret_vault_key: str | None = None
    status: Literal["active", "maintenance", "offline"] = "active"


class OpsChatRequest(BaseModel):
    """AIOps 统一请求体契约
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str"""
    agent_id: int = Field(..., description="智能体 ID (int), 用于反查其绑定的多物理实例资产池")
    instance_id: str = Field(..., description="数据库实例 ID (str), 用于执行运维指令")
    query: str = Field(..., description="自然语言运维指令或故障描述")
    session_id: str | None = Field(None, description="会话 ID, 不传或传 'new_session' 则自动生成")
    user_id: str = Field(..., description="用户 ID, 用于关联会话和指令执行记录")


class UpdateInstanceRequest(BaseModel):
    """更新数据库实例的请求体 — 所有字段均为可选"""
    instance_name: str | None = Field(None, min_length=1, max_length=128)
    environment: Literal["dev", "stg", "prod"] | None = None
    security_level: int | None = Field(None, ge=1, le=5)
    db_type: str | None = Field(None, min_length=1, max_length=32)
    version_code: int | None = None
    charset: str | None = None
    host: str | None = Field(None, min_length=1, max_length=256)
    port: int | None = Field(None, gt=0, lt=65536)
    service_name: str | None = None
    database_name: str | None = None
    dsn: str | None = None
    db_role: Literal["primary", "standby", "cluster_node"] | None = None
    monitor_type: str | None = None
    prometheus_instance_label: str | None = None
    cluster_id: str | None = None
    ops_user: str | None = Field(None, min_length=1, max_length=64)
    password: str | None = Field(None, min_length=1, max_length=512)
    secret_vault_key: str | None = None
    status: Literal["active", "maintenance", "offline"] | None = None


class AgentOpsBindForm(BaseModel):
    """智能体绑定运维实例的请求体
    【⚠️ ID 类型】: agent_id 为 int, instance_id 为 str"""
    agent_id: int = Field(..., description="智能体 ID (int)")
    instance_id: str = Field(..., description="数据库实例 ID (str)")
    is_mutation_allowed: bool = False
    require_approval: bool = True
    max_daily_execution: int = 10


class AgentOpsBindUpdateForm(BaseModel):
    """更新智能体与运维实例绑定策略的请求体"""
    is_mutation_allowed: bool | None = None
    require_approval: bool | None = None
    max_daily_execution: int | None = None


class OpsResumeRequest(BaseModel):
    """HITL 恢复执行请求体 — 用户提交采集到的数据并恢复诊断"""
    request_id: str = Field(
        ..., description="挂起请求 ID（来自 WAIT_FOR_USER 包的 request_id）"
    )
    user_data: dict[str, Any] | None = Field(
        None, description="用户回填的数据，key-value 形式"
    )
    user_note: str | None = Field(
        None, description="用户备注/补充说明"
    )
    user_error: str | None = Field(
        None,
        description="用户执行 SQL 时的报错信息，如 ORA-00942: table or view does not exist"
    )


class OpsApproveRequest(BaseModel):
    """HITL 审批请求体 — 用户对高危变更操作进行审批"""
    request_id: str = Field(
        ..., description="审批请求 ID（来自 REQUIRE_APPROVAL 包的 request_id）"
    )
    approved: bool = Field(
        ..., description="是否批准执行: true=批准, false=拒绝"
    )
    approver_note: str | None = Field(
        None, description="审批人备注（批准或拒绝的理由）"
    )
