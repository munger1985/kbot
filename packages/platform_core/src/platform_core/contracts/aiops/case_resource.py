"""系统案例资源的窄合同；业务空间与模型配置状态分别表达。"""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


CASE_RESOURCE_PATH = "/internal/v1/knowledge/system/aiops-case-collections:ensure"
CASE_RESOURCE_SCOPE = "knowledge.aiops_case.ensure"


class CaseResourceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tenant_id: UUID | None = None
    domain_id: str = Field(min_length=1, max_length=80)
    actor_id: str = Field(min_length=1, max_length=256)
    expected_collection_id: UUID | None = None
    allow_create: bool = True


class CaseResourceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request: CaseResourceRequest
    collection_id: UUID
    missing_model_roles: list[str]
    configuration_status: Literal["MODELS_REQUIRED", "CONFIGURED", "TEXT_POLICY_REQUIRED"]


class CaseResourceStatus(BaseModel):
    resource_status: Literal["UNINITIALIZED", "QUEUED", "RUNNING", "READY", "FAILED", "PAUSED"]
    collection_id: UUID | None = None
    configuration_status: str = "UNKNOWN"
    missing_model_roles: list[str] = Field(default_factory=list)
    error_code: str | None = None
