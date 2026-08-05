"""Main API 后台组合编排的稳定公开契约。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from .agent import CreateAgentDefinitionRequest, UpdateAgentDefinitionRequest


Availability = Literal["AVAILABLE", "MISSING", "UNAVAILABLE", "STALE"]
CompositionStatus = Literal[
    "PRECHECKING",
    "COMMAND_SUBMITTED",
    "SUCCEEDED",
    "FAILED_PRECHECK",
    "COMPENSATION_REQUIRED",
]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CompositionNode(_Contract):
    node_type: str = Field(min_length=1, max_length=80)
    resource_id: str = Field(min_length=1, max_length=256)
    source_service: str = Field(min_length=1, max_length=64)
    source_version: str | None = Field(default=None, max_length=128)
    observed_at: datetime
    availability: Availability
    status: str | None = Field(default=None, max_length=64)
    attributes: dict[str, Any] = Field(default_factory=dict)


class CompositionEdge(_Contract):
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    relation: str
    blocking: bool = False


class ResourceReferenceGraph(_Contract):
    root_type: str
    root_id: str
    observed_at: datetime
    nodes: tuple[CompositionNode, ...]
    edges: tuple[CompositionEdge, ...]
    blockers: tuple[CompositionEdge, ...] = ()
    partial: bool = False


class CompositionReceipt(_Contract):
    receipt_id: UUID
    operation: str
    idempotency_key: str
    status: CompositionStatus
    resource_type: str
    resource_id: str | None = None
    resource_version: str | None = None
    error_code: str | None = None
    verification: dict[str, Any] = Field(default_factory=dict)
    idempotent_replay: bool = False
    created_at: datetime
    updated_at: datetime


class AgentDataQueryBindingDefinition(_Contract):
    semantic_model_id: UUID
    policy_binding_id: UUID


class AgentCompositionCreate(_Contract):
    agent: CreateAgentDefinitionRequest
    collection_ids: tuple[UUID, ...] = ()
    data_query_binding: AgentDataQueryBindingDefinition | None = None


class AgentCompositionUpdate(_Contract):
    agent: UpdateAgentDefinitionRequest
    collection_ids: tuple[UUID, ...] | None = None
    data_query_binding: AgentDataQueryBindingDefinition | None = None


class CollectionCompositionDefinition(_Contract):
    display_name: str = Field(min_length=1, max_length=256)
    models: dict[str, UUID]
    description: str | None = Field(default=None, max_length=1000)
    default_security_level: int = Field(default=1, ge=0, le=999)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionCompositionCreate(_Contract):
    collection: CollectionCompositionDefinition


class CollectionModelsCompositionUpdate(_Contract):
    models: dict[str, UUID]


class SemanticModelBindingDefinition(_Contract):
    agent_id: UUID
    policy_binding_id: UUID


class SemanticModelPublicationComposition(_Contract):
    schema_snapshot_id: UUID
    validation_model_id: UUID
    binding: SemanticModelBindingDefinition | None = None


class ResourceDecommissionPrecheck(_Contract):
    action: Literal["ARCHIVE", "DELETE", "DISABLE"]


class ResourceDecommissionResult(_Contract):
    action: Literal["ARCHIVE", "DELETE", "DISABLE"]
    allowed: bool
    graph: ResourceReferenceGraph


class RunCompositionView(_Contract):
    run_id: UUID
    observed_at: datetime
    run: CompositionNode
    agent: CompositionNode
    models: tuple[CompositionNode, ...]
    collections: tuple[CompositionNode, ...]
    semantic_models: tuple[CompositionNode, ...]
    data_sources: tuple[CompositionNode, ...]
    data_query_runs: tuple[CompositionNode, ...]
    knowledge_evidence: tuple[CompositionNode, ...]
    artifacts: tuple[CompositionNode, ...]
    notifications: tuple[dict[str, Any], ...]
    tasks: tuple[dict[str, Any], ...]
    partial: bool = False
