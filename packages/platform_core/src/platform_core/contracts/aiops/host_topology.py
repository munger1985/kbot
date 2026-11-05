"""共享主机的只读承载投影，不包含连接参数或凭据。"""

from uuid import UUID

from pydantic import Field

from .types import AIOpsContract


class HostTopologySystem(AIOpsContract):
    business_system_id: UUID
    display_name: str
    status: str


class HostTopologyResource(AIOpsContract):
    resource_id: UUID
    resource_type_key: str
    display_name: str
    location_name: str | None = None
    business_systems: tuple[HostTopologySystem, ...] = ()


class HostTopologyParticipant(AIOpsContract):
    resource_id: UUID
    participant_role: str
    display_order: int


class HostTopologyRelation(AIOpsContract):
    relation_id: UUID
    business_system_id: UUID
    relation_type_key: str
    display_name: str | None = None
    participants: tuple[HostTopologyParticipant, ...]


class HostTopologyView(AIOpsContract):
    host_resource_id: UUID
    resources: tuple[HostTopologyResource, ...] = ()
    relations: tuple[HostTopologyRelation, ...] = ()
    truncated: bool = False
    max_nodes: int = Field(ge=1, le=500)
