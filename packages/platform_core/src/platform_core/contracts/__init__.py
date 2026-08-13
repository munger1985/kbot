"""跨平台服务客户端共享的版本化契约。"""

from .api_versions import INTERNAL_API_V1, PUBLIC_API_V1
from .agent import (
    AgentExecutionSpec,
    AgentArtifact,
    AgentArtifactRef,
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
    CreateAgentRunRequest,
)
from .conversation import (
    ConversationItemView,
    ConversationTurnPage,
    ConversationTurnReceipt,
    ConversationTurnView,
    ConversationView,
    CreateConversationRequest,
    CreateConversationTurnRequest,
    ConversationQueryImage,
    MemoryItemView,
    PublicTraceEvent,
    UpdateConversationRequest,
)
from .identity import AuthContext, PrincipalKind, ServiceIdentity
from .slack import SlackWebhookEnvelope, SlackWebhookReceipt
from .model import (
    EmbeddingDataItem,
    ModelArchiveRequest,
    ModelCatalogItem,
    ModelCreateRequest,
    ModelDeleteRequest,
    ModelProviderOption,
    ModelReference,
    ModelReferenceSummary,
    ModelStatusRequest,
    ModelUpdateRequest,
)
from . import data_query

__all__ = [
    "AgentExecutionSpec",
    "AuthContext",
    "AgentArtifactRef",
    "AgentArtifact",
    "AgentRunEvent",
    "AgentRunReceipt",
    "AgentRunSummary",
    "CreateAgentRunRequest",
    "ConversationItemView",
    "ConversationTurnPage",
    "ConversationTurnReceipt",
    "ConversationTurnView",
    "ConversationView",
    "CreateConversationRequest",
    "CreateConversationTurnRequest",
    "ConversationQueryImage",
    "MemoryItemView",
    "PublicTraceEvent",
    "UpdateConversationRequest",
    "EmbeddingDataItem",
    "ModelArchiveRequest",
    "ModelCatalogItem",
    "ModelCreateRequest",
    "ModelDeleteRequest",
    "ModelProviderOption",
    "ModelReference",
    "ModelReferenceSummary",
    "ModelStatusRequest",
    "ModelUpdateRequest",
    "PrincipalKind",
    "ServiceIdentity",
    "SlackWebhookEnvelope",
    "SlackWebhookReceipt",
    "PUBLIC_API_V1",
    "INTERNAL_API_V1",
    "data_query",
]
