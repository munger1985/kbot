"""跨平台服务客户端共享的版本化契约。"""

from .api_versions import INTERNAL_API_V1, PUBLIC_API_V1
from .agent import (
    AgentDefinition,
    AgentArtifact,
    AgentArtifactRef,
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
    CreateAgentRunRequest,
    CreateAgentDefinitionRequest,
    UpdateAgentDefinitionRequest,
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
from .model import EmbeddingDataItem

__all__ = [
    "AuthContext",
    "AgentArtifactRef",
    "AgentArtifact",
    "AgentDefinition",
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
    "CreateAgentDefinitionRequest",
    "EmbeddingDataItem",
    "PrincipalKind",
    "ServiceIdentity",
    "UpdateAgentDefinitionRequest",
    "PUBLIC_API_V1",
    "INTERNAL_API_V1",
]
