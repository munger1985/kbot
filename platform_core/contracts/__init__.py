"""跨平台服务客户端共享的版本化契约。"""

from .api_versions import INTERNAL_API_V1, PUBLIC_API_V1
from .agent import (
    AgentArtifactRef,
    AgentRunEvent,
    AgentRunReceipt,
    AgentRunSummary,
    CreateAgentRunRequest,
)
from .identity import AuthContext, PrincipalKind
from .model import EmbeddingDataItem

__all__ = [
    "AuthContext",
    "AgentArtifactRef",
    "AgentRunEvent",
    "AgentRunReceipt",
    "AgentRunSummary",
    "CreateAgentRunRequest",
    "EmbeddingDataItem",
    "PrincipalKind",
    "PUBLIC_API_V1",
    "INTERNAL_API_V1",
]
