# platform_clients/__init__.py — 跨服务调用客户端

from .agent_runtime import AgentRuntimeClient, AgentRuntimeClientError
from .media_studio import MediaStudioClient, MediaStudioClientError
from .aiops import (
    AIOpsClientAuth,
    AIOpsClientError,
    AIOpsDelegationClient,
    AIOpsManagementClient,
)
from .model import AIModelClient, AIModelConfigClient
from .generative import GenerativeResponsesClient, GenerativeResponsesClientError
from .knowledge_core import (
    KnowledgeCoreClient,
    KnowledgeCoreClientError,
    KnowledgeCoreResponse,
    KnowledgeCoreStreamResponse,
)
from .mcp_data import MCPDataClient, MCPDataClientError
from .data_query import DataQueryClient, DataQueryClientError
from .knowledge_retrieval_app import (
    KnowledgeRetrievalAppClient,
    KnowledgeRetrievalAppClientError,
)
from .km_asset import KmAssetClient, KmAssetClientError
from .km_portal import KmPortalClient, KmPortalClientError

__all__ = [
    "AgentRuntimeClient",
    "AgentRuntimeClientError",
    "MediaStudioClient",
    "MediaStudioClientError",
    "AIOpsClientAuth",
    "AIOpsClientError",
    "AIOpsDelegationClient",
    "AIOpsManagementClient",
    "AIModelClient",
    "AIModelConfigClient",
    "GenerativeResponsesClient",
    "GenerativeResponsesClientError",
    "KnowledgeCoreClient",
    "KnowledgeCoreClientError",
    "KnowledgeCoreResponse",
    "KnowledgeCoreStreamResponse",
    "MCPDataClient",
    "MCPDataClientError",
    "DataQueryClient",
    "DataQueryClientError",
    "KnowledgeRetrievalAppClient",
    "KnowledgeRetrievalAppClientError",
    "KmAssetClient",
    "KmAssetClientError",
    "KmPortalClient",
    "KmPortalClientError",
]
