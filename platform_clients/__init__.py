# platform_clients/__init__.py — 跨服务调用客户端

from .model import AIModelClient, AIModelConfigClient
from .knowledge_core import (
    KnowledgeCoreClient,
    KnowledgeCoreClientError,
    KnowledgeCoreResponse,
)

__all__ = [
    "AIModelClient",
    "AIModelConfigClient",
    "KnowledgeCoreClient",
    "KnowledgeCoreClientError",
    "KnowledgeCoreResponse",
]
