"""跨平台服务客户端共享的版本化契约。"""

from .api_versions import INTERNAL_API_V1, PUBLIC_API_V1
from .model import EmbeddingDataItem

__all__ = ["EmbeddingDataItem", "PUBLIC_API_V1", "INTERNAL_API_V1"]
