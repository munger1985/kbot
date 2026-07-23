"""Main API 的公开 HTTP 路由。"""

from .knowledge import router as knowledge_router

__all__ = ["knowledge_router"]
