"""Main API 的公开 HTTP 路由。"""

from .aiops_app import router as aiops_app_router
from .access_management import router as access_management_router
from .auth import router as auth_router
from .app_api_clients import router as app_api_clients_router
from .conversations import memory_router
from .conversations import router as conversation_router
from .data_query import router as data_query_router
from .development_logs import router as development_logs_router
from .development_agent_runs import router as development_agent_runs_router
from .dify import router as dify_router
from .domains import router as domain_router
from .integrations import router as integration_router
from .knowledge import router as knowledge_router
from .knowledge_retrieval_app import router as knowledge_retrieval_app_router
from .km_asset_app import router as km_asset_app_router
from .models import router as model_catalog_router
from .notifications import router as notification_router
from .ops import router as ops_router
from .runs import router as run_router
from .slack import router as slack_router

__all__ = [
    "conversation_router",
    "access_management_router",
    "aiops_app_router",
    "auth_router",
    "app_api_clients_router",
    "data_query_router",
    "development_logs_router",
    "development_agent_runs_router",
    "dify_router",
    "domain_router",
    "memory_router",
    "integration_router",
    "knowledge_router",
    "knowledge_retrieval_app_router",
    "km_asset_app_router",
    "model_catalog_router",
    "notification_router",
    "ops_router",
    "run_router",
    "slack_router",
]
