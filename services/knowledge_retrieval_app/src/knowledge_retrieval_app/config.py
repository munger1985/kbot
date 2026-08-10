"""知识检索应用配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import ServiceConfig, Settings, load_settings


class KnowledgeRetrievalAppApiConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-retrieval-app-api"
    service_port: int = 18150


class KnowledgeRetrievalAppSettings(Settings):
    api: KnowledgeRetrievalAppApiConfig = Field(
        default_factory=KnowledgeRetrievalAppApiConfig
    )


@lru_cache(maxsize=1)
def get_knowledge_retrieval_app_settings() -> KnowledgeRetrievalAppSettings:
    return load_settings(
        KnowledgeRetrievalAppSettings, service="knowledge_retrieval_app"
    )
