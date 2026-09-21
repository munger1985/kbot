"""知识检索应用配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class KnowledgeRetrievalAppApiConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-retrieval-app-api"
    service_port: int = 18150


class KnowledgeRetrievalAppWorkerConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-retrieval-app-worker"
    service_port: int = 18151
    poll_interval_seconds: int = Field(default=5, ge=1, le=60)
    lease_seconds: int = Field(default=360, ge=15, le=600)


class KnowledgeRetrievalAppSettings(Settings):
    api: KnowledgeRetrievalAppApiConfig = Field(default_factory=KnowledgeRetrievalAppApiConfig)
    worker: KnowledgeRetrievalAppWorkerConfig = Field(default_factory=KnowledgeRetrievalAppWorkerConfig)
    llm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18092",
            audience="kbot-model-llm",
            timeout_seconds=300,
        )
    )


@lru_cache(maxsize=1)
def get_knowledge_retrieval_app_settings() -> KnowledgeRetrievalAppSettings:
    return load_settings(KnowledgeRetrievalAppSettings, service="knowledge_retrieval_app")
