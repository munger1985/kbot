"""知识检索应用配置。"""

from functools import lru_cache
from knowledge_retrieval_app.composition.knowledge_profiles import KnowledgeFoundationConfig

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
    knowledge: KnowledgeFoundationConfig = Field(default_factory=KnowledgeFoundationConfig)
    computation: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(
        base_url="http://127.0.0.1:18090", audience="kbot-knowledge-core-api", timeout_seconds=600))
    embedding: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(
        base_url="http://127.0.0.1:18091", audience="kbot-model-embedding", timeout_seconds=300))
    visual: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(
        base_url="http://127.0.0.1:18093", audience="kbot-model-visual", timeout_seconds=300))

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
