"""Knowledge Core API、Parser 与 Projection Worker 配置。"""

from functools import lru_cache

from pydantic import BaseModel, Field

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


class KnowledgeCoreApiConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-core-api"
    service_port: int = 18090
    receipt_ttl_seconds: int = Field(default=86400, ge=60, le=604800)


class KnowledgeStorageConfig(BaseModel):
    local_object_storage_path: str = "./knowledge_core_storage"


class ParserWorkerConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-core-parser"
    service_port: int = 18095
    worker_id: str = Field(default="kc-parser-local", min_length=1, max_length=256)
    local_artifacts_path: str = "./cached_models/docling"
    claim_interval_seconds: float = Field(default=2.0, ge=0.2, le=60)
    lease_seconds: int = Field(default=600, ge=30, le=3600)
    evidence_batch_size: int = Field(default=100, ge=1, le=500)


class ProjectionWorkerConfig(BaseModel):
    service_name: str = "kbot-knowledge-core-projection"
    worker_id: str = Field(
        default="kc-projection-local", min_length=1, max_length=256
    )
    poll_interval_seconds: float = Field(default=2.0, ge=0.2, le=60)
    lease_seconds: int = Field(default=600, ge=30, le=3600)
    index_batch_size: int = Field(default=64, ge=1, le=500)


class ParsePolicyConfig(BaseModel):
    vlm_model: str | None = None
    visual_description_prompt: str = (
        "请客观描述图片中的可见事实、文字、对象及其关系；"
        "不要推测图片之外的信息。"
    )


class EmbeddingDependencyConfig(ServiceDependencyConfig):
    health_check_timeout_seconds: int = Field(default=10, ge=1, le=120)


class KnowledgeCoreSettings(Settings):
    api: KnowledgeCoreApiConfig = Field(default_factory=KnowledgeCoreApiConfig)
    storage: KnowledgeStorageConfig = Field(default_factory=KnowledgeStorageConfig)
    parser: ParserWorkerConfig = Field(default_factory=ParserWorkerConfig)
    projection: ProjectionWorkerConfig = Field(
        default_factory=ProjectionWorkerConfig
    )
    parse_policy: ParsePolicyConfig = Field(default_factory=ParsePolicyConfig)
    embedding: EmbeddingDependencyConfig = Field(
        default_factory=lambda: EmbeddingDependencyConfig(
            base_url="http://127.0.0.1:18091",
            audience="kbot-model-embedding",
            timeout_seconds=300,
        )
    )
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
            timeout_seconds=600,
        )
    )


@lru_cache(maxsize=1)
def get_knowledge_core_settings() -> KnowledgeCoreSettings:
    return load_settings(KnowledgeCoreSettings, service="knowledge_core")
