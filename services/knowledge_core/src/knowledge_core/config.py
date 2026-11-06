"""无业务数据库的知识解析计算服务配置。"""

from functools import lru_cache
from typing import ClassVar
from pydantic import BaseModel, Field
from platform_core.config import ServiceConfig, ServiceDependencyConfig, Settings, load_settings


class ParseComputeConfig(BaseModel):
    spool_root: str = "./var/spool/knowledge-computation"
    memory_bytes: int = Field(default=2147483648, ge=67108864, le=34359738368)
    cpu_seconds: int = Field(default=300, ge=1, le=1800)
    max_file_bytes: int = Field(default=67108864, ge=1024, le=67108864)
    max_artifact_bytes: int = Field(default=134217728, ge=1024, le=1073741824)
    max_jobs: int = Field(default=2, ge=1, le=32)
    max_unacked_events: int = Field(default=32, ge=1, le=1000)
    converter_artifacts_path: str | None = "./cached_models/docling"
    ocr: dict[str, str] | None = None


class KnowledgeCoreApiConfig(ServiceConfig):
    pass


class KnowledgeCoreSettings(Settings):
    topology_service: ClassVar[str] = "knowledge_core"
    api: KnowledgeCoreApiConfig = Field(default_factory=lambda: KnowledgeCoreApiConfig(service_name="kbot-knowledge-core-api", service_port=18090))
    compute: ParseComputeConfig = Field(default_factory=ParseComputeConfig)
    knowledge_retrieval: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(base_url="http://127.0.0.1:18150", audience="kbot-knowledge-retrieval-app-api", timeout_seconds=600))
    aiops: ServiceDependencyConfig = Field(default_factory=lambda: ServiceDependencyConfig(base_url="http://127.0.0.1:18110", audience="kbot-aiops-api", timeout_seconds=600))


@lru_cache(maxsize=1)
def get_knowledge_core_settings() -> KnowledgeCoreSettings:
    return load_settings(KnowledgeCoreSettings, service="knowledge_core")
