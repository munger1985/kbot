"""Model Serving 各独立模型进程配置。"""

from functools import lru_cache

from pydantic import Field

from platform_core.config import (
    ServiceConfig,
    ServiceDependencyConfig,
    Settings,
    load_settings,
)


class ModelProcessConfig(ServiceConfig):
    timeout: int = Field(default=300, ge=1, le=3600)
    health_check_timeout: int = Field(default=10, ge=1, le=120)
    max_retries: int = Field(default=3, ge=0, le=10)


class EmbeddingConfig(ModelProcessConfig):
    service_name: str = "kbot-model-embedding"
    service_port: int = 18091
    max_tokens: int = Field(default=1024, ge=1, le=65536)
    cache_dir: str = "./cached_models"


class LlmConfig(ModelProcessConfig):
    service_name: str = "kbot-model-llm"
    service_port: int = 18092
    max_tokens: int = Field(default=8192, ge=1, le=65536)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=1.0, ge=0, le=1)
    top_k: int = Field(default=0, ge=0, le=100)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)


class VlmConfig(ModelProcessConfig):
    service_name: str = "kbot-model-vlm"
    service_port: int = 18094
    timeout: int = Field(default=600, ge=1, le=3600)


class VisualConfig(ModelProcessConfig):
    service_name: str = "kbot-model-visual"
    service_port: int = 18093


class ModelServingSettings(Settings):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LlmConfig = Field(default_factory=LlmConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    visual: VisualConfig = Field(default_factory=VisualConfig)
    agent_runtime: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18100",
            audience="kbot-agent-runtime-api",
        )
    )
    knowledge_core: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18090",
            audience="kbot-knowledge-core-api",
        )
    )
    data_query: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18140",
            audience="kbot-data-query-api",
        )
    )


@lru_cache(maxsize=1)
def get_model_serving_settings() -> ModelServingSettings:
    return load_settings(ModelServingSettings, service="model_serving")


def get_embedding_config() -> EmbeddingConfig:
    return get_model_serving_settings().embedding


def get_llm_config() -> LlmConfig:
    return get_model_serving_settings().llm


def get_vlm_config() -> VlmConfig:
    return get_model_serving_settings().vlm


def get_visual_config() -> VisualConfig:
    return get_model_serving_settings().visual
