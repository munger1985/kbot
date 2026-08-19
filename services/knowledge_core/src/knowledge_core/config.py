"""Knowledge Core API、Parser 与 Projection Worker 配置。"""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, model_validator

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
    local_object_storage_path: str = "./var/data/knowledge_core"


class ParserWorkerConfig(ServiceConfig):
    service_name: str = "kbot-knowledge-core-parser"
    service_port: int = 18095
    worker_id: str = Field(default="kc-parser-local", min_length=1, max_length=256)
    local_artifacts_path: str = "./cached_models/docling"
    lease_seconds: int = Field(default=600, ge=30, le=3600)
    evidence_batch_size: int = Field(default=100, ge=1, le=500)


class ProjectionWorkerConfig(BaseModel):
    service_name: str = "kbot-knowledge-core-projection"
    worker_id: str = Field(
        default="kc-projection-local", min_length=1, max_length=256
    )
    lease_seconds: int = Field(default=600, ge=30, le=3600)
    index_batch_size: int = Field(default=64, ge=1, le=500)


class JobWakeupConfig(BaseModel):
    """KC Worker 通知唤醒与故障退避配置。"""

    mode: Literal["DBMS_ALERT", "POLLING"] = "DBMS_ALERT"
    notification_timeout_seconds: float = Field(
        default=30.0, ge=1.0, le=300.0
    )
    fallback_min_seconds: float = Field(default=2.0, ge=0.2, le=60.0)
    fallback_max_seconds: float = Field(default=30.0, ge=1.0, le=300.0)
    fallback_multiplier: float = Field(default=2.0, ge=1.1, le=10.0)
    jitter_ratio: float = Field(default=0.15, ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_fallback_range(self):
        if self.fallback_max_seconds < self.fallback_min_seconds:
            raise ValueError(
                "fallback_max_seconds 不能小于 fallback_min_seconds"
            )
        return self


class ParsePolicyConfig(BaseModel):
    ocr_model: str | None = None
    parse_strategy: Literal["TEXT", "AUTO", "VISUAL", "HYBRID"] = "AUTO"
    visual_description_prompt: str = (
        "请客观描述图片中的可见事实、文字、对象及其关系；"
        "不要推测图片之外的信息。"
    )
    full_page_visual_prompt: str = (
        "请将整张文档页面转换为结构化 Markdown。原文照录文字、数字、"
        "单位和符号；准确还原标题层级、表格、列表、公式及图表关系；"
        "不要编造内容，不要输出代码块或额外解释。"
    )
    visual_min_text_characters: int = Field(default=80, ge=0, le=10000)
    visual_min_mean_confidence: float = Field(default=0.65, ge=0, le=1)
    visual_max_gibberish_ratio: float = Field(default=0.08, ge=0, le=1)
    visual_max_concurrency: int = Field(default=2, ge=1, le=16)


class EmbeddingDependencyConfig(ServiceDependencyConfig):
    health_check_timeout_seconds: int = Field(default=10, ge=1, le=120)


class DsocrConfig(BaseModel):
    """DeepSeek OCR 独立推理端点配置，与模型托管服务无关。"""

    enabled: bool = False
    api_endpoint: str = "http://localhost:18097/v1/chat/completions"
    timeout: int = Field(default=600, ge=10, le=3600)
    crop_mode: bool = True
    max_tokens: int = Field(default=8192, ge=512, le=32768)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)


class KnowledgeCoreSettings(Settings):
    api: KnowledgeCoreApiConfig = Field(default_factory=KnowledgeCoreApiConfig)
    storage: KnowledgeStorageConfig = Field(default_factory=KnowledgeStorageConfig)
    parser: ParserWorkerConfig = Field(default_factory=ParserWorkerConfig)
    projection: ProjectionWorkerConfig = Field(
        default_factory=ProjectionWorkerConfig
    )
    job_wakeup: JobWakeupConfig = Field(default_factory=JobWakeupConfig)
    parse_policy: ParsePolicyConfig = Field(default_factory=ParsePolicyConfig)
    dsocr: DsocrConfig = Field(default_factory=DsocrConfig)
    embedding: EmbeddingDependencyConfig = Field(
        default_factory=lambda: EmbeddingDependencyConfig(
            base_url="http://127.0.0.1:18091",
            audience="kbot-model-embedding",
            timeout_seconds=300,
        )
    )
    vlm: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18094",
            audience="kbot-model-vlm",
            timeout_seconds=600,
        )
    )
    visual: ServiceDependencyConfig = Field(
        default_factory=lambda: ServiceDependencyConfig(
            base_url="http://127.0.0.1:18093",
            audience="kbot-model-visual",
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

    @model_validator(mode="after")
    def validate_parser_models(self):
        if self.parse_policy.ocr_model and not self.dsocr.enabled:
            raise ValueError(
                "配置 parse_policy.ocr_model 时必须启用 dsocr.enabled"
            )
        return self


@lru_cache(maxsize=1)
def get_knowledge_core_settings() -> KnowledgeCoreSettings:
    return load_settings(KnowledgeCoreSettings, service="knowledge_core")
