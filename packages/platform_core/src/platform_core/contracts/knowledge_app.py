"""业务知识库公开输入合同；浏览器不能指定内部 Scope、模型、表或检索 Profile。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class KnowledgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


class KnowledgeLibraryCreate(KnowledgePayload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    profile_ref: str = Field(default="default", min_length=1, max_length=128)


class KnowledgeLibraryUpdate(KnowledgePayload):
    expected_revision: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    status: Literal["ACTIVE", "DISABLED"] | None = None
    grants: dict[str, list[Literal["READ", "UPLOAD", "REVIEW", "MANAGE", "PUBLISH", "WITHDRAW"]]] | None = None

    @model_validator(mode="after")
    def validate_changes(self):
        for field in ("display_name", "status", "grants"):
            if field in self.model_fields_set and getattr(self, field) is None:
                raise ValueError("资料库名称、状态和成员不能显式置空")
        if self.display_name is not None and not self.display_name.strip():
            raise ValueError("资料库名称不能为空白")
        return self


class KnowledgeQueryImage(KnowledgePayload):
    mime_type: Literal["image/png", "image/jpeg", "image/webp"]
    image_base64: str = Field(min_length=1, max_length=12000000)


class KnowledgePredicate(KnowledgePayload):
    field: Literal["kind", "title", "created_at", "document_version_ref", "business_asset_ref"]
    operator: Literal["EQ", "NE", "PREFIX", "LT", "LE", "GT", "GE"]
    value: str | int | float | bool | None


class KnowledgeOrder(KnowledgePayload):
    field: Literal["title", "created_at", "stable_key"]
    descending: bool = False


class KnowledgeSearchRequest(KnowledgePayload):
    text: str | None = Field(default=None, min_length=1, max_length=32000)
    images: list[KnowledgeQueryImage] = Field(default_factory=list, max_length=4)
    requested_count: int = Field(default=10, ge=1, le=100)
    output_types: list[Literal["TEXT", "FIGURE", "PHOTO", "TABLE", "PAGE", "REGION", "MIXED"]] = Field(
        default_factory=lambda: ["TEXT", "FIGURE", "PHOTO", "TABLE", "PAGE"])
    combination: Literal["ANY", "ALL", "COMPOSED"] | None = None
    mode: Literal["RELEVANCE", "FILTERED_ORDER", "EXHAUSTIVE"] = "RELEVANCE"
    predicates: list[KnowledgePredicate] = Field(default_factory=list, max_length=20)
    order_by: list[KnowledgeOrder] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def validate_modes(self):
        if self.mode == "RELEVANCE" and (not (self.text or self.images) or self.order_by):
            raise ValueError("相关性检索必须有文本或图片且不能混入属性排序")
        if self.mode != "RELEVANCE" and (self.text or self.images or not self.order_by):
            raise ValueError("完整范围排序只能使用结构化条件和显式排序")
        if len(self.images) > 1 and self.combination is None:
            raise ValueError("多图查询必须指定组合语义")
        if len(self.images) <= 1 and self.combination is not None:
            raise ValueError("单图不能指定多图组合")
        if len(set(self.output_types)) != len(self.output_types) or not self.output_types:
            raise ValueError("结果类型必须非空且不能重复")
        if sum(len(image.image_base64) for image in self.images) > 12000000:
            raise ValueError("查询图片总大小超过预算")
        return self


class KnowledgeReviewRequest(KnowledgePayload):
    decision: Literal["APPROVE", "REJECT"]
    comment: str | None = Field(default=None, max_length=2000)


class KnowledgeContinuationRequest(KnowledgePayload):
    cursor: str = Field(min_length=1, max_length=8192)


class KnowledgeRelationsRequest(KnowledgeContinuationRequest):
    anchor_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeSearchCapability(KnowledgePayload):
    kind: str
    direction: str | None
    required: bool


class KnowledgeLibraryView(KnowledgePayload):
    library_id: str
    display_name: str
    description: str | None
    status: Literal["ACTIVE", "DISABLED"]
    row_version: int
    permissions: list[str]
    configuration: dict[str, str]
    grants: dict[str, list[str]] | None
    search_capabilities: list[KnowledgeSearchCapability]


class KnowledgeVersionView(KnowledgePayload):
    review: str
    uploaded_by: str
    name: str
    reviewed_by: str | None = None
    comment: str | None = None
    status: str
    job_id: str
    processing_status: str
    error_code: str | None
    generation_ready: bool


class KnowledgeDocumentView(KnowledgePayload):
    document_id: str
    title: str
    status: str
    row_version: int
    versions: dict[str, KnowledgeVersionView]


class KnowledgeDocumentPage(KnowledgePayload):
    items: list[KnowledgeDocumentView]
    page: int
    has_more: bool


class KnowledgeUploadReceipt(KnowledgePayload):
    document_id: str
    version_id: str
    job_id: str
    sha256: str
    name: str
    mime_type: str
    status: str


class KnowledgeActionReceipt(KnowledgePayload):
    status: str


class KnowledgeLocator(KnowledgePayload):
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    logical_ref: str | None = None


class KnowledgeContentRef(KnowledgePayload):
    retrieval_ref: str
    key: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeEvidenceView(KnowledgePayload):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    mime_type: str
    byte_count: int
    locators: list[KnowledgeLocator]
    text: str | None
    content_ref: KnowledgeContentRef | None = None


class KnowledgePrimaryHit(KnowledgePayload):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    rank: int


class KnowledgeRelationView(KnowledgePayload):
    relation_type: str
    source_key: str
    target_key: str


class KnowledgeLinkedGroup(KnowledgePayload):
    anchor_key: str
    linked_keys: list[str]
    manifest_status: str
    continuation: str | None
    incomplete_reason: str | None
    relations: list[KnowledgeRelationView]


class KnowledgeChannelView(KnowledgePayload):
    channel_key: str
    required: bool
    status: str
    reason: str | None
    approximate: bool
    exhausted: bool


class KnowledgeSearchResult(KnowledgePayload):
    retrieval_ref: str
    status: str
    reason: str | None
    mode: str
    continuation: str | None
    primary_hits: list[KnowledgePrimaryHit]
    items: list[KnowledgeEvidenceView]
    linked: list[KnowledgeLinkedGroup]
    channels: list[KnowledgeChannelView]
    model_delivery: str
