"""业务知识库公开输入合同；浏览器不能指定内部 Scope、模型、表或检索 Profile。"""

from typing import Literal
from uuid import UUID
import re
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .knowledge_parsing import LibraryIndexModels, LibraryParseModels, ParseStrategy, ParseStrategyOption

KNOWLEDGE_INLINE_MEDIA_TYPES = frozenset({
    "text/plain", "text/markdown", "image/png", "image/jpeg", "image/webp", "application/pdf",
})

class KnowledgePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")


def parse_library_revision(value: str) -> int:
    match = re.fullmatch(r'"rv-([1-9][0-9]{0,17})"', value)
    if not match:
        raise ValueError("If-Match 格式应为 \"rv-<version>\"")
    return int(match.group(1))


class KnowledgeLibraryCreate(KnowledgePayload):
    display_name: str = Field(min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    parse_models: LibraryParseModels | None = None
    index_models: LibraryIndexModels | None = None


class KnowledgeLibraryUpdate(KnowledgePayload):
    expected_revision: int = Field(ge=1)
    display_name: str | None = Field(default=None, min_length=1, max_length=256)
    description: str | None = Field(default=None, max_length=1000)
    parse_models: LibraryParseModels | None = None
    index_models: LibraryIndexModels | None = None
    rebuild_indexes: bool = False
    status: Literal["ACTIVE", "DISABLED"] | None = None
    grants: dict[str, list[Literal["READ", "UPLOAD", "REVIEW", "MANAGE", "WITHDRAW"]]] | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_changes(self):
        if self.rebuild_indexes and (self.index_models is None or {"grants", "status"} & self.model_fields_set):
            raise ValueError("保存并重建需要模型配置，不与成员或启停操作合并")
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
    parse_strategy: ParseStrategy | None = None

    @model_validator(mode="after")
    def reject_without_strategy(self):
        if self.decision == "REJECT" and self.parse_strategy is not None:
            raise ValueError("拒绝审核不选择解析方案")
        return self


class KnowledgeRetryIndexRequest(KnowledgePayload):
    expected_job_id: UUID


class KnowledgeReprocessRequest(KnowledgeRetryIndexRequest):
    mode: Literal["ORIGINAL", "CURRENT"]
    parse_strategy: ParseStrategy | None = None

    @model_validator(mode="after")
    def validate_strategy(self):
        if (self.mode == "CURRENT") != (self.parse_strategy is not None):
            raise ValueError("使用当前库模型时必须指定解析方案，按原方案重试时不能修改方案")
        return self


class KnowledgeContinuationRequest(KnowledgePayload):
    cursor: str = Field(min_length=1, max_length=8192)


class KnowledgeRelationsRequest(KnowledgeContinuationRequest):
    anchor_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgeSearchCapability(KnowledgePayload):
    kind: str
    direction: str | None
    required: bool


class KnowledgeMemberOption(KnowledgePayload):
    user_id: str
    display_name: str
    can_manage: bool


class KnowledgeDirectoryQuery(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    view: Literal["files", "processing", "reviews"] = "files"
    title: str = Field(default="", max_length=256)
    state: str = Field(default="", max_length=32)
    mine: bool = False
    version_id: UUID | None = None
    document_id: UUID | None = None


IndexRole = Literal["embedding", "visual_embedding"]


class KnowledgeIndexBatchProgress(KnowledgePayload):
    committed_batches: int = Field(ge=0)
    total_batches: int | None = Field(ge=1)
    validated: bool


class KnowledgeIndexRebuildItem(KnowledgePayload):
    version_id: UUID
    title: str
    status: Literal["PENDING", "RUNNING", "READY", "FAILED", "CANCELLED"]
    error_code: str | None
    visual_not_applicable: bool
    index_progress: KnowledgeIndexBatchProgress | None = None


class KnowledgeIndexRebuild(KnowledgePayload):
    batch_id: UUID
    status: Literal["RUNNING", "READY", "FAILED", "STALE", "PUBLISHED", "DISCARDED"]
    scope: Literal["LIBRARY", "FILE"] = "LIBRARY"
    source_published: bool = True
    roles: list[IndexRole]
    items: list[KnowledgeIndexRebuildItem]


class KnowledgeIndexMaintenance(KnowledgePayload):
    published_file_count: int
    library_revision: int
    pending_roles: list[IndexRole]
    active_file_rebuild_count: int = Field(ge=0)
    batch: KnowledgeIndexRebuild | None


class KnowledgeIndexRebuildRequest(KnowledgePayload):
    action: Literal["START", "RETRY", "DISCARD"]
    expected_revision: int | None = Field(default=None, ge=1)
    batch_id: UUID | None = None
    version_id: UUID | None = None
    expected_generation_id: UUID | None = None

    @model_validator(mode="after")
    def validate_action(self):
        if ((self.action == "START" and self.version_id is None) != (self.expected_revision is not None)
            or (self.action != "START") != (self.batch_id is not None)
            or (self.action == "RETRY" and self.version_id is None)
            or (self.action not in {"START", "RETRY"} and self.version_id is not None)
            or (self.expected_generation_id is not None) != (self.action == "START" and self.version_id is not None)):
            raise ValueError("索引任务操作参数不完整或不匹配")
        return self


class KnowledgeLibraryView(KnowledgePayload):
    library_id: str
    display_name: str
    description: str | None
    status: Literal["ACTIVE", "DISABLED"]
    row_version: int
    updated_at: str
    permissions: list[str]
    configuration: dict[str, str]
    parse_models: LibraryParseModels | None = None
    index_models: LibraryIndexModels | None = None
    grants: dict[str, list[str]] | None
    search_capabilities: list[KnowledgeSearchCapability]
    upload_max_bytes: int
    upload_formats: dict[str, str]
    ocr_enabled: bool
    delete_block: Literal["HAS_DOCUMENTS", "AGENT_REFERENCE"] | None = None
    index_maintenance: KnowledgeIndexMaintenance | None = None


class KnowledgeParseTaskHistory(KnowledgePayload):
    job_id: UUID
    status: str
    error_code: str | None
    created_at: str


class KnowledgeVersionView(KnowledgePayload):
    review: str
    uploaded_by: str
    name: str
    reviewed_by: str | None = None
    comment: str | None = None
    parse_strategy: ParseStrategy | None = None
    parse_strategy_options: list[ParseStrategyOption] | None = None
    original_parse_available: bool | None = None
    parse_task_history: list[KnowledgeParseTaskHistory] | None = None
    status: str
    job_id: str
    processing_status: str
    progress_stage: Literal[
        "AWAITING_APPROVAL", "PARSE_QUEUED", "PREPARING", "EXTRACTING", "ORGANIZING",
        "PARSE_RETRY", "INDEX_QUEUED", "INDEXING", "INDEX_RETRY", "INDEX_CONFIRMING",
        "READY", "PARSE_FAILED", "INDEX_FAILED", "CANCELLED", "REJECTED",
        "REBUILD_QUEUED", "REBUILDING", "REBUILD_RETRY", "REBUILD_CONFIRMING", "REBUILD_READY", "REBUILD_FAILED",
        "REBUILD_COMPLETE",
    ] | None = None
    index_progress: KnowledgeIndexBatchProgress | None = None
    parse_status: str
    error_code: str | None
    generation_ready: bool
    generation_id: str | None
    created_at: str
    mime_type: str
    byte_count: int
    allowed_actions: list[str]


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


class KnowledgeMaintenanceReceipt(KnowledgeActionReceipt):
    job_id: str


class KnowledgeLocator(KnowledgePayload):
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    logical_ref: str | None = None


class KnowledgeContentRef(KnowledgePayload):
    retrieval_ref: str
    key: str = Field(pattern=r"^[0-9a-f]{64}$")


class KnowledgePassageDisplay(KnowledgePayload):
    order: int
    kind: str
    title: str | None
    text: str | None
    pages: list[int]
    chapter_key: str | None = None


class KnowledgeEvidenceView(KnowledgePayload):
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: str
    mime_type: str
    byte_count: int
    locators: list[KnowledgeLocator]
    text: str | None
    content_ref: KnowledgeContentRef | None = None
    version_id: str
    source_title: str
    document_id: str
    passage_key: str | None
    chapter: str | None
    passage_display: KnowledgePassageDisplay | None = None


class KnowledgePassageSegment(KnowledgePayload):
    text: str
    matched: bool


class KnowledgePassageBlock(KnowledgePayload):
    key: str
    kind: str
    anchor: bool
    locators: list[KnowledgeLocator]
    segments: list[KnowledgePassageSegment]


class KnowledgePassageContext(KnowledgePayload):
    blocks: list[KnowledgePassageBlock]
    has_before: bool
    has_after: bool
    pdf_page: int | None


class KnowledgePassageQuery(KnowledgePayload):
    before: int = Field(default=2, ge=0, le=10)
    after: int = Field(default=2, ge=0, le=10)


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
    scope_exhausted: bool
