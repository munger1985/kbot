"""从 KC IR 提取的中立内容、来源与位置合同；解析不依赖检索投影。"""

from dataclasses import dataclass
from enum import Enum
import math

from .errors import require
from .identity import EvidenceKey, Locator, nonempty


class ContentKind(str, Enum):
    DOCUMENT = "DOCUMENT"
    SECTION = "SECTION"
    PARAGRAPH = "PARAGRAPH"
    LIST_ITEM = "LIST_ITEM"
    TABLE = "TABLE"
    TABLE_ROW = "TABLE_ROW"
    FIGURE = "FIGURE"
    CAPTION = "CAPTION"
    FORMULA = "FORMULA"
    CODE = "CODE"
    FOOTNOTE = "FOOTNOTE"
    IMAGE = "IMAGE"
    REGION = "REGION"
    PAGE = "PAGE"
    PHOTO = "PHOTO"
    HEADER = "HEADER"
    FOOTER = "FOOTER"


class ProvenanceKind(str, Enum):
    NATIVE = "NATIVE"
    OCR = "OCR"
    MODEL_TRANSCRIPTION = "MODEL_TRANSCRIPTION"
    MODEL_DESCRIPTION = "MODEL_DESCRIPTION"
    APP_DERIVED = "APP_DERIVED"


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: ProvenanceKind
    generator: str
    revision: str
    input_hash: str

    def __post_init__(self) -> None:
        require(isinstance(self.kind, ProvenanceKind), "来源类别无效")
        nonempty(self.generator, self.revision, self.input_hash)


@dataclass(frozen=True, slots=True)
class ContentNode:
    key: EvidenceKey
    kind: ContentKind
    text: str
    locators: tuple[Locator, ...]
    provenance: Provenance
    native_ref: str
    reading_order: int
    heading_level: int | None = None
    parent_key: str | None = None
    numbering_scope: str | None = None
    display_label: str | None = None
    native_anchor: str | None = None

    def __post_init__(self) -> None:
        require(isinstance(self.kind, ContentKind), "内容类型无效")
        require(self.key.projection_ref is None and self.key.source_span is None,
                "规范解析节点不能绑定检索投影或片段身份")
        require(bool(self.locators), "规范节点必须有真实物理或逻辑定位")
        require(type(self.reading_order) is int and self.reading_order >= 0, "阅读顺序无效")
        require(self.heading_level is None or type(self.heading_level) is int
                and 1 <= self.heading_level <= 9, "标题级别无效")
        nonempty(self.native_ref)


@dataclass(frozen=True, slots=True)
class Page:
    local_key: str
    page_no: int
    width: float
    height: float
    rotation: int
    image_handle: str | None = None

    def __post_init__(self) -> None:
        nonempty(self.local_key)
        require(type(self.page_no) is int and self.page_no > 0, "物理页码无效")
        require(all(math.isfinite(v) and v > 0 for v in (self.width, self.height)), "页面几何无效")
        require(self.rotation in {0, 90, 180, 270}, "页面旋转必须规范化")


@dataclass(frozen=True, slots=True)
class VisualAsset:
    key: EvidenceKey
    figure_key: EvidenceKey
    content_handle: str
    sha256: str
    mime_type: str
    byte_count: int
    locator: Locator
    provenance: Provenance
    crop_transform: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        nonempty(self.content_handle, self.sha256, self.mime_type)
        require(type(self.byte_count) is int and self.byte_count > 0, "图像字节数无效")
        require(self.key.projection_ref is None and self.figure_key.projection_ref is None,
                "原图不能绑定尚未构建的检索投影")
        require((self.key.owner_binding_ref, self.key.document_version_ref, self.key.parse_run_ref)
                == (self.figure_key.owner_binding_ref, self.figure_key.document_version_ref,
                    self.figure_key.parse_run_ref), "图像与逻辑图身份不一致")
        if self.crop_transform is not None:
            require(len(self.crop_transform) == 6 and all(math.isfinite(v) for v in self.crop_transform),
                    "裁剪变换必须为有限的仿射矩阵")
