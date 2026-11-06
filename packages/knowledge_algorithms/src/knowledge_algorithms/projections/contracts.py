"""允许视图、精确源映射、单/多向量空间与模型端口。"""

from dataclasses import asdict, dataclass
import math
from typing import AsyncIterator, Protocol

from ..contracts.content import ContentKind, Provenance
from ..contracts.errors import require
from ..contracts.identity import AppBackendBinding, EvidenceKey, Locator, fingerprint, nonempty
from ..retrieval.maxsim import TokenMatrix


@dataclass(frozen=True, slots=True)
class UnitSource:
    unit_start: int
    unit_end: int
    source_key: EvidenceKey
    source_start: int
    source_end: int

    def __post_init__(self) -> None:
        require(all(type(v) is int for v in (self.unit_start, self.unit_end, self.source_start, self.source_end))
                and 0 <= self.unit_start < self.unit_end and 0 <= self.source_start < self.source_end
                and self.unit_end - self.unit_start == self.source_end - self.source_start,
                "直接来源映射必须保留等长字符跨度")
        require(self.source_key.projection_ref is None, "来源映射必须指向规范解析对象")


@dataclass(frozen=True, slots=True)
class AllowedContent:
    key: EvidenceKey
    kind: ContentKind
    text: str
    sources: tuple[UnitSource, ...]
    locators: tuple[Locator, ...]
    provenance: Provenance
    heading_path: tuple[str, ...] = ()
    content_handle: str | None = None
    table_header: tuple[int, int] | None = None
    table_rows: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        require(self.key.projection_ref is None and bool(self.locators), "允许视图身份或定位无效")
        require(all(s.unit_end <= len(self.text) and s.source_key.owner_binding_ref == self.key.owner_binding_ref
                    and s.source_key.document_version_ref == self.key.document_version_ref
                    and s.source_key.parse_run_ref == self.key.parse_run_ref and s.source_key.source_span is None
                    for s in self.sources), "允许视图来源映射越界")
        require(all(a.unit_end <= b.unit_start for a, b in zip(self.sources, self.sources[1:])),
                "允许视图来源映射必须按位置有序且不重叠")
        if self.text:
            require(bool(self.sources), "允许文本必须带原文映射，生成上下文另存")
        if self.table_rows or self.table_header is not None:
            require(self.kind == ContentKind.TABLE and self.table_header is not None, "表格行组缺少表头")
            spans = (self.table_header,) + self.table_rows
            require(all(0 <= start < end <= len(self.text) for start, end in spans), "表格跨度无效")
            require(all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), "表格头/行跨度不能重叠或乱序")


class AllowedDocumentReader(Protocol):
    binding: AppBackendBinding
    document_version_ref: str
    parse_run_ref: str
    allowed_view_ref: str
    allowed_view_hash: str

    def contents(self) -> AsyncIterator[AllowedContent]: ...


@dataclass(frozen=True, slots=True)
class RepresentationSpec:
    model_ref: str
    revision: str
    processor_revision: str
    input_template: str
    dimension: int
    kind: str
    input_kind: str
    similarity: str
    normalization: str
    directions: frozenset[str]

    def __post_init__(self) -> None:
        nonempty(self.model_ref, self.revision, self.processor_revision, self.input_template)
        require(type(self.dimension) is int and self.dimension > 0, "表示维度无效")
        require(self.kind in {"DENSE", "MULTIVECTOR"} and self.input_kind in {"TEXT", "IMAGE"}, "表示类型无效")
        require(self.similarity in {"DOT", "COSINE"} and self.normalization in {"NONE", "L2"}, "评分空间无效")
        require(bool(self.directions), "模型空间必须声明验证过的方向")
        require(self.directions <= {"TEXT_TO_TEXT", "TEXT_TO_IMAGE", "IMAGE_TO_IMAGE", "IMAGE_TO_TEXT"}
                and all(d.endswith("_TO_" + self.input_kind) for d in self.directions),
                "检索方向必须指向当前索引表示模态")

    @property
    def space_id(self) -> str:
        value = asdict(self)
        value["directions"] = sorted(self.directions)
        return fingerprint(value)


@dataclass(frozen=True, slots=True)
class ModelInput:
    kind: str
    text: str | None = None
    image_handle: str | None = None
    input_role: str = "QUERY"

    def __post_init__(self) -> None:
        require(self.input_role in {"QUERY", "DOCUMENT"}, "模型表示输入角色无效")
        require(self.kind in {"TEXT", "IMAGE"}, "模型输入类型无效")
        require((self.text is not None) != (self.image_handle is not None), "模型输入必须唯一")
        require((self.kind == "TEXT") == (self.text is not None), "模型输入与模态不一致")
        nonempty(self.text if self.text is not None else self.image_handle)


@dataclass(frozen=True, slots=True)
class Representation:
    space_id: str
    vector: tuple[float, ...] | None = None
    matrix: TokenMatrix | None = None

    def validate(self, spec: RepresentationSpec, max_elements: int) -> None:
        require(self.space_id == spec.space_id and (self.vector is None) != (self.matrix is None), "模型输出空间或形状错误")
        if spec.kind == "DENSE":
            require(self.vector is not None and len(self.vector) == spec.dimension
                    and len(self.vector) <= max_elements and all(math.isfinite(v) for v in self.vector), "单向量形状或数值无效")
            vectors = (self.vector,)
        else:
            require(self.matrix is not None and self.matrix.space_id == spec.space_id
                    and len(self.matrix.vectors[0]) == spec.dimension
                    and len(self.matrix.vectors) * spec.dimension <= max_elements, "多向量形状、空间或预算错误")
            vectors = tuple(v for v, valid in zip(self.matrix.vectors, self.matrix.mask) if valid)
        if spec.normalization == "L2":
            require(all(math.isclose(math.hypot(*v), 1.0, abs_tol=1e-5) for v in vectors), "模型输出不满足已声明归一化")
        if spec.similarity == "COSINE":
            require(all(math.hypot(*v) > 0 for v in vectors), "余弦表示不能为零向量")


class ModelGateway(Protocol):
    async def represent(self, value: ModelInput, spec: RepresentationSpec) -> Representation: ...


class Tokenizer(Protocol):
    revision: str

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class ProjectionProfile:
    profile_version: str
    max_tokens: int
    max_input_chars: int
    max_units: int
    batch_size: int
    max_representation_elements: int
    timeout_seconds: float
    representations: tuple[RepresentationSpec, ...]

    def __post_init__(self) -> None:
        nonempty(self.profile_version)
        require(all(type(v) is int and v > 0 for v in (self.max_tokens, self.max_input_chars,
                    self.max_units, self.batch_size, self.max_representation_elements)), "投影预算无效")
        require(math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0, "投影超时无效")


@dataclass(frozen=True, slots=True)
class SearchUnit:
    key: EvidenceKey
    source_key: EvidenceKey
    unit_key: str
    kind: ContentKind
    text: str
    context: str
    sources: tuple[UnitSource, ...]
    locators: tuple[Locator, ...]
    provenance: Provenance
    allowed_view_ref: str
    allowed_view_hash: str
    content_handle: str | None


@dataclass(frozen=True, slots=True)
class ProjectedUnit:
    unit: SearchUnit
    representations: tuple[Representation, ...]


@dataclass(frozen=True, slots=True)
class ProjectionBatch:
    projection_ref: str
    sequence: int
    units: tuple[ProjectedUnit, ...]
    payload_hash: str
