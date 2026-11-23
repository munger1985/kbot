"""解析输入、区域提取和有界局部修复合同。"""

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
import math
import re
from typing import AsyncIterator, Protocol

from ..contracts.content import ContentNode, Page, VisualAsset
from ..contracts.errors import require
from ..contracts.identity import fingerprint, nonempty
from .events import ParseRunIdentity, validate_digest


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    owner_binding_ref: str
    document_version_ref: str
    sha256: str
    mime_type: str
    byte_count: int
    source_handle: str

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.document_version_ref, self.mime_type, self.source_handle)
        validate_digest(self.sha256)
        require(type(self.byte_count) is int and self.byte_count > 0, "输入大小无效")
        require(not any(v in self.source_handle for v in ("/", "\\", "://")), "源句柄不能是路径或 URL")


class SourceReader(Protocol):
    descriptor: SourceDescriptor

    def chunks(self) -> AsyncIterator[bytes]: ...


@dataclass(frozen=True, slots=True)
class LayoutProfile:
    full_width_ratio: float
    column_gap: float

    def __post_init__(self) -> None:
        require(all(math.isfinite(v) and 0 < v <= 1 for v in (
            self.full_width_ratio, self.column_gap)), "版面参数必须显式指定且归一化")


@dataclass(frozen=True, slots=True)
class ParseModelBinding:
    """本次审批为解析职责冻结的目录身份，不含地址或凭据。"""

    model_id: str
    revision: str
    inference_ref: str

    def __post_init__(self) -> None:
        require(re.fullmatch(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", self.model_id) is not None, "解析模型目录身份无效")
        nonempty(self.revision, self.inference_ref)
        require(len(self.revision) <= 256 and len(self.inference_ref) <= 256, "解析模型身份超过预算")


class ParseStrategy(str, Enum):
    TEXT = "TEXT"
    AUTO = "AUTO"
    VISUAL = "VISUAL"


@dataclass(frozen=True, slots=True)
class ParseProfile:
    profile_version: str
    algorithm_version: str
    languages: tuple[str, ...]
    max_input_bytes: int
    max_pages: int
    max_objects: int
    max_text_bytes: int
    max_event_bytes: int
    batch_size: int
    timeout_seconds: float
    max_repair_attempts: int
    repair_strategies: tuple[str, ...]
    max_bad_char_ratio: float
    min_region_coverage: float
    allow_partial: bool
    layout: LayoutProfile
    strategy: str | None = field(default=None, metadata={"omit_none": True})
    ocr_model: ParseModelBinding | None = field(default=None, metadata={"omit_none": True})
    vlm_model: ParseModelBinding | None = field(default=None, metadata={"omit_none": True})

    def __post_init__(self) -> None:
        nonempty(self.profile_version, self.algorithm_version, *self.languages)
        require(all(type(v) is int and v > 0 for v in (
            self.max_input_bytes, self.max_pages, self.max_objects, self.max_text_bytes,
            self.max_event_bytes, self.batch_size)), "解析资源预算无效")
        require(type(self.max_repair_attempts) is int and self.max_repair_attempts >= 0, "修复次数无效")
        require(set(self.repair_strategies) <= {"OCR", "MODEL_TRANSCRIPTION"}, "修复不能使用模型描述代替原文")
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in (
            self.max_bad_char_ratio, self.min_region_coverage)), "质量参数无效")
        require(math.isfinite(self.timeout_seconds) and self.timeout_seconds > 0, "解析超时无效")
        require(type(self.allow_partial) is bool, "局部失败策略无效")
        if self.strategy is None or self.strategy == ParseStrategy.TEXT:
            require(self.ocr_model is None and self.vlm_model is None, "原生解析不得冻结模型")
            if self.strategy == ParseStrategy.TEXT:
                require(not self.repair_strategies and self.max_repair_attempts == 0, "原生解析不得执行模型转写")
        else:
            require(self.ocr_model is not None, "自动判断与视觉增强必须冻结 OCR 模型")
            require(self.strategy != ParseStrategy.VISUAL or self.vlm_model is not None,
                    "视觉增强必须冻结 VLM 模型")
        if self.strategy is not None:
            require(self.strategy in {value.value for value in ParseStrategy}, "解析档位无效")

    @property
    def profile_hash(self) -> str:
        value = asdict(self)
        if self.strategy is None:
            # 已冻结任务的摘要不随新审批字段扩展而变化。
            for name in ("strategy", "ocr_model", "vlm_model"):
                value.pop(name)
        return fingerprint(value)

    def for_approval(self, strategy: ParseStrategy, *, ocr: ParseModelBinding | None = None,
                     vlm: ParseModelBinding | None = None) -> "ParseProfile":
        require(isinstance(strategy, ParseStrategy), "审批必须明确选择解析档位")
        native = strategy is ParseStrategy.TEXT
        return replace(self, strategy=strategy.value, ocr_model=None if native else ocr,
                       vlm_model=None if native else vlm,
                       max_repair_attempts=0 if native else 1,
                       repair_strategies=() if native else ("OCR",),
                       allow_partial=True)


@dataclass(frozen=True, slots=True)
class ExtractedRegion:
    region_key: str
    nodes: tuple[ContentNode, ...]
    text_coverage: float

    def __post_init__(self) -> None:
        nonempty(self.region_key)
        require(bool(self.nodes), "提取区域不能没有对象")
        require(math.isfinite(self.text_coverage) and 0 <= self.text_coverage <= 1, "区域覆盖指标无效")
        require(len({n.key for n in self.nodes}) == len(self.nodes), "区域对象重复")


@dataclass(frozen=True, slots=True)
class ExtractionBatch:
    pages: tuple[Page, ...]
    regions: tuple[ExtractedRegion, ...]
    assets: tuple[VisualAsset, ...]


class ParsingAdapter(Protocol):
    mime_types: frozenset[str]
    revision: str

    def extract(self, source: SourceReader, profile: ParseProfile,
                run: ParseRunIdentity) -> AsyncIterator[ExtractionBatch]: ...


class RepairAdapter(Protocol):
    async def repair(self, region: ExtractedRegion, *, strategy: str,
                     profile: ParseProfile, run: ParseRunIdentity) -> ExtractedRegion: ...
