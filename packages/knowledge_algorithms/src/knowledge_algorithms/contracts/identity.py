"""精确证据身份、授权范围和不可混用的索引快照。"""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math

from .errors import AlgorithmError, ErrorCode, require


def canonical_json(value: object) -> str:
    """从既有解析合同提取规范 JSON 规则，同时拒绝非有限数字。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


def nonempty(*values: str) -> None:
    require(all(isinstance(value, str) and bool(value.strip()) for value in values),
            "身份及版本字段不能为空")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    node_key: str
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        nonempty(self.node_key)
        require(type(self.char_start) is int and type(self.char_end) is int
                and 0 <= self.char_start < self.char_end, "原文字符跨度无效")


@dataclass(frozen=True, slots=True)
class EvidenceKey:
    owner_binding_ref: str
    document_version_ref: str
    parse_run_ref: str
    object_key: str
    projection_ref: str | None = None
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.document_version_ref,
                 self.parse_run_ref, self.object_key)
        if self.projection_ref is not None:
            nonempty(self.projection_ref)
        if self.source_span is not None:
            require(isinstance(self.source_span, SourceSpan)
                    and self.source_span.node_key == self.object_key,
                    "证据跨度必须属于其精确来源对象")

    @property
    def stable_key(self) -> str:
        return fingerprint(asdict(self))


@dataclass(frozen=True, slots=True)
class Locator:
    """原生逻辑定位或真实页码；不补造缺失的几何信息。"""

    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    logical_ref: str | None = None

    def __post_init__(self) -> None:
        if self.logical_ref is not None:
            nonempty(self.logical_ref)
            require(self.page_no is None and self.bbox is None, "逻辑与物理定位不能混用")
            return
        require(type(self.page_no) is int and self.page_no > 0, "页码必须是真实正整数")
        if self.bbox is not None:
            require(len(self.bbox) == 4 and all(math.isfinite(v) for v in self.bbox),
                    "坐标必须为四个有限数")
            x0, y0, x1, y1 = self.bbox
            require(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "坐标必须归一化且非空")


@dataclass(frozen=True, slots=True)
class AppBackendBinding:
    """由组合根登记；此对象不是用户可选择的路由参数。"""

    binding_ref: str
    namespace: str

    def __post_init__(self) -> None:
        nonempty(self.binding_ref, self.namespace)

    def check_owner(self, owner_binding_ref: str) -> None:
        if owner_binding_ref != self.binding_ref:
            raise AlgorithmError(ErrorCode.SCOPE_CLOSURE_BROKEN,
                                 "对象不属于当前固定后端", stage="scope")


@dataclass(frozen=True, slots=True)
class AuthorizedScope:
    owner_binding_ref: str
    scope_ref: str
    scope_hash: str
    authorization_revision: str

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.scope_ref, self.scope_hash,
                 self.authorization_revision)


@dataclass(frozen=True, slots=True)
class RetrievalSnapshot:
    owner_binding_ref: str
    scope_hash: str
    snapshot_ref: str
    index_generation: str
    withdrawal_revision: str

    def __post_init__(self) -> None:
        nonempty(self.owner_binding_ref, self.scope_hash, self.snapshot_ref,
                 self.index_generation, self.withdrawal_revision)

    def validate_scope(self, binding: AppBackendBinding, scope: AuthorizedScope) -> None:
        binding.check_owner(scope.owner_binding_ref)
        binding.check_owner(self.owner_binding_ref)
        if self.scope_hash != scope.scope_hash:
            raise AlgorithmError(ErrorCode.SNAPSHOT_STALE,
                                 "快照与授权范围不一致", stage="snapshot")
