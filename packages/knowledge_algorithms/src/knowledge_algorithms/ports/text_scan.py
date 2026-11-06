"""可选精确文本扫描端口；关闭时算法核不加载进程执行器。"""

from dataclasses import dataclass
from typing import Protocol

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, SourceSpan, fingerprint
from ..projections.contracts import UnitSource


@dataclass(frozen=True, slots=True)
class TextScanRequest:
    mode: str
    patterns: tuple[str, ...]
    combination: str
    candidate_keys: tuple[EvidenceKey, ...]
    max_sources: int
    max_bytes: int
    max_matches: int
    max_output_bytes: int
    timeout_seconds: float
    case_sensitive: bool = True

    def __post_init__(self):
        require(self.mode in {"EXACT_RECALL", "WITHIN_CANDIDATES"}
                and self.combination in {"ANY", "ALL"}, "文本扫描语义无效")
        require(0 < len(self.patterns) <= 32 and len(set(self.patterns)) == len(self.patterns)
                and all(type(p) is str and 0 < len(p) <= 512 and not any(c in p for c in "\n\r\0")
                        for p in self.patterns), "字面量扫描模式无效")
        require(all(type(n) is int and n > 0 for n in (self.max_sources, self.max_bytes,
                self.max_matches, self.max_output_bytes)) and 0 < self.timeout_seconds <= 60, "扫描预算无效")
        require(type(self.case_sensitive) is bool and (self.mode != "EXACT_RECALL" or not self.candidate_keys),
                "全范围扫描不能隐式缩成向量候选")


@dataclass(frozen=True, slots=True)
class AuthorizedTextSource:
    key: EvidenceKey
    text: str
    sources: tuple[UnitSource, ...]
    text_hash: str
    map_hash: str

    def validate(self):
        from hashlib import sha256
        from ..contracts.wire import encode
        require(self.text_hash == sha256(self.text.encode()).hexdigest()
                and self.map_hash == fingerprint(encode(self.sources)), "扫描文本或源映射摘要不一致")
        require(all(source.unit_end <= len(self.text)
                and source.source_key.owner_binding_ref == self.key.owner_binding_ref
                and source.source_key.document_version_ref == self.key.document_version_ref
                and source.source_key.parse_run_ref == self.key.parse_run_ref for source in self.sources),
                "扫描来源映射越界")


@dataclass(frozen=True, slots=True)
class TextMatch:
    key: EvidenceKey
    pattern_index: int
    byte_start: int
    byte_end: int
    source_key: EvidenceKey
    source_span: SourceSpan


@dataclass(frozen=True, slots=True)
class TextScanBatch:
    matches: tuple[TextMatch, ...]
    status: str
    exhaustive: bool
    scanned_keys: tuple[EvidenceKey, ...]
    reason: str | None
    mode: str
    policy_revision: str = "rg-literal-source-map/v1"

    def __post_init__(self):
        require(self.status in {"COMPLETE", "PARTIAL", "FAILED"}
                and self.exhaustive == (self.status == "COMPLETE")
                and (self.reason is None) == (self.status == "COMPLETE")
                and self.mode in {"EXACT_RECALL", "WITHIN_CANDIDATES"}, "扫描结果状态不一致")
        require(len(set(self.scanned_keys)) == len(self.scanned_keys), "扫描来源身份重复")


class TextScanPort(Protocol):
    async def scan(self, request, scope, snapshot) -> TextScanBatch: ...


def lexical_family_order(fulltext_keys, matches, *, pattern_count, combination):
    """只产生一份词法名次，不把扫描作为第二次 RRF 投票。"""
    require(combination in {"ANY", "ALL"} and pattern_count > 0, "词法族精确语义无效")
    coverage = {}
    for match in matches:
        require(0 <= match.pattern_index < pattern_count, "扫描模式编号越界")
        coverage.setdefault(match.key, set()).add(match.pattern_index)
    rank = {key: index for index, key in enumerate(fulltext_keys)}
    keys = tuple(dict.fromkeys(tuple(fulltext_keys) + tuple(coverage)))
    def key(value):
        count = len(coverage.get(value, ()))
        complete = count == pattern_count if combination == "ALL" else count > 0
        return -int(complete), -count, rank.get(value, float("inf")), value.stable_key
    return tuple(sorted(keys, key=key))
