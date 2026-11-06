"""可选 rg 的显式组合配置；关闭时不导入、定位或探测 rg。"""

from dataclasses import dataclass
from knowledge_algorithms.contracts.errors import require
from knowledge_algorithms.ports.text_scan import TextScanRequest
from ..domain.knowledge_ingestion import BINDING
from ..adapters.knowledge_text_scan import PublishedTextSourceProvider


@dataclass(frozen=True, slots=True)
class TextScanOptions:
    enabled: bool = False
    executable: str | None = None
    temporary_root: str | None = None
    mode: str = "EXACT_RECALL"
    pattern_combination: str = "ANY"
    max_sources: int = 20000
    max_bytes: int = 16000000
    max_matches: int = 10000
    max_output_bytes: int = 2000000
    timeout_seconds: float = 2.0

    def factory(self, pattern_factory, candidate_factory=None):
        if not self.enabled:
            return None
        require(self.executable and self.executable.startswith("/") and self.temporary_root
                and self.temporary_root.startswith("/"), "启用扫描必须配置受控绝对路径")
        # 调用方只提供可信字面量规划器，不接受浏览器路径或 shell 参数。
        def create(backend):
            from knowledge_text_scan.ripgrep import RipgrepTextScanner
            scanner = RipgrepTextScanner(binding=BINDING, provider=PublishedTextSourceProvider(backend),
                executable=self.executable, temporary_root=self.temporary_root)
            def policy(query):
                patterns = tuple(pattern_factory(query))
                if not patterns:
                    return None
                require(self.mode != "WITHIN_CANDIDATES" or candidate_factory is not None,
                        "候选内扫描必须显式提供当前候选规划器")
                candidates = tuple(candidate_factory(backend, query)) if self.mode == "WITHIN_CANDIDATES" else ()
                return TextScanRequest(self.mode, patterns, self.pattern_combination, candidates,
                    self.max_sources, self.max_bytes, self.max_matches, self.max_output_bytes, self.timeout_seconds)
            return scanner, policy
        return create
