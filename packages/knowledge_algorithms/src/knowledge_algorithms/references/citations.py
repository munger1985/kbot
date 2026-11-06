"""仅按精确标识解析跨文档引用，不按题名相似度猜测连接。"""

from dataclasses import dataclass
import re
from urllib.parse import unquote

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, SourceSpan, fingerprint
from .relations import RelationType, ResolvedRelation


@dataclass(frozen=True, slots=True)
class CitationTarget:
    key: EvidenceKey
    identifiers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CitationResolution:
    occurrence_key: str
    source_key: EvidenceKey
    span: SourceSpan
    identifier: str
    status: str
    target_key: EvidenceKey | None


def canonical_identifier(value):
    require(type(value) is str and 0 < len(value) <= 2048, "文献标识无效")
    normalized = unquote(value.strip())
    normalized = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", normalized, flags=re.I)
    if re.fullmatch(r"10\.[0-9]{4,9}/[^\s<>]+", normalized, re.I):
        return "doi:" + normalized.casefold()
    if normalized.startswith("urn:"):
        return normalized
    raise ValueError("只接受 DOI 或显式规范 URN")


def resolve_citations(text, source_key, targets, *, policy_revision):
    require(len({target.key for target in targets}) == len(targets), "文献目录身份重复")
    require(all(target.key.owner_binding_ref == source_key.owner_binding_ref for target in targets),
            "文献目录不能跨 App")
    catalog = {}
    for target in targets:
        for value in target.identifiers:
            catalog.setdefault(canonical_identifier(value), set()).add(target.key)
    # 只识别正文明确给出的 DOI；不把 [1]、相同标题或相邻文献猜成精确关系。
    matches = re.finditer(r"(?i)(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.[0-9]{4,9}/[^\s<>]+", text)
    result = []
    for match in matches:
        raw = match.group().rstrip(".,;，。；")
        identifier = canonical_identifier(raw)
        span = SourceSpan(source_key.object_key, match.start(), match.start() + len(raw))
        candidates = {key for key in catalog.get(identifier, ())
                      if key.document_version_ref != source_key.document_version_ref}
        status = "RESOLVED" if len(candidates) == 1 else "AMBIGUOUS" if candidates else "UNRESOLVED"
        result.append(CitationResolution(fingerprint([source_key.stable_key, span.char_start, span.char_end,
            policy_revision]), source_key, span, identifier, status, next(iter(candidates)) if len(candidates) == 1 else None))
    return tuple(result)


def citation_edges(resolutions):
    return tuple(ResolvedRelation(fingerprint([entry.occurrence_key, entry.target_key.stable_key]),
        RelationType.CITES_DOCUMENT, entry.source_key, entry.target_key, "EXACT_IDENTIFIER",
        entry.occurrence_key, entry.span) for entry in resolutions if entry.status == "RESOLVED")


@dataclass(frozen=True, slots=True)
class BibliographyEntry:
    label: str
    identifiers: tuple[str, ...]

    def __post_init__(self):
        require(bool(re.fullmatch(r"[1-9][0-9]{0,5}", self.label)) and bool(self.identifiers),
                "文献条目必须有明确数字标签与规范标识")


def bibliography_entries(contents):
    """只从明确参考文献章节的编号条目提取 DOI，不按标题相似度匹配。"""
    entries = []
    for content in contents:
        if not any(heading.strip().casefold() in {"references", "bibliography", "参考文献", "参考资料"}
                   for heading in content.heading_path):
            continue
        for match in re.finditer(r"(?m)^\s*\[([1-9][0-9]{0,5})\]\s*(.+)$", content.text):
            identifiers = tuple(canonical_identifier(value.group().rstrip(".,;，。；")) for value in
                re.finditer(r"(?i)(?:https?://(?:dx\.)?doi\.org/|doi:\s*)?10\.[0-9]{4,9}/[^\s<>]+", match[2]))
            if identifiers:
                entries.append(BibliographyEntry(match[1], identifiers))
    return tuple(entries)


def resolve_numbered_citations(text, source_key, targets, bibliography, *, policy_revision):
    """标签必须对应唯一精确标识；重复标签或多个目标保持歧义，不猜测 [1] 的归属。"""
    labels, catalog = {}, {}
    for entry in bibliography:
        labels.setdefault(entry.label, set()).update(canonical_identifier(value) for value in entry.identifiers)
    for target in targets:
        require(target.key.owner_binding_ref == source_key.owner_binding_ref, "文献目录不能跨 App")
        for value in target.identifiers:
            catalog.setdefault(canonical_identifier(value), set()).add(target.key)
    result = []
    for match in re.finditer(r"\[([1-9][0-9]{0,5})\]", text):
        # 章节中的条目声明本身不重复当作一次正文引用。
        if not text[text.rfind("\n", 0, match.start()) + 1:match.start()].strip():
            continue
        identifiers = labels.get(match[1], set())
        candidates = {key for value in identifiers for key in catalog.get(value, ())
                      if key.document_version_ref != source_key.document_version_ref}
        resolved = len(identifiers) == 1 and len(candidates) == 1
        status = "RESOLVED" if resolved else "AMBIGUOUS" if len(identifiers) > 1 or len(candidates) > 1 else "UNRESOLVED"
        span = SourceSpan(source_key.object_key, match.start(), match.end())
        result.append(CitationResolution(fingerprint([source_key.stable_key, span.char_start,
            span.char_end, policy_revision]), source_key, span, next(iter(identifiers)) if len(identifiers) == 1
            else "label:" + match[1], status, next(iter(candidates)) if resolved else None))
    return tuple(result)


def document_citation_edges(contents, targets, *, policy_revision):
    """先按源文档分隔条目，再把允许视图跨度映射回规范原文。"""
    from dataclasses import replace
    by_version = {}
    for content in contents:
        by_version.setdefault((content.key.document_version_ref, content.key.parse_run_ref), []).append(content)
    edges = {}
    for document in by_version.values():
        bibliography = bibliography_entries(document)
        for content in document:
            resolutions = resolve_citations(content.text, content.key, targets, policy_revision=policy_revision)
            resolutions += resolve_numbered_citations(content.text, content.key, targets, bibliography,
                                                       policy_revision=policy_revision)
            for entry in resolutions:
                source_maps = [value for value in content.sources
                    if value.unit_start <= entry.span.char_start < entry.span.char_end <= value.unit_end]
                if len(source_maps) != 1:
                    continue
                mapping = source_maps[0]
                start = mapping.source_start + entry.span.char_start - mapping.unit_start
                span = SourceSpan(mapping.source_key.object_key, start, start + entry.span.char_end - entry.span.char_start)
                canonical = replace(entry, source_key=mapping.source_key, span=span,
                    occurrence_key=fingerprint([mapping.source_key.stable_key, span.char_start,
                        span.char_end, entry.identifier, policy_revision]))
                for edge in citation_edges((canonical,)):
                    edges[edge.relation_key] = edge
    return tuple(sorted(edges.values(), key=lambda edge: edge.relation_key))
