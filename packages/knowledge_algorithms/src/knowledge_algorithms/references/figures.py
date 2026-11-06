"""保守的图号规则基线；不把 OCR 猜测、相邻图片当成确定引用。"""

from dataclasses import dataclass
from enum import Enum
import re

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, SourceSpan, fingerprint, nonempty


_LABEL = re.compile(
    r"(?:图|(?<![A-Za-z0-9_])Fig(?:ure)?\.?)\s*(?P<number>[0-9]+(?:[-.][0-9]+)*)"
    r"(?:\s*[(（](?P<panel>[a-zA-Z])[)）])?", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FigureLabel:
    number: str
    subfigure: str | None


def parse_figure_label(text: str) -> FigureLabel | None:
    match = _LABEL.fullmatch(text.strip())
    if match is None:
        return None
    panel = match.group("panel")
    return FigureLabel(match.group("number"), panel.lower() if panel else None)


@dataclass(frozen=True, slots=True)
class FigureIdentity:
    key: EvidenceKey
    display_label: str
    numbering_scope: str
    native_anchor: str | None = None

    def __post_init__(self) -> None:
        nonempty(self.numbering_scope)
        require(parse_figure_label(self.display_label) is not None, "图题编号不符合当前规则")


@dataclass(frozen=True, slots=True)
class ReferenceOccurrence:
    occurrence_key: str
    source_key: EvidenceKey
    span: SourceSpan
    raw_label: str
    numbering_scope: str | None
    native_target_anchor: str | None = None

    def __post_init__(self) -> None:
        nonempty(self.occurrence_key, self.raw_label)
        require(self.span.node_key == self.source_key.object_key, "引用跨度不属于源节点")
        require(self.span.char_end - self.span.char_start == len(self.raw_label), "引用跨度与原文不一致")


class ResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    AMBIGUOUS = "AMBIGUOUS"
    UNRESOLVED = "UNRESOLVED"
    MISSING_TARGET = "MISSING_TARGET"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True, slots=True)
class FigureResolution:
    occurrence: ReferenceOccurrence
    status: ResolutionStatus
    target_key: EvidenceKey | None
    basis: str


def extract_figure_references(
    text: str, source_key: EvidenceKey, *, numbering_scope: str | None,
    policy_version: str,
) -> tuple[ReferenceOccurrence, ...]:
    nonempty(policy_version)
    occurrences = []
    for match in _LABEL.finditer(text):
        # 非法编号的前缀不可被截取后强连，例如 OCR 的“图 1-l”。
        remainder = text[match.end():]
        if remainder and (remainder[0] in "-（(" or remainder[0] == "."
                          and len(remainder) > 1 and remainder[1].isascii() and remainder[1].isalnum()
                          or remainder[0].isascii()
                          and remainder[0].isalnum()):
            continue
        span = SourceSpan(source_key.object_key, match.start(), match.end())
        occurrence_key = fingerprint({"source": source_key.stable_key,
                                      "start": span.char_start, "end": span.char_end,
                                      "policy": policy_version})
        occurrences.append(ReferenceOccurrence(occurrence_key, source_key, span,
                           match.group(), numbering_scope))
    return tuple(occurrences)


def resolve_figure_reference(
    occurrence: ReferenceOccurrence, figures: tuple[FigureIdentity, ...],
) -> FigureResolution:
    source = occurrence.source_key
    require(all(f.key.owner_binding_ref == source.owner_binding_ref for f in figures),
            "图片目录不能混用其他 App")
    require(len({f.key for f in figures}) == len(figures), "图片目录身份重复")
    label = parse_figure_label(occurrence.raw_label)
    if label is None:
        return FigureResolution(occurrence, ResolutionStatus.UNRESOLVED, None, "UNSUPPORTED_LABEL")
    catalog = [f for f in figures if f.key.document_version_ref == source.document_version_ref
               and f.key.parse_run_ref == source.parse_run_ref]
    if occurrence.native_target_anchor:
        native = [f for f in catalog if f.native_anchor == occurrence.native_target_anchor]
        if len(native) != 1:
            status = ResolutionStatus.AMBIGUOUS if native else ResolutionStatus.MISSING_TARGET
            return FigureResolution(occurrence, status, None, "NATIVE_ANCHOR")
        target = native[0]
        if (parse_figure_label(target.display_label) != label
                or occurrence.numbering_scope is not None
                and target.numbering_scope != occurrence.numbering_scope):
            return FigureResolution(occurrence, ResolutionStatus.CONFLICT, None, "ANCHOR_LABEL_CONFLICT")
        return FigureResolution(occurrence, ResolutionStatus.RESOLVED, target.key, "NATIVE_ANCHOR")
    candidates = [f for f in catalog if parse_figure_label(f.display_label) == label
                  and (occurrence.numbering_scope is None
                       or f.numbering_scope == occurrence.numbering_scope)]
    if len(candidates) == 1:
        return FigureResolution(occurrence, ResolutionStatus.RESOLVED, candidates[0].key, "LABEL_SCOPE")
    return FigureResolution(occurrence, ResolutionStatus.AMBIGUOUS if candidates
                            else ResolutionStatus.MISSING_TARGET, None, "LABEL_SCOPE")
