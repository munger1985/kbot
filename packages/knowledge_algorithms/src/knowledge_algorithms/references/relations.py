"""纯关系清单组装；输入必须是 App 已授权、同快照的关系页。"""

from dataclasses import dataclass
from enum import Enum

from ..contracts.errors import require
from ..contracts.identity import EvidenceKey, SourceSpan, nonempty


class RelationType(str, Enum):
    REFERS_TO = "REFERS_TO"
    CAPTION_OF = "CAPTION_OF"
    HAS_ASSET = "HAS_ASSET"
    PART_OF = "PART_OF"
    CITES_DOCUMENT = "CITES_DOCUMENT"


@dataclass(frozen=True, slots=True)
class ResolvedRelation:
    relation_key: str
    relation_type: RelationType
    source_key: EvidenceKey
    target_key: EvidenceKey
    basis: str
    occurrence_key: str | None = None
    source_span: SourceSpan | None = None

    def __post_init__(self) -> None:
        nonempty(self.relation_key, self.basis)
        require(isinstance(self.relation_type, RelationType), "关系类型无效")
        require(self.source_key.owner_binding_ref == self.target_key.owner_binding_ref,
                "关系不能跨 App")
        require(self.relation_type == RelationType.CITES_DOCUMENT or
                (self.source_key.document_version_ref, self.source_key.parse_run_ref)
                == (self.target_key.document_version_ref, self.target_key.parse_run_ref),
                "文内图文关系不能跨版本或解析代次")
        if self.relation_type in {RelationType.REFERS_TO, RelationType.CITES_DOCUMENT}:
            require(bool(self.occurrence_key) and self.source_span is not None,
                    "正文引用关系必须保留每次出现与跨度")
            require(self.source_span.node_key == self.source_key.object_key, "引用跨度不属于源对象")


@dataclass(frozen=True, slots=True)
class LinkedManifest:
    anchor_key: EvidenceKey
    linked_keys: tuple[EvidenceKey, ...]
    relations: tuple[ResolvedRelation, ...]
    manifest_status: str
    payload_status: str = "DEFERRED"
    continuation: str | None = None
    incomplete_reason: str | None = None


def direct_figure_manifest(
    anchor: EvidenceKey, *, anchor_kind: str,
    relations: tuple[ResolvedRelation, ...], relation_pages_exhausted: bool,
) -> LinkedManifest:
    """只遍历锚点直接引用及图题/资产/父子图，不递归扩散正文引用。"""
    require(anchor_kind in {"TEXT", "FIGURE", "CAPTION", "REGION"}, "未支持的图文锚点类型")
    require(all(r.source_key.owner_binding_ref == anchor.owner_binding_ref for r in relations),
            "关系页不属于当前 App")
    require(len({r.relation_key for r in relations}) == len(relations), "关系页身份重复")
    figures: set[EvidenceKey] = {anchor} if anchor_kind in {"FIGURE", "REGION"} else set()
    selected: dict[str, ResolvedRelation] = {}
    for relation in relations:
        if relation.relation_type == RelationType.PART_OF and relation.basis == "EXPLICIT_TRANSCRIPTION_REGION":
            if anchor == relation.source_key:
                figures.add(relation.target_key)
                selected[relation.relation_key] = relation
            elif anchor == relation.target_key:
                selected[relation.relation_key] = relation
    for relation in relations:
        if relation.relation_type == RelationType.CITES_DOCUMENT and anchor in {relation.source_key, relation.target_key}:
            selected[relation.relation_key] = relation
    for relation in relations:
        if (anchor_kind == "TEXT" and relation.relation_type == RelationType.REFERS_TO
                and relation.source_key == anchor
                or anchor_kind == "CAPTION" and relation.relation_type == RelationType.CAPTION_OF
                and relation.source_key == anchor):
            figures.add(relation.target_key)
            selected[relation.relation_key] = relation
    # 只补一层必要父图，不把其他子图误当当前子图。
    for relation in relations:
        if relation.relation_type == RelationType.PART_OF and relation.source_key in figures:
            selected[relation.relation_key] = relation
    parents = {r.target_key for r in selected.values() if r.relation_type == RelationType.PART_OF}
    for relation in relations:
        include = (relation.relation_type == RelationType.CAPTION_OF and relation.target_key in figures | parents
                   or relation.relation_type == RelationType.HAS_ASSET and relation.source_key in figures | parents
                   or anchor_kind in {"FIGURE", "REGION", "CAPTION"}
                   and relation.relation_type == RelationType.REFERS_TO and relation.target_key in figures)
        if include:
            selected[relation.relation_key] = relation
    edges = tuple(sorted(selected.values(), key=lambda r: (
        r.source_span.char_start if r.source_span else -1, r.relation_key)))
    linked = tuple(dict.fromkeys(key for relation in edges
                                for key in (relation.source_key, relation.target_key) if key != anchor))
    return LinkedManifest(anchor, linked, edges,
                          "COMPLETE" if relation_pages_exhausted else "PARTIAL")
