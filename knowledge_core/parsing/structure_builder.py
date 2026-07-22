"""Global outline resolution and semantic block construction."""

from dataclasses import dataclass, field
import hashlib
import re
from typing import Any

from .ir import Atom, AtomIr, HeadingDecision, StructureIr, StructureNode
from .reading_order import ReadingOrderResult


_BLOCK_TYPES = {
    "TEXT": "PARAGRAPH",
    "LIST_ITEM": "LIST",
    "TABLE": "TABLE",
    "PICTURE": "FIGURE",
    "CAPTION": "CAPTION",
    "FORMULA": "FORMULA",
    "CODE": "CODE_BLOCK",
    "FOOTNOTE": "FOOTNOTE",
    "VISUAL_DESCRIPTION": "FIGURE",
}


@dataclass
class _MutableNode:
    node_id: str
    node_type: str
    parent_node_id: str | None
    ordinal: int
    atom_ids: list[str] = field(default_factory=list)
    heading: HeadingDecision | None = None
    heading_path: tuple[str, ...] = ()
    continuation_of: str | None = None
    decision_provenance: dict[str, Any] = field(default_factory=dict)


class OutlineResolver:
    def __init__(self, *, resolver_version: str = "outline/v1"):
        self._resolver_version = resolver_version

    def build(self, atom_ir: AtomIr, reading_order: ReadingOrderResult) -> StructureIr:
        atom_ir.validate()
        atom_by_id = {atom.atom_id: atom for atom in atom_ir.atoms}
        root = _MutableNode(
            node_id=self._node_id(atom_ir.document_version_id, "DOCUMENT", ()),
            node_type="DOCUMENT", parent_node_id=None, ordinal=0,
            decision_provenance={"resolver": self._resolver_version},
        )
        nodes: list[_MutableNode] = [root]
        section_stack: list[tuple[int, _MutableNode]] = []
        atom_owner: dict[str, _MutableNode] = {}
        current_path: tuple[str, ...] = ()

        for atom_id in reading_order.ordered_atom_ids:
            atom = atom_by_id[atom_id]
            if atom.atom_type == "TITLE_CANDIDATE":
                level, confidence, reasons = self._heading_level(atom, section_stack)
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                parent = section_stack[-1][1] if section_stack else root
                current_path = tuple(
                    [entry[1].heading.text for entry in section_stack if entry[1].heading]
                    + [atom.content_text]
                )
                node = _MutableNode(
                    node_id=self._node_id(atom_ir.document_version_id, "SECTION", (atom.atom_id,)),
                    node_type="SECTION", parent_node_id=parent.node_id, ordinal=len(nodes),
                    atom_ids=[atom.atom_id],
                    heading=HeadingDecision(
                        atom_id=atom.atom_id, text=atom.content_text, level=level,
                        confidence=confidence, reasons=tuple(reasons),
                    ),
                    heading_path=current_path,
                    decision_provenance={"resolver": self._resolver_version, "reasons": reasons},
                )
                nodes.append(node)
                section_stack.append((level, node))
                atom_owner[atom.atom_id] = node
                continue

            continued_from = reading_order.continuation_of.get(atom.atom_id)
            previous = atom_owner.get(continued_from) if continued_from else None
            node_type = _BLOCK_TYPES.get(atom.atom_type)
            if node_type is None:
                raise ValueError(f"Atom type has no semantic block mapping: {atom.atom_type}")
            if previous is not None and previous.node_type == node_type:
                previous.atom_ids.append(atom.atom_id)
                atom_owner[atom.atom_id] = previous
                continue

            parent = section_stack[-1][1] if section_stack else root
            caption_target = reading_order.caption_parent.get(atom.atom_id)
            if caption_target and caption_target in atom_owner:
                parent = atom_owner[caption_target]
            if node_type == "LIST" and nodes[-1].node_type == "LIST" and nodes[-1].parent_node_id == parent.node_id:
                nodes[-1].atom_ids.append(atom.atom_id)
                atom_owner[atom.atom_id] = nodes[-1]
                continue
            node = _MutableNode(
                node_id=self._node_id(atom_ir.document_version_id, node_type, (atom.atom_id,)),
                node_type=node_type, parent_node_id=parent.node_id, ordinal=len(nodes),
                atom_ids=[atom.atom_id], heading_path=current_path,
                decision_provenance={
                    "builder": "semantic-block/v1",
                    "reading_order_confidence": reading_order.confidence_by_atom.get(atom.atom_id, 0.0),
                },
            )
            nodes.append(node)
            atom_owner[atom.atom_id] = node

        immutable = tuple(self._freeze(node, atom_by_id) for node in nodes)
        result = StructureIr(
            document_version_id=atom_ir.document_version_id,
            atom_ir_sha256=atom_ir.fingerprint(), nodes=immutable,
            generator={"name": "outline-semantic-block-pipeline", "version": self._resolver_version},
        )
        result.validate(atom_ir)
        return result

    def _heading_level(
        self, atom: Atom, section_stack: list[tuple[int, _MutableNode]]
    ) -> tuple[int, float, list[str]]:
        declared = atom.style.get("declared_level")
        numbered = self._numbering_level(atom.content_text)
        reasons: list[str] = []
        candidates: list[int] = []
        if isinstance(declared, int) and declared > 0:
            candidates.append(declared)
            reasons.append(f"docling_level:{declared}")
        if numbered is not None:
            candidates.append(numbered)
            reasons.append(f"numbering_level:{numbered}")
        level = round(sum(candidates) / len(candidates)) if candidates else 1
        confidence = 0.95 if len(set(candidates)) == 1 and candidates else 0.82 if candidates else 0.60
        previous_level = section_stack[-1][0] if section_stack else 0
        if level > previous_level + 1:
            level = previous_level + 1
            confidence = min(confidence, 0.70)
            reasons.append("repair:illegal_level_jump")
        if not reasons:
            reasons.append("fallback:title_candidate")
        return max(1, min(9, level)), confidence, reasons

    @staticmethod
    def _numbering_level(text: str) -> int | None:
        match = re.match(r"^\s*(\d+(?:\.\d+){0,8})(?:[\s、.)]|$)", text)
        if match:
            return match.group(1).count(".") + 1
        if re.match(r"^\s*第[一二三四五六七八九十百千万\d]+[章节篇部]", text):
            return 1
        if re.match(r"^\s*[一二三四五六七八九十]+、", text):
            return 1
        if re.match(r"^\s*[（(][一二三四五六七八九十\d]+[)）]", text):
            return 2
        return None

    @staticmethod
    def _freeze(node: _MutableNode, atom_by_id: dict[str, Atom]) -> StructureNode:
        pages = [
            locator.page_no for atom_id in node.atom_ids for locator in atom_by_id[atom_id].locators
            if locator.page_no is not None
        ]
        page_range = (min(pages), max(pages)) if pages else None
        return StructureNode(
            node_id=node.node_id, node_type=node.node_type,
            parent_node_id=node.parent_node_id, ordinal=node.ordinal,
            atom_ids=tuple(node.atom_ids), heading=node.heading,
            heading_path=node.heading_path, page_range=page_range,
            continuation_of=node.continuation_of,
            decision_provenance=node.decision_provenance,
        )

    @staticmethod
    def _node_id(document_version_id: int, node_type: str, atom_ids: tuple[str, ...]) -> str:
        raw = f"{document_version_id}|{node_type}|{'|'.join(atom_ids)}"
        return f"node:{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}"
