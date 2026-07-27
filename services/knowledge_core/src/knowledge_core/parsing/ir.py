"""Immutable Atom and Structure intermediate representations."""

from uuid import UUID
from dataclasses import asdict, dataclass, field
from typing import Any

from .contracts import canonical_json_hash


ATOM_TYPES = frozenset({
    "TITLE_CANDIDATE", "TEXT", "LIST_ITEM", "TABLE", "PICTURE",
    "CAPTION", "FORMULA", "CODE", "FOOTNOTE", "HEADER", "FOOTER",
    "VISUAL_DESCRIPTION",
})
STRUCTURE_NODE_TYPES = frozenset({
    "DOCUMENT", "SECTION", "PARAGRAPH", "LIST", "TABLE", "FIGURE",
    "CAPTION", "FORMULA", "CODE_BLOCK", "FOOTNOTE",
})
_REPEATED_REGION_TYPES = frozenset({"HEADER", "FOOTER"})


class IrValidationError(ValueError):
    """Raised when an IR violates a hard structural invariant."""


@dataclass(frozen=True)
class PageGeometry:
    page_no: int
    width: float
    height: float

    def validate(self) -> None:
        if self.page_no < 1 or self.width <= 0 or self.height <= 0:
            raise IrValidationError("page geometry requires positive page number and dimensions")


@dataclass(frozen=True)
class AtomLocator:
    page_no: int | None = None
    bbox: tuple[float, float, float, float] | None = None
    logical_ref: str | None = None
    original_bbox: dict[str, Any] | None = None

    def validate(self, atom_id: str, page_numbers: set[int]) -> None:
        if self.logical_ref:
            if self.page_no is not None or self.bbox is not None:
                raise IrValidationError(f"Atom {atom_id} mixes logical and page locators")
            return
        if self.page_no is None or self.bbox is None or self.page_no not in page_numbers:
            raise IrValidationError(f"Atom {atom_id} references an unknown page")
        x0, y0, x1, y1 = self.bbox
        if not (0 <= x0 <= x1 <= 1 and 0 <= y0 <= y1 <= 1):
            raise IrValidationError(f"Atom {atom_id} bbox must use normalized top-left coordinates")


@dataclass(frozen=True)
class Atom:
    atom_id: str
    source_ref: str
    atom_type: str
    content_text: str
    locators: tuple[AtomLocator, ...]
    reading_order_hint: int
    original_label: str
    style: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    provenance: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    repeated_region_key: str | None = None

    def validate(self, page_numbers: set[int]) -> None:
        if not self.atom_id.strip() or not self.source_ref.strip():
            raise IrValidationError("atom_id and source_ref are required")
        if self.atom_type not in ATOM_TYPES:
            raise IrValidationError(f"unsupported atom_type: {self.atom_type}")
        if not self.locators:
            raise IrValidationError(f"Atom {self.atom_id} requires at least one locator")
        for locator in self.locators:
            locator.validate(self.atom_id, page_numbers)
        if self.reading_order_hint < 0 or not 0 <= self.confidence <= 1:
            raise IrValidationError(f"Atom {self.atom_id} has invalid order or confidence")
        if not self.provenance:
            raise IrValidationError(f"Atom {self.atom_id} requires provenance")


@dataclass(frozen=True)
class AtomIr:
    document_version_id: UUID
    pages: tuple[PageGeometry, ...]
    atoms: tuple[Atom, ...]
    generator: dict[str, str]
    ir_version: str = "kc-atom/v1"

    def validate(self) -> None:
        if self.ir_version != "kc-atom/v1":
            raise IrValidationError("invalid Atom IR identity")
        if not self.generator.get("name") or not self.generator.get("version"):
            raise IrValidationError("Atom IR generator name and version are required")
        page_numbers: set[int] = set()
        for page in self.pages:
            page.validate()
            if page.page_no in page_numbers:
                raise IrValidationError(f"duplicate page number: {page.page_no}")
            page_numbers.add(page.page_no)
        atom_ids: set[str] = set()
        source_refs: set[str] = set()
        for atom in self.atoms:
            atom.validate(page_numbers)
            if atom.atom_id in atom_ids:
                raise IrValidationError(f"duplicate atom_id: {atom.atom_id}")
            if atom.source_ref in source_refs:
                raise IrValidationError(f"duplicate source_ref: {atom.source_ref}")
            atom_ids.add(atom.atom_id)
            source_refs.add(atom.source_ref)

    def fingerprint(self) -> str:
        self.validate()
        return canonical_json_hash(asdict(self))


@dataclass(frozen=True)
class HeadingDecision:
    atom_id: str
    text: str
    level: int
    confidence: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class StructureNode:
    node_id: str
    node_type: str
    parent_node_id: str | None
    ordinal: int
    atom_ids: tuple[str, ...]
    heading: HeadingDecision | None = None
    heading_path: tuple[str, ...] = field(default_factory=tuple)
    page_range: tuple[int, int] | None = None
    continuation_of: str | None = None
    decision_provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StructureIr:
    document_version_id: UUID
    atom_ir_sha256: str
    nodes: tuple[StructureNode, ...]
    generator: dict[str, str]
    ir_version: str = "kc-structure/v1"

    def validate(self, atom_ir: AtomIr) -> None:
        atom_ir.validate()
        if self.document_version_id != atom_ir.document_version_id:
            raise IrValidationError("Structure IR and Atom IR document versions differ")
        if self.atom_ir_sha256 != atom_ir.fingerprint():
            raise IrValidationError("Structure IR references a different Atom IR")
        if self.ir_version != "kc-structure/v1" or not self.generator.get("name") or not self.generator.get("version"):
            raise IrValidationError("invalid Structure IR identity or generator")

        nodes_by_id: dict[str, StructureNode] = {}
        ordinals: set[int] = set()
        for node in self.nodes:
            if not node.node_id.strip() or node.node_type not in STRUCTURE_NODE_TYPES:
                raise IrValidationError("Structure node identity and type are required")
            if node.node_id in nodes_by_id or node.ordinal in ordinals:
                raise IrValidationError("Structure node IDs and ordinals must be unique")
            if not node.decision_provenance:
                raise IrValidationError(f"Node {node.node_id} requires decision provenance")
            nodes_by_id[node.node_id] = node
            ordinals.add(node.ordinal)

        roots = [node for node in self.nodes if node.parent_node_id is None]
        if len(roots) != 1 or roots[0].node_type != "DOCUMENT":
            raise IrValidationError("Structure IR requires exactly one DOCUMENT root")

        atom_by_id = {atom.atom_id: atom for atom in atom_ir.atoms}
        atom_owners: dict[str, str] = {}
        for node in self.nodes:
            if node.parent_node_id is not None:
                parent = nodes_by_id.get(node.parent_node_id)
                if parent is None or parent.ordinal > node.ordinal:
                    raise IrValidationError(f"Node {node.node_id} has an invalid parent")
            self._validate_acyclic(node, nodes_by_id)
            for atom_id in node.atom_ids:
                if atom_id not in atom_by_id:
                    raise IrValidationError(f"Node {node.node_id} references unknown Atom {atom_id}")
                if atom_id in atom_owners:
                    raise IrValidationError(f"Atom {atom_id} has multiple primary owners")
                atom_owners[atom_id] = node.node_id
            if node.heading is not None:
                heading = node.heading
                if node.node_type != "SECTION" or heading.atom_id not in node.atom_ids:
                    raise IrValidationError(f"Node {node.node_id} has an invalid heading reference")
                if atom_by_id[heading.atom_id].atom_type != "TITLE_CANDIDATE":
                    raise IrValidationError(f"Heading {heading.atom_id} is not a title candidate")
                if not 1 <= heading.level <= 9 or not 0 <= heading.confidence <= 1 or not heading.reasons:
                    raise IrValidationError(f"Node {node.node_id} has an invalid heading decision")

        expected = {
            atom.atom_id for atom in atom_ir.atoms
            if atom.atom_type not in _REPEATED_REGION_TYPES
        }
        if set(atom_owners) != expected:
            missing = sorted(expected.difference(atom_owners))
            extra = sorted(set(atom_owners).difference(expected))
            raise IrValidationError(f"Atom ownership mismatch; missing={missing}, extra={extra}")

    @staticmethod
    def _validate_acyclic(node: StructureNode, nodes_by_id: dict[str, StructureNode]) -> None:
        visited = {node.node_id}
        parent_id = node.parent_node_id
        while parent_id is not None:
            if parent_id in visited:
                raise IrValidationError(f"Structure cycle detected at {node.node_id}")
            visited.add(parent_id)
            parent = nodes_by_id.get(parent_id)
            if parent is None:
                return
            parent_id = parent.parent_node_id

    def fingerprint(self, atom_ir: AtomIr) -> str:
        self.validate(atom_ir)
        return canonical_json_hash(asdict(self))
