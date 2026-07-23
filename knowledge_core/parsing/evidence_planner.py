"""Plan retrieval Evidence from validated Structure IR without mutating it."""

from uuid import UUID
from dataclasses import asdict, dataclass
import re
from typing import Any

from .contracts import build_evidence_key
from .ir import Atom, AtomIr, StructureIr, StructureNode


_NODE_TO_EVIDENCE = {
    "SECTION": "SECTION",
    "PARAGRAPH": "PARAGRAPH",
    "LIST": "PARAGRAPH",
    "TABLE": "TABLE",
    "FIGURE": "IMAGE",
    "FORMULA": "PARAGRAPH",
    "CODE_BLOCK": "PARAGRAPH",
    "FOOTNOTE": "PARAGRAPH",
}


@dataclass(frozen=True)
class EvidencePolicy:
    min_tokens: int = 80
    target_tokens: int = 450
    max_tokens: int = 900

    def validate(self) -> None:
        if not 0 < self.min_tokens <= self.target_tokens <= self.max_tokens:
            raise ValueError("Evidence token limits must be positive and ordered")


@dataclass(frozen=True)
class PlannedEvidence:
    evidence_key: str
    evidence_type: str
    ordinal: int
    fragment_index: int
    content_text: str
    source_spans: list[dict[str, Any]]
    locator_schema_version: str
    locator: dict[str, Any]
    provenance: dict[str, Any]
    parent_evidence_key: str | None
    source_item_ref: str | None
    heading_path: list[str]
    section_key: str | None
    hierarchy_depth: int
    heading_level: int | None
    page_start: int | None
    page_end: int | None
    language_code: str | None
    token_count: int
    quality_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _Fragment:
    atoms: list[Atom]
    content_text: str
    source_spans: list[dict[str, Any]]
    locator_override: dict[str, Any] | None = None


class EvidencePlanner:
    def __init__(self, policy: EvidencePolicy | None = None):
        self._policy = policy or EvidencePolicy()
        self._policy.validate()

    @property
    def policy(self) -> EvidencePolicy:
        return self._policy

    def plan(
        self, *, parse_view_id: UUID, atom_ir: AtomIr, structure_ir: StructureIr
    ) -> tuple[PlannedEvidence, ...]:
        structure_ir.validate(atom_ir)
        atom_by_id = {atom.atom_id: atom for atom in atom_ir.atoms}
        node_by_id = {node.node_id: node for node in structure_ir.nodes}
        children: dict[str, list[StructureNode]] = {}
        for node in structure_ir.nodes:
            if node.parent_node_id:
                children.setdefault(node.parent_node_id, []).append(node)

        units: list[tuple[StructureNode, list[Atom], int]] = []
        paragraph_buffer: list[Atom] = []
        paragraph_node: StructureNode | None = None

        def flush_paragraphs() -> None:
            nonlocal paragraph_buffer, paragraph_node
            if paragraph_node is not None and paragraph_buffer:
                units.extend(
                    (paragraph_node, atoms, index)
                    for index, atoms in enumerate(self._group_atoms(paragraph_buffer))
                )
            paragraph_buffer, paragraph_node = [], None

        for node in sorted(structure_ir.nodes, key=lambda value: value.ordinal):
            if node.node_type in {"DOCUMENT", "CAPTION"}:
                continue
            atoms = [atom_by_id[atom_id] for atom_id in node.atom_ids]
            if node.node_type == "PARAGRAPH":
                if paragraph_node is not None and paragraph_node.parent_node_id != node.parent_node_id:
                    flush_paragraphs()
                paragraph_node = paragraph_node or node
                paragraph_buffer.extend(atoms)
                if self._tokens(paragraph_buffer) >= self._policy.target_tokens:
                    flush_paragraphs()
                continue
            flush_paragraphs()
            if node.node_type in {"TABLE", "FIGURE"}:
                for caption in children.get(node.node_id, []):
                    if caption.node_type == "CAPTION":
                        atoms.extend(atom_by_id[atom_id] for atom_id in caption.atom_ids)
            for index, atom_group in enumerate(self._group_atoms(atoms, split_single=node.node_type != "TABLE")):
                units.append((node, atom_group, index))
        flush_paragraphs()

        node_primary_key: dict[str, str] = {}
        output: list[PlannedEvidence] = []
        sheet_key_by_ref: dict[str, str] = {}
        sheet_groups: dict[str, tuple[StructureNode, list[Atom]]] = {}
        for node, atoms, _ in units:
            spreadsheet = atoms[0].style.get("spreadsheet") if atoms else None
            if node.node_type == "TABLE" and spreadsheet:
                sheet_ref = spreadsheet["sheet_ref"]
                if sheet_ref not in sheet_groups:
                    sheet_groups[sheet_ref] = (node, [])
                sheet_groups[sheet_ref][1].extend(atoms)
        for sheet_ref, (node, atoms) in sheet_groups.items():
            spans = [{"atom_id": atom.atom_id} for atom in atoms]
            key = build_evidence_key(
                parse_view_id=parse_view_id, source_spans=spans,
                fragment_index=0, evidence_type="SHEET",
            )
            sheet = atoms[0].style["spreadsheet"]
            content = f"{sheet['sheet_name']}\n\n" + "\n\n".join(atom.content_text for atom in atoms)
            output.append(PlannedEvidence(
                evidence_key=key, evidence_type="SHEET", ordinal=len(output), fragment_index=0,
                content_text=content, source_spans=spans,
                locator_schema_version="spreadsheet/v1",
                locator={"sheet_name": sheet["sheet_name"], "sheet_ref": sheet_ref},
                provenance={"planner": "evidence-planner/v1", "source_nodes": [node.node_id], "extractors": self._extractors(atoms)},
                parent_evidence_key=None, source_item_ref=None, heading_path=[],
                section_key=None, hierarchy_depth=0, heading_level=None,
                page_start=None, page_end=None, language_code=None,
                token_count=self._token_count_text(content),
                quality_score=min(atom.confidence for atom in atoms),
            ))
            sheet_key_by_ref[sheet_ref] = key
        fragment_counter: dict[str, int] = {}
        for node, atoms, _ in units:
            spreadsheet = atoms[0].style.get("spreadsheet") if atoms else None
            evidence_type = "CELL_RANGE" if node.node_type == "TABLE" and spreadsheet else _NODE_TO_EVIDENCE[node.node_type]
            fragments = (
                self._spreadsheet_fragments(atoms[0])
                if spreadsheet and self._tokens(atoms) > self._policy.max_tokens
                else self._fragments(atoms, allow_text_split=node.node_type != "TABLE")
            )
            for fragment in fragments:
                if not fragment.content_text.strip():
                    continue
                fragment_index = fragment_counter.get(node.node_id, 0)
                fragment_counter[node.node_id] = fragment_index + 1
                evidence = self._make_evidence(
                    parse_view_id=parse_view_id, node=node, node_by_id=node_by_id,
                    node_primary_key=node_primary_key, fragment=fragment,
                    evidence_type=evidence_type, fragment_index=fragment_index,
                    ordinal=len(output),
                    parent_override=sheet_key_by_ref.get(spreadsheet["sheet_ref"]) if spreadsheet else None,
                )
                output.append(evidence)
                node_primary_key.setdefault(node.node_id, evidence.evidence_key)
            if node.node_type == "TABLE" and atoms and not spreadsheet and self._tokens(atoms) > self._policy.max_tokens:
                for row_index, row_fragment in enumerate(self._table_row_fragments(atoms[0])):
                    row_evidence = self._make_evidence(
                        parse_view_id=parse_view_id, node=node, node_by_id=node_by_id,
                        node_primary_key=node_primary_key, fragment=row_fragment,
                        evidence_type="TABLE_ROW", fragment_index=row_index,
                        ordinal=len(output), parent_override=node_primary_key[node.node_id],
                    )
                    output.append(row_evidence)
        return tuple(output)

    def _make_evidence(
        self, *, parse_view_id: UUID, node: StructureNode,
        node_by_id: dict[str, StructureNode], node_primary_key: dict[str, str],
        fragment: _Fragment, evidence_type: str, fragment_index: int,
        ordinal: int, parent_override: str | None = None,
    ) -> PlannedEvidence:
        evidence_key = build_evidence_key(
            parse_view_id=parse_view_id, source_spans=fragment.source_spans,
            fragment_index=fragment_index, evidence_type=evidence_type,
        )
        if fragment.locator_override is not None:
            locators, locator_schema_version = fragment.locator_override, "spreadsheet/v1"
        else:
            locators, locator_schema_version = self._locator(fragment.atoms)
        page_numbers = [entry["page_no"] for entry in locators.get("pages", [])]
        heading_level = node.heading.level if node.heading else self._nearest_heading_level(node, node_by_id)
        return PlannedEvidence(
            evidence_key=evidence_key, evidence_type=evidence_type,
            ordinal=ordinal, fragment_index=fragment_index,
            content_text=fragment.content_text, source_spans=fragment.source_spans,
            locator_schema_version=locator_schema_version, locator=locators,
            provenance={
                "planner": "evidence-planner/v1", "source_nodes": [node.node_id],
                "extractors": self._extractors(fragment.atoms),
            },
            parent_evidence_key=parent_override or self._parent_key(node, node_by_id, node_primary_key),
            source_item_ref=fragment.atoms[0].source_ref if len(fragment.atoms) == 1 else None,
            heading_path=list(node.heading_path),
            section_key=self._section_key(node, node_by_id),
            hierarchy_depth=len(node.heading_path), heading_level=heading_level,
            page_start=min(page_numbers) if page_numbers else None,
            page_end=max(page_numbers) if page_numbers else None,
            language_code=None, token_count=self._token_count_text(fragment.content_text),
            quality_score=min(atom.confidence for atom in fragment.atoms),
        )

    def _fragments(self, atoms: list[Atom], *, allow_text_split: bool) -> list[_Fragment]:
        content = "\n\n".join(atom.content_text for atom in atoms if atom.content_text.strip())
        if not allow_text_split or len(atoms) != 1 or self._token_count_text(content) <= self._policy.max_tokens:
            return [_Fragment(atoms, content, [{"atom_id": atom.atom_id} for atom in atoms])]
        atom = atoms[0]
        matches = list(re.finditer(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", atom.content_text))
        fragments: list[_Fragment] = []
        for offset in range(0, len(matches), self._policy.max_tokens):
            token_group = matches[offset:offset + self._policy.max_tokens]
            start, end = token_group[0].start(), token_group[-1].end()
            while start < end and atom.content_text[start].isspace():
                start += 1
            while end > start and atom.content_text[end - 1].isspace():
                end -= 1
            fragments.append(_Fragment(
                [atom], atom.content_text[start:end],
                [{"atom_id": atom.atom_id, "char_start": start, "char_end": end}],
            ))
        return fragments

    def _table_row_fragments(self, atom: Atom) -> list[_Fragment]:
        lines = list(re.finditer(r".*(?:\n|$)", atom.content_text))
        lines = [line for line in lines if line.group(0)]
        if len(lines) <= 2:
            return []
        header = lines[:2]
        data = lines[2:]
        result: list[_Fragment] = []
        current: list[Any] = []
        for line in data:
            candidate = "".join(value.group(0) for value in [*header, *current, line]).rstrip()
            if current and self._token_count_text(candidate) > self._policy.max_tokens:
                result.append(self._table_fragment(atom, header, current))
                current = []
            current.append(line)
        if current:
            result.append(self._table_fragment(atom, header, current))
        return result

    def _spreadsheet_fragments(self, atom: Atom) -> list[_Fragment]:
        spreadsheet = atom.style["spreadsheet"]
        cells = spreadsheet.get("cells", [])
        header_cells = [cell for cell in cells if cell.get("column_header")]
        rows: dict[int, list[dict[str, Any]]] = {}
        for cell in cells:
            if cell.get("column_header"):
                continue
            rows.setdefault(int(cell["row_start"]), []).append(cell)
        if not rows:
            return self._fragments([atom], allow_text_split=False)
        fragments: list[_Fragment] = []
        current_rows: list[tuple[int, list[dict[str, Any]]]] = []
        for row in sorted(rows.items()):
            candidate = self._spreadsheet_text(header_cells, [*current_rows, row])
            if current_rows and self._token_count_text(candidate) > self._policy.max_tokens:
                fragments.append(self._spreadsheet_fragment(atom, spreadsheet, header_cells, current_rows))
                current_rows = []
            current_rows.append(row)
        if current_rows:
            fragments.append(self._spreadsheet_fragment(atom, spreadsheet, header_cells, current_rows))
        return fragments

    def _spreadsheet_fragment(
        self, atom: Atom, spreadsheet: dict[str, Any],
        header_cells: list[dict[str, Any]], rows: list[tuple[int, list[dict[str, Any]]]],
    ) -> _Fragment:
        all_cells = [*header_cells, *(cell for _, row_cells in rows for cell in row_cells)]
        row_start = min(cell["row_start"] for cell in all_cells)
        row_end = max(cell["row_end"] for cell in all_cells)
        column_start = min(cell["column_start"] for cell in all_cells)
        column_end = max(cell["column_end"] for cell in all_cells)
        cell_range = f"{self._column_name(column_start)}{row_start}:{self._column_name(column_end)}{row_end}"
        locator = {
            "sheet_name": spreadsheet["sheet_name"], "sheet_ref": spreadsheet["sheet_ref"],
            "table_ref": spreadsheet["table_ref"], "cell_range": cell_range,
            "row_start": row_start, "row_end": row_end,
            "column_start": column_start, "column_end": column_end,
        }
        return _Fragment(
            [atom], self._spreadsheet_text(header_cells, rows),
            [{"atom_id": atom.atom_id, "cell_range": cell_range}], locator,
        )

    @staticmethod
    def _spreadsheet_text(
        header_cells: list[dict[str, Any]], rows: list[tuple[int, list[dict[str, Any]]]],
    ) -> str:
        header = " | ".join(cell["text"] for cell in sorted(header_cells, key=lambda value: value["column_start"]))
        values = [header] if header else []
        for _, cells in rows:
            values.append(" | ".join(cell["text"] for cell in sorted(cells, key=lambda value: value["column_start"])))
        return "\n".join(values)

    @staticmethod
    def _column_name(column: int) -> str:
        letters = ""
        while column:
            column, remainder = divmod(column - 1, 26)
            letters = chr(65 + remainder) + letters
        return letters

    @staticmethod
    def _table_fragment(atom: Atom, header: list[Any], rows: list[Any]) -> _Fragment:
        parts = [*header, *rows]
        spans = [{
            "atom_id": atom.atom_id, "char_start": part.start(), "char_end": part.end(),
        } for part in parts if part.end() > part.start()]
        return _Fragment([atom], "".join(part.group(0) for part in parts).rstrip(), spans)

    def _group_atoms(self, atoms: list[Atom], *, split_single: bool = True) -> list[list[Atom]]:
        if not atoms:
            return []
        groups: list[list[Atom]] = []
        current: list[Atom] = []
        for atom in atoms:
            if current and self._tokens([*current, atom]) > self._policy.max_tokens:
                groups.append(current)
                current = []
            current.append(atom)
        if current:
            groups.append(current)
        if not split_single:
            return groups
        return groups

    @staticmethod
    def _tokens(atoms: list[Atom]) -> int:
        text = "\n".join(atom.content_text for atom in atoms)
        return EvidencePlanner._token_count_text(text)

    @staticmethod
    def _token_count_text(text: str) -> int:
        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))

    @staticmethod
    def _locator(atoms: list[Atom]) -> tuple[dict[str, Any], str]:
        spreadsheet = atoms[0].style.get("spreadsheet") if atoms else None
        if spreadsheet:
            return {
                key: spreadsheet[key] for key in (
                    "sheet_name", "sheet_ref", "table_ref", "cell_range",
                    "row_start", "row_end", "column_start", "column_end",
                )
            }, "spreadsheet/v1"
        pages = []
        logical_refs = []
        seen = set()
        for atom in atoms:
            for locator in atom.locators:
                if locator.logical_ref:
                    logical_refs.append(locator.logical_ref)
                    continue
                key = (locator.page_no, locator.bbox)
                if key in seen:
                    continue
                seen.add(key)
                pages.append({
                    "page_no": locator.page_no,
                    "bbox": list(locator.bbox),
                    "coordinate_space": "page_normalized_top_left",
                })
        if pages:
            return {"pages": pages}, "document/v1"
        return {
            "source_refs": list(dict.fromkeys(logical_refs)),
            "coordinate_space": "logical_document",
        }, "document-logical/v1"

    @staticmethod
    def _extractors(atoms: list[Atom]) -> list[str]:
        return sorted({
            str(entry.get("extractor"))
            for atom in atoms for entry in atom.provenance if entry.get("extractor")
        })

    @staticmethod
    def _parent_key(
        node: StructureNode, node_by_id: dict[str, StructureNode], node_keys: dict[str, str]
    ) -> str | None:
        parent_id = node.parent_node_id
        while parent_id:
            if parent_id in node_keys:
                return node_keys[parent_id]
            parent_id = node_by_id[parent_id].parent_node_id
        return None

    @staticmethod
    def _section_key(node: StructureNode, node_by_id: dict[str, StructureNode]) -> str | None:
        current: StructureNode | None = node
        while current is not None:
            if current.node_type == "SECTION":
                return current.node_id
            current = node_by_id.get(current.parent_node_id) if current.parent_node_id else None
        return None

    @staticmethod
    def _nearest_heading_level(node: StructureNode, node_by_id: dict[str, StructureNode]) -> int | None:
        current = node_by_id.get(node.parent_node_id) if node.parent_node_id else None
        while current is not None:
            if current.heading:
                return current.heading.level
            current = node_by_id.get(current.parent_node_id) if current.parent_node_id else None
        return None
