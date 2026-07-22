"""Deterministic page reading order and cross-page continuity resolution."""

from dataclasses import dataclass
import re

from .ir import Atom, AtomIr


@dataclass(frozen=True)
class ReadingOrderResult:
    ordered_atom_ids: tuple[str, ...]
    excluded_atom_ids: tuple[str, ...]
    continuation_of: dict[str, str]
    caption_parent: dict[str, str]
    confidence_by_atom: dict[str, float]
    resolver_version: str = "reading-order/v1"


class ReadingOrderResolver:
    def resolve(self, atom_ir: AtomIr) -> ReadingOrderResult:
        atom_ir.validate()
        by_page: dict[int, list[Atom]] = {}
        excluded: list[str] = []
        for atom in atom_ir.atoms:
            if atom.atom_type in {"HEADER", "FOOTER"}:
                excluded.append(atom.atom_id)
                continue
            page_no = atom.locators[0].page_no
            by_page.setdefault(page_no if page_no is not None else 0, []).append(atom)

        ordered: list[Atom] = []
        confidence: dict[str, float] = {}
        for page_no in sorted(by_page):
            page_atoms, page_confidence = self._order_page(by_page[page_no])
            ordered.extend(page_atoms)
            confidence.update(page_confidence)

        caption_parent = self._attach_captions(ordered)
        ordered = self._place_captions(ordered, caption_parent)
        continuation = self._cross_page_continuations(ordered)
        return ReadingOrderResult(
            ordered_atom_ids=tuple(atom.atom_id for atom in ordered),
            excluded_atom_ids=tuple(excluded),
            continuation_of=continuation,
            caption_parent=caption_parent,
            confidence_by_atom=confidence,
        )

    def _order_page(self, atoms: list[Atom]) -> tuple[list[Atom], dict[str, float]]:
        if any(atom.locators[0].bbox is None for atom in atoms):
            ordered = sorted(atoms, key=lambda atom: atom.reading_order_hint)
            return ordered, {atom.atom_id: 1.0 for atom in ordered}
        atoms = sorted(atoms, key=lambda atom: (
            atom.locators[0].bbox[1], atom.locators[0].bbox[0], atom.reading_order_hint,
        ))
        full_width = [atom for atom in atoms if self._width(atom) >= 0.62]
        boundaries = sorted({atom.locators[0].bbox[1] for atom in full_width})
        segments: list[list[Atom]] = []
        current: list[Atom] = []
        boundary_index = 0
        for atom in atoms:
            y0 = atom.locators[0].bbox[1]
            while boundary_index < len(boundaries) and y0 > boundaries[boundary_index] + 0.02:
                if current:
                    segments.append(current)
                    current = []
                boundary_index += 1
            current.append(atom)
        if current:
            segments.append(current)

        result: list[Atom] = []
        confidence: dict[str, float] = {}
        for segment in segments:
            narrow = [atom for atom in segment if self._width(atom) < 0.62]
            centers = sorted(self._center_x(atom) for atom in narrow)
            split = self._largest_column_gap(centers)
            if split is None or len(narrow) < 2:
                ordered = sorted(segment, key=lambda atom: (
                    atom.locators[0].bbox[1], atom.locators[0].bbox[0], atom.reading_order_hint,
                ))
                segment_confidence = 0.98
            else:
                left = [atom for atom in segment if self._center_x(atom) <= split]
                right = [atom for atom in segment if self._center_x(atom) > split]
                if not left or not right:
                    ordered = segment
                    segment_confidence = 0.75
                else:
                    ordered = sorted(left, key=self._vertical_key) + sorted(right, key=self._vertical_key)
                    segment_confidence = 0.90
            result.extend(ordered)
            confidence.update({atom.atom_id: segment_confidence for atom in ordered})
        return result, confidence

    @staticmethod
    def _largest_column_gap(centers: list[float]) -> float | None:
        if len(centers) < 2:
            return None
        gaps = [(right - left, (right + left) / 2) for left, right in zip(centers, centers[1:])]
        gap, split = max(gaps)
        return split if gap >= 0.18 else None

    @staticmethod
    def _attach_captions(atoms: list[Atom]) -> dict[str, str]:
        targets = [atom for atom in atoms if atom.atom_type in {"TABLE", "PICTURE"}]
        result: dict[str, str] = {}
        for caption in (atom for atom in atoms if atom.atom_type == "CAPTION"):
            same_page = [
                target for target in targets
            if target.locators[0].page_no == caption.locators[0].page_no
            ]
            if not same_page:
                continue
            nearest = min(same_page, key=lambda target: ReadingOrderResolver._vertical_distance(caption, target))
            if ReadingOrderResolver._vertical_distance(caption, nearest) <= 0.12:
                result[caption.atom_id] = nearest.atom_id
        return result

    @staticmethod
    def _place_captions(atoms: list[Atom], parents: dict[str, str]) -> list[Atom]:
        atom_by_id = {atom.atom_id: atom for atom in atoms}
        children: dict[str, list[Atom]] = {}
        for caption_id, parent_id in parents.items():
            children.setdefault(parent_id, []).append(atom_by_id[caption_id])
        result: list[Atom] = []
        attached = set(parents)
        for atom in atoms:
            if atom.atom_id in attached:
                continue
            result.append(atom)
            result.extend(sorted(children.get(atom.atom_id, []), key=lambda value: value.reading_order_hint))
        return result

    @staticmethod
    def _cross_page_continuations(atoms: list[Atom]) -> dict[str, str]:
        by_page: dict[int, list[Atom]] = {}
        for atom in atoms:
            if atom.locators[0].page_no is not None:
                by_page.setdefault(atom.locators[0].page_no, []).append(atom)
        result: dict[str, str] = {}
        pages = sorted(by_page)
        for current_page, next_page in zip(pages, pages[1:]):
            if next_page != current_page + 1:
                continue
            previous, following = by_page[current_page][-1], by_page[next_page][0]
            if previous.atom_type not in {"TEXT", "LIST_ITEM"} or following.atom_type != previous.atom_type:
                continue
            previous_text = previous.content_text.rstrip()
            if previous_text and not re.search(r"[。！？.!?:：；;]$", previous_text):
                if abs(previous.locators[-1].bbox[0] - following.locators[0].bbox[0]) <= 0.12:
                    result[following.atom_id] = previous.atom_id
        return result

    @staticmethod
    def _vertical_key(atom: Atom) -> tuple[float, float, int]:
        bbox = atom.locators[0].bbox or (0.0, 0.0, 1.0, 1.0)
        return bbox[1], bbox[0], atom.reading_order_hint

    @staticmethod
    def _width(atom: Atom) -> float:
        bbox = atom.locators[0].bbox or (0.0, 0.0, 1.0, 1.0)
        return bbox[2] - bbox[0]

    @staticmethod
    def _center_x(atom: Atom) -> float:
        bbox = atom.locators[0].bbox or (0.0, 0.0, 1.0, 1.0)
        return (bbox[0] + bbox[2]) / 2

    @staticmethod
    def _vertical_distance(first: Atom, second: Atom) -> float:
        a, b = first.locators[0].bbox, second.locators[0].bbox
        if a is None or b is None:
            return 1.0
        return max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
