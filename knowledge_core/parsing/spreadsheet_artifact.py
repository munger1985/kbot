"""Deterministic spreadsheet projection for data-query handoff.

The projection is deliberately separate from retrieval Evidence: Evidence is
human-readable and citation-oriented, while this artifact preserves typed cell
coordinates for a later Data Query service.
"""
from typing import Any


def _column_name(index: int) -> str:
    result = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def build_spreadsheet_artifact(document: Any) -> dict[str, Any] | None:
    rows: dict[str, list[dict[str, Any]]] = {}
    groups = {
        getattr(group, "self_ref", ""): getattr(group, "name", None) or getattr(group, "label", "Sheet")
        for group in getattr(document, "groups", []) or []
    }
    for item, _ in document.iterate_items():
        data = getattr(item, "data", None)
        cells = getattr(data, "table_cells", None) if data is not None else None
        if not cells:
            continue
        parent = getattr(getattr(item, "parent", None), "cref", "")
        sheet_ref = str(parent or "#sheet:unknown")
        sheet_name = str(groups.get(sheet_ref, sheet_ref.rsplit("/", 1)[-1] or "Sheet"))
        target = rows.setdefault(sheet_ref, [])
        for cell in cells:
            row = int(getattr(cell, "start_row_offset_idx", 0))
            col = int(getattr(cell, "start_col_offset_idx", 0))
            target.append({
                "address": f"{_column_name(col)}{row + 1}",
                "row": row,
                "column": col,
                "value": getattr(cell, "text", None),
                "column_header": bool(getattr(cell, "column_header", False)),
                "row_header": bool(getattr(cell, "row_header", False)),
            })
    if not rows:
        return None
    sheets = [{
        "sheet_ref": ref, "sheet_name": str(groups.get(ref, ref.rsplit("/", 1)[-1] or "Sheet")),
        "cells": sorted(cells, key=lambda cell: (cell["row"], cell["column"])),
    } for ref, cells in sorted(rows.items())]
    return {"schema": "kc-spreadsheet/v1", "sheets": sheets}
