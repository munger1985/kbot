"""Versioned intermediate representations for the KC parser pipeline."""

from .contracts import (
    EVIDENCE_TYPES,
    build_evidence_key,
    build_output_fingerprint,
    canonical_json_hash,
    evidence_fingerprint,
    validate_artifact_manifest,
    validate_quality_report,
    validate_locator,
    validate_source_spans,
)
from .spreadsheet_artifact import build_spreadsheet_artifact
from .ir import (
    ATOM_TYPES,
    STRUCTURE_NODE_TYPES,
    Atom,
    AtomIr,
    AtomLocator,
    HeadingDecision,
    IrValidationError,
    PageGeometry,
    StructureIr,
    StructureNode,
)

__all__ = [
    "ATOM_TYPES",
    "EVIDENCE_TYPES",
    "STRUCTURE_NODE_TYPES",
    "Atom",
    "AtomIr",
    "AtomLocator",
    "HeadingDecision",
    "IrValidationError",
    "PageGeometry",
    "StructureIr",
    "StructureNode",
    "canonical_json_hash",
    "build_evidence_key",
    "build_output_fingerprint",
    "evidence_fingerprint",
    "validate_artifact_manifest",
    "validate_quality_report",
    "validate_locator",
    "validate_source_spans",
    "build_spreadsheet_artifact",
]
