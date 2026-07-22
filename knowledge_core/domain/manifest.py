"""Deterministic, KC-owned rendering of Bundle metadata into a Manifest document."""
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from knowledge_core.domain.intake import BundleDeclaration


@dataclass(frozen=True)
class RenderedManifest:
    content: bytes
    content_sha256: str
    mime_type: str = "text/markdown"


def render_bundle_manifest(bundle: BundleDeclaration) -> RenderedManifest:
    """Render searchable source facts without accepting caller-provided Markdown."""
    metadata = json.dumps(bundle.metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    facets = json.dumps(bundle.facet, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    lines = [
        f"# {bundle.title}",
        "", f"Source ID: {bundle.source_id}", f"Source revision: {bundle.source_revision}",
        f"Security level: {bundle.security_level}",
    ]
    if bundle.canonical_url:
        lines.append(f"Source URL: {bundle.canonical_url}")
    lines.extend(["", "## Facets", facets, "", "## Source metadata", metadata, ""])
    content = "\n".join(lines).encode("utf-8")
    return RenderedManifest(content=content, content_sha256=hashlib.sha256(content).hexdigest())
