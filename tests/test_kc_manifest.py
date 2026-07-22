"""Manifest content is deterministic and only derived from structured source facts."""
import unittest

from knowledge_core.domain.intake import BundleDeclaration
from knowledge_core.domain.manifest import render_bundle_manifest


class ManifestTest(unittest.TestCase):
    def test_rendering_is_stable_and_searchable_without_attachments(self):
        bundle = BundleDeclaration(source_id="A-1", source_revision="r1", title="Asset title", security_level=1, facet={"product": "db"}, metadata={"briefing": "useful content"})
        rendered = render_bundle_manifest(bundle)
        self.assertEqual(rendered.content_sha256, render_bundle_manifest(bundle).content_sha256)
        self.assertIn(b"Asset title", rendered.content)
        self.assertIn(b"useful content", rendered.content)


if __name__ == "__main__":
    unittest.main()
