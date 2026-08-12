import unittest

from knowledge_core.application.discovery import BundleProfileInput, MemberProfileInput, build_bundle_profile


class DiscoveryProfileTest(unittest.TestCase):
    def test_profile_is_deterministic_and_tracks_missing_members(self):
        value = BundleProfileInput(
            title="Asset A", source_system="km", source_type="asset", source_id="A-1",
            canonical_url=None, facet={"domain": "infra"},
            metadata={"author_mail": "user@example.com", "asset_solution": "AI"},
            members=(MemberProfileInput(
                external_document_id="doc-1", declared_name="brief.pdf", document_role="PRIMARY",
                mime_type="application/pdf", member_status="READY", evidence_count=2,
                section_titles=("Overview",),
            ),),
            missing_members=("broken.xlsx",),
        )
        first = build_bundle_profile(value)
        second = build_bundle_profile(value)
        self.assertEqual(first.profile_hash, second.profile_hash)
        self.assertIn('\"author_mail\":\"user@example.com\"', first.profile_text)
        self.assertIn("broken.xlsx", first.profile_text)
        self.assertEqual(first.coverage["ready_member_count"], 1)


if __name__ == "__main__":
    unittest.main()
