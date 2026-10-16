import unittest
from types import SimpleNamespace
from uuid import UUID

from knowledge_core.domain.parse_settings import (
    normalize_collection_parse_policy,
)
from knowledge_core.workers.parser.visual_evidence import (
    build_visual_page_evidence,
)


class ParseSettingsTest(unittest.TestCase):
    def test_text_disables_ocr_and_all_image_generation(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "TEXT",
        })

        self.assertFalse(policy["do_ocr"])
        self.assertFalse(policy["force_full_page_ocr"])
        self.assertFalse(policy["extract_page_images"])
        self.assertFalse(policy["extract_picture_images"])

    def test_auto_uses_only_object_level_images(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "AUTO",
        })

        self.assertTrue(policy["do_ocr"])
        self.assertFalse(policy["force_full_page_ocr"])
        self.assertFalse(policy["extract_page_images"])
        self.assertTrue(policy["extract_picture_images"])

    def test_visual_baseline_does_not_generate_page_images(self):
        policy = normalize_collection_parse_policy({
            "parse_strategy": "VISUAL",
        })

        self.assertTrue(policy["do_ocr"])
        self.assertFalse(policy["extract_page_images"])

    def test_hybrid_is_rejected_and_image_switch_cannot_override_mode(self):
        with self.assertRaises(ValueError):
            normalize_collection_parse_policy({"parse_strategy": "HYBRID"})
        policy = normalize_collection_parse_policy({
            "parse_strategy": "AUTO",
            "extract_page_images": True,
        })
        self.assertFalse(policy["extract_page_images"])

    def test_visual_page_evidence_is_split_at_token_limit(self):
        items = build_visual_page_evidence(
            parse_view_id=UUID(int=1),
            enrichment=SimpleNamespace(page_results=[SimpleNamespace(
                page_no=1,
                markdown="字" * 1801,
                served_model_name="vlm-a",
                confidence=0.82,
            )]),
        )

        self.assertEqual(3, len(items))
        self.assertLessEqual(max(item["token_count"] for item in items), 900)


if __name__ == "__main__":
    unittest.main()
