import unittest
from types import SimpleNamespace

from knowledge_core.workers.parser.visual_enricher import KcVisualEnricher


class FakeModelClient:
    async def get_vlm_answer(self, model_name, image, prompt):
        return "A server topology diagram"


class VisualEnricherTest(unittest.IsolatedAsyncioTestCase):
    async def test_adds_policy_model_description_with_provenance(self):
        picture = SimpleNamespace(
            image=SimpleNamespace(pil_image=object()), annotations=[],
        )
        document = SimpleNamespace(pictures=[picture])
        enricher = KcVisualEnricher(client_factory=FakeModelClient)

        count = await enricher.enrich(
            document, model_name="vlm-a", prompt="Describe visible facts",
        )

        self.assertEqual(count, 1)
        self.assertEqual(picture.annotations[0].text, "A server topology diagram")
        self.assertIn("vlm_kc_v2", picture.annotations[0].provenance)

    async def test_does_nothing_without_policy_model(self):
        document = SimpleNamespace(pictures=[])
        count = await KcVisualEnricher(client_factory=FakeModelClient).enrich(
            document, model_name=None, prompt="Describe",
        )
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
