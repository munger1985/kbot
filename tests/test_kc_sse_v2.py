import unittest

from knowledge_core.application.grounding import GroundingResult
from knowledge_core.application.sse_v2 import build_grounded_sse_payload


class SseV2Test(unittest.TestCase):
    def test_payload_uses_v2_names_only(self):
        result = GroundingResult(
            answer_markdown="a", claims=[], used_citation_labels=[], citations=[],
            doc_results_v2=[], grounding_status="INSUFFICIENT",
        )
        payload = build_grounded_sse_payload(answer_markdown="a", result=result)
        self.assertIn("citations_v2", payload)
        self.assertIn("doc_results_v2", payload)
        self.assertNotIn("doc_results", payload)


if __name__ == "__main__":
    unittest.main()
