import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from PIL import Image

from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.client import ParseTask
from knowledge_core.workers.parser.worker import KcParserWorker
from knowledge_core.workers.parser.visual_enricher import (
    VisualEnrichmentResult,
    VisualPageResult,
)
from knowledge_core.workers.job_wait import AdaptiveJobWait
from platform_core.dictionary import ModelCategory
from test_kc_docling_atom_normalizer import FakeDocument, item


class FakeClient:
    def __init__(self):
        self.artifacts = []
        self.evidences = []
        self.completed = None
        self.failed = None
        self.visual_assets = []
        self.visual_completed = False

    async def heartbeat(self, task, *, lease_seconds):
        return None

    async def upload_artifact(self, task, **values):
        self.artifacts.append(values)
        return {
            "uri": f"memory://{values['name']}", "sha256": values["sha256"],
            "schema": values["schema"], "generator": values["generator"],
        }

    async def submit_evidence(self, task, items):
        self.evidences.extend(items)
        return len(items)

    async def complete(self, task, **values):
        self.completed = values
        return len(self.evidences)

    async def submit_visual_assets(self, task, items):
        self.visual_assets.extend(items)
        return len(items)

    async def complete_visual(self, task):
        self.visual_completed = True
        return len(self.evidences)

    async def fail(self, task, **values):
        self.failed = values


class FakeConverter:
    def __init__(self, document):
        self.document = document
        self.received_path = None
        self.received_options = None

    async def convert(self, *, source_path, **kwargs):
        self.received_path = source_path
        self.received_options = kwargs
        return self.document


class FakeVisualEnricher:
    async def enrich(self, document, **kwargs):
        return VisualEnrichmentResult(
            strategy=kwargs["strategy"],
            picture_description_count=0,
            page_assessments=(),
            page_results=(VisualPageResult(
                page_no=1,
                markdown="# 架构\n\n> 服务通过消息队列连接",
                served_model_name=kwargs["served_model_name"],
                confidence=0.82,
                replace_docling=False,
                selection_reasons=(),
            ),),
            failed_page_numbers=(),
        )


class FakeModelConfigClient:
    def __init__(
        self, *, category=int(ModelCategory.VLM), served_model_name="vlm-a",
    ):
        self.category = category
        self.served_model_name = served_model_name
        self.model_ids = []

    async def get_model(self, model_id):
        self.model_ids.append(model_id)
        return {
            "category": self.category,
            "status": "ACTIVE",
            "served_model_name": self.served_model_name,
        }


class ParserWorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_runs_full_claim_result_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "content-addressed-object"
            source.write_bytes(b"fake pdf")
            task = ParseTask(
                job_id=1, job_type="PARSE",
                lease_owner="worker", lease_until="later",
                input_fingerprint=hashlib.sha256(b"fake pdf").hexdigest(),
                source_fingerprint=hashlib.sha256(b"fake pdf").hexdigest(),
                document_version_id=10, parse_view_id=20,
                source_read_url=str(source), detected_mime_type="application/pdf",
                view_kind="TEXT", parse_config_fingerprint="a" * 64,
                policy_snapshot={},
            )
            client = FakeClient()
            converter = FakeConverter(FakeDocument([
                item(source_ref="#/texts/0", label="section_header", text="1 Overview"),
                item(source_ref="#/texts/1", label="text", text="Install before startup."),
            ]))
            worker = KcParserWorker(
                client=client, converter=converter,
                pipeline=KcParsingPipeline(parser_version="test/1"),
                worker_id="worker", lease_seconds=600,
                job_wait=AdaptiveJobWait(
                    listener=None,
                    notification_timeout_seconds=30,
                    fallback_min_seconds=1,
                    fallback_max_seconds=30,
                    fallback_multiplier=2,
                    jitter_ratio=0,
                ),
                evidence_batch_size=2,
            )

            await worker._process(task)

            self.assertEqual(len(client.artifacts), 4)
            self.assertTrue(client.evidences)
            self.assertIsNotNone(client.completed)
            self.assertIsNone(client.failed)
            self.assertEqual(converter.received_path.suffix, ".pdf")
            self.assertTrue(converter.received_options["generate_page_images"])
            self.assertTrue(converter.received_options["generate_picture_images"])

    async def test_visual_job_generates_pages_only_and_completes_additively(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "content-addressed-object"
            source.write_bytes(b"fake pdf")
            model_id = UUID("00000000-0000-0000-0000-000000000005")
            task = ParseTask(
                job_id=2, job_type="VISUAL_ENRICH",
                lease_owner="worker", lease_until="later",
                input_fingerprint="b" * 64,
                source_fingerprint=hashlib.sha256(b"fake pdf").hexdigest(),
                document_version_id=UUID(int=10),
                parse_view_id=UUID(int=20),
                source_read_url=str(source),
                detected_mime_type="application/pdf",
                view_kind="VISUAL",
                parse_config_fingerprint="a" * 64,
                policy_snapshot={
                    "parse_strategy": "VISUAL",
                    "models": {"parser_vlm": str(model_id)},
                },
            )
            document = SimpleNamespace(
                pages={1: SimpleNamespace(
                    page_no=1,
                    image=SimpleNamespace(
                        pil_image=Image.new("RGB", (256, 256), "white")
                    ),
                )},
                pictures=[],
            )
            client = FakeClient()
            converter = FakeConverter(document)
            worker = KcParserWorker(
                client=client,
                converter=converter,
                pipeline=KcParsingPipeline(parser_version="test/1"),
                worker_id="worker",
                lease_seconds=600,
                job_wait=AdaptiveJobWait(
                    listener=None,
                    notification_timeout_seconds=30,
                    fallback_min_seconds=1,
                    fallback_max_seconds=30,
                    fallback_multiplier=2,
                    jitter_ratio=0,
                ),
                evidence_batch_size=2,
                visual_enricher=FakeVisualEnricher(),
                model_config_clients={
                    int(ModelCategory.VLM): FakeModelConfigClient()
                },
            )

            await worker._process(task)

            self.assertTrue(converter.received_options["generate_page_images"])
            self.assertFalse(converter.received_options["generate_picture_images"])
            self.assertTrue(client.evidences)
            self.assertTrue(client.visual_assets)
            self.assertTrue(client.visual_completed)
            self.assertIsNone(client.failed)

    async def test_model_roles_use_their_category_catalog_clients(self):
        vlm_model_id = UUID("00000000-0000-4000-8000-000000000005")
        ocr_model_id = UUID("00000000-0000-4000-8000-000000000006")
        vlm_catalog = FakeModelConfigClient(
            category=int(ModelCategory.VLM), served_model_name="vlm-a"
        )
        ocr_catalog = FakeModelConfigClient(
            category=int(ModelCategory.OCR), served_model_name="ocr-a"
        )
        worker = KcParserWorker(
            client=FakeClient(),
            converter=FakeConverter(FakeDocument([])),
            pipeline=KcParsingPipeline(parser_version="test/1"),
            worker_id="worker",
            lease_seconds=600,
            job_wait=AdaptiveJobWait(
                listener=None,
                notification_timeout_seconds=30,
                fallback_min_seconds=1,
                fallback_max_seconds=30,
                fallback_multiplier=2,
                jitter_ratio=0,
            ),
            evidence_batch_size=2,
            model_config_clients={
                int(ModelCategory.VLM): vlm_catalog,
                int(ModelCategory.OCR): ocr_catalog,
            },
        )
        policy = {"models": {
            "parser_vlm": str(vlm_model_id),
            "parser_ocr": str(ocr_model_id),
        }}

        self.assertEqual("vlm-a", await worker._resolve_vlm_model(policy))
        self.assertEqual(ocr_model_id, await worker._resolve_ocr_model(policy))
        self.assertEqual([vlm_model_id], vlm_catalog.model_ids)
        self.assertEqual([ocr_model_id], ocr_catalog.model_ids)


if __name__ == "__main__":
    unittest.main()
