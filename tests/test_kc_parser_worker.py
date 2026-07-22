import hashlib
import tempfile
import unittest
from pathlib import Path

from knowledge_core.parsing.pipeline import KcParsingPipeline
from knowledge_core.workers.parser.client import ParseTask
from knowledge_core.workers.parser.worker import KcParserWorker
from tests.test_kc_docling_atom_normalizer import FakeDocument, item


class FakeClient:
    def __init__(self):
        self.artifacts = []
        self.evidences = []
        self.completed = None
        self.failed = None

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

    async def fail(self, task, **values):
        self.failed = values


class FakeConverter:
    def __init__(self, document):
        self.document = document
        self.received_path = None

    async def convert(self, *, source_path, **kwargs):
        self.received_path = source_path
        return self.document


class ParserWorkerTest(unittest.IsolatedAsyncioTestCase):
    async def test_worker_runs_full_claim_result_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "content-addressed-object"
            source.write_bytes(b"fake pdf")
            task = ParseTask(
                job_id=1, lease_owner="worker", lease_until="later",
                input_fingerprint=hashlib.sha256(b"fake pdf").hexdigest(),
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
                poll_interval=1, evidence_batch_size=2,
            )

            await worker._process(task)

            self.assertEqual(len(client.artifacts), 4)
            self.assertTrue(client.evidences)
            self.assertIsNotNone(client.completed)
            self.assertIsNone(client.failed)
            self.assertEqual(converter.received_path.suffix, ".pdf")


if __name__ == "__main__":
    unittest.main()
