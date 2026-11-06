"""历史证据读取：短期检索过期不重搜，撤权/撤回与 pin 期限仍生效。"""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(path) for path in ROOT.glob("packages/*/src")] + [str(path) for path in ROOT.glob("services/*/src")]
sys.path[:0] = [str(ROOT / "tests/unit/knowledge_ingestion"), str(ROOT / "tests/unit/knowledge_algorithms")]
import test_app_knowledge_retrieval as support
from knowledge_algorithms.contracts.errors import AlgorithmError
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_retrieval_app.domain.knowledge_retrieval import RetrievalAccess


class HistoricalContentTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = support.AppRetrievalTests.asyncSetUp
    asyncTearDown = support.AppRetrievalTests.asyncTearDown

    async def prepare_history(self):
        fixture = self.fixture
        access = RetrievalAccess(fixture.scope.domain_id, fixture.scope.tenant_id, fixture.scope.subject_ref,
            run_ref="historical-run", spec_hash="frozen-spec", delegated_scope_hash=None)
        scope = await self.service.create_scope(access, resource_ref="resource-1")
        access = replace(access, delegated_scope_hash=scope.scope_hash)
        receipt = await self.service.retrieve(access, scope=scope, profile_ref="baseline",
            query=QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="图像检索"))
        key = receipt.evidence_set.primary_hits[0].evidence_key
        pins = await self.service.retain_evidence(access, retrieval_ref=receipt.retrieval_ref,
            keys=(key,), expires_at=int(fixture.now.timestamp()) + 3600)
        fixture.now += timedelta(seconds=1000)
        return access, receipt, key, pins

    async def test_expired_scope_history_reads_exact_pinned_bytes_without_new_search(self):
        access, receipt, key, _ = await self.prepare_history()
        with self.assertRaises(AlgorithmError):
            await self.service.stream_content(access, retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)
        item, chunks = await self.service.stream_historical_content(access,
            retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)
        raw = b"".join([chunk async for chunk in chunks])
        self.assertEqual(len(raw), item.byte_count)
        self.assertIn("图像检索", raw.decode())
        for other in (replace(access, run_ref="another"), replace(access, spec_hash="another"),
                      replace(access, delegated_scope_hash="another")):
            with self.assertRaises(AlgorithmError):
                await self.service.stream_historical_content(other,
                    retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)

    async def test_release_and_expiry_do_not_grant_history(self):
        access, receipt, key, pins = await self.prepare_history()
        await self.service.release_evidence(access, pin_ids=pins)
        with self.assertRaises(AlgorithmError):
            await self.service.stream_historical_content(access,
                retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)

    async def test_withdrawal_blocks_already_open_historical_stream(self):
        access, receipt, key, _ = await self.prepare_history()
        _, chunks = await self.service.stream_historical_content(access,
            retrieval_ref=receipt.retrieval_ref, stable_key=key.stable_key)
        async with self.uow_factory() as uow:
            records = await uow.knowledge_retrieval.version_records(access, (self.fixture.version,))
        await self.service.withdraw(access, resource_ref="resource-1", version_id=self.fixture.version,
                                    expected_revision=records[0]["version_revision"])
        with self.assertRaises(AlgorithmError):
            _ = b"".join([chunk async for chunk in chunks])


if __name__ == "__main__":
    unittest.main()
