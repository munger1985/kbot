"""K0 工件一致性与 K1 资源/故障的退出门禁。"""

import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/knowledge_algorithms/src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from knowledge_algorithm_fixtures import (FakeBackend, FakeModel, allowed, node, original, parse_profile,
    retrieval_profile, run_for, source, spec)
from knowledge_algorithms.contracts.content import ContentKind, Page, ProvenanceKind
from knowledge_algorithms.contracts.errors import AlgorithmError, ErrorCode
from knowledge_algorithms.contracts.identity import Locator
from knowledge_algorithms.contracts.query import QueryInput, QueryMode
from knowledge_algorithms.contracts.retrieval import ChannelSpec
from knowledge_algorithms.parsing.contracts import ExtractedRegion, ExtractionBatch
from knowledge_algorithms.parsing.engine import ParseEngine
from knowledge_algorithms.parsing.plain_text import PlainTextAdapter
from knowledge_algorithms.projections.contracts import Representation
from knowledge_algorithms.retrieval.engine import RetrievalEngine


class InventoryGateTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "tests/contract/knowledge_algorithms"
        self.inventory = json.loads((base / "inventory-v1.json").read_text())
        self.experiments = json.loads((base / "experiments-v1.json").read_text())

    def test_fixed_physical_maps_are_disjoint_and_bounded(self):
        self.assertEqual(set(self.inventory["apps"]), {"knowledge_retrieval", "aiops"})
        oracle, pg = [], []
        for owner, app in self.inventory["apps"].items():
            self.assertEqual(app["binding"]["binding_ref"], owner)
            self.assertEqual(len(app["tables"]), 18)
            self.assertIn("knowledge_scope_snapshot", app["tables"])
            self.assertIn("knowledge_scope_member", app["tables"])
            for logical, table in app["tables"].items():
                self.assertEqual(table["oracle"], app["oracle_prefix"] + logical.upper())
                self.assertEqual(table["postgresql"], app["postgresql_prefix"] + logical)
                self.assertLessEqual(len(table["oracle"].encode()), 128)
                self.assertLessEqual(len(table["postgresql"].encode()), 63)
                self.assertTrue(table["columns"] and table["nonunique_indexes"])
                self.assertTrue(all(set(i) <= set(table["columns"] + ["domain_id"]) for i in table["nonunique_indexes"]))
                oracle.append(table["oracle"])
                pg.append(table["postgresql"])
            self.assertTrue((ROOT / app["fixture"]).is_file())
        self.assertEqual(len(oracle), len(set(oracle)))
        self.assertEqual(len(pg), len(set(pg)))

    def test_namespaces_cache_and_scope_never_share_owners(self):
        apps = list(self.inventory["apps"].values())
        for field in ("index_namespace", "temporary_root", "cache_namespace"):
            self.assertNotEqual(apps[0][field], apps[1][field])
        self.assertFalse(set(apps[0]["object_roots"].values()) & set(apps[1]["object_roots"].values()))
        self.assertTrue(all(a["scope_persistence"] == "OWN_SCOPE_SNAPSHOT_AND_MEMBER" for a in apps))

    def test_data_authority_is_explicitly_unapproved(self):
        self.assertFalse(self.inventory["real_database_inspected"])
        self.assertEqual(self.inventory["data_actions_approved"], [])
        self.assertEqual(len(self.inventory["data_authority"]), 4)
        self.assertFalse(self.inventory["constraints"]["unique_indexes"])
        self.assertFalse(self.inventory["constraints"]["runtime_schema_probes"])

    def test_existing_source_mappings_are_evidence_backed(self):
        for mapping in self.inventory["existing_mappings"]:
            if "source" in mapping:
                body = (ROOT / mapping["source"]).read_text()
                for field in mapping.get("preserve", []):
                    self.assertIn(field, body)
        self.assertTrue(any("asset_version_id" in m.get("preserve", []) for m in self.inventory["existing_mappings"]))

    def test_each_app_has_separate_golden_permission_fault_and_holdout(self):
        owners, revisions = set(), set()
        for app in self.inventory["apps"].values():
            data = json.loads((ROOT / app["fixture"]).read_text())
            owners.add(data["owner"])
            revisions.add(data["dataset_revision"])
            self.assertFalse(data["real_data"])
            self.assertEqual({c["split"] for c in data["cases"]}, {"development", "validation", "contract_holdout"})
            kinds = {c["kind"] for c in data["cases"]}
            self.assertTrue({"TEXT_TO_TEXT", "TEXT_TO_IMAGE", "IMAGE_TO_IMAGE", "FINAL_REVOCATION", "CROSS_APP",
                             "OPTIONAL_FAILURE", "REQUIRED_FAILURE"} <= kinds)
        self.assertEqual(len(owners), 2)
        self.assertEqual(len(revisions), 2)

    def test_only_fake_models_admitted_and_budgets_frozen(self):
        self.assertFalse(self.experiments["production_quality_claim"])
        self.assertEqual(self.experiments["tuning_trials"], 0)
        self.assertTrue(self.experiments["runner_budget"]["max_traced_memory_bytes"] > 0)
        self.assertFalse(any("rerank" in role.lower() for app in self.inventory["apps"].values() for role in app["model_roles"]))


class FailureGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_app_golden_runner(self):
        path = ROOT / "tests/evaluation/knowledge_algorithms/run_baseline.py"
        specification = importlib.util.spec_from_file_location("knowledge_algorithm_baseline_runner", path)
        runner = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(runner)
        report = await runner.evaluate()
        frozen = json.loads((ROOT / "tests/evaluation/knowledge_algorithms/results-v1.json").read_text())
        frozen.pop("peak_traced_memory_bytes")
        self.assertEqual(report, frozen)
        self.assertEqual(len(report["results"]), 20)
        self.assertEqual(report["gate"], "G_A")
        self.assertFalse(report["production_quality_claim"])

    async def test_channel_timeout_retains_other_valid_results(self):
        backend = FakeBackend()
        p = backend.add("p")
        backend.rows = {"lexical": ((p, 1),)}
        async def blocked(*args):
            await asyncio.Event().wait()
        backend.search_dense = blocked
        dense = ChannelSpec("dense", "dense", "DENSE", "TEXT_TO_TEXT", 1, 10, .01, False, spec())
        profile = retrieval_profile(channels=retrieval_profile().channels + (dense,))
        result = await RetrievalEngine(backend, FakeModel()).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="x"),
            backend.scope, profile)
        self.assertEqual(result.primary_hits[0].evidence_key, p)
        self.assertEqual(result.channel_status[1].reason, "TIMEOUT")

    async def test_total_timeout_not_no_match(self):
        backend = FakeBackend()
        async def blocked(*args):
            await asyncio.Event().wait()
        backend.snapshot = blocked
        with self.assertRaises(AlgorithmError) as error:
            await RetrievalEngine(backend, None).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="x"),
                backend.scope, retrieval_profile(timeout_seconds=.01))
        self.assertEqual(error.exception.code, ErrorCode.TIMEOUT)

    async def test_shard_budget_failure_not_healthy_empty(self):
        backend = FakeBackend()
        original_search = backend.search_lexical
        async def duplicate(*args):
            shards = await original_search(*args)
            return shards * 2
        backend.search_lexical = duplicate
        result = await RetrievalEngine(backend, None).retrieve(QueryInput(QueryMode.RELEVANCE, 1, ("TEXT",), text="x"),
            backend.scope, retrieval_profile(max_shards=1))
        self.assertEqual(result.status, "FAILED")
        self.assertEqual(result.channel_status[0].reason, "RESOURCE_LIMIT")

    async def parse_bad(self, profile=None, broken=None):
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                item = node(text="\ufffd\ufffd")
                if broken is not None:
                    item = broken(item)
                yield ExtractionBatch((), (ExtractedRegion("bad", (item,), .01),), ())
        reader, profile = source(), profile or parse_profile()
        return await ParseEngine(FakeBackend().binding, (Adapter(),)).parse(reader, profile, run_for(reader, profile))

    async def test_failed_quality_not_published_and_original_retained(self):
        parsed = await self.parse_bad(parse_profile(allow_partial=False))
        self.assertEqual(parsed.quality_status, "FAILED")
        self.assertEqual(parsed.failed_regions, ("bad",))
        self.assertEqual(parsed.alternatives[0].nodes[0].text, "\ufffd\ufffd")
        self.assertEqual(parsed.alternatives[0].reason, "ORIGINAL")

    async def test_missing_figure_resolution_survives_memory_result(self):
        reader, profile = source("如图 9-9 所示。"), parse_profile()
        parsed = await ParseEngine(FakeBackend().binding, (PlainTextAdapter(),)).parse(reader, profile, run_for(reader, profile))
        self.assertEqual(parsed.resolutions[0].status.value, "MISSING_TARGET")
        self.assertIsNone(parsed.resolutions[0].target_key)

    async def test_adapter_identity_version_and_locator_rejected(self):
        for broken in (lambda n: replace(n, key=original("noncanonical")),
                       lambda n: replace(n, key=replace(n.key, document_version_ref="other")),
                       lambda n: replace(n, locators=(Locator(page_no=2),))):
            with self.assertRaises(AlgorithmError):
                await self.parse_bad(broken=broken)

    async def test_page_count_budget_is_resource_limit(self):
        class Adapter(PlainTextAdapter):
            async def extract(self, source, profile, run):
                async for _ in source.chunks():
                    pass
                yield ExtractionBatch((Page("page-1", 1, 100, 100, 0), Page("page-2", 2, 100, 100, 0)),
                    (ExtractedRegion("a", (node(),), 1.0),), ())
        reader, profile = source(), parse_profile(max_pages=1)
        with self.assertRaises(AlgorithmError) as error:
            await ParseEngine(FakeBackend().binding, (Adapter(),)).parse(reader, profile, run_for(reader, profile))
        self.assertEqual(error.exception.code, ErrorCode.RESOURCE_LIMIT)

    def test_model_description_not_admitted_as_repair(self):
        with self.assertRaises(AlgorithmError):
            parse_profile(repair_strategies=("MODEL_DESCRIPTION",))

    def test_cosine_query_zero_vector_rejected(self):
        specification = replace(spec(), similarity="COSINE")
        with self.assertRaises(AlgorithmError):
            Representation(specification.space_id, vector=(0.0, 0.0)).validate(specification, 100)


if __name__ == "__main__":
    unittest.main()
