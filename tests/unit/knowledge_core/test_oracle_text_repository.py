"""Knowledge Core Oracle Text SQL 编译回归测试。"""

import unittest

from sqlalchemy import Float
from sqlalchemy.dialects import oracle

from knowledge_core.application.evidence_retrieval import EvidenceScope
from knowledge_core.repositories.discovery_repo import DiscoveryRepository
from knowledge_core.repositories.ingestion_repo import EvidenceRepository
from knowledge_core.repositories.oracle_text_query import (
    build_oracle_text_query,
)
from platform_core.identity import uuid7


class _Result:
    def all(self):
        return []


class _Session:
    def __init__(self):
        self.statements = []

    async def execute(self, statement):
        self.statements.append(statement)
        return _Result()


class OracleTextRepositoryTest(unittest.IsolatedAsyncioTestCase):
    def test_chinese_question_is_rewritten_to_content_terms(self) -> None:
        self.assertEqual(
            "{员工} ACCUM {套餐}",
            build_oracle_text_query("员工有哪些套餐"),
        )

    async def test_oracle_text_labels_are_numeric_literals(self) -> None:
        discovery_session = _Session()
        await DiscoveryRepository(discovery_session).search_text(
            collection_id=uuid7(),
            query="员工套餐",
        )
        evidence_session = _Session()
        await EvidenceRepository(evidence_session).search_text(
            scope=EvidenceScope(
                collection_id=uuid7(),
                bundle_id=uuid7(),
                bundle_revision_id=uuid7(),
            ),
            query="员工套餐",
        )

        for statement in (
            discovery_session.statements[0],
            evidence_session.statements[0],
        ):
            sql = str(statement.compile(dialect=oracle.dialect())).upper()
            with self.subTest(sql=sql):
                self.assertIn("SCORE(1)", sql)
                self.assertIn(", 1) >", sql)
                self.assertNotIn("SCORE(:", sql)

    async def test_vector_distance_uses_float_result_type(self) -> None:
        discovery_session = _Session()
        await DiscoveryRepository(discovery_session).search_vector(
            collection_id=uuid7(),
            vector=[0.1, 0.2],
        )
        evidence_session = _Session()
        await EvidenceRepository(evidence_session).search_vector(
            scope=EvidenceScope(
                collection_id=uuid7(),
                bundle_id=uuid7(),
                bundle_revision_id=uuid7(),
            ),
            vector=[0.1, 0.2],
        )
        for statement in (
            discovery_session.statements[0],
            evidence_session.statements[0],
        ):
            selected = list(statement.selected_columns)
            distance = next(
                column for column in selected
                if column.key == "distance"
            )
            self.assertIsInstance(distance.type, Float)

    async def test_all_retrieval_channels_apply_user_security_level(self) -> None:
        discovery_session = _Session()
        discovery = DiscoveryRepository(discovery_session)
        await discovery.search_text(
            collection_id=uuid7(), query="员工套餐", max_security_level=2
        )
        await discovery.search_vector(
            collection_id=uuid7(), vector=[0.1, 0.2], max_security_level=2
        )
        evidence_session = _Session()
        evidence = EvidenceRepository(evidence_session)
        scope = EvidenceScope(
            collection_id=uuid7(),
            bundle_id=uuid7(),
            bundle_revision_id=uuid7(),
        )
        await evidence.search_text(
            scope=scope, query="员工套餐", max_security_level=2
        )
        await evidence.search_vector(
            scope=scope, vector=[0.1, 0.2], max_security_level=2
        )

        statements = (
            discovery_session.statements[0],
            discovery_session.statements[-1],
            evidence_session.statements[0],
            evidence_session.statements[-1],
        )
        for statement in statements:
            compiled = statement.compile(dialect=oracle.dialect())
            sql = str(compiled).upper()
            with self.subTest(sql=sql):
                self.assertIn("SECURITY_LEVEL <=", sql)
                self.assertIn(2, compiled.params.values())


if __name__ == "__main__":
    unittest.main()
