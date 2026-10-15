import unittest
from datetime import UTC, datetime
from uuid import UUID

from knowledge_core.application.scope import KnowledgeCoreScopeService, KnowledgeScopeError
from platform_core.contracts import KnowledgeScopeSnapshotV1


COLLECTION_1 = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a21")
COLLECTION_2 = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a22")
REVISION_1 = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a31")


class _Collection:
    def __init__(self, status="ACTIVE"):
        self.status = status


class _Collections:
    async def get_by_id_scope(self, *, domain_id, collection_id):
        if (
            domain_id != 7
            or collection_id not in {COLLECTION_1, COLLECTION_2}
        ):
            return None
        return _Collection(
            "DISABLED" if collection_id == COLLECTION_2 else "ACTIVE"
        )


class _Bindings:
    async def get_by_consumer_collection(self, *, consumer_type, consumer_id, collection_id):
        if consumer_type == "AGENT" and consumer_id == UUID(
            "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11"
        ) and collection_id in {COLLECTION_1, COLLECTION_2}:
            return type("Binding", (), {"status": "ACTIVE"})()
        return None


class _Revisions:
    async def list_current_scope(self, *, collection_ids):
        return [REVISION_1] if collection_ids == [COLLECTION_1] else []


class _Uow:
    collections = _Collections()
    bindings = _Bindings()
    revisions = _Revisions()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class ScopeTest(unittest.IsolatedAsyncioTestCase):
    def test_scope_hash_survives_json_round_trip(self):
        payload = {
            "schema_version": "1.0",
            "owner_app_id": "knowledge_retrieval",
            "domain_id": 7,
            "consumer_agent_id": "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
            "consumer_agent_version_id": (
                "019c03b5-4b88-7ab2-8c19-7b6ea34f2a12"
            ),
            "collection_ids": [str(COLLECTION_1)],
            "bundle_revision_ids": [str(REVISION_1)],
            "security_level": 3,
            "business_context": {},
            "business_resource_refs": [],
            "policy_version": "test.v1",
            "issued_at": datetime.now(UTC).isoformat(),
        }
        payload["scope_hash"] = KnowledgeScopeSnapshotV1.calculate_hash(
            payload
        )
        first = KnowledgeScopeSnapshotV1.model_validate(payload)
        second = KnowledgeScopeSnapshotV1.model_validate(
            first.model_dump(mode="json")
        )
        self.assertEqual(first.scope_hash, second.scope_hash)

    async def test_disabled_collection_is_skipped_but_binding_is_retained(self):
        service = KnowledgeCoreScopeService(uow_factory=_Uow)
        self.assertEqual(await service.resolve_agent_collections(
            domain_id=7,
            agent_id=UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a11"),
            collection_ids=(COLLECTION_1, COLLECTION_2),
        ), (COLLECTION_1,))

    async def test_unbound_collection_is_rejected(self):
        service = KnowledgeCoreScopeService(uow_factory=_Uow)
        with self.assertRaises(KnowledgeScopeError):
            await service.resolve_agent_collections(
                domain_id=7,
                agent_id=UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a12"),
                collection_ids=(COLLECTION_1,),
            )

    async def test_resolve_current_revision_scope_is_agent_bound(self):
        service = KnowledgeCoreScopeService(uow_factory=_Uow)
        collections, revisions = await service.resolve_current_revision_scope(
            domain_id=7,
            agent_id=UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a11"),
            collection_ids=(COLLECTION_1,),
        )
        self.assertEqual(collections, (COLLECTION_1,))
        self.assertEqual(revisions, (REVISION_1,))


if __name__ == "__main__":
    unittest.main()
