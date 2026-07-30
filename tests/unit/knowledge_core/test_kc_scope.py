import unittest
from uuid import UUID

from knowledge_core.application.scope import KnowledgeCoreScopeService, KnowledgeScopeError


COLLECTION_1 = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a21")
COLLECTION_2 = UUID("019c03b5-4b88-7ab2-8c19-7b6ea34f2a22")


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


class _Uow:
    collections = _Collections()
    bindings = _Bindings()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


class ScopeTest(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
