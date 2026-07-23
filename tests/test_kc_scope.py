import unittest

from knowledge_core.application.scope import KnowledgeCoreScopeService, KnowledgeScopeError


class _Collection:
    def __init__(self, status="ACTIVE"):
        self.status = status


class _Collections:
    async def get_by_id_scope(self, *, app_id, domain_id, collection_id):
        if app_id != 100 or domain_id != 7 or collection_id not in {1, 2}:
            return None
        return _Collection("DISABLED" if collection_id == 2 else "ACTIVE")


class _Bindings:
    async def get_by_consumer_collection(self, *, consumer_type, consumer_id, collection_id):
        if consumer_type == "AGENT" and consumer_id == "019c03b5-4b88-7ab2-8c19-7b6ea34f2a11" and collection_id in {1, 2}:
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
        service = KnowledgeCoreScopeService(app_id=100, uow_factory=_Uow)
        self.assertEqual(await service.resolve_agent_collections(
            domain_id=7,
            agent_id="019c03b5-4b88-7ab2-8c19-7b6ea34f2a11",
            collection_ids=(1, 2),
        ), (1,))

    async def test_unbound_collection_is_rejected(self):
        service = KnowledgeCoreScopeService(app_id=100, uow_factory=_Uow)
        with self.assertRaises(KnowledgeScopeError):
            await service.resolve_agent_collections(
                domain_id=7,
                agent_id="019c03b5-4b88-7ab2-8c19-7b6ea34f2a12",
                collection_ids=(1,),
            )


if __name__ == "__main__":
    unittest.main()
