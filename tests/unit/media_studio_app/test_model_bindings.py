"""多媒体创作模型绑定能力边界测试。"""

import unittest

from media_studio_app.application import (
    MediaStudioApplicationError,
    ModelBindingService,
    UpsertBindingCommand,
)
from platform_core.identity import uuid7


class _Catalog:
    def __init__(self, row):
        self.row = row

    async def get_model(self, model_id):
        if str(model_id) != str(self.row["model_id"]):
            raise LookupError(model_id)
        return self.row


class _Repository:
    def __init__(self):
        self.row = None

    async def get_by_role(self, *, domain_id, role, lock=False):
        del lock
        if self.row is not None and int(self.row.domain_id) == domain_id and self.row.role == role:
            return self.row
        return None

    async def add(self, row):
        row.row_version = 1
        self.row = row

    async def list(self, *, domain_id):
        return [self.row] if self.row is not None and int(self.row.domain_id) == domain_id else []

    async def model_references(self, *, model_id):
        if self.row is not None and self.row.model_id == model_id:
            return [self.row]
        return []


class _UnitOfWork:
    def __init__(self, repository):
        self.bindings = repository
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return None

    async def commit(self):
        self.committed = True


class ModelBindingTest(unittest.IsolatedAsyncioTestCase):
    async def test_image_binding_requires_active_image_generation_model(self):
        model_id = uuid7()
        for status, category in (("DRAFT", 8), ("ACTIVE", 1)):
            with self.subTest(status=status, category=category):
                row = {
                    "model_id": str(model_id),
                    "display_name": "图片模型",
                    "served_model_name": "image-model",
                    "provider": "api_oci_image",
                    "status": status,
                    "category": category,
                    "api_key": "不能进入快照",
                }
                service = ModelBindingService(
                    uow_factory=lambda: _UnitOfWork(_Repository()),
                    catalog_client=_Catalog(row),
                )

                with self.assertRaises(MediaStudioApplicationError) as raised:
                    await service.upsert(UpsertBindingCommand(
                        domain_id=41,
                        role="IMAGE_GENERATION",
                        model_id=model_id,
                        actor_id="user-41",
                    ))

                self.assertEqual("MODEL_NOT_ACTIVE", raised.exception.code)

    async def test_active_image_generation_model_is_bound_with_safe_snapshot(self):
        model_id = uuid7()
        repository = _Repository()
        service = ModelBindingService(
            uow_factory=lambda: _UnitOfWork(repository),
            catalog_client=_Catalog({
                "model_id": str(model_id),
                "display_name": "图片模型",
                "served_model_name": "image-model",
                "provider": "api_oci_image",
                "status": "ACTIVE",
                "category": 8,
                "api_key": "不能进入快照",
            }),
        )

        result = await service.upsert(UpsertBindingCommand(
            domain_id=41,
            role="IMAGE_GENERATION",
            model_id=model_id,
            actor_id="user-41",
        ))

        self.assertTrue(result["ready"])
        self.assertEqual("IMAGE_GENERATION", result["role"])
        self.assertNotIn("api_key", repository.row.model_snapshot_json)

        references = await service.model_references(model_id=model_id)
        self.assertEqual("image_generation", references[0]["usage"])

    async def test_binding_list_uses_current_status_instead_of_stale_snapshot(self):
        model_id = uuid7()
        repository = _Repository()
        catalog = _Catalog({
            "model_id": str(model_id),
            "display_name": "图片模型",
            "served_model_name": "image-model",
            "provider": "api_oci_image",
            "status": "ACTIVE",
            "category": 8,
        })
        service = ModelBindingService(
            uow_factory=lambda: _UnitOfWork(repository),
            catalog_client=catalog,
        )
        await service.upsert(UpsertBindingCommand(
            domain_id=41,
            role="IMAGE_GENERATION",
            model_id=model_id,
            actor_id="user-41",
        ))

        catalog.row["status"] = "DRAFT"
        current = await service.list(domain_id=41)

        self.assertFalse(current[0]["ready"])
        self.assertEqual("ACTIVE", repository.row.model_snapshot_json["status"])

    async def test_binding_list_does_not_read_entity_after_uow_closes(self):
        model_id = uuid7()
        repository = _Repository()
        catalog = _Catalog({
            "model_id": str(model_id),
            "display_name": "图片模型",
            "served_model_name": "image-model",
            "provider": "api_oci_image",
            "status": "ACTIVE",
            "category": 8,
        })
        service = ModelBindingService(
            uow_factory=lambda: _UnitOfWork(repository),
            catalog_client=catalog,
        )
        await service.upsert(UpsertBindingCommand(
            domain_id=41,
            role="IMAGE_GENERATION",
            model_id=model_id,
            actor_id="user-41",
        ))

        state = {"closed": False}
        persisted = repository.row

        class _DetachedGuard:
            def __getattr__(self, name):
                if state["closed"]:
                    raise AssertionError("UoW 关闭后禁止读取 ORM 实体")
                return getattr(persisted, name)

        class _ClosingUnitOfWork(_UnitOfWork):
            async def __aenter__(self):
                state["closed"] = False
                return await super().__aenter__()

            async def __aexit__(self, exc_type, exc, traceback):
                state["closed"] = True
                return await super().__aexit__(exc_type, exc, traceback)

        repository.row = _DetachedGuard()
        service = ModelBindingService(
            uow_factory=lambda: _ClosingUnitOfWork(repository),
            catalog_client=catalog,
        )

        current = await service.list(domain_id=41)

        self.assertTrue(current[0]["ready"])


if __name__ == "__main__":
    unittest.main()
