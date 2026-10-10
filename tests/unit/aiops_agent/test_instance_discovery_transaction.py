"""验证实例发现不会在事务关闭后读取 ORM 实体。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from aiops_agent.application.configuration.common import ConfigurationScope
from aiops_agent.application.configuration.service import AIOpsConfigurationService
from aiops_agent.ports.diagnostic_source import (
    InstanceDiscoveryCandidate,
    InstanceDiscoveryResult,
)
from platform_core.contracts.aiops import InstanceDiscoveryRequest
from platform_core.identity import uuid7


class _Binding:
    def __init__(self, state: SimpleNamespace, *, locator_key: str, target_id):
        self._state = state
        self._locator_key = locator_key
        self._target_id = target_id

    def _require_attached(self) -> None:
        if self._state.detached:
            raise RuntimeError("事务关闭后不得读取映射实体")

    @property
    def source_locator_key(self) -> str:
        self._require_attached()
        return self._locator_key

    @property
    def target_id(self):
        self._require_attached()
        return self._target_id


class _UnitOfWork:
    def __init__(self, state: SimpleNamespace, binding: _Binding):
        self._state = state
        self.diagnostic_sources = AsyncMock()
        self.diagnostic_sources.get_scoped.return_value = SimpleNamespace(
            source_type="PROMETHEUS"
        )
        self.targets = AsyncMock()
        self.targets.list_source_bindings_by_source.return_value = [binding]

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args) -> None:
        self._state.detached = True


class InstanceDiscoveryTransactionTest(unittest.IsolatedAsyncioTestCase):
    async def test_materializes_mapping_before_unit_of_work_closes(self) -> None:
        source_id = uuid7()
        target_id = uuid7()
        state = SimpleNamespace(detached=False)
        binding = _Binding(state, locator_key="oracle-test", target_id=target_id)
        uow = _UnitOfWork(state, binding)
        adapter = AsyncMock()
        adapter.discover_instances.return_value = InstanceDiscoveryResult(
            candidates=(
                InstanceDiscoveryCandidate(
                    source_locator_key="oracle-test",
                    source_locator={"target_key": "oracle-test"},
                    db_type="ORACLE",
                    display_name="Oracle Test",
                ),
            )
        )
        service = object.__new__(AIOpsConfigurationService)
        service._uow_factory = Mock(return_value=uow)
        service._instance_discovery_adapter = AsyncMock(return_value=adapter)
        service._instance_candidate_refs = Mock()
        service._instance_candidate_refs.encode.return_value = "candidate-ref"

        page = await service.discover_source_instances(
            scope=ConfigurationScope(
                domain_id=100,
                principal_id="PORTAL:aiops",
                actor_id="user-1",
                request_id="request-1",
                trace_id="trace-1",
            ),
            source_id=source_id,
            request=InstanceDiscoveryRequest(
                db_types=("ORACLE",),
                page_size=50,
            ),
        )

        self.assertTrue(state.detached)
        self.assertEqual("MAPPED", page.items[0].mapping_status)
        self.assertEqual(target_id, page.items[0].mapped_target_id)


if __name__ == "__main__":
    unittest.main()
