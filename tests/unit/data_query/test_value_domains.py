"""维度值域采集、观察与编译期解析。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from uuid import uuid4

from data_query.application.value_domains import (
    ValueDomainObserver,
    planning_dimension_payload,
    should_observe_dimension,
)
from data_query.connectors.filter_values import resolve_dimension_filter
from data_query.connectors.value_members import (
    CLOSED_DOMAIN_FILTER_OPERATORS,
    VALUE_MEMBER_SAMPLE_LIMIT,
    dimension_value_domain,
    members_from_stored_values,
    parse_check_string_members,
)
from data_query.contracts import (
    DatasetDefinition,
    DimensionDefinition,
    DimensionValueMember,
    MeasureDefinition,
    PlanFilter,
    PlanMeasure,
    SemanticModelDefinition,
    DataQueryPlanV1,
)
from data_query.connectors import compile_dialect_query
from platform_core.identity import uuid7


def _status_dimension(**kwargs) -> DimensionDefinition:
    payload = {
        "name": "lifecycle_status",
        "dataset": "customers",
        "physical_column": "LIFECYCLE_STATUS",
        "value_type": "STRING",
        "value_members": (
            DimensionValueMember(value="AT_RISK", aliases=("风险状态",)),
            DimensionValueMember(value="ACTIVE"),
        ),
    }
    payload.update(kwargs)
    return DimensionDefinition(**payload)


class ValueDomainCollectionTest(unittest.TestCase):
    def test_oracle_check_in_list_is_parsed(self) -> None:
        values = parse_check_string_members(
            definition="LIFECYCLE_STATUS IN ('PROSPECT', 'QUALIFIED', 'ACTIVE', 'AT_RISK', 'CHURNED')",
            physical_column="LIFECYCLE_STATUS",
        )
        self.assertEqual(
            ("PROSPECT", "QUALIFIED", "ACTIVE", "AT_RISK", "CHURNED"),
            values,
        )

    def test_postgres_any_array_is_parsed(self) -> None:
        values = parse_check_string_members(
            definition='"lifecycle_status"::text = ANY (ARRAY[\'ACTIVE\'::text, \'AT_RISK\'::text])',
            physical_column="lifecycle_status",
        )
        self.assertEqual(("ACTIVE", "AT_RISK"), values)

    def test_check_members_narrow_filter_operators(self) -> None:
        members, operators = dimension_value_domain(
            physical_column="LIFECYCLE_STATUS",
            column_type="VARCHAR2(32)",
            constraints=[{
                "name": "CK_STATUS",
                "type": "C",
                "definition": "LIFECYCLE_STATUS IN ('ACTIVE', 'AT_RISK')",
            }],
        )
        self.assertEqual(("ACTIVE", "AT_RISK"), tuple(item.value for item in members))
        self.assertEqual(CLOSED_DOMAIN_FILTER_OPERATORS, operators)

    def test_high_cardinality_samples_are_discarded(self) -> None:
        values = [f"value_{index:03d}" for index in range(VALUE_MEMBER_SAMPLE_LIMIT)]
        self.assertIsNone(members_from_stored_values(values))

    def test_planning_payload_includes_logical_value_domain(self) -> None:
        payload = planning_dimension_payload(_status_dimension())
        self.assertEqual("NONE", payload["value_normalization"])
        self.assertEqual(
            ({"value": "AT_RISK", "aliases": ("风险状态",)}, {"value": "ACTIVE"}),
            payload["value_members"],
        )
        self.assertNotIn("physical_column", payload)


class ValueDomainObserverTest(unittest.IsolatedAsyncioTestCase):
    def _dataset(self) -> DatasetDefinition:
        return DatasetDefinition(
            name="customers",
            display_name="客户",
            physical_schema="DEMO",
            physical_object="DEMO_CRM_CUSTOMER_OVERVIEW",
        )

    async def test_observer_reuses_cache_for_same_locator(self) -> None:
        sampler = AsyncMock(return_value=("AT_RISK", "ACTIVE"))
        observer = ValueDomainObserver(sampler=sampler, cache_ttl_seconds=600)
        dimension = DimensionDefinition(
            name="lifecycle_status",
            dataset="customers",
            physical_column="LIFECYCLE_STATUS",
            value_type="STRING",
        )
        data_source_id = uuid4()
        dataset = self._dataset()
        await observer.resolve_planning_members(
            dimension=dimension, dataset=dataset,
            source_type="ORACLE", data_source_id=data_source_id,
        )
        await observer.resolve_planning_members(
            dimension=dimension, dataset=dataset,
            source_type="ORACLE", data_source_id=data_source_id,
        )
        self.assertEqual(1, sampler.await_count)

    async def test_sensitive_and_alias_dimensions_are_not_observed(self) -> None:
        sampler = AsyncMock(return_value=("secret",))
        observer = ValueDomainObserver(sampler=sampler)
        dataset = self._dataset()
        sensitive = DimensionDefinition(
            name="email",
            dataset="customers",
            physical_column="EMAIL",
            value_type="STRING",
            sensitivity="SENSITIVE",
        )
        aliased = DimensionDefinition(
            name="topic",
            dataset="customers",
            physical_column="TITLE",
            value_type="STRING",
            filter_alias_columns=("PRODUCT", "SOLUTION"),
        )
        self.assertFalse(should_observe_dimension(sensitive))
        self.assertFalse(should_observe_dimension(aliased))
        await observer.resolve_planning_members(
            dimension=sensitive, dataset=dataset,
            source_type="ORACLE", data_source_id=uuid4(),
        )
        await observer.resolve_planning_members(
            dimension=aliased, dataset=dataset,
            source_type="ORACLE", data_source_id=uuid4(),
        )
        sampler.assert_not_awaited()

    async def test_high_cardinality_observation_is_dropped(self) -> None:
        sampler = AsyncMock(
            return_value=tuple(f"value_{index:03d}" for index in range(VALUE_MEMBER_SAMPLE_LIMIT))
        )
        observer = ValueDomainObserver(sampler=sampler)
        dimension = DimensionDefinition(
            name="customer_code",
            dataset="customers",
            physical_column="CUSTOMER_CODE",
            value_type="STRING",
        )
        members = await observer.resolve_planning_members(
            dimension=dimension,
            dataset=self._dataset(),
            source_type="ORACLE",
            data_source_id=uuid4(),
        )
        self.assertEqual((), members)


class ValueDomainCompilerTest(unittest.TestCase):
    def test_compiler_promotes_alias_contains_to_eq(self) -> None:
        operator, values = resolve_dimension_filter(
            dimension=_status_dimension(),
            operator="CONTAINS",
            values=("风险状态",),
        )
        self.assertEqual("EQ", operator)
        self.assertEqual(("AT_RISK",), values)

    def test_compiler_sql_uses_stored_code_for_alias(self) -> None:
        model = SemanticModelDefinition(
            datasets=(DatasetDefinition(
                name="customers",
                display_name="客户",
                physical_schema="DEMO",
                physical_object="DEMO_CRM_CUSTOMER_OVERVIEW",
            ),),
            dimensions=(_status_dimension(),),
            measures=(MeasureDefinition(
                name="customer_count",
                dataset="customers",
                physical_column=None,
                aggregation="COUNT",
                value_type="INTEGER",
            ),),
        )
        plan = DataQueryPlanV1(
            semantic_model_id=uuid7(),
            semantic_model_version=1,
            dataset="customers",
            measures=(PlanMeasure(name="customer_count", aggregation="COUNT"),),
            dimensions=("lifecycle_status",),
            filters=(PlanFilter(
                field="lifecycle_status",
                operator="CONTAINS",
                values=("风险状态",),
            ),),
            limit=20,
        )
        compiled = compile_dialect_query(
            dialect="ORACLE", plan=plan, model=model, guardrail_max_limit=100,
        )
        self.assertIn("=", compiled.sql)
        self.assertNotIn("LIKE", compiled.sql)
        self.assertEqual("AT_RISK", compiled.parameters[0])


if __name__ == "__main__":
    unittest.main()
