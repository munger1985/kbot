"""闭域维度筛选值解析契约。"""

from __future__ import annotations

import unittest

from platform_core.contracts.data_query import (
    DimensionValueMember,
    MemberFilterResolutionError,
    normalize_catalog_dimension_filter,
    resolve_member_filter,
)


def _members() -> tuple[DimensionValueMember, ...]:
    return (
        DimensionValueMember(value="AT_RISK", aliases=("风险状态",)),
        DimensionValueMember(value="ACTIVE"),
        DimensionValueMember(value="华辰智能制造"),
    )


class ValueMemberFilterTest(unittest.TestCase):
    def test_alias_resolves_to_stored_value(self) -> None:
        operator, values = resolve_member_filter(
            members=_members(),
            operator="EQ",
            values=("风险状态",),
            strict=True,
            field_name="lifecycle_status",
        )
        self.assertEqual("EQ", operator)
        self.assertEqual(("AT_RISK",), values)

    def test_contains_unique_member_promotes_to_eq(self) -> None:
        operator, values = resolve_member_filter(
            members=_members(),
            operator="CONTAINS",
            values=("AT_RISK",),
            allowed_filter_operators=("EQ", "IN", "CONTAINS"),
            strict=True,
            field_name="lifecycle_status",
        )
        self.assertEqual("EQ", operator)
        self.assertEqual(("AT_RISK",), values)

    def test_contains_stored_substring_is_kept(self) -> None:
        operator, values = resolve_member_filter(
            members=_members(),
            operator="CONTAINS",
            values=("华辰",),
            strict=True,
            field_name="customer_name",
        )
        self.assertEqual("CONTAINS", operator)
        self.assertEqual(("华辰",), values)

    def test_strict_unresolved_value_lists_available_codes(self) -> None:
        with self.assertRaises(MemberFilterResolutionError) as raised:
            resolve_member_filter(
                members=_members(),
                operator="CONTAINS",
                values=("风险",),
                strict=True,
                field_name="lifecycle_status",
            )
        self.assertIn("AT_RISK", str(raised.exception))
        self.assertIn("风险", str(raised.exception))
        self.assertEqual(
            ("AT_RISK", "ACTIVE", "华辰智能制造"),
            raised.exception.available_values,
        )

    def test_non_strict_keeps_unresolved_contains(self) -> None:
        operator, values = resolve_member_filter(
            members=_members(),
            operator="CONTAINS",
            values=("风险",),
            strict=False,
            field_name="lifecycle_status",
        )
        self.assertEqual("CONTAINS", operator)
        self.assertEqual(("风险",), values)

    def test_catalog_filter_repairs_operator_then_resolves_member(self) -> None:
        operator, values = normalize_catalog_dimension_filter(
            field="lifecycle_status",
            operator="CONTAINS",
            values=["风险状态"],
            catalog_dimension={
                "name": "lifecycle_status",
                "allowed_filter_operators": ["EQ", "IN"],
                "value_members": [
                    {"value": "AT_RISK", "aliases": ["风险状态"]},
                    {"value": "ACTIVE"},
                ],
            },
            strict=True,
        )
        self.assertEqual("EQ", operator)
        self.assertEqual(["AT_RISK"], values)


if __name__ == "__main__":
    unittest.main()
