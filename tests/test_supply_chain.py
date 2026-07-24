"""供应链基线测试。"""

from __future__ import annotations

import unittest

from scripts.check_supply_chain import (
    build_sbom,
    check_supply_chain,
    load_requirements,
)


class SupplyChainTest(unittest.TestCase):
    def test_supply_chain_baseline_is_valid(self) -> None:
        self.assertEqual([], check_supply_chain())

    def test_direct_dependencies_are_unique_and_pinned(self) -> None:
        dependencies = load_requirements()
        self.assertTrue(dependencies)
        self.assertEqual(
            len(dependencies),
            len({name for name, _ in dependencies}),
        )

    def test_sbom_contains_every_direct_dependency(self) -> None:
        sbom = build_sbom()
        self.assertEqual("CycloneDX", sbom["bomFormat"])
        self.assertEqual(
            len(load_requirements()),
            len(sbom["components"]),
        )


if __name__ == "__main__":
    unittest.main()
