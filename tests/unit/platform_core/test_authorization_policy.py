"""核心授权策略执行测试。"""

import unittest
from uuid import UUID

from platform_core.authorization import can_read_agent, filter_readable_agents
from platform_core.contracts import PrincipalKind


ALLOWED_AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a001")
OTHER_AGENT_ID = UUID("019f8eae-2c25-7d48-b044-350ec3f5a002")
DOMAIN_ID = 100


class CoreAgentAuthorizationTest(unittest.TestCase):
    def test_human_use_reads_only_active_agents(self) -> None:
        rows = filter_readable_agents(
            app_id="aiops",
            domain_id=DOMAIN_ID,
            principal_kind=PrincipalKind.PORTAL,
            permissions=("aiops:use",),
            authorized_agent_ids=(),
            agents=(
                {
                    "agent_id": str(ALLOWED_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "ACTIVE",
                },
                {
                    "agent_id": str(OTHER_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "DISABLED",
                },
                {
                    "agent_id": str(OTHER_AGENT_ID),
                    "domain_id": DOMAIN_ID + 1,
                    "status": "ACTIVE",
                },
            ),
        )

        self.assertEqual([str(ALLOWED_AGENT_ID)], [row["agent_id"] for row in rows])

    def test_human_agent_manager_reads_all_lifecycle_states(self) -> None:
        self.assertTrue(
            can_read_agent(
                app_id="aiops",
                domain_id=DOMAIN_ID,
                principal_kind=PrincipalKind.PORTAL,
                permissions=("aiops:use", "aiops:agent_manage"),
                authorized_agent_ids=(),
                agent={
                    "agent_id": str(OTHER_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "DISABLED",
                },
            )
        )

    def test_machine_requires_active_allowlisted_agent(self) -> None:
        common = {
            "app_id": "aiops",
            "domain_id": DOMAIN_ID,
            "principal_kind": PrincipalKind.APP_API_CLIENT,
            "permissions": ("aiops:use", "aiops:agent_manage"),
            "authorized_agent_ids": (ALLOWED_AGENT_ID,),
        }

        self.assertTrue(
            can_read_agent(
                **common,
                agent={
                    "agent_id": str(ALLOWED_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "ACTIVE",
                },
            )
        )
        self.assertFalse(
            can_read_agent(
                **common,
                agent={
                    "agent_id": str(OTHER_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "ACTIVE",
                },
            )
        )
        self.assertFalse(
            can_read_agent(
                **common,
                agent={
                    "agent_id": str(ALLOWED_AGENT_ID),
                    "domain_id": DOMAIN_ID,
                    "status": "DISABLED",
                },
            )
        )

    def test_agent_read_requires_use_permission_and_current_domain(self) -> None:
        agent = {
            "agent_id": str(ALLOWED_AGENT_ID),
            "domain_id": DOMAIN_ID,
            "status": "ACTIVE",
        }
        self.assertFalse(
            can_read_agent(
                app_id="aiops",
                domain_id=DOMAIN_ID,
                principal_kind=PrincipalKind.PORTAL,
                permissions=("aiops:agent_manage",),
                authorized_agent_ids=(),
                agent=agent,
            )
        )
        self.assertFalse(
            can_read_agent(
                app_id="aiops",
                domain_id=DOMAIN_ID,
                principal_kind=PrincipalKind.SERVICE,
                permissions=("aiops:use", "aiops:agent_manage"),
                authorized_agent_ids=(),
                agent=agent,
            )
        )
        self.assertFalse(
            can_read_agent(
                app_id="aiops",
                domain_id=DOMAIN_ID + 1,
                principal_kind=PrincipalKind.PORTAL,
                permissions=("aiops:use", "aiops:agent_manage"),
                authorized_agent_ids=(),
                agent=agent,
            )
        )


if __name__ == "__main__":
    unittest.main()
