"""巡检展示贯穿真实内部与公开 HTTP 响应校验，不以直接调用投影函数代替。"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from aiops_agent.api.runtime import routes as internal
from aiops_agent.entities import ReportSourceEntity
from main_api.api import ops as public
from platform_clients.aiops import AIOpsManagementClient
from platform_core.contracts.aiops.public import ReportPresentation
from platform_core.identity import uuid7
from tests.support.inspection_report_samples import inspection_samples


class InspectionPresentationHttpTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = inspection_samples()

    def check_sample(self, name):
        async def run():
            sample = deepcopy(self.samples[name])
            report_id, agent_id = uuid7(), uuid7()
            domain_id = 7 if ReportSourceEntity.__tablename__.startswith("KBOT_") else uuid7()
            context = SimpleNamespace(domain_id=domain_id)
            service = SimpleNamespace(
                get_report_source_agent_id=AsyncMock(return_value=agent_id),
                get_report_presentation=AsyncMock(return_value=sample),
            )
            internal_app = FastAPI()
            internal_app.include_router(internal.router)
            internal_app.dependency_overrides[internal.get_service] = lambda: service
            internal_app.dependency_overrides[internal.get_aiops_auth_context] = lambda: context
            async with AsyncClient(
                transport=ASGITransport(app=internal_app, raise_app_exceptions=False),
                base_url="http://internal",
            ) as internal_http:
                management = AIOpsManagementClient.__new__(AIOpsManagementClient)

                async def dispatch(method, path, *, auth_context):
                    self.assertIs(context, auth_context)
                    response = await internal_http.request(method, path)
                    self.assertEqual(200, response.status_code, response.text)
                    return response.json()

                management._json = AsyncMock(side_effect=dispatch)
                public_app = FastAPI()
                public_app.add_api_route(
                    public.router.prefix + "/reports/{report_id}/presentation",
                    public.get_report_presentation, response_model=ReportPresentation,
                )

                @public_app.middleware("http")
                async def set_context(request, call_next):
                    request.state.auth_context = context
                    return await call_next(request)

                with patch.object(internal, "require_service_scope") as service_scope, patch.object(
                    internal, "_ensure_agent_authorized",
                ) as authorization, patch.object(public, "_client", return_value=management):
                    async with AsyncClient(
                        transport=ASGITransport(app=public_app),
                        base_url="http://public",
                    ) as http:
                        path = public.router.prefix.replace("{domain_code}", "aiops-default")
                        response = await http.get(f"{path}/reports/{report_id}/presentation")
                self.assertEqual(200, response.status_code)
                body = response.json()
                key = "monthly_inspection" if name == "monthly" else "daily_inspection"
                self.assertEqual(sample[key]["header"], body[key]["header"])
                self.assertEqual(sample[key]["conclusion"], body[key]["conclusion"])
                self.assertEqual(sample[key]["metrics"], body[key]["metrics"])
                self.assertEqual(sample[key]["limitations"], body[key]["limitations"])
                if key == "monthly_inspection":
                    self.assertEqual(3, len(body[key]["target_results"]))
                    self.assertEqual(sample[key]["target_results"], body[key]["target_results"])
                else:
                    self.assertEqual(sample[key]["checks"], body[key]["checks"])
                    self.assertNotIn("baseline_metrics", body[key])
                service.get_report_presentation.assert_awaited_once_with(
                    report_id=report_id, domain_id=domain_id,
                )
                service_scope.assert_called_once()
                authorization.assert_called_once_with(context, agent_id)
        asyncio.run(run())

    def test_normal_daily_survives_both_http_boundaries(self):
        self.check_sample("normal")

    def test_attention_daily_survives_both_http_boundaries(self):
        self.check_sample("attention")

    def test_abnormal_daily_survives_both_http_boundaries(self):
        self.check_sample("abnormal")

    def test_incomplete_daily_survives_both_http_boundaries(self):
        self.check_sample("incomplete")

    def test_monthly_survives_both_http_boundaries(self):
        self.check_sample("monthly")

    def test_report_headers_still_reject_undeclared_fields(self):
        for name, key in (("normal", "daily_inspection"), ("monthly", "monthly_inspection")):
            sample = deepcopy(self.samples[name])
            sample[key]["header"]["undeclared_field"] = "不可绕过合同"
            with self.subTest(report=name), self.assertRaises(ValidationError):
                ReportPresentation.model_validate(sample)

    def test_openapi_declares_professional_business_fields(self):
        definitions = ReportPresentation.model_json_schema()["$defs"]
        self.assertIn("timezone", definitions["DailyInspectionHeader"]["properties"])
        self.assertIn("business_system_name", definitions["DailyInspectionHeader"]["properties"])
        self.assertIn("timezone", definitions["MonthlyInspectionHeader"]["properties"])
        for field in ("metrics", "checks", "limitations"):
            self.assertIn(field, definitions["DailyInspectionPresentation"]["properties"])
        for field in ("target_results", "limitations"):
            self.assertIn(field, definitions["MonthlyInspectionPresentation"]["properties"])
        self.assertNotIn("baseline_metrics", definitions["DailyInspectionPresentation"]["properties"])


if __name__ == "__main__":
    unittest.main()
