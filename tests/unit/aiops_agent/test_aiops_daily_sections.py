"""日报共享分区、版式冻结、模板发布与模拟预览的跨数据库回归。"""

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from aiops_agent.application.inspections.daily_sections import daily_section_catalog, normalize_daily_layout, project_daily_sections
from aiops_agent.application.inspections.daily_preview import daily_preview
from aiops_agent.application.inspections.check_catalog import load_check_catalog, default_selected_check_ids
from aiops_agent.application.report_templates import InspectionTemplateService
from aiops_agent.entities import InspectionTemplateEntity
from platform_core.contracts.aiops.daily_report import DAILY_SECTION_IDS, DailyReportLayout
from platform_core.contracts.aiops.public import ReportPresentation
from platform_core.identity import uuid7

IS_KBOT = InspectionTemplateEntity.__tablename__.startswith("KBOT_")


@pytest.mark.parametrize("engine", ["ORACLE", "MYSQL", "POSTGRESQL", "FUTURE_DATABASE"])
@pytest.mark.parametrize("scenario", ["NORMAL", "ATTENTION", "INCOMPLETE"])
def test_all_engines_share_one_section_contract_and_simulation(engine, scenario):
    presentation = daily_preview(database_type=engine, scenario=scenario)
    ReportPresentation.model_validate(presentation)
    data = presentation["daily_inspection"]
    assert data["preview"] is True
    assert data["header"]["database_type"] == engine
    assert [item["section_id"] for item in data["sections"]] == list(DAILY_SECTION_IDS)
    assert data["conclusion"]["health_status"] == {"NORMAL": "NORMAL", "ATTENTION": "ATTENTION", "INCOMPLETE": "INCOMPLETE"}[scenario]
    checks = [check for region in data["sections"] if region["section_id"] not in {"OVERVIEW", "BASELINE", "ACTIONS"} for check in region["checks"]]
    assert len(checks) == len(data["checks"])
    assert checks
    sql = next(item for item in data["sections"] if item["section_id"] == "SQL")
    if scenario == "ATTENTION":
        assert sql["sql_details"][0]["statements"][0]["agent_review"]["status"] == "COMPLETE"
        sql_ids = {item["problem_id"] for group in sql["sql_details"] for item in group["statements"]}
        assert sql_ids <= {item["anchor"] for item in data["finding_index"]}
        assert any("模拟" in item["action"] and item["section_id"] == "SQL" for item in data["actions"])
    else:
        assert sql["status"] == "NORMAL"
        assert "当前没有问题 SQL" in sql["summary"]


def test_catalog_entries_have_explicit_shared_regions_and_no_engine_layout():
    ids = set(DAILY_SECTION_IDS)
    for group in load_check_catalog().groups:
        for check in group.checks:
            assert check.report_section in ids
    roles = [role for region in daily_section_catalog()["sections"] for role in region["semantic_roles"]]
    assert len(roles) == len(set(roles))
    assert "database_type" not in DailyReportLayout.model_fields


def test_pdf_keeps_shared_regions_and_uses_readable_severity_labels():
    from reportlab.lib.styles import getSampleStyleSheet
    from aiops_agent.application.daily_inspection_pdf import daily_inspection_pdf_story

    data = daily_preview(database_type="FUTURE_DATABASE", scenario="ATTENTION")["daily_inspection"]
    frozen = deepcopy(data)
    styles = getSampleStyleSheet()
    story = daily_inspection_pdf_story(
        data=data, font_name="Helvetica", body=styles["BodyText"],
        section=styles["Heading2"], cover_title=styles["Title"],
    )

    def plain_text(item):
        if hasattr(item, "getPlainText"):
            return item.getPlainText()
        return " ".join(plain_text(cell) for row in getattr(item, "_cellvalues", ()) for cell in row)

    text = " ".join(plain_text(item) for item in story)
    assert all(region["title"] in text for region in data["sections"])
    assert "MEDIUM" not in text and "HIGH" not in text
    assert "Agent 初步分析" in text and "附录：统计方法与技术追溯" in text
    assert data == frozen


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "first", "last", "hide_first"])
def test_invalid_layout_cannot_hide_or_invent_regions(mutation):
    layout = normalize_daily_layout()
    if mutation == "missing":
        layout["sections"].pop()
    elif mutation == "duplicate":
        layout["sections"][2] = deepcopy(layout["sections"][1])
    elif mutation == "unknown":
        layout["sections"][2]["section_id"] = "CUSTOM_SQL"
    elif mutation == "first":
        layout["sections"][0],layout["sections"][1] = layout["sections"][1],layout["sections"][0]
    elif mutation == "last":
        layout["sections"][-1],layout["sections"][-2] = layout["sections"][-2],layout["sections"][-1]
    else:
        layout["sections"][0]["visibility"] = "WHEN_RELEVANT"
    with pytest.raises(ValidationError):
        DailyReportLayout.model_validate(layout)


def test_template_controls_presentation_without_changing_facts_or_health():
    data = daily_preview(database_type="MYSQL", scenario="ATTENTION")["daily_inspection"]
    frozen = deepcopy(data)
    layout = normalize_daily_layout()
    layout["title"] = "生产系统每日巡检"
    layout["sections"][2],layout["sections"][3] = layout["sections"][3],layout["sections"][2]
    for item in layout["sections"][1:-1]:
        item.update(visibility="WHEN_RELEVANT", normal_detail="SUMMARY", expanded=False, show_statistics=False, show_trend=False)
    projected = project_daily_sections(data, layout=layout)
    assert projected["conclusion"] == data["conclusion"]
    assert projected["findings"] == data["findings"]
    assert projected["header"]["title"] == layout["title"]
    assert projected["sections"][2]["section_id"] == "CAPACITY"
    assert {item["section_id"] for item in projected["sections"]} == set(DAILY_SECTION_IDS)
    assert data == frozen
    for region in projected["sections"]:
        if region["findings"]:
            assert region["findings"]


def test_empty_unknown_optional_and_incomplete_content_is_not_silently_normal():
    data = daily_preview(database_type="MYSQL", scenario="INCOMPLETE")["daily_inspection"]
    data["checks"] = [{"title":"新数据库扩展检查","check_ids":["future.capacity"],"required":True,"status":"MISSING",
                      "summary":"读取超时","report_section":"NEW_ENGINE_TOPIC"}]
    data.update(metrics=[],findings=[],sql_details=[],recommendations=["建议一","建议二"])
    projected = project_daily_sections(data)
    performance = next(region for region in projected["sections"] if region["section_id"] == "PERFORMANCE")
    assert performance["checks"][0]["classification_note"]
    assert performance["status"] == "INCOMPLETE"
    capacity = next(region for region in projected["sections"] if region["section_id"] == "CAPACITY")
    assert capacity["status"] == "NOT_INCLUDED"
    assert "不据此判断正常" in capacity["summary"]
    assert {"建议一","建议二"} <= {action["action"] for action in projected["actions"]}


def test_completed_check_with_problem_remains_attention_not_incomplete():
    data = daily_preview(database_type="MYSQL", scenario="NORMAL")["daily_inspection"]
    data["checks"] = [{"title":"备份保障","required":True,"kind":"backup_policy_check","check_ids":["backup"],
                      "status":"FINDING","severity":"HIGH","summary":"未发现有效备份成功记录"}]
    data.update(metrics=[],findings=[],sql_details=[])
    projected = project_daily_sections(data)
    backup = next(region for region in projected["sections"] if region["section_id"] == "BACKUP")
    assert backup["status"] == "ABNORMAL"
    assert backup["checks"][0]["summary"] == "未发现有效备份成功记录"


def test_layout_freezes_in_definition_and_does_not_enter_monthly_definition():
    selected = default_selected_check_ids("DAILY")
    layout = normalize_daily_layout()
    layout["sections"][2]["normal_detail"] = "SUMMARY"
    definition = InspectionTemplateService._definition(display_name="测试日报", selected_check_ids=selected, daily_report_layout=layout)
    assert definition["daily_report_layout"] == layout
    layout["title"] = "不会修改冻结定义"
    assert definition["daily_report_layout"]["title"] != layout["title"]
    monthly = InspectionTemplateService._definition(display_name="测试月报", selected_check_ids=selected, schedule_type="MONTHLY")
    assert "daily_report_layout" not in monthly


def test_create_publish_history_disable_and_domain_scope_preserve_old_version():
    async def run():
        domain = 7 if IS_KBOT else uuid7()
        actor = "test-admin" if IS_KBOT else uuid7()
        rows, versions = {}, {}
        async def add_row(row):
            row.row_version = 1
            row.updated_at = datetime.now(UTC)
            rows[row.inspection_template_id] = row
        async def add_version(version):
            version.created_at = datetime.now(UTC)
            versions[version.inspection_template_version_id] = version
        def scoped_row(**kwargs):
            row = rows.get(kwargs["inspection_template_id"])
            return row if row and row.domain_id == kwargs["domain_id"] else None
        repo = SimpleNamespace(
            add_inspection_template=AsyncMock(side_effect=add_row),
            add_inspection_template_version=AsyncMock(side_effect=add_version),
            get_inspection_template=AsyncMock(side_effect=scoped_row),
            get_inspection_template_version=AsyncMock(side_effect=lambda **kwargs: versions.get(kwargs["inspection_template_version_id"])),
            next_inspection_template_version=AsyncMock(return_value=2),
            list_inspection_template_versions=AsyncMock(side_effect=lambda **kwargs: sorted(
                [item for item in versions.values() if item.inspection_template_id == kwargs["inspection_template_id"]],
                key=lambda item:item.version_no, reverse=True)),
        )
        uow = SimpleNamespace(inspections=repo, commit=AsyncMock())
        @asynccontextmanager
        async def context():
            yield uow
        service = InspectionTemplateService(uow_factory=context)
        identity = {"domain_id":domain,"actor_id":actor, **({} if IS_KBOT else {"tenant_id":uuid7()})}
        selected = default_selected_check_ids("DAILY")
        original_layout = normalize_daily_layout()
        first = await service.create(**identity, display_name="共享日报", selected_check_ids=selected, daily_report_layout=original_layout)
        template_id = next(iter(rows))
        initial = deepcopy(next(iter(versions.values())).definition_json)
        layout = normalize_daily_layout()
        layout["title"] = "新版日报"
        await service.create_version(**identity, template_id=template_id, expected_row_version=1, selected_check_ids=selected, daily_report_layout=layout)
        history = await service.list_versions(domain_id=domain,template_id=template_id)
        assert [item["version_no"] for item in history] == [2,1]
        assert history[0]["current"] is True and history[1]["current"] is False
        assert history[1]["definition"] == initial
        assert first["daily_report_layout"] == original_layout
        with pytest.raises(Exception):
            await service.list_versions(domain_id=8 if IS_KBOT else uuid7(),template_id=template_id)
        assert repo.list_inspection_template_versions.await_count == 1
        await service.update(domain_id=domain,actor_id=actor,template_id=template_id,expected_row_version=1,status="INACTIVE")
        assert rows[template_id].status == "INACTIVE"
        assert len(versions) == 2
        with pytest.raises(Exception):
            await service.update(domain_id=domain,actor_id=actor,template_id=template_id,expected_row_version=999,status="ACTIVE")
    asyncio.run(run())


def test_preview_http_is_scoped_strict_and_uses_formal_contract():
    from aiops_agent.api import report_templates as api
    async def run():
        app = FastAPI()
        app.include_router(api.router)
        context = SimpleNamespace(domain_id=7 if IS_KBOT else uuid7(),principal_id=uuid7(),asserted_user_id="tester",client_id="portal")
        app.dependency_overrides[api.get_aiops_auth_context] = lambda:context
        async with AsyncClient(transport=ASGITransport(app=app),base_url="http://preview") as client:
            with patch.object(api,"require_service_scope") as guard:
                response = await client.post("/internal/v1/aiops/inspection-templates:preview",json={"database_type":"POSTGRESQL","scenario":"ATTENTION"})
                assert response.status_code == 200, response.text
                ReportPresentation.model_validate(response.json())
                assert response.json()["daily_inspection"]["preview"] is True
                guard.assert_called_once_with(guard.call_args.args[0],"aiops.manage")
                response = await client.post("/internal/v1/aiops/inspection-templates:preview",json={"sql":"SELECT 1"})
                assert response.status_code == 422
            with patch.object(api,"require_service_scope",side_effect=HTTPException(403,"禁止")):
                response = await client.post("/internal/v1/aiops/inspection-templates:preview",json={})
                assert response.status_code == 403
    asyncio.run(run())
