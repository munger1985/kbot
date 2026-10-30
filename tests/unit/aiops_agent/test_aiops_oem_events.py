"""OEM官方Incident与通用事件事实的生产评估、模型输入及来源重放。"""

import asyncio
from datetime import timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.application.diagnosis.event_evidence import source_event_fact
from aiops_agent.application.diagnosis.source_history_validation import validate_source_history_facts
from aiops_agent.contracts.evidence import ObservationSet
from aiops_agent.playbooks import canonical_hash
from aiops_agent.ports.diagnostic_source import EventEvidenceRequest
from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler
from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection, project, GUID, NOW


def event_request(**changes):
    return EventEvidenceRequest(target_id=str(uuid7()), binding_id=str(uuid7()), source_locator_key=GUID,
        window_start=NOW-timedelta(minutes=15), window_end=NOW, max_events=10,
        trace_id="events-test", **changes)


def incident(**changes):
    return {"id": "A"*32, "displayId": 1, "message": "CPU 使用率超过供应商阈值",
        "targets": [{"targetName": "mes-db", "targetType": "oracle_database"}],
        "severity": "WARNING", "status": "new", "isOpen": True,
        "timeCreated": "2026-10-28T06:50:00.000Z", "timeUpdated": "2026-10-28T06:55:00.000Z",
        **changes}


def events_adapter(payload=None, **kwargs):
    value, calls = adapter({"targetName": "mes-db", "targetType": "oracle_database"},
        collection([incident()]) if payload is None else payload, **kwargs)
    value.context = value.context.model_copy(update={"declared_capabilities": {"event.query": {}}})
    return value, calls


async def observe(items=None, **changes):
    value, calls = events_adapter(collection([incident()] if items is None else items, **changes))
    request = event_request()
    result = await value.query_events(request)
    payload = ObservationSet(target_id=request.target_id, binding_id=request.binding_id,
        source_id=value.context.source_id, active_alerts=result.events, event_query=result.query,
        gaps=result.gaps, collected_at=NOW)
    return payload, calls, result


def test_official_incident_query_rebinds_target_uses_basic_and_disables_member_targets_and_redirects():
    async def run():
        value, calls = events_adapter(credentials={"username": "reader", "password": " password "})
        result = await value.query_events(event_request())
        assert not result.gaps and len(result.events) == 1
        assert calls[0][0].endswith("/em/api/targets/"+GUID)
        assert calls[1][0].endswith("/em/api/incidents")
        params = calls[1][1]["params"]
        assert params["targetName"] == "mes-db" and params["targetTypeName"] == "oracle_database"
        assert params["includeMemberTargets"] == "false" and params["sort"] == "timeCreated:DESC"
        assert params["ageInHoursLessThanOrEqualTo"] >= 1 and "startTime" not in params
        assert all(not kw["allow_redirects"] and kw["auth"].password == " password " for _, kw in calls)
        assert result.events[0]["incident_id"] == "A"*32 and result.events[0]["status"] == "NEW"
        assert result.query.query_scope == "CURRENT_OPEN_INCIDENTS"
    asyncio.run(run())


@pytest.mark.parametrize("invalid", [None, "mes-db", "../other", "https://evil.invalid", "0"*33])
def test_incident_locator_requires_bound_guid_without_following_arbitrary_paths(invalid):
    value, calls = events_adapter()
    req = event_request().model_copy(update={"source_locator_key": invalid})
    result = asyncio.run(value.query_events(req))
    assert not result.events and result.query is None and result.gaps[0].code == "SOURCE_TARGET_LOCATOR_INVALID"
    assert not calls


def test_missing_event_authorization_is_not_borrowed_from_metric_permissions():
    value, calls = events_adapter()
    value.context = value.context.model_copy(update={"declared_capabilities": {"metric.query_range": {}}})
    result = asyncio.run(value.query_events(event_request()))
    assert result.gaps[0].code == "SOURCE_CAPABILITY_FORBIDDEN" and not calls


@pytest.mark.parametrize("changes", [
    {"id": "short"}, {"id": "Z"*32}, {"message": None}, {"message": "x"*4001},
    {"severity": []}, {"severity": "UNKNOWN"}, {"status": None}, {"isOpen": "true"},
    {"timeCreated": "bad"}, {"timeCreated": "2026-10-28T06:50:00"},
    {"timeUpdated": "2026-10-28T06:49:00Z"}, {"targets": []},
    {"targets": [{"targetName": "other", "targetType": "oracle_database"}]},
    {"targets": [{"targetName": "mes-db", "targetType": "host"}]},
    {"targets": [{"targetName": "mes-db", "targetType": "oracle_database"}]*2},
])
def test_wrong_target_or_malformed_incident_invalidates_entire_batch(changes):
    value, _ = events_adapter(collection([incident(), incident(**{"id": "B"*32, **changes})]))
    result = asyncio.run(value.query_events(event_request()))
    assert not result.events and result.query is None and result.gaps and not result.gaps[0].retryable


@pytest.mark.parametrize("payload", [
    collection([incident(), incident()]),
    collection([incident()], count=3),
    collection([incident(), incident(id="B"*32, timeCreated="2026-10-28T06:51:00Z")]),
])
def test_duplicates_count_mismatch_and_wrong_order_are_not_silently_accepted(payload):
    value, _ = events_adapter(payload)
    result = asyncio.run(value.query_events(event_request()))
    assert not result.events and result.query is None and result.gaps


@pytest.mark.parametrize("status,code", [(302, "SOURCE_RESPONSE_INVALID"), (401, "SOURCE_AUTH_FAILED"),
    (429, "SOURCE_RATE_LIMITED"), (503, "SOURCE_UNREACHABLE")])
def test_http_failures_and_redirects_never_claim_empty_success(status, code):
    value, calls = events_adapter(statuses=[200, status])
    result = asyncio.run(value.query_events(event_request()))
    assert result.gaps[0].code == code and result.query is None and len(calls) == 2


def test_oversize_total_byte_budget_cannot_return_a_partial_success():
    value, _ = events_adapter(collection([incident(message="x"*3000)]))
    result = asyncio.run(value.query_events(event_request(max_response_bytes=1024)))
    assert result.gaps[0].code == "SOURCE_RESULT_TRUNCATED" and not result.events and result.query is None


@pytest.mark.parametrize("items", [[], [incident(isOpen=False)], [incident(timeCreated="2026-10-28T06:44:59Z")],
    [incident(timeCreated="2026-10-28T07:00:00Z", timeUpdated="2026-10-28T07:00:00Z")]])
def test_zero_rows_preserve_open_only_query_scope_without_claiming_no_historical_alerts(items):
    async def run():
        payload, _, result = await observe(items)
        assert result.query is not None and not result.events and not result.gaps
        assessment, _, _, _ = await project(payload)
        fact = next(item for item in assessment.evidence if item.tool_id == "monitor.event.query")
        assert fact.row_count == 0 and fact.measurement_semantics == "CURRENT_ACTIVITY"
        assert "零行不等于历史无告警" in fact.warnings[0]
    asyncio.run(run())


def test_page_boundary_and_next_link_are_visible_without_following_a_source_url():
    async def run():
        payload, calls, _ = await observe(links={"next": {"href": "https://evil.invalid"}})
        fact = source_event_fact(artifact_id=uuid7(), result=payload)
        assert fact.truncated and len(calls) == 2 and "TRUNCATED" in fact.provenance.quality_flags
        value, _ = events_adapter()
        result = await value.query_events(event_request().model_copy(update={"max_events": 1}))
        assert result.query.truncated
    asyncio.run(run())


@pytest.mark.parametrize("source_type", ["OEM", "ZABBIX", "PROMETHEUS"])
def test_all_existing_event_sources_reach_real_assessor_and_answer_input_without_requiring_metrics(source_type):
    async def run():
        if source_type == "OEM":
            payload, _, _ = await observe()
        else:
            payload = ObservationSet(target_id=str(uuid7()), binding_id=str(uuid7()), source_id=str(uuid7()), collected_at=NOW,
                active_alerts=({"source_type": source_type, "event_id": "123", "name": "告警",
                    "severity": "WARNING", "status": "FIRING", "active_at": NOW.isoformat()},))
        assessment, _, _, _ = await project(payload)
        assert len(assessment.evidence) == 1 and assessment.evidence[0].tool_id == "monitor.event.query"
        encoded = json.dumps(DbaAnswerComposeHandler._model_assessment_payload(assessment), ensure_ascii=False)
        assert "monitor.event.query" in encoded and "监控告警本身不是数据库根因" in encoded
        assert "EVENT_HISTORY_INCOMPLETE" in assessment.evidence[0].provenance.quality_flags
    asyncio.run(run())


@pytest.mark.parametrize("field", ["warnings", "rows", "query_scope", "truncated", "captured_at", "hash", "target", "run"])
def test_report_source_replay_rejects_forged_event_or_scope_even_without_investigation_record(field):
    async def run():
        payload, _, _ = await observe()
        assessment, uow, current, raw = await project(payload)
        frozen = {"evidence": [item.model_dump(mode="json") for item in assessment.evidence]}
        assert await validate_source_history_facts(uow=uow, run=current, payload=frozen)
        if field == "warnings": frozen["evidence"][0]["warnings"] = []
        elif field == "rows": frozen["evidence"][0]["rows"][0][2] = "forged"
        elif field == "query_scope": frozen["evidence"][0]["measurement_semantics"] = "HISTORICAL_SAMPLES"
        elif field == "truncated": frozen["evidence"][0]["truncated"] = True
        elif field == "captured_at": frozen["evidence"][0]["captured_at"] = NOW.isoformat()
        elif field == "hash": raw.content_hash = "b"*64
        elif field == "target": current.target_id = uuid7()
        else: raw.ops_run_id = uuid7()
        assert not await validate_source_history_facts(uow=uow, run=current, payload=frozen)
    asyncio.run(run())


def test_worker_event_only_binding_reaches_assessment_and_rejects_wrong_returned_window():
    async def run():
        req = event_request()
        for wrong_window in (False, True):
            value, _ = events_adapter()
            if wrong_window:
                result = await value.query_events(req)
                wrong = result.query.model_copy(update={"window_start": req.window_start-timedelta(seconds=1)})
                value.query_events = AsyncMock(return_value=result.model_copy(update={"query": wrong}))
            binding = {"binding_id": str(req.binding_id), "source_locator_key": GUID,
                "unsupported_metrics": [], "metrics": [], "effective_capabilities": ["event.query"],
                "source": {"source_id": str(value.context.source_id), "source_type": "OEM", "adapter_id": "oem",
                    "adapter_version": "1.0.0", "config_version": 1, "endpoint": "https://oem.example.com/em",
                    "declared_capabilities": {"event.query": {}}, "config": {}}}
            handler = EvidenceObserveHandler(
                diagnostic_source_registry=SimpleNamespace(create=lambda *a, **kw: value), secret_store=object())
            context = SimpleNamespace(target_id=req.target_id, task_key=f"observe:{req.binding_id}", trace_id="test",
                plan_snapshot={"monitoring": {"bindings": [binding], "max_response_bytes": 262144,
                    "window": {"start": req.window_start.isoformat(), "end": req.window_end.isoformat()}}})
            result = await handler.execute(context)
            if wrong_window:
                assert result.gaps and result.event_query is None and not result.active_alerts
            else:
                assert result.event_query is not None and result.active_alerts
                assessment, _, _, _ = await project(result)
                assert assessment.evidence[0].tool_id == "monitor.event.query"
    asyncio.run(run())

@pytest.mark.parametrize("tamper", [None, "remove_warning", "raw_source", "missing_source"])
def test_case_publish_and_retrieval_replay_original_event_without_investigation_record(tamper):
    from aiops_agent.application.case_source_validation import validate_case_sources

    async def run():
        payload, _, _ = await observe()
        assessment, uow, current, raw = await project(payload)
        frozen = {"evidence": [item.model_dump(mode="json") for item in assessment.evidence]}
        if tamper == "remove_warning":
            frozen["evidence"][0]["warnings"] = []
        if tamper == "raw_source":
            raw.payload_json["active_alerts"][0]["severity"] = "CRITICAL"
            raw.content_hash = canonical_hash(raw.payload_json)
        final_id = uuid7()
        final = SimpleNamespace(artifact_id=final_id, ops_run_id=current.ops_run_id, payload_json=frozen,
            schema_version="AIOPS_TURN_RESULT.v1", content_hash=canonical_hash(frozen), provenance_json={})

        async def get_artifact(*, artifact_id):
            if str(artifact_id) == str(final_id):
                return final
            return None if tamper == "missing_source" else raw

        uow.runs.get_artifact = AsyncMock(side_effect=get_artifact)
        current.agent_id, current.plan_snapshot_json = uuid7(), {}
        uow.runs.get_run_scoped = AsyncMock(return_value=current)
        source = SimpleNamespace(source_report_id=None, source_artifact_id=final_id,
            source_run_id=current.ops_run_id, content_hash=final.content_hash)
        uow.operations_knowledge = SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources": (source,)}))
        uow.agents = SimpleNamespace(
            get=AsyncMock(return_value=SimpleNamespace(current_version_id=uuid7())),
            version_has_target=AsyncMock(return_value=True), version_source_ids=AsyncMock(return_value=()))
        uow.turns = SimpleNamespace(get_run_link_by_ops_run_id=AsyncMock(return_value=None))
        assert await validate_case_sources(uow=uow, domain_id=uuid7(), version_id=uuid7()) == (tamper is None)
    asyncio.run(run())


def test_real_replanner_input_retains_event_scope_and_history_gap():
    from aiops_agent.application.investigation.reasoner import InvestigationReasoner

    async def run():
        payload, _, _ = await observe([])
        assessment, _, _, _ = await project(payload)
        model = SimpleNamespace(generate_structured=AsyncMock(side_effect=RuntimeError("captured")))
        prompts = SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="诊断", ref=lambda: {})))
        with pytest.raises(RuntimeError, match="captured"):
            await InvestigationReasoner(model, prompts).replan(
                content=(), conversation_context=(), target_context={"database_type": "ORACLE"},
                prompt_snapshot={}, source_run_evidence=None, task_frame={}, prior_plan={},
                assessment=assessment.model_dump(mode="json"), available_tools=(), available_playbooks=(),
                model_snapshot={"test": "model"}, deadline=None, idempotency_key="oem-events", revision_no=2)
        supplied = model.generate_structured.await_args.kwargs["input_payload"]
        encoded = json.dumps(supplied, ensure_ascii=False)
        assert "monitor.event.query" in encoded and "零行不等于历史无告警" in encoded
        assert "CURRENT_OPEN_INCIDENTS" in encoded and "HISTORICAL_SAMPLES" not in encoded
    asyncio.run(run())

@pytest.mark.parametrize("items", [[], [incident()]])
def test_observation_report_retains_event_fact_scope_and_original_reference_even_for_zero_rows(items):
    from aiops_agent.workers.evidence_handlers import EvidenceReportHandler

    async def run():
        payload, _, _ = await observe(items)
        artifact_id = str(uuid7())
        context = SimpleNamespace(target_id=payload.target_id,
            input_artifacts=({"artifact_id": artifact_id, "schema_version": "OBSERVATION_SET.v1",
                "payload": payload.model_dump(mode="json")},),
            plan_snapshot={"monitoring": {"window": {"start": (NOW-timedelta(minutes=15)).isoformat(),
                "end": NOW.isoformat()}, "catalog_version": "test", "catalog_hash": "a"*64}})
        report = await EvidenceReportHandler().execute(context)
        assert len(report.event_evidence) == 1
        fact = report.event_evidence[0]
        assert fact.row_count == len(items) and fact.evidence_ref == f"artifact:{artifact_id}#source-events"
        assert "零行不等于历史无告警" in fact.warnings[0]
        assert report.root_cause_level == "INCONCLUSIVE"
    asyncio.run(run())

@pytest.mark.parametrize("source_type", ["PROMETHEUS", "ZABBIX"])
def test_real_adapter_zero_event_query_scope_reaches_assessor(source_type):
    from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
    from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter
    from tests.unit.aiops_agent.test_aiops_oem_native_metrics import Response

    async def run():
        req = event_request()
        base, _ = events_adapter()
        context = base.context.model_copy(update={"source_type": source_type, "adapter_id": source_type.lower(),
            "endpoint": "https://source.example.com", "declared_capabilities": {"event.query": {}}})
        cls = PrometheusAdapter if source_type == "PROMETHEUS" else ZabbixAdapter
        source = cls(context=context, session=SimpleNamespace(
            get=lambda *a, **kw: Response({"status": "success", "data": {"alerts": []}})),
            request_timeout_seconds=5, webhook_replay_seconds=60)
        if source_type == "ZABBIX":
            source._rpc = AsyncMock(side_effect=[
                ([{"hostid": "1", "host": GUID}], "a"*64), ([], "b"*64)])
        result = await source.query_events(req)
        assert not result.gaps and result.query is not None and not result.events
        assert result.query.query_scope == ("CURRENT_ALERTS" if source_type == "PROMETHEUS" else "RECENT_PROBLEMS")
        payload = ObservationSet(target_id=req.target_id, binding_id=req.binding_id, source_id=context.source_id,
            event_query=result.query, active_alerts=result.events, collected_at=NOW)
        assessment, _, _, _ = await project(payload)
        fact = assessment.evidence[0]
        assert fact.row_count == 0 and "零行不等于历史无告警" in fact.warnings[0]
    asyncio.run(run())


@pytest.mark.parametrize("recovery,status,is_open", [("0", "FIRING", True), ("123", "RESOLVED", False),
    (None, "UNKNOWN", None)])
def test_zabbix_recent_recovered_problems_are_not_reported_as_current_firing(recovery, status, is_open):
    from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter

    async def run():
        base, _ = events_adapter()
        source = ZabbixAdapter(context=base.context.model_copy(update={"source_type": "ZABBIX", "adapter_id": "zabbix",
            "endpoint": "https://source.example.com"}), session=object(), request_timeout_seconds=5, webhook_replay_seconds=60)
        source._rpc = AsyncMock(side_effect=[([{"hostid": "1", "host": GUID}], "a"*64),
            ([{"eventid": "123", "name": "告警", "severity": "4", "clock": int(NOW.timestamp()), "r_eventid": recovery}], "b"*64)])
        result = await source.query_events(event_request())
        assert not result.gaps and result.events[0]["status"] == status and result.events[0]["is_open"] is is_open
        assert "r_eventid" in source._rpc.await_args.args[1]["output"]
    asyncio.run(run())


@pytest.mark.parametrize("payload", [
    {"status": "error", "error": "not authorized"}, {"status": "success", "data": {}},
    {"status": "success", "data": {"alerts": None}},
])
def test_prometheus_error_or_missing_alerts_is_not_a_zero_row_success(payload):
    from aiops_agent.adapters.diagnostic_sources.prometheus import PrometheusAdapter
    from tests.unit.aiops_agent.test_aiops_oem_native_metrics import Response

    async def run():
        base, _ = events_adapter()
        source = PrometheusAdapter(context=base.context.model_copy(update={"source_type": "PROMETHEUS", "adapter_id": "prometheus",
            "endpoint": "https://source.example.com"}),
            session=SimpleNamespace(get=lambda *a, **kw: Response(payload)), request_timeout_seconds=5, webhook_replay_seconds=60)
        result = await source.query_events(event_request())
        assert result.gaps and result.query is None and not result.events
    asyncio.run(run())

@pytest.mark.parametrize("recovery", [True, -1, "bad", "", []])
def test_zabbix_invalid_recovery_identity_never_produces_false_resolved_state(recovery):
    from aiops_agent.adapters.diagnostic_sources.zabbix import ZabbixAdapter

    async def run():
        base, _ = events_adapter()
        source = ZabbixAdapter(context=base.context.model_copy(update={"source_type": "ZABBIX", "adapter_id": "zabbix",
            "endpoint": "https://source.example.com"}), session=object(), request_timeout_seconds=5, webhook_replay_seconds=60)
        source._rpc = AsyncMock(side_effect=[([{"hostid": "1", "host": GUID}], "a"*64),
            ([{"eventid": "123", "clock": int(NOW.timestamp()), "r_eventid": recovery}], "b"*64)])
        result = await source.query_events(event_request())
        assert result.gaps and not result.events and result.query is None
    asyncio.run(run())

@pytest.mark.parametrize("alerts", [[], [{"source_type": "ZABBIX", "event_id": "1", "name": "历史告警"}]])
def test_absent_event_metadata_preserves_original_observation_and_provenance_hash(alerts):
    from aiops_agent.application.diagnosis.provenance import monitoring_provenance

    original = ObservationSet(
        target_id=str(uuid7()), binding_id=str(uuid7()), source_id=str(uuid7()), collected_at=NOW, active_alerts=tuple(alerts),
    ).model_dump(mode="json")
    original.pop("event_query", None)
    source = ObservationSet.model_validate(original)
    assert source.event_query is None and "event_query" not in source.model_dump(mode="json")
    assert canonical_hash(original) == canonical_hash(source.model_dump(mode="json"))
    assert monitoring_provenance(source).content_hash == canonical_hash(original)


def test_absent_event_facts_preserve_original_observation_report_hash():
    from aiops_agent.contracts.artifacts import ObserveReport

    report = ObserveReport(
        target_id=str(uuid7()), status="READY", window_start=NOW-timedelta(minutes=15), window_end=NOW,
        source_count=0, metric_count=0, alert_count=0, gap_count=0,
    )
    original = report.model_dump(mode="json")
    original.pop("event_evidence", None)
    restored = ObserveReport.model_validate(original)
    assert not restored.event_evidence and "event_evidence" not in restored.model_dump(mode="json")
    assert canonical_hash(original) == canonical_hash(restored.model_dump(mode="json"))
