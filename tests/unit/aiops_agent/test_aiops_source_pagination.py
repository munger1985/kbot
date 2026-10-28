"""原生来源续取贯穿冻结、Adapter、Worker 与证据边界的回归。"""

import asyncio
import hashlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.loki import LokiAdapter, _epoch_nanoseconds
from aiops_agent.adapters.diagnostic_sources.pagination import SourcePageRow, read_source_pages
from aiops_agent.application.investigation.query_freezing import prepare_source_queries
from aiops_agent.application.investigation.progress import action_fingerprint, drop_repeated_plan_actions
from aiops_agent.contracts.evidence import NativeMetricSamples, SourcePaginationObservation
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, LogEvidenceRequest
from platform_core.contracts.aiops.investigation import DiagnosticPagination, InvestigationPlanningOutput
from platform_core.identity import uuid7
from tests.unit.aiops_agent.test_aiops_native_metrics import adapter, item, request
from tests.unit.aiops_agent.test_aiops_log_evidence import _LokiResponse


def page_row(number, *, same_clock=False):
    return SourcePageRow(position=100 if same_clock else 100 + number, order=number,
        identity=str(number), content_hash=hashlib.sha256(str(number).encode()).hexdigest(), value=number)


@pytest.mark.parametrize("size", [0, 1, 3, 6, 9])
@pytest.mark.parametrize("page_size", [1, 2, 3])
@pytest.mark.parametrize("same_clock", [False, True])
def test_overlap_never_skips_same_timestamp_or_invents_complete_coverage(size, page_size, same_clock):
    calls = []
    records = [page_row(number, same_clock=same_clock) for number in range(size)]
    async def fetch(frontier, limit):
        calls.append((frontier, limit))
        return [row for row in records if row.position >= frontier][:limit], "a" * 64
    values, summary = asyncio.run(read_source_pages(fetch_page=fetch,
        pagination=DiagnosticPagination(page_size=page_size, max_pages=3), result_limit=100, start=100))
    assert values == tuple(range(min(size, page_size * 3)))
    assert (summary.stop_reason == "EXHAUSTED") == (size <= page_size * 3)
    assert len(calls) <= 3 and sum(summary.returned_page_rows) == len(values)
    assert summary.snapshot_consistency == "NOT_GUARANTEED"
    if same_clock:
        assert all(frontier == 100 for frontier, _ in calls)


def test_later_page_failure_retains_prior_evidence_and_explicit_source_error():
    fetch = AsyncMock(side_effect=[([page_row(0), page_row(1), page_row(2)], "a" * 64),
        DiagnosticSourceAdapterError("SOURCE_RESULT_TRUNCATED", "字节预算已用尽")])
    values, summary = asyncio.run(read_source_pages(fetch_page=fetch,
        pagination=DiagnosticPagination(page_size=2, max_pages=3), result_limit=6, start=100))
    assert values == (0, 1)
    assert summary.stop_reason == "SOURCE_ERROR" and summary.error_code == "SOURCE_RESULT_TRUNCATED"


@pytest.mark.parametrize("bad", ["foreign_boundary", "duplicate", "changed"])
def test_invalid_source_page_does_not_pollute_accepted_rows(bad):
    first = [page_row(0, same_clock=True), page_row(1, same_clock=True), page_row(2, same_clock=True)]
    corrupt = first[0]
    if bad == "foreign_boundary":
        corrupt = SourcePageRow(position=99, order=0, identity="0", content_hash="b" * 64, value=0)
        second = [corrupt]
    elif bad == "duplicate":
        second = [corrupt, corrupt]
    else:
        corrupt = SourcePageRow(position=100, order=0, identity="0", content_hash="b" * 64, value=0)
        second = [corrupt]
    fetch = AsyncMock(side_effect=[(first, "a" * 64), (second, "b" * 64)])
    values, summary = asyncio.run(read_source_pages(fetch_page=fetch,
        pagination=DiagnosticPagination(page_size=2, max_pages=3), result_limit=6, start=100))
    assert values == (0, 1) and summary.stop_reason == "SOURCE_ERROR"


def test_cancelled_paging_does_not_turn_into_success_or_restart():
    fetch = AsyncMock(side_effect=[([page_row(i) for i in range(3)], "a" * 64), asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(read_source_pages(fetch_page=fetch, pagination=DiagnosticPagination(page_size=2, max_pages=3),
            result_limit=6, start=100))
    assert fetch.await_count == 2


@pytest.mark.parametrize("size", [5, 6, 7])
def test_real_zabbix_history_uses_inclusive_seconds_and_retains_nanosecond_identity(size):
    req = request().model_copy(update={"pagination": DiagnosticPagination(page_size=2, max_pages=3)})
    value = adapter()
    clock = int(req.window_start.timestamp())
    rows = [{"itemid": "22", "clock": str(clock), "ns": str(index + 1),
        "value": str(90071992547409930 + index)} for index in range(size)]
    calls = []
    async def rpc(method, params, **kwargs):
        calls.append((method, params, kwargs))
        if method == "host.get":
            return [{"hostid": "11", "host": "bound-host"}], "a" * 64
        if method == "item.get":
            return [item()], "b" * 64
        assert params["time_from"] == clock and params["sortfield"] == ["clock", "ns"]
        return rows[:params["limit"]], "c" * 64
    value._rpc = AsyncMock(side_effect=rpc)
    samples = asyncio.run(value.query_native_metric(req, metric_name=item()["key_"], limit=100))
    assert [metadata.nanoseconds for metadata in samples.sample_metadata] == list(range(1, min(size, 6) + 1))
    assert samples.points[0][1] == "90071992547409930"
    assert samples.truncated == (size > 6)
    assert samples.pagination.stop_reason == ("ROW_LIMIT" if size > 6 else "EXHAUSTED")
    assert sum(call[2]["max_bytes"] for call in calls) <= req.max_response_bytes
    assert [call[1]["limit"] for call in calls if call[0] == "history.get"] == [3, 5, 7]


def make_loki(size, *, same_stamp=True):
    start = datetime(2026, 10, 28, 4, tzinfo=timezone.utc)
    req = LogEvidenceRequest(target_id=str(uuid7()), binding_id=str(uuid7()), source_locator_key="target",
        selector_labels={"instance": "bound"}, window_start=start, window_end=start + timedelta(minutes=1),
        max_entries=100, max_response_bytes=262144, trace_id="test",
        pagination={"page_size": 2, "max_pages": 3})
    base = _epoch_nanoseconds(start)
    records = [[str(base + (1 if same_stamp else index + 1)), f"line {index} password=secret-{index}"] for index in range(size)]
    calls = []
    class Session:
        def get(self, url, **kwargs):
            params = kwargs["params"]
            calls.append(params)
            selected = [row for row in records if params["start"] <= int(row[0]) <= params["end"]][:params["limit"]]
            return _LokiResponse({"status": "success", "data": {"resultType": "streams",
                "result": [{"stream": {"instance": "bound"}, "values": selected}]}})
    value = LokiAdapter(context=DiagnosticSourceContext(source_id=str(uuid7()), source_type="LOKI", adapter_id="loki",
        adapter_version="1.0.0", config_version=1, endpoint="https://loki.example.com"),
        session=Session(), request_timeout_seconds=5, webhook_replay_seconds=60)
    return value, req, calls


@pytest.mark.parametrize("same_stamp", [False, True])
@pytest.mark.parametrize("size", [0, 5, 6, 7])
def test_real_loki_pages_keep_nanosecond_boundaries_and_redaction(size, same_stamp):
    value, req, calls = make_loki(size, same_stamp=same_stamp)
    result = asyncio.run(value.query_logs(req))
    assert len(result.entries) == min(size, 6)
    assert not any("secret-" in entry.line for entry in result.entries)
    assert len({entry.entry_fingerprint for entry in result.entries}) == len(result.entries)
    assert all(entry.timestamp_ns is not None for entry in result.entries)
    assert result.truncated == (size > 6)
    assert all(call["direction"] == "forward" for call in calls)
    assert len(calls) <= 3 and sum(result.provenance["source_pagination"]["returned_page_rows"]) == len(result.entries)
    if size > 6:
        assert result.gaps[0].code == "SOURCE_PAGINATION_ROW_LIMIT"


def test_loki_second_page_byte_failure_preserves_first_page():
    value, req, _ = make_loki(7)
    original = value._response_json
    async def reader(response, **kwargs):
        if reader.calls:
            raise DiagnosticSourceAdapterError("SOURCE_RESULT_TRUNCATED", "字节预算已用尽")
        reader.calls += 1
        return await original(response, **kwargs)
    reader.calls = 0
    value._response_json = reader
    result = asyncio.run(value.query_logs(req))
    assert len(result.entries) == 2 and result.truncated
    assert result.gaps[0].code == "SOURCE_RESULT_TRUNCATED"


def test_source_paging_plan_is_frozen_and_changing_page_budget_does_not_bypass_repeat_guard():
    source = InvestigationPlanningOutput.model_validate({
        "input_envelope": {"materials": [{"item_no": 1, "material_kind": "QUESTION", "summary": "历史取证", "confidence": 1}], "explicit_question": "核验历史"},
        "task_frame": {"objectives": ["DIAGNOSE"], "problem_statement": "历史取证", "success_criteria": ["补齐证据"]},
        "plan": {"revision_no": 1, "actions": [{
            "action_id": "a1", "question": "读取历史", "tool_id": "monitor.query_native_metric",
            "input": {"metric_name": "business.calls"}, "expected_evidence_kind": "METRIC_SAMPLES",
            "measurement_semantics": "HISTORICAL_SAMPLES", "evidence_need": "需要更多样本", "decision_impact": "验证负载",
            "pagination": {"page_size": 2, "max_pages": 3},
        }]},
    })
    planned, frozen = prepare_source_queries(source)
    assert frozen["native_metric_queries"][0]["pagination"] == {"page_size": 2, "max_pages": 3}
    action = planned.plan.actions[0]
    altered = action.model_copy(update={"pagination": DiagnosticPagination(page_size=3, max_pages=3)})
    assert drop_repeated_plan_actions(planned.plan.model_copy(update={"actions": (altered,)}),
        executed_fingerprints=(action_fingerprint(action.tool_id, action.input, paged=True),))[1] == ("a1",)
    loki = action.model_copy(update={"tool_id": "loki.query_range", "input": {"query": "${binding_selector}"}})
    _, log_frozen = prepare_source_queries(source.model_copy(update={"plan": source.plan.model_copy(update={"actions": (loki,)})}))
    assert log_frozen["ad_hoc_log_queries"][0]["pagination"] == {"page_size": 2, "max_pages": 3}
    with pytest.raises(ValueError):
        prepare_source_queries(source.model_copy(update={"plan": source.plan.model_copy(update={"actions": (action, loki)})}))


@pytest.mark.parametrize("change", [
    {"returned_page_rows": [3]}, {"returned_page_rows": [True]}, {"response_hashes": ["not-hash"]},
    {"stop_reason": "ROW_LIMIT"}, {"stop_reason": "PAGE_LIMIT"},
    {"stop_reason": "SOURCE_ERROR"}, {"error_code": "invented"}, {"snapshot_consistency": "COMPLETE"},
])
def test_summary_rejects_forged_source_coverage(change):
    with pytest.raises(ValueError):
        SourcePaginationObservation.model_validate({"page_size": 2, "max_pages": 3, "result_limit": 6,
            "returned_page_rows": [2], "response_hashes": ["a" * 64], "stop_reason": "EXHAUSTED", **change})


def test_zabbix_second_precision_api_does_not_lose_valid_fractional_window_samples():
    req = request().model_copy(update={"pagination": DiagnosticPagination(page_size=2, max_pages=4)})
    req = req.model_copy(update={"window_start": req.window_start + timedelta(microseconds=500000),
        "window_end": req.window_start + timedelta(seconds=1, microseconds=500000)})
    clock = int(req.window_start.timestamp())
    raw = [(clock, 0), (clock, 400000000), (clock, 600000000), (clock, 900000000),
        (clock + 1, 100000000), (clock + 1, 900000000)]
    rows = [{"itemid": "22", "clock": str(seconds), "ns": str(ns), "value": str(ns)} for seconds, ns in raw]
    value = adapter()
    async def rpc(method, params, **kwargs):
        if method == "host.get":
            return [{"hostid": "11", "host": "bound-host"}], "a" * 64
        if method == "item.get":
            return [item()], "b" * 64
        return [row for row in rows if int(row["clock"]) >= params["time_from"]][:params["limit"]], "c" * 64
    value._rpc = AsyncMock(side_effect=rpc)
    samples = asyncio.run(value.query_native_metric(req, metric_name=item()["key_"], limit=100))
    assert [value for _, value in samples.points] == ["600000000", "900000000", "100000000"]
    assert not samples.truncated and samples.pagination.returned_page_rows == (0, 2, 1)


def native_payload():
    from aiops_agent.contracts.evidence import ObservationSet
    req = request()
    samples = NativeMetricSamples(action_id="a1", target_id=req.target_id, binding_id=req.binding_id,
        source_id=str(uuid7()), metric_name="business.calls", native_value_type="3", source_version="1:1",
        window_start=req.window_start, window_end=req.window_end, captured_at=req.window_end,
        points=((req.window_start, "90071992547409933"), (req.window_end, "90071992547409934")),
        response_sha256="a" * 64,
        pagination={"page_size": 2, "max_pages": 3, "result_limit": 6,
            "returned_page_rows": [2], "response_hashes": ["a" * 64], "stop_reason": "EXHAUSTED"})
    return ObservationSet(target_id=req.target_id, binding_id=req.binding_id, source_id=samples.source_id,
        native_metrics=(samples,), collected_at=req.window_end)


async def project_source(payload):
    from aiops_agent.workers.handlers import TaskExecutionContext
    from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
    from aiops_agent.playbooks import canonical_hash
    identity, run_id = uuid7(), uuid7()
    artifact = SimpleNamespace(artifact_id=identity, ops_run_id=run_id, schema_version=payload.schema_version,
        trust_level="SOURCE_VERIFIED", payload_json=payload.model_dump(mode="json"),
        content_hash=canonical_hash(payload.model_dump(mode="json")))
    result = await DbaEvidenceAssessmentHandler().execute(TaskExecutionContext(
        run_id=str(run_id), task_id=str(uuid7()), task_key="assess", target_id=payload.target_id,
        agent_id=str(uuid7()), actor_id="tester", trigger_type="USER", trace_id="test", attempt=1,
        deadline_at=None, plan_snapshot={"answer_context": {"task_frame": {"objectives": ["DIAGNOSE"]}}},
        policy_snapshot={}, input_artifacts=({"artifact_id": str(identity), "schema_version": payload.schema_version,
            "payload": artifact.payload_json},)))
    uow = SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)))
    return result, uow, SimpleNamespace(ops_run_id=run_id, target_id=payload.target_id), artifact


@pytest.mark.parametrize("source_kind", ["native", "loki"])
def test_original_source_pages_replay_in_formal_report_and_case_evidence(source_kind):
    from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
    async def run():
        if source_kind == "native":
            payload = native_payload()
        else:
            value, req, _ = make_loki(7)
            payload = await value.query_logs(req)
        result, uow, current, artifact = await project_source(payload)
        facts = [fact for fact in result.evidence if fact.tool_id in {"monitor.query_native_metric", "loki.query_range"}]
        assert len(facts) == 1 and facts[0].warnings
        if source_kind == "native":
            assert facts[0].source_pagination == payload.native_metrics[0].pagination
        original = {"evidence": [facts[0].model_dump(mode="json")]}
        assert await validate_investigation_sources(uow=uow, run=current, payload=original)
        for change in ("rows", "truncated", "page_budget", "source_hash", "source_run"):
            copy = {"evidence": [facts[0].model_dump(mode="json")]}
            if change == "rows":
                copy["evidence"][0]["rows"][0][0] = datetime.now(timezone.utc).isoformat()
            elif change == "truncated":
                copy["evidence"][0]["truncated"] = not facts[0].truncated
            elif change == "page_budget":
                if source_kind == "native":
                    copy["evidence"][0]["source_pagination"]["max_pages"] = 4
                else:
                    copy["evidence"][0]["warnings"] = []
            elif change == "source_hash":
                artifact.content_hash = "b" * 64
            else:
                artifact.ops_run_id = uuid7()
            assert not await validate_investigation_sources(uow=uow, run=current, payload=copy)
            from aiops_agent.playbooks import canonical_hash
            artifact.content_hash = canonical_hash(artifact.payload_json)
            artifact.ops_run_id = current.ops_run_id
    asyncio.run(run())


@pytest.mark.parametrize("source_kind", ["native", "loki"])
@pytest.mark.parametrize("corrupt", [None, "missing_receipt", "wrong_budget", "wrong_target"])
def test_formal_worker_checks_source_identity_and_frozen_page_budget(source_kind, corrupt):
    from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler, LogEvidenceHandler
    from aiops_agent.monitoring import LogQueryPolicy, LogQueryPolicySnapshot
    async def run():
        if source_kind == "native":
            payload = native_payload()
            samples = payload.native_metrics[0]
            if corrupt == "missing_receipt":
                samples = samples.model_copy(update={"pagination": None})
            elif corrupt == "wrong_budget":
                samples = samples.model_copy(update={"pagination": samples.pagination.model_copy(update={"max_pages": 4})})
            elif corrupt == "wrong_target":
                samples = samples.model_copy(update={"target_id": uuid7()})
            api = SimpleNamespace(query_native_metric=AsyncMock(return_value=samples))
            handler = EvidenceObserveHandler(diagnostic_source_registry=SimpleNamespace(create=lambda *a, **kw: api), secret_store=object())
            prefix, capability, source_type = "observe", "item.history", "ZABBIX"
            window_start, window_end = samples.window_start, samples.window_end
            query = {"native_metric_queries": [{"action_id": "a1", "metric_name": samples.metric_name,
                "window_seconds": 900, "limit": 100, "pagination": {"page_size": 2, "max_pages": 3}}]}
        else:
            value, req, _ = make_loki(7)
            payload = await value.query_logs(req)
            if corrupt == "missing_receipt":
                payload = payload.model_copy(update={"provenance": {}})
            elif corrupt == "wrong_budget":
                receipt = dict(payload.provenance["source_pagination"])
                receipt["max_pages"] = 4
                payload = payload.model_copy(update={"provenance": {**payload.provenance, "source_pagination": receipt}})
            elif corrupt == "wrong_target":
                payload = payload.model_copy(update={"target_id": uuid7()})
            api = SimpleNamespace(query_logs=AsyncMock(return_value=payload))
            handler = LogEvidenceHandler(diagnostic_source_registry=SimpleNamespace(create=lambda *a, **kw: api), secret_store=object())
            prefix, capability, source_type = "log", "log.query", "LOKI"
            window_start, window_end = req.window_start, req.window_end
            query = {"ad_hoc_log_queries": [{"action_id": "a1",
                "validated_query": LogQueryPolicy(LogQueryPolicySnapshot()).validate("${binding_selector}", window_seconds=60).model_dump(mode="json"),
                "pagination": {"page_size": 2, "max_pages": 3}}]}
        source_id = samples.source_id if source_kind == "native" else value.context.source_id
        target_id = payload.target_id
        if source_kind == "loki":
            target_id = req.target_id
        source = {"source_id": str(source_id), "source_type": source_type,
            "adapter_id": source_type.lower(), "adapter_version": "1.0.0", "config_version": 1,
            "endpoint": "https://source.example.com", "declared_capabilities": {capability: {}}, "config": {}}
        binding_id = payload.binding_id
        context = SimpleNamespace(target_id=target_id, task_key=f"{prefix}:{binding_id}", trace_id="test",
            plan_snapshot={"monitoring": {"window": {"start": window_start.isoformat(), "end": window_end.isoformat()},
                "max_response_bytes": 262144, "bindings": [{"binding_id": str(binding_id),
                    "source_locator_key": "bound", "source_locator": {"labels": {"instance": "bound"}},
                    "source": source, "effective_capabilities": [capability], "unsupported_metrics": []}], **query}})
        result = await handler.execute(context)
        if corrupt:
            assert not (result.native_metrics if source_kind == "native" else result.entries)
            assert result.gaps[0].code == "SOURCE_RESPONSE_INVALID"
        elif source_kind == "native":
            assert result.native_metrics[0].pagination == samples.pagination
            api.query_native_metric.assert_awaited_once()
        else:
            assert len(result.entries) == 6 and result.truncated
            assert result.provenance["source_queries"][0]["provenance"]["source_pagination"]["stop_reason"] == "ROW_LIMIT"
    asyncio.run(run())
