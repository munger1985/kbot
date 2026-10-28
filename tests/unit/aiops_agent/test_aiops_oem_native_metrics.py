"""OEM官方目录/历史合同及真实生产Worker、回答证据、报告/案例来源回放。"""

import asyncio
import copy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem import OEMAdapter
from aiops_agent.adapters.diagnostic_sources.oem_native import metric_selector, parse_metric_selector
from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.diagnosis.source_validation import validate_investigation_sources
from aiops_agent.application.investigation.discovery import available_tools
from aiops_agent.contracts.evidence import NativeMetricSamples, ObservationSet
from aiops_agent.domain.evidence.source_history import NATIVE_COLUMNS, native_history_rows
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, MetricsDiscoveryRequest, NativeMetricQueryRequest, SourceHealthRequest
from aiops_agent.playbooks import canonical_hash
from aiops_agent.workers.evidence_handlers import EvidenceObserveHandler
from aiops_agent.workers.handlers import TaskExecutionContext
from aiops_agent.workers.turn_answer_handlers import DbaEvidenceAssessmentHandler
from platform_core.contracts.aiops.playbooks import DbaCapabilitySnapshot
from platform_core.identity import uuid7


GUID = "CF99A10F233254B78ED96ED1B5C15140"
NOW = datetime(2026, 10, 28, 7, tzinfo=timezone.utc)


def request(discovery=False, **changes):
    cls = MetricsDiscoveryRequest if discovery else NativeMetricQueryRequest
    return cls(action_id="a1", target_id=str(uuid7()), binding_id=str(uuid7()),
        source_locator_key=GUID, window_start=NOW-timedelta(minutes=15), window_end=NOW,
        trace_id="test").model_copy(update=changes)


def metadata(**changes):
    return {"name": "Filesystems", "displayName": "文件系统", "keys": [{"name": "mountPoint"}],
        "metrics": [{"name": "size", "dataType": "NUMBER", "unitDisplayName": "MB", "displayName": "容量"},
                    {"name": "fileSystem", "dataType": "STRING"}], **changes}


def collection(items, **changes):
    return {"count": len(items), "items": items, "links": {}, **changes}


def series(mount="/", **changes):
    return {"targetName": "mes-db", "targetTypeName": "oracle_database", "targetId": GUID,
        "metricGroupName": "Filesystems", "metricName": "size",
        "keyValues": [{"name": "mountPoint", "value": mount}],
        "datapoints": [["2026-10-28T06:50:00.000Z", "90071992547409931"]], **changes}


class Response:
    def __init__(self, payload, status=200):
        self.status = status
        raw = json.dumps(payload, ensure_ascii=False).encode()
        async def chunks(size):
            for offset in range(0, len(raw), size):
                yield raw[offset:offset+size]
        self.content = SimpleNamespace(iter_chunked=chunks)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


def adapter(*payloads, credentials=None, statuses=None):
    replies, calls = list(payloads), []
    statuses = list(statuses or [200]*len(replies))
    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert replies, "不允许未声明的额外网络调用"
        return Response(replies.pop(0), statuses.pop(0))
    value = OEMAdapter(context=DiagnosticSourceContext(
        source_id=str(uuid7()), source_type="OEM", adapter_id="oem", adapter_version="1.0.0",
        config_version=1, endpoint="https://oem.example.com/em",
        declared_capabilities={"metric.discover": {}, "metric.query_range": {}},
        credentials=credentials or {},
    ), session=SimpleNamespace(get=get), request_timeout_seconds=5, webhook_replay_seconds=60)
    return value, calls


def native_adapter(payload=None, meta=None, **kwargs):
    return adapter({"targetName": "mes-db", "targetType": "oracle_database"},
        meta or metadata(), collection([series()]) if payload is None else payload, **kwargs)


def test_native_discovery_uses_official_target_directory_and_does_not_guess_counter_unit_or_coverage():
    value, calls = adapter({"targetName": "mes-db", "targetType": "oracle_database"},
        collection([metadata()]))
    result = asyncio.run(value.discover_metrics(request(True)))
    assert [entry.name for entry in result.entries] == ["Filesystems/size", "Filesystems/fileSystem"]
    first, text = result.entries
    assert first.unit == "MB" and first.metric_type == "UNKNOWN" and first.metadata_state == "VERIFIED"
    assert first.label_names == ("mountPoint",) and first.query_supported and not text.query_supported
    assert calls[0][0].endswith("/em/api/targets/"+GUID)
    assert calls[1][0].endswith("/metricGroups") and calls[1][1]["params"] == {"include": "metrics"}
    assert all(not kwargs["allow_redirects"] for _, kwargs in calls)
    assert result.warnings and not result.truncated


@pytest.mark.parametrize("scope,locator", [("HOST", GUID), ("DATABASE", "mes-db"), ("DATABASE", "../other"),
    ("DATABASE", "https://other.example/targets"), ("DATABASE", GUID+"00")])
def test_discovery_rejects_unbound_host_and_non_guid_before_network(scope, locator):
    value, calls = adapter()
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request(True, scope=scope, source_locator_key=locator)))
    assert not calls


@pytest.mark.parametrize("selector", ["", "Filesystems", "Filesystems/size/other", "../size", "./size",
    "Filesystems/%73ize", "Filesystems/%0A", "Filesystems/%ZZ", "Filesystems/", "/size", "x"*257])
def test_native_selector_cannot_smuggle_url_path_or_noncanonical_encoding(selector):
    with pytest.raises(DiagnosticSourceAdapterError):
        parse_metric_selector(selector)


@pytest.mark.parametrize("group,metric", [("Load","cpuUtil"), ("业务/组","吞吐率"), ("Foo bar","A?B")])
def test_native_selector_roundtrip_keeps_native_names_without_using_them_as_urls(group, metric):
    assert parse_metric_selector(metric_selector(group, metric)) == (group, metric)


def test_discovery_marks_both_entry_limit_and_native_next_page_without_following_links():
    for changes in ({}, {"links": {"next": {"href": "https://untrusted.invalid/"}}}):
        value, calls = adapter({"targetName": "mes-db", "targetType": "oracle_database"},
            collection([metadata()], **changes))
        result = asyncio.run(value.discover_metrics(request(True, limit=1)))
        assert len(result.entries) == 1 and result.truncated and len(calls) == 2


@pytest.mark.parametrize("bad", [
    {"items": []}, collection([metadata()], count=999),
    collection([metadata(keys=[{"name": "mountPoint"},{"name":"mountPoint"}])]),
    collection([metadata(metrics=[{"name":"size","dataType":"NUMBER"}]*2)]),
    collection([metadata(metrics=[{"name":"size","dataType":"NUMBER","unitDisplayName":"x"*65}])]),
    collection([metadata(), {"name":"bad","metrics":None}]),
])
def test_discovery_rejects_malformed_or_ambiguous_metadata_as_a_whole(bad):
    value, _ = adapter({"targetName":"mes-db","targetType":"oracle_database"}, bad)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request(True)))


def test_native_history_preserves_exact_values_units_dimensions_and_unknown_retention_rollup():
    value, calls = native_adapter(collection([series("/"), series("/data")]))
    result = asyncio.run(value.query_native_metric(request(), metric_name="Filesystems/size", limit=100))
    assert result.points[0][1] == "90071992547409931" and result.unit == "MB"
    assert result.sample_dimensions == ({"mountPoint":"/"},{"mountPoint":"/data"})
    assert result.source_type == "OEM" and result.history_resolution == "PROVIDER_MANAGED"
    assert result.pagination is None and not result.truncated
    params = calls[-1][1]["params"]
    assert params == {"targetId":GUID, "metricGroupName":"Filesystems", "metricName":"size",
        "timeCollectedGreaterThanOrEqualTo":"2026-10-28T06:45:00.000Z",
        "timeCollectedLessThan":"2026-10-28T07:00:00.000Z", "limit":20}
    assert calls[-1][0].endswith("/em/api/metricTimeSeries")


@pytest.mark.parametrize("changes", [
    {"targetName":"other"}, {"targetTypeName":"host"}, {"targetId":"0"*32},
    {"metricGroupName":"Other"}, {"metricName":"available"}, {"keyValues":[]},
    {"keyValues":[{"name":"mountPoint","value":"/"},{"name":"mountPoint","value":"/data"}]},
    {"datapoints":[["2026-10-28T06:50:00.000Z","NaN"]]},
    {"datapoints":[["2026-10-28T06:50:00.000Z","Infinity"]]},
    {"datapoints":[["2026-10-28T06:50:00.000Z","1e10000"]]},
    {"datapoints":[["2026-10-28T06:50:00.000Z",True]]},
    {"datapoints":[["2026-10-28T06:50:00","1"]]},
    {"datapoints":[["2026-10-28T07:00:00.000Z","1"]]},
    {"datapoints":[["2026-10-28T06:40:00.000Z","1"]]},
    {"datapoints":[["2026-10-28T06:50:00.000Z","1"],["2026-10-28T06:50:00.000Z","2"]]},
])
def test_native_history_rejects_wrong_scope_invalid_time_value_and_duplicate_points(changes):
    value, _ = native_adapter(collection([series(), series("/bad",**changes)]))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_native_metric(request(),metric_name="Filesystems/size",limit=1))


def test_native_history_empty_zero_and_budget_truncation_are_distinct_and_no_next_url_is_followed():
    for payload, expected, truncated in [
        (collection([]), (), False),
        (collection([series(datapoints=[["2026-10-28T06:50:00.000Z","0"]])]), ("0",), False),
        (collection([series(),series("/data")]), ("90071992547409931",), True),
        (collection([],links={"next":{"href":"http://untrusted.invalid/"}}), (), True),
    ]:
        value, calls = native_adapter(payload)
        result = asyncio.run(value.query_native_metric(request(),metric_name="Filesystems/size",limit=1))
        assert tuple(point[1] for point in result.points) == expected and result.truncated == truncated
        assert len(calls) == 3


def test_native_microsecond_window_is_filtered_after_official_millisecond_bounds():
    req = request(window_start=NOW-timedelta(seconds=1,microseconds=500), window_end=NOW-timedelta(microseconds=500))
    points = [["2026-10-28T06:59:58.999Z","1"],["2026-10-28T06:59:59.000Z","2"],
              ["2026-10-28T06:59:59.999Z","3"]]
    value, calls = native_adapter(collection([series(datapoints=points)]))
    result = asyncio.run(value.query_native_metric(req,metric_name="Filesystems/size",limit=100))
    assert tuple(v for _,v in result.points) == ("2","3")
    assert calls[-1][1]["params"]["timeCollectedLessThan"] == "2026-10-28T07:00:00.000Z"


@pytest.mark.parametrize("changes", [
    {"metrics":[{"name":"size","dataType":"STRING"}]},
    {"metrics":[{"name":"size","dataType":"FUTURE_TYPE"}]},
    {"metrics":[]}, {"name":"Other"}, {"keys":[{"name":"mountPoint"}]*2},
])
def test_history_rechecks_actual_native_metadata_and_never_guesses_text_as_numeric(changes):
    value, calls = native_adapter(meta=metadata(**changes))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.query_native_metric(request(),metric_name="Filesystems/size",limit=100,allow_text=True))
    assert len(calls) == 2


def test_history_native_pagination_is_explicitly_unsupported_and_does_not_execute_any_request():
    req = NativeMetricQueryRequest.model_validate({**request().model_dump(),
        "pagination":{"page_size":2,"max_pages":2}})
    value, calls = native_adapter()
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.query_native_metric(req,metric_name="Filesystems/size",limit=100))
    assert error.value.code == "SOURCE_QUERY_UNSUPPORTED" and not calls


@pytest.mark.parametrize("status,code", [(302,"SOURCE_RESPONSE_INVALID"),(401,"SOURCE_AUTH_FAILED"),
    (403,"SOURCE_AUTH_FAILED"),(404,"SOURCE_QUERY_UNSUPPORTED"),(429,"SOURCE_RATE_LIMITED"),(503,"SOURCE_UNREACHABLE")])
def test_official_endpoint_errors_are_not_empty_success_and_auth_cannot_redirect(status, code):
    value, calls = adapter({},statuses=[status])
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.discover_metrics(request(True)))
    assert error.value.code == code and len(calls) == 1 and not calls[0][1]["allow_redirects"]


def test_source_byte_budget_and_actual_basic_auth_apply_to_all_requests():
    value, calls = adapter({"targetName":"mes-db","targetType":"oracle_database"},
        collection([metadata(displayName="x"*3000)]),credentials={"username":"reader","password":"example"})
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.discover_metrics(request(True,max_response_bytes=1024)))
    assert error.value.code == "SOURCE_RESULT_TRUNCATED"
    assert all(kwargs["auth"].login == "reader" and not kwargs["headers"] for _,kwargs in calls)


def test_health_uses_real_readonly_api_and_auth_failure_is_not_connected():
    value,calls=adapter(collection([]),credentials={"username":"reader","password":"example"})
    result=asyncio.run(value.health_check(SourceHealthRequest(trace_id="test")))
    assert result.healthy and calls[0][0].endswith("/em/api/targets")
    value,_=adapter({},statuses=[401])
    assert not asyncio.run(value.health_check(SourceHealthRequest(trace_id="test"))).healthy


@pytest.mark.parametrize("capabilities,discover,native", [
    ((),False,False),(("metric.discover",),True,False),(("metric.query_range",),False,True),
    (("item.history",),False,False),
])
def test_oem_tools_require_current_source_capability_not_unrelated_history_permission(capabilities,discover,native):
    snapshot = DbaCapabilitySnapshot(agent_id=str(uuid7()),agent_version_id=str(uuid7()),target_id=str(uuid7()),
        database_type="ORACLE",database_version="19",target_enabled=True,target_reachable=True,
        source_snapshots=({"source_id":str(uuid7()),"source_type":"OEM","enabled":True,
            "reachable":True,"capabilities":capabilities},))
    tools={item["tool_id"] for item in available_tools(SimpleNamespace(discover_tools=lambda _:()),snapshot)}
    assert "monitor.query_range" not in tools
    assert ("monitor.discover_metrics" in tools) == discover
    assert ("monitor.query_native_metric" in tools) == native
    assert "metric.discover" in DiagnosticSourceAdapterCatalog().describe_source_type(source_type="OEM").capabilities


@pytest.mark.parametrize("discovery", [True,False])
def test_adapter_also_rechecks_declared_capabilities_before_network(discovery):
    value,calls=native_adapter()
    value.context=value.context.model_copy(update={"declared_capabilities":{}})
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.discover_metrics(request(True)) if discovery else value.query_native_metric(request(),metric_name="Filesystems/size",limit=100))
    assert not calls


async def project(payload):
    run_id,artifact_id=uuid7(),uuid7()
    result=await DbaEvidenceAssessmentHandler().execute(TaskExecutionContext(
        run_id=str(run_id),task_id=str(uuid7()),task_key="assess",target_id=payload.target_id,
        agent_id=str(uuid7()),actor_id="tester",trigger_type="USER",trace_id="test",attempt=1,deadline_at=None,
        plan_snapshot={"answer_context":{"task_frame":{"objectives":["DIAGNOSE"]}}},policy_snapshot={},
        input_artifacts=({"artifact_id":str(artifact_id),"schema_version":"OBSERVATION_SET.v1",
            "payload":payload.model_dump(mode="json")},)))
    artifact=SimpleNamespace(artifact_id=artifact_id,ops_run_id=run_id,schema_version="OBSERVATION_SET.v1",
        trust_level="SOURCE_VERIFIED",payload_json=payload.model_dump(mode="json"),
        content_hash=canonical_hash(payload.model_dump(mode="json")))
    uow=SimpleNamespace(runs=SimpleNamespace(get_artifact=AsyncMock(return_value=artifact)))
    return result,uow,SimpleNamespace(ops_run_id=run_id,target_id=payload.target_id),artifact


@pytest.mark.parametrize("kind",["catalog","history"])
def test_original_scope_units_dimensions_rollup_and_warnings_replay_for_report_and_case(kind):
    async def run():
        req=request(kind=="catalog")
        if kind=="catalog":
            value,_=adapter({"targetName":"mes-db","targetType":"oracle_database"},collection([metadata()]))
            source=await value.discover_metrics(req)
            content={"metric_catalogs":(source,)}
        else:
            value,_=native_adapter(collection([series(),series("/data")]))
            source=await value.query_native_metric(req,metric_name="Filesystems/size",limit=100)
            content={"native_metrics":(source,)}
        payload=ObservationSet(target_id=req.target_id,binding_id=req.binding_id,source_id=value.context.source_id,
            collected_at=NOW,**content)
        assessment,uow,current,artifact=await project(payload)
        fact=assessment.evidence[0]
        assert fact.warnings
        if kind=="history":
            assert fact.rows[1][NATIVE_COLUMNS.index("dimensions")] == {"mountPoint":"/data"}
            assert fact.rows[0][NATIVE_COLUMNS.index("history_resolution")] == "PROVIDER_MANAGED"
            assert "PROVIDER_RETENTION_ROLLUP_POSSIBLE" in fact.provenance.quality_flags
        original={"evidence":[fact.model_dump(mode="json")]}
        assert await validate_investigation_sources(uow=uow,run=current,payload=original)
        for field in ("warnings","rows","truncated","source","run"):
            changed=copy.deepcopy(original)
            if field=="warnings": changed["evidence"][0]["warnings"]=[]
            elif field=="rows": changed["evidence"][0]["rows"][0][0]="forged"
            elif field=="truncated": changed["evidence"][0]["truncated"]=not fact.truncated
            elif field=="source": artifact.content_hash="b"*64
            else: artifact.ops_run_id=uuid7()
            assert not await validate_investigation_sources(uow=uow,run=current,payload=changed)
            artifact.ops_run_id=current.ops_run_id
            artifact.content_hash=canonical_hash(artifact.payload_json)
    asyncio.run(run())


@pytest.mark.parametrize("authorization",["metric.query_range","item.history"])
def test_worker_runs_native_oem_only_under_numeric_binding_permission(authorization):
    async def run():
        req=request()
        value,calls=native_adapter()
        binding={"binding_id":str(req.binding_id),"source_locator_key":GUID,"source_locator":{},
            "unsupported_metrics":[],"metrics":[],"effective_capabilities":[authorization],
            "source":{"source_id":str(value.context.source_id),"source_type":"OEM","adapter_id":"oem",
                "adapter_version":"1.0.0","config_version":1,"endpoint":"https://oem.example.com/em",
                "declared_capabilities":{authorization:{}},"config":{}}}
        handler=EvidenceObserveHandler(diagnostic_source_registry=SimpleNamespace(create=lambda *a,**kw:value),secret_store=object())
        context=SimpleNamespace(target_id=req.target_id,task_key=f"observe:{req.binding_id}",trace_id="test",
            plan_snapshot={"monitoring":{"window":{"start":req.window_start.isoformat(),"end":req.window_end.isoformat()},
                "bindings":[binding],"max_response_bytes":262144,
                "native_metric_queries":[{"action_id":"a1","metric_name":"Filesystems/size","window_seconds":900,"limit":100}]}})
        result=await handler.execute(context)
        if authorization=="metric.query_range":
            assert len(result.native_metrics)==1 and len(calls)==3
            assessment,_,_,_=await project(result)
            assert any(fact.tool_id=="monitor.query_native_metric" for fact in assessment.evidence)
        else:
            assert not result.native_metrics and not calls and result.gaps
    asyncio.run(run())


@pytest.mark.parametrize("changes", [{"history_resolution":"RAW"}, {"sample_dimensions":()},
    {"sample_kind":"TEXT"}, {"native_value_type":"STRING"},
    {"sample_dimensions":({"mountPoint":"x"*257},)}])
def test_contract_cannot_strip_oem_resolution_or_dimensions(changes):
    value,_=native_adapter()
    result=asyncio.run(value.query_native_metric(request(),metric_name="Filesystems/size",limit=100))
    with pytest.raises(ValueError):
        NativeMetricSamples.model_validate({**result.model_dump(),**changes})


@pytest.mark.parametrize("stage",["replan","answer"])
def test_actual_model_input_preserves_native_keys_and_retention_warning(stage):
    from aiops_agent.application.investigation.reasoner import InvestigationReasoner
    from aiops_agent.workers.turn_answer_handlers import DbaAnswerComposeHandler
    async def run():
        req=request()
        value,_=native_adapter()
        samples=await value.query_native_metric(req,metric_name="Filesystems/size",limit=100)
        source=ObservationSet(target_id=req.target_id,binding_id=req.binding_id,source_id=value.context.source_id,
            native_metrics=(samples,),collected_at=NOW)
        result,_,_,_=await project(source)
        if stage=="answer":
            supplied=DbaAnswerComposeHandler._model_assessment_payload(result)
        else:
            model=SimpleNamespace(generate_structured=AsyncMock(side_effect=RuntimeError("captured")))
            prompts=SimpleNamespace(resolve=AsyncMock(return_value=SimpleNamespace(content="诊断",ref=lambda:{})))
            with pytest.raises(RuntimeError,match="captured"):
                await InvestigationReasoner(model,prompts).replan(content=(),conversation_context=(),
                    target_context={"database_type":"ORACLE"},prompt_snapshot={},source_run_evidence=None,
                    task_frame={},prior_plan={},assessment=result.model_dump(mode="json"),
                    available_tools=(),available_playbooks=(),model_snapshot={"test":"model"},deadline=None,
                    idempotency_key="oem",revision_no=2)
            supplied=model.generate_structured.await_args.kwargs["input_payload"]
        encoded=json.dumps(supplied,ensure_ascii=False)
        assert "mountPoint" in encoded and "PROVIDER_MANAGED" in encoded and "小时或日均值" in encoded
    asyncio.run(run())


@pytest.mark.parametrize("kind",["catalog","history"])
@pytest.mark.parametrize("tamper",[None,"remove_warning","raw_source","missing_source"])
def test_actual_case_publish_retrieval_gate_reads_oem_originals_even_without_investigation_record(kind,tamper):
    from aiops_agent.application.case_source_validation import validate_case_sources
    async def run():
        req=request(kind=="catalog")
        if kind=="catalog":
            value,_=adapter({"targetName":"mes-db","targetType":"oracle_database"},collection([metadata()]))
            content={"metric_catalogs":(await value.discover_metrics(req),)}
        else:
            value,_=native_adapter()
            content={"native_metrics":(await value.query_native_metric(req,metric_name="Filesystems/size",limit=100),)}
        payload=ObservationSet(target_id=req.target_id,binding_id=req.binding_id,source_id=value.context.source_id,
            collected_at=NOW,**content)
        assessment,uow,current,raw=await project(payload)
        frozen={"evidence":[item.model_dump(mode="json") for item in assessment.evidence]}
        if tamper=="remove_warning": frozen["evidence"][0]["warnings"]=[]
        if tamper=="raw_source":
            if kind=="history": raw.payload_json["native_metrics"][0]["points"][0][1]="999"
            else: raw.payload_json["metric_catalogs"][0]["entries"][0]["name"]="forged"
            raw.content_hash=canonical_hash(raw.payload_json)
        final_id=uuid7()
        final=SimpleNamespace(artifact_id=final_id,ops_run_id=current.ops_run_id,payload_json=frozen,
            content_hash=canonical_hash(frozen),provenance_json={})
        async def get_artifact(*,artifact_id):
            if str(artifact_id)==str(final_id): return final
            return None if tamper=="missing_source" else raw
        uow.runs.get_artifact=AsyncMock(side_effect=get_artifact)
        current.agent_id,current.plan_snapshot_json=uuid7(),{}
        uow.runs.get_run_scoped=AsyncMock(return_value=current)
        source=SimpleNamespace(source_report_id=None,source_artifact_id=final_id,source_run_id=current.ops_run_id,
            content_hash=final.content_hash)
        uow.operations_knowledge=SimpleNamespace(get_version_relations=AsyncMock(return_value={"sources":(source,)}))
        uow.agents=SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(current_version_id=uuid7())),
            version_has_target=AsyncMock(return_value=True),version_source_ids=AsyncMock(return_value=()))
        uow.turns=SimpleNamespace(get_run_link_by_ops_run_id=AsyncMock(return_value=None))
        assert await validate_case_sources(uow=uow,domain_id=uuid7(),version_id=uuid7()) == (tamper is None)
    asyncio.run(run())
