"""官方关系读取的多层归属、过滤完整性、唯一 GUID 和网络安全边界。"""

import asyncio
import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem_exadata_inventory import ASSOCIATION_COLUMNS, DATA_SERVICE
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter, collection


ROOT, GRID, CELL = "a" * 32, "c" * 32, "b" * 32


def target(guid, name, kind):
    return {"id": guid, "name": name, "typeName": kind}


ROOT_ROW = target(ROOT, "测试整机", "oracle_dbmachine")
GRID_ROW = target(GRID, "测试 Grid", "oracle_exadata_grid")
CELL_ROW = target(CELL, "测试 Cell", "oracle_exadata")


def association_rows(*rows, **changes):
    return {"columnNames": sorted(ASSOCIATION_COLUMNS), "exceedsMaxRows": False,
        "data": [dict(zip(("TARGET_NAME", "TARGET_TYPE", "ASSOC_TARGET_NAME", "ASSOC_TARGET_TYPE", "ASSOC_TYPE"),
                          (*row, "internal_contains"))) for row in rows], **changes}


def tree_payloads():
    return [ROOT_ROW, association_rows(("测试整机", "oracle_dbmachine", "测试 Grid", "oracle_exadata_grid")),
        collection([GRID_ROW]), GRID_ROW,
        association_rows(("测试 Grid", "oracle_exadata_grid", "测试 Cell", "oracle_exadata")),
        collection([CELL_ROW]), CELL_ROW]


def test_verified_tree_preserves_grid_and_cell_and_uses_only_fixed_get_endpoints():
    value, calls = adapter(*tree_payloads(), credentials={"username": "reader", "password": "test-only"})
    root, members, digest = asyncio.run(value.exadata_inventory(ROOT))
    assert root["target_id"] == ROOT and len(digest) == 64
    assert [(row["target_id"], row["target_type"]) for row in members] == [(CELL, "oracle_exadata"), (GRID, "oracle_exadata_grid")]
    relations = [(url, kw) for url, kw in calls if url.endswith(DATA_SERVICE)]
    assert len(relations) == 2
    for url, kw in relations:
        assert url == "https://oem.example.com/em" + DATA_SERVICE
        assert kw["params"]["name"] == "TargetAssociations"
        assert "ASSOC_TYPE='internal_contains'" in kw["params"]["search"]
        assert kw["auth"].login == "reader" and not kw["headers"]
    assert all(not kw["allow_redirects"] for _, kw in calls)
    assert calls[2][1]["params"] == {"name": "测试 Grid", "typeName": "oracle_exadata_grid", "limit": 2}


def deployed_host_rows(**changes):
    return {"columnNames": sorted(ASSOCIATION_COLUMNS), "exceedsMaxRows": False, "data": [{
        "TARGET_NAME": "测试 DB Server", "TARGET_TYPE": "host", "ASSOC_TYPE": "deployed_on",
        "ASSOC_TARGET_NAME": "测试 ILOM", "ASSOC_TARGET_TYPE": "oracle_si_server_map", **changes}]}


def server_payloads():
    server = target("d" * 32, "测试 ILOM", "oracle_si_server_map")
    return [ROOT_ROW, association_rows(("测试整机", "oracle_dbmachine", "测试 Cell", "oracle_exadata"),
            ("测试整机", "oracle_dbmachine", "测试 ILOM", "oracle_si_server_map")),
            collection([CELL_ROW]), CELL_ROW, collection([server]), server]


def test_db_server_requires_confirmed_ilom_and_explicit_reverse_deployed_on():
    host = target("e" * 32, "测试 DB Server", "host")
    value, calls = adapter(*server_payloads(), deployed_host_rows(), collection([host]), host)
    _, members, _ = asyncio.run(value.exadata_inventory(ROOT))
    assert any(row["target_id"] == "e" * 32 and row["target_type"] == "host" for row in members)
    params = next(kw["params"] for url, kw in calls if url.endswith(DATA_SERVICE) and "deployed_on" in kw["params"]["search"])
    assert params["search"] == "ASSOC_TARGET_NAME='测试 ILOM',ASSOC_TARGET_TYPE='oracle_si_server_map',TARGET_TYPE='host',ASSOC_TYPE='deployed_on'"
    assert calls[-2][1]["params"] == {"name": "测试 DB Server", "typeName": "host", "limit": 2}


@pytest.mark.parametrize("changes", [
    {"ASSOC_TYPE": "hosted_by"}, {"ASSOC_TYPE": "internal_contains"},
    {"TARGET_TYPE": "oracle_database"}, {"ASSOC_TARGET_NAME": "其他 ILOM"},
    {"ASSOC_TARGET_TYPE": "oracle_si_netswitch"},
])
def test_wrong_reverse_direction_type_or_filter_never_adds_db_server(changes):
    value, calls = adapter(*server_payloads(), deployed_host_rows(**changes))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))
    assert len(calls) == 7


def test_ilom_without_deployed_host_is_not_agent_host_or_missing_inventory():
    payload = deployed_host_rows()
    payload["data"] = []
    value, _ = adapter(*server_payloads(), payload)
    _, members, _ = asyncio.run(value.exadata_inventory(ROOT))
    assert not any(row["target_type"] == "host" for row in members)


def test_ambiguous_multiple_hosts_on_one_ilom_fails_closed():
    payload = deployed_host_rows()
    payload["data"].append({**payload["data"][0], "TARGET_NAME": "其他主机"})
    value, _ = adapter(*server_payloads(), payload)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


def test_same_host_cannot_be_owned_by_two_confirmed_iloms():
    first = target("d" * 32, "测试 ILOM", "oracle_si_server_map")
    second = target("f" * 32, "其他 ILOM", "oracle_si_server_map")
    host = target("e" * 32, "测试 DB Server", "host")
    value, _ = adapter(ROOT_ROW, association_rows(
        ("测试整机", "oracle_dbmachine", "测试 Cell", "oracle_exadata"),
        ("测试整机", "oracle_dbmachine", "测试 ILOM", "oracle_si_server_map"),
        ("测试整机", "oracle_dbmachine", "其他 ILOM", "oracle_si_server_map")),
        collection([CELL_ROW]), CELL_ROW, collection([first]), first, collection([second]), second,
        deployed_host_rows(), collection([host]), host,
        deployed_host_rows(ASSOC_TARGET_NAME="其他 ILOM"), collection([host]), host)
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.exadata_inventory(ROOT))
    assert error.value.code == "SOURCE_ITEM_AMBIGUOUS"


@pytest.mark.parametrize("change", [
    {"exceedsMaxRows": True}, {"exceedsMaxRows": "false"}, {"exceedsMaxRows": 0},
    {"columnNames": ["TARGET_NAME"]}, {"columnNames": ["TARGET_NAME"] * 5},
    {"columnNames": [None] * 5}, {"data": None}, {"data": [None]},
    {"data": [{}]}, {"data": "not-rows"},
])
def test_truncation_or_malformed_rows_fail_closed_without_member_lookup(change):
    value, calls = adapter(ROOT_ROW, association_rows(**change))
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))
    assert len(calls) == 2


def test_missing_completeness_flag_is_not_assumed_complete():
    payload = association_rows()
    payload.pop("exceedsMaxRows")
    value, _ = adapter(ROOT_ROW, payload)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


@pytest.mark.parametrize("field,wrong", [
    ("TARGET_NAME", "其他整机"), ("TARGET_TYPE", "host"),
    ("ASSOC_TYPE", "hosted_by"), ("ASSOC_TYPE", "monitored_by"),
    ("ASSOC_TYPE", "stores_on"), ("ASSOC_TARGET_NAME", ""), ("ASSOC_TARGET_TYPE", "host\n"),
])
def test_filter_not_applied_or_non_containment_never_proves_membership(field, wrong):
    payload = association_rows(("测试整机", "oracle_dbmachine", "测试 Cell", "oracle_exadata"))
    payload["data"][0][field] = wrong
    value, calls = adapter(ROOT_ROW, payload)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))
    assert len(calls) == 2


@pytest.mark.parametrize("lookup", [collection([]), collection([GRID_ROW, GRID_ROW]),
    collection([GRID_ROW], links={"next": {"href": "https://untrusted.invalid"}}),
    collection([target(GRID, "同类型其他名称", "oracle_exadata_grid")]),
    collection([target(GRID, "测试 Grid", "host")]), collection([target("not-guid", "测试 Grid", "oracle_exadata_grid")])])
def test_member_lookup_requires_unique_exact_name_type_and_guid(lookup):
    payloads = tree_payloads()[:2] + [lookup]
    value, calls = adapter(*payloads)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))
    assert len(calls) == 3


@pytest.mark.parametrize("detail", [target("d" * 32, "测试 Grid", "oracle_exadata_grid"),
    target(GRID, "改名 Grid", "oracle_exadata_grid"), target(GRID, "测试 Grid", "host")])
def test_guid_detail_drift_is_rejected(detail):
    value, _ = adapter(*tree_payloads()[:3], detail)
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


def test_grid_without_cells_does_not_sign_usable_inventory():
    value, _ = adapter(*tree_payloads()[:4], association_rows())
    with pytest.raises(DiagnosticSourceAdapterError):
        asyncio.run(value.exadata_inventory(ROOT))


def test_duplicates_cycles_and_arbitrary_grid_descendants_are_rejected():
    for kind in ("oracle_exadata_grid", "oracle_dbmachine", "host"):
        rows = association_rows(("测试 Grid", "oracle_exadata_grid", "循环或越界", kind))
        value, calls = adapter(*tree_payloads()[:4], rows)
        with pytest.raises(DiagnosticSourceAdapterError):
            asyncio.run(value.exadata_inventory(ROOT))
        assert len(calls) == 5
    row = ("测试整机", "oracle_dbmachine", "测试 Cell", "oracle_exadata")
    for rows in ((row, row), (row,) * 129):
        value, calls = adapter(ROOT_ROW, association_rows(*rows))
        with pytest.raises(DiagnosticSourceAdapterError):
            asyncio.run(value.exadata_inventory(ROOT))
        assert len(calls) == 2


@pytest.mark.parametrize("status,code", [(401, "SOURCE_AUTH_FAILED"), (403, "SOURCE_AUTH_FAILED"),
    (404, "SOURCE_QUERY_UNSUPPORTED"), (302, "SOURCE_RESPONSE_INVALID"), (429, "SOURCE_RATE_LIMITED"),
    (503, "SOURCE_UNREACHABLE")])
def test_official_dataservice_http_failures_never_fall_back(status, code):
    value, calls = adapter(ROOT_ROW, {}, statuses=[200, status])
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        asyncio.run(value.exadata_inventory(ROOT))
    assert error.value.code == code and len(calls) == 2


def test_bearer_auth_and_endpoint_api_suffix_do_not_create_a_second_origin():
    value, calls = adapter(*tree_payloads(), credentials={"token": "test-only"})
    value.context = value.context.model_copy(update={"endpoint": "https://oem.example.com/em/api"})
    asyncio.run(value.exadata_inventory(ROOT))
    assert calls[1][0] == "https://oem.example.com/em" + DATA_SERVICE
    assert calls[1][1]["headers"] == {"Authorization": "Bearer test-only"}


def test_quoted_name_remains_encoded_parameter_and_filter_is_rechecked():
    name = "测试 ' 引号, /? Grid"
    root = target(ROOT, name, "oracle_dbmachine")
    proof = association_rows((name, "oracle_dbmachine", "测试 Cell", "oracle_exadata"))
    value, calls = adapter(root, proof, collection([CELL_ROW]), CELL_ROW)
    asyncio.run(value.exadata_inventory(ROOT))
    assert calls[1][0].endswith(DATA_SERVICE) and "?" not in calls[1][0]
    assert "测试 '' 引号, /? Grid" in calls[1][1]["params"]["search"]
