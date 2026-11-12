"""原厂 EMCLI 的维度、原值、时间窗口、显式路由和子进程安全回归。"""

import asyncio
import copy
import hashlib
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem_emcli import OEMEMCLIClient
from aiops_agent.adapters.diagnostic_sources.oem_exadata_emcli import parse_values, query_emcli
from aiops_agent.adapters.diagnostic_sources.registry import DiagnosticSourceAdapterCatalog
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.ports.exadata import ExadataMetricRead
from tests.unit.aiops_agent.test_aiops_oem_native_metrics import adapter


NOW = datetime(2026, 11, 12, 3, tzinfo=UTC)
TARGET = {"targetId": "b" * 32, "targetName": "SIM_CELL", "targetType": "oracle_exadata"}


def definition(selector="Aggregated_HardNFlashDisk_Metric/avg_cd_io_load"):
    return next(item for item in load_catalog()[2].values() if item["selector"] == selector)


def payload(value="9007199254740993.123456789", *, selector=None):
    metric = definition(selector) if selector else definition()
    group, column = metric["selector"].split("/")
    return {"Entity Type": TARGET["targetType"], "Entity Name": TARGET["targetName"],
        "Metric Group Name": group, **{f"keyColumn_{i}_name": name for i, name in enumerate(metric["keys"], 1)},
        "data_columns": [{"Metric Column Name": column, "keyValues": [{
            **{f"key{i}": "SIM_" + name for i, name in enumerate(metric["keys"], 1)},
            "values": [{"First Collection Time": "2026-11-12 02:50:00",
                        "Last Collection Time": "2026-11-12 02:55:00", "Value": value}]}]}]}


def parsed(body=None, **kwargs):
    return parse_values(payload() if body is None else body, target=TARGET, definition=definition(),
                        timezone="UTC", captured_at=NOW, **kwargs)


def test_native_precision_dimensions_and_retained_timestamp_not_expanded():
    points, dimensions, truncated = parsed(start=NOW-timedelta(hours=1), end=NOW)
    assert points == ((NOW-timedelta(minutes=5), "9007199254740993.123456789"),)
    assert dimensions == ({"celldisk_type": "SIM_celldisk_type"},) and not truncated


@pytest.mark.parametrize("value", ["0", "", None])
def test_zero_empty_and_null_are_distinct(value):
    assert parsed(payload(value))[0][0][1] == value


def test_history_clips_window_and_does_not_expand_state_interval():
    assert parsed(start=NOW-timedelta(minutes=4), end=NOW) == ((), (), False)
    assert len(parsed(start=NOW-timedelta(minutes=6), end=NOW)[0]) == 1


def test_latest_uses_last_per_dimension_without_filling_null():
    body = payload()
    rows = body["data_columns"][0]["keyValues"]
    rows[0]["values"].append({**rows[0]["values"][0], "Value": None})
    other = copy.deepcopy(rows[0])
    other["key1"] = "SIM_OTHER_DISK"
    other["values"][-1]["Value"] = "0"
    rows.append(other)
    points, dimensions, _ = parsed(body)
    assert len(points) == 2 and {value for _, value in points} == {None, "0"}
    assert len({tuple(d.items()) for d in dimensions}) == 2


def test_malformed_native_timestamp_is_a_safe_source_error():
    body = payload()
    body["data_columns"][0]["keyValues"][0]["values"][0]["Last Collection Time"] = "not a timestamp"
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        parsed(body)
    assert error.value.code == "SOURCE_RESPONSE_INVALID"


@pytest.mark.parametrize("native,code", [(b"not-json", "SOURCE_RESPONSE_INVALID"),
    (b"\xff", "SOURCE_RESPONSE_INVALID"), (TimeoutError(), "SOURCE_EMCLI_TIMEOUT")])
def test_native_invalid_output_and_tls_timeout_are_safe_errors(tmp_path, native, code):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions")
        if isinstance(native, TimeoutError):
            client._initialize = AsyncMock(side_effect=native)
        else:
            client._initialize = AsyncMock()
            client._run = AsyncMock(return_value=native)
        try:
            with pytest.raises(DiagnosticSourceAdapterError) as error:
                await client.read(context(), TARGET, definition(), history=False, max_bytes=1024)
            assert error.value.code == code
            assert not next(iter(client._sessions.values())).initialized
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("field,value,code", [
    ("Entity Name", "OTHER_CELL", "SOURCE_TARGET_MISMATCH"),
    ("Entity Type", "host", "SOURCE_TARGET_MISMATCH"),
    ("Metric Group Name", "OTHER_GROUP", "SOURCE_TARGET_MISMATCH"),
    ("keyColumn_1_name", "wrong_dimension", "SOURCE_METRIC_DIMENSION_MISMATCH"),
])
def test_native_identity_and_dimension_names_fail_closed(field, value, code):
    body = payload()
    body[field] = value
    with pytest.raises(DiagnosticSourceAdapterError) as error:
        parsed(body)
    assert error.value.code == code


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e2000", {}, 1.2, True])
def test_number_is_finite_decimal_text_without_float_coercion(value):
    with pytest.raises(DiagnosticSourceAdapterError):
        parsed(payload(value))


def test_text_status_keeps_native_string_and_asm_size_is_not_retyped():
    for selector, value in [("Exadata_Disk_Status_Metric/status", "normal"),
                            ("Aggregated_Exadata_Diskgroup_Capacity_Metric/size", "8192")]:
        metric = definition(selector)
        assert metric["native_value_type"] == "STRING"
        points, _, _ = parse_values(payload(value, selector=selector), target=TARGET,
            definition=metric, timezone="UTC", captured_at=NOW)
        assert points[0][1] == value


def test_future_time_dst_ambiguity_and_reversed_intervals_rejected():
    for first, last, timezone, code in [
        ("2026-11-12 03:00:06", "2026-11-12 03:00:06", "UTC", "SOURCE_RESPONSE_INVALID"),
        ("2026-11-12 02:59:00", "2026-11-12 02:55:00", "UTC", "SOURCE_RESPONSE_INVALID"),
        ("2026-11-01 01:30:00", "2026-11-01 01:30:00", "America/New_York", "SOURCE_TIMEZONE_AMBIGUOUS"),
    ]:
        body = payload()
        row = body["data_columns"][0]["keyValues"][0]["values"][0]
        row.update({"First Collection Time": first, "Last Collection Time": last})
        with pytest.raises(DiagnosticSourceAdapterError) as error:
            parse_values(body, target=TARGET, definition=definition(), timezone=timezone, captured_at=NOW)
        assert error.value.code == code


def test_truncation_is_explicit_and_point_labels_stay_aligned():
    body = payload()
    rows = body["data_columns"][0]["keyValues"]
    rows.append({**copy.deepcopy(rows[0]), "key1": "SIM_B"})
    points, dimensions, truncated = parsed(body, limit=1)
    assert len(points) == len(dimensions) == 1 and truncated


def test_mode_requires_explicit_timezone_pin_and_rejects_command_options():
    normalize = lambda value: DiagnosticSourceAdapterCatalog.normalize_config(source_type="OEM", config=value)
    good = {"exadata_metric_transport": "EMCLI", "exadata_emcli_timezone": "UTC",
            "exadata_emcli_certificate_sha256": "a" * 64}
    assert normalize({}) == {} and normalize(good) == good
    for bad in [{**good, "executable": "/bin/sh"}, {"exadata_metric_transport": "FALLBACK"},
                {**good, "exadata_emcli_timezone": "not-a-zone"},
                {**good, "exadata_emcli_certificate_sha256": "A" * 64},
                {"exadata_metric_transport": "EMCLI"}]:
        with pytest.raises(ValueError):
            normalize(bad)


def context(**changes):
    return SimpleNamespace(source_id="source-one", config_version=1, endpoint="https://oem.example.com/em/api",
        credentials={"username": "test-reader", "password": "test-not-a-secret"}, config={}, **changes)


def test_sessions_are_private_bounded_and_rotated_for_credentials(tmp_path):
    client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions", max_sessions=1)
    async def run():
        first = client._session(context())
        assert Path(first.directory.name).stat().st_mode & 0o777 == 0o700
        new = context()
        new.credentials = {**new.credentials, "password": "test-rotated"}
        second = client._session(new)
        assert first is not second and not Path(first.directory.name).exists()
        await client.aclose()
        assert not Path(second.directory.name).exists()
    asyncio.run(run())


def test_symlink_or_shared_session_root_rejected(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o755)
    link = tmp_path / "link"
    link.symlink_to(real)
    for root in (link, real):
        client = OEMEMCLIClient(executable=sys.executable, session_root=root)
        with pytest.raises(DiagnosticSourceAdapterError):
            client._session(context())


def test_process_timeout_and_output_budget_reap_children(tmp_path):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions", timeout_seconds=.2)
        entry = client._session(context())
        try:
            for arguments, code in [(["-c", "import time; time.sleep(60)"], "SOURCE_EMCLI_TIMEOUT"),
                                    (["-c", "print('x'*2048)"], "SOURCE_RESPONSE_TOO_LARGE")]:
                with pytest.raises(DiagnosticSourceAdapterError) as error:
                    await client._run(entry, arguments, max_bytes=1024)
                assert error.value.code == code and not client._processes
        finally:
            await client.aclose()
    asyncio.run(run())


def test_process_cancellation_reaps_children_and_close_prevents_reuse(tmp_path):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions")
        entry = client._session(context())
        task = asyncio.create_task(client._run(entry, ["-c", "import time; time.sleep(60)"]))
        for _ in range(100):
            if client._processes:
                break
            await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not client._processes
        await client.aclose()
        with pytest.raises(DiagnosticSourceAdapterError):
            client._session(context())
    asyncio.run(run())


def test_metric_allowlist_and_text_history_fail_before_process_creation(tmp_path):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions")
        client._run = AsyncMock()
        for metric, target, history in [({**definition(), "selector": "OTHER/metric"}, TARGET, False),
            (definition(), {**TARGET, "targetType": "host"}, False),
            (definition("Exadata_Disk_Status_Metric/status"), TARGET, True)]:
            with pytest.raises(DiagnosticSourceAdapterError):
                await client.read(context(), target, metric, history=history, max_bytes=262144)
        client._run.assert_not_awaited()
        await client.aclose()
    asyncio.run(run())


def test_certificate_pin_mismatch_rejects_before_password_is_sent(tmp_path):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions")
        value = context()
        value.config = {"exadata_emcli_certificate_sha256": "a" * 64}
        writer = SimpleNamespace(get_extra_info=lambda _: SimpleNamespace(getpeercert=lambda **_: b"unexpected-cert"),
                                 close=lambda: None, wait_closed=AsyncMock())
        client._run = AsyncMock()
        try:
            with patch("asyncio.open_connection", new=AsyncMock(return_value=(None, writer))):
                with pytest.raises(DiagnosticSourceAdapterError) as error:
                    await client._initialize(client._session(value), value)
            assert error.value.code == "SOURCE_TLS_PIN_MISMATCH"
            client._run.assert_not_awaited()
            writer.wait_closed.assert_awaited_once()
        finally:
            await client.aclose()
    asyncio.run(run())


def test_cli_uses_argument_array_and_password_only_via_stdin(tmp_path):
    async def run():
        client = OEMEMCLIClient(executable=sys.executable, session_root=tmp_path / "sessions")
        value = context()
        process = SimpleNamespace(stdin=SimpleNamespace(write=lambda data: sent.append(data),
            drain=AsyncMock(), close=lambda: None), stdout=SimpleNamespace(read=AsyncMock(side_effect=[b"{}", b""])),
            wait=AsyncMock(return_value=0))
        # subprocess 对象须可哈希，用简单对象代替命名空间。
        process = type("Process", (), dict(vars(process)))()
        sent = []
        with patch.dict(os.environ, {"AMMOLITE_DATABASE_PASSWORD": "must-not-leak", "EMCLI_OPTS": "untrusted-options"}):
            with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as launch:
                await client._run(client._session(value), ["setup", "-url=https://oem.example.com/em"],
                                  password=value.credentials["password"])
        options = launch.call_args.kwargs
        assert value.credentials["password"] not in str(launch.call_args)
        assert sent == [(value.credentials["password"] + "\n").encode()]
        assert "AMMOLITE_DATABASE_PASSWORD" not in options["env"]
        assert "untrusted-options" not in options["env"]["EMCLI_OPTS"]
        assert options["start_new_session"] is True and "shell" not in options
        assert options["env"]["HOME"] == options["cwd"]
        assert "keystore.type=JKS" in (Path(options["cwd"]) / "emcli.security").read_text()
        await client.aclose()
    asyncio.run(run())


def test_native_bindings_use_private_client_copy_not_shared_installation(tmp_path):
    async def run():
        installation = tmp_path / "installed"
        installation.mkdir()
        executable = installation / "emcli"
        executable.write_text("test native launcher")
        executable.chmod(0o755)
        (installation / "emCLI.jar").write_bytes(b"test native jar")
        (installation / "version.txt").write_text("24.1")
        (installation / "bindings").mkdir()
        (installation / "bindings" / "another-source-cookie").write_text("never copy")
        (installation / "another-source-credentials").write_text("never copy")
        client = OEMEMCLIClient(executable=str(executable), session_root=tmp_path / "sessions")
        value = context()
        value.config = {"exadata_emcli_certificate_sha256": hashlib.sha256(b"expected-cert").hexdigest()}
        writer = SimpleNamespace(get_extra_info=lambda _: SimpleNamespace(getpeercert=lambda **_: b"expected-cert"),
                                 close=lambda: None, wait_closed=AsyncMock())
        client._run = AsyncMock(return_value=b"Emcli setup successful")
        entry = client._session(value)
        try:
            with patch("asyncio.open_connection", new=AsyncMock(return_value=(None, writer))) as connect:
                await client._initialize(entry, value)
                await client._initialize(entry, value)
            assert connect.await_count == 2  # 缓存会话仍复核证书指纹。
            assert entry.client_home != installation
            assert {path.name for path in entry.client_home.iterdir()} == {"emcli", "emCLI.jar", "version.txt"}
            assert (entry.client_home / "emCLI.jar").read_bytes() == b"test native jar"
            assert entry.client_home.stat().st_mode & 0o777 == 0o700
            client._run.assert_awaited_once()
            assert client._run.call_args.kwargs["password"] == value.credentials["password"]
            assert "-noautologin" in client._run.call_args.args[1]
            assert (installation / "bindings" / "another-source-cookie").read_text() == "never copy"
        finally:
            await client.aclose()
    asyncio.run(run())


def test_emcli_is_opt_in_and_missing_dependency_never_falls_back():
    async def run():
        value, _ = adapter()
        request = ExadataMetricRead(source_locator_key="b" * 32)
        with patch("aiops_agent.adapters.diagnostic_sources.oem_exadata.query_latest", new_callable=AsyncMock) as rest:
            await value.exadata_latest(request, definition(), "oracle_exadata")
            rest.assert_awaited_once()
            value.context = value.context.model_copy(update={"config": {"exadata_metric_transport": "EMCLI"}})
            with pytest.raises(DiagnosticSourceAdapterError) as error:
                await value.exadata_latest(request, definition(), "oracle_exadata")
            assert error.value.code == "SOURCE_EMCLI_UNAVAILABLE"
            assert rest.await_count == 1
    asyncio.run(run())


def test_online_metadata_type_and_keys_must_match_before_cli_read():
    async def run():
        metric = definition()
        for native_type, keys, code in [("STRING", ["celldisk_type"], "SOURCE_METRIC_TYPE_MISMATCH"),
                                        ("NUMBER", ["wrong"], "SOURCE_METRIC_DIMENSION_MISMATCH")]:
            value, _ = adapter(TARGET, {"name": metric["selector"].split("/")[0],
                "keys": [{"name": key} for key in keys], "metrics": [{"name": "avg_cd_io_load", "dataType": native_type}]})
            value._exadata_emcli_client = SimpleNamespace(read=AsyncMock())
            with pytest.raises(DiagnosticSourceAdapterError) as error:
                await query_emcli(value, ExadataMetricRead(source_locator_key="b" * 32), metric, "oracle_exadata", history=False)
            assert error.value.code == code
            value._exadata_emcli_client.read.assert_not_awaited()
    asyncio.run(run())


def test_editing_exadata_transport_preserves_independently_managed_standard_mappings():
    from aiops_agent.application.configuration.diagnostic_source_service import DiagnosticSourceConfigurationMixin
    from platform_core.contracts.aiops import DiagnosticSourcePatch

    async def run():
        mappings = {"db.storage.used": {"selector": "SIM_STORAGE/used", "unit": "GB"}}
        entity = SimpleNamespace(source_type="OEM", config_json={"standard_metric_mappings": mappings},
            row_version=3, adapter_id="oem", adapter_version="1.0.0", declared_capabilities_json={},
            endpoint="https://oem.example.com/em/api", webhook_credential_id=None, connectivity_version=1)
        uow = SimpleNamespace(diagnostic_sources=SimpleNamespace(get_scoped=AsyncMock(return_value=entity)),
            managed_credentials=object(), session=SimpleNamespace(flush=AsyncMock()), flush=AsyncMock(), commit=AsyncMock())

        class UowContext:
            async def __aenter__(self):
                return uow

            async def __aexit__(self, *_):
                return False

        service = object.__new__(DiagnosticSourceConfigurationMixin)
        service._uow_factory = UowContext
        service._check_version = Mock()
        service._normalize_source_config = DiagnosticSourceAdapterCatalog.normalize_config
        service._validate_diagnostic_source_adapter = Mock()
        service._check_diagnostic_source_now = AsyncMock()
        module = "aiops_agent.application.configuration.diagnostic_source_service"
        with patch(module + ".add_configuration_event", new=AsyncMock()), \
                patch(module + "._diagnostic_source_detail", side_effect=lambda value: value):
            result = await service.patch_diagnostic_source(scope=SimpleNamespace(domain_id=87, actor_id="SIM_ACTOR"),
                source_id="SIM_SOURCE", expected_version=3, request=DiagnosticSourcePatch(config={
                    "exadata_metric_transport": "EMCLI", "exadata_emcli_timezone": "UTC",
                    "exadata_emcli_certificate_sha256": "a" * 64}))
        assert result.config_json["standard_metric_mappings"] == mappings
        assert result.config_json["exadata_metric_transport"] == "EMCLI"
        assert result.status == "DISABLED" and result.connectivity_version == 2
        uow.commit.assert_awaited_once()
    asyncio.run(run())
