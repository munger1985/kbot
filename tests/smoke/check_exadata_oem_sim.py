"""只读联调真实 OEM 中的 SIM 整机；凭据仅从标准输入读取，不写入文件或日志。"""

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import sys

import aiohttp

from aiops_agent.adapters.diagnostic_sources.base import DiagnosticSourceAdapterError
from aiops_agent.adapters.diagnostic_sources.oem import OEMAdapter
from aiops_agent.adapters.diagnostic_sources.oem_target_identity import normalize_target
from aiops_agent.application.exadata_catalog import load_catalog
from aiops_agent.ports.diagnostic_source import DiagnosticSourceContext, NativeMetricQueryRequest


async def probe(arguments, credentials):
    context = DiagnosticSourceContext(source_id="00000000-0000-4000-8000-000000000001", source_type="OEM",
        adapter_id="oem", adapter_version="1.0.0", config_version=1, endpoint=arguments.endpoint,
        credentials=credentials, declared_capabilities={"metric.query_range": {}, "instance.discover": {}})
    connector = aiohttp.TCPConnector(ssl=False) if arguments.insecure_lab_tls else aiohttp.TCPConnector()
    async with aiohttp.ClientSession(connector=connector) as session:
        adapter = OEMAdapter(context=context, session=session, request_timeout_seconds=15,
                             webhook_replay_seconds=300)
        adapter._remaining_response_bytes = 4 * 1024 * 1024
        matches, cells = [], []
        async for rows in adapter._target_pages("/targets"):
            for row in rows:
                target = normalize_target(row)
                if target["targetName"] == "AIOPS_SIM_EXADATA_DBM01" and target["targetType"] == "oracle_dbmachine":
                    matches.append(target["targetId"])
                if target["targetName"] == "AIOPS_SIM_EXADATA_CELL01" and target["targetType"] == "oracle_exadata":
                    cells.append(target)
        if arguments.cell_only:
            if len(cells) != 1:
                raise DiagnosticSourceAdapterError("SIM_CELL_NOT_UNIQUE", "未发现唯一 SIM Cell")
            root, digest = None, None
            members = [{"target_id": cells[0]["targetId"], "target_type": "oracle_exadata",
                        "display_name": cells[0]["targetName"]}]
        else:
            if len(matches) != 1:
                raise DiagnosticSourceAdapterError("SIM_ROOT_NOT_UNIQUE", "未发现唯一 SIM 整机")
            root, members, digest = await adapter.exadata_inventory(matches[0])
        if any(not member["display_name"].startswith("AIOPS_SIM_") for member in members):
            raise DiagnosticSourceAdapterError("SIM_MEMBER_SCOPE_INVALID", "SIM 整机含非模拟成员，停止读取")
        _, catalog_hash, definitions, profiles, _ = load_catalog()
        profile = next(item for item in profiles if item["id"] == arguments.scene)
        end = datetime.now(UTC)
        results = []
        for member in members:
            for code in profile["metric_codes"]:
                definition = definitions[code]
                if definition["target_type"] != member["target_type"]:
                    continue
                if arguments.history and definition["native_value_type"] != "NUMBER":
                    continue
                request = NativeMetricQueryRequest(action_id="a1", target_id="00000000-0000-4000-8000-000000000002",
                    binding_id="00000000-0000-4000-8000-000000000003", source_locator_key=member["target_id"],
                    window_start=end - timedelta(hours=1), window_end=end, trace_id="exadata-sim-probe")
                try:
                    samples = await (adapter.exadata_history(request, definition, member["target_type"])
                                     if arguments.history else adapter.exadata_latest(request, definition, member["target_type"]))
                    points = samples["points"]
                    results.append({"metric": code, "sample_count": len(points), "unit": samples["unit"],
                        "points": [[stamp.isoformat(), value] for stamp, value in points[-3:]],
                        "dimensions": samples["dimensions"][-3:], "truncated": samples["truncated"],
                        "non_empty": any(value not in (None, "") for _, value in points)})
                except DiagnosticSourceAdapterError as exc:
                    results.append({"metric": code, "error_code": exc.code})
        passed = bool(results) and all(item.get("non_empty") for item in results)
        print(json.dumps({"ok": passed, "simulation": True, "read_only": True,
            "inventory_verified": not arguments.cell_only,
            "acceptance_scope": "METRIC_TRANSPORT_ONLY" if arguments.cell_only else "VERIFIED_EXADATA_INVENTORY_AND_METRICS",
            "repository_adapter": OEMAdapter.__module__, "repository_root": str(Path(__file__).resolve().parents[2]),
            "root": root, "members": members,
            "inventory_digest": digest, "catalog_hash": catalog_hash,
            "mode": "HISTORY" if arguments.history else "LATEST_SNAPSHOT", "metrics": results}, ensure_ascii=False))
        return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--insecure-lab-tls", action="store_true", help="仅用于本次隔离实验环境自签名证书")
    parser.add_argument("--scene", choices=("overview", "cell", "disk", "flash", "network", "capacity", "iorm"), default="overview")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--cell-only", action="store_true", help="仅验证 Cell 指标传输；不代表整机成员或页面验收通过")
    arguments = parser.parse_args()
    try:
        credentials = json.load(sys.stdin)
        if (not isinstance(credentials, dict) or set(credentials) != {"username", "password"}
                or any(not isinstance(value, str) or not value for value in credentials.values())):
            raise ValueError
        return asyncio.run(probe(arguments, credentials))
    except DiagnosticSourceAdapterError as exc:
        print(json.dumps({"ok": False, "error_code": exc.code}))
    except Exception as exc:
        print(json.dumps({"ok": False, "error_type": type(exc).__name__}))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
