"""OEM 数据库发现：固定只读端点、受控分页及属性白名单。"""

import asyncio
import re
from urllib.parse import parse_qs, urlsplit

import aiohttp

from aiops_agent.ports.diagnostic_source import HostDiscoveryCandidate, InstanceDiscoveryCandidate, InstanceDiscoveryResult
from .base import DiagnosticSourceAdapterError
from .oem_target_identity import normalize_target, target_error, target_text


# 只登记明确的插件目标类型，不从显示名称猜测引擎或把集群当实例。
DATABASE_TARGET_TYPES = {
    "oracle_database": "ORACLE",
    "oracle_pdb": "ORACLE",
    "mysql_database": "MYSQL",
}
IDENTITY_PROPERTIES = frozenset({
    "CDB", "IS_ROOT", "Name", "DBName", "DBUniqueName", "SID", "ServiceName",
    "Version", "DBVersion", "RACOption", "RACInstNum", "PDBGuid",
})


class OEMInstanceDiscovery:
    async def _host_candidate(self, target):
        guid = target["targetId"]
        detail, _ = await self._native_get("/targets/" + guid, max_bytes=65536)
        detail = normalize_target(detail, expected_guid=guid)
        if detail["targetType"] != "host" or detail["targetName"] != target["targetName"]:
            target_error("SOURCE_TARGET_MISMATCH")
        return HostDiscoveryCandidate(
            source_locator_key=guid,
            display_name=target_text(detail.get("displayName") or detail["targetName"]),
            source_locator={
                "resource_scope": "HOST", "target_id": guid,
                "target_type": "host", "target_name": detail["targetName"],
                "host_name": target_text(detail.get("hostName") or detail["targetName"]),
            },
        )

    async def _target_pages(self, suffix, *, limit=100):
        from .oem_native import _collection

        params = {"limit": limit}
        seen = set()
        for _ in range(100):
            payload, _ = await self._native_get(suffix, params=params, max_bytes=524288)
            items, more = _collection(payload)
            yield items
            if not more:
                return
            link = payload["links"]["next"]
            href = link.get("href") if isinstance(link, dict) else None
            if not isinstance(href, str) or len(href) > 16384:
                target_error()
            parsed = urlsplit(href)
            expected_path = urlsplit(self._native_api()).path + suffix
            query = parse_qs(parsed.query, strict_parsing=True)
            # 只提取该固定端点的页令牌，不执行响应 URL，也不接受浏览器查询。
            if (parsed.scheme or parsed.netloc or parsed.fragment or parsed.path != expected_path
                or set(query) != {"page"} or len(query["page"]) != 1):
                target_error()
            token = query["page"][0]
            if not re.fullmatch(r"[A-Za-z0-9_+/=-]{1,8192}", token) or token in seen:
                target_error("SOURCE_PAGINATION_INVALID")
            seen.add(token)
            params = {"page": token}
        target_error("SOURCE_DISCOVERY_LIMIT")

    async def _database_candidate(self, target):
        guid = target["targetId"]
        detail, _ = await self._native_get("/targets/" + guid, max_bytes=65536)
        detail = normalize_target(detail, expected_guid=guid)
        if (detail["targetName"] != target["targetName"]
            or detail["targetType"] != target["targetType"]):
            target_error("SOURCE_TARGET_MISMATCH")
        properties = {}
        async for items in self._target_pages("/targets/" + guid + "/properties"):
            for item in items:
                key = item.get("id")
                if not isinstance(key, str):
                    target_error()
                if key not in IDENTITY_PROPERTIES or item.get("value") in (None, ""):
                    continue
                value = target_text(item["value"])
                if key in properties and properties[key] != value:
                    target_error("SOURCE_ITEM_AMBIGUOUS")
                properties[key] = value
        kind = detail["targetType"]
        locator = {
            "target_id": guid, "target_name": detail["targetName"],
            "target_type": kind, "database_type": DATABASE_TARGET_TYPES[kind],
        }
        host = detail.get("hostName")
        if host:
            locator["host_name"] = target_text(host)
        if detail.get("typeVersion"):
            locator["plugin_version"] = target_text(detail["typeVersion"], 64)
        for key, name in (("DBName", "database_name"), ("DBUniqueName", "database_unique_name"),
                          ("SID", "instance_name"), ("PDBGuid", "pdb_guid"),
                          ("RACOption", "rac_option"), ("RACInstNum", "rac_instance_number")):
            if key in properties:
                locator[name] = properties[key]
        version = properties.get("DBVersion") or properties.get("Version")
        if version:
            locator["database_version"] = version
        if kind == "oracle_database":
            cdb = properties.get("CDB")
            if cdb not in {"YES", "NO"}:
                target_error("SOURCE_TARGET_IDENTITY_UNVERIFIED")
            locator["oracle_container_scope"] = "CDB_ROOT" if cdb == "YES" else "NON_CDB"
        elif kind == "oracle_pdb":
            root = properties.get("IS_ROOT")
            if root not in {"YES", "NO"}:
                target_error("SOURCE_TARGET_IDENTITY_UNVERIFIED")
            # OEM 将 CDB$ROOT 也登记为 oracle_pdb；它不是可另选的普通 PDB。
            if root == "YES":
                return None
            locator["oracle_container_scope"] = "PDB"
            locator["oracle_pdb_name"] = target_text(properties.get("Name"), 128)
        label = detail.get("displayName") or detail["targetName"]
        context = [locator.get("oracle_container_scope") or locator["database_type"],
                   locator.get("oracle_pdb_name") or locator.get("database_name"),
                   version, host]
        display = " · ".join([target_text(label), *(part for part in context if part)])
        return InstanceDiscoveryCandidate(
            source_locator_key=guid, source_locator=locator,
            db_type=locator["database_type"], display_name=display[:256],
        )

    async def discover_instances(self, request):
        self._remaining_response_bytes = 4 * 1024 * 1024
        try:
            async with asyncio.timeout(45):
                requested = set(request.db_types)
                targets = {}
                locator_keys = (*request.locator_keys, *request.host_locator_keys)
                if locator_keys:
                    if len(locator_keys) > 100:
                        target_error("SOURCE_QUERY_INVALID")
                    for guid in locator_keys:
                        if not re.fullmatch(r"[0-9a-fA-F]{32}", guid):
                            target_error("SOURCE_TARGET_LOCATOR_INVALID")
                        payload, _ = await self._native_get("/targets/" + guid, max_bytes=65536)
                        target = normalize_target(payload, expected_guid=guid)
                        targets[target["targetId"]] = target
                else:
                    async for items in self._target_pages("/targets"):
                        for item in items:
                            target = normalize_target(item)
                            guid = target["targetId"]
                            if guid in targets:
                                target_error("SOURCE_PAGINATION_INVALID")
                            targets[guid] = target
                candidates, host_candidates = [], []
                for guid, target in sorted(targets.items()):
                    if request.after_locator_key is not None and guid <= request.after_locator_key:
                        continue
                    if target["targetType"] == "host" and (request.include_hosts or request.host_locator_keys):
                        if len(host_candidates) < request.limit:
                            host_candidates.append(await self._host_candidate(target))
                        continue
                    if request.host_only:
                        continue
                    db_type = DATABASE_TARGET_TYPES.get(target["targetType"])
                    if db_type is None or (requested and db_type not in requested):
                        continue
                    if len(candidates) >= request.limit:
                        continue
                    candidate = await self._database_candidate(target)
                    if candidate is not None:
                        candidates.append(candidate)
                return InstanceDiscoveryResult(candidates=tuple(candidates), host_candidates=tuple(host_candidates))
        except (aiohttp.ClientError, TimeoutError) as exc:
            raise DiagnosticSourceAdapterError(
                "SOURCE_UNREACHABLE", "OEM 数据库发现超时或连接失败", retryable=True
            ) from exc
        except ValueError as exc:
            raise DiagnosticSourceAdapterError(
                "SOURCE_RESPONSE_INVALID", "OEM 数据库目录分页或属性无效"
            ) from exc
