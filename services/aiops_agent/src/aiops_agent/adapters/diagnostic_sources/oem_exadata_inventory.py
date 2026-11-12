"""使用官方 Data Service 显式包含关系确认 Exadata 成员，不猜名称或同机归属。"""

import aiohttp

from .oem_native import _collection
from .oem_target_identity import normalize_target, target_error, target_text


ASSOCIATION_COLUMNS = frozenset({"TARGET_NAME", "TARGET_TYPE", "ASSOC_TYPE",
                                 "ASSOC_TARGET_NAME", "ASSOC_TARGET_TYPE"})
DATA_SERVICE = "/websvcs/restful/emws/core/v0/list/dataservice"


async def associations(adapter, parent, *, deployed_hosts=False):
    """资源、关联方向和过滤字段均固定；不接受 SQL、响应地址或任意资源名。"""
    credentials = adapter.context.credentials
    username, password = credentials.get("username"), credentials.get("password")
    if bool(username) != bool(password):
        target_error("SOURCE_AUTH_FAILED")
    auth = aiohttp.BasicAuth(username, password) if username and password else None
    # 只在固定搜索语法内编码 OEM 返回的身份；绝不将其拼到 URL 路径或 SQL。
    name = parent["targetName"].replace("'", "''")
    kind = parent["targetType"]
    if deployed_hosts and kind != "oracle_si_server_map":
        target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
    search = (f"ASSOC_TARGET_NAME='{name}',ASSOC_TARGET_TYPE='{kind}',TARGET_TYPE='host',ASSOC_TYPE='deployed_on'"
              if deployed_hosts else f"TARGET_NAME='{name}',TARGET_TYPE='{kind}',ASSOC_TYPE='internal_contains'")
    params = {"name": "TargetAssociations", "search": search}
    async with adapter._session.get(
        adapter._native_api().removesuffix("/api") + DATA_SERVICE, params=params, auth=auth,
        headers={} if auth else adapter._headers(), timeout=adapter._timeout, allow_redirects=False,
    ) as response:
        if response.status not in {200, 401, 403, 429} and response.status < 500:
            target_error("SOURCE_QUERY_UNSUPPORTED" if response.status == 404 else "SOURCE_RESPONSE_INVALID")
        payload, _ = await adapter._response_json(response, max_bytes=262144)
    if not isinstance(payload, dict) or payload.get("exceedsMaxRows") is not False:
        target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
    columns, rows = payload.get("columnNames"), payload.get("data")
    if (not isinstance(columns, list) or len(columns) != len(ASSOCIATION_COLUMNS)
            or any(not isinstance(value, str) for value in columns)
            or set(columns) != ASSOCIATION_COLUMNS or not isinstance(rows, list)):
        target_error()
    if len(rows) > 128:
        target_error("SOURCE_DISCOVERY_LIMIT")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != ASSOCIATION_COLUMNS:
            target_error()
        actual = ((row["ASSOC_TARGET_NAME"], row["ASSOC_TARGET_TYPE"], row["TARGET_TYPE"], row["ASSOC_TYPE"])
                  if deployed_hosts else (row["TARGET_NAME"], row["TARGET_TYPE"], row["ASSOC_TYPE"]))
        expected = ((parent["targetName"], kind, "host", "deployed_on") if deployed_hosts
                    else (parent["targetName"], kind, "internal_contains"))
        if actual != expected:
            target_error("SOURCE_TARGET_MISMATCH")
        identity = ((target_text(row["TARGET_NAME"]), "host") if deployed_hosts else
                    (target_text(row["ASSOC_TARGET_NAME"]), target_text(row["ASSOC_TARGET_TYPE"], 64)))
        if identity in seen:
            target_error("SOURCE_ITEM_AMBIGUOUS")
        seen.add(identity)
        result.append(identity)
    return result


async def resolve_member(adapter, name, kind):
    """仅将已证明关系的精确名称/类型查回 GUID；不做通配或模糊匹配。"""
    payload, _ = await adapter._native_get("/targets", params={"name": name, "typeName": kind, "limit": 2},
                                           max_bytes=65536)
    rows, more = _collection(payload)
    if more or len(rows) != 1:
        target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
    item = normalize_target(rows[0])
    if (item["targetName"], item["targetType"]) != (name, kind):
        target_error("SOURCE_TARGET_MISMATCH")
    detail, _ = await adapter._native_get("/targets/" + item["targetId"], max_bytes=65536)
    detail = normalize_target(detail, expected_guid=item["targetId"])
    if (detail["targetName"], detail["targetType"]) != (name, kind):
        target_error("SOURCE_TARGET_MISMATCH")
    return detail


async def verified_members(adapter, root):
    """只展开根、直接 Grid 和已确认 ILOM 的反向 host deployed_on；不使用 hosted_by。"""
    members, parents, servers = {}, [root], []
    for parent in parents:
        for name, kind in await associations(adapter, parent):
            if (kind == "oracle_dbmachine" or
                    parent["targetType"] == "oracle_exadata_grid" and kind != "oracle_exadata"):
                target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
            if len(members) >= 128:
                target_error("SOURCE_DISCOVERY_LIMIT")
            item = await resolve_member(adapter, name, kind)
            guid = item["targetId"]
            if guid == root["targetId"] or guid in members:
                target_error("SOURCE_ITEM_AMBIGUOUS")
            members[guid] = {"target_id": guid, "target_type": kind, "display_name": name}
            if kind == "oracle_exadata_grid":
                parents.append(item)
            elif kind == "oracle_si_server_map":
                servers.append(item)
    host_owners = set()
    for server in servers:
        hosts = await associations(adapter, server, deployed_hosts=True)
        if len(hosts) > 1:
            target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
        for name, kind in hosts:
            item = await resolve_member(adapter, name, kind)
            guid = item["targetId"]
            if guid in host_owners or guid == root["targetId"]:
                target_error("SOURCE_ITEM_AMBIGUOUS")
            host_owners.add(guid)
            if guid in members:
                if members[guid] != {"target_id": guid, "target_type": kind, "display_name": name}:
                    target_error("SOURCE_TARGET_MISMATCH")
                continue
            if len(members) >= 128:
                target_error("SOURCE_DISCOVERY_LIMIT")
            members[guid] = {"target_id": guid, "target_type": kind, "display_name": name}
    if not any(member["target_type"] == "oracle_exadata" for member in members.values()):
        target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
    return sorted(members.values(), key=lambda item: item["target_id"])
