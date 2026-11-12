"""显式导入的 SI 子组件：管理员确认归属，原厂接口复核身份，不伪称自动发现。"""

from .oem_target_identity import normalize_target, target_error, target_text


PARENT_TYPES = {
    "oracle_si_hw_component_server": "oracle_si_server_map",
    "oracle_si_network_port_server": "oracle_si_server_map",
    "oracle_si_hw_component_switch": "oracle_si_netswitch",
    "oracle_si_network_port_switch": "oracle_si_netswitch",
    "oracle_si_hw_component_pdu": "oracle_si_pdu",
}


async def confirm_components(adapter, members, manifest):
    """只接受已确认父设备下的白名单组件；不递归扩张导入关系。"""
    if not isinstance(manifest, (list, tuple)) or len(manifest) > 64:
        target_error("SOURCE_DISCOVERY_LIMIT")
    parents = {item["target_id"]: item for item in members}
    seen, imported = set(parents), []
    for row in manifest:
        if not isinstance(row, dict) or set(row) != {"target_id", "target_type", "display_name", "parent_target_id", "evidence_ref"}:
            target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
        parent = parents.get(row["parent_target_id"])
        if parent is None or PARENT_TYPES.get(row["target_type"]) != parent["target_type"]:
            target_error("SOURCE_EXADATA_MEMBERS_UNVERIFIED")
        guid = row["target_id"]
        import re
        if not isinstance(guid, str) or not re.fullmatch(r"[0-9a-f]{32}", guid) or guid in seen:
            target_error("SOURCE_ITEM_AMBIGUOUS")
        seen.add(guid)
        target_text(row["display_name"])
        target_text(row["evidence_ref"])
        payload, _ = await adapter._native_get("/targets/" + guid, max_bytes=65536)
        item = normalize_target(payload, expected_guid=guid)
        if (item["targetType"], item["targetName"]) != (row["target_type"], row["display_name"]):
            target_error("SOURCE_TARGET_MISMATCH")
        imported.append({**row, "membership_origin": "IMPORT_CONFIRMED"})
    if len(members) + len(imported) > 128:
        target_error("SOURCE_DISCOVERY_LIMIT")
    return sorted([*members, *imported], key=lambda item: item["target_id"])
