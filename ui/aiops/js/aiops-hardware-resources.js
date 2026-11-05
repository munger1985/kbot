(function () {
  "use strict";
  const hostTypes = new Set(["PHYSICAL_SERVER", "VIRTUAL_MACHINE"]);
  const hardwareTypes = new Set([...hostTypes, "STORAGE", "NETWORK_DEVICE"]);
  function draft(bindings = []) {
    const binding = bindings.find(row => row.status === "ACTIVE") || bindings[0];
    return { source_id: binding?.source_id || "", candidate_ref: binding ? `current:${binding.binding_id}` : "", locator_key: binding?.source_locator_key || "", candidates: binding ? [{ candidate_ref: `current:${binding.binding_id}`, locator_key: binding.source_locator_key, display_name: "当前主机" }] : [], discovering: false, revision: 0, original_binding: binding };
  }
  function create(type, layer, owner, uuid) {
    return { draft_resource_id: uuid(), mode: "NEW", resource_type_key: type, display_name: "", identity_kind: "STABLE", owner_group_id: owner, architecture_layer_key: layer, responsibility_group_id: "", location_name: "", display_order: 0, saved: false, attributes: {}, hardware: draft() };
  }
  function changed(item) {
    const state = item.hardware;
    return state.source_id !== (state.original_binding?.source_id || "") || state.locator_key !== (state.original_binding?.source_locator_key || "");
  }
  function mapping(item, patch = false) {
    const state = item.hardware;
    if (!state?.source_id || (patch && !changed(item))) return undefined;
    const result = { source_id: state.source_id, candidate_ref: state.candidate_ref };
    if (patch) Object.assign(result, { binding_id: state.original_binding?.binding_id || null, binding_row_version: state.original_binding?.row_version || null });
    return result;
  }
  function error(item) {
    if (!item.display_name.trim() || !item.owner_group_id) return "请填写资源名称和技术责任组。";
    if (item.hardware?.source_id && (!item.hardware.candidate_ref || (changed(item) && item.hardware.candidate_ref.startsWith("current:")))) return "请重新发现并选择主机候选。";
    if (item.hardware?.original_binding && !item.hardware.source_id) return "不能隐式移除已有主机绑定；请保留或重新选择监控源。";
    return "";
  }
  function relationSpec(item) {
    return hostTypes.has(item.resource_type_key) ? ["RUNS_ON", "WORKLOAD", "HOST"] : item.resource_type_key === "STORAGE" ? ["USES_STORAGE", "CONSUMER", "PROVIDER"] : ["DEPENDS_ON", "SOURCE", "TARGET"];
  }
  function linked(item, relations) {
    const [type, sourceRole, targetRole] = relationSpec(item);
    return relations.filter(row => row.relation_type_key === type && row.participants.some(p => p.participant_role === targetRole && p.draft_resource_id === item.draft_resource_id)).flatMap(row => row.participants.filter(p => p.participant_role === sourceRole).map(p => p.draft_resource_id));
  }
  function setLinked(item, ids, resources, relations, uuid) {
    const [type, sourceRole, targetRole] = relationSpec(item);
    const current = relations.filter(row => row.relation_type_key === type && row.participants.some(p => p.participant_role === targetRole && p.draft_resource_id === item.draft_resource_id));
    const retained = relations.filter(row => !current.includes(row));
    for (const id of ids) {
      const workload = resources.find(row => row.draft_resource_id === id);
      if (!workload || id === item.draft_resource_id) continue;
      const existing = current.find(row => row.participants.length === 2 && row.participants.some(p => p.participant_role === sourceRole && p.draft_resource_id === id));
      retained.push(existing || { draft_relation_id: uuid(), relation_type_key: type, display_name: `${workload.display_name} → ${item.display_name}`, origin: "MANUAL", confirmation_status: "CONFIRMED", responsibility_group_id: "", attributes: {}, participants: [{ draft_resource_id: id, participant_role: sourceRole, display_order: 0 }, { draft_resource_id: item.draft_resource_id, participant_role: targetRole, display_order: 1 }] });
    }
    return retained;
  }
  async function discover(item, request) {
    const state = item.hardware, sourceId = state.source_id, ticket = ++state.revision;
    state.discovering = true;
    const candidates = [];
    let cursor = null;
    try {
      do {
        const page = await request(`/diagnostic-sources/${encodeURIComponent(sourceId)}/instance-discoveries`, { method: "POST", body: JSON.stringify({ resource_scope: "HOST", page_size: 100, cursor }) });
        if (ticket !== state.revision || state.source_id !== sourceId) return;
        candidates.push(...page.host_items);
        cursor = page.has_more ? page.next_cursor : null;
        if (page.has_more && !cursor) throw new Error("主机发现返回了无效分页，请重试。");
        if (cursor && candidates.length >= 1000) throw new Error("主机候选超过 1000 个，请缩小监控源范围。");
      } while (cursor);
      state.candidates = candidates;
      state.candidate_ref = candidates.find(row => row.locator_key === state.locator_key)?.candidate_ref || "";
    } finally { if (ticket === state.revision) state.discovering = false; }
  }
  globalThis.KBotAIOpsHardware = { hostTypes, hardwareTypes, draft, create, changed, mapping, error, relationSpec, linked, setLinked, discover };
})();
