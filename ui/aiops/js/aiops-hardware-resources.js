(function () {
  "use strict";
  const hostTypes = new Set(["PHYSICAL_SERVER", "VIRTUAL_MACHINE", "CONTAINER"]);
  const serviceTypes = new Set(["ORACLE_ASM", "ORACLE_RAC", "ORACLE_LISTENER"]);
const hardwareTypes = new Set([...hostTypes, ...serviceTypes, "STORAGE", "NETWORK_DEVICE", "EXADATA"]);
  function draft(bindings = []) {
    const binding = bindings.find(row => row.status === "ACTIVE") || bindings[0];
    const sourceBinding = JSON.stringify({ source_locator_key: binding?.source_locator_key || '', source_locator: binding?.source_locator || {}, mapping_overrides: binding?.mapping_overrides || {}, capability_scope: binding?.capability_scope || {}, query_budget: binding?.query_budget || {}, role: binding?.role || 'PRIMARY', priority: binding?.priority ?? 100 }, null, 2);
    return { component_manifest_text: '[]', confirmed_manifest: [], source_binding_text: sourceBinding, original_source_binding_text: sourceBinding, source_id: binding?.source_id || "", candidate_ref: binding ? `current:${binding.binding_id}` : "", locator_key: binding?.source_locator_key || "", candidates: binding ? [{ candidate_ref: `current:${binding.binding_id}`, locator_key: binding.source_locator_key, display_name: "当前绑定" }] : [], discovering: false, revision: 0, original_binding: binding };
  }
  function create(type, layer, owner, uuid) {
    return { draft_resource_id: uuid(), mode: "NEW", resource_type_key: type, display_name: "", identity_kind: "STABLE", owner_group_id: owner, architecture_layer_key: layer, responsibility_group_id: "", location_name: "", display_order: 0, saved: false, attributes: {}, hardware: draft() };
  }
  function changed(item) {
    const state = item.hardware;
    return state.source_id !== (state.original_binding?.source_id || "") || state.locator_key !== (state.original_binding?.source_locator_key || "") || state.source_binding_text !== state.original_source_binding_text || (state.confirmed_manifest?.length || 0) > 0;
  }
  function mapping(item, patch = false) {
    const state = item.hardware;
    if (!state?.source_id || (patch && !changed(item))) return undefined;
    const result = { source_id: state.source_id, candidate_ref: state.candidate_ref };
  if (serviceTypes.has(item.resource_type_key)) {
    delete result.candidate_ref;
    result.source_binding = { ...JSON.parse(state.source_binding_text), source_id: state.source_id };
  } else if (item.resource_type_key === 'EXADATA') result.component_manifest = state.confirmed_manifest;
  if (patch) Object.assign(result, { binding_id: state.original_binding?.binding_id || null, binding_row_version: state.original_binding?.row_version || null });
    return result;
  }
  function error(item) {
    if (!item.display_name.trim() || !item.owner_group_id) return "请填写资源名称和技术责任组。";
  if (serviceTypes.has(item.resource_type_key)) {
    if (item.hardware?.original_binding && !item.hardware.source_id) return '不能隐式移除已有服务绑定。';
    if (!item.hardware?.source_id) return '';
    try { const value = JSON.parse(item.hardware.source_binding_text); if (!value.source_locator_key || !value.source_locator || value.source_locator.resource_scope !== item.resource_type_key) return '请填写服务定位键及与资源类型一致的 resource_scope。'; } catch { return '服务绑定配置必须是 JSON 对象。'; }
    return '';
  }
  if (item.resource_type_key === 'EXADATA') {
    try {
      const manifest = JSON.parse(item.hardware.component_manifest_text || '[]');
      if (!Array.isArray(manifest)) return '组件清单必须是 JSON 数组。';
      if (JSON.stringify(manifest) !== JSON.stringify(item.hardware.confirmed_manifest || [])) return '组件清单已变化，请重新发现、复核并选择整机候选。';
    } catch { return '组件清单必须是有效的 JSON 数组。'; }
  }
  if (item.hardware?.source_id && (!item.hardware.candidate_ref || (changed(item) && item.hardware.candidate_ref.startsWith("current:")))) return "请重新发现并选择主机候选。";
    if (item.hardware?.original_binding && !item.hardware.source_id) return "不能隐式移除已有主机绑定；请保留或重新选择监控源。";
    return "";
  }
  function relationSpec(item) {
    return item.resource_type_key === "EXADATA" ? ["USES_EXADATA", "DATABASE", "EXADATA"] : hostTypes.has(item.resource_type_key) ? ["RUNS_ON", "WORKLOAD", "HOST"] : item.resource_type_key === "STORAGE" ? ["USES_STORAGE", "CONSUMER", "PROVIDER"] : serviceTypes.has(item.resource_type_key) ? ["DEPENDS_ON", "CONSUMER", "PROVIDER"] : ["DEPENDS_ON", "SOURCE", "TARGET"];
  }
  function linked(item, relations) {
    const [type, sourceRole, targetRole] = relationSpec(item);
    return relations.filter(row => row.relation_type_key === type && row.participants.some(p => p.participant_role === targetRole && p.draft_resource_id === item.draft_resource_id)).flatMap(row => row.participants.filter(p => p.participant_role === sourceRole).map(p => p.draft_resource_id));
  }
  function workloads(item, resources) {
    return resources.filter(row => row.draft_resource_id !== item.draft_resource_id && (row.database || row.resource_type_key === "DATABASE" || (hostTypes.has(item.resource_type_key) && row.resource_type_key === "CONTAINER")));
  }
  function setLinked(item, ids, resources, relations, uuid) {
    const [type, sourceRole, targetRole] = relationSpec(item);
    const visible = new Set(workloads(item, resources).map(row => row.draft_resource_id));
    const current = relations.filter(row => row.relation_type_key === type && row.participants.some(p => p.participant_role === targetRole && p.draft_resource_id === item.draft_resource_id) && row.participants.some(p => p.participant_role === sourceRole && visible.has(p.draft_resource_id)));
    const retained = relations.filter(row => !current.includes(row));
    for (const id of ids) {
      const workload = resources.find(row => row.draft_resource_id === id);
      if (!workload || !visible.has(id)) continue;
      const existing = current.find(row => row.participants.length === 2 && row.participants.some(p => p.participant_role === sourceRole && p.draft_resource_id === id));
      retained.push(existing || { draft_relation_id: uuid(), relation_type_key: type, display_name: `${workload.display_name} → ${item.display_name}`, origin: "MANUAL", confirmation_status: "CONFIRMED", responsibility_group_id: "", attributes: {}, participants: [{ draft_resource_id: id, participant_role: sourceRole, display_order: 0 }, { draft_resource_id: item.draft_resource_id, participant_role: targetRole, display_order: 1 }] });
    }
    return retained;
  }
  function exadataMembership(members) {
    if (!Array.isArray(members)) return null;
    // 只比较成员身份与归属证据；接口返回顺序和对象字段顺序不影响确认结果。
    return JSON.stringify(members.map(row => JSON.stringify([
      row.target_id, row.target_type, row.display_name, row.parent_target_id ?? null,
      row.membership_origin ?? "OEM_ASSOCIATION", row.evidence_ref ?? null,
    ])).sort());
  }
  async function discover(item, request, { requireSelection = false } = {}) {
    const state = item.hardware, sourceId = state.source_id, ticket = ++state.revision;
    state.discovering = true;
    const candidates = [];
    let cursor = null;
    try {
      if (item.resource_type_key === "EXADATA") {
        const locatorKey = state.locator_key, manifestText = state.component_manifest_text;
        const selected = state.candidates.find(row => row.locator_key === locatorKey && row.candidate_ref === state.candidate_ref);
        const previousMembers = exadataMembership(selected?.members);
        state.candidate_ref = "";
        const manifest = JSON.parse(manifestText || "[]");
        if (!Array.isArray(manifest)) throw new Error("组件清单必须是 JSON 数组。");
        if (requireSelection && (!locatorKey || !selected || previousMembers === null)) throw new Error("请重新发现并确认 Exadata 整机及成员。");
        if (requireSelection && JSON.stringify(manifest) !== JSON.stringify(state.confirmed_manifest || [])) throw new Error("组件清单已变化，请重新发现并确认 Exadata 整机及成员。");
        const body = manifest.length || requireSelection ? { root_id: locatorKey, component_manifest: manifest } : {};
        const page = await request(`/diagnostic-sources/${encodeURIComponent(sourceId)}/exadata-discoveries`, { method: "POST", body: JSON.stringify(body) });
        if (ticket !== state.revision || state.source_id !== sourceId || state.locator_key !== locatorKey || state.component_manifest_text !== manifestText || state.candidate_ref) {
          if (requireSelection) throw new Error("Exadata 发现期间配置已变化，请重新发现并确认。");
          return;
        }
        state.candidates = page.items;
        if (requireSelection) {
          const matches = page.items.filter(row => row.locator_key === locatorKey);
          if (matches.length !== 1) throw new Error("原 Exadata 整机候选已不存在或不唯一，请重新发现并确认。");
          const candidate = matches[0];
          if (exadataMembership(candidate.members) !== previousMembers) throw new Error("Exadata 成员已变化，请重新发现、复核并选择整机候选。");
          if (!candidate.candidate_ref) throw new Error("Exadata 整机候选引用无效，请重新发现并确认。");
          state.candidate_ref = candidate.candidate_ref;
        }
        state.confirmed_manifest = manifest;
        return;
      }
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
  globalThis.KBotAIOpsHardware = { hostTypes, serviceTypes, hardwareTypes, draft, create, changed, mapping, error, relationSpec, linked, workloads, setLinked, discover };
})();
