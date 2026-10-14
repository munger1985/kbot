(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const dialog = document.getElementById("system-dialog");
  const form = document.getElementById("system-form");
  const result = document.getElementById("system-result");
  const panels = [...document.querySelectorAll("[data-system-panel]")];
  const stepButtons = [...document.querySelectorAll("[data-system-step]")];
  const previous = document.getElementById("system-previous");
  const next = document.getElementById("system-next");
  const enable = document.getElementById("enable-system");
  const versions = { ORACLE: ["19c", "26ai"], MYSQL: ["8.4"], POSTGRESQL: ["16"] };
  const architectureOptions = {
    ORACLE: [["RAC", "Oracle RAC"], ["DATA_GUARD", "Data Guard"], ["RAC_DG", "RAC + Data Guard"]],
    MYSQL: [["MYSQL_REPLICATION", "主从复制"]],
    POSTGRESQL: [["POSTGRES_STREAMING", "Streaming Replication"]],
  };
  const protectionSources = {
    ORACLE: [["ORACLE_RMAN", "Oracle RMAN"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    MYSQL: [["MYSQL_XTRABACKUP", "Percona XtraBackup"], ["MYSQL_ENTERPRISE_BACKUP", "MySQL Enterprise Backup"], ["MYSQL_LOGICAL_DUMP", "逻辑导出"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    POSTGRESQL: [["POSTGRESQL_BASEBACKUP", "pg_basebackup"], ["POSTGRESQL_PGBACKREST", "pgBackRest"], ["POSTGRESQL_BARMAN", "Barman"], ["POSTGRESQL_WALG", "WAL-G"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
  };
  let step = "shape";
  let system = null;
  let systems = [];
  let sources = [];
  let groups = [];
  let sites = [];
  let members = [];
  let relations = [];
  let activeMember = 0;
  let detail = null;
  let openedRequestedSystem = false;

  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const idempotency = () => crypto.randomUUID();
  const field = (name) => form.elements[name];
  const shape = () => field("deployment_shape").value;
  const dbType = () => field("db_type").value;
  const selectedArchitecture = () => form.querySelector('[name="architecture"]:checked')?.value || "";

  function setResult(message, tone) {
    result.textContent = message || "";
    result.dataset.tone = tone || "";
  }

  function architecturePayload() {
    if (shape() === "SINGLE_INSTANCE") return { deployment_category: "SINGLE_INSTANCE", capabilities: [] };
    const architecture = selectedArchitecture();
    if (architecture === "RAC") return { deployment_category: "RAC", capabilities: ["RAC"] };
    if (architecture === "DATA_GUARD") return { deployment_category: "REPLICATION", capabilities: ["DATA_GUARD"] };
    if (architecture === "RAC_DG") return { deployment_category: "COMPOSITE", capabilities: ["RAC", "DATA_GUARD"] };
    if (architecture === "MYSQL_REPLICATION") return { deployment_category: "REPLICATION", capabilities: ["MYSQL_REPLICATION"] };
    return { deployment_category: "REPLICATION", capabilities: ["POSTGRES_STREAMING"] };
  }

  function renderArchitecture() {
    field("version_code").innerHTML = versions[dbType()].map((item) => `<option>${item}</option>`).join("");
    const container = document.getElementById("architecture-options");
    container.hidden = shape() !== "HIGH_AVAILABILITY";
    container.innerHTML = '<legend>兼容架构</legend>' + architectureOptions[dbType()].map(([value, label], index) => `<label><input name="architecture" type="radio" value="${value}" ${index === 0 ? "checked" : ""}><strong>${label}</strong></label>`).join("");
  }

  function setStep(value) {
    step = value;
    panels.forEach((panel) => { panel.hidden = panel.dataset.systemPanel !== value; });
    stepButtons.forEach((button) => button.classList.toggle("active", button.dataset.systemStep === value));
    previous.hidden = value === "shape";
    enable.hidden = value !== "topology";
    next.hidden = value === "topology";
    next.textContent = value === "shape" ? "下一步：数据库与架构" : value === "system" ? "保存草稿并配置成员" : "生成并复核拓扑";
    updateStepStatuses();
  }

  function updateStepStatuses() {
    const state = {
      shape: Boolean(shape()),
      system: Boolean(system),
      members: members.length > 0 && members.every((item) => item.saved),
      topology: Boolean(detail && !(detail.validation_issues || []).some((item) => item.severity === "BLOCKING")),
    };
    stepButtons.forEach((button) => {
      const label = button.querySelector("em");
      label.textContent = state[button.dataset.systemStep] ? "已完成" : "待配置";
      label.className = state[button.dataset.systemStep] ? "good" : "";
    });
  }

  function member(memberKey, label, siteCode, role, order, groupKey = "", memberType = "INSTANCE") {
    const port = dbType() === "ORACLE" ? 1521 : dbType() === "MYSQL" ? 3306 : 5432;
    return { member_key: memberKey, label, site_code: siteCode, expected_role: role, sort_order: order, group_key: groupKey, member_type: memberType, target_id: "", saved: false, display_name: `${field("display_name").value}-${label}`, host: "", port, database: "", oracle_scope: "NON_CDB", oracle_pdb_name: "", username: "", password: "", source_id: "", target_key: "", candidate_ref: "", host_candidate_ref: "", discovery_candidates: [], host_candidates: [], discovery_loading: false };
  }

  const memberSource = (item) => sources.find((source) => source.source_id === item.source_id);
  const memberUsesDiscovery = (item) => ["PROMETHEUS", "ZABBIX"].includes(memberSource(item)?.source_type || "");

  async function discoverMemberLabels(item) {
    saveEditor(item);
    if (!item.source_id) throw new Error("请先选择监控源");
    item.discovery_loading = true; renderMembers();
    try {
      const page = await request(`/diagnostic-sources/${encodeURIComponent(item.source_id)}/instance-discoveries`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ db_types: [dbType()], page_size: 100 }) });
      item.discovery_candidates = (page.items || []).filter((candidate) => candidate.mapping_status === "UNMAPPED");
      item.host_candidates = page.host_items || [];
      if (!item.discovery_candidates.length) throw new Error("该监控源没有可映射的数据库候选");
    } finally {
      item.discovery_loading = false; renderMembers();
    }
  }

  function buildTemplate() {
    const architecture = shape() === "SINGLE_INSTANCE" ? "SINGLE_INSTANCE" : selectedArchitecture();
    if (architecture === "SINGLE_INSTANCE") {
      sites = [{ site_code: "MAIN", display_name: "主站点", expected_role: "ACTIVE", sort_order: 0 }];
      members = [member("INSTANCE-1", "实例 1", "MAIN", "INSTANCE", 0)]; relations = [];
    } else if (architecture === "RAC") {
      sites = [{ site_code: "PRIMARY", display_name: "RAC 站点", expected_role: "PRIMARY", sort_order: 0 }];
      members = [member("RAC-1", "RAC 节点 1", "PRIMARY", "RAC_INSTANCE", 0, "RAC-PRIMARY", "RAC_INSTANCE"), member("RAC-2", "RAC 节点 2", "PRIMARY", "RAC_INSTANCE", 1, "RAC-PRIMARY", "RAC_INSTANCE")]; relations = [];
    } else if (architecture === "RAC_DG") {
      sites = [{ site_code: "PRIMARY", display_name: "主站点", expected_role: "PRIMARY", sort_order: 0 }, { site_code: "STANDBY", display_name: "备站点", expected_role: "STANDBY", sort_order: 1 }];
      members = [member("RAC-1", "主站 RAC 节点 1", "PRIMARY", "RAC_INSTANCE", 0, "RAC-PRIMARY", "RAC_INSTANCE"), member("RAC-2", "主站 RAC 节点 2", "PRIMARY", "RAC_INSTANCE", 1, "RAC-PRIMARY", "RAC_INSTANCE"), member("STANDBY-1", "备库实例", "STANDBY", "STANDBY", 2)];
      relations = [{ relation_key: "DG-1", source_member_key: "RAC-1", target_member_key: "STANDBY-1", relation_type: "DATA_GUARD", transport_mode: "ASYNC", sort_order: 0 }];
    } else {
      const mysql = architecture === "MYSQL_REPLICATION";
      sites = [{ site_code: "PRIMARY", display_name: "主站点", expected_role: "PRIMARY", sort_order: 0 }, { site_code: "STANDBY", display_name: "备站点", expected_role: "STANDBY", sort_order: 1 }];
      members = [member(mysql ? "SOURCE-1" : "PRIMARY-1", mysql ? "Source" : "Primary", "PRIMARY", mysql ? "SOURCE" : "PRIMARY", 0), member(mysql ? "REPLICA-1" : "STANDBY-1", mysql ? "Replica" : "Standby", "STANDBY", mysql ? "REPLICA" : "STANDBY", 1)];
      relations = [{ relation_key: "REPLICATION-1", source_member_key: members[0].member_key, target_member_key: members[1].member_key, relation_type: mysql ? "MYSQL_REPLICATION" : architecture === "DATA_GUARD" ? "DATA_GUARD" : "POSTGRES_STREAMING", transport_mode: "ASYNC", sort_order: 0 }];
    }
    activeMember = 0; renderMembers();
  }

  function renderMembers() {
    const slots = document.getElementById("member-slots");
    slots.innerHTML = members.map((item, index) => `<button type="button" data-member-index="${index}" class="${activeMember === index ? "active" : ""}"><span>${index + 1}</span><span><strong>${shell.escape(item.label)}</strong><small>${shell.escape(item.site_code)} · ${shell.escape(item.expected_role)}</small></span><em class="${item.saved ? "good" : ""}">${item.saved ? "验证通过" : "待配置"}</em></button>`).join("");
    slots.querySelectorAll("[data-member-index]").forEach((button) => { button.onclick = () => { activeMember = Number(button.dataset.memberIndex); renderMembers(); }; });
    const item = members[activeMember];
    const selectedSource = memberSource(item);
    const discoveryFields = memberUsesDiscovery(item)
      ? `<div class="ops-field span-4"><label>数据库 Label</label><select name="member_candidate_ref"><option value="">请选择</option>${item.discovery_candidates.map((candidate) => `<option value="${shell.escape(candidate.candidate_ref)}" ${candidate.candidate_ref === item.candidate_ref ? "selected" : ""}>${shell.escape(candidate.display_name)} · ${shell.escape(candidate.locator_hint)}</option>`).join("")}</select></div>${selectedSource?.source_type === "PROMETHEUS" ? `<div class="ops-field span-4"><label>主机 Label</label><select name="member_host_candidate_ref"><option value="">请选择</option>${item.host_candidates.map((candidate) => `<option value="${shell.escape(candidate.candidate_ref)}" ${candidate.candidate_ref === item.host_candidate_ref ? "selected" : ""}>${shell.escape(candidate.display_name)} · ${shell.escape(candidate.locator_hint)}</option>`).join("")}</select></div>` : ""}<div class="ops-field span-4"><button id="discover-member-labels" type="button">${item.discovery_loading ? "发现中…" : "发现数据库与主机 Label"}</button></div>`
      : `<div class="ops-field span-4"><label>稳定定位键</label><input name="member_target_key" value="${shell.escape(item.target_key)}"><small>Loki 使用精确 target_key 标签；OEM 等来源使用稳定目标标识。</small></div>`;
    document.getElementById("member-editor").innerHTML = `
      <div class="ops-field span-12"><h3>${shell.escape(item.label)}</h3><small>${shell.escape(item.site_code)} · 期望角色 ${shell.escape(item.expected_role)} · 该成员归属于当前数据库系统</small></div>
      <div id="member-new" class="ops-form span-12">
        <div class="ops-field span-6"><label>Target 名称</label><input name="member_display_name" value="${shell.escape(item.display_name)}"></div><div class="ops-field span-6"><label>Host</label><input name="member_host" value="${shell.escape(item.host)}"></div>
        <div class="ops-field span-3"><label>端口</label><input name="member_port" type="number" value="${item.port}"></div><div class="ops-field span-5"><label>${dbType() === "ORACLE" ? "Service Name" : "Database"}</label><input name="member_database" value="${shell.escape(item.database)}"></div>${dbType() === "ORACLE" ? `<div class="ops-field span-4"><label>Oracle 容器范围</label><select name="member_oracle_scope"><option value="NON_CDB" ${item.oracle_scope === "NON_CDB" ? "selected" : ""}>Non-CDB</option><option value="CDB_ROOT" ${item.oracle_scope === "CDB_ROOT" ? "selected" : ""}>CDB Root</option><option value="PDB" ${item.oracle_scope === "PDB" ? "selected" : ""}>PDB</option></select></div>${item.oracle_scope === "PDB" ? `<div class="ops-field span-4"><label>PDB Name</label><input name="member_oracle_pdb_name" value="${shell.escape(item.oracle_pdb_name)}"><small>Service Name 只负责路由，PDB Name 用于核验实际容器。</small></div>` : ""}` : ""}
        <div class="ops-field span-4"><label>诊断用户名</label><input name="member_username" value="${shell.escape(item.username)}" autocomplete="off"></div><div class="ops-field span-4"><label>诊断密码</label><input name="member_password" type="password" value="${shell.escape(item.password)}" autocomplete="new-password"></div>
        <div class="ops-field span-4"><label>监控源</label><select name="member_source_id"><option value="">请选择</option>${sources.map((source) => `<option value="${shell.escape(source.source_id)}" ${source.source_id === item.source_id ? "selected" : ""}>${shell.escape(source.display_name)} · ${shell.escape(source.source_type)}</option>`).join("")}</select></div>${discoveryFields}
      </div><div class="ops-actions span-12"><button id="save-member" class="primary" type="button">${item.saved ? "重新验证成员" : "验证并保存成员"}</button></div>`;
    document.querySelector('[name="member_source_id"]')?.addEventListener("change", (event) => {
      saveEditor(item); item.source_id = event.target.value; item.candidate_ref = ""; item.host_candidate_ref = ""; item.discovery_candidates = []; item.host_candidates = []; renderMembers();
    });
    document.querySelector('[name="member_oracle_scope"]')?.addEventListener("change", (event) => {
      saveEditor(item); item.oracle_scope = event.target.value; renderMembers();
    });
    document.getElementById("discover-member-labels")?.addEventListener("click", () => discoverMemberLabels(item).catch((error) => setResult(error.message, "bad")));
    document.getElementById("save-member").onclick = () => saveMember(item);
  }

  function saveEditor(item) {
    const editor = document.getElementById("member-editor");
    item.display_name = editor.querySelector('[name="member_display_name"]')?.value || item.display_name;
    item.host = editor.querySelector('[name="member_host"]')?.value || item.host;
    item.port = Number(editor.querySelector('[name="member_port"]')?.value || item.port);
    item.database = editor.querySelector('[name="member_database"]')?.value || item.database;
    item.oracle_scope = editor.querySelector('[name="member_oracle_scope"]')?.value || item.oracle_scope;
    item.oracle_pdb_name = editor.querySelector('[name="member_oracle_pdb_name"]')?.value || item.oracle_pdb_name;
    item.username = editor.querySelector('[name="member_username"]')?.value || item.username;
    item.password = editor.querySelector('[name="member_password"]')?.value || item.password;
    item.source_id = editor.querySelector('[name="member_source_id"]')?.value || item.source_id;
    item.target_key = editor.querySelector('[name="member_target_key"]')?.value || item.target_key;
    item.candidate_ref = editor.querySelector('[name="member_candidate_ref"]')?.value || item.candidate_ref;
    item.host_candidate_ref = editor.querySelector('[name="member_host_candidate_ref"]')?.value || item.host_candidate_ref;
  }

  function endpoint(item) {
    return { host: item.host, port: item.port, service: dbType() === "ORACLE" ? item.database : null, database: dbType() === "ORACLE" ? null : item.database, tls_enabled: dbType() !== "ORACLE" };
  }

  async function saveMember(item) {
    saveEditor(item); setResult("", "");
    try {
      const source = memberSource(item);
        if (![item.display_name, item.host, item.database, item.username, item.password, source].every(Boolean)) throw new Error("请完整填写连接、凭据和监控源");
        if (dbType() === "ORACLE" && item.oracle_scope === "PDB" && !item.oracle_pdb_name) throw new Error("PDB 范围必须填写独立的 PDB Name");
        if (memberUsesDiscovery(item) && (!item.candidate_ref || (source.source_type === "PROMETHEUS" && !item.host_candidate_ref))) throw new Error("请发现并选择数据库 Label；Prometheus 还必须选择主机 Label");
        if (!memberUsesDiscovery(item) && !item.target_key) throw new Error("请填写监控源稳定定位键");
        const targetRequest = { display_name: item.display_name, db_type: dbType(), version_code: field("version_code").value, environment: field("environment").value, db_role: ["PRIMARY", "STANDBY"].includes(item.expected_role) ? item.expected_role : "UNKNOWN", oracle_container_scope: dbType() === "ORACLE" ? item.oracle_scope : null, oracle_pdb_name: dbType() === "ORACLE" && item.oracle_scope === "PDB" ? item.oracle_pdb_name : null, endpoint: endpoint(item), readonly_connection_enabled: true, controlled_change_enabled: false, diagnostic_credential: { username: item.username, password: item.password }, importance_level: Number(field("importance_level").value), responsibility_group_id: field("responsibility_group_id").value, capabilities: {} };
        const tested = await request("/targets/test-connection", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ db_type: dbType(), version_code: field("version_code").value, oracle_container_scope: targetRequest.oracle_container_scope, oracle_pdb_name: targetRequest.oracle_pdb_name, endpoint: targetRequest.endpoint, diagnostic_credential: targetRequest.diagnostic_credential }) });
        if (!tested.ok) throw new Error(tested.error_code || "数据库连接验证失败");
        const target = await request("/targets", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotency() }, body: JSON.stringify(targetRequest) });
        if (memberUsesDiscovery(item)) {
          await request(`/diagnostic-sources/${encodeURIComponent(item.source_id)}/instance-mappings`, { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotency() }, body: JSON.stringify({ mappings: [{ candidate_ref: item.candidate_ref, host_candidate_ref: source.source_type === "PROMETHEUS" ? item.host_candidate_ref : null, target_id: target.target_id }] }) });
        } else {
          const locator = source.source_type === "LOKI" ? { labels: { target_key: item.target_key } } : { target_key: item.target_key };
          await request(`/targets/${target.target_id}/source-bindings`, { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotency() }, body: JSON.stringify({ source_id: item.source_id, source_locator_key: item.target_key, source_locator: locator }) });
        }
        item.target_id = target.target_id;
      item.saved = true;
      const pending = members.findIndex((candidate) => !candidate.saved);
      if (pending >= 0) activeMember = pending;
      renderMembers(); updateStepStatuses(); shell.toast(`${item.label} 已验证`);
    } catch (error) { setResult(error.message, "bad"); }
  }

  function renderTopology() {
    const board = document.getElementById("topology-board");
    board.innerHTML = (detail.sites || []).map((site) => `<section><header><strong>${shell.escape(site.display_name)}</strong><small>${shell.escape(site.site_code)}</small></header><div>${detail.members.filter((item) => item.site_code === site.site_code).map((node) => `<button type="button" data-member-key="${shell.escape(node.member_key)}"><strong>${shell.escape(node.target_display_name)}</strong><span>${shell.escape(node.expected_role)} / ${shell.escape(node.observed_role)}</span><small>${node.monitoring_ready ? "监控已映射" : "监控未映射"}</small></button>`).join("")}</div></section>`).join("") + `<div class="system-relation-list">${(detail.relations || []).map((relation) => `<span>${shell.escape(relation.source_member_key)} → ${shell.escape(relation.target_member_key)} · ${shell.escape(relation.relation_type)} · ${shell.escape(relation.transport_mode)}</span>`).join("")}</div>`;
    board.querySelectorAll("[data-member-key]").forEach((button) => { button.onclick = () => { activeMember = members.findIndex((item) => item.member_key === button.dataset.memberKey); renderMembers(); setStep("members"); }; });
    const issues = detail.validation_issues || [];
    document.getElementById("topology-issues").innerHTML = issues.length ? `<div class="system-issue-list">${issues.map((issue) => `<span class="${issue.severity === "BLOCKING" ? "bad" : "warn"}">${shell.escape(issue.message)}</span>`).join("")}</div>` : "";
    enable.disabled = issues.some((issue) => issue.severity === "BLOCKING");
  }

  async function createDraft() {
    if (!field("display_name").value.trim() || !field("responsibility_group_id").value || (shape() === "HIGH_AVAILABILITY" && !selectedArchitecture())) throw new Error("请完整填写系统名称、架构和责任组");
    system = await request("/database-systems", { method: "POST", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotency() }, body: JSON.stringify({ display_name: field("display_name").value.trim(), db_type: dbType(), version_code: field("version_code").value, deployment_shape: shape(), ...architecturePayload(), environment: field("environment").value, importance_level: Number(field("importance_level").value), responsibility_group_id: field("responsibility_group_id").value, description: field("description").value.trim() || null }) });
    document.getElementById("system-draft-state").textContent = `草稿 ${system.database_system_id}`;
    buildTemplate(); setStep("members");
  }

  async function saveTopology() {
    if (!members.every((item) => item.saved)) throw new Error("请先完成全部成员配置");
    detail = await request(`/database-systems/${system.database_system_id}/topology`, { method: "PUT", headers: { "Content-Type": "application/json", "If-Match": `"rv-${system.row_version}"` }, body: JSON.stringify({ sites, members: members.map((item) => ({ member_key: item.member_key, target_id: item.target_id, site_code: item.site_code, group_key: item.group_key || null, member_type: item.member_type, expected_role: item.expected_role, sort_order: item.sort_order })), relations }) });
    system = detail; renderTopology(); setStep("topology");
  }

  async function advance() {
    setResult("", "");
    try {
      if (step === "shape") {
        if (!shape()) throw new Error("请选择单实例或高可用系统");
        renderArchitecture(); setStep("system");
      } else if (step === "system") await createDraft();
      else if (step === "members") await saveTopology();
    } catch (error) { setResult(error.message, "bad"); }
  }

  function retreat() {
    if (step === "system") setStep("shape");
    else if (step === "members") setStep("system");
    else if (step === "topology") setStep("members");
  }

  async function enableSystem(event) {
    event.preventDefault();
    try {
      detail = await request(`/database-systems/${system.database_system_id}/enable`, { method: "POST", headers: { "Idempotency-Key": idempotency(), "If-Match": `"rv-${system.row_version}"` } });
      shell.toast("数据库系统已创建并启用监控；保护状态为未配置"); dialog.close(); await load();
    } catch (error) { setResult(error.message, "bad"); }
  }

  function reset() {
    form.reset(); system = null; detail = null; sites = []; members = []; relations = []; activeMember = 0; setResult("", ""); document.getElementById("system-draft-state").textContent = "尚未保存服务端草稿"; renderArchitecture(); setStep("shape");
  }

  function renderList() {
    document.getElementById("system-list").innerHTML = systems.length ? `<div class="system-card-grid">${systems.map((item) => `<article><header><div><small>DATABASE SYSTEM</small><h3>${shell.escape(item.display_name)}</h3></div>${shell.badge(item.status)}</header><p>${shell.escape(item.db_type)} ${shell.escape(item.version_code)} · ${shell.escape(item.deployment_category)} · ${item.member_count} 个成员</p><dl><dt>保护状态</dt><dd>${item.protection_status === "CONFIGURED" ? "已配置" : "未配置"}</dd><dt>更新时间</dt><dd>${shell.fmt(item.updated_at)}</dd></dl><button type="button" data-system-id="${shell.escape(item.database_system_id)}">查看系统</button></article>`).join("")}</div>` : '<div class="ops-empty">尚未创建数据库系统。</div>';
    document.querySelectorAll("[data-system-id]").forEach((button) => { button.onclick = () => openDetail(button.dataset.systemId); });
  }

  function memberDetailMarkup(item, memberItem) {
    const systemId = encodeURIComponent(item.database_system_id);
    const targetId = encodeURIComponent(memberItem.target_id);
    return `<details class="system-member-card"><summary><span><strong>${shell.escape(memberItem.target_display_name)}</strong><small>${shell.escape(memberItem.expected_role)} / ${shell.escape(memberItem.observed_role)}</small></span>${shell.badge(memberItem.connectivity_status)}</summary><dl><dt>成员类型</dt><dd>${shell.escape(memberItem.member_type)}</dd><dt>连接状态</dt><dd>${shell.escape(memberItem.connectivity_status)}</dd><dt>监控映射</dt><dd>${memberItem.monitoring_ready ? "已配置" : "未配置"}</dd><dt>成员键</dt><dd>${shell.escape(memberItem.member_key)}</dd></dl><div class="ops-actions"><a class="ops-button" href="./target-detail.html?id=${targetId}&database_system_id=${systemId}">成员详情</a><a class="ops-button" href="./targets.html?edit=${targetId}&database_system_id=${systemId}">维护成员配置</a></div></details>`;
  }

  async function openDetail(systemId) {
    const [item, profile] = await Promise.all([
      request(`/database-systems/${systemId}`),
      request(`/database-systems/${systemId}/protection-profile`),
    ]);
    const selectedSources = new Set(profile?.required_backup_source_types || []);
    const value = (name, fallback) => profile?.[name] ?? fallback;
    document.getElementById("system-detail").innerHTML = `<header><div><small>DATABASE SYSTEM</small><h2>${shell.escape(item.display_name)}</h2><p>${shell.escape(item.db_type)} ${shell.escape(item.version_code)} · ${shell.escape(item.deployment_category)}</p></div></header><div class="ops-dialog-body"><section class="ops-panel"><div class="ops-panel-head"><div><h3>系统拓扑与成员</h3><p>${item.member_count} 个成员；展开节点查看连接、监控与维护入口。</p></div></div><div class="ops-panel-body system-topology-board">${item.sites.map((site) => `<section><header><strong>${shell.escape(site.display_name)}</strong><small>${shell.escape(site.site_code)}</small></header><div>${item.members.filter((memberItem) => memberItem.site_code === site.site_code).map((memberItem) => memberDetailMarkup(item, memberItem)).join("")}</div></section>`).join("")}</div></section><section class="ops-panel"><div class="ops-panel-head"><div><h3>备份与恢复保护</h3><p>保护配置归属整个数据库系统；未配置只告警，不影响监控启用。</p></div><a class="ops-button" href="./recovery-drills.html?database_system_id=${encodeURIComponent(item.database_system_id)}">恢复演练</a></div><div class="ops-panel-body"><form id="system-protection-form" class="ops-form"><label class="ops-field span-6">备份策略校验<select name="backup_policy_enabled"><option value="true" ${value("backup_policy_enabled", true) ? "selected" : ""}>启用</option><option value="false" ${!value("backup_policy_enabled", true) ? "selected" : ""}>停用</option></select></label><label class="ops-field span-3">备份频率（小时）<input name="backup_frequency_hours" type="number" min="1" value="${value("backup_frequency_hours", 24)}"></label><label class="ops-field span-3">成功备份最大时效（小时）<input name="backup_success_max_age_hours" type="number" min="1" value="${value("backup_success_max_age_hours", 26)}"></label><label class="ops-field span-3">保留天数<input name="backup_retention_days" type="number" min="1" value="${value("backup_retention_days", 30)}"></label><label class="ops-field span-3">RPO（秒）<input name="rpo_seconds" type="number" min="0" value="${value("rpo_seconds", 3600)}"></label><label class="ops-field span-3">RTO（秒）<input name="rto_seconds" type="number" min="60" value="${value("rto_seconds", 14400)}"></label><label class="ops-field span-3">定期恢复演练<select name="drill_required"><option value="true" ${value("drill_required", true) ? "selected" : ""}>要求</option><option value="false" ${!value("drill_required", true) ? "selected" : ""}>不要求</option></select></label><label class="ops-field span-3">演练周期（天）<input name="required_drill_interval_days" type="number" min="1" value="${value("required_drill_interval_days", 180)}"></label><label class="ops-field span-6">最低保障等级<select name="required_assurance_level"><option value="BACKUP_METADATA">备份元数据</option><option value="RESTORE_VALIDATE">恢复校验</option><option value="DATABASE_OPEN">数据库可打开</option><option value="APPLICATION_VALIDATED">业务验证通过</option></select></label><label class="ops-field span-12">必需备份来源<select name="required_backup_source_types" multiple size="${Math.min(6, (protectionSources[item.db_type] || []).length)}">${(protectionSources[item.db_type] || []).map(([source, label]) => `<option value="${source}" ${selectedSources.has(source) ? "selected" : ""}>${label}</option>`).join("")}</select></label><label class="ops-field span-12">策略说明<textarea name="source_note" rows="3" maxlength="1000">${shell.escape(profile?.source_note || "")}</textarea></label><div class="ops-actions span-12"><button class="primary" type="submit">${profile ? "保存新版本" : "创建保护配置"}</button><span id="system-protection-result" class="ops-connection-result"></span></div></form></div></section></div>`;
    const protectionForm = document.getElementById("system-protection-form");
    protectionForm.elements.required_assurance_level.value = value("required_assurance_level", "DATABASE_OPEN");
    protectionForm.onsubmit = async (event) => {
      event.preventDefault();
      const submit = protectionForm.querySelector('button[type="submit"]');
      const status = document.getElementById("system-protection-result");
      const requiredSources = [...protectionForm.elements.required_backup_source_types.selectedOptions].map((option) => option.value);
      const payload = {
        backup_policy_enabled: protectionForm.elements.backup_policy_enabled.value === "true",
        backup_frequency_hours: Number(protectionForm.elements.backup_frequency_hours.value),
        backup_success_max_age_hours: Number(protectionForm.elements.backup_success_max_age_hours.value),
        backup_retention_days: Number(protectionForm.elements.backup_retention_days.value),
        rpo_seconds: Number(protectionForm.elements.rpo_seconds.value),
        rto_seconds: Number(protectionForm.elements.rto_seconds.value),
        drill_required: protectionForm.elements.drill_required.value === "true",
        required_drill_interval_days: Number(protectionForm.elements.required_drill_interval_days.value),
        required_assurance_level: protectionForm.elements.required_assurance_level.value,
        required_backup_source_types: requiredSources,
        rto_clock_basis: "SERVICE_UNAVAILABLE_TO_VALIDATED",
        source_note: protectionForm.elements.source_note.value.trim() || null,
      };
      if ((payload.backup_policy_enabled || payload.drill_required) && !requiredSources.length) {
        status.textContent = "启用备份校验或恢复演练时，请至少选择一个备份来源。";
        status.dataset.tone = "bad";
        return;
      }
      submit.disabled = true;
      try {
        await request(`/database-systems/${systemId}/protection-profile`, { method: "PUT", headers: { "Content-Type": "application/json", "Idempotency-Key": idempotency() }, body: JSON.stringify(payload) });
        shell.toast("数据库系统保护配置已保存为新版本");
        await openDetail(systemId);
      } catch (error) {
        status.textContent = error.message;
        status.dataset.tone = "bad";
        submit.disabled = false;
      }
    };
    document.getElementById("system-detail-dialog").showModal();
  }

  async function load() {
    const [systemPage, sourcePage, groupPage] = await Promise.all([request("/database-systems?limit=200"), request("/diagnostic-sources?limit=200"), request("/responsibility-groups")]);
    systems = systemPage.items || []; sources = (sourcePage.items || []).filter((item) => item.status === "ENABLED"); groups = (groupPage.items || []).filter((item) => item.status === "ACTIVE");
    field("responsibility_group_id").innerHTML = '<option value="">请选择责任组</option>' + groups.map((group) => `<option value="${shell.escape(group.responsibility_group_id)}">${shell.escape(group.name)}</option>`).join(""); renderList();
    const requestedSystemId = new URLSearchParams(location.search).get("open");
    if (requestedSystemId && !openedRequestedSystem && systems.some((item) => item.database_system_id === requestedSystemId)) {
      openedRequestedSystem = true;
      await openDetail(requestedSystemId);
    }
  }

  shell.ready.then(() => {
    document.getElementById("create-system").onclick = () => { reset(); dialog.showModal(); };
    document.getElementById("close-system-dialog").onclick = () => dialog.close();
    document.getElementById("refresh-systems").onclick = load;
    document.getElementById("close-system-detail").onclick = () => document.getElementById("system-detail-dialog").close();
    field("db_type").onchange = renderArchitecture;
    next.onclick = advance; previous.onclick = retreat; form.onsubmit = enableSystem;
    stepButtons.forEach((button) => { button.onclick = () => { if (button.dataset.systemStep === step || (system && button.dataset.systemStep !== "topology")) setStep(button.dataset.systemStep); }; });
    return load();
  }).catch((error) => shell.toast(error.message));
})();
