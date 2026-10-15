(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const esc = shell.escape;
  const panel = document.getElementById("workspace-panel");
  const systemId = new URLSearchParams(location.search).get("id") || "new";
  const creating = systemId === "new";
  const draftKey = `kbot.aiops.business-system.${systemId}`;

  const steps = ["BUSINESS_INFO", "SYSTEM_COMPOSITION", "TOPOLOGY_CONFIRMATION"];
  const layers = [
    ["ACCESS", "接入层", "负载均衡、网关与反向代理"],
    ["APPLICATION", "应用层", "应用服务及其专属计算资源"],
    ["MIDDLEWARE", "中间件层", "消息队列、应用服务器与集成平台"],
    ["DATA", "数据层", "数据库与缓存服务"],
    ["INFRASTRUCTURE", "基础设施层", "共享计算、网络和存储"],
  ];
  const resourceCatalog = [
    ["DATABASE", "数据库", "数据平台", true], ["CACHE", "缓存", "数据平台", false],
    ["PHYSICAL_SERVER", "物理服务器", "计算资源", false],
    ["VIRTUAL_MACHINE", "虚拟机", "计算资源", false], ["KUBERNETES_CLUSTER", "容器集群", "计算资源", false],
    ["APPLICATION_SERVICE", "应用服务", "应用与服务", false], ["APPLICATION_SERVER", "应用服务器", "中间件", false],
    ["MESSAGE_QUEUE", "消息队列", "中间件", false], ["LOAD_BALANCER", "负载均衡", "网络设施", false],
    ["NETWORK_DEVICE", "网络设备", "网络设施", false], ["STORAGE", "存储设备", "存储设施", false],
  ];
  const catalogTypesByLayer = {
    ACCESS: ["LOAD_BALANCER", "NETWORK_DEVICE"],
    APPLICATION: ["APPLICATION_SERVICE"],
    MIDDLEWARE: ["APPLICATION_SERVER", "MESSAGE_QUEUE"],
    DATA: ["DATABASE", "CACHE"],
    INFRASTRUCTURE: ["PHYSICAL_SERVER", "VIRTUAL_MACHINE", "KUBERNETES_CLUSTER", "NETWORK_DEVICE", "STORAGE"],
  };
  const versions = { ORACLE: ["19c", "26ai"], MYSQL: ["8.4"], POSTGRESQL: ["16", "17"] };
  const relationTypes = [
    ["ORACLE_RAC", "组成 Oracle RAC"], ["DATA_GUARD", "Data Guard 保护"],
    ["OGG", "Oracle GoldenGate 复制"], ["MYSQL_REPLICATION", "MySQL 主从复制"],
    ["POSTGRES_STREAMING", "PostgreSQL 流复制"], ["RUNS_ON", "运行于"],
    ["USES_STORAGE", "使用存储"], ["DEPENDS_ON", "依赖"], ["LOAD_BALANCES", "负载均衡到"],
  ];
  const roleOptions = {
    ORACLE_RAC: ["MEMBER"], DATA_GUARD: ["PRIMARY", "STANDBY"],
    OGG: ["SOURCE", "TARGET"], MYSQL_REPLICATION: ["SOURCE", "REPLICA"],
    POSTGRES_STREAMING: ["PRIMARY", "STANDBY"], RUNS_ON: ["WORKLOAD", "HOST"],
    USES_STORAGE: ["CONSUMER", "PROVIDER"], DEPENDS_ON: ["CONSUMER", "PROVIDER"],
    LOAD_BALANCES: ["BALANCER", "BACKEND"],
  };
  const roleLabels = {
    MEMBER: "成员", PRIMARY: "主库", STANDBY: "备库", SOURCE: "源端", TARGET: "目标端",
    REPLICA: "从库", WORKLOAD: "工作负载", HOST: "承载主机", CONSUMER: "使用方",
    PROVIDER: "提供方", BALANCER: "负载均衡", BACKEND: "后端服务",
  };

  let step = 0;
  let furthest = 0;
  let loading = true;
  let saving = false;
  let dirty = false;
  let resourceDrawerOpen = false;
  let resourceDrawerMode = "CATALOG";
  let detail = null;
  let groups = [];
  let sources = [];
  let resources = [];
  let relations = [];
  let activeLayer = "DATA";
  let activeResourceId = "";
  let activeRelationId = "";
  let catalogType = "DATABASE";
  let pendingLeaveHref = "";
  let system = { display_name: "", environment: "PROD", importance_level: 3, responsibility_group_id: "", description: "" };
  let lastTimestamp = -1;
  let sequence = 0;

  function uuid7() {
    const bytes = new Uint8Array(16);
    if (globalThis.crypto?.getRandomValues) globalThis.crypto.getRandomValues(bytes);
    else for (let index = 0; index < bytes.length; index += 1) bytes[index] = Math.floor(Math.random() * 256);
    const timestamp = Math.max(Date.now(), lastTimestamp);
    if (timestamp === lastTimestamp) sequence = (sequence + 1) & 0x0fff;
    else {
      sequence = ((bytes[6] & 0x0f) << 8) | bytes[7];
      lastTimestamp = timestamp;
    }
    let remaining = timestamp;
    for (let index = 5; index >= 0; index -= 1) {
      bytes[index] = remaining % 256;
      remaining = Math.floor(remaining / 256);
    }
    bytes[6] = 0x70 | (sequence >> 8);
    bytes[7] = sequence & 0xff;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const value = Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
    return `${value.slice(0, 8)}-${value.slice(8, 12)}-${value.slice(12, 16)}-${value.slice(16, 20)}-${value.slice(20)}`;
  }

  function mark() { dirty = true; updateFooter(); }
  function currentResource() { return resources.find((item) => item.draft_resource_id === activeResourceId); }
  function currentRelation() { return relations.find((item) => item.draft_relation_id === activeRelationId); }
  function groupName(value) { return groups.find((item) => item.responsibility_group_id === value)?.name || "继承业务责任组"; }
  function layerName(value) { return layers.find((item) => item[0] === value)?.[1] || value; }
  function typeName(value) { return resourceCatalog.find((item) => item[0] === value)?.[1] || value; }
  function relationName(value) { return relationTypes.find((item) => item[0] === value)?.[1] || value; }
  function notifyValidationIssues(value) {
    if (!value?.validation_issues?.length) return;
    shell.toast(value.validation_issues.map((item) => item.message).join("；"));
  }
  function sourceFor(item) { return sources.find((source) => source.source_id === item.database?.source_id); }
  function usesDiscovery(item) { return ["PROMETHEUS", "ZABBIX"].includes(sourceFor(item)?.source_type || ""); }
  function isRacResource(item) { return item?.database?.deployment_type === "RAC" && Boolean(item.rac_group_id); }
  function racMembers(item) {
    if (!item) return [];
    if (!item.rac_group_id) return [item];
    return resources
      .filter((row) => row.rac_group_id === item.rac_group_id)
      .sort((left, right) => Number(left.database?.rac_node_number || 0) - Number(right.database?.rac_node_number || 0));
  }
  function managedRacRelation(item) {
    return item?.rac_group_id ? relations.find((relation) => relation.draft_relation_id === item.rac_group_id && relation.relation_type_key === "ORACLE_RAC") : undefined;
  }
  function isManagedRacRelation(relation) {
    return relation?.relation_type_key === "ORACLE_RAC"
      && relation.participants.length >= 2
      && relation.participants.every((participant) => resources.find((resource) => resource.draft_resource_id === participant.draft_resource_id)?.rac_group_id === relation.draft_relation_id);
  }
  function updateRacRelation(item) {
    const relation = managedRacRelation(item);
    if (!relation) return;
    relation.display_name = item.rac_cluster_name?.trim() || "";
    relation.participants = racMembers(item).map((member, index) => ({ draft_resource_id: member.draft_resource_id, participant_role: "MEMBER", display_order: index }));
  }
  function syncRacNames(item) {
    const name = item.rac_cluster_name || "";
    racMembers(item).forEach((member) => {
      member.rac_cluster_name = name;
      member.display_name = name.trim() ? `${name.trim()} · 节点 ${member.database?.rac_node_number}` : "";
      member.saved = Boolean(member.system_resource_id);
    });
    updateRacRelation(item);
    mark();
  }
  function syncRacResourceField(item, field) {
    racMembers(item).forEach((member) => {
      member[field] = item[field];
      member.saved = Boolean(member.system_resource_id);
    });
    mark();
  }
  function syncRacDatabaseField(item, field) {
    if (!item.database) return;
    racMembers(item).forEach((member) => {
      if (!member.database) return;
      member.database[field] = item.database[field];
      member.saved = Boolean(member.system_resource_id);
    });
    mark();
  }
  function selectOptions(items, selected, labels = null) {
    return items.map((value) => {
      const key = Array.isArray(value) ? value[0] : value;
      const label = Array.isArray(value) ? value[1] : labels?.[key] ? `${labels[key]}（${key}）` : key;
      return `<option value="${esc(key)}" ${key === selected ? "selected" : ""}>${esc(label)}</option>`;
    }).join("");
  }
  function groupOptions(selected, emptyLabel) {
    return `<option value="">${esc(emptyLabel)}</option>${groups.map((item) => `<option value="${esc(item.responsibility_group_id)}" ${item.responsibility_group_id === selected ? "selected" : ""}>${esc(item.name)}</option>`).join("")}`;
  }

  function emptyDatabaseResource() {
    return {
      draft_resource_id: uuid7(), mode: "NEW", resource_type_key: "DATABASE", display_name: "",
      owner_group_id: system.responsibility_group_id, architecture_layer_key: "DATA",
      responsibility_group_id: "", location_name: "", display_order: resources.length, saved: false,
      database: {
        db_type: "ORACLE", version_code: "19c", environment: system.environment,
        deployment_type: "SINGLE_INSTANCE", rac_node_number: null, db_role: "UNKNOWN",
        host: "", port: 1521, database: "", oracle_scope: "NON_CDB", oracle_pdb_name: "",
        username: "", password: "", source_id: "", locator_key: "", candidate_ref: "",
        host_candidate_ref: "", candidates: [], host_candidates: [], discovering: false, discovery_revision: 0,
      },
    };
  }

  function newRacNode(source, nodeNumber) {
    const database = source.database;
    const item = {
      draft_resource_id: uuid7(), rac_group_id: source.rac_group_id, rac_cluster_name: source.rac_cluster_name,
      mode: "NEW", resource_type_key: "DATABASE", display_name: "",
      owner_group_id: source.owner_group_id, architecture_layer_key: source.architecture_layer_key,
      responsibility_group_id: "", location_name: source.location_name, display_order: resources.length, saved: false,
      database: {
        ...database, deployment_type: "RAC", rac_node_number: nodeNumber,
        host: "", database: "", locator_key: "", candidate_ref: "", host_candidate_ref: "",
        candidates: [...database.candidates], host_candidates: [...database.host_candidates], discovering: false,
      },
    };
    item.display_name = item.rac_cluster_name?.trim() ? `${item.rac_cluster_name.trim()} · 节点 ${nodeNumber}` : "";
    return item;
  }

  function ensureRacGroup(item) {
    if (!item.database || item.mode !== "NEW") return;
    item.rac_cluster_name ||= item.display_name;
    item.database.db_type = "ORACLE";
    item.database.deployment_type = "RAC";
    item.database.rac_node_number ||= 1;
    if (!item.rac_group_id) {
      const relationId = uuid7();
      item.rac_group_id = relationId;
      relations.push({ draft_relation_id: relationId, relation_type_key: "ORACLE_RAC", display_name: item.rac_cluster_name || "", responsibility_group_id: "", participants: [], attributes: {} });
    }
    if (racMembers(item).length < 2) resources.push(newRacNode(item, 2));
    syncRacNames(item);
    updateRacRelation(item);
  }

  function addRacNode(item) {
    ensureRacGroup(item);
    const nextNumber = Math.max(...racMembers(item).map((member) => Number(member.database?.rac_node_number || 0))) + 1;
    resources.push(newRacNode(item, nextNumber));
    syncRacNames(item);
    updateRacRelation(item);
  }

  function removeRacNode(item, node) {
    const members = racMembers(item);
    if (members.length <= 2) { shell.toast("Oracle RAC 至少需要两个节点。"); return; }
    resources = resources.filter((row) => row !== node);
    if (activeResourceId === node.draft_resource_id) activeResourceId = members.find((row) => row !== node)?.draft_resource_id || "";
    updateRacRelation(item);
    mark();
  }

  function closeResourceDrawer() {
    resourceDrawerOpen = false;
    const drawer = document.getElementById("business-resource-drawer");
    if (drawer?.open) drawer.close();
  }

  function openResourceCatalog() {
    const layerCatalog = resourceCatalog.filter((item) => catalogTypesByLayer[activeLayer].includes(item[0]));
    catalogType = layerCatalog.find((item) => item[3])?.[0] || layerCatalog[0]?.[0] || "";
    resourceDrawerMode = "CATALOG";
    resourceDrawerOpen = true;
    renderComposition();
  }

  function openResourceEditor(resourceId) {
    activeResourceId = resourceId;
    resourceDrawerMode = "EDITOR";
    resourceDrawerOpen = true;
    renderComposition();
  }

  function renderBusiness() {
    panel.innerHTML = `<section class="business-step-surface">
      <header><div><p class="business-eyebrow">STEP 1</p><h2>业务信息</h2></div></header>
      <div class="business-form-section"><div><h3>基本信息</h3><p>名称面向业务人员和运维人员共同识别。</p></div><div class="business-form-grid"><label class="span-8"><span>业务系统名称</span><input name="display_name" value="${esc(system.display_name)}" placeholder="例如：MES 生产系统"></label><label class="span-4"><span>运行环境</span><select name="environment">${selectOptions([["PROD", "生产"], ["STG", "预发布"], ["DEV", "开发"]], system.environment)}</select></label></div></div>
      <div class="business-form-section"><div><h3>责任与影响</h3><p>业务责任组处理整体影响，资源可在下一步指定技术责任组。</p></div><div class="business-form-grid"><label class="span-6"><span>业务责任组</span><select name="responsibility_group_id">${groupOptions(system.responsibility_group_id, "请选择")}</select></label><label class="span-6"><span>重要级</span><select name="importance_level">${selectOptions([["1", "L1 低"], ["2", "L2 一般"], ["3", "L3 重要"], ["4", "L4 关键"], ["5", "L5 核心"]], String(system.importance_level))}</select></label></div></div>
      <div class="business-form-section"><div><h3>运维说明</h3><p>记录服务范围、影响窗口或跨团队协作背景。</p></div><label class="business-description"><span>说明（可选）</span><textarea name="description" rows="5" maxlength="1000">${esc(system.description)}</textarea></label></div>
    </section>`;
    panel.querySelectorAll("[name]").forEach((input) => {
      input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
        system[input.name] = input.name === "importance_level" ? Number(input.value) : input.value;
        mark();
      });
    });
  }

  function catalogHtml() {
    const groupsByName = new Map();
    const layerCatalog = resourceCatalog.filter((item) => catalogTypesByLayer[activeLayer].includes(item[0]));
    layerCatalog.forEach((item) => { if (!groupsByName.has(item[2])) groupsByName.set(item[2], []); groupsByName.get(item[2]).push(item); });
    const options = [...groupsByName.entries()].map(([name, items]) => `<optgroup label="${esc(name)}">${items.map((item) => `<option value="${esc(item[0])}" ${catalogType === item[0] ? "selected" : ""}>${esc(item[1])}${item[3] ? "" : "（后续开放）"}</option>`).join("")}</optgroup>`).join("");
    const enabled = layerCatalog.find((item) => item[0] === catalogType)?.[3];
    return `<div class="business-resource-drawer-shell"><header><div><span>${esc(layerName(activeLayer))}</span><h2>添加资源</h2><p>选择要接入当前架构层的运维资源类型。</p></div><button id="close-resource-drawer" type="button" aria-label="关闭添加资源">关闭</button></header><div class="business-resource-drawer-body"><label class="business-resource-type-field"><span>资源类型</span><select id="catalog-type">${options}</select></label><article class="business-resource-type-summary"><span class="business-resource-icon">${catalogType === "DATABASE" ? "DB" : "R"}</span><div><strong>${esc(typeName(catalogType))}</strong><p>${enabled ? "接入新数据库，并从监控源自动发现数据库和承载主机。" : "该资源类型将在后续版本开放。"}</p></div></article></div><footer><button id="cancel-resource-drawer" type="button">取消</button><button id="confirm-add-resource" class="primary" type="button">配置${esc(typeName(catalogType))}</button></footer></div>`;
  }

  function resourceEditorHtml(item) {
    if (!item) return `<aside class="business-resource-guidance"><h3>选择资源查看详情</h3><p>大型系统仍以列表为主；拓扑确认页只加载当前焦点范围。</p></aside>`;
    const database = item.database;
    const source = sourceFor(item);
    const editing = Boolean(item.system_resource_id);
    const rac = isRacResource(item);
    const baseFields = `<div class="business-editor-grid"><label class="span-6"><span>${rac ? "RAC 名称" : "数据库名称"}</span><input name="${rac ? "rac_cluster_name" : "resource_name"}" value="${esc(rac ? item.rac_cluster_name : item.display_name)}"></label><label class="span-6"><span>技术责任组</span><select name="owner_group_id">${groupOptions(item.owner_group_id, "请选择")}</select></label><label class="span-12"><span>位置（可选）</span><input name="location_name" value="${esc(item.location_name)}" placeholder="机房、地域或可用区"></label></div>`;
    const databaseFields = database ? `<div class="business-editor-grid business-database-grid"><label class="span-3"><span>数据库引擎</span><select name="db_type" ${editing || rac ? "disabled" : ""}>${selectOptions([["ORACLE", "Oracle"], ["MYSQL", "MySQL"], ["POSTGRESQL", "PostgreSQL"]], database.db_type)}</select></label><label class="span-3"><span>数据库版本</span><select name="version_code">${selectOptions(versions[database.db_type], database.version_code)}</select></label><label class="span-3"><span>数据库环境</span><select name="db_environment">${selectOptions([["PROD", "生产"], ["STG", "预发布"], ["DEV", "开发"]], database.environment)}</select></label><label class="span-3"><span>部署形态</span><select name="deployment_type" ${database.db_type !== "ORACLE" || editing ? "disabled" : ""}>${selectOptions([["SINGLE_INSTANCE", "单实例"], ["RAC", "RAC（多节点）"]], database.deployment_type)}</select></label><label class="span-3"><span>数据库角色</span><select name="db_role">${selectOptions([["UNKNOWN", "待识别"], ["PRIMARY", "主库"], ["STANDBY", "备库"]], database.db_role)}</select></label></div>` : "";
    let connection = "";
    if (item.mode === "NEW" && database && !rac) {
      const discovery = usesDiscovery(item);
      connection = `<div class="business-editor-grid business-connection-grid"><label class="${database.db_type === "ORACLE" ? "span-3" : "span-4"}"><span>数据库地址</span><input name="host" value="${esc(database.host)}" placeholder="主机名或 IP"></label><label class="span-2"><span>端口</span><input name="port" value="${esc(database.port)}" type="number"></label><label class="${database.db_type === "ORACLE" ? "span-4" : "span-6"}"><span>${database.db_type === "ORACLE" ? "Service Name" : "Database"}</span><input name="database" value="${esc(database.database)}"></label>${database.db_type === "ORACLE" ? `<label class="span-3"><span>容器范围</span><select name="oracle_scope">${selectOptions([["NON_CDB", "非 CDB"], ["CDB", "CDB"], ["PDB", "PDB"]], database.oracle_scope)}</select>${database.oracle_scope === "PDB" ? `<input name="oracle_pdb_name" value="${esc(database.oracle_pdb_name)}" aria-label="PDB 名称" placeholder="PDB 名称">` : ""}</label>` : ""}<label class="span-6"><span>诊断用户名</span><input name="username" value="${esc(database.username)}" autocomplete="off"></label><label class="span-6"><span>诊断密码</span><input name="password" value="${esc(database.password)}" type="password" autocomplete="new-password"></label><label class="${discovery ? "span-4" : "span-6"}"><span>监控源（必选）</span><select name="source_id" required><option value="">请选择</option>${sources.map((row) => `<option value="${esc(row.source_id)}" ${row.source_id === database.source_id ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.source_type)}</option>`).join("")}</select></label>${!discovery ? `<label class="span-6"><span>监控定位键</span><input name="locator_key" value="${esc(database.locator_key)}" placeholder="与监控 Label 完全一致"></label>` : `<label class="span-4"><span>数据库候选</span><select name="candidate_ref" ${database.discovering ? "disabled" : ""}><option value="">请选择</option>${database.candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === database.candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label><label class="span-4"><span>承载主机候选</span><select name="host_candidate_ref" ${database.discovering ? "disabled" : ""}><option value="">请选择</option>${database.host_candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === database.host_candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label>${database.discovering ? `<p class="span-12 business-inline-note">正在从监控源发现数据库和承载主机…</p>` : ""}`}${source ? `<p class="span-12 business-inline-note">监控映射来自 ${esc(source.display_name)}；承载主机指标随数据库一次接入，无需重复登记服务器。</p>` : ""}</div>`;
    }
    if (item.mode === "NEW" && database && rac) {
      const discovery = usesDiscovery(item);
      const nodeCards = racMembers(item).map((node) => {
        const nodeDatabase = node.database;
        const nodeId = esc(node.draft_resource_id);
        const monitoring = discovery
          ? `<label class="span-6"><span>数据库监控 Target</span><select data-rac-node="${nodeId}" data-rac-field="candidate_ref" ${nodeDatabase.discovering ? "disabled" : ""}><option value="">请选择</option>${nodeDatabase.candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === nodeDatabase.candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label><label class="span-6"><span>主机监控 Target</span><select data-rac-node="${nodeId}" data-rac-field="host_candidate_ref" ${nodeDatabase.discovering ? "disabled" : ""}><option value="">请选择</option>${nodeDatabase.host_candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === nodeDatabase.host_candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label>`
          : `<label class="span-12"><span>监控定位键</span><input data-rac-node="${nodeId}" data-rac-field="locator_key" value="${esc(nodeDatabase.locator_key)}"></label>`;
        return `<article class="business-rac-node-card"><header><div><strong>RAC 节点 ${esc(nodeDatabase.rac_node_number)}</strong><small>${esc(node.display_name || "待命名节点")}</small></div>${racMembers(item).length > 2 ? `<button type="button" class="business-danger" data-remove-rac-node="${nodeId}">移除节点</button>` : ""}</header><div class="business-editor-grid"><label class="span-2"><span>节点编号</span><input type="number" min="1" max="1024" required data-rac-node="${nodeId}" data-rac-field="rac_node_number" value="${esc(nodeDatabase.rac_node_number)}"></label><label class="span-4"><span>Host</span><input data-rac-node="${nodeId}" data-rac-field="host" value="${esc(nodeDatabase.host)}" placeholder="节点主机名或 IP"></label><label class="span-6"><span>Service Name</span><input data-rac-node="${nodeId}" data-rac-field="database" value="${esc(nodeDatabase.database)}"></label>${monitoring}</div></article>`;
      }).join("");
      connection = `<div class="business-editor-grid business-connection-grid business-rac-editor"><p class="span-12 business-rac-guidance">公共凭据与监控源只填写一次；每个节点分别配置 Host、Service Name、数据库监控 Target 和主机监控 Target。</p><label class="span-3"><span>端口</span><input name="port" value="${esc(database.port)}" type="number"></label><label class="span-3"><span>容器范围</span><select name="oracle_scope">${selectOptions([["NON_CDB", "非 CDB"], ["CDB", "CDB"], ["PDB", "PDB"]], database.oracle_scope)}</select></label>${database.oracle_scope === "PDB" ? `<label class="span-6"><span>PDB 名称</span><input name="oracle_pdb_name" value="${esc(database.oracle_pdb_name)}"></label>` : ""}<label class="span-6"><span>诊断用户名（所有节点共用）</span><input name="username" value="${esc(database.username)}" autocomplete="off"></label><label class="span-6"><span>诊断密码（所有节点共用）</span><input name="password" value="${esc(database.password)}" type="password" autocomplete="new-password"></label><label class="span-12"><span>监控源（所有节点共用）</span><select name="source_id" required><option value="">请选择</option>${sources.map((row) => `<option value="${esc(row.source_id)}" ${row.source_id === database.source_id ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.source_type)}</option>`).join("")}</select></label>${database.discovering ? `<p class="span-12 business-inline-note">正在从监控源发现数据库和承载主机…</p>` : ""}<section class="span-12 business-rac-node-list">${nodeCards}<button id="add-rac-node" type="button">＋ 增加 RAC 节点</button></section>${source ? `<p class="span-12 business-inline-note">监控映射来自 ${esc(source.display_name)}；每个 RAC 节点分别绑定数据库与承载主机 Target。</p>` : ""}</div>`;
    }
    return `<form id="resource-editor" class="business-resource-editor"><header><div><span>${esc(layerName(item.architecture_layer_key))} · ${esc(typeName(item.resource_type_key))}</span><h2>${esc(item.display_name || "配置数据库")}</h2><p>${editing ? "编辑资源配置" : "接入资源并验证连接与监控映射"}</p></div><button id="close-resource-drawer" type="button" aria-label="关闭资源配置">关闭</button></header><div class="business-resource-drawer-body">${baseFields}${databaseFields}${connection}</div><footer><button id="remove-resource" class="business-danger" type="button">移除资源</button><p>${editing ? "修改将在保存业务系统时提交" : item.saved ? "连接与监控映射已验证" : "连接验证通过后才可进入下一步"}</p><button id="finish-resource-drawer" type="button">完成</button>${!editing ? `<button id="validate-resource" class="primary" type="submit" ${saving ? "disabled" : ""}>验证数据库</button>` : ""}</footer></form>`;
  }

  function renderComposition() {
    const visible = resources.filter((item) => item.architecture_layer_key === activeLayer);
    const cards = visible.map((item) => `<button type="button" data-resource="${esc(item.draft_resource_id)}" class="business-resource-card"><header><span class="business-resource-icon">${item.resource_type_key === "DATABASE" ? "DB" : "R"}</span><em class="${item.saved ? "ready" : ""}">${item.saved ? "已就绪" : "待验证"}</em></header><div><small>${esc(typeName(item.resource_type_key))}</small><h3>${esc(item.display_name || "未命名数据库")}</h3><p>${item.database ? `${esc(item.database.db_type)} ${esc(item.database.version_code)} · ${item.database.deployment_type === "RAC" ? `RAC 节点 ${esc(item.database.rac_node_number)}` : "单实例"} · ${item.database.db_role === "PRIMARY" ? "主库" : item.database.db_role === "STANDBY" ? "备库" : "角色待识别"}` : ""}</p></div><dl><div><dt>技术责任组</dt><dd>${esc(groupName(item.owner_group_id))}</dd></div><div><dt>位置</dt><dd>${esc(item.location_name || "未填写")}</dd></div></dl><footer><span>${item.system_resource_id ? "已接入资源" : "新接入资源"}</span><strong>编辑 →</strong></footer></button>`).join("");
    const content = activeLayer === "DATA" ? `<div class="business-resource-card-grid">${cards}<button id="add-resource-card" class="business-resource-add-card" type="button" aria-label="＋ 添加资源"><span>＋</span><strong>添加资源</strong><small>接入当前架构层</small></button></div>` : `<div class="business-layer-coming-soon"><span aria-hidden="true">◇</span><h3>功能未完成</h3><p>敬请期待</p></div>`;
    panel.innerHTML = `<section class="business-composition-layout"><aside class="business-layer-nav" role="tablist" aria-label="系统构成"><header><p class="business-eyebrow">STEP 2</p><h2>系统构成</h2></header>${layers.map((layer) => `<button type="button" role="tab" aria-selected="${activeLayer === layer[0]}" data-layer="${layer[0]}" class="${activeLayer === layer[0] ? "active" : ""}"><span><strong>${layer[1]}</strong><small>${layer[2]}</small></span><em>${resources.filter((item) => item.architecture_layer_key === layer[0]).length}</em></button>`).join("")}</aside><section class="business-resource-panel" role="tabpanel"><header><div><p class="business-eyebrow">${esc(activeLayer)}</p><h2>${esc(layerName(activeLayer))}</h2><p>${activeLayer === "DATA" ? `已配置 ${visible.length} 个运维资源` : "功能未完成，敬请期待"}</p></div></header>${content}</section></section>`;
    renderResourceDrawer();
    bindComposition();
  }

  function renderResourceDrawer() {
    const drawer = document.getElementById("business-resource-drawer");
    if (!resourceDrawerOpen) {
      if (drawer.open) drawer.close();
      drawer.innerHTML = "";
      return;
    }
    drawer.innerHTML = resourceDrawerMode === "CATALOG" ? catalogHtml() : resourceEditorHtml(currentResource());
    if (!drawer.open) drawer.showModal();
  }

  function bindComposition() {
    panel.querySelectorAll("[data-layer]").forEach((button) => button.onclick = () => { closeResourceDrawer(); activeLayer = button.dataset.layer; activeResourceId = ""; renderComposition(); });
    panel.querySelectorAll("[data-resource]").forEach((button) => button.onclick = () => openResourceEditor(button.dataset.resource));
    const addResource = document.getElementById("add-resource-card");
    if (addResource) addResource.onclick = openResourceCatalog;
    bindResourceDrawer();
  }

  function bindResourceDrawer() {
    const drawer = document.getElementById("business-resource-drawer");
    drawer.onclose = () => { resourceDrawerOpen = false; };
    const close = document.getElementById("close-resource-drawer");
    if (close) close.onclick = closeResourceDrawer;
    const cancel = document.getElementById("cancel-resource-drawer");
    if (cancel) cancel.onclick = closeResourceDrawer;
    const finish = document.getElementById("finish-resource-drawer");
    if (finish) finish.onclick = closeResourceDrawer;
    const catalog = document.getElementById("catalog-type");
    if (catalog) catalog.onchange = () => { catalogType = catalog.value; renderResourceDrawer(); bindResourceDrawer(); };
    const confirmAdd = document.getElementById("confirm-add-resource");
    if (confirmAdd) confirmAdd.onclick = () => {
      const selected = resourceCatalog.find((item) => item[0] === catalogType);
      if (!selected?.[3]) { shell.toast(`${selected?.[1] || "该资源类型"}将在后续版本开放接入。`); return; }
      const item = emptyDatabaseResource();
      item.architecture_layer_key = activeLayer;
      resources.push(item); activeResourceId = item.draft_resource_id; resourceDrawerMode = "EDITOR"; mark(); renderComposition();
    };
    const item = currentResource();
    if (resourceDrawerMode !== "EDITOR" || !item) return;
    const editor = document.getElementById("resource-editor");
    if (!editor) return;
    editor.querySelectorAll("[name]").forEach((input) => input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => updateResourceField(item, input)));
    editor.querySelectorAll("[data-rac-node][data-rac-field]").forEach((input) => input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      const node = resources.find((row) => row.draft_resource_id === input.dataset.racNode);
      if (!node?.database) return;
      node.database[input.dataset.racField] = input.dataset.racField === "rac_node_number" ? Number(input.value) : input.value;
      node.saved = false;
      if (input.dataset.racField === "rac_node_number") syncRacNames(item);
      else mark();
    }));
    const addNode = document.getElementById("add-rac-node");
    if (addNode) addNode.onclick = () => { addRacNode(item); renderComposition(); };
    editor.querySelectorAll("[data-remove-rac-node]").forEach((button) => button.onclick = () => {
      const node = resources.find((row) => row.draft_resource_id === button.dataset.removeRacNode);
      if (!node) return;
      removeRacNode(item, node);
      renderComposition();
    });
    document.getElementById("remove-resource").onclick = () => {
      const removed = new Set(racMembers(item).map((member) => member.draft_resource_id));
      resources = resources.filter((row) => !removed.has(row.draft_resource_id));
      relations = relations.filter((relation) => relation.draft_relation_id !== item.rac_group_id && !relation.participants.some((participant) => removed.has(participant.draft_resource_id)));
      activeResourceId = resources.find((row) => row.architecture_layer_key === activeLayer)?.draft_resource_id || resources[0]?.draft_resource_id || "";
      closeResourceDrawer(); mark(); renderComposition();
    };
    editor.onsubmit = (event) => { event.preventDefault(); void validateResource(item); };
  }

  function updateResourceField(item, input) {
    const name = input.name;
    if (name === "rac_cluster_name") {
      item.rac_cluster_name = input.value;
      syncRacNames(item);
      return;
    }
    if (["resource_name", "owner_group_id", "location_name"].includes(name)) {
      const key = { resource_name: "display_name" }[name] || name;
      item[key] = input.value;
      if (isRacResource(item) && name !== "resource_name") {
        syncRacResourceField(item, key);
        return;
      }
      if (name === "resource_name" && !item.system_resource_id) item.saved = false;
      mark();
      return;
    }
    if (!item.database) return;
    const key = name === "db_environment" ? "environment" : name;
    item.database[key] = ["port", "rac_node_number"].includes(name) ? Number(input.value) : input.value;
    if (name === "db_type") {
      item.database.version_code = versions[item.database.db_type][0];
      item.database.port = item.database.db_type === "ORACLE" ? 1521 : item.database.db_type === "MYSQL" ? 3306 : 5432;
      if (item.database.db_type !== "ORACLE") {
        item.database.deployment_type = "SINGLE_INSTANCE";
        item.database.rac_node_number = null;
      }
      item.saved = false; mark(); renderComposition();
      if (item.database.source_id && usesDiscovery(item)) void discover(item);
      return;
    }
    if (name === "deployment_type") {
      if (item.database.deployment_type === "RAC") ensureRacGroup(item);
      else {
        const groupId = item.rac_group_id;
        const nameBefore = item.rac_cluster_name || item.display_name.replace(/\s*·\s*节点\s*\d+$/, "");
        if (groupId) {
          resources = resources.filter((row) => row === item || row.rac_group_id !== groupId);
          relations = relations.filter((relation) => relation.draft_relation_id !== groupId);
        }
        item.rac_group_id = undefined;
        item.rac_cluster_name = undefined;
        item.display_name = nameBefore;
        item.database.rac_node_number = null;
      }
      item.saved = Boolean(item.system_resource_id); mark(); renderComposition(); return;
    }
    if (name === "source_id") {
      racMembers(item).forEach((member) => {
        if (!member.database) return;
        member.database.source_id = item.database.source_id;
        member.database.discovery_revision += 1;
        Object.assign(member.database, { locator_key: "", candidate_ref: "", host_candidate_ref: "", candidates: [], host_candidates: [] });
        member.saved = Boolean(member.system_resource_id);
      });
      mark(); renderComposition();
      if (item.database.source_id && usesDiscovery(item)) void discover(item);
      return;
    }
    if (isRacResource(item)) syncRacDatabaseField(item, key);
    if (name === "oracle_scope") { item.saved = false; mark(); renderComposition(); return; }
    item.saved = Boolean(item.system_resource_id);
    if (["host", "port", "database", "oracle_pdb_name", "username", "password", "locator_key", "candidate_ref", "host_candidate_ref"].includes(name)) item.saved = false;
    mark();
  }

  async function discover(item) {
    if (!item.database?.source_id) { shell.toast("请先选择监控源。"); return; }
    const sourceId = item.database.source_id;
    const dbType = item.database.db_type;
    const revision = ++item.database.discovery_revision;
    racMembers(item).forEach((member) => { if (member.database) member.database.discovering = true; });
    renderComposition();
    try {
      const found = await request(`/diagnostic-sources/${encodeURIComponent(sourceId)}/instance-discoveries`, { method: "POST", body: JSON.stringify({ db_types: [dbType], page_size: 100 }) });
      if (revision !== item.database.discovery_revision || sourceId !== item.database.source_id || dbType !== item.database.db_type) return;
      const candidates = (found.items || []).filter((row) => row.mapping_status !== "MAPPED");
      const hostCandidates = found.host_items || [];
      racMembers(item).forEach((member) => {
        if (!member.database) return;
        member.database.candidates = [...candidates];
        member.database.host_candidates = [...hostCandidates];
      });
      shell.toast(`发现 ${candidates.length} 个数据库候选、${hostCandidates.length} 个主机候选。`);
    } catch (error) { shell.toast(error.message); }
    finally {
      if (revision === item.database.discovery_revision) {
        racMembers(item).forEach((member) => { if (member.database) member.database.discovering = false; });
        renderComposition();
      }
    }
  }

  function targetPayload(item) {
    const database = item.database;
    const oracle = database.db_type === "ORACLE";
    return { display_name: item.display_name.trim(), db_type: database.db_type, version_code: database.version_code, environment: database.environment, deployment_type: database.deployment_type, rac_node_number: database.rac_node_number, db_role: database.db_role, importance_level: Number(system.importance_level), responsibility_group_id: item.owner_group_id, readonly_connection_enabled: true, controlled_change_enabled: false, endpoint: { host: database.host.trim(), port: Number(database.port), service: oracle ? database.database.trim() : null, database: oracle ? null : database.database.trim(), tls_enabled: !oracle }, oracle_container_scope: oracle ? database.oracle_scope : null, oracle_pdb_name: oracle && database.oracle_scope === "PDB" ? database.oracle_pdb_name.trim() : null, diagnostic_credential: { username: database.username, password: database.password }, capabilities: {} };
  }
  function monitoringPayload(item) {
    const database = item.database;
    const source = sourceFor(item);
    if (usesDiscovery(item)) return { source_id: database.source_id, candidate_ref: database.candidate_ref, host_candidate_ref: database.host_candidate_ref || null };
    return { source_id: database.source_id, source_binding: { source_id: database.source_id, source_locator_key: database.locator_key.trim(), source_locator: source.source_type === "LOKI" ? { labels: { target_key: database.locator_key.trim() } } : { target_key: database.locator_key.trim() } } };
  }

  async function validateResource(item) {
    const members = isRacResource(item) ? racMembers(item) : [item];
    const database = item.database;
    const source = sourceFor(item);
    if (!database || members.some((member) => !member.database || !member.display_name.trim() || !member.owner_group_id || !member.database.host.trim() || !member.database.database.trim()) || !database.username || !database.password || !source) { shell.toast("请完整填写数据库名称、责任组、连接地址、凭据和监控映射。"); return; }
    if (isRacResource(item) && members.length < 2) { shell.toast("Oracle RAC 至少需要两个节点。"); return; }
    if (database.db_type === "ORACLE" && database.oracle_scope === "PDB" && !database.oracle_pdb_name.trim()) { shell.toast("PDB 接入必须填写 PDB 名称。"); return; }
    if (members.some((member) => member.database?.deployment_type === "RAC" && (!member.database.rac_node_number || member.database.rac_node_number < 1))) { shell.toast("每个 RAC 节点都必须填写有效的节点编号。"); return; }
    if (members.some((member) => usesDiscovery(member) ? !member.database?.candidate_ref || (source.source_type === "PROMETHEUS" && !member.database?.host_candidate_ref) : !member.database?.locator_key.trim())) { shell.toast("请完成数据库与监控源的定位映射。"); return; }
    if (isRacResource(item)) {
      const uniqueFields = [
        ["节点编号", members.map((member) => member.database?.rac_node_number)],
        ["Host", members.map((member) => member.database?.host.trim())],
        ["Service Name", members.map((member) => member.database?.database.trim())],
        ["数据库监控 Target", members.map((member) => member.database?.candidate_ref || member.database?.locator_key.trim())],
      ];
      if (source.source_type === "PROMETHEUS") uniqueFields.push(["主机监控 Target", members.map((member) => member.database?.host_candidate_ref)]);
      const duplicate = uniqueFields.find((entry) => new Set(entry[1]).size !== entry[1].length);
      if (duplicate) { shell.toast(`RAC 各节点的${duplicate[0]}不能重复。`); return; }
    }
    saving = true; updateFooter();
    try {
      for (const member of members) {
        const target = targetPayload(member);
        const response = await request("/targets/test-connection", { method: "POST", body: JSON.stringify({ db_type: target.db_type, version_code: target.version_code, endpoint: target.endpoint, oracle_container_scope: target.oracle_container_scope, oracle_pdb_name: target.oracle_pdb_name, diagnostic_credential: target.diagnostic_credential }) });
        if (!response.ok) throw new Error(`${member.display_name}：${response.error_code || "连接验证失败"}`);
      }
      members.forEach((member) => { member.saved = true; });
      mark(); shell.toast(isRacResource(item) ? `${members.length} 个 RAC 节点连接验证通过；正式提交前不会创建资源。` : "数据库连接验证通过；正式提交前不会创建资源。");
    } catch (error) { shell.toast(error.message); }
    finally { saving = false; renderComposition(); updateFooter(); }
  }

  function topologyBoardHtml() {
    const visibleLayers = layers.map((layer) => [layer, resources.filter((item) => item.architecture_layer_key === layer[0])]).filter((entry) => entry[1].length);
    const resourceName = (draftId) => resources.find((item) => item.draft_resource_id === draftId)?.display_name || "未选择资源";
    const lanes = visibleLayers.length ? visibleLayers.map(([layer, items]) => `<section class="business-topology-lane"><header><strong>${esc(layer[1])}</strong><span>${items.length}</span></header><div>${items.map((item) => `<button type="button" data-topology-resource="${esc(item.draft_resource_id)}" class="${activeResourceId === item.draft_resource_id ? "active" : ""} ${item.saved ? "" : "pending"}"><span class="business-node-kind">${item.resource_type_key === "DATABASE" ? "DB" : "RES"}</span><span><strong>${esc(item.display_name || "未命名资源")}</strong><small>${esc(item.database ? `${item.database.db_type} ${item.database.version_code} · ${item.database.deployment_type === "RAC" ? `RAC 节点 ${item.database.rac_node_number}` : "单实例"} · ${item.database.db_role}` : item.resource_type_key)}</small></span></button>`).join("")}</div></section>`).join("") : `<div class="business-topology-empty"><h3>拓扑将在接入资源后自动生成</h3><p>先返回“系统构成”添加数据库或其他运维资源。</p></div>`;
    const rail = relations.length ? `<section class="business-relation-rail"><header><strong>资源关系</strong><span>关系是拓扑事实，不创建隐藏分组</span></header><div>${relations.map((relation) => `<button type="button" data-topology-relation="${esc(relation.draft_relation_id)}" class="${activeRelationId === relation.draft_relation_id ? "active" : ""}"><strong>${esc(relation.display_name || relationName(relation.relation_type_key))}</strong><span>${relation.participants.map((participant) => `<em>${esc(resourceName(participant.draft_resource_id))} <small>${esc(roleLabels[participant.participant_role] || participant.participant_role)}</small></em>`).join("")}</span></button>`).join("")}</div></section>` : "";
    return `<section class="business-topology-board"><header><span>BUSINESS SYSTEM</span><strong>${esc(system.display_name || "未命名业务系统")}</strong><small>${resources.length} 个资源 · ${relations.length} 个关系</small></header><div class="business-topology-lanes">${lanes}</div>${rail}</section>`;
  }

  function relationEditorHtml(relation) {
    const roles = roleOptions[relation.relation_type_key] || ["MEMBER"];
    const managed = isManagedRacRelation(relation);
    return `<header><div><p class="business-eyebrow">RELATION</p><h3>${esc(relation.display_name || relationName(relation.relation_type_key))}</h3></div>${managed ? "" : `<button id="remove-relation" class="business-danger" type="button">删除</button>`}</header><label><span>关系类型</span><select name="relation_type" ${managed ? "disabled" : ""}>${selectOptions(relationTypes, relation.relation_type_key)}</select></label><label><span>关系名称（可选）</span><input name="relation_name" value="${esc(relation.display_name)}" ${managed ? "disabled" : ""}></label><label><span>关系责任组</span><select name="relation_group">${groupOptions(relation.responsibility_group_id, "继承参与资源")}</select></label><div class="business-participant-list">${relation.participants.map((participant, index) => `<div><select data-participant-resource="${index}" ${managed ? "disabled" : ""}>${resources.map((item) => `<option value="${esc(item.draft_resource_id)}" ${item.draft_resource_id === participant.draft_resource_id ? "selected" : ""}>${esc(item.display_name || "未命名资源")}</option>`).join("")}</select><select data-participant-role="${index}" ${managed ? "disabled" : ""}>${selectOptions(roles, participant.participant_role, roleLabels)}</select>${managed ? "" : `<button type="button" data-remove-participant="${index}" aria-label="移除参与者">×</button>`}</div>`).join("")}${managed ? "" : `<button id="add-participant" type="button" ${relation.participants.length >= resources.length ? "disabled" : ""}>添加参与资源</button>`}</div>`;
  }
  function resourceSummaryHtml(item) {
    if (!item) return `<div class="business-topology-empty"><h3>选择节点或关系</h3><p>点击画布中的资源或关系，在这里查看和编辑属性。</p></div>`;
    return `<header><div><p class="business-eyebrow">RESOURCE</p><h3>${esc(item.display_name || "未命名资源")}</h3></div></header><dl><div><dt>资源类型</dt><dd>${esc(typeName(item.resource_type_key))}</dd></div><div><dt>架构层</dt><dd>${esc(layerName(item.architecture_layer_key))}</dd></div><div><dt>技术责任组</dt><dd>${esc(groupName(item.owner_group_id))}</dd></div>${item.database ? `<div><dt>数据库</dt><dd>${esc(`${item.database.db_type} ${item.database.version_code} · ${item.database.db_role}`)}</dd></div>` : ""}</dl><button id="open-resource" type="button">打开完整配置</button>`;
  }

  function renderTopology() {
    const relation = currentRelation();
    panel.innerHTML = `<section class="business-topology-step"><header><div><p class="business-eyebrow">STEP 3</p><h2>拓扑确认</h2><p>系统根据资源和关系推导单实例、RAC、Data Guard 或复制架构。</p></div><button id="add-relation" class="primary" type="button">＋ 添加关系</button></header><div class="business-topology-layout">${topologyBoardHtml()}<aside class="business-topology-property">${relation ? relationEditorHtml(relation) : resourceSummaryHtml(currentResource())}</aside></div></section>`;
    bindTopology();
  }
  function defaultParticipants(type) {
    const roles = roleOptions[type] || ["SOURCE", "TARGET"];
    return resources.slice(0, 2).map((item, index) => ({ draft_resource_id: item.draft_resource_id, participant_role: roles[Math.min(index, roles.length - 1)], display_order: index }));
  }

  function bindTopology() {
    document.getElementById("add-relation").onclick = () => {
      if (resources.length < 2) { shell.toast("至少有两个资源后才能建立关系。"); return; }
      const relation = { draft_relation_id: uuid7(), relation_type_key: "ORACLE_RAC", display_name: "", responsibility_group_id: "", participants: defaultParticipants("ORACLE_RAC"), attributes: {} };
      relations.push(relation); activeRelationId = relation.draft_relation_id; activeResourceId = ""; mark(); renderTopology();
    };
    panel.querySelectorAll("[data-topology-resource]").forEach((button) => button.onclick = () => { activeResourceId = button.dataset.topologyResource; activeRelationId = ""; renderTopology(); });
    panel.querySelectorAll("[data-topology-relation]").forEach((button) => button.onclick = () => { activeRelationId = button.dataset.topologyRelation; activeResourceId = ""; renderTopology(); });
    const relation = currentRelation();
    if (relation) {
      const type = panel.querySelector('[name="relation_type"]');
      type.onchange = () => { relation.relation_type_key = type.value; relation.participants = defaultParticipants(type.value); mark(); renderTopology(); };
      const name = panel.querySelector('[name="relation_name"]'); name.oninput = () => { relation.display_name = name.value; mark(); };
      const group = panel.querySelector('[name="relation_group"]'); group.onchange = () => { relation.responsibility_group_id = group.value; mark(); };
      panel.querySelectorAll("[data-participant-resource]").forEach((select) => select.onchange = () => { relation.participants[Number(select.dataset.participantResource)].draft_resource_id = select.value; mark(); });
      panel.querySelectorAll("[data-participant-role]").forEach((select) => select.onchange = () => { relation.participants[Number(select.dataset.participantRole)].participant_role = select.value; mark(); });
      panel.querySelectorAll("[data-remove-participant]").forEach((button) => button.onclick = () => { relation.participants.splice(Number(button.dataset.removeParticipant), 1); mark(); renderTopology(); });
      const addParticipant = document.getElementById("add-participant");
      if (addParticipant) addParticipant.onclick = () => {
        const candidate = resources.find((resource) => !relation.participants.some((participant) => participant.draft_resource_id === resource.draft_resource_id));
        if (!candidate) return;
        relation.participants.push({ draft_resource_id: candidate.draft_resource_id, participant_role: (roleOptions[relation.relation_type_key] || ["MEMBER"])[0], display_order: relation.participants.length }); mark(); renderTopology();
      };
      const removeRelation = document.getElementById("remove-relation");
      if (removeRelation) removeRelation.onclick = () => { relations = relations.filter((row) => row !== relation); activeRelationId = ""; mark(); renderTopology(); };
    }
    const open = document.getElementById("open-resource");
    if (open) open.onclick = () => { const item = currentResource(); step = 1; activeLayer = item.architecture_layer_key; resourceDrawerMode = "EDITOR"; resourceDrawerOpen = true; render(); };
  }

  function validateStep() {
    if (step === 0 && (!system.display_name.trim() || !system.responsibility_group_id)) { shell.toast("请填写业务系统名称并选择业务责任组。"); return false; }
    if (step === 1 && (!resources.length || resources.some((item) => !item.display_name.trim() || !item.owner_group_id || !item.saved))) { shell.toast("至少接入一个资源，并完成所有数据库的连接验证。"); return false; }
    if (step === 2 && relations.some((item) => item.participants.length < 2 || new Set(item.participants.map((participant) => participant.draft_resource_id)).size !== item.participants.length)) { shell.toast("每个关系至少需要两个不重复的参与资源。"); return false; }
    return true;
  }
  function systemPayload() { return { display_name: system.display_name.trim(), environment: system.environment, importance_level: Number(system.importance_level), responsibility_group_id: system.responsibility_group_id, description: system.description.trim() || null }; }
  function resourcePayload(item) {
    return { draft_resource_id: item.draft_resource_id, mode: "NEW", resource_type_key: item.resource_type_key, display_name: item.display_name.trim(), identity_kind: "STABLE", owner_group_id: item.owner_group_id, architecture_layer_key: item.architecture_layer_key, responsibility_group_id: null, location_name: item.location_name.trim() || null, display_order: item.display_order, attributes: {}, database: { target: targetPayload(item), monitoring: monitoringPayload(item) } };
  }
  function relationPayload(item, systemIdByDraft = null) {
    return { relation_type_key: item.relation_type_key, display_name: item.display_name.trim() || null, responsibility_group_id: item.responsibility_group_id || null, origin: "MANUAL", confirmation_status: "CONFIRMED", attributes: item.attributes || {}, participants: item.participants.map((participant) => systemIdByDraft ? { system_resource_id: systemIdByDraft.get(participant.draft_resource_id), participant_role: participant.participant_role, display_order: participant.display_order } : participant) };
  }
  async function createSystem() {
    return request("/business-systems", { method: "POST", headers: { "Idempotency-Key": uuid7() }, body: JSON.stringify({ system: systemPayload(), resources: resources.map(resourcePayload), relations: relations.map((item) => ({ draft_relation_id: item.draft_relation_id, ...relationPayload(item) })) }) });
  }

  async function updateSystem() {
    await request(`/business-systems/${encodeURIComponent(systemId)}`, { method: "PATCH", headers: { "If-Match": `"rv-${detail.row_version}"` }, body: JSON.stringify(systemPayload()) });
    const originalResources = new Map(detail.resources.map((item) => [item.system_resource_id, item]));
    const retainedSystemIds = new Set(resources.map((item) => item.system_resource_id).filter(Boolean));
    const retainedRelationIds = new Set(relations.map((item) => item.relation_id).filter(Boolean));
    for (const original of detail.relations) if (!retainedRelationIds.has(original.relation_id)) await request(`/business-systems/${encodeURIComponent(systemId)}/relations/${encodeURIComponent(original.relation_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${original.row_version}"` } });
    let current = await request(`/business-systems/${encodeURIComponent(systemId)}`);
    for (const item of resources) {
      if (item.system_resource_id) {
        await request(`/managed-resources/${encodeURIComponent(item.resource_id)}`, { method: "PATCH", headers: { "If-Match": `"rv-${item.resource_row_version}"` }, body: JSON.stringify({ display_name: item.display_name.trim(), owner_group_id: item.owner_group_id, location_name: item.location_name.trim() || null, database: item.database ? { version_code: item.database.version_code, environment: item.database.environment, deployment_type: item.database.deployment_type, rac_node_number: item.database.rac_node_number, db_role: item.database.db_role } : null }) });
        current = await request(`/business-systems/${encodeURIComponent(systemId)}/resources/${encodeURIComponent(item.system_resource_id)}`, { method: "PATCH", headers: { "If-Match": `"rv-${item.system_row_version}"` }, body: JSON.stringify({ architecture_layer_key: item.architecture_layer_key, responsibility_group_id: null, display_order: item.display_order }) });
      } else {
        const before = new Set(current.resources.map((row) => row.system_resource_id));
        current = await request(`/business-systems/${encodeURIComponent(systemId)}/resources`, { method: "POST", headers: { "Idempotency-Key": uuid7() }, body: JSON.stringify(resourcePayload(item)) });
        const added = current.resources.find((row) => !before.has(row.system_resource_id));
        if (!added) throw new Error(`无法确认新增资源“${item.display_name}”的服务端标识`);
        Object.assign(item, { system_resource_id: added.system_resource_id, resource_id: added.resource.resource_id, system_row_version: added.row_version, resource_row_version: added.resource.row_version });
      }
    }
    const systemIdByDraft = new Map(resources.map((item) => [item.draft_resource_id, item.system_resource_id]));
    for (const item of relations) {
      if (item.relation_id) current = await request(`/business-systems/${encodeURIComponent(systemId)}/relations/${encodeURIComponent(item.relation_id)}`, { method: "PATCH", headers: { "If-Match": `"rv-${item.row_version}"` }, body: JSON.stringify(relationPayload(item, systemIdByDraft)) });
      else current = await request(`/business-systems/${encodeURIComponent(systemId)}/relations`, { method: "POST", headers: { "Idempotency-Key": uuid7() }, body: JSON.stringify(relationPayload(item, systemIdByDraft)) });
    }
    for (const original of originalResources.values()) if (!retainedSystemIds.has(original.system_resource_id)) await request(`/business-systems/${encodeURIComponent(systemId)}/resources/${encodeURIComponent(original.system_resource_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${original.row_version}"` } });
    return request(`/business-systems/${encodeURIComponent(systemId)}`);
  }

  async function commit() {
    if (!validateStep()) return;
    saving = true; updateFooter();
    try {
      const saved = creating ? await createSystem() : await updateSystem();
      dirty = false; sessionStorage.removeItem(draftKey); shell.toast(creating ? "业务系统与资源拓扑已创建。" : "业务系统与资源拓扑已更新。");
      if (creating) location.replace(`./business-system-workspace.html?id=${encodeURIComponent(saved.business_system_id)}`);
      else { hydrate(saved); notifyValidationIssues(saved); render(); }
    } catch (error) { shell.toast(error.message); }
    finally { saving = false; updateFooter(); }
  }
  function saveDraft() {
    const safeResources = resources.map((item) => ({ ...item, database: item.database ? { ...item.database, username: "", password: "", candidates: [], host_candidates: [], discovering: false } : undefined, saved: Boolean(item.system_resource_id) }));
    sessionStorage.setItem(draftKey, JSON.stringify({ system, resources: safeResources, relations, step, furthest }));
    dirty = false; updateFooter(); shell.toast("草稿已保存在当前浏览器会话；数据库凭据不会保存。");
  }

  function hydrate(value) {
    detail = value;
    system = { display_name: value.display_name, environment: value.environment, importance_level: value.importance_level, responsibility_group_id: value.responsibility_group_id, description: value.description || "" };
    const draftBySystem = new Map();
    resources = value.resources.filter((item) => item.status === "ACTIVE").map((item, index) => {
      const draftId = uuid7(); draftBySystem.set(item.system_resource_id, draftId);
      const database = item.resource.database;
      return { draft_resource_id: draftId, mode: "EXISTING", resource_id: item.resource.resource_id, system_resource_id: item.system_resource_id, resource_row_version: item.resource.row_version, system_row_version: item.row_version, resource_type_key: item.resource.resource_type_key, display_name: item.resource.display_name, owner_group_id: item.resource.owner_group_id, architecture_layer_key: item.architecture_layer_key, responsibility_group_id: item.responsibility_group_id || "", location_name: item.resource.location_name || "", display_order: item.display_order ?? index, saved: true, database: database ? { db_type: database.db_type, version_code: database.version_code, environment: database.environment, deployment_type: database.deployment_type, rac_node_number: database.rac_node_number ?? null, db_role: database.db_role, host: "", port: 0, database: "", oracle_scope: "NON_CDB", oracle_pdb_name: "", username: "", password: "", source_id: "", locator_key: "", candidate_ref: "", host_candidate_ref: "", candidates: [], host_candidates: [], discovering: false, discovery_revision: 0 } : undefined };
    });
    relations = value.relations.map((item) => ({ draft_relation_id: uuid7(), relation_id: item.relation_id, row_version: item.row_version, relation_type_key: item.relation_type_key, display_name: item.display_name || "", responsibility_group_id: item.responsibility_group_id || "", attributes: item.attributes || {}, participants: item.participants.map((participant) => ({ draft_resource_id: draftBySystem.get(participant.system_resource_id) || "", participant_role: participant.participant_role, display_order: participant.display_order })) }));
    relations.filter((item) => item.relation_type_key === "ORACLE_RAC").forEach((relation) => {
      const memberIds = new Set(relation.participants.map((participant) => participant.draft_resource_id));
      const members = resources.filter((item) => memberIds.has(item.draft_resource_id));
      const clusterName = relation.display_name || members[0]?.display_name.replace(/\s*·\s*节点\s*\d+$/, "") || "";
      members.forEach((member) => {
        member.rac_group_id = relation.draft_relation_id;
        member.rac_cluster_name = clusterName;
      });
    });
    activeResourceId = resources[0]?.draft_resource_id || "";
    activeLayer = resources[0]?.architecture_layer_key || "DATA";
    document.getElementById("workspace-code").textContent = value.system_code || "业务系统";
    document.getElementById("workspace-title").textContent = value.display_name;
  }

  function updateFooter() {
    document.getElementById("step-copy").textContent = `${dirty ? "有未保存修改 · " : ""}第 ${step + 1} 步，共 3 步`;
    document.getElementById("previous").hidden = step === 0;
    document.getElementById("next").hidden = step === 2;
    document.getElementById("commit").hidden = step !== 2;
    document.getElementById("commit").disabled = saving;
    document.getElementById("commit").textContent = saving ? "正在提交…" : creating ? "确认并创建" : "保存业务系统";
  }
  function render() {
    document.querySelectorAll("#wizard-steps [data-step]").forEach((button, index) => { button.classList.toggle("active", index === step); button.classList.toggle("done", index < step); button.disabled = index > furthest; });
    if (loading) panel.innerHTML = `<div class="ops-empty">正在加载业务系统工作区…</div>`;
    else if (step === 0) renderBusiness();
    else if (step === 1) renderComposition();
    else renderTopology();
    updateFooter();
  }

  async function load() {
    loading = true; render();
    try {
      const calls = [request("/responsibility-groups"), request("/diagnostic-sources?limit=200")];
      if (!creating) calls.push(request(`/business-systems/${encodeURIComponent(systemId)}`));
      const [groupPage, sourcePage, loadedDetail] = await Promise.all(calls);
      groups = (groupPage.items || []).filter((item) => item.status === "ACTIVE");
      sources = (sourcePage.items || []).filter((item) => item.status === "ENABLED");
      if (loadedDetail) { hydrate(loadedDetail); notifyValidationIssues(loadedDetail); }
      const saved = sessionStorage.getItem(draftKey);
      if (saved) {
        const draft = JSON.parse(saved);
        system = { ...system, ...draft.system }; resources = draft.resources || resources; relations = draft.relations || relations;
        step = Number(draft.step || 0); furthest = Math.max(Number(draft.furthest || 0), step);
        activeResourceId = resources[0]?.draft_resource_id || ""; activeLayer = resources[0]?.architecture_layer_key || "DATA";
      }
      dirty = false;
    } catch (error) { shell.toast(error.message); }
    finally { loading = false; render(); }
  }

  document.getElementById("previous").onclick = () => { if (step > 0) { step -= 1; render(); } };
  document.getElementById("next").onclick = () => { if (validateStep() && step < 2) { step += 1; furthest = Math.max(furthest, step); render(); } };
  document.getElementById("commit").onclick = () => void commit();
  document.getElementById("save-draft").onclick = saveDraft;
  document.querySelectorAll("#wizard-steps [data-step]").forEach((button) => button.onclick = () => { const next = Number(button.dataset.step); if (next <= furthest) { step = next; render(); } });
  document.addEventListener("click", (event) => {
    const anchor = event.target.closest?.("a[href]");
    if (!dirty || !anchor || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || anchor.target === "_blank" || anchor.hasAttribute("download")) return;
    const destination = new URL(anchor.href, location.href);
    if (destination.href === location.href || (destination.origin === location.origin && destination.pathname === location.pathname && destination.search === location.search)) return;
    event.preventDefault();
    pendingLeaveHref = destination.href;
    const dialog = document.getElementById("workspace-leave-dialog");
    if (!dialog.open) dialog.showModal();
  });
  document.getElementById("workspace-leave-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const destination = pendingLeaveHref;
    pendingLeaveHref = "";
    dirty = false;
    document.getElementById("workspace-leave-dialog").close();
    if (destination) location.assign(destination);
  });
  document.getElementById("cancel-workspace-leave").onclick = () => { pendingLeaveHref = ""; document.getElementById("workspace-leave-dialog").close(); };
  document.getElementById("workspace-leave-dialog").addEventListener("cancel", (event) => { event.preventDefault(); pendingLeaveHref = ""; event.currentTarget.close(); });
  addEventListener("beforeunload", (event) => { if (dirty) event.preventDefault(); });
  shell.ready.then(load).catch((error) => shell.toast(error.message));
})();
