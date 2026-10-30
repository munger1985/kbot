(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const esc = shell.escape;
  const panel = document.getElementById("workspace-panel");
  const searchParams = new URLSearchParams(location.search);
  const systemId = searchParams.get("id") || "new";
  const creating = systemId === "new";
  let workspaceReadOnly = !creating && searchParams.get("view") === "readonly";

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
  const oracleScopeOptions = [["NON_CDB", "非 CDB"], ["CDB_ROOT", "CDB"], ["PDB", "PDB"]];
  const relationTypes = [
    ["ORACLE_RAC", "组成 Oracle RAC"], ["DATA_GUARD", "Data Guard 保护"],
    ["OGG", "Oracle GoldenGate 复制"],
    ["DEBEZIUM_CDC", "Debezium CDC"], ["FLINK_CDC", "Flink CDC"], ["DATAX", "DataX 批量同步"],
    ["MYSQL_REPLICATION", "MySQL 主从复制"],
    ["POSTGRES_STREAMING", "PostgreSQL 流复制"], ["RUNS_ON", "运行于"],
    ["USES_STORAGE", "使用存储"], ["DEPENDS_ON", "依赖"], ["LOAD_BALANCES", "负载均衡到"],
  ];
  let step = 0;
  let furthest = 0;
  let loading = true;
  let saving = false;
  let dirty = false;
  let draftId = searchParams.get("draft") || "";
  let draftRevision = 0;
  let resourceDrawerOpen = false;
  let resourceDrawerMode = "CATALOG";
  let snapshotState = null;
  let snapshotGeneration = 0;
  let detail = null;
  let groups = [];
  let sources = [];
  let resources = [];
  let relations = [];
  let dismissedInferenceKeys = [];
  const connectionTestStates = new Map();
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
  function isPublishedView() { return !creating && workspaceReadOnly; }
  function currentResource() { return resources.find((item) => item.draft_resource_id === activeResourceId); }
  function currentRelation() { return relations.find((item) => item.draft_relation_id === activeRelationId); }
  function groupName(value) { return groups.find((item) => item.responsibility_group_id === value)?.name || "继承业务责任组"; }
  function layerName(value) { return layers.find((item) => item[0] === value)?.[1] || value; }
  function typeName(value) { return resourceCatalog.find((item) => item[0] === value)?.[1] || value; }
  function relationName(value) { return relationTypes.find((item) => item[0] === value)?.[1] || value; }
  function environmentName(value) { return { PROD: "生产", STG: "预发布", DEV: "开发" }[value] || value || "—"; }
  function deploymentName(value) { return value === "RAC" ? "Oracle RAC" : "单实例"; }
  function databaseRoleName(value) { return { PRIMARY: "主库", STANDBY: "备库 / 从库", UNKNOWN: "角色待识别" }[value] || value || "—"; }
  function connectivityName(value) { return { CONNECTED: "连接正常", CHECKING: "检查中", DEGRADED: "连接降级", MISCONFIGURED: "配置异常", UNREACHABLE: "无法连接", UNKNOWN: "尚未检查" }[value] || value || "—"; }
  function oracleScopeName(value) { return { CDB_ROOT: "CDB Root", CDB: "CDB Root", PDB: "PDB", NON_CDB: "非 CDB" }[value] || "—"; }
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
  const relationCompatibility = {
      DATA_GUARD: { source: ['ORACLE'], target: ['ORACLE'] },
      OGG: { source: ['ORACLE'], target: ['ORACLE', 'MYSQL', 'POSTGRESQL'] },
      DEBEZIUM_CDC: { source: ['ORACLE', 'MYSQL', 'POSTGRESQL'], target: ['ORACLE', 'MYSQL', 'POSTGRESQL'] },
      FLINK_CDC: { source: ['ORACLE', 'MYSQL', 'POSTGRESQL'], target: ['ORACLE', 'MYSQL', 'POSTGRESQL'] },
      DATAX: { source: ['ORACLE', 'MYSQL', 'POSTGRESQL'], target: ['ORACLE', 'MYSQL', 'POSTGRESQL'] },
      MYSQL_REPLICATION: { source: ['MYSQL'], target: ['MYSQL'] },
      POSTGRES_STREAMING: { source: ['POSTGRESQL'], target: ['POSTGRESQL'] },
  };
  const relationEndpointRoles = {
      DATA_GUARD: { source: 'PRIMARY', target: 'STANDBY' }, OGG: { source: 'SOURCE', target: 'TARGET' },
      DEBEZIUM_CDC: { source: 'SOURCE', target: 'TARGET' }, FLINK_CDC: { source: 'SOURCE', target: 'TARGET' }, DATAX: { source: 'SOURCE', target: 'TARGET' },
      MYSQL_REPLICATION: { source: 'SOURCE', target: 'REPLICA' }, POSTGRES_STREAMING: { source: 'PRIMARY', target: 'STANDBY' },
  };
  const databaseRelationTypes = relationTypes.filter(item => Object.hasOwn(relationCompatibility,item[0])).map(([value,label]) => ({value,label}));
  function databaseUnits() {
    const seen = new Set();return resources.flatMap(resource => {
      if (!resource.database) return [];const unitId = resource.rac_group_id ? `cluster:${resource.rac_group_id}` : `resource:${resource.draft_resource_id}`;if (seen.has(unitId)) return [];seen.add(unitId);
      return [{unit_id:unitId,resources:resource.rac_group_id ? racMembers(resource) : [resource],db_type:resource.database.db_type,display_name:resource.rac_cluster_name || resource.display_name || "未命名数据库",db_role:resource.database.db_role,is_rac:Boolean(resource.rac_group_id)}];
    });
  }
  function databaseUnitForResource(resourceId) {
      return databaseUnits().find(unit => unit.resources.some(resource => resource.draft_resource_id === resourceId));
  }
  function databaseUnit(unitId) { return databaseUnits().find(unit => unit.unit_id === unitId); }
  function compatibleDatabaseUnits(relationType, endpoint) {
      const engines = relationCompatibility[relationType]?.[endpoint] || [];
      return databaseUnits().filter(unit => engines.includes(unit.db_type));
  }
  function supportsRelationEndpoints(relationType, sourceUnit, targetUnit) {
      const compatibility = relationCompatibility[relationType];
      return Boolean(compatibility?.source.includes(sourceUnit.db_type) && compatibility.target.includes(targetUnit.db_type));
  }
  function relationSourceUnitId(item) {
      const sourceRole = relationEndpointRoles[item.relation_type_key]?.source;
      const participant = item.participants.find(row => row.participant_role === sourceRole)
          || item.participants.find(row => ['PRIMARY', 'SOURCE'].includes(row.participant_role))
          || item.participants[0];
      return participant ? databaseUnitForResource(participant.draft_resource_id)?.unit_id || '' : '';
  }
  function relationTargetUnitId(item) {
      const targetRole = relationEndpointRoles[item.relation_type_key]?.target;
      const sourceUnitId = relationSourceUnitId(item);
      const participant = item.participants.find(row => row.participant_role === targetRole && databaseUnitForResource(row.draft_resource_id)?.unit_id !== sourceUnitId)
          || item.participants.find(row => databaseUnitForResource(row.draft_resource_id)?.unit_id !== sourceUnitId);
      return participant ? databaseUnitForResource(participant.draft_resource_id)?.unit_id || '' : '';
  }
  function relationTypeOptions(item) {
      const sourceUnit = databaseUnit(relationSourceUnitId(item));
      const targetUnit = databaseUnit(relationTargetUnitId(item));
      if (!sourceUnit || !targetUnit)
          return [];
      return databaseRelationTypes.filter(option => supportsRelationEndpoints(option.value, sourceUnit, targetUnit));
  }
  function buildRelationParticipants(type, sourceUnit, targetUnit) {
      const roles = relationEndpointRoles[type];
      if (!roles)
          return [];
      const sourceParticipants = sourceUnit.resources.map((resource, index) => ({
          draft_resource_id: resource.draft_resource_id,
          participant_role: roles.source,
          display_order: index,
      }));
      return [
          ...sourceParticipants,
          ...targetUnit.resources.map((resource, index) => ({
              draft_resource_id: resource.draft_resource_id,
              participant_role: roles.target,
              display_order: sourceParticipants.length + index,
          })),
      ];
  }
  function generatedRelationName(type, sourceUnit, targetUnit) {
      return `${relationName(type)} · ${sourceUnit.display_name} → ${targetUnit.display_name}`;
  }
  function relationDatabaseAttributes(sourceUnit, targetUnit) {
      return { source_database_type: sourceUnit.db_type, target_database_type: targetUnit.db_type };
  }
  function relationEndpointSummary(item) {
      const source = databaseUnit(relationSourceUnitId(item))?.display_name || '未识别源端';
      const target = databaseUnit(relationTargetUnitId(item))?.display_name || '未识别目标端';
      return `${source} → ${target}`;
  }
  function hasDatabaseRelation(type, sourceUnitId, targetUnitId, except) {
      return relations.some(relation => relation !== except
          && relation.relation_type_key === type
          && relationSourceUnitId(relation) === sourceUnitId
          && relationTargetUnitId(relation) === targetUnitId);
  }
  function relationInferenceKey(item) {
      return typeof item.attributes.inference_key === 'string' ? item.attributes.inference_key : '';
  }
  function isInferenceOverride(item) {
      return item.attributes.inference_override === true && Boolean(relationInferenceKey(item));
  }
  function makeRelationManual(item) {
      if (!isInferredRelation(item) || isManagedRacRelation(item))
          return;
      item.origin = 'MANUAL';
      item.attributes = { ...item.attributes, auto_inferred: false, inference_override: true };
  }
  function removeRelation(item) {
      const inferenceKey = relationInferenceKey(item);
      if (inferenceKey && !dismissedInferenceKeys.includes(inferenceKey))
          dismissedInferenceKeys.push(inferenceKey);
      relations = relations.filter(relation => relation !== item);
      activeRelationId = '';
      mark();
  }
  function restoreInferredRelation(item) {
      const inferenceKey = relationInferenceKey(item);
      relations = relations.filter(relation => relation !== item);
      dismissedInferenceKeys = dismissedInferenceKeys.filter(key => key !== inferenceKey);
      syncInferredDatabaseRelations();
      activeRelationId = relations.find(relation => relationInferenceKey(relation) === inferenceKey)?.draft_relation_id || '';
      mark();
  }
  function restoreDismissedRelations() {
      if (!dismissedInferenceKeys.length)
          return;
      dismissedInferenceKeys = [];
      syncInferredDatabaseRelations();
      mark();
      shell.toast('已重新生成当前能够确定的数据库关系。');
  }
  function managedRacRelation(item) {
    return item?.rac_group_id ? relations.find((relation) => relation.draft_relation_id === item.rac_group_id && relation.relation_type_key === "ORACLE_RAC") : undefined;
  }
  function isManagedRacRelation(relation) {
    return relation?.relation_type_key === "ORACLE_RAC"
      && relation.participants.length >= 2
      && relation.participants.every((participant) => resources.find((resource) => resource.draft_resource_id === participant.draft_resource_id)?.rac_group_id === relation.draft_relation_id);
  }
  function isInferredRelation(relation) {
    return relation?.attributes?.auto_inferred === true;
  }
  function isManagedRelation(relation) {
    return isManagedRacRelation(relation);
  }
  function syncInferredDatabaseRelations() {
      const definitions = {
          ORACLE: { relationType: 'DATA_GUARD', label: 'Data Guard' },
          MYSQL: { relationType: 'MYSQL_REPLICATION', label: 'MySQL 主从复制' },
          POSTGRESQL: { relationType: 'POSTGRES_STREAMING', label: 'PostgreSQL 流复制' },
      };
      const desired = new Map();
      for (const [dbType, definition] of Object.entries(definitions)) {
          const seen = new Set();
          const units = resources.flatMap(resource => {
              if (resource.database?.db_type !== dbType)
                  return [];
              const key = resource.rac_group_id || resource.draft_resource_id;
              if (seen.has(key))
                  return [];
              seen.add(key);
              const members = resource.rac_group_id ? racMembers(resource) : [resource];
              return [{ key, members, role: members[0]?.database?.db_role, name: resource.rac_cluster_name || resource.display_name || dbType }];
          });
          const primaries = units.filter(unit => unit.role === 'PRIMARY');
          const standbys = units.filter(unit => unit.role === 'STANDBY');
          if (primaries.length !== 1 || !standbys.length)
              continue;
          for (const standby of standbys) {
              const sourceUnit = { unit_id: primaries[0].key, resources: primaries[0].members, db_type: dbType, display_name: primaries[0].name, db_role: primaries[0].role || 'UNKNOWN', is_rac: primaries[0].members.length > 1 };
              const targetUnit = { unit_id: standby.key, resources: standby.members, db_type: dbType, display_name: standby.name, db_role: standby.role || 'UNKNOWN', is_rac: standby.members.length > 1 };
              const inferenceKey = `DATABASE_ROLE:${dbType}:${primaries[0].key}:${standby.key}`;
              desired.set(inferenceKey, {
                  draft_relation_id: '', origin: 'DISCOVERY', relation_type_key: definition.relationType,
                  display_name: `${definition.label} · ${sourceUnit.display_name} → ${targetUnit.display_name}`,
                  responsibility_group_id: '', participants: buildRelationParticipants(definition.relationType, sourceUnit, targetUnit),
                  attributes: { auto_inferred: true, inference_rule: 'DATABASE_ROLE', inference_key: inferenceKey, database_type: dbType },
              });
          }
      }
      for (const key of dismissedInferenceKeys)
          desired.delete(key);
      for (const relation of relations) {
          if (isInferenceOverride(relation))
              desired.delete(relationInferenceKey(relation));
      }
      let changed = false;
      const retained = [];
      for (const relation of relations) {
          const inferenceKey = typeof relation.attributes.inference_key === 'string' ? relation.attributes.inference_key : '';
          if (!isInferredRelation(relation) || relation.attributes.inference_rule !== 'DATABASE_ROLE') {
              retained.push(relation);
              continue;
          }
          const next = desired.get(inferenceKey);
          if (!next) {
              changed = true;
              continue;
          }
          desired.delete(inferenceKey);
          const nextParticipants = JSON.stringify(next.participants);
          if (relation.display_name !== next.display_name || relation.relation_type_key !== next.relation_type_key || JSON.stringify(relation.participants) !== nextParticipants) {
              relation.display_name = next.display_name;
              relation.relation_type_key = next.relation_type_key;
              relation.participants = next.participants;
              relation.attributes = next.attributes;
              relation.origin = 'DISCOVERY';
              changed = true;
          }
          retained.push(relation);
      }
      for (const relation of desired.values()) {
          const participantIds = new Set(relation.participants.map(item => item.draft_resource_id));
          const equivalent = retained.some(item => item.relation_type_key === relation.relation_type_key
              && item.participants.length === relation.participants.length
              && item.participants.every(participant => participantIds.has(participant.draft_resource_id)));
          if (equivalent)
              continue;
          relation.draft_relation_id = uuid7();
          retained.push(relation);
          changed = true;
      }
      if (changed)
          relations = retained;
      return changed;
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
  function normalizeDraftDatabase(database) {
    if (!database) return undefined;
    const selectedCandidate = (database.candidates || []).find((row) => row.candidate_ref === database.candidate_ref);
    const selectedHostCandidate = (database.host_candidates || []).find((row) => row.candidate_ref === database.host_candidate_ref);
    return {
      ...database,
      tls_enabled: database.tls_enabled === true,
      oracle_scope: database.db_type === "ORACLE" && database.oracle_scope === "CDB" ? "CDB_ROOT" : database.oracle_scope,
      candidate_locator_key: database.candidate_locator_key || selectedCandidate?.locator_key || "",
      host_candidate_locator_key: database.host_candidate_locator_key || selectedHostCandidate?.locator_key || "",
      candidates: database.candidates || [],
      host_candidates: database.host_candidates || [],
      discovering: false,
    };
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
        tls_enabled: false,
        username: "", password: "", source_id: "", locator_key: "", candidate_ref: "",
        candidate_locator_key: "", host_candidate_ref: "", host_candidate_locator_key: "",
        candidates: [], host_candidates: [], discovering: false, discovery_revision: 0,
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
        host: "", database: "", locator_key: "", candidate_ref: "", candidate_locator_key: "",
        host_candidate_ref: "", host_candidate_locator_key: "",
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
      relations.push({ draft_relation_id: relationId, origin: "DISCOVERY", relation_type_key: "ORACLE_RAC", display_name: item.rac_cluster_name || "", responsibility_group_id: "", participants: [], attributes: { auto_inferred: true, inference_rule: "RAC_DEPLOYMENT" } });
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
    if (snapshotState?.mutating) { shell.toast("采集配置操作正在进行，请稍候。"); return; }
    snapshotGeneration += 1;
    snapshotState = null;
    resourceDrawerOpen = false;
    const drawer = document.getElementById("business-resource-drawer");
    if (drawer?.open) drawer.close();
    drawer?.replaceChildren();
  }

  function removeResource(item) {
    const removed = new Set(racMembers(item).map((member) => member.draft_resource_id));
    removed.forEach((resourceId) => connectionTestStates.delete(resourceId));
    resources = resources.filter((row) => !removed.has(row.draft_resource_id));
    relations = relations.filter((relation) => relation.draft_relation_id !== item.rac_group_id && !relation.participants.some((participant) => removed.has(participant.draft_resource_id)));
    syncInferredDatabaseRelations();
    activeResourceId = resources.find((row) => row.architecture_layer_key === activeLayer)?.draft_resource_id || resources[0]?.draft_resource_id || "";
    closeResourceDrawer(); mark(); renderComposition();
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
    snapshotState = null;
    renderComposition();
    if (snapshotTargetId(currentResource())) void loadSnapshotPolicy();
  }

  function snapshotTargetId(item) {
    return detail?.resources?.find((row) => row.system_resource_id === item?.system_resource_id)?.resource?.database?.target_id || "";
  }

  function snapshotError(error) {
    return [409, 412].includes(error?.status) ? "Target 配置已被其他操作修改，请重新打开业务系统后检查草稿。" : error?.message || "操作失败，请重试。";
  }

  function snapshotRender() {
    updateFooter();
    if (resourceDrawerOpen) {
      const scrollTop = document.querySelector(".resource-drawer-body")?.scrollTop || 0;
      renderResourceDrawer(); bindResourceDrawer();
      const body = document.querySelector(".resource-drawer-body");
      if (body) body.scrollTop = scrollTop;
    }
  }

  function snapshotToast(message) {
    const node = document.createElement("div");
    node.className = "ops-toast";
    node.textContent = message;
    node.setAttribute("role", "status");
    document.getElementById("business-resource-drawer").append(node);
    if (typeof node.showPopover === "function") {
      node.setAttribute("popover", "manual");
      node.style.inset = "auto 24px 24px auto";
      node.style.margin = "0";
      node.style.border = "0";
      node.showPopover();
    }
    setTimeout(() => node.remove(), 5000);
  }

  async function loadSnapshotPolicy() {
    const targetId = snapshotTargetId(currentResource());
    const ticket = ++snapshotGeneration;
    snapshotState = { targetId, target: null, draft: null, busy: true, error: "" };
    snapshotRender();
    if (!targetId) { snapshotState.busy = false; snapshotState.error = "未找到当前数据库对应的 Target。"; snapshotRender(); return; }
    try {
      const target = await request(`/targets/${encodeURIComponent(targetId)}`);
      if (ticket !== snapshotGeneration) return;
      snapshotState.target = target;
      const staged = currentResource()?.workload_snapshot;
      snapshotState.draft = { ...(staged?.target_id === targetId ? staged.policy : target.workload_snapshot_policy) };
    } catch (error) { if (ticket === snapshotGeneration) snapshotState.error = snapshotError(error); }
    finally { if (ticket === snapshotGeneration) { snapshotState.busy = false; snapshotRender(); } }
  }

  function stageSnapshotPolicy() {
    const item = currentResource();
    const state = snapshotState;
    if (isPublishedView() || !item || !state?.target || !state.draft) return;
    const old = item.workload_snapshot?.target_id === state.targetId ? item.workload_snapshot : null;
    const original = old?.original_policy || state.target.workload_snapshot_policy;
    item.workload_snapshot = JSON.stringify(state.draft) === JSON.stringify(original) ? null : { target_id: state.targetId, target_row_version: old?.target_row_version || state.target.row_version, original_policy: { ...original }, policy: { ...state.draft } };
    mark();
  }

  async function snapshotOperation() {
    const item = currentResource();
    const state = snapshotState;
    if (isPublishedView() || !item || !state?.target || !state.draft || state.busy) return;
    const ticket = snapshotGeneration;
    state.busy = true; state.mutating = true; state.error = ""; snapshotRender();
    let notification = "";
    try {
      const url = `/targets/${encodeURIComponent(state.targetId)}`;
      const headers = { "If-Match": `"rv-${item.workload_snapshot?.target_row_version || state.target.row_version}"`, "Idempotency-Key": uuid7() };
      const target = await request(`${url}/connectivity-checks`, { method: "POST", headers });
      if (ticket !== snapshotGeneration) return;
      state.target = target;
      if (item.workload_snapshot) item.workload_snapshot.target_row_version = target.row_version;
      notification = `连接检查已完成：${target.connectivity_status || "状态未知"}。连接检查立即执行；采集策略仍需发布业务系统。`;
    } catch (error) { if (ticket === snapshotGeneration) state.error = snapshotError(error); }
    finally { if (ticket === snapshotGeneration) { state.busy = false; state.mutating = false; snapshotRender(); if (notification) snapshotToast(notification); } }
  }

  function snapshotHtml(item) {
    const source = { POSTGRESQL: "pg_stat_statements（须启用扩展并赋予只读权限）", MYSQL: "performance_schema SQL digest（须启用统计并赋予只读权限）", ORACLE: "V$SQLSTATS（非 AWR，不改变 AWR 快照周期）" }[item.database?.db_type] || "数据库统计视图";
    const state = snapshotState;
    const status = state?.target;
    const policy = state?.draft;
    const disabled = state?.busy ? "disabled" : "";
    const formatTime = (value) => value ? new Date(value).toLocaleString("zh-CN", { hour12: false }) : "暂无";
    let content = state?.busy && !status ? "<p>正在加载采集策略…</p>" : "";
    if (state?.error) content += `<p class="snapshot-error" role="alert">${esc(state.error)} ${!status ? `<button type="button" id="snapshot-reload" ${disabled}>重试</button>` : ""}</p>`;
    if (status && policy) {
      content += `<div class="snapshot-status">当前策略：${status.workload_snapshot_policy.enabled ? "已启用" : "未启用"} · 最近采集尝试：${esc(formatTime(status.workload_last_collected_at))} · 下次计划：${esc(formatTime(status.workload_next_run_at))}<br>最近采集错误：${esc(status.workload_last_error_code || "无")}</div>`;
      if (!isPublishedView()) content += `<label class="snapshot-toggle"><input data-snapshot-field="enabled" type="checkbox" ${policy.enabled ? "checked" : ""} ${disabled}> 启用周期快照</label><div class="snapshot-fields"><label>采集范围<select data-snapshot-field="collection_mode" ${disabled}><option value="BOUNDED_CANDIDATES" ${policy.collection_mode !== "VISIBLE_SET" ? "selected" : ""}>有限累计候选</option><option value="VISIBLE_SET" ${policy.collection_mode === "VISIBLE_SET" ? "selected" : ""}>尝试读取可见集合（有界）</option></select></label>${policy.collection_mode === "VISIBLE_SET" ? `<label>可见集合行数上限<input data-snapshot-field="visible_statement_limit" type="number" min="10" max="500" step="1" value="${esc(policy.visible_statement_limit || 500)}" ${disabled}></label>` : ""}<label>采集间隔（分钟）<input data-snapshot-field="interval_minutes" type="number" min="1" max="60" step="1" value="${esc(policy.interval_minutes)}" ${disabled}></label><label>保留天数<input data-snapshot-field="retention_days" type="number" min="1" max="365" step="1" value="${esc(policy.retention_days)}" ${disabled}></label><label>候选 SQL 数<input data-snapshot-field="top_statement_limit" type="number" min="10" max="500" step="1" value="${esc(policy.top_statement_limit)}" ${disabled}></label></div><div class="snapshot-actions"><button class="secondary-button" type="button" id="snapshot-refresh" ${disabled}>测试连接并刷新能力</button><span id="snapshot-dirty" ${item.workload_snapshot ? "" : "hidden"}>策略已修改，将随业务系统发布</span></div>`;
      else content += `<p>采集范围：${policy.collection_mode === "VISIBLE_SET" ? "尝试读取可见集合（有界）" : "有限累计候选"} · 行数上限：${esc(policy.collection_mode === "VISIBLE_SET" ? policy.visible_statement_limit : policy.top_statement_limit)} · 间隔：${esc(policy.interval_minutes)} 分钟 · 保留：${esc(policy.retention_days)} 天</p>`;
      content += `<button type="button" id="snapshot-status-refresh" class="secondary-button snapshot-reload" ${disabled}>刷新采集状态</button>`;
    }
    const hint = isPublishedView() ? "运行中的业务系统仅供查看；下线后可编辑采集策略。" : "修改保存在业务系统草稿中，发布时生效；RAC 按当前节点配置。";
    return `<section class="snapshot-settings readonly-detail-section"><header><span>04</span><div><strong>历史 SQL 采集</strong><small>${hint}</small></div></header><div class="snapshot-body"><p>统计来源：${esc(source)}。采集需要已启用的 Target、可用的只读连接、有效 Agent 绑定和运行中的采集服务。</p><p>启用后至少需要两次可比较的快照才能计算窗口增量，不回填历史；窗口可能未被完整覆盖。</p>${content}</div></section>`;
  }

  function renderBusiness() {
    const disabled = isPublishedView() ? "disabled" : "";
    panel.innerHTML = `<section class="step-surface business-step">
      <header><div><p class="eyebrow">STEP 1</p><h2>业务信息</h2></div></header>
      <div class="form-section"><div class="section-copy"><h3>基本信息</h3><p>名称面向业务人员和运维人员共同识别。</p></div><div class="form-grid"><label class="span-8"><span>业务系统名称</span><input name="display_name" value="${esc(system.display_name)}" placeholder="例如：MES 生产系统" ${disabled}></label><label class="span-4"><span>运行环境</span><select name="environment" ${disabled}>${selectOptions([["PROD", "生产"], ["STG", "预发布"], ["DEV", "开发"]], system.environment)}</select></label></div></div>
      <div class="form-section"><div class="section-copy"><h3>责任与影响</h3><p>业务责任组处理整体影响，资源可在下一步指定技术责任组。</p></div><div class="form-grid"><label class="span-6"><span>业务责任组</span><select name="responsibility_group_id" ${disabled}>${groupOptions(system.responsibility_group_id, "请选择")}</select></label><label class="span-6"><span>重要级</span><select name="importance_level" ${disabled}>${selectOptions([["1", "L1 低"], ["2", "L2 一般"], ["3", "L3 重要"], ["4", "L4 关键"], ["5", "L5 核心"]], String(system.importance_level))}</select></label></div></div>
      <div class="form-section"><div class="section-copy"><h3>运维说明</h3><p>记录服务范围、影响窗口或跨团队协作背景。</p></div><label class="description-field"><span>说明（可选）</span><textarea name="description" rows="5" maxlength="1000" ${disabled}>${esc(system.description)}</textarea></label></div>
    </section>`;
    if (isPublishedView()) return;
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
    return `<header class="admin-drawer-header"><div><p class="eyebrow">${esc(layerName(activeLayer))}</p><h2>添加资源</h2><p>选择要接入当前架构层的运维资源类型。</p></div><button id="close-resource-drawer" class="icon-button" type="button" aria-label="关闭添加资源">×</button></header><div class="resource-drawer-body"><label class="resource-type-field"><span>资源类型</span><select id="catalog-type">${options}</select></label><article class="resource-type-summary"><span class="resource-icon">${catalogType === "DATABASE" ? "DB" : "R"}</span><div><strong>${esc(typeName(catalogType))}</strong><p>${enabled ? "接入新数据库，并从监控源自动发现数据库和承载主机。" : "该资源类型将在后续版本开放。"}</p></div></article></div><footer class="admin-drawer-footer"><button id="cancel-resource-drawer" class="secondary-button" type="button">取消</button><button id="confirm-add-resource" class="primary-button" type="button">配置${esc(typeName(catalogType))}</button></footer>`;
  }

  function readonlyResourceHtml(item) {
    if (!item) return `<aside class="resource-guidance"><h3>选择资源查看详情</h3><p>点击实例卡片查看运行配置。</p></aside>`;
    const database = item.database || {};
    const serviceLabel = database.db_type === "ORACLE" ? "Service Name" : "Database";
    const details = (rows) => rows.map(([label, value, technical = false]) => `<div><dt>${esc(label)}</dt><dd class="${technical ? "technical-value" : ""}">${esc(value ?? "—")}</dd></div>`).join("");
    const instanceRows = [
      ["数据库名称", item.display_name || "—"],
      ["数据库引擎", `${database.db_type || "—"} ${database.version_code || ""}`.trim()],
      ["运行环境", environmentName(database.environment)],
      ["部署形态", `${deploymentName(database.deployment_type)}${database.rac_node_number ? ` · 节点 ${database.rac_node_number}` : ""}`],
      ["数据库角色", databaseRoleName(database.db_role)],
      ["技术责任组", groupName(item.owner_group_id)],
      ["位置", item.location_name || "—"],
      ["资源状态", "已接入"],
    ];
    const connectionRows = [
      ["IP / 主机", database.host || "—", true],
      ["端口", database.port || "—", true],
      [serviceLabel, database.database || "—", true],
      ["登录用户", database.username || "—", true],
    ];
    if (database.db_type === "ORACLE") connectionRows.push(["容器范围", oracleScopeName(database.oracle_scope)]);
    if (database.db_type === "ORACLE" && database.oracle_scope === "PDB") connectionRows.push(["PDB 名称", database.oracle_pdb_name || "—", true]);
    const statusRows = [
      ["数据库连接", connectivityName(database.connectivity_status)],
      ["监控映射", database.monitoring_ready ? "已就绪" : "未就绪"],
    ];
    const section = (number, title, hint, rows) => `<section class="readonly-detail-section"><header><span>${number}</span><div><strong>${esc(title)}</strong><small>${esc(hint)}</small></div></header><dl>${details(rows)}</dl></section>`;
    return `<header class="admin-drawer-header"><div><p class="eyebrow">${esc(layerName(item.architecture_layer_key))} · ${esc(typeName(item.resource_type_key))}</p><h2>${esc(item.display_name || "数据库实例")}</h2><p>查看运行中业务系统的数据库实例与连接配置。</p></div><button id="close-resource-drawer" class="icon-button" type="button" aria-label="关闭实例详情">×</button></header><div class="resource-drawer-body readonly-resource-detail">${section("01", "实例信息", "数据库身份、部署形态与责任归属。", instanceRows)}${section("02", "连接信息", "只展示连接所需的非敏感元数据，不展示密码。", connectionRows)}${section("03", "运行状态", "连接检查与监控映射的当前结果。", statusRows)}${snapshotTargetId(item) && item.database ? snapshotHtml(item) : "<p>未找到当前数据库对应的 Target，无法配置历史 SQL 采集。</p>"}</div><footer class="admin-drawer-footer"><button id="finish-resource-drawer" class="primary-button" type="button">关闭</button></footer>`;
  }

  function resourceEditorHtml(item) {
    if (!item) return `<aside class="resource-guidance"><h3>选择资源查看详情</h3><p>大型系统仍以列表为主；拓扑确认页只加载当前焦点范围。</p></aside>`;
    const database = item.database;
    const editing = Boolean(item.system_resource_id);
    const rac = isRacResource(item);
    const connectionTestState = connectionTestStates.get(item.draft_resource_id);
    const connectionTestMessage = connectionTestState?.message || (editing ? "修改将在保存业务系统时提交" : item.saved ? "数据库连接已测试" : "连接测试为可选项，不影响保存");
    const connectionTestTone = connectionTestState?.status === "SUCCEEDED" ? "good" : connectionTestState?.status === "FAILED" ? "bad" : "";
    const connectionTestRunning = connectionTestState?.status === "CHECKING";
    const baseFields = `<div class="editor-grid"><label class="span-6"><span>${rac ? "RAC 名称" : "数据库名称"}</span><input name="${rac ? "rac_cluster_name" : "resource_name"}" value="${esc(rac ? item.rac_cluster_name : item.display_name)}"></label><label class="span-6"><span>技术责任组</span><select name="owner_group_id">${groupOptions(item.owner_group_id, "请选择")}</select></label><label class="span-all"><span>位置（可选）</span><input name="location_name" value="${esc(item.location_name)}" placeholder="机房、地域或可用区"></label></div>`;
    const databaseFields = database ? `<div class="editor-grid database-grid"><label class="span-3"><span>数据库引擎</span><select name="db_type" ${editing || rac ? "disabled" : ""}>${selectOptions([["ORACLE", "Oracle"], ["MYSQL", "MySQL"], ["POSTGRESQL", "PostgreSQL"]], database.db_type)}</select></label><label class="span-3"><span>数据库版本</span><select name="version_code">${selectOptions(versions[database.db_type], database.version_code)}</select></label><label class="span-3"><span>数据库环境</span><select name="db_environment">${selectOptions([["PROD", "生产"], ["STG", "预发布"], ["DEV", "开发"]], database.environment)}</select></label><label class="span-3"><span>部署形态</span><select name="deployment_type" ${database.db_type !== "ORACLE" || editing ? "disabled" : ""}>${selectOptions([["SINGLE_INSTANCE", "单实例"], ["RAC", "RAC（多节点）"]], database.deployment_type)}</select></label><label class="span-3"><span>数据库角色</span><select name="db_role">${selectOptions([["UNKNOWN", "待识别"], ["PRIMARY", "主库"], ["STANDBY", "备库"]], database.db_role)}</select></label></div>` : "";
    let connection = "";
    if (item.mode === "NEW" && database && !rac) {
      const discovery = usesDiscovery(item);
      const oracle = database.db_type === "ORACLE";
      const pdb = oracle && database.oracle_scope === "PDB";
      const endpointFields = `<label><span>数据库地址</span><input name="host" value="${esc(database.host)}" placeholder="主机名或 IP"></label><label><span>端口</span><input name="port" value="${esc(database.port)}" type="number"></label><label><span>${oracle ? "Service Name" : "Database"}</span><input name="database" value="${esc(database.database)}"></label>${oracle ? `<label><span>容器范围</span><select name="oracle_scope">${selectOptions(oracleScopeOptions, database.oracle_scope)}</select></label>` : ""}`;
      const accessFields = `${pdb ? `<label><span>PDB 名称</span><input name="oracle_pdb_name" value="${esc(database.oracle_pdb_name)}"></label>` : ""}<label><span>诊断用户名</span><input name="username" value="${esc(database.username)}" autocomplete="off"></label><label><span>诊断密码</span><input name="password" value="${esc(database.password)}" type="password" autocomplete="new-password"></label>`;
      const monitoringFields = `<label><span>监控源（必选）</span><select name="source_id" required><option value="">请选择</option>${sources.map((row) => `<option value="${esc(row.source_id)}" ${row.source_id === database.source_id ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.source_type)}</option>`).join("")}</select></label>${!discovery ? `<label><span>监控定位键</span><input name="locator_key" value="${esc(database.locator_key)}" placeholder="与监控 Label 完全一致"></label>` : `<label><span>数据库候选</span><select name="candidate_ref" ${database.discovering ? "disabled" : ""}><option value="">请选择</option>${database.candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === database.candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label><label><span>承载主机候选</span><select name="host_candidate_ref" ${database.discovering ? "disabled" : ""}><option value="">请选择</option>${database.host_candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === database.host_candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label>`}`;
      connection = `<div class="connection-layout"><section class="connection-field-group"><p>连接端点</p><div class="connection-row connection-endpoint-row ${oracle ? "oracle" : ""}">${endpointFields}</div></section><section class="connection-field-group"><p>数据库访问</p><div class="connection-row connection-access-row ${pdb ? "pdb" : ""}">${accessFields}</div></section><section class="connection-field-group"><p>监控映射</p><div class="connection-row connection-monitoring-row ${discovery ? "discovery" : ""}">${monitoringFields}</div>${database.discovering ? `<p class="discovery-status">正在从监控源发现数据库和承载主机…</p>` : ""}</section></div>`;
    }
    if (item.mode === "NEW" && database && rac) {
      const discovery = usesDiscovery(item);
      const nodeCards = racMembers(item).map((node) => {
        const nodeDatabase = node.database;
        const nodeId = esc(node.draft_resource_id);
        const monitoring = discovery
          ? `<label class="span-6"><span>数据库监控 Target</span><select data-rac-node="${nodeId}" data-rac-field="candidate_ref" ${nodeDatabase.discovering ? "disabled" : ""}><option value="">请选择</option>${nodeDatabase.candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === nodeDatabase.candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label><label class="span-6"><span>主机监控 Target</span><select data-rac-node="${nodeId}" data-rac-field="host_candidate_ref" ${nodeDatabase.discovering ? "disabled" : ""}><option value="">请选择</option>${nodeDatabase.host_candidates.map((row) => `<option value="${esc(row.candidate_ref)}" ${row.candidate_ref === nodeDatabase.host_candidate_ref ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.locator_key)}</option>`).join("")}</select></label>`
          : `<label class="span-all"><span>监控定位键</span><input data-rac-node="${nodeId}" data-rac-field="locator_key" value="${esc(nodeDatabase.locator_key)}"></label>`;
        return `<article class="rac-node-card"><header><div><strong>RAC 节点 ${esc(nodeDatabase.rac_node_number)}</strong><small>${esc(node.display_name || "待命名节点")}</small></div>${racMembers(item).length > 2 ? `<button type="button" class="danger-link" data-remove-rac-node="${nodeId}">移除节点</button>` : ""}</header><div class="editor-grid"><label class="span-2"><span>节点编号</span><input type="number" min="1" max="1024" required data-rac-node="${nodeId}" data-rac-field="rac_node_number" value="${esc(nodeDatabase.rac_node_number)}"></label><label class="span-4"><span>Host</span><input data-rac-node="${nodeId}" data-rac-field="host" value="${esc(nodeDatabase.host)}" placeholder="节点主机名或 IP"></label><label class="span-6"><span>Service Name</span><input data-rac-node="${nodeId}" data-rac-field="database" value="${esc(nodeDatabase.database)}"></label>${monitoring}</div></article>`;
      }).join("");
      connection = `<div class="editor-grid connection-grid rac-editor"><p class="span-all rac-guidance">公共凭据与监控源只填写一次；每个节点分别配置 Host、Service Name、数据库监控 Target 和主机监控 Target。</p><label class="span-3"><span>端口</span><input name="port" value="${esc(database.port)}" type="number"></label><label class="span-3"><span>容器范围</span><select name="oracle_scope">${selectOptions(oracleScopeOptions, database.oracle_scope)}</select></label>${database.oracle_scope === "PDB" ? `<label class="span-6"><span>PDB 名称</span><input name="oracle_pdb_name" value="${esc(database.oracle_pdb_name)}"></label>` : ""}<label class="span-6"><span>诊断用户名（所有节点共用）</span><input name="username" value="${esc(database.username)}" autocomplete="off"></label><label class="span-6"><span>诊断密码（所有节点共用）</span><input name="password" value="${esc(database.password)}" type="password" autocomplete="new-password"></label><label class="span-all"><span>监控源（所有节点共用）</span><select name="source_id" required><option value="">请选择</option>${sources.map((row) => `<option value="${esc(row.source_id)}" ${row.source_id === database.source_id ? "selected" : ""}>${esc(row.display_name)} · ${esc(row.source_type)}</option>`).join("")}</select></label>${database.discovering ? `<p class="span-all discovery-status">正在从监控源发现数据库和承载主机…</p>` : ""}<section class="span-all rac-node-list">${nodeCards}<button id="add-rac-node" class="secondary-button rac-add-node" type="button">＋ 增加 RAC 节点</button></section></div>`;
    }
    return `<header class="admin-drawer-header"><div><p class="eyebrow">${esc(layerName(item.architecture_layer_key))} · ${esc(typeName(item.resource_type_key))}</p><h2>${esc(item.display_name || "配置数据库")}</h2><p>${editing ? "编辑资源配置" : "配置连接与监控映射，可按需测试连接"}</p></div><button id="close-resource-drawer" class="icon-button" type="button" aria-label="关闭资源配置">×</button></header><form id="resource-editor" class="resource-editor resource-drawer-form"><div class="resource-drawer-body">${baseFields}${databaseFields}${connection}${snapshotTargetId(item) ? snapshotHtml(item) : ""}</div><footer class="admin-drawer-footer resource-drawer-editor-footer"><span id="resource-connection-result" data-tone="${connectionTestTone}">${esc(connectionTestMessage)}</span><div class="resource-drawer-editor-actions"><button id="remove-resource" class="danger-button" type="button">移除资源</button><button id="finish-resource-drawer" class="secondary-button" type="button">完成</button>${!editing ? `<button id="validate-resource" class="primary-button" type="submit" ${saving || connectionTestRunning ? "disabled" : ""}>${connectionTestRunning ? "正在测试…" : "测试连接"}</button>` : ""}</div></footer></form>`;
  }

  function renderComposition() {
    const visible = resources.filter((item) => item.architecture_layer_key === activeLayer);
    const cards = visible.map((item) => `<article class="resource-card-shell"><button type="button" data-resource="${esc(item.draft_resource_id)}" class="resource-card"><header><span class="resource-icon">${item.resource_type_key === "DATABASE" ? "DB" : "R"}</span><em class="${item.saved ? "ready" : ""}">${item.saved ? "连接已测试" : "连接未测试"}</em></header><div><small>${esc(typeName(item.resource_type_key))}</small><h3>${esc(item.display_name || "未命名数据库")}</h3><p>${item.database ? `${esc(item.database.db_type)} ${esc(item.database.version_code)} · ${item.database.deployment_type === "RAC" ? `RAC 节点 ${esc(item.database.rac_node_number)}` : "单实例"} · ${item.database.db_role === "PRIMARY" ? "主库" : item.database.db_role === "STANDBY" ? "备库" : "角色待识别"}` : ""}</p></div><dl><div><dt>技术责任组</dt><dd>${esc(groupName(item.owner_group_id))}</dd></div><div><dt>位置</dt><dd>${esc(item.location_name || "未填写")}</dd></div></dl><footer><span>${item.system_resource_id ? "已接入资源" : ""}</span><strong>${isPublishedView() ? "查看 →" : "编辑 →"}</strong></footer></button>${!isPublishedView() && !item.system_resource_id ? `<button type="button" data-delete-resource="${esc(item.draft_resource_id)}" class="resource-card-delete" aria-label="删除 ${esc(item.display_name || "未命名数据库")}">删除</button>` : ""}</article>`).join("");
    const addCard = isPublishedView() ? "" : `<button id="add-resource-card" class="resource-add-card" type="button" aria-label="＋ 添加资源"><span>＋</span><strong>添加资源</strong><small>接入当前架构层</small></button>`;
    const content = activeLayer === "DATA" ? `<div class="resource-card-grid">${cards}${addCard}</div>` : `<div class="layer-coming-soon"><span aria-hidden="true">◇</span><h3>功能未完成</h3><p>敬请期待</p></div>`;
    panel.innerHTML = `<section class="composition-layout"><aside class="layer-nav" role="tablist" aria-label="系统构成"><header><p class="eyebrow">STEP 2</p><h2>系统构成</h2></header>${layers.map((layer) => `<button type="button" role="tab" aria-selected="${activeLayer === layer[0]}" data-layer="${layer[0]}" class="${activeLayer === layer[0] ? "active" : ""}"><span><strong>${layer[1]}</strong><small>${layer[2]}</small></span><em>${resources.filter((item) => item.architecture_layer_key === layer[0]).length}</em></button>`).join("")}</aside><section class="resource-panel" role="tabpanel"><header><div><p class="eyebrow">${esc(activeLayer)}</p><h2>${esc(layerName(activeLayer))}</h2><p>${activeLayer === "DATA" ? `已配置 ${visible.length} 个运维资源` : "功能未完成，敬请期待"}</p></div></header>${content}</section></section>`;
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
    drawer.innerHTML = resourceDrawerMode === "CATALOG" ? catalogHtml() : isPublishedView() ? readonlyResourceHtml(currentResource()) : resourceEditorHtml(currentResource());
    if (!drawer.open) drawer.showModal();
  }

  function bindComposition() {
    panel.querySelectorAll("[data-layer]").forEach((button) => button.onclick = () => { closeResourceDrawer(); activeLayer = button.dataset.layer; activeResourceId = ""; renderComposition(); });
    panel.querySelectorAll("[data-resource]").forEach((button) => button.onclick = () => openResourceEditor(button.dataset.resource));
    panel.querySelectorAll("[data-delete-resource]").forEach((button) => button.onclick = () => {
      const item = resources.find((row) => row.draft_resource_id === button.dataset.deleteResource);
      if (item && !item.system_resource_id && !isPublishedView()) removeResource(item);
    });
    const addResource = document.getElementById("add-resource-card");
    if (addResource) addResource.onclick = openResourceCatalog;
    bindResourceDrawer();
  }

  function bindResourceDrawer() {
    const drawer = document.getElementById("business-resource-drawer");
    drawer.onclose = () => { if (!drawer.open) closeResourceDrawer(); };
    drawer.oncancel = event => { event.preventDefault();closeResourceDrawer(); };
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
    if (snapshotState?.draft && !isPublishedView()) drawer.querySelectorAll("[data-snapshot-field]").forEach((input) => input.addEventListener(input.type === "checkbox" ? "change" : "input", () => {
      snapshotState.draft[input.dataset.snapshotField] = input.type === "checkbox" ? input.checked : input.dataset.snapshotField === "collection_mode" ? input.value : input.value === "" ? "" : Number(input.value);
      stageSnapshotPolicy();
      document.getElementById("snapshot-dirty").hidden = !item.workload_snapshot;
      if (input.dataset.snapshotField === "collection_mode") snapshotRender();
    }));
    const snapshotRefresh = document.getElementById("snapshot-refresh");
    if (snapshotRefresh) snapshotRefresh.onclick = () => void snapshotOperation();
    const snapshotStatusRefresh = document.getElementById("snapshot-status-refresh");
    if (snapshotStatusRefresh) snapshotStatusRefresh.onclick = () => void loadSnapshotPolicy();
    const snapshotReload = document.getElementById("snapshot-reload");
    if (snapshotReload) snapshotReload.onclick = () => void loadSnapshotPolicy();
    const editor = document.getElementById("resource-editor");
    if (!editor) return;
    editor.querySelectorAll("[name]").forEach((input) => input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => updateResourceField(item, input)));
    editor.querySelectorAll("[data-rac-node][data-rac-field]").forEach((input) => input.addEventListener(input.tagName === "SELECT" ? "change" : "input", () => {
      const node = resources.find((row) => row.draft_resource_id === input.dataset.racNode);
      if (!node?.database) return;
      connectionTestStates.delete(item.draft_resource_id);
      node.database[input.dataset.racField] = input.dataset.racField === "rac_node_number" ? Number(input.value) : input.value;
      if (input.dataset.racField === "candidate_ref") node.database.candidate_locator_key = node.database.candidates.find((row) => row.candidate_ref === input.value)?.locator_key || "";
      if (input.dataset.racField === "host_candidate_ref") node.database.host_candidate_locator_key = node.database.host_candidates.find((row) => row.candidate_ref === input.value)?.locator_key || "";
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
    document.getElementById("remove-resource").onclick = () => removeResource(item);
    editor.onsubmit = (event) => { event.preventDefault(); void validateResource(item); };
  }

  function updateResourceField(item, input) {
    const name = input.name;
    connectionTestStates.delete(item.draft_resource_id);
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
    item.database[key] = name === "tls_enabled" ? input.value === "true" : ["port", "rac_node_number"].includes(name) ? Number(input.value) : input.value;
    if (name === "db_type") {
      item.database.version_code = versions[item.database.db_type][0];
      item.database.port = item.database.db_type === "ORACLE" ? 1521 : item.database.db_type === "MYSQL" ? 3306 : 5432;
      if (item.database.db_type !== "ORACLE") {
        item.database.deployment_type = "SINGLE_INSTANCE";
        item.database.rac_node_number = null;
      }
      item.database.tls_enabled = false;
      syncInferredDatabaseRelations();
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
      syncInferredDatabaseRelations();
      item.saved = Boolean(item.system_resource_id); mark(); renderComposition(); return;
    }
    if (name === "source_id") {
      racMembers(item).forEach((member) => {
        if (!member.database) return;
        member.database.source_id = item.database.source_id;
        member.database.discovery_revision += 1;
        Object.assign(member.database, { locator_key: "", candidate_ref: "", candidate_locator_key: "", host_candidate_ref: "", host_candidate_locator_key: "", candidates: [], host_candidates: [] });
        member.saved = Boolean(member.system_resource_id);
      });
      mark(); renderComposition();
      if (item.database.source_id && usesDiscovery(item)) void discover(item);
      return;
    }
    if (name === "candidate_ref") {
      item.database.candidate_locator_key = item.database.candidates.find((row) => row.candidate_ref === input.value)?.locator_key || "";
    }
    if (name === "host_candidate_ref") {
      item.database.host_candidate_locator_key = item.database.host_candidates.find((row) => row.candidate_ref === input.value)?.locator_key || "";
    }
    if (isRacResource(item)) syncRacDatabaseField(item, key);
    if (name === "db_role") syncInferredDatabaseRelations();
    if (name === "oracle_scope") { item.saved = false; mark(); renderComposition(); return; }
    item.saved = Boolean(item.system_resource_id);
    if (["host", "port", "database", "oracle_pdb_name", "username", "password", "locator_key", "candidate_ref", "host_candidate_ref"].includes(name)) item.saved = false;
    mark();
  }

  function selectedDiscoveryLocators(database) {
    const candidate = database.candidates.find((row) => row.candidate_ref === database.candidate_ref);
    const hostCandidate = database.host_candidates.find((row) => row.candidate_ref === database.host_candidate_ref);
    return {
      candidate: database.candidate_locator_key || candidate?.locator_key || "",
      host: database.host_candidate_locator_key || hostCandidate?.locator_key || "",
    };
  }

  function applyDiscoveryResult(items, found, selections, requireSelections = false) {
    const candidates = (found.items || []).map((row) => ({
      ...row,
      display_name: `${row.display_name}${row.mapping_status === "MAPPED" ? "（已配置，可复用）" : ""}`,
    }));
    const hostCandidates = found.host_items || [];
    items.forEach((member) => {
      if (!member.database) return;
      const selected = selections.get(member.draft_resource_id) || { candidate: "", host: "" };
      const candidate = candidates.find((row) => row.locator_key === selected.candidate);
      const hostCandidate = hostCandidates.find((row) => row.locator_key === selected.host);
      if (requireSelections && !selected.candidate) throw new Error(`${member.display_name}：草稿缺少数据库监控 Target 的稳定定位信息，请重新选择监控 Target。`);
      if (requireSelections && selected.candidate && !candidate) throw new Error(`${member.display_name}：原数据库监控 Target 已不存在，请重新选择。`);
      if (requireSelections && sourceFor(member)?.source_type === "PROMETHEUS" && !selected.host) throw new Error(`${member.display_name}：草稿缺少主机监控 Target 的稳定定位信息，请重新选择监控 Target。`);
      if (requireSelections && selected.host && !hostCandidate) throw new Error(`${member.display_name}：原主机监控 Target 已不存在，请重新选择。`);
      Object.assign(member.database, {
        candidates: [...candidates],
        host_candidates: [...hostCandidates],
        candidate_ref: candidate?.candidate_ref || "",
        candidate_locator_key: candidate?.locator_key || "",
        host_candidate_ref: hostCandidate?.candidate_ref || "",
        host_candidate_locator_key: hostCandidate?.locator_key || "",
      });
    });
    return { candidates, hostCandidates };
  }

  async function discover(item, options = {}) {
    if (!item.database?.source_id) { shell.toast("请先选择监控源。"); return; }
    const members = racMembers(item);
    const selections = new Map(members.map((member) => [member.draft_resource_id, selectedDiscoveryLocators(member.database)]));
    const sourceId = item.database.source_id;
    const dbType = item.database.db_type;
    const revision = ++item.database.discovery_revision;
    members.forEach((member) => { if (member.database) member.database.discovering = true; });
    renderComposition();
    try {
      const found = await request(`/diagnostic-sources/${encodeURIComponent(sourceId)}/instance-discoveries`, { method: "POST", body: JSON.stringify({ db_types: [dbType], page_size: 100 }) });
      if (revision !== item.database.discovery_revision || sourceId !== item.database.source_id || dbType !== item.database.db_type) return;
      const { candidates, hostCandidates } = applyDiscoveryResult(members, found, selections, Boolean(options.requireSelections));
      if (!options.quiet) shell.toast(`发现 ${candidates.length} 个数据库候选、${hostCandidates.length} 个主机候选。`);
    } catch (error) {
      if (options.requireSelections) throw error;
      shell.toast(error.message);
    } finally {
      if (revision === item.database.discovery_revision) {
        members.forEach((member) => { if (member.database) member.database.discovering = false; });
        renderComposition();
      }
    }
  }

  async function refreshDiscoveryReferences() {
    const groups = new Map();
    resources.filter((item) => item.mode === "NEW" && usesDiscovery(item)).forEach((item) => {
      const key = `${item.database.source_id}:${item.database.db_type}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(item);
    });
    await Promise.all([...groups.values()].map(async (items) => {
      const sample = items[0];
      const selections = new Map(items.map((item) => [item.draft_resource_id, selectedDiscoveryLocators(item.database)]));
      const found = await request(`/diagnostic-sources/${encodeURIComponent(sample.database.source_id)}/instance-discoveries`, { method: "POST", body: JSON.stringify({ db_types: [sample.database.db_type], page_size: 100 }) });
      applyDiscoveryResult(items, found, selections, true);
    }));
  }

  function targetPayload(item) {
    const database = item.database;
    const oracle = database.db_type === "ORACLE";
    return { display_name: item.display_name.trim(), db_type: database.db_type, version_code: database.version_code, environment: database.environment, deployment_type: database.deployment_type, rac_node_number: database.rac_node_number, db_role: database.db_role, importance_level: Number(system.importance_level), responsibility_group_id: item.owner_group_id, readonly_connection_enabled: true, controlled_change_enabled: false, endpoint: { host: database.host.trim(), port: Number(database.port), service: oracle ? database.database.trim() : null, database: oracle ? null : database.database.trim(), tls_enabled: !oracle && database.tls_enabled === true }, oracle_container_scope: oracle ? database.oracle_scope : null, oracle_pdb_name: oracle && database.oracle_scope === "PDB" ? database.oracle_pdb_name.trim() : null, diagnostic_credential: { username: database.username, password: database.password }, capabilities: {} };
  }
  function monitoringPayload(item) {
    const database = item.database;
    const source = sourceFor(item);
    if (usesDiscovery(item)) return { source_id: database.source_id, candidate_ref: database.candidate_ref, host_candidate_ref: database.host_candidate_ref || null };
    return { source_id: database.source_id, source_binding: { source_id: database.source_id, source_locator_key: database.locator_key.trim(), source_locator: source?.source_type === "LOKI" ? { labels: { target_key: database.locator_key.trim() } } : { target_key: database.locator_key.trim() } } };
  }

  function resourceConfigurationError(item) {
    const members = isRacResource(item) ? racMembers(item) : [item];
    const database = item.database;
    const source = sourceFor(item);
    if (!database) return `${item.display_name || "未命名资源"}：缺少数据库配置。`;
    for (const member of members) {
      const label = isRacResource(item) ? `RAC 节点 ${member.database?.rac_node_number || "?"}` : (member.display_name || "未命名数据库");
      if (!member.display_name.trim()) return `${label}：请填写数据库名称。`;
      if (!member.owner_group_id) return `${label}：请选择技术责任组。`;
      if (!member.database?.host.trim()) return `${label}：请填写 Host。`;
      if (!member.database.database.trim()) return `${label}：请填写 ${database.db_type === "ORACLE" ? "Service Name" : "Database"}。`;
    }
    if (!database.username) return `${item.display_name || "数据库"}：请填写诊断用户名。`;
    if (!database.password) return `${item.display_name || "数据库"}：请填写诊断密码。`;
    if (!source) return `${item.display_name || "数据库"}：请选择监控源。`;
    if (isRacResource(item) && members.length < 2) return "Oracle RAC 至少需要两个节点。";
    if (database.db_type === "ORACLE" && database.oracle_scope === "PDB" && !database.oracle_pdb_name.trim()) return "PDB 接入必须填写 PDB 名称。";
    if (members.some((member) => member.database?.deployment_type === "RAC" && (!member.database.rac_node_number || member.database.rac_node_number < 1))) return "每个 RAC 节点都必须填写有效的节点编号。";
    for (const member of members) {
      const label = isRacResource(item) ? `RAC 节点 ${member.database?.rac_node_number || "?"}` : (member.display_name || "未命名数据库");
      if (usesDiscovery(member) && !member.database?.candidate_ref) return `${label}：未配置数据库监控 Target。`;
      if (usesDiscovery(member) && source.source_type === "PROMETHEUS" && !member.database?.host_candidate_ref) return `${label}：未配置主机监控 Target。`;
      if (!usesDiscovery(member) && !member.database?.locator_key.trim()) return `${label}：未填写监控定位键。`;
    }
    if (isRacResource(item)) {
      const uniqueFields = [
        ["节点编号", members.map((member) => member.database?.rac_node_number)],
        ["Host", members.map((member) => member.database?.host.trim())],
        ["Service Name", members.map((member) => member.database?.database.trim())],
      ];
      const duplicate = uniqueFields.find((entry) => new Set(entry[1]).size !== entry[1].length);
      if (duplicate) return `RAC 各节点的${duplicate[0]}不能重复。`;
    }
    return null;
  }

  async function validateResource(item) {
    const configurationError = resourceConfigurationError(item);
    if (configurationError) {
      connectionTestStates.set(item.draft_resource_id, { status: "FAILED", message: configurationError });
      renderComposition();
      return;
    }
    const members = isRacResource(item) ? racMembers(item) : [item];
    connectionTestStates.set(item.draft_resource_id, { status: "CHECKING", message: "正在测试数据库连接…" });
    saving = true; renderComposition(); updateFooter();
    try {
      for (const member of members) {
        const target = targetPayload(member);
        const response = await request("/targets/test-connection", { method: "POST", body: JSON.stringify({ db_type: target.db_type, version_code: target.version_code, endpoint: target.endpoint, oracle_container_scope: target.oracle_container_scope, oracle_pdb_name: target.oracle_pdb_name, diagnostic_credential: target.diagnostic_credential }) });
        if (!response.ok) {
          const messages = {
            TARGET_UNREACHABLE: "无法连接数据库，请检查地址、端口和 TLS 设置。",
            AUTH_FAILED: "数据库认证失败，请检查诊断用户名和密码。",
            TIMEOUT: "连接数据库超时，请检查网络和防火墙设置。",
          };
          throw new Error(`${member.display_name}：${messages[response.error_code] || response.error_code || "连接测试失败"}`);
        }
      }
      members.forEach((member) => { member.saved = true; });
      connectionTestStates.set(item.draft_resource_id, { status: "SUCCEEDED", message: isRacResource(item) ? `${members.length} 个 RAC 节点连接测试通过；正式提交前不会创建资源。` : "数据库连接测试通过；正式提交前不会创建资源。" });
      mark();
    } catch (error) {
      connectionTestStates.set(item.draft_resource_id, { status: "FAILED", message: error.message });
    }
    finally { saving = false; renderComposition(); updateFooter(); }
  }

  function addRelation() {
      const selectedUnit = databaseUnitForResource(activeResourceId);
      if (!selectedUnit) {
          shell.toast('请先在拓扑中选择一个数据库。');
          return;
      }
      const nativeRelationType = selectedUnit.db_type === 'ORACLE' ? 'DATA_GUARD' : selectedUnit.db_type === 'MYSQL' ? 'MYSQL_REPLICATION' : 'POSTGRES_STREAMING';
      const relationTypeCandidates = [nativeRelationType, ...databaseRelationTypes.map(item => item.value).filter(type => type !== nativeRelationType)];
      const candidates = databaseUnits().filter(unit => unit.unit_id !== selectedUnit.unit_id);
      const selectedIsTarget = selectedUnit.db_role === 'STANDBY';
      const preferredRole = selectedIsTarget ? 'PRIMARY' : 'STANDBY';
      const orderedCandidates = [...candidates].sort((left, right) => Number(right.db_role === preferredRole) - Number(left.db_role === preferredRole));
      const choices = relationTypeCandidates.flatMap(type => orderedCandidates.flatMap(counterpart => {
          const orientations = selectedIsTarget
              ? [{ source: counterpart, target: selectedUnit }, { source: selectedUnit, target: counterpart }]
              : [{ source: selectedUnit, target: counterpart }, { source: counterpart, target: selectedUnit }];
          const orientation = orientations.find(({ source, target }) => supportsRelationEndpoints(type, source, target));
          return orientation ? [{ type, ...orientation }] : [];
      }));
      if (!choices.length) {
          shell.toast('当前没有可与该数据库建立关系的兼容数据库。');
          return;
      }
      const choice = choices.find(({ type, source, target }) => !hasDatabaseRelation(type, source.unit_id, target.unit_id));
      if (!choice) {
          shell.toast('当前数据库与兼容数据库之间的关系均已存在，请直接选择关系进行修改。');
          return;
      }
      const relationType = choice.type;
      const sourceUnit = choice.source;
      const targetUnit = choice.target;
      const item = {
          draft_relation_id: uuid7(), origin: 'MANUAL', relation_type_key: relationType,
          display_name: generatedRelationName(relationType, sourceUnit, targetUnit), responsibility_group_id: '',
          participants: buildRelationParticipants(relationType, sourceUnit, targetUnit), attributes: relationDatabaseAttributes(sourceUnit, targetUnit),
      };
      relations.push(item);
      activeRelationId = item.draft_relation_id;
      activeResourceId = '';
      mark();
  }
  function changeRelationType(item, event) {
      const select = event.target;
      const relationType = select.value;
      const sourceUnit = databaseUnit(relationSourceUnitId(item));
      const targetUnit = databaseUnit(relationTargetUnitId(item));
      if (!sourceUnit || !targetUnit || !supportsRelationEndpoints(relationType, sourceUnit, targetUnit)) {
          select.value = item.relation_type_key;
          return;
      }
      if (hasDatabaseRelation(relationType, sourceUnit.unit_id, targetUnit.unit_id, item)) {
          select.value = item.relation_type_key;
          shell.toast('相同源端、目标端和类型的关系已经存在。');
          return;
      }
      makeRelationManual(item);
      item.relation_type_key = relationType;
      item.participants = buildRelationParticipants(relationType, sourceUnit, targetUnit);
      item.display_name = generatedRelationName(relationType, sourceUnit, targetUnit);
      item.attributes = { ...item.attributes, ...relationDatabaseAttributes(sourceUnit, targetUnit) };
      mark();
  }
  function changeRelationEndpoint(item, endpoint, event) {
      const select = event.target;
      const previousUnitId = endpoint === 'source' ? relationSourceUnitId(item) : relationTargetUnitId(item);
      const nextUnit = databaseUnit(select.value);
      const sourceUnit = endpoint === 'source' ? nextUnit : databaseUnit(relationSourceUnitId(item));
      const targetUnit = endpoint === 'target' ? nextUnit : databaseUnit(relationTargetUnitId(item));
      if (!sourceUnit || !targetUnit) {
          select.value = previousUnitId;
          return;
      }
      if (sourceUnit.unit_id === targetUnit.unit_id) {
          select.value = previousUnitId;
          shell.toast('源数据库和目标数据库不能相同。');
          return;
      }
      if (hasDatabaseRelation(item.relation_type_key, sourceUnit.unit_id, targetUnit.unit_id, item)) {
          select.value = previousUnitId;
          shell.toast('相同源端、目标端和类型的关系已经存在。');
          return;
      }
      makeRelationManual(item);
      item.participants = buildRelationParticipants(item.relation_type_key, sourceUnit, targetUnit);
      item.display_name = generatedRelationName(item.relation_type_key, sourceUnit, targetUnit);
      item.attributes = { ...item.attributes, ...relationDatabaseAttributes(sourceUnit, targetUnit) };
      mark();
  }

  function topologyBoardHtml() {
    return KBotAIOpsTopology.render({systemName:system.display_name || "未命名业务系统",resources,relations,selectedResourceId:activeResourceId,selectedRelationId:activeRelationId});
  }
  function relationEditorHtml(relation) {
    const title = `<header><div><p class="eyebrow">RELATION</p><h3>${esc(relationName(relation.relation_type_key))}</h3></div>${!isPublishedView() && !isManagedRelation(relation) ? '<button id="remove-relation" class="danger-link" type="button">删除关系</button>' : ""}</header>`;
    if (isManagedRelation(relation)) return title + `<p class="inferred-relation-note">RAC 关系不可在此修改。RAC 成员由 Step 2 的部署形态维护。</p><dl class="relation-readonly-summary"><div><dt>RAC 名称</dt><dd>${esc(relation.display_name || "未命名 RAC")}</dd></div><div><dt>成员节点</dt><dd>${relation.participants.length} 个</dd></div></dl><div class="rac-relation-members">${relation.participants.map(participant => {const item = resources.find(resource => resource.draft_resource_id === participant.draft_resource_id);return `<article><strong>${esc(item?.display_name || "未命名节点")}</strong><span>${esc(item?.database?.host || "Host 待填写")}</span></article>`;}).join("")}</div>`;
    if (isPublishedView()) return title + `<p class="inferred-relation-note">当前业务系统正在运行，关系配置仅供查看。</p><dl class="relation-readonly-summary"><div><dt>关系类型</dt><dd>${esc(relationName(relation.relation_type_key))}</dd></div><div><dt>连接方向</dt><dd>${esc(relationEndpointSummary(relation))}</dd></div><div><dt>来源</dt><dd>${isInferredRelation(relation) ? "系统自动推导" : "人工配置"}</dd></div></dl>`;
    const sourceId = relationSourceUnitId(relation), targetId = relationTargetUnitId(relation);
    const options = endpoint => compatibleDatabaseUnits(relation.relation_type_key,endpoint).map(unit => `<option value="${esc(unit.unit_id)}"${unit.unit_id === (endpoint === "source" ? sourceId : targetId) ? " selected" : ""}${unit.unit_id === (endpoint === "source" ? targetId : sourceId) ? " disabled" : ""}>${esc(unit.display_name)} · ${esc(unit.db_type)}</option>`).join("");
    return title + `${isInferredRelation(relation) ? '<p class="inferred-relation-note">系统已根据 Step 2 的数据库引擎和主备角色自动推导。若判断不准确，可直接调整；首次修改后将转为人工校正。</p>' : isInferenceOverride(relation) ? '<p class="inferred-relation-note relation-override-note"><span>此关系已由自动推导转为人工校正。</span><button id="restore-inferred-relation" class="secondary-button" type="button">恢复自动推导</button></p>' : ""}<div class="relation-sentence" aria-label="数据库关系编辑"><label><span>源数据库 / RAC</span><select data-relation-endpoint="source">${options("source")}</select></label><span>通过</span><label><span>关系类型</span><select id="relation-type">${relationTypeOptions(relation).map(option => `<option value="${esc(option.value)}"${option.value === relation.relation_type_key ? " selected" : ""}>${esc(option.label)}</option>`).join("")}</select></label><span>连接到</span><label><span>目标数据库 / RAC</span><select data-relation-endpoint="target">${options("target")}</select></label></div><p class="relation-editor-help">异构同步可选择 OGG、Debezium CDC、Flink CDC 或 DataX；选择 RAC 时会自动包含全部节点。</p>`;
  }
  function resourceSummaryHtml(item) {
    if (!item) return '<div class="empty-state"><h3>选择数据库或关系</h3><p>点击数据库查看已有关系，点击连线修改或删除关系。</p></div>';
    const unit = databaseUnitForResource(item.draft_resource_id), ids = new Set(unit?.resources.map(resource => resource.draft_resource_id) || []), linked = relations.filter(relation => relation.participants.some(participant => ids.has(participant.draft_resource_id)));
    return `<header><div><p class="eyebrow">DATABASE</p><h3>${esc(unit?.display_name)}</h3></div><span class="readonly-badge">资源只读</span></header><p class="resource-readonly-note">数据库配置只在 Step 2 维护；这里用于确认它参与的关系。</p><dl><div><dt>数据库引擎</dt><dd>${esc(unit?.db_type)} ${esc(unit?.resources[0]?.database?.version_code)}</dd></div><div><dt>部署形态</dt><dd>${unit?.is_rac ? `Oracle RAC · ${unit.resources.length} 节点` : "单实例"}</dd></div><div><dt>数据库角色</dt><dd>${esc(databaseRoleName(unit?.db_role || "UNKNOWN"))}</dd></div></dl><section class="resource-relations"><header><strong>已有关系</strong><span>${linked.length}</span></header>${linked.length ? `<div>${linked.map(relation => `<button class="" type="button" data-topology-relation="${esc(relation.draft_relation_id)}"><span><strong>${esc(relationName(relation.relation_type_key))}</strong><small>${esc(relation.relation_type_key === "ORACLE_RAC" ? `${relation.participants.length} 个 RAC 成员` : relationEndpointSummary(relation))}</small></span><em>${isManagedRelation(relation) ? "固定" : isPublishedView() ? "查看 →" : "编辑 →"}</em></button>`).join("")}</div>` : "<p>当前数据库尚未建立关系。</p>"}</section>`;
  }
  function renderTopology() {
    const relation = currentRelation(), resource = currentResource();
    panel.innerHTML = `<section class="topology-step"><div class="topology-layout">${topologyBoardHtml()}<aside class="topology-properties"><div class="topology-properties-body">${relation ? relationEditorHtml(relation) : resourceSummaryHtml(resource)}</div><footer class="topology-properties-actions"><span>${isPublishedView() ? "运行中业务系统仅供查看" : resource ? "Step 3 只调整关系" : "自动关系可人工校正"}</span>${isPublishedView() ? "" : `<div>${dismissedInferenceKeys.length ? '<button id="restore-dismissed-relations" class="secondary-button" type="button">恢复自动识别</button>' : ""}${resource ? '<button id="add-relation" class="primary-button" type="button">＋ 为该数据库添加关系</button>' : ""}</div>`}</footer></aside></div></section>`;
    bindTopology();KBotAIOpsTopology.mount();
  }
  function bindTopology() {
    panel.onclick = event => {
      const selectedResource = event.target.closest("[data-topology-resource]"), selectedRelation = event.target.closest("[data-topology-relation]");
      if (selectedResource) {activeResourceId = selectedResource.dataset.topologyResource;activeRelationId = "";renderTopology();return;}
      if (selectedRelation) {activeRelationId = selectedRelation.dataset.topologyRelation;activeResourceId = "";renderTopology();return;}
      if (isPublishedView()) return;
      if (event.target.closest("#add-relation")) addRelation();
      else if (event.target.closest("#remove-relation")) removeRelation(currentRelation());
      else if (event.target.closest("#restore-inferred-relation")) restoreInferredRelation(currentRelation());
      else if (event.target.closest("#restore-dismissed-relations")) restoreDismissedRelations();
      else return;
      renderTopology();
    };
    panel.onkeydown = event => {if (["Enter"," "].includes(event.key) && event.target.matches(".graph-edge-label")) {event.preventDefault();event.target.dispatchEvent(new MouseEvent("click",{bubbles:true}));}};
    panel.querySelectorAll("[data-relation-endpoint]").forEach(select => {select.onchange = event => {changeRelationEndpoint(currentRelation(),select.dataset.relationEndpoint,event);renderTopology();};});
    const type = panel.querySelector("#relation-type");if (type) type.onchange = event => {changeRelationType(currentRelation(),event);renderTopology();};
  }
  function publicationErrors() {
    const errors = [];
    if (!system.display_name.trim()) errors.push("业务信息：请填写业务系统名称。");
    if (!system.responsibility_group_id) errors.push("业务信息：请选择业务责任组。");
    if (!resources.length) errors.push("系统构成：至少接入一个资源。");
    const visited = new Set();
    resources.filter((item) => !item.system_resource_id).forEach((item) => {
      const key = item.rac_group_id || item.draft_resource_id;
      if (visited.has(key)) return;
      visited.add(key);
      const error = resourceConfigurationError(item);
      if (error) errors.push(error);
    });
    resources.forEach((item) => {
      const staged = item.workload_snapshot;
      if (!staged) return;
      const original = detail?.resources?.find((row) => row.system_resource_id === item.system_resource_id)?.resource?.database;
      const valid = ["BOUNDED_CANDIDATES", "VISIBLE_SET"].includes(staged.policy.collection_mode) && [[staged.policy.visible_statement_limit, 10, 500], [staged.policy.interval_minutes, 1, 60], [staged.policy.retention_days, 1, 365], [staged.policy.top_statement_limit, 10, 500]].every(([value, min, max]) => Number.isInteger(value) && value >= min && value <= max);
      if (!valid || !original || original.target_id !== staged.target_id || !Number.isInteger(staged.target_row_version) || staged.target_row_version < 1) errors.push(`${item.display_name}：历史 SQL 采集策略无效，请检查间隔、保留天数、候选数量及 Target。`);
    });
    relations.forEach((relation) => {
      const name = relation.display_name || relationName(relation.relation_type_key);
      if (relation.participants.length < 2) errors.push(`${name}：至少需要两个参与资源。`);
      else if (new Set(relation.participants.map((item) => item.draft_resource_id)).size !== relation.participants.length) errors.push(`${name}：参与资源不能重复。`);
    });
    return [...new Set(errors)];
  }
  function systemPayload() { return { display_name: system.display_name.trim(), environment: system.environment, importance_level: Number(system.importance_level), responsibility_group_id: system.responsibility_group_id, description: system.description.trim() || null }; }
  function resourcePayload(item) {
    return { draft_resource_id: item.draft_resource_id, mode: "NEW", resource_type_key: item.resource_type_key, display_name: item.display_name.trim(), identity_kind: "STABLE", owner_group_id: item.owner_group_id, architecture_layer_key: item.architecture_layer_key, responsibility_group_id: null, location_name: item.location_name.trim() || null, display_order: item.display_order, attributes: {}, database: { target: targetPayload(item), monitoring: monitoringPayload(item) } };
  }
  function relationPayload(item, systemIdByDraft = null) {
    return { relation_type_key: item.relation_type_key, display_name: item.display_name.trim() || null, responsibility_group_id: item.responsibility_group_id || null, origin: item.origin || (isInferredRelation(item) ? "DISCOVERY" : "MANUAL"), confirmation_status: "CONFIRMED", attributes: item.attributes || {}, participants: item.participants.map((participant) => systemIdByDraft ? { system_resource_id: systemIdByDraft.get(participant.draft_resource_id), participant_role: participant.participant_role, display_order: participant.display_order } : participant) };
  }
  function draftPayload() {
    return {
      system: { ...system },
      resources: resources.map((item) => ({ ...item, database: item.database ? { ...item.database, discovering: false } : undefined })),
      relations,
      dismissed_inference_keys: dismissedInferenceKeys,
      step: steps[step],
      furthest,
      publish: { system: systemPayload(), resources: resources.map(resourcePayload), relations: relations.map((item) => ({ draft_relation_id: item.draft_relation_id, ...relationPayload(item) })) },
    };
  }
  function clearStaleDraftReference() {
    draftId = "";
    draftRevision = 0;
    const currentUrl = new URL(location.href);
    currentUrl.searchParams.delete("draft");
    history.replaceState(null, "", currentUrl);
  }
  async function persistDraft(showToast = true) {
    if (isPublishedView()) throw new Error("运行中的业务系统仅支持查看。");
    const payload = draftPayload();
    const canUpdateDraft = Boolean(draftId) && Number.isInteger(draftRevision) && draftRevision >= 1;
    const saved = canUpdateDraft
      ? await request(`/business-system-drafts/${encodeURIComponent(draftId)}`, { method: "PUT", body: JSON.stringify({ revision: draftRevision, current_step: steps[step], payload }) })
      : await request("/business-system-drafts", { method: "POST", body: JSON.stringify({ mode: creating ? "CREATE" : "EDIT", business_system_id: creating ? null : systemId, current_step: steps[step], payload }) });
    draftId = saved.draft_id;
    draftRevision = Number(saved.revision);
    dirty = false;
    const currentUrl = new URL(location.href);
    currentUrl.searchParams.set("draft", draftId);
    history.replaceState(null, "", currentUrl);
    updateFooter();
    if (showToast) shell.toast("草稿已保存到服务器，可稍后继续编辑。");
    return saved;
  }

  async function updateSystem() {
    if (isPublishedView()) throw new Error("运行中的业务系统仅支持查看。");
    await request(`/business-systems/${encodeURIComponent(systemId)}`, { method: "PATCH", headers: { "If-Match": `"rv-${detail.row_version}"` }, body: JSON.stringify(systemPayload()) });
    const originalResources = new Map(detail.resources.map((item) => [item.system_resource_id, item]));
    const retainedSystemIds = new Set(resources.map((item) => item.system_resource_id).filter(Boolean));
    const retainedRelationIds = new Set(relations.map((item) => item.relation_id).filter(Boolean));
    for (const original of detail.relations) if (!retainedRelationIds.has(original.relation_id)) await request(`/business-systems/${encodeURIComponent(systemId)}/relations/${encodeURIComponent(original.relation_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${original.row_version}"` } });
    let current = await request(`/business-systems/${encodeURIComponent(systemId)}`);
    for (const item of resources) {
      if (item.system_resource_id) {
        await request(`/managed-resources/${encodeURIComponent(item.resource_id)}`, { method: "PATCH", headers: { "If-Match": `"rv-${item.resource_row_version}"` }, body: JSON.stringify({ display_name: item.display_name.trim(), owner_group_id: item.owner_group_id, location_name: item.location_name.trim() || null, database: item.database ? { version_code: item.database.version_code, environment: item.database.environment, deployment_type: item.database.deployment_type, rac_node_number: item.database.rac_node_number, db_role: item.database.db_role, ...(item.workload_snapshot ? { workload_snapshot_policy: item.workload_snapshot.policy, target_row_version: item.workload_snapshot.target_row_version, business_system_id: systemId, system_resource_id: item.system_resource_id } : {}) } : null }) });
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
    if (isPublishedView() || saving || snapshotState?.mutating) return;
    if (syncInferredDatabaseRelations()) mark();
    const errors = publicationErrors();
    if (errors.length) { shell.toast(errors.join("；")); return; }
    saving = true; updateFooter();
    try {
      await refreshDiscoveryReferences();
      let saved;
      if (creating) {
        const draft = await persistDraft(false);
        saved = await request(`/business-system-drafts/${encodeURIComponent(draft.draft_id)}:commit`, { method: "POST", headers: { "If-Match": `"rv-${draft.revision}"`, "Idempotency-Key": uuid7() }, body: JSON.stringify({}) });
      } else {
        saved = await updateSystem();
        if (draftId) await request(`/business-system-drafts/${encodeURIComponent(draftId)}`, { method: "DELETE", headers: { "If-Match": `"rv-${draftRevision}"` } });
      }
      draftId = "";
      draftRevision = 0;
      dirty = false; shell.toast(creating ? "业务系统资源已发布并上线。" : "业务系统资源已发布。");
      const publishedUrl = new URL(location.href);
      publishedUrl.searchParams.set("id", saved.business_system_id);
      publishedUrl.searchParams.delete("draft");
      if (creating) location.replace(publishedUrl.href);
      else {
        history.replaceState(null, "", publishedUrl);
        hydrate(saved); notifyValidationIssues(saved); render();
      }
    } catch (error) { shell.toast(error.message); }
    finally { saving = false; updateFooter(); }
  }
  async function saveDraft() {
    if (isPublishedView() || saving || snapshotState?.mutating) return;
    saving = true; updateFooter();
    try { await persistDraft(); }
    catch (error) { shell.toast(error.message); }
    finally { saving = false; updateFooter(); }
  }
  function hydrateDraft(value) {
    if (isPublishedView()) return;
    const payload = value.payload || {};
    draftId = value.draft_id;
    draftRevision = Number(value.revision);
    system = { ...system, ...(payload.system || {}) };
    resources = Array.isArray(payload.resources) ? payload.resources.map((item) => ({ ...item, database: normalizeDraftDatabase(item.database) })) : [];
    relations = Array.isArray(payload.relations) ? payload.relations : [];
    dismissedInferenceKeys = Array.isArray(payload.dismissed_inference_keys) ? payload.dismissed_inference_keys.filter(item => typeof item === "string") : [];
    const loadedStep = steps.indexOf(payload.step || value.current_step);
    step = loadedStep >= 0 ? loadedStep : 0;
    furthest = Math.max(Number(payload.furthest || 0), step);
    activeResourceId = resources[0]?.draft_resource_id || "";
    activeLayer = resources[0]?.architecture_layer_key || "DATA";
  }

  function hydrate(value) {
    dismissedInferenceKeys = [];
    detail = value;
    workspaceReadOnly = !creating && value.status === "ENABLED";
    system = { display_name: value.display_name, environment: value.environment, importance_level: value.importance_level, responsibility_group_id: value.responsibility_group_id, description: value.description || "" };
    const draftBySystem = new Map();
    resources = value.resources.filter((item) => item.status === "ACTIVE").map((item, index) => {
      const draftId = uuid7(); draftBySystem.set(item.system_resource_id, draftId);
      const database = item.resource.database;
      return { draft_resource_id: draftId, mode: "EXISTING", resource_id: item.resource.resource_id, system_resource_id: item.system_resource_id, resource_row_version: item.resource.row_version, system_row_version: item.row_version, resource_type_key: item.resource.resource_type_key, display_name: item.resource.display_name, owner_group_id: item.resource.owner_group_id, architecture_layer_key: item.architecture_layer_key, responsibility_group_id: item.responsibility_group_id || "", location_name: item.resource.location_name || "", display_order: item.display_order ?? index, saved: true, database: database ? { db_type: database.db_type, version_code: database.version_code, environment: database.environment, deployment_type: database.deployment_type, rac_node_number: database.rac_node_number ?? null, db_role: database.db_role, host: database.endpoint?.host || "", port: database.endpoint?.port || 0, database: database.endpoint?.service || database.endpoint?.database || "", oracle_scope: database.oracle_container_scope || "NON_CDB", oracle_pdb_name: database.oracle_pdb_name || "", username: database.diagnostic_username || "", password: "", source_id: "", locator_key: "", candidate_ref: "", host_candidate_ref: "", candidates: [], host_candidates: [], discovering: false, discovery_revision: 0, connectivity_status: database.connectivity_status, monitoring_ready: database.monitoring_ready } : undefined };
    });
    relations = value.relations.map((item) => ({ draft_relation_id: uuid7(), relation_id: item.relation_id, row_version: item.row_version, origin: item.origin || "MANUAL", relation_type_key: item.relation_type_key, display_name: item.display_name || "", responsibility_group_id: item.responsibility_group_id || "", attributes: item.attributes || {}, participants: item.participants.map((participant) => ({ draft_resource_id: draftBySystem.get(participant.system_resource_id) || "", participant_role: participant.participant_role, display_order: participant.display_order })) }));
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
    const published = isPublishedView();
    const editable = !loading && !published;
    document.querySelector(".workspace-actions").hidden = loading;
    document.getElementById("step-copy").textContent = published ? "业务系统运行中，当前为只读查看" : dirty ? "有未保存修改" : draftId ? "草稿已保存到服务器" : "尚未保存草稿";
    document.getElementById("save-draft").hidden = !editable;
    document.getElementById("save-draft").disabled = saving || Boolean(snapshotState?.mutating);
    document.getElementById("previous").hidden = !editable || step === 0;
    document.getElementById("next").hidden = !editable || step === 2;
    document.getElementById("commit").hidden = !editable || step !== 2;
    document.getElementById("commit").disabled = saving || Boolean(snapshotState?.mutating);
    document.getElementById("commit").textContent = saving ? "正在发布…" : "发布资源";
  }
  function render() {
    if (step !== 2) { KBotAIOpsTopology.dispose();panel.onclick = null;panel.onkeydown = null; }
    document.querySelectorAll("#wizard-steps [data-step]").forEach((button, index) => { button.classList.toggle("active", index === step); button.classList.toggle("done", index < step); button.disabled = !isPublishedView() && index > furthest; });
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
      if (draftId && isPublishedView()) {
        clearStaleDraftReference();
        shell.toast("业务系统正在运行，已切换为只读查看。");
      } else if (draftId) {
        try {
          hydrateDraft(await request(`/business-system-drafts/${encodeURIComponent(draftId)}`));
        } catch (error) {
          if (error.status !== 404) throw error;
          clearStaleDraftReference();
          shell.toast("原草稿不存在或已失效，已切换为新草稿，请重新保存。");
        }
      }
      dirty = false;
    } catch (error) { shell.toast(error.message); }
    finally { loading = false; render(); }
  }

  document.getElementById("previous").onclick = () => { if (step > 0) { step -= 1; render(); } };
  document.getElementById("next").onclick = () => { if (step < 2) { if (step === 1 && syncInferredDatabaseRelations()) mark(); step += 1; furthest = Math.max(furthest, step); render(); } };
  document.getElementById("commit").onclick = () => void commit();
  document.getElementById("save-draft").onclick = () => void saveDraft();
  document.querySelectorAll("#wizard-steps [data-step]").forEach((button) => button.onclick = () => { const next = Number(button.dataset.step); if (isPublishedView() || next <= furthest) { if (!isPublishedView() && next === 2 && syncInferredDatabaseRelations()) mark(); step = next; render(); } });
  document.addEventListener("click", (event) => {
    const anchor = event.target.closest?.("a[href]");
    if (anchor && snapshotState?.mutating) { event.preventDefault(); shell.toast("采集配置操作正在进行，请稍候。"); return; }
    if (!dirty || !anchor || event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || anchor.target === "_blank" || anchor.hasAttribute("download")) return;
    const destination = new URL(anchor.href, location.href);
    if (destination.href === location.href || (destination.origin === location.origin && destination.pathname === location.pathname && destination.search === location.search)) return;
    event.preventDefault();
    pendingLeaveHref = destination.href;
    const dialog = document.getElementById("workspace-leave-dialog");
    if (!dialog.open) dialog.showModal();
    document.getElementById("cancel-workspace-leave").focus();
  });
  document.getElementById("workspace-leave-form").addEventListener("submit", (event) => {
    event.preventDefault();
    const destination = pendingLeaveHref;
    pendingLeaveHref = "";
    dirty = false;
    snapshotGeneration += 1;
    snapshotState = null;
    document.getElementById("workspace-leave-dialog").close();
    if (destination) location.assign(destination);
  });
  document.getElementById("cancel-workspace-leave").onclick = () => { pendingLeaveHref = ""; document.getElementById("workspace-leave-dialog").close(); };
  document.getElementById("workspace-leave-dialog").addEventListener("cancel", (event) => { event.preventDefault(); pendingLeaveHref = ""; event.currentTarget.close(); });
  addEventListener("beforeunload", (event) => { if (dirty || snapshotState?.mutating) event.preventDefault(); });
  addEventListener("pagehide",() => { KBotAIOpsTopology.dispose();snapshotGeneration += 1;snapshotState = null; });
  shell.ready.then(load).catch((error) => shell.toast(error.message));
})();
