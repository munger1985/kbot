(function () {
  "use strict";
  const esc = value => KBotAIOpsShell.escape(value ?? "");
  const unique = values => [...new Set(values)];
  const relationLabels = {ORACLE_RAC:"Oracle RAC",DATA_GUARD:"Data Guard",OGG:"GoldenGate",DEBEZIUM_CDC:"Debezium CDC",FLINK_CDC:"Flink CDC",DATAX:"DataX",MYSQL_REPLICATION:"MySQL 主从复制",POSTGRES_STREAMING:"PostgreSQL 流复制",RUNS_ON:"运行于",USES_STORAGE:"使用存储",DEPENDS_ON:"依赖",LOAD_BALANCES:"负载均衡"};
  const roles = {MEMBER:"RAC 成员",PRIMARY:"主库",STANDBY:"备库",SOURCE:"源端",TARGET:"目标端",REPLICA:"从库",WORKLOAD:"工作负载",HOST:"承载主机",CONSUMER:"使用方",PROVIDER:"提供方",BALANCER:"负载均衡",BACKEND:"后端服务"};
  const kinds = {DATABASE:"DB",PHYSICAL_SERVER:"SRV",VIRTUAL_MACHINE:"VM",CONTAINER:"CTR",POD:"POD",KUBERNETES_CLUSTER:"K8S",APPLICATION_SERVICE:"APP",APPLICATION_SERVER:"MID",MESSAGE_QUEUE:"MQ",LOAD_BALANCER:"LB",NETWORK_DEVICE:"NET",STORAGE:"STG"};
  let observer = null, frame = null, model = null;
  function dispose() {observer?.disconnect();observer = null;if (frame !== null) cancelAnimationFrame(frame);frame = null;model = null;}
  function graphModel(resources,relations) {
    const byId = new Map(resources.map(item => [item.draft_resource_id,item])), resourceNodes = new Map(), rac = relations.filter(item => item.relation_type_key === "ORACLE_RAC");
    for (const relation of rac) for (const participant of relation.participants) resourceNodes.set(participant.draft_resource_id,`cluster:${relation.draft_relation_id}`);
    for (const resource of resources) if (!resourceNodes.has(resource.draft_resource_id)) resourceNodes.set(resource.draft_resource_id,`resource:${resource.draft_resource_id}`);
    const edges = relations.filter(item => item.relation_type_key !== "ORACLE_RAC").map(relation => {
      const participants = [...relation.participants].sort((a,b) => Number(a.display_order || 0) - Number(b.display_order || 0));
      const sources = participants.filter(item => ["PRIMARY","SOURCE","WORKLOAD","CONSUMER","BALANCER"].includes(item.participant_role)), effective = sources.length ? sources : participants.slice(0,1), ids = new Set(effective.map(item => item.draft_resource_id));
      return {relation,sourceIds:unique(effective.map(item => resourceNodes.get(item.draft_resource_id) || "")),targetIds:unique(participants.filter(item => !ids.has(item.draft_resource_id)).map(item => resourceNodes.get(item.draft_resource_id) || ""))};
    }).filter(item => item.sourceIds.length && item.targetIds.length);
    const nodes = [], clustered = new Set();
    for (const relation of rac) {
      const members = [...relation.participants].sort((a,b) => Number(a.display_order || 0) - Number(b.display_order || 0)).map(item => byId.get(item.draft_resource_id)).filter(Boolean);members.forEach(item => clustered.add(item.draft_resource_id));nodes.push({id:`cluster:${relation.draft_relation_id}`,relation,resources:members});
    }
    for (const resource of resources.filter(item => !clustered.has(item.draft_resource_id))) nodes.push({id:`resource:${resource.draft_resource_id}`,resources:[resource]});
    const ranks = new Map(nodes.map(item => [item.id,0])), indegree = new Map(nodes.map(item => [item.id,0])), next = new Map(nodes.map(item => [item.id,new Set()]));
    for (const edge of edges) for (const source of edge.sourceIds) for (const target of edge.targetIds) {
      if (source === target || !next.has(source) || !indegree.has(target) || next.get(source).has(target)) continue;next.get(source).add(target);indegree.set(target,indegree.get(target) + 1);
    }
    const queue = nodes.filter(item => indegree.get(item.id) === 0).map(item => item.id), visited = new Set();
    while (queue.length) {const id = queue.shift();visited.add(id);for (const target of next.get(id)) {ranks.set(target,Math.max(ranks.get(target),ranks.get(id) + 1));indegree.set(target,indegree.get(target) - 1);if (indegree.get(target) === 0) queue.push(target);}}
    for (const node of nodes) node.rank = visited.has(node.id) ? ranks.get(node.id) : 0;
    const related = new Set([...rac.map(item => `cluster:${item.draft_relation_id}`),...edges.flatMap(item => [...item.sourceIds,...item.targetIds])]);
    return {edges,network:nodes.filter(item => related.has(item.id)).sort((a,b) => a.rank - b.rank),standalone:nodes.filter(item => !related.has(item.id))};
  }
  function render({systemName,resources,relations,selectedResourceId,selectedRelationId}) {
    dispose();model = {...graphModel(resources,relations),selectedRelationId};
    const resourceRole = id => unique(relations.flatMap(relation => relation.participants.filter(item => item.draft_resource_id === id && item.participant_role !== "MEMBER").map(item => roles[item.participant_role] || item.participant_role))).join(" / ") || "独立资源";
    const resourceButton = (resource,role) => {
      const database = resource.database, summary = database ? `${database.db_type} ${database.version_code} · ${database.deployment_type === "RAC" ? `RAC 节点 ${database.rac_node_number || "-"}` : "单实例"}` : resource.resource_type_key || "资源";
      return `<button type="button" class="topology-node${resource.draft_resource_id === selectedResourceId ? " active" : ""}${!resource.saved ? " pending" : ""}" data-topology-resource="${esc(resource.draft_resource_id)}"><span class="node-kind">${esc(kinds[resource.resource_type_key] || "RES")}</span><span><small>${esc(role)}</small><strong>${esc(resource.display_name || "未命名资源")}</strong><em>${esc(summary)}</em></span></button>`;
    };
    const network = model.network.length ? `<div class="graph-stage"><svg class="graph-edges" aria-hidden="true"></svg><div class="graph-node-grid" style="grid-template-columns:repeat(${Math.max(1,...model.network.map(item => item.rank + 1))},minmax(220px,1fr))">${model.network.map(node => `<article class="graph-node${node.relation ? " cluster" : ""}${node.relation?.draft_relation_id === selectedRelationId ? " active" : ""}" data-graph-node="${esc(node.id)}" style="grid-column:${node.rank + 1}">${node.relation ? `<header><button class="" type="button" data-topology-relation="${esc(node.relation.draft_relation_id)}"><span>${esc(relationLabels[node.relation.relation_type_key] || node.relation.relation_type_key)}</span><strong>${esc(node.relation.display_name || relationLabels[node.relation.relation_type_key] || node.relation.relation_type_key)}</strong></button><em>${node.resources.length} 节点 · ${node.relation.relation_type_key === "ORACLE_RAC" || node.relation.origin === "DISCOVERY" || node.relation.attributes?.auto_inferred === true ? "自动推导" : "手工关系"}</em></header>` : ""}<div class="cluster-members${node.relation ? "" : " single"}">${node.resources.map(resource => resourceButton(resource,`${node.relation ? "RAC 成员 · " : ""}${resourceRole(resource.draft_resource_id)}`)).join("")}</div></article>`).join("")}</div></div>` : "";
    const standalone = model.standalone.length ? `<section class="standalone-strip" aria-label="独立节点"><header><div><strong>独立节点</strong><span>尚未建立任何关系的资源</span></div><em>${model.standalone.length}</em></header><div>${model.standalone.map(node => resourceButton(node.resources[0],"独立资源")).join("")}</div></section>` : "";
    return `<section class="topology-canvas" aria-label="业务系统拓扑预览"><header class="topology-heading"><div><span>TOPOLOGY PREVIEW</span><strong>${esc(systemName)}</strong></div><small>${resources.length} 个资源 · ${relations.length} 个关系</small></header>${resources.length ? `<div class="topology-content"><section class="unified-graph" aria-label="统一资源关系图">${network}${standalone}</section><p class="topology-hint">同一资源在画布中只出现一次；自动关系来自 Step 2 的部署形态和数据库角色，尚无关系的资源统一排列在画布下方。</p></div>` : '<div class="topology-empty"><h3>拓扑将在接入资源后自动生成</h3><p>先返回“系统构成”添加数据库或其他运维资源。</p></div>'}</section>`;
  }
  function updateEdges() {
    const stage = document.querySelector(".graph-stage"), svg = stage?.querySelector(".graph-edges");if (!stage || !svg || !model) return;
    const rect = stage.getBoundingClientRect(), elements = new Map([...stage.querySelectorAll("[data-graph-node]")].map(node => [node.dataset.graphNode,node]));
    const curve = (fromX,fromY,toX,toY) => {const direction = toX >= fromX ? 1 : -1,bend = Math.max(36,Math.abs(toX - fromX) * 0.42);return `M ${fromX} ${fromY} C ${fromX + bend * direction} ${fromY}, ${toX - bend * direction} ${toY}, ${toX} ${toY}`;};
    svg.innerHTML = '<defs><marker id="topology-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M 0 0 L 8 4 L 0 8 z" fill="var(--accent)"/></marker></defs>' + model.edges.map(edge => {
      const sources = edge.sourceIds.map(id => elements.get(id)).filter(Boolean),targets = edge.targetIds.map(id => elements.get(id)).filter(Boolean);if (!sources.length || !targets.length) return "";
      const sourceAnchors = sources.map(node => {const box = node.getBoundingClientRect();return {x:box.right - rect.left,y:box.top - rect.top + box.height / 2};}),targetAnchors = targets.map(node => {const box = node.getBoundingClientRect();return {x:box.left - rect.left - 8,y:box.top - rect.top + box.height / 2};});
      const sourceX = Math.max(...sourceAnchors.map(point => point.x)),targetX = Math.min(...targetAnchors.map(point => point.x)),labelX = (sourceX + targetX) / 2,junctionY = (sourceAnchors.reduce((sum,point) => sum + point.y,0) / sourceAnchors.length + targetAnchors.reduce((sum,point) => sum + point.y,0) / targetAnchors.length) / 2;
      return `<g class="graph-edge${edge.relation.draft_relation_id === model.selectedRelationId ? " active" : ""}">${sourceAnchors.map(point => `<path d="${curve(point.x,point.y,labelX,junctionY)}"/>`).join("")}${targetAnchors.map(point => `<path d="${curve(labelX,junctionY,point.x,point.y)}" marker-end="url(#topology-arrow)"/>`).join("")}<g class="graph-edge-label" transform="translate(${labelX},${junctionY - 30})" data-topology-relation="${esc(edge.relation.draft_relation_id)}" role="button" tabindex="0" aria-label="${esc(relationLabels[edge.relation.relation_type_key] || edge.relation.relation_type_key)}"><rect x="-56" y="-17" width="112" height="34" rx="17"/><text text-anchor="middle" dominant-baseline="central">${esc(relationLabels[edge.relation.relation_type_key] || edge.relation.relation_type_key)}</text></g></g>`;
    }).join("");
  }
  function mount() {const stage = document.querySelector(".graph-stage");if (stage) {observer = new ResizeObserver(updateEdges);observer.observe(stage);}frame = requestAnimationFrame(updateEdges);}
  globalThis.KBotAIOpsTopology = {render,mount,dispose};
})();
