// 自动同步自 Ammolite topologyLayout.ts；请修改源文件后重新生成。
(function () {
const exports = {};
const require = () => ({ default: globalThis.KBotAIOpsTopologyLayout });
"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.hostTypes = exports.resourceKinds = exports.relationLabels = void 0;
exports.relationLabel = relationLabel;
exports.databaseSummary = databaseSummary;
exports.hostProjection = hostProjection;
exports.graphModel = graphModel;
exports.relationLabels = { ORACLE_RAC: 'Oracle RAC', DATA_GUARD: 'Data Guard', OGG: 'GoldenGate', DEBEZIUM_CDC: 'Debezium CDC', FLINK_CDC: 'Flink CDC', DATAX: 'DataX', MYSQL_REPLICATION: 'MySQL 主从复制', POSTGRES_STREAMING: 'PostgreSQL 流复制', RUNS_ON: '运行于', USES_STORAGE: '使用存储', DEPENDS_ON: '依赖', LOAD_BALANCES: '负载均衡' };
exports.resourceKinds = { DATABASE: 'DB', PHYSICAL_SERVER: 'SRV', VIRTUAL_MACHINE: 'VM', CONTAINER: 'CTR', POD: 'POD', KUBERNETES_CLUSTER: 'K8S', APPLICATION_SERVICE: 'APP', APPLICATION_SERVER: 'MID', MESSAGE_QUEUE: 'MQ', LOAD_BALANCER: 'LB', NETWORK_DEVICE: 'NET', STORAGE: 'STG' };
exports.hostTypes = new Set(['PHYSICAL_SERVER', 'VIRTUAL_MACHINE', 'CONTAINER', 'POD', 'KUBERNETES_CLUSTER']);
const replicationTypes = new Set(['DATA_GUARD', 'OGG', 'DEBEZIUM_CDC', 'FLINK_CDC', 'DATAX', 'MYSQL_REPLICATION', 'POSTGRES_STREAMING']);
const sourceRoles = new Set(['PRIMARY', 'SOURCE', 'WORKLOAD', 'CONSUMER', 'BALANCER']);
const roleLabels = { MEMBER: 'RAC 成员', PRIMARY: '主库', STANDBY: '备库', SOURCE: '源端', TARGET: '目标端', REPLICA: '从库', WORKLOAD: '工作负载', HOST: '承载主机', CONSUMER: '使用方', PROVIDER: '提供方', BALANCER: '负载均衡', BACKEND: '后端服务' };
const unique = (values) => [...new Set(values)];
function relationLabel(key) { return exports.relationLabels[key] || key; }
function databaseSummary(resource) {
    if (!resource.database)
        return resource.business_system_names?.join(' / ') || { PHYSICAL_SERVER: '物理服务器', VIRTUAL_MACHINE: '虚拟机', APPLICATION_SERVICE: '应用服务', STORAGE: '存储', NETWORK_DEVICE: '网络设备', CONTAINER: '容器', POD: 'Pod' }[resource.resource_type_key] || resource.resource_type_key;
    const db = resource.database;
    return `${db.db_type} ${db.version_code} · ${db.deployment_type === 'RAC' ? `RAC 实例 ${db.rac_node_number || '-'}` : '单实例'}`;
}
function hostProjection(value) {
    return {
        resources: value.resources.map(resource => ({ ...resource, draft_resource_id: resource.resource_id, saved: true, business_system_names: unique(resource.business_systems.map(system => system.display_name)) })),
        relations: value.relations.map(relation => ({ ...relation, draft_relation_id: relation.relation_id, participants: relation.participants.map(item => ({ ...item, draft_resource_id: item.resource_id })) })),
    };
}
function graphModel(resources, relations, view = 'deployment') {
    const byId = new Map(resources.map(resource => [resource.draft_resource_id, resource]));
    const valid = relations.map(relation => ({ ...relation, participants: relation.participants.filter(p => byId.has(p.draft_resource_id)) })).filter(relation => relation.participants.length > 1);
    const rac = valid.filter(relation => relation.relation_type_key === 'ORACLE_RAC');
    const resourceToVisualNode = new Map();
    for (const relation of rac)
        for (const member of relation.participants) {
            if (!resourceToVisualNode.has(member.draft_resource_id))
                resourceToVisualNode.set(member.draft_resource_id, `cluster:${relation.draft_relation_id}`);
        }
    const related = new Set(valid.flatMap(relation => relation.participants.map(p => p.draft_resource_id)));
    const standalone = resources.filter(resource => !related.has(resource.draft_resource_id));
    const network = resources.filter(resource => related.has(resource.draft_resource_id));
    const hosting = valid.filter(relation => relation.relation_type_key === 'RUNS_ON');
    const parentCandidates = new Map();
    for (const relation of hosting) {
        const hosts = relation.participants.filter(p => p.participant_role === 'HOST');
        for (const member of relation.participants.filter(p => p.participant_role === 'WORKLOAD')) {
            const parents = parentCandidates.get(member.draft_resource_id) || new Set();
            hosts.forEach(host => { if (exports.hostTypes.has(byId.get(host.draft_resource_id).resource_type_key))
                parents.add(host.draft_resource_id); });
            parentCandidates.set(member.draft_resource_id, parents);
        }
    }
    const parents = new Map();
    for (const [id, values] of parentCandidates)
        if (values.size === 1 && !values.has(id))
            parents.set(id, [...values][0]);
    // 环路和多承载不强行嵌套，仍保留真实的关系连线。
    for (const id of [...parents.keys()]) {
        const seen = new Set([id]);
        let cursor = parents.get(id);
        while (cursor) {
            if (seen.has(cursor)) {
                parents.delete(id);
                break;
            }
            ;
            seen.add(cursor);
            cursor = parents.get(cursor);
        }
    }
    const childrenOf = (id) => [...parents].filter(([, parent]) => parent === id).map(([child]) => child);
    const domainsOf = (id) => {
        const children = childrenOf(id);
        const own = resourceToVisualNode.get(id);
        return unique([...(own ? [own] : !children.length ? ['unclustered'] : []), ...children.flatMap(domainsOf)]);
    };
    if (view === 'deployment') {
        // 共享主机独立展示，不能为了集群外框重复主机或把它错误归某个数据库独占。
        const sharedParents = unique([...parents.values()]).filter(id => domainsOf(id).length > 1);
        for (const [id, parent] of [...parents])
            if (sharedParents.includes(parent))
                parents.delete(id);
    }
    const hostHeaderHeight = 86, hostChildGap = 16;
    const makeTree = (id) => {
        const children = childrenOf(id).map(makeTree);
        if (!children.length)
            return { id, children, width: 264, height: 84 };
        return { id, children, width: Math.max(296, ...children.map(child => child.width + 32)), height: hostHeaderHeight + children.reduce((height, child) => height + child.height + hostChildGap, 0) };
    };
    const roots = network.filter(resource => !parents.has(resource.draft_resource_id)).map(resource => makeTree(resource.draft_resource_id));
    const treeIds = (tree) => [tree.id, ...tree.children.flatMap(treeIds)];
    const siteOf = (trees) => {
        const locations = trees.flatMap(treeIds).map(id => byId.get(id)?.location_name?.trim() || '');
        return locations.length && locations.every(value => value && value === locations[0]) ? locations[0] : '';
    };
    const rowOf = (tree) => {
        const values = treeIds(tree).map(id => byId.get(id));
        if (values.some(resource => resource.resource_type_key === 'DATABASE'))
            return 1;
        const type = byId.get(tree.id).resource_type_key;
        if (values.some(resource => ['APPLICATION_SERVICE', 'APPLICATION_SERVER', 'MESSAGE_QUEUE', 'LOAD_BALANCER'].includes(resource.resource_type_key)))
            return 0;
        if (type === 'STORAGE')
            return 3;
        return 2;
    };
    const units = [];
    const assigned = new Set();
    for (const relation of rac) {
        const id = `cluster:${relation.draft_relation_id}`;
        const members = view === 'deployment' ? roots.filter(root => {
            const owners = domainsOf(root.id);
            return owners.length === 1 && owners[0] === id;
        }) : [];
        members.forEach(root => assigned.add(root.id));
        const columns = Math.min(3, Math.max(1, members.length));
        const cellWidth = Math.max(264, ...members.map(root => root.width));
        let height = 84;
        for (let offset = 0; offset < members.length; offset += columns)
            height += Math.max(...members.slice(offset, offset + columns).map(root => root.height)) + 22;
        units.push({ id, roots: members, relation, row: 1, site: siteOf(members), width: view === 'host' ? 320 : columns * cellWidth + (columns - 1) * 20 + 40, height: view === 'host' ? 96 : height + 4, summary: view === 'host' });
    }
    for (const root of roots.filter(tree => !assigned.has(tree.id))) {
        units.push({ id: `resource:${root.id}`, roots: [root], row: view === 'host' && treeIds(root).some(id => parents.has(id)) ? 2 : rowOf(root), site: siteOf([root]), width: root.width, height: root.height });
    }
    // 主备关系只决定同层左右次序，不参与应用、主机、存储的纵向分层。
    const logical = valid.filter(relation => replicationTypes.has(relation.relation_type_key));
    const logicalRank = new Map(units.map(unit => [unit.id, 0]));
    for (let pass = 0; pass < units.length; pass++) {
        let changed = false;
        for (const relation of logical) {
            const sources = relation.participants.filter(p => sourceRoles.has(p.participant_role));
            const targets = relation.participants.filter(p => !sourceRoles.has(p.participant_role));
            for (const source of sources)
                for (const target of targets) {
                    const from = resourceToVisualNode.get(source.draft_resource_id) || `resource:${source.draft_resource_id}`;
                    const to = resourceToVisualNode.get(target.draft_resource_id) || `resource:${target.draft_resource_id}`;
                    if (from === to || !logicalRank.has(from) || !logicalRank.has(to))
                        continue;
                    const rank = Math.min(units.length, (logicalRank.get(from) || 0) + 1);
                    if (rank > (logicalRank.get(to) || 0)) {
                        logicalRank.set(to, rank);
                        changed = true;
                    }
                }
        }
        if (!changed)
            break;
    }
    units.sort((a, b) => a.row - b.row || (logicalRank.get(a.id) || 0) - (logicalRank.get(b.id) || 0));
    const boxes = [], resourceNodes = [], anchors = new Map(), rows = [];
    const rolesFor = (id) => {
        const cluster = rac.find(relation => relation.participants.some(p => p.draft_resource_id === id));
        const roles = unique(valid.flatMap(relation => relation.participants.filter(p => p.draft_resource_id === id && p.participant_role !== 'MEMBER').map(p => roleLabels[p.participant_role] || p.participant_role)));
        return [cluster ? `${cluster.display_name || 'Oracle RAC'} · 成员` : '', ...roles].filter(Boolean).join(' / ') || '资源';
    };
    const placeTree = (tree, x, y) => {
        const resource = byId.get(tree.id), nested = tree.children.length > 0;
        anchors.set(`resource:${tree.id}`, { x, y, width: tree.width, height: tree.height });
        if (nested)
            boxes.push({ id: `host:${tree.id}`, kind: 'host', title: '', subtitle: '', x, y, width: tree.width, height: tree.height, resourceId: tree.id });
        resourceNodes.push({ resource, role: rolesFor(tree.id), host: nested, x: x + (nested ? 12 : 0), y: y + (nested ? 10 : 0), width: tree.width - (nested ? 24 : 0), height: nested ? 64 : 84 });
        let top = y + hostHeaderHeight;
        for (const child of tree.children) {
            placeTree(child, x + 16, top);
            top += child.height + hostChildGap;
        }
    };
    let top = 52, width = 0;
    const siteOrder = unique(units.map(unit => unit.site).filter(Boolean));
    const siteWidths = new Map(siteOrder.map(site => [site, Math.max(...[0, 1, 2, 3].map(row => {
            const batch = units.filter(unit => unit.site === site && unit.row === row);
            return batch.reduce((sum, unit) => sum + unit.width, 0) + Math.max(0, batch.length - 1) * 180 + 40;
        }))]));
    const siteLefts = new Map();
    let columnLeft = 28;
    for (const site of siteOrder) {
        siteLefts.set(site, columnLeft);
        columnLeft += siteWidths.get(site) + 180;
    }
    const rowNames = view === 'host' ? ['应用与接入', '数据库与集群', '主机与运行实例', '存储与基础设施'] : ['应用与接入', '数据库部署', '共享承载与基础设施', '存储'];
    for (let row = 0; row < 4; row++) {
        const current = units.filter(unit => unit.row === row);
        if (!current.length)
            continue;
        rows.push({ y: top - 22, title: rowNames[row] });
        let left = 28;
        const sites = [...siteOrder.filter(site => current.some(unit => unit.site === site)), ...(current.some(unit => !unit.site) ? [''] : [])];
        const siteGroups = sites.map(site => ({ site, units: current.filter(unit => unit.site === site) }));
        for (const group of siteGroups) {
            left = group.site ? siteLefts.get(group.site) : sites.some(Boolean) ? columnLeft : 28;
            const start = left, startTop = top, sitePadding = group.site ? 20 : 0, siteTitle = group.site ? 42 : 0;
            let siteHeight = 0;
            left += sitePadding;
            for (const unit of group.units) {
                const y = top + siteTitle;
                anchors.set(unit.id, { x: left, y, width: unit.width, height: unit.height });
                if (unit.relation) {
                    const names = unit.relation.participants.map(p => byId.get(p.draft_resource_id)).filter(Boolean);
                    const dbRoles = unique(names.map(resource => resource.database?.db_role).filter(Boolean));
                    const origin = unit.relation.origin === 'DISCOVERY' || unit.relation.attributes?.auto_inferred === true ? '自动推导' : '手工关系';
                    boxes.push({ id: unit.id, kind: 'cluster', title: unit.relation.display_name || 'Oracle RAC', subtitle: `${names.length} 个实例 · ${dbRoles.map(role => roleLabels[role] || role).join(' / ') || '角色待识别'} · ${origin}${unit.summary ? ' · 成员在主机内' : ''}`, relation: unit.relation, x: left, y, width: unit.width, height: unit.height });
                    const columns = Math.min(3, Math.max(1, unit.roots.length)), cellWidth = Math.max(264, ...unit.roots.map(root => root.width));
                    let treeTop = y + 84;
                    for (let offset = 0; offset < unit.roots.length; offset += columns) {
                        const batch = unit.roots.slice(offset, offset + columns);
                        batch.forEach((root, i) => placeTree(root, left + 20 + i * (cellWidth + 20), treeTop));
                        treeTop += Math.max(...batch.map(root => root.height)) + 22;
                    }
                }
                else
                    placeTree(unit.roots[0], left, y);
                left += unit.width + 180;
                siteHeight = Math.max(siteHeight, unit.height);
            }
            left -= 180;
            if (group.site) {
                boxes.unshift({ id: `site:${row}:${group.site}`, kind: 'site', title: `位置：${group.site}`, subtitle: '', x: start, y: startTop, width: siteWidths.get(group.site), height: siteHeight + siteTitle + 20 });
                left = start + siteWidths.get(group.site);
            }
            left += 180;
        }
        width = Math.max(width, left - 152, ...boxes.map(box => box.x + box.width + 28), ...resourceNodes.map(node => node.x + node.width + 28));
        const maxBottom = Math.max(...boxes.filter(box => box.y >= top).map(box => box.y + box.height), ...resourceNodes.filter(node => node.y >= top).map(node => node.y + node.height));
        top = Math.max(top, maxBottom) + 106;
    }
    const edges = [];
    const resourceRects = new Map(resourceNodes.map(node => [node.resource.draft_resource_id, node]));
    const labelWidth = (relation) => Math.max(82, [...relationLabel(relation.relation_type_key)].reduce((w, char) => w + (/[^\u0000-\u007f]/.test(char) ? 13 : 8), 24));
    for (const relation of valid.filter(relation => relation.relation_type_key !== 'ORACLE_RAC')) {
        const participants = [...relation.participants].sort((a, b) => (a.display_order || 0) - (b.display_order || 0));
        const sources = participants.filter(p => sourceRoles.has(p.participant_role));
        const effective = sources.length ? sources : participants.slice(0, 1);
        const sourceIds = new Set(effective.map(p => p.draft_resource_id));
        const targets = participants.filter(p => !sourceIds.has(p.draft_resource_id));
        const kind = replicationTypes.has(relation.relation_type_key) ? 'replication' : relation.relation_type_key === 'USES_STORAGE' ? 'storage' : relation.relation_type_key === 'RUNS_ON' ? 'hosting' : 'dependency';
        if (kind === 'hosting' && effective.length === 1 && targets.length === 1 && parents.get(effective[0].draft_resource_id) === targets[0].draft_resource_id) {
            const node = resourceRects.get(effective[0].draft_resource_id);
            edges.push({ relation, kind, contained: true, paths: [], x: node.x + node.width / 2, y: node.y - 14, width: 82 });
            continue;
        }
        const keyFor = (id) => kind === 'replication' && resourceToVisualNode.has(id) ? resourceToVisualNode.get(id) : `resource:${id}`;
        const fromKeys = unique(effective.map(p => keyFor(p.draft_resource_id))), toKeys = unique(targets.map(p => keyFor(p.draft_resource_id)));
        const paths = [];
        let lx = 0, ly = 0, pairs = 0;
        for (const fromKey of fromKeys)
            for (const toKey of toKeys) {
                const from = anchors.get(fromKey), to = anchors.get(toKey);
                if (!from || !to || fromKey === toKey)
                    continue;
                let x, y, route;
                if (kind === 'replication' && (to.x >= from.x + from.width || from.x >= to.x + to.width)) {
                    const forward = to.x >= from.x + from.width;
                    const sx = from.x + (forward ? from.width : 0), tx = to.x + (forward ? 0 : to.width), sy = from.y + 46, ty = to.y + 46;
                    x = (sx + tx) / 2;
                    y = (sy + ty) / 2;
                    route = `M ${sx} ${sy} H ${x} V ${ty} H ${tx}`;
                }
                else {
                    const downward = to.y >= from.y;
                    const sx = from.x + from.width / 2, tx = to.x + to.width / 2, sy = from.y + (downward ? from.height : 0), ty = to.y + (downward ? 0 : to.height);
                    x = (sx + tx) / 2;
                    y = (sy + ty) / 2;
                    // 同层关系从下方绕行，避免穿过资源卡片正文。
                    if (Math.abs(from.y - to.y) < 85) {
                        y = Math.max(from.y + from.height, to.y + to.height) + 36;
                        route = `M ${sx} ${from.y + from.height} V ${y} H ${tx} V ${to.y + to.height}`;
                    }
                    else
                        route = `M ${sx} ${sy} V ${y} H ${tx} V ${ty}`;
                }
                paths.push(route);
                lx += x;
                ly += y;
                pairs++;
            }
        if (pairs)
            edges.push({ relation, kind, contained: false, paths, x: lx / pairs, y: ly / pairs - 18, width: labelWidth(relation) });
    }
    return { boxes, resourceNodes, edges, rows, standalone, width: Math.max(360, width), height: Math.max(160, top - 54), resourceToVisualNode };
}
exports.default = { graphModel, hostProjection, hostTypes: exports.hostTypes, resourceKinds: exports.resourceKinds, relationLabel, databaseSummary };

globalThis.KBotAIOpsTopologyLayout = exports.default;
})();
