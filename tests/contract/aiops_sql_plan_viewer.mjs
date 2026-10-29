import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

const source = readFileSync(new URL("../../ui/aiops/js/aiops-sql-plan-viewer.js", import.meta.url), "utf8");
const context = {};
runInNewContext(source, context);
const viewer = context.AIOpsSqlPlanViewer;
const nodes = [
  { node_id: 0, parent_id: null, operation: "SELECT STATEMENT", metrics: {} },
  { node_id: 1, parent_id: 0, operation: "HASH JOIN", metrics: { actual_elapsed_time_us: "0" } },
  { node_id: 2, parent_id: 1, operation: "FIXED TABLE FULL", relation_name: "X$KSPPI", metrics: {} },
  { node_id: 3, parent_id: 1, operation: "Index Scan", index_name: "equipment_idx", metrics: {} }
];
const payload = {
  plan_graph: { schema_version: "SQL_PLAN_GRAPH.v1", database_type: "ORACLE", nodes },
  columns: [{ name: "node_id", label: "节点" }, { name: "operation", label: "执行操作" }],
  rows: nodes.map(n => [n.node_id, n.operation])
};
assert.equal(viewer.graphFrom(payload).nodes.length, 4);
const placed = viewer.layout(nodes);
assert.equal(placed.nodes.length, 4);
assert.equal(placed.edges.length, 3);
for (const edge of placed.edges) {
  assert.equal(edge.from.node.parent_id, edge.to.node.node_id);
  assert.ok(edge.from.x + 258 < edge.to.x);
}
assert.equal(viewer.category(nodes[1]), "join");
assert.equal(viewer.category(nodes[2]), "access");
assert.equal(viewer.category({ operation: "grouping_operation" }), "aggregate");
assert.equal(viewer.category({ operation: "ordering_operation" }), "sort");
assert.equal(viewer.category({ operation: "eq_ref", relation_name: "orders" }), "access");
assert.equal(viewer.category({ operation: "Unknown Operator" }), "control");
const html = viewer.html(payload);
assert.ok(html.includes("图形计划") && html.includes("原始计划"));
assert.ok(html.includes('role="tab"') && html.includes('role="tabpanel"'));
assert.ok(html.includes("sql-plan-branch tee") && html.includes("sql-plan-branch elbow"));
assert.ok(!html.includes("onclick="));
const unsafe = structuredClone(payload);
unsafe.plan_graph.nodes[2].relation_name = '<img src=x onerror="attack()">';
assert.ok(!viewer.html(unsafe).includes("<img"));
assert.equal(viewer.graphFrom({}), null);
const cyclic = structuredClone(payload);
cyclic.plan_graph.nodes[0].parent_id = 1;
assert.equal(viewer.graphFrom(cyclic), null);
const duplicate = structuredClone(payload);
duplicate.plan_graph.nodes[1].node_id = 0;
assert.equal(viewer.graphFrom(duplicate), null);
const large = Array.from({ length: 128 }, (_, node_id) => ({
  node_id, parent_id: node_id ? node_id - 1 : null, operation: "VIEW"
}));
assert.equal(viewer.layout(large).nodes.length, 128);
assert.equal(viewer.layout(large).edges.length, 127);
const forest = viewer.layout([{ node_id: 0, parent_id: null, operation: "SELECT" }, { node_id: 1, parent_id: null, operation: "SELECT" }]);
assert.equal(forest.nodes.length, 2);
assert.notEqual(forest.nodes[0].y, forest.nodes[1].y);
process.stdout.write("计划结构、原生类别、紧凑树线、转义与有界布局通过\n");
