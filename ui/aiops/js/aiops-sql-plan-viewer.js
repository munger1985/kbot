/* 执行计划展示器：两端共用同一实现，只消费结构化计划，不解析字符缩进。 */
(function () {
  "use strict";
  if (globalThis.AIOpsSqlPlanViewer) return;
  const kinds = { join: "连接", aggregate: "聚合", sort: "排序", access: "数据访问", control: "控制 / 结果" };
  const metricNames = {
    optimizer_cost: "优化器成本（非毫秒）", optimizer_total_cost: "优化器总成本（非毫秒）",
    optimizer_query_cost: "查询成本（非毫秒）", optimizer_read_cost: "读取成本（非毫秒）",
    optimizer_eval_cost: "计算成本（非毫秒）", optimizer_prefix_cost: "前缀成本（非毫秒）",
    estimated_rows: "估算行数", estimated_rows_examined_per_scan: "估算扫描行数／扫描",
    estimated_rows_produced_per_join: "估算输出行数／连接", estimated_filtered_percent: "估算过滤比例（%）",
    actual_rows_total: "LAST 实际行数（合计）", actual_rows_per_loop: "实际行数／循环",
    actual_rows_per_start: "LAST 实际行数／启动", actual_starts: "LAST 启动次数", actual_loops: "实际循环次数",
    actual_elapsed_time_us: "LAST 耗时（微秒）", actual_total_time_per_loop_ms: "实际耗时／循环（毫秒）",
    actual_consistent_gets: "LAST 一致性读", actual_disk_reads: "LAST 物理读", actual_disk_writes: "LAST 物理写",
    actual_to_estimated_rows_ratio: "实际／估算行数比", rows_removed_by_filter_per_loop: "过滤移除行数／循环",
    rows_removed_by_join_filter_per_loop: "连接过滤移除行数／循环", workers_planned: "计划工作进程",
    workers_launched: "已启动工作进程", shared_hit_blocks: "共享块命中", shared_read_blocks: "共享块读取",
    temp_read_blocks: "临时块读取", temp_written_blocks: "临时块写入"
  };
  let serial = 0;
  const states = new WeakMap();
  const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const shorten = (value, length) => String(value ?? "").length > length ? String(value).slice(0, length - 1) + "…" : String(value ?? "");
  const objectName = node => [node.schema_name, node.relation_name].filter(Boolean).join(".");
  function category(node) {
    const op = node.operation.toUpperCase().replaceAll("_", " ");
    if (/JOIN|NESTED LOOP/.test(op)) return "join";
    if (/AGGREGATE|GROUPING|GROUP BY/.test(op)) return "aggregate";
    if (/SORT|ORDERING/.test(op)) return "sort";
    if (node.relation_name || node.index_name || /SCAN|TABLE ACCESS|FIXED TABLE|INDEX|BITMAP/.test(op)) return "access";
    return "control";
  }
  function graphFrom(payload) {
    const graph = payload.plan_graph;
    if (graph?.schema_version !== "SQL_PLAN_GRAPH.v1" || !Array.isArray(graph.nodes)
      || !graph.nodes.length || graph.nodes.length > 128) return null;
    const seen = new Set();
    for (const n of graph.nodes) {
      if (!n || !Number.isInteger(n.node_id) || seen.has(n.node_id) || typeof n.operation !== "string"
        || (n.parent_id !== null && (!Number.isInteger(n.parent_id) || !seen.has(n.parent_id)))) return null;
      seen.add(n.node_id);
    }
    return graph;
  }
  function descendants(nodes, id) {
    const ids = new Set([id]);
    for (const node of nodes) if (ids.has(node.parent_id)) ids.add(node.node_id);
    ids.delete(id);
    return ids;
  }
  function layout(nodes) {
    const children = new Map(nodes.map(n => [n.node_id, []]));
    const byId = new Map(nodes.map(n => [n.node_id, n]));
    for (const n of nodes) if (children.has(n.parent_id)) children.get(n.parent_id).push(n);
    const positions = new Map();
    let leaves = 0;
    function visit(n) {
      const inputs = children.get(n.node_id);
      const inner = inputs.map(visit);
      const rank = inner.length ? Math.max(...inner.map(p => p.rank)) + 1 : 0;
      const y = inner.length ? (inner[0].y + inner[inner.length - 1].y) / 2 : 28 + leaves++ * 140;
      const position = { node: n, rank, x: 24 + rank * 304, y };
      positions.set(n.node_id, position);
      return position;
    }
    nodes.filter(n => !byId.has(n.parent_id)).forEach(visit);
    const placed = Array.from(positions.values());
    return {
      nodes: placed,
      edges: placed.filter(p => positions.has(p.node.parent_id)).map(p => ({ from: p, to: positions.get(p.node.parent_id) })),
      width: Math.max(400, ...placed.map(p => p.x + 282)), height: Math.max(222, ...placed.map(p => p.y + 126))
    };
  }
  function button(action, label, extra = "") {
    return '<button type="button" class="secondary-button" data-plan-action="' + action + '" ' + extra + '>' + label + "</button>";
  }
  function tableHtml(payload, graph, uid) {
    const columns = Array.isArray(payload.columns) ? payload.columns : [];
    const rows = Array.isArray(payload.rows) ? payload.rows : [];
    const byId = new Map((graph?.nodes || []).map(n => [n.node_id, n]));
    const idIndex = columns.findIndex(c => c.name === "node_id");
    const siblings = new Map();
    for (const n of byId.values()) {
      if (!siblings.has(n.parent_id)) siblings.set(n.parent_id, []);
      siblings.get(n.parent_id).push(n.node_id);
    }
    function branchHtml(node) {
      if (!node || node.parent_id === null) return "";
      const ancestors = [];
      let cursor = node;
      while (cursor.parent_id !== null && byId.has(cursor.parent_id)) {
        ancestors.unshift(cursor);
        cursor = byId.get(cursor.parent_id);
      }
      const lines = ancestors.map((n, index) => {
        const group = siblings.get(n.parent_id) || [];
        const last = group[group.length - 1] === n.node_id;
        const kind = index === ancestors.length - 1 ? (last ? "elbow" : "tee") : (last ? "empty" : "through");
        return '<i class="sql-plan-branch ' + kind + '"></i>';
      }).join("");
      return '<span class="sql-plan-branches" aria-hidden="true">' + lines + "</span>";
    }
    const heads = columns.map(c => '<th scope="col">' + esc(c.label || c.name) + "</th>").join("");
    const body = rows.map(row => {
      const node = byId.get(Number(row[idIndex]));
      return "<tr>" + columns.map((c, i) => {
        if (c.name === "operation" && node) return '<td class="sql-plan-operation"><span class="sql-plan-operation-inner">' + branchHtml(node)
          + '<span title="' + esc(node.operation) + '">' + esc(node.operation) + "</span></span></td>";
        return '<td' + (["object", "index", "operation"].includes(c.name) ? ' class="sql-plan-raw-name"' : ' class="sql-plan-raw-value"')
          + ' title="' + esc(row[i] ?? "—") + '">' + esc(row[i] ?? "—") + "</td>";
      }).join("") + "</tr>";
    }).join("");
    const native = payload.native_plan;
    const text = native && Array.isArray(native.lines)
      ? '<details class="sql-plan-native"><summary>DBMS_XPLAN 原生输出' + (native.truncated ? " · 已截断" : "")
        + '</summary><p>' + esc(native.note || "") + (native.captured_at ? " · 采集时间 " + esc(native.captured_at) : "")
        + '</p><pre tabindex="0">' + esc(native.lines.slice(0, 256).join("\n")) + "</pre></details>"
      : "";
    return '<div class="sql-plan-raw-scroll" tabindex="0" aria-label="原始执行计划，可横向滚动"><table class="sql-plan-raw"><thead><tr>'
      + heads + "</tr></thead><tbody>" + body + "</tbody></table></div>" + text;
  }
  function dataFor(payload, uid) {
    return { uid, payload, graph: graphFrom(payload), selected: null, focused: false, collapsed: new Set(), scale: 1, tab: "graph" };
  }
  function graphHtml(state) {
    const graph = state.graph;
    if (!graph) return '<p class="sql-plan-empty">此份证据未附结构化计划节点；可在“原始计划”查看已保存输出，不猜测连接关系。</p>';
    let visible = graph.nodes;
    for (const id of state.collapsed) {
      const hidden = descendants(graph.nodes, id);
      visible = visible.filter(n => !hidden.has(n.node_id));
    }
    if (state.focused && state.selected !== null) {
      const current = graph.nodes.find(n => n.node_id === state.selected);
      visible = graph.nodes.filter(n => n.node_id === state.selected || n.node_id === current?.parent_id || n.parent_id === state.selected);
    }
    const placed = layout(visible);
    const marker = state.uid + "-arrow";
    const lines = placed.edges.map(({ from, to }) => {
      const start = from.x + 258, end = to.x, middle = (start + end) / 2;
      const hot = state.selected === from.node.node_id || state.selected === to.node.node_id;
      return '<path class="sql-plan-edge' + (hot ? " is-selected" : "") + '" marker-end="url(#' + marker
        + ')" d="M' + start + " " + (from.y + 48) + " H" + middle + " V" + (to.y + 48) + " H" + end + '"/>';
    }).join("");
    const boxes = placed.nodes.map(({ node, x, y }) => {
      const kind = category(node);
      const words = node.operation.match(/.{1,26}/g) || [node.operation];
      const operation = words.slice(0, 2).map((line, i) => '<tspan x="15" dy="' + (i ? 18 : 0) + '">'
        + esc(i === 1 && words.length > 2 ? line.slice(0, 25) + "…" : line) + "</tspan>").join("");
      const collapsed = state.collapsed.has(node.node_id) ? " · 已折叠 " + descendants(graph.nodes, node.node_id).size + " 节点" : "";
      const caption = objectName(node) || node.index_name || kinds[kind];
      return '<g class="sql-plan-node is-' + kind + (state.selected === node.node_id ? " is-selected" : "")
        + '" role="button" tabindex="0" aria-pressed="' + (state.selected === node.node_id)
        + '" aria-label="节点 ' + node.node_id + " " + esc(node.operation) + '" data-plan-node="' + node.node_id
        + '" transform="translate(' + x + " " + y + ')"><title>#' + node.node_id + " " + esc(node.operation)
        + (objectName(node) ? " · " + esc(objectName(node)) : "") + '</title><rect width="258" height="100" rx="7"/>'
        + '<text class="sql-plan-node-kind" x="15" y="20">#' + node.node_id + " · " + kinds[kind] + collapsed
        + '</text><text class="sql-plan-node-operation" x="15" y="43">' + operation
        + '</text><text class="sql-plan-node-object" x="15" y="84">' + esc(shorten(caption, 30)) + "</text></g>";
    }).join("");
    return '<svg class="sql-plan-graph" width="' + (placed.width * state.scale) + '" height="' + (placed.height * state.scale)
      + '" viewBox="0 0 ' + placed.width + " " + placed.height + '" role="group" aria-label="执行计划数据流图，方向从数据源到结果">'
      + '<defs><marker id="' + marker + '" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">'
      + '<path d="M0,0 L8,4 L0,8 Z"/></marker></defs>' + lines + boxes + "</svg>";
  }
  function detailsHtml(state) {
    const node = state.graph?.nodes.find(n => n.node_id === state.selected);
    if (!node) return '<div class="sql-plan-detail-empty">选择图中节点，查看对象、原生指标和上下游。</div>';
    const children = state.graph.nodes.filter(n => n.parent_id === node.node_id);
    const origin = { DATABASE_CURSOR: "缓存计划 / LAST 行源", CONTROLLED_ESTIMATE: "受控估算计划", USER_UPLOAD: "用户提供计划（待核验）" }[state.payload.plan_origin] || "以证据来源为准";
    const metrics = Object.entries(node.metrics || {}).map(([key, value]) => '<div><dt>' + esc(metricNames[key] || key) + "</dt><dd>" + esc(value) + "</dd></div>").join("");
    const navigation = (id, label) => '<button class="text-button" type="button" data-plan-node="' + id + '">' + esc(label) + "</button>";
    return '<div class="sql-plan-detail-head"><span>节点 #' + node.node_id + " · " + kinds[category(node)] + '</span>'
      + button("close", "关闭", 'aria-label="关闭节点详情"') + '</div><h4>' + esc(node.operation) + "</h4>"
      + '<dl><div><dt>对象</dt><dd>' + esc(objectName(node) || "未提供") + '</dd></div><div><dt>索引</dt><dd>'
      + esc(node.index_name || "未提供") + '</dd></div><div><dt>父节点</dt><dd>'
      + (node.parent_id === null ? "结果根节点" : navigation(node.parent_id, "#" + node.parent_id))
      + '</dd></div><div><dt>输入节点</dt><dd>' + (children.map(n => navigation(n.node_id, "#" + n.node_id)).join(" / ") || "未提供输入节点")
      + '</dd></div><div><dt>计划来源</dt><dd>' + esc(origin) + "</dd></div>" + metrics + "</dl>"
      + (!metrics ? '<p class="sql-plan-detail-note">未提供节点指标，不显示耗时占比或热点。</p>' : "")
      + '<p class="sql-plan-detail-note">成本不是耗时；父子节点耗时可能包含，不能直接相加。FULL 不自动代表异常。</p>'
      + '<div class="sql-plan-detail-actions">' + button("focus", state.focused ? "查看完整图" : "聚焦上下游")
      + (children.length ? button("collapse", state.collapsed.has(node.node_id) ? "展开输入分支" : "折叠输入分支") : "") + "</div>";
  }
  function html(payload) {
    const uid = "sql-plan-" + (++serial);
    const state = dataFor(payload, uid);
    if (!state.graph) state.tab = "raw";
    const encoded = esc(JSON.stringify({ uid, payload }));
    const legend = Object.entries(kinds).map(([kind, label]) => '<span><i class="sql-plan-dot is-' + kind + '"></i>' + label + "</span>").join("");
    return '<div class="sql-plan-viewer" data-plan-state="' + encoded + '">'
      + '<div class="sql-plan-toolbar"><div class="sql-plan-tabs" role="tablist" aria-label="执行计划视图">'
      + button("tab-graph", "图形计划", 'role="tab" tabindex="' + (state.tab === "graph" ? 0 : -1) + '" id="' + uid + '-tab-graph" aria-selected="' + (state.tab === "graph") + '" aria-controls="' + uid + '-graph"')
      + button("tab-raw", "原始计划", 'role="tab" tabindex="' + (state.tab === "raw" ? 0 : -1) + '" id="' + uid + '-tab-raw" aria-selected="' + (state.tab === "raw") + '" aria-controls="' + uid + '-raw"')
      + '</div><div class="sql-plan-tools"' + (state.tab === "raw" ? " hidden" : "") + ">"
      + button("out", "−", 'aria-label="缩小计划"') + '<span data-plan-scale>100%</span>' + button("in", "+", 'aria-label="放大计划"')
      + button("reset", "还原视图") + button("fullscreen", "全屏查看") + "</div></div>"
      + '<div role="tabpanel" tabindex="0" aria-labelledby="' + uid + '-tab-graph" id="' + uid + '-graph" data-plan-panel="graph"' + (state.tab !== "graph" ? " hidden" : "") + '>'
      + '<div class="sql-plan-legend">' + legend + '<span class="sql-plan-color-note">颜色区分类别，不表示故障程度</span></div>'
      + '<div class="sql-plan-workspace"><div class="sql-plan-canvas" tabindex="0" aria-label="图形画布，可横向或纵向滚动">' + graphHtml(state)
      + '</div><aside class="sql-plan-inspector" aria-label="计划节点详情" hidden>' + detailsHtml(state) + "</aside></div>"
      + '<p class="sql-plan-caption">数据源 → 处理算子 → 结果；箭头不是严格执行顺序。大计划可滚动、聚焦或折叠查看。</p></div>'
      + '<div role="tabpanel" tabindex="0" aria-labelledby="' + uid + '-tab-raw" id="' + uid + '-raw" data-plan-panel="raw"' + (state.tab !== "raw" ? " hidden" : "") + ">"
      + tableHtml(payload, state.graph, uid) + "</div></div>";
  }
  function stateFor(root) {
    if (!states.has(root)) {
      const encoded = JSON.parse(root.dataset.planState);
      states.set(root, dataFor(encoded.payload, encoded.uid));
    }
    return states.get(root);
  }
  function paint(root, state) {
    const canvas = root.querySelector(".sql-plan-canvas");
    const scroll = { left: canvas.scrollLeft, top: canvas.scrollTop };
    canvas.innerHTML = graphHtml(state);
    canvas.scrollLeft = scroll.left;
    canvas.scrollTop = scroll.top;
    const inspector = root.querySelector(".sql-plan-inspector");
    inspector.hidden = state.selected === null;
    inspector.innerHTML = detailsHtml(state);
    root.querySelector(".sql-plan-workspace").classList.toggle("has-inspector", state.selected !== null);
    root.querySelector("[data-plan-scale]").textContent = Math.round(state.scale * 100) + "%";
  }
  function interact(event) {
    const target = event.target instanceof Element ? event.target.closest("[data-plan-action],[data-plan-node]") : null;
    const root = target?.closest(".sql-plan-viewer");
    if (!root) return;
    const state = stateFor(root);
    if (target.hasAttribute("data-plan-node")) {
      state.selected = Number(target.dataset.planNode);
      paint(root, state);
      const node = root.querySelector('.sql-plan-node[data-plan-node="' + state.selected + '"]');
      const canvas = root.querySelector(".sql-plan-canvas");
      if (node) {
        const box = node.getBoundingClientRect(), viewport = canvas.getBoundingClientRect();
        if (box.left < viewport.left || box.right > viewport.right) canvas.scrollLeft += box.left - viewport.left - 20;
        if (box.top < viewport.top || box.bottom > viewport.bottom) canvas.scrollTop += box.top - viewport.top - 20;
        if (target.classList.contains("sql-plan-node")) node.focus({ preventScroll: true });
      }
      return;
    }
    const action = target.dataset.planAction;
    if (action.startsWith("tab-")) {
      state.tab = action.slice(4);
      root.querySelectorAll("[role=tab]").forEach(tab => {
        tab.setAttribute("aria-selected", String(tab === target));
        tab.tabIndex = tab === target ? 0 : -1;
      });
      root.querySelectorAll("[data-plan-panel]").forEach(panel => { panel.hidden = panel.dataset.planPanel !== state.tab; });
      root.querySelector(".sql-plan-tools").hidden = state.tab !== "graph";
      return;
    }
    if (action === "fullscreen") {
      const label = () => { target.textContent = document.fullscreenElement === root || root.classList.contains("is-expanded") ? "退出全屏" : "全屏查看"; };
      if (root.classList.contains("is-expanded")) { root.classList.remove("is-expanded"); label(); return; }
      if (document.fullscreenElement === root) document.exitFullscreen?.().catch(() => {});
      else if (root.requestFullscreen) root.requestFullscreen().catch(() => { root.classList.add("is-expanded"); label(); });
      else { root.classList.add("is-expanded"); label(); }
      return;
    }
    if (action === "close") { state.selected = null; state.focused = false; }
    if (action === "focus") state.focused = !state.focused;
    if (action === "collapse") {
      if (state.collapsed.has(state.selected)) state.collapsed.delete(state.selected);
      else state.collapsed.add(state.selected);
      state.focused = false;
    }
    if (action === "in" || action === "out") state.scale = Math.max(.75, Math.min(1.75, state.scale + (action === "in" ? .1 : -.1)));
    if (action === "reset") { state.scale = 1; state.focused = false; state.collapsed.clear(); }
    paint(root, state);
    if (action === "reset") root.querySelector(".sql-plan-canvas").scrollTo(0, 0);
  }
  globalThis.AIOpsSqlPlanViewer = { html, layout, category, graphFrom };
  if (typeof document !== "undefined") {
    document.addEventListener("click", interact);
    document.addEventListener("fullscreenchange", () => {
      document.querySelectorAll(".sql-plan-viewer [data-plan-action=fullscreen]").forEach(button => {
        button.textContent = document.fullscreenElement === button.closest(".sql-plan-viewer") ? "退出全屏" : "全屏查看";
      });
    });
    document.addEventListener("keydown", event => {
      const target = event.target instanceof Element ? event.target : null;
      if (!target?.closest(".sql-plan-viewer")) return;
      if (event.key === "Escape") {
        const expanded = target.closest(".sql-plan-viewer.is-expanded");
        if (expanded) {
          expanded.classList.remove("is-expanded");
          expanded.querySelector("[data-plan-action=fullscreen]").textContent = "全屏查看";
        }
      }
      if (target.matches(".sql-plan-node") && ["Enter", " "].includes(event.key)) {
        event.preventDefault();
        interact(event);
      }
      if (target.getAttribute("role") === "tab" && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
        event.preventDefault();
        const tabs = Array.from(target.parentElement.querySelectorAll("[role=tab]"));
        const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
          : (tabs.indexOf(target) + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
        tabs[index].click();
        tabs[index].focus();
      }
    });
  }
})();
