(function () {
  "use strict";
  if (document.body.dataset.page !== "inspections") return;
  const shell = globalThis.KBotAIOpsShell, esc = shell.escape;
  const request = path => KBotAIOpsAuth.request(`/api/v1/apps/aiops${path}`);
  const rows = value => Array.isArray(value) ? value : [];
  const text = (id,value) => {document.getElementById(id).textContent = value;};
  const systemFilter = new URLSearchParams(location.search).get("businessSystemId") || "";
  const state = {items:[],planId:new URLSearchParams(location.search).get("planId") || "",sequence:0};
  const statusLabel = value => ({QUEUED:"等待执行",RUNNING:"运行中",COMPLETED:"已完成",PARTIAL:"部分完成",FAILED:"失败",CANCELLED:"已取消"})[value] || value;
  const statusClass = value => ["FAILED","CANCELLED"].includes(value) ? "status-danger" : ["QUEUED","RUNNING","PARTIAL"].includes(value) ? "status-warning" : "status-success";
  function renderList() {
    const items = state.items.filter(item => (!state.planId || item.plan_id === state.planId) && (!systemFilter || item.business_system_id === systemFilter));
    const attention = items.filter(item => ["PARTIAL","FAILED"].includes(item.status)).length;
    text("inspection-total",`${items.length} 批`);text("inspection-running",`${items.filter(item => ["QUEUED","RUNNING"].includes(item.status)).length} 批`);text("inspection-attention",`${attention} 批`);text("inspection-completed",`${items.filter(item => item.status === "COMPLETED").length} 批`);
    document.getElementById("inspection-attention-card").classList.toggle("is-danger",Boolean(attention));
    text("inspection-list-title",state.planId ? "计划巡检报告" : "巡检报告");text("inspection-count",`${items.length} 批`);document.getElementById("inspection-clear-plan").hidden = !state.planId;
    document.getElementById("inspection-body").innerHTML = items.length ? items.map(item => `<tr><td><strong class="table-primary">${esc(item.plan_display_name)}</strong>${item.business_system_name ? `<small class="table-secondary">${esc(item.business_system_name)}</small>` : ""}<small class="table-secondary technical-value">${esc(item.fire_id)}</small></td><td>${esc(shell.fmt(item.scheduled_at))}${item.completed_at ? `<small class="table-secondary">完成于 ${esc(shell.fmt(item.completed_at))}</small>` : ""}</td><td><span class="status-pill ${statusClass(item.status)}">${esc(statusLabel(item.status))}</span></td><td>${item.target_count} 个目标</td><td>${item.completed_count} / ${item.failed_count}</td><td><a class="row-action" href="./inspection-detail.html?id=${encodeURIComponent(item.fire_id)}${state.planId ? `&amp;planId=${encodeURIComponent(state.planId)}` : ""}">查看批次</a></td></tr>`).join("") : `<tr><td colspan="6" class="empty-cell">${state.planId ? "该计划尚未产生巡检报告。" : "当前业务空间还没有巡检报告。"}</td></tr>`;
  }
  async function load() {
    const sequence = ++state.sequence;
    document.getElementById("inspection-loading").hidden = false;
    try {const fires = await request("/inspection-fires?limit=100");if (sequence === state.sequence) {state.items = rows(fires.items);renderList();}}
    catch(error) {shell.toast(error.message || "无法读取数据库巡检");}
    finally {if (sequence === state.sequence) document.getElementById("inspection-loading").hidden = true;}
  }
  shell.ready.then(access => {
    document.querySelectorAll("[data-plan-manage]").forEach(node => {node.hidden = !rows(access.permissions).includes("aiops:plan_manage");});
    document.querySelectorAll("[data-inspection-plans]").forEach(node => {node.onclick = () => {location.href = "./inspection-plans.html";};});
    document.getElementById("refresh-workspace").onclick = load;
    document.getElementById("inspection-clear-plan").onclick = () => {state.planId = "";const url = new URL(location.href);url.searchParams.delete("planId");history.replaceState(null,"",url);renderList();};
    return load();
  }).catch(error => shell.toast(error.message));
})();
