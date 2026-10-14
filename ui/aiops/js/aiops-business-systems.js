(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const state = { items: [], groupsById: new Map(), status: "", busyId: "", offlineItem: null };
  const environmentLabels = { PROD: "生产", STG: "预发布", DEV: "开发" };
  const importanceLabels = { 1: "低", 2: "一般", 3: "重要", 4: "关键", 5: "核心" };
  const architectureLabels = {
    SINGLE_INSTANCE: "单实例",
    RAC: "Oracle RAC",
    DATA_GUARD: "Data Guard",
    RAC_DATA_GUARD: "RAC + Data Guard",
    MYSQL_REPLICATION: "MySQL 主从复制",
    POSTGRES_STREAMING: "PostgreSQL 流复制",
    OGG: "Oracle GoldenGate",
    MIXED_VERSION: "混合版本",
  };

  function environmentLabel(value) {
    return environmentLabels[value] || "其他环境";
  }

  function importanceLabel(value) {
    return `L${value} ${importanceLabels[value] || "未分级"}`;
  }

  function architectureLabel(value) {
    return architectureLabels[value] || "待识别架构";
  }

  function resourceScale(item) {
    const count = Number(item.resource_count || 0);
    if (!count) return "尚未添加资源";
    const details = (item.resource_type_summary || []).slice(0, 2).map((row) => `${Number(row.count)} 个${row.display_name}`);
    return details.length ? details.join(" · ") : `${count} 个资源`;
  }

  function protectionLabel(value) {
    if (value === "CONFIGURED") return "已配置";
    if (value === "PARTIAL") return "部分已配置";
    return "未配置";
  }

  function statusLabel(value) {
    if (value === "ENABLED") return "运行中";
    if (value === "DRAFT") return "草稿";
    return "已下线";
  }

  function responsibilityGroupName(groupId, groupsById) {
    if (!groupId) return "未设置责任组";
    return groupsById.get(String(groupId))?.name || "责任组已不可用";
  }

  function card(item, groupsById) {
    const labels = (item.architecture_labels || []).length
      ? item.architecture_labels.map((label) => `<span>${shell.escape(architectureLabel(label))}</span>`).join("")
      : "<span>待推导</span>";
    const workspaceLabel = item.status === "DRAFT" ? "继续配置" : "进入系统";
    const lifecycleAction = item.status === "ENABLED"
      ? `<details class="business-system-more"><summary>更多⌄</summary><div><button type="button" data-offline-system="${shell.escape(item.business_system_id)}">下线业务系统</button></div></details>`
      : item.status === "DRAFT"
        ? `<button type="button" data-enable-system="${shell.escape(item.business_system_id)}">上线</button><details class="business-system-more"><summary>更多⌄</summary><div><button class="danger" type="button" data-delete-system="${shell.escape(item.business_system_id)}">删除草稿</button></div></details>`
        : `<button type="button" data-enable-system="${shell.escape(item.business_system_id)}">重新上线</button>`;
    return `<article class="business-system-directory-card">
      <header><div><small>${shell.escape(item.system_code)}</small><h3>${shell.escape(item.display_name)}</h3></div>${shell.badge(item.status, statusLabel(item.status))}</header>
      <div class="business-architecture-tags">${labels}</div>
      <dl>
        <div><dt>环境 / 重要级</dt><dd>${shell.escape(environmentLabel(item.environment))} · ${shell.escape(importanceLabel(Number(item.importance_level)))}</dd></div>
        <div><dt>资源构成</dt><dd>${shell.escape(resourceScale(item))}</dd></div>
        <div><dt>业务责任组</dt><dd>${shell.escape(responsibilityGroupName(item.responsibility_group_id, groupsById))}</dd></div>
        <div><dt>数据保护</dt><dd>${shell.escape(protectionLabel(item.protection_status))}</dd></div>
      </dl>
      <footer class="business-system-card-actions"><span>更新于 ${shell.escape(shell.fmt(item.updated_at))}</span><div><a href="./business-system-workspace.html?id=${encodeURIComponent(item.business_system_id)}">${workspaceLabel}</a>${lifecycleAction}</div></footer>
    </article>`;
  }

  function addCard(empty) {
    const help = empty
      ? "当前暂无业务系统；从业务归属开始建立第一个运维对象"
      : "登记业务归属，再接入数据库、应用服务、中间件、网络或存储资源";
    return `<a class="business-system-add-card" href="./business-system-workspace.html?id=new" data-write-ready>
      <span aria-hidden="true">＋</span><strong>新增业务系统</strong><small>${help}</small>
    </a>`;
  }

  async function load() {
    const [page, groupPage] = await Promise.all([
      request("/business-systems?limit=200"),
      request("/responsibility-groups"),
    ]);
    state.items = page.items || [];
    state.groupsById = new Map((groupPage.items || []).map((item) => [String(item.responsibility_group_id), item]));
    render();
  }

  function render() {
    const items = state.status ? state.items.filter((item) => item.status === state.status) : state.items;
    document.getElementById("system-count").textContent = `${items.length} 个`;
    document.querySelectorAll("#system-status-filter [data-status]").forEach((button) => button.classList.toggle("active", button.dataset.status === state.status));
    const empty = items.length ? "" : state.items.length ? '<p class="ops-empty">当前筛选条件下没有业务系统。</p>' : "";
    document.getElementById("system-list").innerHTML = `<div class="business-system-directory-grid">${items.map((item) => card(item, state.groupsById)).join("")}${addCard(!state.items.length)}</div>${empty}`;
  }

  function itemById(id) {
    return state.items.find((item) => String(item.business_system_id) === String(id));
  }

  function confirmAction(title, message, confirmLabel, danger = false) {
    const dialog = document.getElementById("business-system-action-dialog");
    document.getElementById("business-system-action-title").textContent = title;
    document.getElementById("business-system-action-message").textContent = message;
    const confirm = document.getElementById("business-system-action-confirm");
    confirm.textContent = confirmLabel;
    confirm.className = danger ? "danger" : "primary";
    dialog.returnValue = "";
    dialog.showModal();
    return new Promise((resolve) => dialog.addEventListener("close", () => resolve(dialog.returnValue === "confirm"), { once: true }));
  }

  async function command(item, action, reason = null) {
    state.busyId = item.business_system_id;
    try {
      await request(`/business-systems/${encodeURIComponent(item.business_system_id)}/${action}`, {
        method: "POST",
        headers: { "If-Match": `"rv-${item.row_version}"`, "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(reason ? { reason } : {}),
      });
      shell.toast(action === "enable" ? (item.status === "DRAFT" ? "业务系统已上线" : "业务系统已重新上线") : "业务系统已下线");
      state.offlineItem = null;
      document.getElementById("business-system-offline-dialog").close();
      await load();
    } catch (error) { shell.toast(error.message); }
    finally { state.busyId = ""; }
  }

  async function enable(item) {
    const first = item.status === "DRAFT";
    if (!await confirmAction(first ? "上线业务系统？" : "重新上线业务系统？", `确认${first ? "上线" : "重新上线"}“${item.display_name}”？系统会先校验资源与拓扑配置。`, first ? "确认上线" : "重新上线")) return;
    await command(item, "enable");
  }

  async function removeDraft(item) {
    if (!await confirmAction("删除草稿业务系统？", `“${item.display_name}”将从业务系统目录中移除。已接入的资源会保留，不会删除资源、拓扑或历史数据。`, "删除草稿", true)) return;
    state.busyId = item.business_system_id;
    try {
      await request(`/business-systems/${encodeURIComponent(item.business_system_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${item.row_version}"` } });
      shell.toast("草稿业务系统已删除，已接入资源仍然保留");
      await load();
    } catch (error) { shell.toast(error.message); }
    finally { state.busyId = ""; }
  }

  document.getElementById("system-status-filter").onclick = (event) => {
    const button = event.target.closest("[data-status]");
    if (!button) return;
    state.status = button.dataset.status || "";
    render();
  };
  document.getElementById("system-list").onclick = (event) => {
    const enableButton = event.target.closest("[data-enable-system]");
    const offlineButton = event.target.closest("[data-offline-system]");
    const deleteButton = event.target.closest("[data-delete-system]");
    if (enableButton) enable(itemById(enableButton.dataset.enableSystem));
    if (deleteButton) removeDraft(itemById(deleteButton.dataset.deleteSystem));
    if (offlineButton) {
      state.offlineItem = itemById(offlineButton.dataset.offlineSystem);
      document.getElementById("business-system-offline-title").textContent = `下线 ${state.offlineItem.display_name}`;
      const form = document.getElementById("business-system-offline-form");
      form.reset();
      document.getElementById("business-system-offline-dialog").showModal();
    }
  };
  document.getElementById("business-system-offline-form").onsubmit = async (event) => {
    event.preventDefault();
    const reason = String(new FormData(event.currentTarget).get("reason") || "").trim();
    if (!reason) { shell.toast("请填写下线原因"); return; }
    if (state.offlineItem) await command(state.offlineItem, "disable", reason);
  };
  document.querySelectorAll("[data-close-offline]").forEach((button) => button.onclick = () => {
    if (!state.busyId) document.getElementById("business-system-offline-dialog").close();
  });

  document.getElementById("refresh-systems").onclick = load;
  shell.ready.then(load).catch((error) => shell.toast(error.message));
})();
