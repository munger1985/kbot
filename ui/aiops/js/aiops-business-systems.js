(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const environmentLabels = { PROD: "生产", STG: "预发布", DEV: "开发" };
  const importanceLabels = { 1: "一般", 2: "较低", 3: "重要", 4: "核心", 5: "关键" };
  const architectureLabels = {
    SINGLE_INSTANCE: "单实例",
    RAC: "Oracle RAC",
    DG: "Data Guard",
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

  function databaseScale(count) {
    const value = Number(count || 0);
    return value === 0 ? "尚未添加数据库" : value === 1 ? "1 个数据库" : `${value} 个数据库实例`;
  }

  function protectionLabel(value) {
    if (value === "CONFIGURED") return "已配置";
    if (value === "PARTIAL") return "部分已配置";
    return "未配置";
  }

  function statusLabel(value) {
    if (value === "ENABLED") return "已启用";
    if (value === "DRAFT") return "草稿";
    return "已停用";
  }

  function responsibilityGroupName(groupId, groupsById) {
    if (!groupId) return "未设置责任组";
    return groupsById.get(String(groupId))?.name || "责任组已不可用";
  }

  function card(item, groupsById) {
    const labels = (item.architecture_labels || []).length
      ? item.architecture_labels.map((label) => `<span>${shell.escape(architectureLabel(label))}</span>`).join("")
      : "<span>待推导</span>";
    const action = item.status === "DRAFT" ? "继续配置" : "进入系统";
    return `<article class="business-system-directory-card">
      <header><div><small>${shell.escape(item.system_code)}</small><h3>${shell.escape(item.display_name)}</h3></div>${shell.badge(item.status, statusLabel(item.status))}</header>
      <div class="business-architecture-tags">${labels}</div>
      <dl>
        <div><dt>环境 / 重要级</dt><dd>${shell.escape(environmentLabel(item.environment))} · ${shell.escape(importanceLabel(Number(item.importance_level)))}</dd></div>
        <div><dt>数据库规模</dt><dd>${shell.escape(databaseScale(item.target_count))}</dd></div>
        <div><dt>责任组</dt><dd>${shell.escape(responsibilityGroupName(item.responsibility_group_id, groupsById))}</dd></div>
        <div><dt>恢复目标</dt><dd>${shell.escape(protectionLabel(item.protection_status))}</dd></div>
      </dl>
      <footer><span>更新于 ${shell.escape(shell.fmt(item.updated_at))}</span><a href="./business-system-workspace.html?id=${encodeURIComponent(item.business_system_id)}">${action} →</a></footer>
    </article>`;
  }

  function addCard(empty) {
    const help = empty
      ? "当前暂无业务系统；从业务归属开始建立第一个运维对象"
      : "先登记业务归属，再添加数据库、应用服务器、中间件或网络组件";
    return `<a class="business-system-add-card" href="./business-system-workspace.html?id=new" data-write-ready>
      <span aria-hidden="true">＋</span><strong>新增业务系统</strong><small>${help}</small>
    </a>`;
  }

  async function load() {
    const [page, groupPage] = await Promise.all([
      request("/business-systems?limit=200"),
      request("/responsibility-groups"),
    ]);
    const items = page.items || [];
    const groupsById = new Map((groupPage.items || []).map((item) => [String(item.responsibility_group_id), item]));
    document.getElementById("system-list").innerHTML = `<div class="business-system-directory-grid">${items.map((item) => card(item, groupsById)).join("")}${addCard(!items.length)}</div>`;
  }

  document.getElementById("refresh-systems").onclick = load;
  shell.ready.then(load).catch((error) => shell.toast(error.message));
})();
