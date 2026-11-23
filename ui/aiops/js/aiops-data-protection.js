(function () {
  "use strict";
  const shell = globalThis.KBotAIOpsShell;
  const request = (path, init) => KBotAIOpsAuth.request("/api/v1/apps/aiops" + path, init);
  const element = id => document.getElementById("protection-" + id);
  const form = element("form");
  const query = new URLSearchParams(location.search);
  const fields = form.elements;
  const plain = ["topology", "rto_seconds", "rpo_seconds", "storage_key",
    "backup_directory", "backup_parallelism", "restore_template_key", "drill_destination_key", "timeout_seconds", "schedule_window_seconds"];
  const url = () => "/data-protection/resources/" + encodeURIComponent(resourceId);
  const labels = {MISSING:"缺失", UNVERIFIED:"未验证", PASS:"通过", DATABASE:"数据库", HOST:"主机", STORAGE:"存储", NETWORK_TOOL:"网络与工具", VERIFY:"恢复后验证"};
  const levels = {CRITICAL:"极高", HIGH:"高", MEDIUM:"中", LOWER:"较低", LOW:"低"};
  const permissionDrawer = element("permissions");
  const sections = {
    config:["保护配置", "管理期望恢复目标和专用身份。"], backup:["备份计划", "设置工具、存储、并行度和执行时间。"],
    drill:["恢复演练", "仅在专用隔离空库验证备份，不恢复生产主库或备库。"], history:["历史与报告", "查看最近的备份校验、演练结果及正式报告。"],
  };
  const selectionGuide = "请在左侧选择业务系统，再选择要查看或配置保护的数据库。";
  const systemRows = new Map();
  let selectedSystem = null, resourceId = "", resources = [], systemVersion = 0, preferredResourceId = "";
  let view = null, canEdit = false, busy = false, systemImportance = "";
  let section = "backup", editing = false, calendarDate = "";
  const node = (tag, content, className) => {
    const value = document.createElement(tag);
    if (content !== undefined) value.textContent = content;
    if (className) value.className = className;
    return value;
  };
  function facts(container, rows) {
    container.replaceChildren(...rows.map(([label, value]) => {
      const row = node("div"), detail = node("dd");
      detail.append(value);
      row.append(node("dt", label), detail);
      return row;
    }));
  }
  function state() {
    for (const {button} of systemRows.values()) button.disabled = busy || editing;
    for (const button of element("databases").querySelectorAll("button")) {
      button.disabled = busy || editing;
      button.setAttribute("aria-pressed", String(button.dataset.resourceId === resourceId));
    }
    for (const group of form.querySelectorAll("[data-editor]")) {
      group.hidden = group.dataset.editor !== section;
      group.disabled = group.hidden || !editing || busy || !canEdit;
    }
    for (const button of element("nav").querySelectorAll("[data-section]")) {
      const active = button.dataset.section === section;
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
      button.disabled = busy || editing;
    }
    for (const group of element("panel").querySelectorAll("[data-overview]")) group.hidden = group.dataset.overview !== section;
    element("section-title").textContent = sections[section][0];
    element("section-description").textContent = sections[section][1];
    element("help").hidden = section !== "config";
    element("schedule").hidden = section !== "backup" || editing;
    element("history").hidden = section !== "history" && (section !== "backup" || editing);
    form.hidden = !editing;
    element("cancel").disabled = busy;
    const edit = element("edit");
    edit.hidden = !canEdit || editing || section === "history";
    edit.disabled = busy;
    edit.textContent = section === "config" ? "编辑保护配置" : section === "drill"
      ? (view?.settings.drill_schedule ? "编辑演练计划" : "新建演练计划")
      : (view?.settings.backup_rotation || view?.settings.backup_schedule ? "编辑备份计划" : "新建备份计划");
    element("save").disabled = busy || !canEdit;
    const toggle = element("toggle-backup");
    toggle.hidden = !canEdit || editing || section !== "backup" || !(view?.next_backup_at || (view?.backup_template_key &&
      (view.settings.backup_schedule || view.settings.backup_rotation) && view.has_backup_identity));
    toggle.disabled = busy;
    toggle.textContent = view?.next_backup_at ? "停用定时备份" : "启用已保存的备份计划";
    const drill = element("toggle-drill");
    drill.hidden = !canEdit || editing || section !== "drill" || !(view?.next_drill_at || view?.drill_preflight_status === "PENDING" || (view?.settings.drill_schedule && view.settings.drill_destination_key &&
      view.settings.restore_template_key && view.has_restore_identity));
    drill.disabled = busy;
    drill.textContent = view?.next_drill_at ? "停用空库演练" :
      view?.drill_preflight_status === "PENDING" ? "取消演练预检" : "预检并启用演练";
  }
  function openSection(next) {
    if (!view || busy || editing) return;
    section = next;
    element("message").textContent = "";
    renderHistory();
    state();
  }
  element("nav").addEventListener("click", event => {
    const button = event.target.closest("button[data-section]");
    if (button) openSection(button.dataset.section);
  });
  element("show-history").addEventListener("click", () => openSection("history"));
  element("edit").addEventListener("click", () => {
    if (!view || !canEdit || busy) return;
    assignDraft(view);
    editing = true;
    element("message").textContent = "";
    state();
  });
  element("cancel").addEventListener("click", () => {
    assignDraft(view);
    editing = false;
    element("message").textContent = "";
    state();
  });
  function clearIdentity() {
    for (const name of ["backup_username", "backup_password", "restore_username", "restore_password"]) fields[name].value = "";
  }
  function usesOracleOs() {
    return view?.settings.tool === "RMAN" && fields.backup_auth_mode.value === "ORACLE_OS";
  }
  function updateBackupAuth() {
    element("backup-auth").hidden = view?.settings.tool !== "RMAN";
    element("backup-password-field").hidden = usesOracleOs();
    element("backup-user-label").textContent = usesOracleOs() ? "备份 OS 用户名（已配置 SYSBACKUP）" : "备份用户名";
  }
  function updateBackupStrategy() {
    const strategy = fields.backup_strategy.value;
    const cron = strategy === "CRON", rotation = strategy === "ROTATION";
    for (const [id, name, visible, required] of [
      ["full-cron", "backup_cron", !rotation, cron],
      ["incremental-cron", "incremental_cron", cron, cron],
      ...["cycle-start", "cycle-days", "full-time", "incremental-time"].map(
        id => [id, id.replace("-", "_"), rotation, rotation]),
    ]) {
      element(id).hidden = !visible;
      fields[name].required = required;
      fields[name].disabled = !visible;
    }
    element("cron-hint").hidden = !cron;
    element("rotation-hint").hidden = !rotation;
    element("baseline-hint").hidden = strategy === "FULL";
  }
  fields.backup_strategy.addEventListener("change", updateBackupStrategy);
  function adviceMatches() {
    return view && (view.settings.importance ?? "") === systemImportance && view.settings.topology === fields.topology.value;
  }
  function renderAdvice() {
    const matches = adviceMatches(), advice = view.advice;
    element("advice").textContent = matches ? advice.explanation : "保存保护配置后，可查看与当前等级和主备形态匹配的经验建议。";
    element("use-advice").hidden = !matches || advice.rto_seconds == null || !canEdit;
    element("use-advice").textContent = `采用建议：RTO ${advice.rto_seconds} 秒 / RPO ${advice.rpo_seconds} 秒`;
  }
  function assignDraft(value) {
    clearIdentity();
    fields.backup_auth_mode.value = "DATABASE_PASSWORD";
    updateBackupAuth();
    for (const name of plain) fields[name].value = value.settings[name] ?? "";
    const rotation = value.settings.backup_rotation;
    fields.backup_timezone.value = rotation?.timezone ?? value.settings.backup_schedule?.timezone ?? "UTC";
    fields.drill_timezone.value = value.settings.drill_schedule?.timezone ?? "UTC";
    fields.backup_strategy.value = rotation ? "ROTATION" : value.settings.backup_incremental_schedule ? "CRON" : "FULL";
    fields.backup_cron.value = value.settings.backup_schedule?.cron ?? "";
    fields.incremental_cron.value = value.settings.backup_incremental_schedule?.cron ?? "";
    fields.cycle_start.value = rotation?.start_date ?? "";
    fields.cycle_days.value = rotation?.cycle_days ?? 3;
    fields.full_time.value = rotation?.full_time ?? "02:00";
    fields.incremental_time.value = rotation?.incremental_time ?? "02:00";
    updateBackupStrategy();
    fields.drill_cron.value = value.settings.drill_schedule?.cron ?? "";
    fields.drill_once.value = value.settings.drill_schedule?.once_at ?? "";
    renderAdvice();
  }
  function assign(value) {
    const previous = view?.settings.backup_rotation;
    view = value;
    assignDraft(value);
    const rotation = value.settings.backup_rotation;
    if (JSON.stringify(previous) !== JSON.stringify(rotation)) calendarDate = rotation ? initialDate(rotation) : "";
    element("selection-guide").hidden = true;
    element("panel").hidden = false;
    element("system-name").textContent = selectedSystem.display_name;
    element("status").textContent = ({NOT_ENROLLED:"未纳入", DRAFT:"已保存 · 未调度", SCHEDULED:"已调度"})[view.status] || view.status;
    element("database-facts").textContent = [view.db_type, view.db_version, view.db_role].filter(Boolean).join(" · ");
    const preflight = {PENDING:"待独立演练主机预检，尚未调度。刷新可查看结果。",
      REJECTED:"演练预检未通过，未调度。请核对隔离空目标、身份和仓库。"};
    element("preflight").textContent = preflight[view.drill_preflight_status] || "";
    element("next-backup").textContent = view.next_backup_at ? "下次备份：" + view.next_backup_at : "";
    element("tool-note").textContent = view.backup_template_key
      ? `内置预设 ${view.backup_template_key} v${view.backup_template_version}，无需提供脚本。`
      : "业务系统登记的数据库版本暂无备份预设，不能启用定时备份。";
    const settings = view.settings, pg = settings.tool === "PGBACKREST", drillPlan = settings.drill_schedule;
    const systemImportanceLabel = levels[systemImportance] || "未设置";
    element("importance").textContent = systemImportanceLabel;
    facts(element("backup-facts"), [["内置备份工具", view.backup_tool_requirement || settings.tool],
      [pg ? "备份 stanza（repo1）" : "备份目录", (pg ? settings.storage_key : settings.backup_directory) || "未设置"], ["并行度", settings.backup_parallelism]]);
    facts(element("config-facts"), [["期望 RTO", settings.rto_seconds == null ? "未设置" : `${settings.rto_seconds} 秒`],
      ["期望 RPO", settings.rpo_seconds == null ? "未设置" : `${settings.rpo_seconds} 秒`],
      ["业务系统重要程度 · 主备形态", `${systemImportanceLabel} · ${fields.topology.selectedOptions[0]?.textContent || "未确认"}`]]);
    facts(element("drill-facts"), [["隔离空环境", settings.drill_destination_key || "未设置"],
      ["执行计划", drillPlan ? `${drillPlan.once_at ? "一次性" : "周期"} · ${drillPlan.once_at || drillPlan.cron} · ${drillPlan.timezone}` : "未设置"],
      ["下次演练", view.next_drill_at || "未调度"]]);
    element("repository").hidden = view.settings.tool !== "PGBACKREST";
    element("directory").hidden = view.settings.tool === "PGBACKREST";
    element("identities").textContent = "备份：" + (view.has_backup_identity ? "已保管" : "未配置") +
      "；演练：" + (view.has_restore_identity ? "已保管" : "未配置") + "。留空保留原凭据。";
    element("checks").replaceChildren(...(view.readiness.length ? [["BACKUP", "备份"], ["DRILL", "恢复演练"]].map(([operation, title]) => {
      const group = node("section", undefined, "protection-permission-group"), list = node("ul");
      for (const check of view.readiness.filter(item => item.operation === operation)) {
        const li = node("li");
        li.append(node("strong", `${labels[check.area]} · ${labels[check.status]}`), node("p", check.guidance));
        list.append(li);
      }
      group.append(node("h3", title), list);
      return group;
    }) : [node("p", "暂无权限检查信息。")]));
    renderSchedule();
    renderHistory();
  }
  function renderHistory() {
    const runs = section === "history" ? view.recent_runs : view.recent_runs.filter(run => run.operation === "BACKUP").slice(0, 2);
    const states = {RUNNING:"运行中", SUCCEEDED:"成功", FAILED:"失败", REJECTED:"已拒绝", SKIPPED:"已跳过", UNKNOWN:"状态未确认"};
    element("recent").hidden = section !== "backup";
    element("no-history").hidden = Boolean(runs.length);
    element("history-table").hidden = !runs.length;
    element("history-rows").replaceChildren(...runs.map(run => {
      const row = node("tr"), report = node("td");
      const kind = run.operation === "BACKUP" ? ({INCREMENTAL:"增量备份", FULL:"全量备份"}[run.backup_kind] || "备份") : "恢复演练";
      if (run.report_id) {
        const link = node("a", "演练报告");
        link.href = "./report-detail.html?id=" + encodeURIComponent(run.report_id);
        report.append(link);
      } else report.textContent = "—";
      row.append(...[run.scheduled_at, kind, states[run.status] || run.status, run.message || "—"].map(value => node("td", value)), report);
      return row;
    }));
  }
  const dayMs = 86_400_000;
  const civilDay = iso => new Date(iso + "T00:00:00Z");
  const dayLabel = value => value.toISOString().slice(0, 10);
  function initialDate(plan) {
    try {
      const today = new Intl.DateTimeFormat("sv-SE", {timeZone:plan.timezone, year:"numeric", month:"2-digit", day:"2-digit"}).format();
      return today > plan.start_date ? today : plan.start_date;
    } catch { return plan.start_date; }
  }
  function plannedDay(date, plan) {
    // 轮转使用自然日差，不使用浏览器时区或 24 小时的实际调度间隔。
    const elapsed = (civilDay(date).getTime() - civilDay(plan.start_date).getTime()) / dayMs;
    if (elapsed < 0) return null;
    const phase = elapsed % plan.cycle_days;
    return phase === 0 ? {kind:"全量", time:plan.full_time, baseline:null}
      : {kind:"增量", time:plan.incremental_time, baseline:dayLabel(new Date(civilDay(date).getTime() - phase * dayMs))};
  }
  function selectDay(date) {
    calendarDate = date;
    renderSchedule();
    element("month-grid").querySelector('[aria-pressed="true"]')?.focus();
  }
  function moveMonth(delta) {
    if (!calendarDate) return;
    const first = civilDay(calendarDate.slice(0, 7) + "-01");
    first.setUTCMonth(first.getUTCMonth() + delta);
    if (first.getUTCFullYear() < 1 || first.getUTCFullYear() > 9999) return;
    const last = new Date(first);
    last.setUTCMonth(first.getUTCMonth() + 1, 0);
    first.setUTCDate(Math.min(Number(calendarDate.slice(8)), last.getUTCDate()));
    calendarDate = dayLabel(first);
    renderSchedule();
  }
  element("previous-month").addEventListener("click", () => moveMonth(-1));
  element("next-month").addEventListener("click", () => moveMonth(1));
  element("first-day").addEventListener("click", () => selectDay(view.settings.backup_rotation.start_date));
  function renderSchedule() {
    const plan = view.settings.backup_rotation;
    element("schedule-status").textContent = view.next_backup_at ? "计划已调度" : "计划未调度";
    element("calendar-layout").hidden = !plan;
    element("rotation-summary").textContent = plan ? `按天轮转 · ${plan.cycle_days} 天一轮 · 自 ${plan.start_date} 起 · ${plan.timezone}` : "";
    const cron = element("cron-summary");
    cron.hidden = Boolean(plan);
    cron.replaceChildren();
    if (!plan) {
      const full = view.settings.backup_schedule, inc = view.settings.backup_incremental_schedule;
      if (full) {
        cron.append(node("p", `全量：${full.cron} · ${full.timezone}`));
        if (inc) cron.append(node("p", `增量：${inc.cron} · 同一时刻重合时全量优先。`));
        cron.append(node("p", "此处仅显示 Cron 规则，不在月历推算日期。"));
      } else cron.append(node("p", "尚未配置备份计划。"));
      return;
    }
    const month = calendarDate.slice(0, 7), first = civilDay(month + "-01"), last = new Date(first);
    last.setUTCMonth(first.getUTCMonth() + 1, 0);
    const offset = (first.getUTCDay() + 6) % 7, chosen = plannedDay(calendarDate, plan);
    element("month-caption").textContent = `${month.slice(0, 4)} 年 ${Number(month.slice(5))} 月`;
    const days = Array.from({length:Math.ceil((offset + last.getUTCDate()) / 7) * 7}, (_, index) => {
      const day = index - offset + 1;
      if (day < 1 || day > last.getUTCDate()) {
        const blank = node("span", "", "protection-day-empty");
        blank.setAttribute("aria-hidden", "true");
        return blank;
      }
      const date = month + "-" + String(day).padStart(2, "0"), slot = plannedDay(date, plan);
      const button = node("button", undefined, "protection-day" + (date === chosen?.baseline ? " is-baseline" : ""));
      button.type = "button";
      button.setAttribute("aria-pressed", String(date === calendarDate));
      button.setAttribute("aria-label", `${date} ${slot ? `${slot.kind} ${slot.time}` : "无排程"}`);
      button.append(node("strong", day));
      if (slot) button.append(node("span", slot.kind, "protection-tag" + (slot.kind === "全量" ? " is-full" : "")));
      button.addEventListener("click", () => selectDay(date));
      return button;
    });
    element("month-grid").replaceChildren(...[..."一二三四五六日"].map(day => node("span", "周" + day, "protection-weekday")), ...days);
    const detail = element("date-detail");
    detail.replaceChildren(node("h4", "所选日期"), node("strong", calendarDate, "protection-date"));
    if (chosen) {
      const list = node("dl"), rows = [["备份类型", chosen.kind], ["计划时间", chosen.time], ["时区", plan.timezone]];
      if (chosen.baseline) {
        const link = node("button", chosen.baseline + " →", "protection-link");
        link.type = "button";
        link.addEventListener("click", () => selectDay(chosen.baseline));
        rows.push(["计划全量基线", link]);
      }
      facts(list, rows);
      detail.append(list);
      if (chosen.baseline) detail.append(node("p", "对应全量未通过校验时，不执行该增量，也不会改跑全量。"));
    } else detail.append(node("p", "早于轮转起点，无排程。"));
  }
  function resetProtection() {
    if (permissionDrawer.open) permissionDrawer.close();
    view = null;
    resourceId = "";
    section = "backup"; editing = false; calendarDate = "";
    clearIdentity();
    element("panel").hidden = true;
    element("selection-guide").hidden = false;
    element("guide").textContent = selectionGuide;
    element("message").textContent = "";
  }
  async function selectResource(resource) {
    if (busy || (resourceId === resource.resource_id && view)) return;
    resetProtection();
    resourceId = resource.resource_id;
    element("database-name").textContent = resource.display_name;
    element("guide").textContent = "正在读取数据保护配置…";
    busy = true; state();
    try { assign(await request(url())); }
    catch { element("guide").textContent = "无法读取数据保护，请检查资源访问权限。"; }
    finally { busy = false; state(); }
  }
  const matches = (query, ...values) => {
    const needle = query.trim().toLocaleLowerCase();
    return values.some(value => value?.toLocaleLowerCase().includes(needle));
  };
  function renderDatabases() {
    const search = element("database-search").value;
    const visible = resources.filter(item => matches(search, item.display_name, item.resource_code,
      item.database.db_type, item.database.version_code));
    element("database-search-field").hidden = resources.length < 2;
    element("database-status").textContent = visible.length ? ""
      : resources.length ? "没有匹配的数据库。" : "此业务系统没有可配置保护的数据库。";
    element("database-status").hidden = Boolean(visible.length);
    element("databases").replaceChildren(...visible.map(resource => {
      const row = document.createElement("li"), button = document.createElement("button");
      button.className = "protection-database-button";
      button.type = "button";
      button.dataset.resourceId = resource.resource_id;
      const title = document.createElement("strong"), description = document.createElement("small");
      title.textContent = resource.display_name;
      description.textContent = [resource.database.db_type, resource.database.version_code].filter(Boolean).join(" ");
      button.append(title, description);
      button.addEventListener("click", () => selectResource(resource));
      row.append(button);
      return row;
    }));
    state();
  }
  async function selectSystem(system, preferred = "") {
    if (busy) return;
    const generation = ++systemVersion;
    preferredResourceId = preferred;
    selectedSystem = system;
    resetProtection();
    resources = [];
    systemImportance = "";
    element("retry-system").hidden = true;
    element("database-search-field").hidden = true;
    element("database-search").value = "";
    element("databases").replaceChildren();
    const branch = element("branch");
    branch.hidden = !system;
    for (const {item, button, chevron} of systemRows.values()) {
      const active = item === system;
      button.setAttribute("aria-expanded", String(active));
      chevron.textContent = active ? "▾" : "▸";
    }
    if (!system) return;
    systemRows.get(system.business_system_id).row.append(branch);
    element("database-status").hidden = false;
    element("database-status").textContent = "正在读取数据库…";
    try {
      const detail = await request("/business-systems/" + encodeURIComponent(system.business_system_id));
      if (generation !== systemVersion) return;
      resources = detail.resources.filter(entry => entry.status === "ACTIVE"
        && entry.resource.resource_type_key === "DATABASE" && entry.resource.database).map(entry => entry.resource);
      systemImportance = ["", "LOW", "LOWER", "MEDIUM", "HIGH", "CRITICAL"][detail.importance_level] || "";
      renderDatabases();
      const resource = resources.find(item => item.resource_id === preferred);
      if (resource) await selectResource(resource);
    } catch {
      if (generation !== systemVersion) return;
      element("database-status").textContent = "无法读取该业务系统的数据库，请重试。";
      element("retry-system").hidden = false;
    }
  }
  function filterSystems() {
    let visible = false;
    for (const {item, row} of systemRows.values()) {
      row.hidden = !matches(element("system-search").value, item.display_name, item.system_code);
      visible ||= !row.hidden;
    }
    element("catalog-status").hidden = visible;
    element("catalog-status").textContent = visible ? "" : systemRows.size
      ? "没有匹配的业务系统。" : "当前资源域没有包含数据库的业务系统。";
  }
  async function loadCatalog() {
    element("retry-catalog").hidden = true;
    element("system-search").disabled = true;
    element("catalog-status").hidden = false;
    element("catalog-status").textContent = "正在读取业务系统…";
    try {
      const page = await request("/business-systems?limit=200");
      const list = element("systems");
      systemRows.clear();
      list.replaceChildren();
      for (const item of page.items) {
        const count = item.resource_type_summary.find(entry => entry.resource_type_key === "DATABASE")?.count ?? 0;
        if (count < 1) continue;
        const row = document.createElement("li"), button = document.createElement("button"), chevron = document.createElement("span");
        button.type = "button";
        button.className = "protection-system-button";
        button.setAttribute("aria-expanded", "false");
        chevron.setAttribute("aria-hidden", "true");
        chevron.textContent = "▸";
        const total = node("small", count + " 库");
        total.setAttribute("aria-hidden", "true");
        button.append(chevron, item.display_name, total);
        button.addEventListener("click", () => selectSystem(selectedSystem === item ? null : item));
        row.append(button);
        list.append(row);
        systemRows.set(item.business_system_id, {item, row, button, chevron});
      }
      element("system-search").disabled = false;
      filterSystems();
      const preferred = systemRows.get(query.get("businessSystemId"));
      if (preferred) await selectSystem(preferred.item, query.get("resourceId"));
    } catch {
      element("catalog-status").textContent = "无法读取业务系统，请重试。";
      element("retry-catalog").hidden = false;
    }
  }
  element("system-search").addEventListener("input", filterSystems);
  element("database-search").addEventListener("input", renderDatabases);
  element("retry-catalog").addEventListener("click", loadCatalog);
  element("retry-system").addEventListener("click", () => selectSystem(selectedSystem, preferredResourceId));
  element("help").addEventListener("click", () => permissionDrawer.showModal());
  element("close-help").addEventListener("click", () => permissionDrawer.close());
  permissionDrawer.addEventListener("cancel", event => event.preventDefault());
  element("use-advice").addEventListener("click", () => {
    if (!adviceMatches() || view.advice.rto_seconds == null) return;
    fields.rto_seconds.value = view.advice.rto_seconds;
    fields.rpo_seconds.value = view.advice.rpo_seconds;
  });
  fields.topology.addEventListener("change", renderAdvice);
  fields.backup_auth_mode.addEventListener("change", () => {
    fields.backup_password.value = "";
    updateBackupAuth();
  });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !canEdit || !view || !editing) return;
    const value = name => fields[name].value;
    for (const kind of ["backup", "restore"]) {
      if (kind === "backup" && usesOracleOs()) continue;
      if (Boolean(value(kind + "_username")) !== Boolean(value(kind + "_password"))) {
        element("message").textContent = "更新账号时，用户名与密码须同时填写。"; return;
      }
    }
    if (value("drill_cron") && value("drill_once")) {
      element("message").textContent = "演练只能选择一次性时间或 Cron。"; return;
    }
    const once = value("drill_once") ? new Date(value("drill_once")) : null;
    if (once && (!/(Z|[+-]\d{2}:\d{2})$/i.test(value("drill_once")) || !Number.isFinite(once.getTime()))) {
      element("message").textContent = "一次性演练时间须包含有效的时区偏移。"; return;
    }
    const schedule = (timezone, cron, when = null) => cron || when ? {timezone, cron:cron || null, once_at:when?.toISOString() ?? null} : null;
    const strategy = value("backup_strategy");
    if (strategy === "CRON" && !(value("backup_cron").trim() && value("incremental_cron").trim())) {
      element("message").textContent = "请填写全量和增量 Cron。"; return;
    }
    const settings = {tool:view.settings.tool,
      importance:(section === "config" ? systemImportance : "") || view.settings.importance || null,
      ...Object.fromEntries(plain.map(name => {
      const entry = value(name);
      return [name, !entry ? null : name.endsWith("_seconds") || name === "backup_parallelism" ? Number(entry) : entry];
    })), backup_schedule:strategy !== "ROTATION" ? schedule(value("backup_timezone").trim(), value("backup_cron").trim()) : null,
    backup_incremental_schedule:strategy === "CRON" ? schedule(value("backup_timezone").trim(), value("incremental_cron").trim()) : null,
    backup_rotation:strategy === "ROTATION" ? {
      timezone:value("backup_timezone").trim(), start_date:value("cycle_start"), cycle_days:Number(value("cycle_days")),
      full_time:value("full_time"), incremental_time:value("incremental_time"),
    } : null,
    drill_schedule:schedule(value("drill_timezone").trim(), value("drill_cron"), once)};
    const identity = kind => value(kind + "_password") ? {username:value(kind + "_username"), password:value(kind + "_password")} : null;
    const backup = usesOracleOs()
      ? (value("backup_username") ? {auth_mode:"ORACLE_OS", username:value("backup_username").trim()} : null)
      : identity("backup");
    const payload = {expected_revision:view.revision, settings, backup_identity:backup, restore_identity:identity("restore")};
    busy = true; state();
    try {
      assign(await request(url(), {method:"PUT", body:JSON.stringify(payload)}));
      editing = false;
      element("message").textContent = "草稿已保存。备份和演练需分别重新启用。";
    } catch { element("message").textContent = "未能保存：请检查字段或刷新后重试。"; }
    finally { clearIdentity(); busy = false; state(); }
  });
  const changeSchedule = async kind => {
    if (busy || !canEdit || !view) return;
    const previous = view, id = resourceId, drill = kind === "drill";
    const enabled = !(drill ? view.next_drill_at || view.drill_preflight_status === "PENDING" : view.next_backup_at);
    const label = drill ? "空库演练" : "定时备份";
    if (!await shell.confirmAction({
      title: enabled ? "预检并启用已保存的" + label + "？" : "停用" + label + "？",
      message: enabled ? (drill ? "只使用已保存配置。未核验专用隔离空环境时，不会启用。" : "仅使用已保存的参数。预检通过后才会调度。")
        : "停止后续调度，不删除已有记录或产物，也不中断已开始写入的作业。",
    })) return;
    if (busy || view !== previous || resourceId !== id) return;
    busy = true; state();
    try {
      assign(await request(url() + "/" + kind + "-schedule", {
        method:"PUT", body:JSON.stringify({expected_revision:previous.revision, enabled}),
      }));
      element("message").textContent = drill && enabled && !view.next_drill_at
        ? "未开始调度，请查看演练预检状态。" : label + (enabled ? "已启用。" : "已停用。");
    } catch {
      element("message").textContent = enabled
        ? (drill ? "未启用：请检查已保存的演练计划与独立空环境预检。" : "未启用：请检查已保存的计划和执行环境预检。")
        : "未能停用，请刷新后重试。";
    } finally { busy = false; state(); }
  };
  element("toggle-backup").addEventListener("click", () => changeSchedule("backup"));
  element("toggle-drill").addEventListener("click", () => changeSchedule("drill"));
  shell.ready.then(async access => {
    canEdit = (access.permissions || []).includes("aiops:target_manage");
    await loadCatalog();
  }).catch(() => { element("catalog-status").textContent = "无法读取业务系统访问权限。"; });
})();
