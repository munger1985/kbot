(function () {
  "use strict";
  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const select = document.getElementById("protection-system");
  const form = document.getElementById("protection-form");
  const empty = document.getElementById("protection-empty");
  const result = document.getElementById("protection-result");
  const request = (path, options) => KBotAIOpsAuth.request(`${api}${path}`, options);
  const sourceTypes = {
    ORACLE: [["ORACLE_RMAN", "Oracle RMAN"]],
    MYSQL: [["MYSQL_XTRABACKUP", "Percona XtraBackup"], ["MYSQL_ENTERPRISE_BACKUP", "MySQL Enterprise Backup"], ["MYSQL_LOGICAL_DUMP", "逻辑导出"]],
    POSTGRESQL: [["POSTGRESQL_BASEBACKUP", "pg_basebackup"], ["POSTGRESQL_PGBACKREST", "pgBackRest"], ["POSTGRESQL_BARMAN", "Barman"], ["POSTGRESQL_WALG", "WAL-G"]],
  };
  const common = [["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]];
  let databaseResources = [];

  function set(name, value) { const control = form.elements[name]; if (control) control.value = value; }
  async function loadProfile() {
    if (!select.value) { form.hidden = true; empty.hidden = false; return; }
    const item = databaseResources.find((row) => row.resource_id === select.value);
    const profile = await request(`/managed-resources/${encodeURIComponent(select.value)}/protection-profile`);
    form.hidden = false;
    empty.hidden = true;
    const value = (name, fallback) => profile?.[name] ?? fallback;
    ["backup_policy_enabled", "drill_required"].forEach((name) => set(name, String(value(name, true))));
    [["backup_frequency_hours", 24], ["backup_success_max_age_hours", 26], ["backup_retention_days", 30], ["rpo_seconds", 3600], ["rto_seconds", 14400], ["required_drill_interval_days", 180], ["required_assurance_level", "DATABASE_OPEN"], ["source_note", ""]].forEach(([name, fallback]) => set(name, value(name, fallback)));
    const selected = new Set(profile?.required_backup_source_types || []);
    const options = [...new Map([...(sourceTypes[item.db_type] || []), ...common].map((row) => [row[0], row])).values()];
    form.elements.required_backup_source_types.innerHTML = options.map(([key, label]) => `<option value="${key}" ${selected.has(key) ? "selected" : ""}>${label}</option>`).join("");
  }

  async function load() {
    const page = await request("/business-systems?limit=200");
    const systems = await Promise.all((page.items || []).filter((item) => item.status === "ENABLED").map((item) => request(`/business-systems/${encodeURIComponent(item.business_system_id)}`)));
    databaseResources = systems.flatMap((system) => (system.resources || []).filter((item) => item.status === "ACTIVE" && item.resource.resource_type_key === "DATABASE" && item.resource.database).map((item) => ({ resource_id: item.resource.resource_id, display_name: item.resource.display_name, db_type: item.resource.database.db_type, business_system_name: system.display_name })));
    select.innerHTML = '<option value="">请选择数据库资源</option>' + databaseResources.map((item) => `<option value="${shell.escape(item.resource_id)}">${shell.escape(item.business_system_name)} / ${shell.escape(item.display_name)}</option>`).join("");
    const requested = new URLSearchParams(location.search).get("resource_id");
    if (requested && databaseResources.some((item) => item.resource_id === requested)) { select.value = requested; await loadProfile(); }
  }

  form.onsubmit = async (event) => {
    event.preventDefault();
    const sources = [...form.elements.required_backup_source_types.selectedOptions].map((option) => option.value);
    if ((form.elements.backup_policy_enabled.value === "true" || form.elements.drill_required.value === "true") && !sources.length) { result.textContent = "请至少选择一个备份来源"; result.dataset.tone = "bad"; return; }
    const payload = {
      backup_policy_enabled: form.elements.backup_policy_enabled.value === "true",
      backup_frequency_hours: Number(form.elements.backup_frequency_hours.value),
      backup_success_max_age_hours: Number(form.elements.backup_success_max_age_hours.value),
      backup_retention_days: Number(form.elements.backup_retention_days.value),
      rpo_seconds: Number(form.elements.rpo_seconds.value), rto_seconds: Number(form.elements.rto_seconds.value),
      drill_required: form.elements.drill_required.value === "true",
      required_drill_interval_days: Number(form.elements.required_drill_interval_days.value),
      required_assurance_level: form.elements.required_assurance_level.value,
      required_backup_source_types: sources, rto_clock_basis: "SERVICE_UNAVAILABLE_TO_VALIDATED",
      source_note: form.elements.source_note.value.trim() || null,
    };
    try {
      await request(`/managed-resources/${encodeURIComponent(select.value)}/protection-profile`, { method: "PUT", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify(payload) });
      shell.toast("数据库资源保护配置已保存为新版本");
      result.textContent = "保存成功";
      result.dataset.tone = "good";
    } catch (error) { result.textContent = error.message; result.dataset.tone = "bad"; }
  };
  select.onchange = () => void loadProfile().catch((error) => shell.toast(error.message));
  shell.ready.then(load).catch((error) => shell.toast(error.message));
})();
