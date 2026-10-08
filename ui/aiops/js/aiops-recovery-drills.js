(function () {
  "use strict";
  const appApi = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const sourceOptions = {
    ORACLE: [["ORACLE_RMAN", "Oracle RMAN"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    POSTGRESQL: [["POSTGRESQL_BASEBACKUP", "pg_basebackup"], ["POSTGRESQL_PGBACKREST", "pgBackRest"], ["POSTGRESQL_BARMAN", "Barman"], ["POSTGRESQL_WALG", "WAL-G"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
    MYSQL: [["MYSQL_XTRABACKUP", "Percona XtraBackup"], ["MYSQL_ENTERPRISE_BACKUP", "MySQL Enterprise Backup"], ["MYSQL_LOGICAL_DUMP", "逻辑导出"], ["FILESYSTEM_SNAPSHOT", "文件系统快照"], ["STORAGE_SNAPSHOT", "存储快照"], ["CLOUD_MANAGED_BACKUP", "云托管备份"], ["THIRD_PARTY_BACKUP", "第三方备份"]],
  };
  let currentTarget = null;

  function renderProfile(profile, target) {
    const panel = document.getElementById("recovery-profile-summary");
    const state = document.getElementById("recovery-workspace-state");
    state.className = `ops-badge ${profile ? "good" : "warn"}`;
    state.textContent = profile ? `目标 v${profile.version_no}` : "目标未配置";
    panel.innerHTML = `<dl class="ops-detail"><dt>数据库</dt><dd>${shell.escape(target.db_type)}</dd><dt>RPO</dt><dd>${profile ? `${profile.rpo_seconds / 60} 分钟` : "未配置"}</dd><dt>RTO</dt><dd>${profile ? `${profile.rto_seconds / 60} 分钟` : "未配置"}</dd><dt>最低等级</dt><dd>${shell.escape(profile?.required_assurance_level || "未配置")}</dd></dl>`;
  }

  function renderHistory(items) {
    const panel = document.getElementById("recovery-drill-history");
    if (!items.length) {
      panel.innerHTML = '<p class="ops-empty">当前 Target 尚未登记恢复演练。</p>';
      return;
    }
    panel.innerHTML = `<table class="ops-table"><thead><tr><th>演练时间</th><th>等级</th><th>结果</th><th>审核状态</th><th>来源与信任</th><th>实际 RPO/RTO</th><th>操作</th></tr></thead><tbody>${items.map((item) => `<tr><td>${shell.escape(shell.fmt(item.simulated_failure_at))}</td><td>${shell.escape(item.assurance_level)}</td><td>${shell.badge(item.result)}</td><td>${shell.badge(item.status)}</td><td>${shell.escape(item.backup_source_type)}<br><small>${shell.escape(item.source_trust_level)}</small></td><td>${item.achieved_rpo_seconds ?? "—"}s / ${item.achieved_rto_seconds ?? "—"}s</td><td>${item.status === "SUBMITTED" ? `<div class="ops-actions"><button type="button" data-review-drill="${shell.escape(item.drill_id)}" data-version="${shell.escape(item.row_version)}" data-decision="VERIFY">通过</button><button type="button" data-review-drill="${shell.escape(item.drill_id)}" data-version="${shell.escape(item.row_version)}" data-decision="REJECT">拒绝</button></div>` : "—"}</td></tr>`).join("")}</tbody></table>`;
  }

  async function loadWorkspace(targetId) {
    const [target, profile, drills] = await Promise.all([
      KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}`),
      KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/recovery-profile`),
      KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(targetId)}/recovery-drills`),
    ]);
    currentTarget = target;
    renderProfile(profile, target);
    renderHistory(drills?.items || []);
    document.getElementById("recovery-target-detail-link").href = `./target-detail.html?id=${encodeURIComponent(targetId)}`;
    document.getElementById("recovery-backup-source").innerHTML = (sourceOptions[target.db_type] || []).map(([value, label]) => `<option value="${value}">${label}</option>`).join("");
    document.getElementById("submit-recovery-drill").disabled = false;
  }

  async function reviewDrill(button) {
    const verb = button.dataset.decision === "VERIFY" ? "通过" : "拒绝";
    if (!confirm(`确认${verb}这条演练记录吗？审核不会把人工证据提升为系统直采证据。`)) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(currentTarget.target_id)}/recovery-drills/${encodeURIComponent(button.dataset.reviewDrill)}:review`, {
        method: "POST",
        headers: { "If-Match": `"rv-${button.dataset.version}"`, "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({ decision: button.dataset.decision, review_note: null }),
      });
      shell.toast(`演练记录已${verb}`);
      await loadWorkspace(currentTarget.target_id);
    } catch (error) {
      shell.toast(error.message);
      button.disabled = false;
    }
  }

  async function submitDrill(event) {
    event.preventDefault();
    if (!currentTarget) return;
    const form = event.currentTarget;
    const button = document.getElementById("submit-recovery-drill");
    const result = document.getElementById("recovery-drill-result");
    const iso = (name) => form.elements[name].value ? new Date(form.elements[name].value).toISOString() : null;
    const failureAt = new Date(form.simulated_failure_at.value);
    const recoveredThrough = form.recovered_through_at.value ? new Date(form.recovered_through_at.value) : null;
    const validatedAt = form.service_validated_at.value ? new Date(form.service_validated_at.value) : null;
    if (recoveredThrough && recoveredThrough > failureAt) {
      result.textContent = "实际恢复到的数据时间不能晚于模拟故障时间。";
      result.dataset.tone = "bad";
      return;
    }
    if (validatedAt && validatedAt < failureAt) {
      result.textContent = "验证完成时间不能早于模拟故障时间。";
      result.dataset.tone = "bad";
      return;
    }
    if (
      form.result.value === "PASS"
      && ["DATABASE_OPEN", "APPLICATION_VALIDATED"].includes(form.assurance_level.value)
      && !form.service_validated_at.value
    ) {
      result.textContent = "验证范围为数据库启动或业务可用性且结果为通过，请填写验证完成时间。";
      result.dataset.tone = "bad";
      return;
    }
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${appApi}/targets/${encodeURIComponent(currentTarget.target_id)}/recovery-drills`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({
          scenario: form.scenario.value,
          assurance_level: form.assurance_level.value,
          backup_source_type: form.backup_source_type.value,
          environment: form.environment.value,
          result: form.result.value,
          simulated_failure_at: iso("simulated_failure_at"),
          recovered_through_at: iso("recovered_through_at"),
          service_validated_at: iso("service_validated_at"),
          recovery_marker: { kind: currentTarget.db_type },
          evidence: [],
          notes: String(form.notes.value || "").trim() || null,
        }),
      });
      form.reset();
      result.textContent = "演练记录已提交，等待人工审核。";
      result.dataset.tone = "good";
      await loadWorkspace(currentTarget.target_id);
    } catch (error) {
      result.textContent = error.message;
      result.dataset.tone = "bad";
    } finally {
      button.disabled = !currentTarget;
    }
  }

  async function initialize() {
    const payload = await KBotAIOpsAuth.request(`${appApi}/targets?limit=200`);
    const targets = payload?.items || [];
    const select = document.getElementById("recovery-target-select");
    select.insertAdjacentHTML("beforeend", targets.map((target) => `<option value="${shell.escape(target.target_id)}">${shell.escape(target.display_name)} · ${shell.escape(target.db_type)}</option>`).join(""));
    select.addEventListener("change", async () => {
      if (!select.value) return;
      currentTarget = null;
      document.getElementById("submit-recovery-drill").disabled = true;
      document.getElementById("recovery-drill-form").reset();
      document.getElementById("recovery-drill-result").textContent = "";
      history.replaceState(null, "", `?target_id=${encodeURIComponent(select.value)}`);
      try { await loadWorkspace(select.value); }
      catch (error) { shell.toast(error.message); }
    });
    document.getElementById("recovery-drill-form").addEventListener("submit", submitDrill);
    document.getElementById("recovery-drill-history").addEventListener("click", (event) => {
      const button = event.target.closest("[data-review-drill]");
      if (button) void reviewDrill(button);
    });
    const requested = new URLSearchParams(location.search).get("target_id");
    const selected = targets.some((target) => String(target.target_id) === requested)
      ? requested : targets[0]?.target_id;
    if (selected) {
      select.value = selected;
      await loadWorkspace(selected);
    } else {
      document.getElementById("recovery-profile-summary").innerHTML = '<p class="ops-empty">当前没有可管理的 Target。</p>';
    }
  }

  shell.ready.then(() => initialize().catch((error) => shell.toast(error.message)));
})();
