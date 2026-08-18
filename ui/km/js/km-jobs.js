(function () {
  "use strict";
  const base = "/api/v1/apps/km-asset";
  const $ = (id) => document.getElementById(id);
  const expandedAssets = new Set();
  const expandedChains = new Set();
  const expandedSourceJobs = new Set();
  const sourceJobVisibleCounts = new Map();
  const detailRows = [];
  const SOURCE_JOB_PAGE_SIZE = 10;
  let rows = [];
  let assets = [];
  let processing = [];
  let sources = [];

  const stepNames = {
    SOURCE_SYNC: "MetaDB 来源扫描",
    ATTACHMENT_DOWNLOAD: "附件下载与 Bundle 组装",
    KC_INGEST: "提交 Knowledge Core",
    KC_STATUS_SYNC: "Knowledge Core 状态跟踪",
    SOURCE_STATUS_UPDATE: "回写 MetaDB 状态",
    RETRY: "重新执行 Asset 同步",
    PARSE: "文档解析",
    PROFILE: "检索画像生成",
    INDEX: "全文与向量索引",
    VISUAL: "视觉内容处理",
  };

  async function initialize() {
    try {
      sources = KBotKmApi.items(await KBotKmApi.request(`${base}/sources`));
      $("job-source").insertAdjacentHTML("beforeend", sources.map((row) => `<option value="${KBotKmShell.escapeHtml(row.source_id)}">${KBotKmShell.escapeHtml(row.display_name)}</option>`).join(""));
      await load();
    } catch (error) { KBotKmShell.showError(error, "任务页面加载失败"); }
  }

  function sourceName(id) {
    return sources.find((row) => String(row.source_id) === String(id))?.display_name || KBotKmShell.shortId(id);
  }

  async function loadAllAssets(sourceId) {
    const output = [];
    const limit = 500;
    for (let offset = 0; ; offset += limit) {
      const page = KBotKmApi.items(await KBotKmApi.request(`${base}/assets${KBotKmApi.query({ source_id: sourceId, offset, limit })}`));
      output.push(...page);
      if (page.length < limit) return output;
    }
  }

  async function load() {
    const sourceId = $("job-source").value;
    const limit = Number($("job-limit").value);
    const note = $("job-data-note");
    note.hidden = true;
    try {
      const [jobRows, assetRows, processingResult] = await Promise.all([
        KBotKmApi.request(`${base}/jobs${KBotKmApi.query({ source_id: sourceId, limit })}`),
        loadAllAssets(sourceId),
        KBotKmApi.request(`${base}/jobs/processing${KBotKmApi.query({ source_id: sourceId, limit: 2000 })}`).catch((error) => ({ error })),
      ]);
      rows = KBotKmApi.items(jobRows);
      assets = assetRows;
      if (processingResult.error) {
        processing = [];
        note.textContent = "Knowledge Core 步骤暂时不可用；当前仅显示 KM Asset 本地任务。";
        note.hidden = false;
      } else {
        processing = KBotKmApi.items(processingResult);
      }
      render();
    } catch (error) { KBotKmShell.showError(error, "同步任务查询失败"); }
  }

  function timeValue(value) { return value ? new Date(value).getTime() || 0 : 0; }
  function latestTime(items) { return Math.max(0, ...items.map((item) => timeValue(item.completed_at || item.started_at || item.created_at))); }
  function chainStatus(steps) {
    for (const status of ["RUNNING", "RETRY_WAIT", "PENDING"]) {
      if (steps.some((item) => item.row.status === status)) return status;
    }
    if (!steps.length) return "PENDING";
    const latest = [...steps].sort((left, right) => timeValue(right.row.completed_at || right.row.started_at || right.row.created_at) - timeValue(left.row.completed_at || left.row.started_at || left.row.created_at))[0];
    return latest.row.status || "PENDING";
  }
  function stepDetail(row, origin) {
    const index = detailRows.push({ origin, ...row }) - 1;
    return `<button class="small" data-detail="${index}">详情</button>`;
  }
  function renderStep(step, index) {
    const row = step.row;
    const status = row.status || "PENDING";
    const attempt = row.attempt_count == null ? "—" : `${row.attempt_count} / ${row.max_attempts || "—"}`;
    const document = row.document_version_id ? `<span class="km-step-document">文件 ${KBotKmShell.escapeHtml(KBotKmShell.shortId(row.document_version_id))}</span>` : "";
    const error = row.failure_message || row.error_message;
    return `<li class="km-job-step"><span class="km-step-index">${index + 1}</span><span class="km-step-line" aria-hidden="true"></span><div class="km-step-body"><div class="km-step-head"><div><strong>${KBotKmShell.escapeHtml(stepNames[row.job_type] || row.job_type)}</strong><span class="km-step-origin">${KBotKmShell.escapeHtml(step.origin)}</span>${document}</div>${KBotKmShell.badge(status)}</div><div class="km-step-meta"><span>${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.started_at || row.created_at || row.available_at))}</span><span>完成 ${KBotKmShell.escapeHtml(KBotKmShell.formatDate(row.completed_at))}</span><span>尝试 ${KBotKmShell.escapeHtml(attempt)}</span></div>${error ? `<p class="km-step-error">${KBotKmShell.escapeHtml(error)}</p>` : ""}<div class="km-step-action">${stepDetail(row, step.origin)}</div></div></li>`;
  }
  function processingSteps(item) {
    return (item?.jobs || []).map((row) => ({ origin: "Knowledge Core", row: { ...row, status: row.job_status } }));
  }
  function localSteps(items) { return items.map((row) => ({ origin: "KM Asset", row })); }

  function chainsFor(asset) {
    const local = rows.filter((row) => String(row.km_asset_id) === String(asset.km_asset_id));
    const groups = new Map();
    for (const row of local) {
      const key = String(row.asset_revision_id || `unversioned:${row.job_id}`);
      if (!groups.has(key)) groups.set(key, { key, revisionId: row.asset_revision_id, local: [], kc: null });
      groups.get(key).local.push(row);
    }
    const kc = processing.filter((item) => String(item.bundle_revision_id) === String(asset.kc_bundle_revision_id));
    for (const item of kc) {
      const key = String(asset.current_revision_id || `kc:${item.bundle_revision_id}`);
      if (!groups.has(key)) groups.set(key, { key, revisionId: asset.current_revision_id, local: [], kc: null });
      groups.get(key).kc = item;
    }
    return [...groups.values()].map((chain) => {
      const steps = [...localSteps(chain.local), ...processingSteps(chain.kc)].sort((left, right) => timeValue(left.row.started_at || left.row.created_at || left.row.available_at) - timeValue(right.row.started_at || right.row.created_at || right.row.available_at));
      return { ...chain, steps, status: chainStatus(steps), latest: latestTime(steps.map((item) => item.row)) };
    }).sort((left, right) => right.latest - left.latest);
  }

  function renderChain(asset, chain, index) {
    const key = `${asset.km_asset_id}:${chain.key}`;
    const expanded = expandedChains.has(key);
    const kcMeta = chain.kc ? ` · KC ${KBotKmShell.shortId(chain.kc.bundle_revision_id)}` : "";
    return `<section class="km-job-chain"><button class="km-chain-toggle" data-chain="${KBotKmShell.escapeHtml(key)}" aria-expanded="${expanded}"><span class="km-tree-caret" aria-hidden="true"></span><span><strong>同步链 ${index + 1}</strong><small>Asset Revision ${KBotKmShell.escapeHtml(KBotKmShell.shortId(chain.revisionId))}${KBotKmShell.escapeHtml(kcMeta)}</small></span><span class="km-chain-summary">${chain.steps.length} 个步骤 · ${KBotKmShell.formatDate(chain.latest)}</span>${KBotKmShell.badge(chain.status)}</button><ol class="km-job-steps" ${expanded ? "" : "hidden"}>${chain.steps.map(renderStep).join("") || '<li class="km-tree-empty">该同步链暂无步骤记录</li>'}</ol></section>`;
  }

  function renderAsset(asset) {
    const chains = chainsFor(asset);
    const expanded = expandedAssets.has(String(asset.km_asset_id));
    const jobCount = chains.reduce((total, chain) => total + chain.steps.length, 0);
    const latest = Math.max(timeValue(asset.completed_at || asset.last_update_time), ...chains.map((chain) => chain.latest));
    return `<article class="km-asset-node"><button class="km-asset-toggle" data-asset="${KBotKmShell.escapeHtml(asset.km_asset_id)}" aria-expanded="${expanded}"><span class="km-tree-caret" aria-hidden="true"></span><span class="km-asset-identity"><strong>${KBotKmShell.escapeHtml(asset.asset_title || "未命名 Asset")}</strong><small>${KBotKmShell.escapeHtml(asset.external_asset_id)} · ${KBotKmShell.escapeHtml(sourceName(asset.source_id))}</small></span><span class="km-asset-summary">${chains.length} 条同步链 · ${jobCount} 个步骤</span><span class="km-asset-time">${KBotKmShell.escapeHtml(KBotKmShell.formatDate(latest))}</span>${KBotKmShell.badge(asset.ingestion_status)}</button><div class="km-asset-children" ${expanded ? "" : "hidden"}>${chains.map((chain, index) => renderChain(asset, chain, index)).join("") || '<p class="km-tree-empty">该 Asset 尚无同步任务</p>'}</div></article>`;
  }

  function renderSourceJobs() {
    const sourceRows = rows.filter((row) => !row.km_asset_id);
    const container = $("source-job-tree");
    if (!sourceRows.length) { container.innerHTML = '<p class="km-tree-empty">暂无来源级任务</p>'; return; }
    const groups = new Map();
    for (const row of sourceRows) {
      const key = String(row.source_id || "unknown");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(row);
    }
    const ordered = [...groups.entries()].sort((left, right) => {
      return latestTime(right[1]) - latestTime(left[1]);
    });
    container.innerHTML = ordered.map(([key, items]) => {
      const expanded = expandedSourceJobs.has(key);
      const sorted = [...items].sort((left, right) => {
        return timeValue(right.created_at) - timeValue(left.created_at);
      });
      const visibleCount = Math.min(
        sourceJobVisibleCounts.get(key) || SOURCE_JOB_PAGE_SIZE,
        sorted.length,
      );
      const visible = sorted.slice(0, visibleCount);
      const remaining = sorted.length - visible.length;
      const status = chainStatus(localSteps(sorted));
      const latest = latestTime(sorted);
      const more = remaining > 0
        ? `<button class="km-source-more" data-source-more="${KBotKmShell.escapeHtml(key)}">再显示 ${Math.min(SOURCE_JOB_PAGE_SIZE, remaining)} 条<span>剩余 ${remaining} 条</span></button>`
        : "";
      return `<article class="km-source-node"><button class="km-source-toggle" data-source-jobs="${KBotKmShell.escapeHtml(key)}" aria-expanded="${expanded}"><span class="km-tree-caret" aria-hidden="true"></span><span class="km-source-identity"><strong>${KBotKmShell.escapeHtml(sourceName(key))}</strong><small>MetaDB 扫描与来源调度</small></span><span class="km-source-summary">${sorted.length} 条任务</span><span class="km-source-time">最近 ${KBotKmShell.escapeHtml(KBotKmShell.formatDate(latest))}</span>${KBotKmShell.badge(status)}</button><div class="km-source-children" ${expanded ? "" : "hidden"}><ol class="km-job-steps km-source-steps">${localSteps(visible).map(renderStep).join("")}</ol>${more}</div></article>`;
    }).join("");
  }

  function render() {
    detailRows.length = 0;
    const ordered = [...assets].sort((left, right) => {
      const leftLatest = latestTime(rows.filter((row) => String(row.km_asset_id) === String(left.km_asset_id)));
      const rightLatest = latestTime(rows.filter((row) => String(row.km_asset_id) === String(right.km_asset_id)));
      return rightLatest - leftLatest || String(left.asset_title || left.external_asset_id).localeCompare(String(right.asset_title || right.external_asset_id));
    });
    $("job-status-text").textContent = `${ordered.length} 个 Asset · 当前加载 ${rows.length} 条 KM 任务 · ${processing.length} 条 KC Revision`;
    $("job-tree").innerHTML = ordered.map(renderAsset).join("") || '<p class="km-tree-empty">当前来源下没有 Asset</p>';
    renderSourceJobs();
  }

  function toggle(event) {
    const assetButton = event.target.closest("[data-asset]");
    const chainButton = event.target.closest("[data-chain]");
    const sourceButton = event.target.closest("[data-source-jobs]");
    const sourceMoreButton = event.target.closest("[data-source-more]");
    const detailButton = event.target.closest("[data-detail]");
    if (assetButton) {
      const key = assetButton.dataset.asset;
      expandedAssets.has(key) ? expandedAssets.delete(key) : expandedAssets.add(key);
      render();
    } else if (chainButton) {
      const key = chainButton.dataset.chain;
      expandedChains.has(key) ? expandedChains.delete(key) : expandedChains.add(key);
      render();
    } else if (sourceButton) {
      const key = sourceButton.dataset.sourceJobs;
      expandedSourceJobs.has(key) ? expandedSourceJobs.delete(key) : expandedSourceJobs.add(key);
      render();
    } else if (sourceMoreButton) {
      const key = sourceMoreButton.dataset.sourceMore;
      const current = sourceJobVisibleCounts.get(key) || SOURCE_JOB_PAGE_SIZE;
      sourceJobVisibleCounts.set(key, current + SOURCE_JOB_PAGE_SIZE);
      expandedSourceJobs.add(key);
      render();
    } else if (detailButton) {
      $("job-detail-json").textContent = JSON.stringify(detailRows[Number(detailButton.dataset.detail)], null, 2);
      KBotKmShell.openDialog("job-detail-dialog");
    }
  }

  window.addEventListener("DOMContentLoaded", () => {
    $("job-form").addEventListener("submit", (event) => { event.preventDefault(); load(); });
    $("refresh-jobs").addEventListener("click", load);
    $("job-tree").addEventListener("click", toggle);
    $("source-job-tree").addEventListener("click", toggle);
  });
  KBotKmShell.ready.then(initialize).catch(() => {});
})();
