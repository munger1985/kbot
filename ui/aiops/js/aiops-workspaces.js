(function () {
  "use strict";

  const api = "/api/v1/apps/aiops";
  const shell = globalThis.KBotAIOpsShell;
  const markdown = globalThis.KBotMarkdown;
  const state = {
    agents: [], targets: [], conversation: null, selectedFiles: [],
    caseAgents: [], caseTargets: [],
    permissions: new Set(), starters: [], starterCatalogVersion: "",
    starterCatalogLoaded: false,
  };
  const maxDiagnosticFiles = 15;
  const typingFrameMs = 22;
  const streamRecoveryAttempts = 120;
  const activeTurnFollowers = new Set();
  const terminalRunStatuses = new Set([
    "COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED",
  ]);
  const terminalTurnStatuses = new Set([
    "WAITING_USER", "COMPLETED", "PARTIAL", "FAILED", "CANCELLED",
  ]);
  const workloadReportDefinitions = {
    "db.oracle.awr.report": {
      label: "下载原生 AWR 报告",
      filename: "oracle-awr-report.html",
    },
    "db.oracle.awr.diff_report": {
      label: "下载原生 AWR 对比报告",
      filename: "oracle-awr-diff-report.html",
    },
    "db.oracle.ash.report": {
      label: "下载原生 ASH 报告",
      filename: "oracle-ash-report.html",
    },
    "db.oracle.sql_monitor.report": {
      label: "下载原生 SQL Monitor 报告",
      filename: "oracle-sql-monitor.html",
    },
  };
  let activeSituationId = null;
  let situationRefreshTimer = null;
  const graphemeSegmenter = typeof Intl?.Segmenter === "function"
    ? new Intl.Segmenter(undefined, { granularity: "grapheme" })
    : null;

  const esc = shell.escape;
  const values = (items) => Array.isArray(items) ? items : [];
  const bullets = (items) => values(items).length
    ? values(items).map((item) => `- ${typeof item === "string" ? item : item.fact_summary || item.summary || item.title || "已记录"}`).join("\n")
    : "- 无";
  const inspectionItemText = (item) => {
    if (typeof item === "string") return item;
    return item?.fact_summary || item?.summary || item?.title || item?.detail || item?.code || "";
  };
  const inspectionBullets = (items, emptyText) => {
    const lines = values(items).map(inspectionItemText).map((item) => String(item || "").trim()).filter(Boolean);
    return lines.length ? lines.map((item) => `- ${item}`).join("\n") : `- ${emptyText}`;
  };
  const inspectionEvidenceFacts = (payload) => values(payload?.evidence).map((item) => {
    const title = item.tool_id || "检查项";
    const count = Number(item.row_count || 0);
    if (count <= 0) return `${title}：检查已完成，本期没有需要报告的记录，结果正常。`;
    return `${title}：检查已完成，采集 ${count} 条可验证观测，结果正常。`;
  });

  function uploadMediaType(file) {
    const suffix = String(file.name || "").toLowerCase().split(".").pop();
    if (["log", "txt", "trc", "trace", "out", "lst"].includes(suffix)) return "text/plain";
    if (["html", "htm"].includes(suffix)) return "text/html";
    if (suffix === "csv") return "text/csv";
    if (suffix === "json") return "application/json";
    if (suffix === "sql") return "application/sql";
    if (file.type) return file.type;
    return "application/octet-stream";
  }

  function inspectionMarkdown(result) {
    const payload = result?.payload || {};
    const schemaVersion = result?.final_artifact?.schema_version;
    if (!result?.final_artifact) {
      return `### 巡检尚未形成最终报告\n\n当前状态：${result?.status || "处理中"}`;
    }
    const healthyRecommendation = "继续按既定周期执行该巡检模板并关注趋势变化。";
    const emptyFindings = "本期检查均已完成，未发现异常。";
    const emptyGaps = "未发现数据缺口，全部检查已形成可验证观测。";
    if (schemaVersion === "AIOPS_TURN_RESULT.v1") {
      const conclusion = values(payload.blocks)
        .filter((block) => block.block_type === "MARKDOWN")
        .map((block) => String(block.payload?.markdown || "").trim())
        .filter(Boolean)
        .join("\n\n");
      const gaps = values(payload.evidence_gaps);
      const recommendations = gaps.length
        ? ["请先处理本报告列出的数据缺口，再重新执行同一巡检模板。"]
        : [healthyRecommendation];
      return `## 巡检报告\n\n${conclusion || "本期巡检已完成，所有计划检查均已形成可追溯观测。"}\n\n### 发现\n${inspectionBullets(inspectionEvidenceFacts(payload), emptyFindings)}\n\n### 建议\n${inspectionBullets(recommendations, healthyRecommendation)}\n\n### 数据缺口\n${inspectionBullets(gaps, emptyGaps)}`;
    }
    const conclusion = values(payload.facts)
      .filter((item) => item && item.kind === "agent_health_inspection")
      .map((item) => String(item.markdown || "").trim())
      .filter(Boolean)
      .join("\n\n");
    const checks = values(payload.facts).filter((item) => !(item && item.kind === "agent_health_inspection"));
    const summary = payload.summary || conclusion || "本期巡检已完成，所有计划检查均已形成可追溯观测。";
    const body = conclusion && payload.summary ? `${summary}\n\n${conclusion}` : summary;
    return `## ${payload.title || "巡检报告"}\n\n${body}\n\n### 发现\n${inspectionBullets(checks.length ? checks : payload.facts, emptyFindings)}\n\n### 建议\n${inspectionBullets(payload.recommendations, healthyRecommendation)}\n\n### 数据缺口\n${inspectionBullets(payload.gaps, emptyGaps)}`;
  }

  function inspectionReportHtml(result) {
    const payload = result?.payload || {};
    const facts = values(payload.facts);
    const checks = facts.filter((item) => item?.kind === "inspection_check");
    const findings = checks.flatMap((item) => values(item?.findings));
    const checksWithoutFindings = checks.filter((item) => !values(item?.findings).length);
    const healthyRecommendation = "继续按既定周期执行该巡检模板并关注趋势变化。";
    const emptyFindings = "本期未发现达到规则阈值的异常。";
    const emptyGaps = "未发现数据缺口，全部检查已形成可验证观测。";
    const summary = payload.summary || "本期巡检已完成。";
    const heading = markdown.render(`## ${payload.title || "巡检报告"}\n\n${summary}`);
    const findingCards = findingCardsHtml({
      findings,
      empty_reasons: findings.length ? [] : [emptyFindings],
    });
    const checksMarkdown = `### 其他检查结果\n${inspectionBullets(checksWithoutFindings, "全部检查结果已在发现卡片中展示。")}`;
    const recommendationsMarkdown = `### 建议\n${inspectionBullets(payload.recommendations, healthyRecommendation)}`;
    const gapsMarkdown = `### 数据缺口\n${inspectionBullets(payload.gaps, emptyGaps)}`;
    return `${heading}${findingCards}${markdown.render(`${checksMarkdown}\n\n${recommendationsMarkdown}\n\n${gapsMarkdown}`)}`;
  }

  function inspectionAnswerHtml(result, turn) {
    const schemaVersion = result?.final_artifact?.schema_version;
    if (schemaVersion === "AIOPS_TURN_RESULT.v1") {
      const narrative = values(result?.payload?.blocks).filter(
        (block) => !["TABLE", "CHART", "EVIDENCE_REFERENCES"].includes(block.block_type),
      );
      return narrative.map((block) => answerBlockHtml(block, turn)).join("")
        || markdown.render("本次巡检已完成，但未生成可展示的巡检结论。");
    }
    if (schemaVersion === "REPORT_CONTENT.v1") {
      return inspectionReportHtml(result);
    }
    return markdown.render(inspectionMarkdown(result));
  }

  function conversationAnswerMarkdown(result) {
    const payload = result?.payload || {};
    if (!result?.final_artifact) {
      return `诊断尚未完成，当前状态为 ${result?.status || "处理中"}。`;
    }
    const direct = payload.direct_answer?.answer_text;
    const solution = payload.solution || {};
    let answer = direct || payload.diagnosis_rationale || "现有证据还不足以回答这个问题。";
    const limitations = values(payload.direct_answer?.limitations).filter((item) => !answer.includes(String(item)));
    if (limitations.length) answer += `\n\n${limitations.map((item) => `> ${item}`).join("\n")}`;
    if (!direct) {
      const recommendations = [
        ...values(solution.immediate_mitigations),
        ...values(solution.long_term_remediations),
      ].filter(Boolean);
      if (recommendations.length) answer += `\n\n接下来可以这样处理：\n\n${bullets(recommendations)}`;
    }
    return answer;
  }

  function conversationAnswerHtml(result, turn) {
    const schemaVersion = result?.final_artifact?.schema_version;
    if (schemaVersion === "AIOPS_TURN_RESULT.v1") {
      const narrative = values(result?.payload?.blocks).filter(
        (block) => !["TABLE", "CHART", "EVIDENCE_REFERENCES"].includes(block.block_type),
      );
      return narrative.map((block) => answerBlockHtml(block, turn)).join("")
        || markdown.render("Agent 已完成诊断，但本轮没有生成可展示的文字结论。");
    }
    return markdown.render(conversationAnswerMarkdown(result));
  }

  function evidenceDetails(result) {
    if (result?.final_artifact?.schema_version === "AIOPS_TURN_RESULT.v1") {
      return turnEvidenceHtml(values(result?.payload?.blocks));
    }
    const payload = result?.payload || {};
    const facts = values(payload.facts);
    const gaps = values(payload.gaps);
    const root = payload.root_cause || {};
    if (!facts.length && !gaps.length && !root.effective_level) return "";
    const factRows = facts.length
      ? `<ol class="ops-evidence-list">${facts.map((fact) => `<li><span>${esc(fact.fact_summary || "已验证事实")}</span><small>${esc(fact.source_type || "EVIDENCE")}${fact.captured_at ? ` · ${esc(shell.fmt(fact.captured_at))}` : ""}</small></li>`).join("")}</ol>`
      : '<p class="ops-evidence-empty">本次没有形成可展示的事实条目。</p>';
    const rootRow = root.effective_level
      ? `<div class="ops-evidence-assessment"><span>根因判断</span><strong>${esc(root.effective_level)}</strong></div>`
      : "";
    const gapRows = gaps.length
      ? `<div class="ops-evidence-gaps"><strong>仍缺少的证据</strong><ul>${gaps.map((item) => `<li>${esc(typeof item === "string" ? item : item.code || item.summary || "EVIDENCE_GAP")}</li>`).join("")}</ul></div>`
      : "";
    return `<details class="ops-evidence"><summary>诊断依据 <span>${facts.length} 项已验证事实</span></summary><div class="ops-evidence-body">${rootRow}${factRows}${gapRows}</div></details>`;
  }

  function monitoringSourceSummary(detail) {
    const sources = values(detail?.monitoring_sources);
    if (!sources.length) return '<p class="ops-evidence-empty">当前情境没有可展示的监控来源。</p>';
    const rows = sources.map((item) => `<li><span><strong>${esc(item.display_name || "未命名监控来源")}</strong> ${shell.badge(item.latest_status)}</span><small>${esc(item.source_type || "UNKNOWN")} · 累计 ${esc(item.event_count)} 次观测 · 最近观测 ${esc(shell.fmt(item.last_observed_at))}</small></li>`).join("");
    return `<details class="ops-evidence" open><summary>监控来源 <span>${sources.length} 个</span></summary><div class="ops-evidence-body"><ol class="ops-evidence-list">${rows}</ol></div></details>`;
  }

  function situationAlertContent(detail) {
    const sources = values(detail?.monitoring_sources);
    const latestRows = sources.map((item) => `<li><span>${esc(item.latest_summary || detail.summary || detail.title)}</span><small>${esc(item.latest_event_class || "ALERT")} · ${esc(item.latest_severity || detail.severity)} · ${esc(shell.fmt(item.last_observed_at))}</small></li>`).join("");
    const content = latestRows || `<p>${esc(detail.summary || detail.title)}</p>`;
    return `<details class="ops-evidence" open><summary>告警内容 <span>最新状态</span></summary><div class="ops-evidence-body">${latestRows ? `<ol class="ops-evidence-list">${content}</ol>` : content}</div></details>`;
  }

  function situationStatusText(status) {
    const labels = {
      OPEN: "持续告警，仍有未恢复信号",
      RESOLVED: "已恢复",
      ACKNOWLEDGED: "已确认",
      INVESTIGATING: "诊断中",
      DIAGNOSED: "已诊断",
      MITIGATING: "处理中",
      OBSERVING: "观察中",
      SUPPRESSED: "已抑制",
    };
    return labels[String(status || "").toUpperCase()] || String(status || "未知状态");
  }

  function messageHtml(role, text, meta = "", supplemental = "") {
    const user = role === "USER";
    return `<article class="ops-message ${user ? "user" : "agent"}"><div class="ops-avatar">${user ? "我" : "AI"}</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${markdown.render(text)}</div>${supplemental}${meta ? `<div class="ops-message-meta">${esc(meta)}</div>` : ""}</div></article>`;
  }

  function imageAttachmentsHtml(conversationId, turn) {
    const user = values(turn.messages).find((item) => item.message_type === "USER_MESSAGE");
    const content = values(user?.payload?.content);
    const images = content
      .map((item, index) => ({ ...item, item_no: index + 1 }))
      .filter((item) => item.content_type === "IMAGE" && String(item.media_type || "").startsWith("image/"));
    if (!images.length) return "";
    return `<div class="ops-image-attachments">${images.map((item) => `<figure><img class="ops-conversation-image" alt="用户上传的诊断截图" data-image-content-path="${esc(`${api}/conversations/${conversationId}/turns/${turn.turn_id}/inputs/${item.item_no}/content`)}"><figcaption>诊断截图 · 将作为本轮证据保存</figcaption></figure>`).join("")}</div>`;
  }

  async function hydrateConversationImages(root) {
    const images = Array.from(root.querySelectorAll("img[data-image-content-path]"));
    await Promise.all(images.map(async (image) => {
      try {
        const blob = await KBotAIOpsAuth.requestBlob(image.dataset.imageContentPath);
        const url = URL.createObjectURL(blob);
        image.src = url;
        image.addEventListener("load", () => URL.revokeObjectURL(url), { once: true });
      } catch (_) {
        image.closest("figure")?.remove();
      }
    }));
  }

  function investigationPlanHtml(plan) {
    const actions = values(plan?.actions);
    const frame = plan?.task_frame || {};
    const hypotheses = values(plan?.hypotheses);
    if (!actions.length && !frame.problem_statement && !hypotheses.length) return "";
    const list = (title, items) => values(items).length
      ? `<div class="ops-plan-section"><strong>${esc(title)}</strong><ul>${values(items).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>`
      : "";
    const frameHtml = frame.problem_statement
      ? `<div class="ops-plan-frame"><p><strong>问题定义：</strong>${esc(frame.problem_statement)}</p>${list("当前已知", frame.known_facts)}${list("待验证", frame.unknowns)}${list("完成标准", frame.success_criteria)}</div>`
      : "";
    const hypothesisHtml = hypotheses.length
      ? `<div class="ops-plan-hypotheses"><strong>待验证假设</strong><ol>${hypotheses.map((item) => {
        const confidence = Number.isFinite(Number(item.confidence)) ? ` · 初始置信度 ${Math.round(Number(item.confidence) * 100)}%` : "";
        return `<li><span>${esc(item.statement || "待验证假设")}</span>${item.rationale || confidence ? `<small>${item.rationale ? `判断依据摘要：${esc(item.rationale)}` : ""}${esc(confidence)}</small>` : ""}</li>`;
      }).join("")}</ol></div>`
      : "";
    const rows = actions.map((action) => {
      const approval = action.execution_mode === "APPROVAL_REQUIRED";
      const mode = approval ? "需人工审批" : "自动只读执行";
      const status = action.status || "PLANNED";
      const evidence = action.expected_evidence_kind ? ` · 预期证据 ${action.expected_evidence_kind}` : "";
      const dependency = values(action.depends_on).length ? ` · 依赖 ${values(action.depends_on).join("、")}` : "";
      const query = action.sql_text
        ? `<details class="ops-plan-query"><summary>查看待执行 SQL 与参数</summary><pre><code>${esc(action.sql_text)}</code></pre><strong>绑定参数</strong><pre><code>${esc(JSON.stringify(action.parameters || {}, null, 2))}</code></pre></details>`
        : "";
      return `<li data-plan-action="${esc(action.action_id)}"><span>${esc(action.question || "执行诊断步骤")}</span><small>${esc(action.tool_class || action.tool_id || "DIAGNOSTIC")} · ${esc(mode)} · ${esc(status)}${esc(evidence)}${esc(dependency)}</small>${query}</li>`;
    }).join("");
    const actionsHtml = actions.length ? `<div class="ops-plan-actions"><strong>取证步骤</strong><ol>${rows}</ol></div>` : `<p class="ops-plan-empty">现有材料已足够，本轮不需要调用额外诊断工具。</p>`;
    return `<section class="ops-investigation-plan" data-plan-revision="${esc(plan.revision_no || 1)}"><header><strong>调查计划与判断依据</strong><span>第 ${esc(plan.revision_no || 1)} 版 · ${actions.length} 个步骤</span></header>${frameHtml}${hypothesisHtml}${actionsHtml}</section>`;
  }

  function showInvestigationPlan(progress, plan) {
    const html = investigationPlanHtml(plan);
    if (!html) return;
    const existing = progress.parentElement?.querySelector(".ops-investigation-plan.is-live");
    if (existing) existing.remove();
    progress.insertAdjacentHTML("beforebegin", html.replace("ops-investigation-plan", "ops-investigation-plan is-live"));
  }

  function ensureProgressTimeline(progress) {
    if (progress.dataset.timelineReady === "true") return;
    const initial = progress.textContent.trim() || "正在建立诊断计划：先固定执行上下文，再理解问题并选择证据…";
    progress.dataset.timelineReady = "true";
    progress.dataset.startedAt = String(Date.now());
    progress.innerHTML = `<header><strong>诊断过程</strong><span class="ops-progress-elapsed">已运行 0 秒</span></header><ol class="ops-progress-timeline"></ol>`;
    appendProgress(progress, "client.started", {
      public_summary: initial,
      public_sections: [{
        title: "计划将包含",
        items: ["问题定义与完成标准", "当前已知和待验证项", "候选假设、取证步骤及每步预期证据"],
      }],
    }, "client.started");
  }

  function appendProgress(progress, event, payload = {}, eventId = "") {
    ensureProgressTimeline(progress);
    const summary = String(payload.public_summary || payload.summary || `当前状态：${payload.status || "处理中"}`);
    const key = String(eventId || `${event}:${payload.action_id || payload.status || payload.revision_no || "current"}`);
    const timeline = progress.querySelector(".ops-progress-timeline");
    timeline.querySelectorAll("li.is-active").forEach((item) => item.classList.remove("is-active"));
    let row = [...timeline.children].find((item) => item.dataset.progressKey === key);
    if (!row) {
      row = document.createElement("li");
      row.dataset.progressKey = key;
      row.innerHTML = `<i aria-hidden="true"></i><div class="ops-progress-content"><span></span><div class="ops-progress-details"></div></div><small></small>`;
      timeline.append(row);
    }
    row.classList.add("is-active");
    row.querySelector(".ops-progress-content > span").textContent = summary;
    const details = row.querySelector(".ops-progress-details");
    const sections = values(payload.public_sections).filter((section) => values(section?.items).length);
    details.innerHTML = sections.map((section) => `<section><strong>${esc(section.title || "阶段详情")}</strong><ul>${values(section.items).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></section>`).join("");
    details.hidden = !sections.length;
    row.querySelector("small").textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function updateProgressElapsed(progress) {
    const startedAt = Number(progress.dataset.startedAt || Date.now());
    const seconds = Math.max(0, Math.floor((Date.now() - startedAt) / 1000));
    const label = progress.querySelector(".ops-progress-elapsed");
    if (label) label.textContent = `已运行 ${seconds} 秒`;
  }

  function diagnosticQueryApprovalHtml(pending) {
    const request = pending?.request || {};
    if (pending?.hitl_type !== "DIAGNOSTIC_QUERY_APPROVAL") return "";
    const reasons = values(request.reason_codes).length
      ? `<p><strong>审批原因：</strong>${values(request.reason_codes).map(esc).join("、")}</p>`
      : "";
    return `<section class="ops-query-approval" data-query-approval="${esc(pending.hitl_id)}"><header><strong>动态只读查询待审批</strong><span>${esc(request.target_display_name || "当前 Target")}</span></header><p>${esc(request.purpose || "执行动态只读诊断")}</p>${reasons}<strong>规范化 SQL</strong><pre><code>${esc(request.sql_text || "")}</code></pre><strong>绑定参数</strong><pre><code>${esc(JSON.stringify(request.parameters || {}, null, 2))}</code></pre><p class="ops-query-limits">最多 ${esc(request.max_rows || "-")} 行 · 超时 ${esc(request.timeout_seconds || "-")} 秒 · ${esc(shell.fmt(request.expires_at))} 前有效</p><div class="ops-query-actions"><button type="button" class="primary" data-query-decision="APPROVE" data-hitl-id="${esc(pending.hitl_id)}" data-row-version="${esc(pending.row_version)}">批准并继续</button><button type="button" data-query-decision="REJECT" data-hitl-id="${esc(pending.hitl_id)}" data-row-version="${esc(pending.row_version)}">拒绝并继续分析</button></div></section>`;
  }

  async function diagnosticQueryDecision(button) {
    const approving = button.dataset.queryDecision === "APPROVE";
    const note = approving
      ? "用户已核对并批准该动态只读查询"
      : prompt("请输入拒绝原因");
    if (!note) return;
    const card = button.closest(".ops-query-approval");
    card.querySelectorAll("button").forEach((item) => { item.disabled = true; });
    try {
      await KBotAIOpsAuth.request(`${api}/hitl/${encodeURIComponent(button.dataset.hitlId)}/decision`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({
          expected_row_version: Number(button.dataset.rowVersion),
          decision: approving ? "APPROVE" : "REJECT",
          note,
        }),
      });
      card.querySelector(".ops-query-actions").innerHTML = `<p>${approving ? "已批准，诊断继续执行。" : "已拒绝，Agent 将依据现有证据继续分析。"}</p>`;
      const progress = card.nextElementSibling?.matches("[data-turn-progress]")
        ? card.nextElementSibling
        : null;
      const conversationId = state.conversation?.conversation_id;
      if (progress && progress.id !== "live-progress" && conversationId) {
        appendProgress(progress, "approval.submitted", { public_summary: "审批已提交，正在继续诊断…" });
        followTurn(conversationId, progress.dataset.turnProgress, progress)
          .then(() => loadConversation(conversationId))
          .catch((error) => shell.toast(error.message));
      }
    } catch (error) {
      shell.toast(error.message);
      card.querySelectorAll("button").forEach((item) => { item.disabled = false; });
    }
  }

  function bindDiagnosticQueryActions(root) {
    root.querySelectorAll("[data-query-decision]").forEach((button) => {
      button.onclick = () => diagnosticQueryDecision(button);
    });
  }

  async function showDiagnosticQueryApproval(progress, hitlId) {
    const pending = await KBotAIOpsAuth.request(`${api}/hitl/${encodeURIComponent(hitlId)}`);
    const html = diagnosticQueryApprovalHtml(pending);
    if (!html) return;
    progress.parentElement?.querySelector(`[data-query-approval="${String(hitlId)}"]`)?.remove();
    progress.insertAdjacentHTML("beforebegin", html);
    bindDiagnosticQueryActions(progress.previousElementSibling);
  }

  async function agents() {
    const [rows, targetPage] = await Promise.all([
      KBotAIOpsAuth.request(`${api}/agents`),
      KBotAIOpsAuth.request(`${api}/targets?status=ENABLED&limit=200`),
    ]);
    state.caseAgents = values(rows);
    state.agents = state.caseAgents.filter((item) => item.status === "ACTIVE");
    state.targets = values(targetPage?.items);
    return state.agents;
  }

  function syncAgentTargetContext() {
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    const context = document.getElementById("agent-target-context");
    const agent = state.agents.find((item) => String(item.agent_id) === String(agentId));
    const target = state.targets.find((item) => String(item.target_id) === String(targetId));
    context.textContent = !targetId
      ? "先选择要运维的数据库对象。"
      : !agentId
        ? `已选择 ${target?.display_name || shell.short(targetId)}，请选择 Agent。`
      : target
        ? `诊断 Target：${target.display_name || shell.short(target.target_id)} · ${target.readonly_connection_enabled ? (target.connectivity_status || "UNKNOWN") : "仅监控模式"}`
        : "当前 Target 不可用";
  }

  async function confirmTargetFact(form) {
    const conversationId = form.dataset.conversationId;
    const turnId = form.dataset.turnId;
    const targetId = form.dataset.targetId;
    const factType = form.dataset.factType;
    const keyField = form.dataset.keyField;
    const raw = String(new FormData(form).get(keyField) || "").trim();
    const note = String(new FormData(form).get("note") || "").trim();
    if (!conversationId || !turnId || !targetId || !factType || !keyField || !raw) {
      shell.toast("请完整填写需要确认的运维事实");
      return;
    }
    if (!confirm("确认把这条事实写入 Target 运维记忆吗？系统只会记下来，不会执行 SQL。")) return;
    const button = form.querySelector("button[type=submit]");
    if (button) button.disabled = true;
    try {
      await KBotAIOpsAuth.request(
        `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}/target-facts:confirm`,
        {
          method: "POST",
          body: JSON.stringify({
            target_id: targetId,
            fact_type: factType,
            fact_key: raw,
            fact_value: { [keyField]: raw },
            note: note || null,
          }),
        },
      );
      shell.toast("运维记忆已确认写入，不会执行 SQL");
      if (state.conversation) await loadConversation(state.conversation.conversation_id);
    } catch (error) {
      shell.toast(error.message);
      if (button) button.disabled = false;
    }
  }

  async function proposalAction(button) {
    const approving = Boolean(button.dataset.approveProposal);
    const proposalId = button.dataset.approveProposal || button.dataset.rejectProposal;
    if (approving && !confirm("确认批准并执行这一条受控变更吗？系统会继续执行前置校验，并在完成后验证效果。")) return;
    const reason = approving ? "用户在诊断对话中逐条确认" : prompt("请输入拒绝原因");
    if (!reason) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/proposals/${encodeURIComponent(proposalId)}/${approving ? "approve" : "reject"}`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(approving ? { expected_row_version: Number(button.dataset.version), expected_proposal_hash: button.dataset.hash, note: reason } : { expected_row_version: Number(button.dataset.version), reason }),
      });
      shell.toast(approving ? "审批已提交，等待执行与验证" : "已拒绝该变更");
      if (state.conversation) await loadConversation(state.conversation.conversation_id);
      else button.closest(".ops-message")?.remove();
    } catch (error) { shell.toast(error.message); button.disabled = false; }
  }

  async function submitManualResult(button) {
    const proposal = button.closest(".ops-proposal");
    const status = proposal.querySelector("[data-manual-status]").value;
    const note = proposal.querySelector("[data-manual-note]").value.trim();
    const boundedOutput = proposal.querySelector("[data-manual-output]").value.trim();
    if (!confirm(`确认回填人工处理结果为“${status}”吗？系统不会执行页面中的命令。`)) return;
    button.disabled = true;
    try {
      await KBotAIOpsAuth.request(`${api}/proposals/${encodeURIComponent(button.dataset.manualProposal)}/manual-result`, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify({
          expected_row_version: Number(button.dataset.version),
          status,
          occurred_at: new Date().toISOString(),
          note: note || null,
          bounded_output: boundedOutput || null,
        }),
      });
      shell.toast(status === "EXECUTED" ? "人工结果已回填，系统将进行只读验证" : "人工结果已回填");
      if (state.conversation) await loadConversation(state.conversation.conversation_id);
    } catch (error) { shell.toast(error.message); button.disabled = false; }
  }

  const findingTypeLabels = {
    LOCK_WAIT: "会话阻塞",
    LONG_SESSION: "长会话",
    DG_LAG: "Data Guard 延迟",
    WAIT_CLASS: "等待类",
    TABLESPACE: "表空间",
    SQL_STATS_STALE: "统计过期",
    EXACHECK_FAIL: "ExaCheck 失败",
    EXACHECK_WARNING: "ExaCheck 警告",
    INVALID_OBJECT: "无效对象",
    ARCHIVE_HEADROOM: "FRA 余量",
    BACKUP_FAILED: "备份失败",
    LONG_TRANSACTION: "长事务",
    TOP_SQL: "高耗时 SQL",
  };
  const findingSeverityLabels = {
    CRITICAL: "严重",
    HIGH: "高",
    MEDIUM: "中",
    LOW: "低",
    INFO: "信息",
  };
  const findingConfirmationLabels = {
    CONFIRMED: "已确认",
    LIKELY: "很可能",
    POSSIBLE: "可能",
    UNKNOWN: "无法判断",
  };
  const findingFieldLabels = {
    waiting_session_id: "等待 SID",
    waiting_serial_number: "等待 Serial",
    waiting_sql_id: "等待 SQL_ID",
    blocking_session_id: "持有 SID",
    blocking_serial_number: "持有 Serial",
    blocking_sql_id: "持有 SQL_ID",
    lock_type: "锁类型",
    lock_mode: "锁模式",
    lock_ctime_seconds: "持有时间（秒）",
    waiting_instance_id: "等待实例",
    waiting_username: "等待用户",
    waiting_prev_sql_id: "等待上一 SQL_ID",
    waiting_status: "等待状态",
    blocking_instance_id: "持有实例",
    blocking_username: "持有用户",
    blocking_prev_sql_id: "持有上一 SQL_ID",
    blocking_status: "持有状态",
    wait_event: "等待事件",
    wait_seconds: "等待时间（秒）",
    chain_depth: "阻塞链深度",
    is_holder: "是否持有者",
    session_id: "SID",
    serial_number: "Serial",
    username: "用户",
    status: "状态",
    sql_id: "SQL_ID",
    prev_sql_id: "上一 SQL_ID",
    instance_id: "实例",
    client_host: "客户端主机",
    metric_name: "指标",
    metric_value: "当前值",
    metric_unit: "单位",
    lag_seconds: "延迟（秒）",
    wait_class: "等待类",
    time_waited_seconds: "等待秒数",
    foreground_waited_seconds: "前台等待秒数",
    total_waits: "等待次数",
    total_waits_fg: "前台等待次数",
    tablespace_name: "表空间",
    used_percent: "使用率（%）",
    free_mb: "剩余 MB",
    maximum_headroom_mb: "最大余量 MB",
    file_count: "文件数",
    allocated_mb: "已分配 MB",
    used_mb: "已用 MB",
    maximum_mb: "最大 MB",
    owner: "所有者",
    object_name: "对象名",
    object_type: "对象类型",
    last_analyzed: "上次分析",
    stale_stats: "统计过期",
    file_name: "文件名",
    check_name: "检查项",
    host: "主机",
    message: "说明",
    check_id: "检查 ID",
    object_count: "对象数量",
    database_role: "数据库角色",
    open_mode: "打开模式",
    log_mode: "日志模式",
    force_logging: "强制日志",
    flashback_on: "Flashback",
    recovery_file_dest: "恢复区路径",
    fra_limit_mb: "FRA 上限 MB",
    fra_used_mb: "FRA 已用 MB",
    fra_reclaimable_mb: "FRA 可回收 MB",
    fra_file_count: "FRA 文件数",
    fra_used_percent: "FRA 使用率（%）",
    session_key: "会话键",
    input_type: "输入类型",
    start_time: "开始时间",
    end_time: "结束时间",
    elapsed_seconds: "耗时（秒）",
    input_mb: "输入 MB",
    output_mb: "输出 MB",
    output_device_type: "输出设备",
    transaction_started_at: "事务开始时间",
    undo_blocks: "UNDO 块数",
    undo_records: "UNDO 记录数",
    plan_hash_value: "Plan Hash",
    executions: "执行次数",
    cpu_seconds: "CPU 秒",
    buffer_gets: "逻辑读",
    disk_reads: "物理读",
    rows_processed: "处理行数",
    last_active_time: "最近活动时间",
  };
  const findingPriorityFields = {
    LOCK_WAIT: [
      "waiting_session_id",
      "waiting_serial_number",
      "waiting_sql_id",
      "blocking_session_id",
      "blocking_serial_number",
      "blocking_sql_id",
      "lock_type",
      "lock_mode",
      "lock_ctime_seconds",
    ],
    SQL_STATS_STALE: [
      "owner",
      "object_name",
      "object_type",
      "stale_stats",
      "last_analyzed",
      "sql_id",
    ],
    EXACHECK_FAIL: [
      "check_name",
      "status",
      "host",
      "message",
      "check_id",
    ],
    EXACHECK_WARNING: [
      "check_name",
      "status",
      "host",
      "message",
      "check_id",
    ],
    INVALID_OBJECT: [
      "owner",
      "object_type",
      "status",
      "object_count",
    ],
    ARCHIVE_HEADROOM: [
      "fra_used_percent",
      "fra_used_mb",
      "fra_limit_mb",
      "recovery_file_dest",
    ],
    BACKUP_FAILED: [
      "status",
      "input_type",
      "start_time",
      "end_time",
      "output_device_type",
    ],
    LONG_TRANSACTION: [
      "session_id",
      "username",
      "elapsed_seconds",
      "transaction_started_at",
      "undo_blocks",
    ],
    TOP_SQL: [
      "sql_id",
      "elapsed_seconds",
      "cpu_seconds",
      "executions",
      "buffer_gets",
      "disk_reads",
    ],
  };

  function findingFieldValue(value) {
    return value == null || value === "" ? "空" : value;
  }

  function findingFieldsHtml(fields, findingType) {
    const source = fields || {};
    const keys = Object.keys(source);
    if (!keys.length) return "";
    const priority = findingPriorityFields[findingType] || [];
    const remaining = keys.filter((key) => !priority.includes(key));
    const ordered = [...priority.filter((key) => Object.prototype.hasOwnProperty.call(source, key)), ...remaining];
    return `<ul class="ops-finding-fields">${ordered.map((key) => `<li><span>${esc(findingFieldLabels[key] || key)}</span><strong>${esc(findingFieldValue(source[key]))}</strong></li>`).join("")}</ul>`;
  }

  function findingCardsHtml(payload) {
    const findings = values(payload.findings);
    const emptyReasons = values(payload.empty_reasons);
    const gaps = values(payload.gaps);
    const cards = findings.map((card) => {
      const severity = String(card.severity || "INFO");
      const tone = ["CRITICAL", "HIGH"].includes(severity) ? "bad" : severity === "MEDIUM" ? "warn" : "good";
      return `<article class="ops-finding-card is-${esc(severity.toLowerCase())}"><header><div><strong>${esc(findingTypeLabels[card.finding_type] || card.finding_type || "发现")}</strong><small>${esc(findingConfirmationLabels[card.confirmation] || card.confirmation || "")}</small></div><span class="ops-badge ${tone}">${esc(findingSeverityLabels[severity] || severity)}</span></header><p class="ops-finding-impact">${esc(card.impact || "")}</p>${findingFieldsHtml(card.fields, card.finding_type)}</article>`;
    }).join("");
    const empty = !findings.length
      ? `<p class="ops-finding-empty">${esc(emptyReasons.join(" ") || "当前没有需要报告的发现，不等于未取证。")}</p>`
      : emptyReasons.length
        ? `<p class="ops-finding-empty">${esc(emptyReasons.join(" "))}</p>`
        : "";
    const gapRows = gaps.length
      ? `<p class="ops-finding-gaps">${gaps.map((item) => esc(item.detail || item.column || "字段缺失")).join("；")}</p>`
      : "";
    return `<section class="ops-findings"><header><strong>发现</strong></header>${cards}${empty}${gapRows}</section>`;
  }

  function factConfirmationHtml(payload, turn) {
    const candidates = values(payload.candidates);
    if (!candidates.length) return "";
    const forms = candidates.map((candidate) => {
      const fields = values(candidate.fields).map((field) => (
        `<label>${esc(field.label || field.name)}`
        + `<input name="${esc(field.name)}" maxlength="256" ${field.required ? "required" : ""}>`
        + `</label>`
      )).join("");
      return (
        `<form class="ops-fact-confirmation-form" data-confirm-target-fact`
        + ` data-conversation-id="${esc(turn?.conversation_id || "")}"`
        + ` data-turn-id="${esc(turn?.turn_id || "")}"`
        + ` data-target-id="${esc(payload.target_id || "")}"`
        + ` data-fact-type="${esc(candidate.fact_type || "")}"`
        + ` data-key-field="${esc(candidate.key_field || "")}">`
        + `<strong>${esc(candidate.label || candidate.fact_type || "运维事实")}</strong>`
        + fields
        + `<label>备注（可选）<textarea name="note" maxlength="1000" rows="2"></textarea></label>`
        + `<button type="submit" class="primary">确认写入运维记忆</button>`
        + `</form>`
      );
    }).join("");
    return (
      `<section class="ops-fact-confirmation">`
      + `<header><strong>确认运维记忆</strong></header>`
      + `<p>${esc(payload.instruction || "确认后才会写入 Target 运维记忆，不会执行 SQL。")}</p>`
      + forms
      + `</section>`
    );
  }

  function htmlReportLinksHtml(payload, turn) {
    const reports = values(payload.reports);
    if (!reports.length) return "";
    const items = reports.map((report) => {
      const label = report.label || report.tool_id || "原生报告";
      if (turn && report.action_id) {
        const definition = workloadReportDefinitions[report.tool_id] || {
          filename: "oracle-workload-report.html",
        };
        const filename = workloadReportFilename(definition.filename, report.action_id);
        return (
          `<button type="button" data-download-workload-report="${esc(report.tool_id)}" `
          + `data-workload-report-action="${esc(report.action_id)}" `
          + `data-conversation-id="${esc(turn.conversation_id)}" `
          + `data-turn-id="${esc(turn.turn_id)}" `
          + `data-workload-report-filename="${esc(filename)}">`
          + `${esc(label)}</button>`
        );
      }
      return `<span class="ops-html-report-label">${esc(label)}</span>`;
    }).join("");
    return `<div class="ops-workload-report-actions">${items}</div>`;
  }

  function answerBlockHtml(block, turn) {
    const payload = block.payload || {};
    if (block.block_type === "MARKDOWN") return markdown.render(payload.markdown || payload.text || "");
    if (block.block_type === "ANALYSIS_MARKDOWN") {
      return `<section class="ops-analysis"><header><strong>分析</strong></header>${markdown.render(payload.markdown || payload.text || "")}</section>`;
    }
    if (block.block_type === "SOLUTION_MARKDOWN") {
      return `<section class="ops-solution"><header><strong>解决方案</strong></header>${markdown.render(payload.markdown || payload.text || "")}</section>`;
    }
    if (block.block_type === "FACT_CONFIRMATION") return factConfirmationHtml(payload, turn);
    if (block.block_type === "FINDING_CARDS") return findingCardsHtml(payload);
    if (block.block_type === "HTML_REPORT_LINKS") return htmlReportLinksHtml(payload, turn);
    if (block.block_type === "IMPLEMENTATION_RUNBOOK") {
      const applicabilityLabel = {
        REQUIRED: "必须实施",
        ALREADY_SATISFIED: "当前已满足",
        CONDITIONAL: "按条件实施",
        BLOCKED: "等待必要事实",
      };
      const isManualCommand = (command) => [command.command_type, command.executor]
        .some((value) => String(value || "").toUpperCase() === "MANUAL");
      const commandHtml = (command, manual = false) => {
        const metadata = [
          manual ? "" : command.executor || command.command_type,
          command.run_as ? `身份 ${command.run_as}` : "",
          values(command.node_scope).length ? `节点 ${values(command.node_scope).join("、")}` : "",
          command.container_name ? `容器 ${command.container_name}` : "",
          command.risk_level ? `风险 ${command.risk_level}` : "",
        ].filter(Boolean).map((item) => `<span>${esc(item)}</span>`).join("");
        if (manual) {
          return `<div class="ops-runbook-manual"><h6>${esc(command.title || "人工确认")}</h6><p>${esc(command.content || "")}</p>${metadata ? `<div class="ops-runbook-command-meta">${metadata}</div>` : ""}${values(command.notes).length ? `<ul>${values(command.notes).map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}</div>`;
        }
        return `<div class="agent-code-block ops-runbook-command"><div class="agent-code-toolbar"><span><b>${esc(command.executor || command.command_type || "COMMAND")}</b>${esc(command.title || "")}</span><button type="button" data-copy-code>复制命令</button></div>${metadata ? `<div class="ops-runbook-command-meta">${metadata}</div>` : ""}<pre><code>${esc(command.content || "")}</code></pre>${values(command.expected_result).length ? `<div class="ops-runbook-expected"><strong>预期结果</strong><ul>${values(command.expected_result).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></div>` : ""}${values(command.notes).length ? `<ul>${values(command.notes).map((item) => `<li>${esc(item)}</li>`).join("")}</ul>` : ""}</div>`;
      };
      const commandGroup = (title, commands, manual = false) => values(commands).length
        ? `<section class="ops-runbook-command-group${manual ? " is-manual" : ""}"><h5>${esc(title)}</h5>${values(commands).map((command) => commandHtml(command, manual)).join("")}</section>`
        : "";
      const stateItems = values(payload.current_state).map((item) => `<li class="is-${esc(String(item.status || "unknown").toLowerCase())}"><span>${esc(item.label || "-")}</span><strong>${esc(item.value || "-")}</strong><small>${esc(item.status || "UNKNOWN")}</small></li>`).join("");
      const resolvedParameters = values(payload.resolved_parameters).map((item) => `<li><div><strong>${esc(item.label || item.key)}</strong><small>${esc(item.source || "")}</small></div><code>${esc(item.value || "")}</code><span>${esc(item.status || "")}</span></li>`).join("");
      const requiredInputs = values(payload.required_inputs).map((item) => `<li><div><strong>${esc(item.label || item.key)}</strong><small>${esc(item.description || "")}</small></div><code>${esc(item.placeholder || "")}</code></li>`).join("");
      const missingFacts = values(payload.missing_facts).map((item) => `<li><div><strong>${esc(item.fact_key || "必要事实")}</strong><small>${esc(item.reason || "")}</small></div><code>${esc(item.resolution_source || "")}</code><span>${values(item.blocking_steps).length ? `阻断 ${esc(values(item.blocking_steps).join("、"))}` : ""}</span></li>`).join("");
      const artifactItems = values(payload.artifacts).map((item) => `<li><div><strong>${esc(item.file_name || item.artifact_id)}</strong><small>${esc(item.description || "")}</small></div><code>${esc(item.target_path || item.relative_path || "")}</code><span>${esc(item.file_mode || "")} · ${esc(item.run_as || "")} · SHA256 ${esc(String(item.sha256 || "").slice(0, 12))}…</span></li>`).join("");
      const runbookId = `runbook-${String(turn?.turn_id || "current").replace(/[^a-zA-Z0-9_-]/g, "")}`;
      const phaseValues = values(payload.phases);
      const phases = phaseValues.map((phase, phaseIndex) => {
        const phaseNumber = phaseIndex + 1;
        const phaseId = `${runbookId}-phase-${phaseNumber}`;
        const steps = values(phase.steps).map((step, stepIndex) => {
          const stepNumber = `${phaseNumber}.${stepIndex + 1}`;
          const stepId = `${phaseId}-step-${stepIndex + 1}`;
          const implementationCommands = values(step.commands).filter((command) => !isManualCommand(command));
          const manualItems = values(step.commands).filter(isManualCommand);
          return `<article class="ops-runbook-step" id="${stepId}"><header><div><span class="ops-runbook-step-number">${stepNumber}</span><h4>${esc(step.title || step.step_id)}</h4><p>${esc(step.rationale || "")}</p></div><span class="ops-runbook-applicability is-${esc(String(step.applicability || "required").toLowerCase())}">${esc(applicabilityLabel[step.applicability] || step.applicability || "")}</span></header>${values(step.required_inputs).length ? `<p class="ops-runbook-needs"><strong>所需输入：</strong>${values(step.required_inputs).map((item) => `<code>${esc(item)}</code>`).join(" ")}</p>` : ""}${commandGroup("人工确认项", manualItems, true)}${commandGroup("实施命令", implementationCommands)}${commandGroup("验证命令", step.verification_commands)}${commandGroup("回退命令", step.rollback)}${values(step.risks).length ? `<aside class="ops-runbook-risks"><strong>风险与注意事项</strong><ul>${values(step.risks).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></aside>` : ""}</article>`;
        }).join("");
        return `<section class="ops-runbook-phase" id="${phaseId}"><header><span class="ops-runbook-phase-number">${phaseNumber}</span><div><h3>${esc(phase.title || phase.phase_id)}</h3><p>${esc(phase.objective || "")}</p></div></header><div class="ops-runbook-steps">${steps}</div></section>`;
      }).join("");
      const tableOfContents = phaseValues.length
        ? `<nav class="ops-runbook-toc" aria-label="实施文档目录"><h3>目录</h3><ol>${phaseValues.map((phase, phaseIndex) => { const phaseNumber = phaseIndex + 1; const phaseId = `${runbookId}-phase-${phaseNumber}`; return `<li><a href="#${phaseId}"><span>${phaseNumber}. ${esc(phase.title || phase.phase_id)}</span></a>${values(phase.steps).length ? `<ol>${values(phase.steps).map((step, stepIndex) => `<li><a href="#${phaseId}-step-${stepIndex + 1}">${phaseNumber}.${stepIndex + 1} ${esc(step.title || step.step_id)}</a></li>`).join("")}</ol>` : ""}</li>`; }).join("")}</ol></nav>`
        : "";
      const stopConditions = values(payload.stop_conditions).length ? `<section class="ops-runbook-stop"><h5>停止条件</h5><ul>${values(payload.stop_conditions).map((item) => `<li>${esc(item)}</li>`).join("")}</ul></section>` : "";
      const download = turn?.conversation_id && turn?.turn_id
        ? `<div class="ops-runbook-downloads"><button type="button" data-download-implementation-runbook="pdf" data-runbook-profile="${esc(payload.profile || "database-implementation")}" data-conversation-id="${esc(turn.conversation_id)}" data-turn-id="${esc(turn.turn_id)}">下载 PDF</button><button type="button" data-download-implementation-runbook="markdown" data-runbook-profile="${esc(payload.profile || "database-implementation")}" data-conversation-id="${esc(turn.conversation_id)}" data-turn-id="${esc(turn.turn_id)}">下载 Markdown</button>${artifactItems ? `<button type="button" data-download-implementation-runbook="zip" data-runbook-profile="${esc(payload.profile || "database-implementation")}" data-conversation-id="${esc(turn.conversation_id)}" data-turn-id="${esc(turn.turn_id)}">下载脚本 ZIP</button>` : ""}</div>`
        : "";
      const adjustParameters = payload.generation?.starter_id && turn?.turn_id
        ? `<button type="button" class="ops-runbook-adjust" data-adjust-implementation-runbook="${esc(turn.turn_id)}">调整参数并重新生成</button>`
        : "";
      const appendix = stateItems || resolvedParameters || requiredInputs || missingFacts || artifactItems
        ? `<section class="ops-runbook-appendix"><h3>附录：当前状态与实施参数</h3>${stateItems ? `<section><h4>当前环境摘要</h4><ul class="ops-runbook-state">${stateItems}</ul></section>` : ""}${resolvedParameters ? `<section class="ops-runbook-parameters"><h4>已解析实施参数</h4><ul>${resolvedParameters}</ul></section>` : ""}${missingFacts ? `<section class="ops-runbook-missing-facts"><h4>缺失的必要事实</h4><p>请在对应 Target 的部署拓扑、主机采集或策略配置中补齐，重新生成后解除阻断。</p><ul>${missingFacts}</ul></section>` : ""}${requiredInputs ? `<section class="ops-runbook-inputs"><h4>历史档案所需输入</h4><ul>${requiredInputs}</ul></section>` : ""}${artifactItems ? `<section class="ops-runbook-artifacts"><h4>脚本与配置清单</h4><ul>${artifactItems}</ul></section>` : ""}</section>`
        : "";
      return `<article class="ops-runbook"><header class="ops-runbook-document-header"><div><p class="ops-runbook-kicker">KBot 智能运维 · 数据库实施操作文档</p><h2>${esc(payload.title || "数据库实施 Runbook")}</h2><p class="ops-runbook-meta">${esc(payload.profile || "")} · ${esc(payload.schema_version || "")}</p></div><div class="ops-runbook-header-actions"><span class="ops-runbook-status">${esc(payload.status || "UNKNOWN")}</span>${adjustParameters}${download}</div></header><section class="ops-runbook-policy"><h3>执行边界</h3><p>${esc(payload.execution_policy || "")}</p></section>${tableOfContents}<div class="ops-runbook-body">${phases}</div>${stopConditions}${appendix}</article>`;
    }
    if (block.block_type === "TABLE") {
      const columns = values(payload.columns);
      const cell = (row, column, index) => Array.isArray(row)
        ? row[index]
        : row?.[column.key || column.name || column];
      return `<div class="ops-table-wrap"><table><thead><tr>${columns.map((column) => `<th>${esc(column.label || column.name || column.key || column)}</th>`).join("")}</tr></thead><tbody>${values(payload.rows).map((row) => `<tr>${columns.map((column, index) => `<td>${esc(cell(row, column, index) ?? "-")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }
    if (block.block_type === "CHART") {
      const categories = values(payload.categories);
      const sourceSeries = values(payload.series);
      const series = sourceSeries.map((item, index) => typeof item === "object"
        ? item
        : { label: categories[index] ?? "-", value: item });
      const maximum = Math.max(0, ...series.map((item) => Number(item.value)).filter(Number.isFinite));
      return `<figure class="ops-tablespace-chart"><figcaption>${esc(payload.title || "指标对比")}</figcaption><div class="ops-chart-rows">${series.map((item) => { const raw = Number(item.value); const width = Number.isFinite(raw) && maximum > 0 ? Math.max(0, Math.min(100, raw / maximum * 100)) : 0; return `<div class="ops-chart-row"><span>${esc(item.label || item.name || "-")}</span><div class="ops-chart-track"><i style="width:${width}%"></i></div><strong>${esc(item.display_value ?? item.value ?? "-")}</strong></div>`; }).join("")}</div></figure>`;
    }
    if (block.block_type === "PROPOSAL_SUMMARY") {
      const parameters = Object.entries(payload.parameters || {}).map(([key, value]) => `<li><code>${esc(key)}</code><span>${esc(typeof value === "object" ? JSON.stringify(value) : value)}</span></li>`).join("");
      const pending = payload.status === "PENDING_APPROVAL";
      const manual = payload.execution_mode === "MANUAL_ONLY" && payload.status === "ADVISORY_READY";
      const canApprove = state.permissions.has("aiops:proposal:approve");
      const actions = pending && canApprove
        ? `<div class="ops-proposal-actions"><button type="button" class="primary" data-approve-proposal="${esc(payload.proposal_id)}" data-version="${esc(payload.row_version || 1)}" data-hash="${esc(payload.proposal_hash)}">批准并执行</button><button type="button" data-reject-proposal="${esc(payload.proposal_id)}" data-version="${esc(payload.row_version || 1)}">拒绝</button></div>`
        : manual && canApprove
          ? `<div class="ops-manual-result"><strong>DBA 人工执行结果</strong><select data-manual-status><option value="EXECUTED">已执行</option><option value="FAILED">执行失败</option><option value="CANCELLED">已取消</option></select><textarea data-manual-note maxlength="4000" placeholder="处理说明（可选）"></textarea><textarea data-manual-output maxlength="16000" placeholder="受限输出（可选，请勿填写密码或密钥）"></textarea><button type="button" class="primary" data-manual-proposal="${esc(payload.proposal_id)}" data-version="${esc(payload.row_version || 1)}">回填结果</button></div>`
          : `<p class="ops-proposal-status">当前状态：${esc(payload.status || "UNKNOWN")}</p>`;
      const command = payload.command_preview ? `<div class="agent-code-block"><div class="agent-code-toolbar"><span>仅供 DBA 人工核对${manual ? "并在 KBot 外执行" : ""}</span><button type="button" data-copy-code>复制命令</button></div><pre><code>${esc(payload.command_preview)}</code></pre></div>` : "";
      const verification = values(payload.verification_plan).length ? `<p><strong>验证计划：</strong>${values(payload.verification_plan).map(esc).join("、")}</p>` : "";
      const title = manual ? "仅供人工执行" : pending ? "受控变更待审批" : "受控动作建议";
      return `<section class="ops-proposal"><header><div><strong>${title}</strong><small>${esc(payload.action_template_id || "Action Template")} · ${esc(payload.risk_level || "UNKNOWN")}</small></div></header><p>${esc(payload.rationale || "")}</p><p><strong>影响范围：</strong>${esc(payload.impact || "-")}</p><p><strong>锁影响：</strong>${esc(payload.lock_impact || "-")}</p>${parameters ? `<ul class="ops-proposal-parameters">${parameters}</ul>` : ""}${command}${verification}${actions}</section>`;
    }
    if (block.block_type === "EVIDENCE_REFERENCES") return "";
    return markdown.render(payload.markdown || payload.text || payload.instruction || "");
  }

  function turnEvidenceHtml(blocks, gaps = []) {
    const evidence = new Map();
    const dataBlocks = blocks.filter((block) => ["TABLE", "CHART"].includes(block.block_type));
    const add = (key, label, meta) => {
      const normalizedLabel = String(label || "诊断证据").trim();
      const normalizedKey = String(key || normalizedLabel.toLowerCase());
      if (!evidence.has(normalizedKey)) evidence.set(normalizedKey, { label: normalizedLabel, meta });
    };
    blocks.forEach((block) => {
      values(block.evidence_refs).forEach((item) => add(
        item,
        item,
        "VERIFIED_EVIDENCE",
      ));
      values(block.citations).forEach((item) => add(
        item.turn_evidence_id || `citation:${item.label || item.citation_no}`,
        item.label || `证据 ${item.citation_no}`,
        shell.short(item.turn_evidence_id),
      ));
      if (block.block_type !== "EVIDENCE_REFERENCES") return;
      values(block.payload?.items).forEach((item, index) => add(
        item.turn_evidence_id || item.artifact_id || `reference:${item.label || item.summary || index}`,
        item.label || item.summary || "诊断证据",
        `${item.source || "EVIDENCE"}${item.observed_at ? ` · ${shell.fmt(item.observed_at)}` : ""}`,
      ));
    });
    const rows = Array.from(evidence.values());
    const gapRows = values(gaps).map((item) => ({
      label: item.detail || item.code || "本次未取得证据",
      meta: `${item.code || "EVIDENCE_GAP"}${item.step_id ? ` · ${item.step_id}` : ""}`,
    }));
    if (!rows.length && !gapRows.length && !dataBlocks.length) return "";
    const evidenceRows = rows.length
      ? `<ol class="ops-evidence-list">${rows.map((item) => `<li><span>${esc(item.label)}</span><small>${esc(item.meta || "EVIDENCE")}</small></li>`).join("")}</ol>`
      : dataBlocks.length
        ? ""
        : '<p class="ops-evidence-empty">本次没有形成可展示的有效证据。</p>';
    const evidenceData = dataBlocks.length
      ? `<div class="ops-evidence-data"><strong>原始取证结果</strong>${dataBlocks.map((block) => { const payload = block.payload || {}; const meta = [payload.measurement_semantics, payload.captured_at ? shell.fmt(payload.captured_at) : ""].filter(Boolean).join(" · "); return `<section><header><span>${esc(payload.title || (block.block_type === "CHART" ? "指标图表" : "查询结果"))}</span>${meta ? `<small>${esc(meta)}</small>` : ""}</header>${answerBlockHtml(block)}</section>`; }).join("")}</div>`
      : "";
    const missingRows = gapRows.length
      ? `<div class="ops-evidence-gaps"><strong>未取得的证据</strong><ol class="ops-evidence-list">${gapRows.map((item) => `<li><span>${esc(item.label)}</span><small>${esc(item.meta)}</small></li>`).join("")}</ol></div>`
      : "";
    return `<details class="ops-evidence"><summary>诊断依据 <span>${rows.length} 项证据${dataBlocks.length ? ` · ${dataBlocks.length} 份原始结果` : ""}${gapRows.length ? ` · ${gapRows.length} 项缺口` : ""}</span></summary><div class="ops-evidence-body">${evidenceRows}${evidenceData}${missingRows}</div></details>`;
  }

  function reportAction({ runId, conversationId, sourceKind, periodKind = "AD_HOC" }) {
    if (!runId && !conversationId) return "";
    const source = conversationId
      ? `data-generate-report-conversation="${esc(conversationId)}"`
      : `data-generate-report-run="${esc(runId)}"`;
    return `<div class="ops-filter-actions"><button type="button" class="primary" ${source} data-report-source-kind="${esc(sourceKind)}" data-report-period-kind="${esc(periodKind)}">生成正式报告</button></div>`;
  }

  function workloadReportFilename(filename, actionId) {
    return String(filename || "oracle-workload-report.html").replace(
      /\.html$/i,
      `-${actionId}.html`,
    );
  }

  function workloadReportLabel(definition, action, actions) {
    const sameTool = actions.filter((item) => item.tool_id === action.tool_id);
    if (sameTool.length <= 1) return definition.label;
    const question = String(action.question || "").trim();
    const uniqueQuestion = question && sameTool.filter(
      (item) => String(item.question || "").trim() === question,
    ).length === 1;
    if (uniqueQuestion) return `${definition.label}：${question}`;
    return `${definition.label}（${action.action_id}）`;
  }

  function workloadReportActions(turn) {
    const actions = values(turn.investigation_plan?.actions).filter((action) => (
      action.status === "SUCCEEDED" && workloadReportDefinitions[action.tool_id]
    ));
    if (!actions.length) return "";
    const buttons = actions.map((action) => {
      const definition = workloadReportDefinitions[action.tool_id];
      const filename = workloadReportFilename(definition.filename, action.action_id);
      const label = workloadReportLabel(definition, action, actions);
      return (
        `<button type="button" data-download-workload-report="${esc(action.tool_id)}" `
        + `data-workload-report-action="${esc(action.action_id)}" `
        + `data-conversation-id="${esc(turn.conversation_id)}" `
        + `data-turn-id="${esc(turn.turn_id)}" `
        + `data-workload-report-filename="${esc(filename)}">`
        + `${esc(label)}</button>`
      );
    }).join("");
    return `<div class="ops-workload-report-actions">${buttons}</div>`;
  }

  function bindWorkloadReportActions(root = document) {
    root.querySelectorAll("[data-download-workload-report]").forEach((button) => {
      button.onclick = async () => {
        button.disabled = true;
        const label = button.textContent;
        button.textContent = "正在下载…";
        try {
          const conversationId = encodeURIComponent(button.dataset.conversationId);
          const turnId = encodeURIComponent(button.dataset.turnId);
          const toolId = encodeURIComponent(button.dataset.downloadWorkloadReport);
          const actionId = encodeURIComponent(button.dataset.workloadReportAction);
          await KBotAIOpsAuth.download(
            `${api}/conversations/${conversationId}/turns/${turnId}/workload-reports/${toolId}?action_id=${actionId}`,
            button.dataset.workloadReportFilename,
            "text/html",
          );
          shell.toast("原生 Oracle 报告已开始下载");
        } catch (error) {
          shell.toast(error.message || "无法下载原生 Oracle 报告");
        } finally {
          button.disabled = false;
          button.textContent = label;
        }
      };
    });
  }

  function bindImplementationRunbookActions(root = document) {
    root.querySelectorAll("[data-adjust-implementation-runbook]").forEach((button) => {
      button.onclick = () => {
        const turn = values(state.turns).find((item) => item.turn_id === button.dataset.adjustImplementationRunbook);
        const block = values(turn?.answer_blocks).find((item) => item.block_type === "IMPLEMENTATION_RUNBOOK");
        const generation = block?.payload?.generation || {};
        if (!generation.starter_id) return shell.toast("该历史文档没有可重新生成的功能入口信息");
        selectStarter(generation.starter_id, generation.supplied_parameters || {});
      };
    });
    root.querySelectorAll("[data-download-implementation-runbook]").forEach((button) => {
      button.onclick = async () => {
        button.disabled = true;
        const label = button.textContent;
        button.textContent = "正在下载…";
        try {
          const conversationId = encodeURIComponent(button.dataset.conversationId);
          const turnId = encodeURIComponent(button.dataset.turnId);
          const formats = {
            pdf: { extension: "pdf", mediaType: "application/pdf", label: "PDF" },
            markdown: { extension: "md", mediaType: "text/markdown", label: "Markdown" },
            zip: { extension: "zip", mediaType: "application/zip", label: "脚本 ZIP" },
          };
          const format = formats[button.dataset.downloadImplementationRunbook] || formats.pdf;
          const profile = String(button.dataset.runbookProfile || "database-implementation").toLowerCase().replaceAll("_", "-");
          await KBotAIOpsAuth.download(
            `${api}/conversations/${conversationId}/turns/${turnId}/implementation-runbook.${format.extension}`,
            `${profile}-${button.dataset.turnId}.${format.extension}`,
            format.mediaType,
          );
          shell.toast(`数据库实施文档 ${format.label} 已开始下载`);
        } catch (error) {
          shell.toast(error.message || "无法下载数据库实施文档");
        } finally {
          button.disabled = false;
          button.textContent = label;
        }
      };
    });
  }

  async function openReportGenerator(button) {
    button.disabled = true;
    try {
      const sourceKind = button.dataset.reportSourceKind;
      const periodKind = button.dataset.reportPeriodKind;
      const conversationId = button.dataset.generateReportConversation;
      const rows = await KBotAIOpsAuth.request(
        `${api}/${conversationId ? "session-report-templates" : "report-layouts"}`,
      );
      const templates = values(rows).filter((item) => values(item.applicable_source_kinds).includes(sourceKind)
        && (sourceKind === "INSPECTION" || values(item.allowed_period_kinds).includes(periodKind)));
      if (!templates.length) throw new Error("当前诊断没有可用的报告模板");
      const periods = sourceKind === "INSPECTION"
        ? ["DAILY", "MONTHLY", "QUARTERLY", "ANNUAL"] : [periodKind];
      const periodLabels = { DAILY: "日常报告", MONTHLY: "月度报告", QUARTERLY: "季度报告", ANNUAL: "年度报告", AD_HOC: "单次诊断报告" };
      const dialog = document.createElement("dialog");
      dialog.className = "ops-dialog";
      const scopeHint = conversationId
        ? "报告将冻结本次会话全部已完成 Turn 的事实、证据引用和数据缺口。"
        : "报告将冻结当前已验证事实、证据引用和数据缺口。";
      dialog.innerHTML = `<form method="dialog"><header><h2>生成正式报告</h2><p>${scopeHint}</p></header><div class="ops-dialog-body">${sourceKind === "INSPECTION" ? `<label class="ops-field">报告周期<select name="period_kind">${periods.map((kind) => `<option value="${kind}">${periodLabels[kind]}</option>`).join("")}</select></label>` : ""}<label class="ops-field">${conversationId ? "会话报告模板" : "系统报告版式"}<select name="template_ref"></select></label><p class="ops-connection-result">${conversationId ? "报告范围为当前会话，进行中的 Turn 结束后才可生成。" : "周期报告只汇总最近一个完整自然周期内的巡检结果。"}</p></div><footer><button value="cancel">取消</button><button class="primary" value="confirm">生成报告</button></footer></form>`;
      document.body.append(dialog);
      const templateSelect = dialog.querySelector('[name="template_ref"]');
      const periodSelect = dialog.querySelector('[name="period_kind"]');
      const populateTemplates = () => {
        const selectedPeriod = periodSelect?.value || periodKind;
        const available = templates.filter((item) => values(item.allowed_period_kinds).includes(selectedPeriod));
        templateSelect.innerHTML = available.map((item) => `<option value="${esc(item.template_ref)}">${esc(item.display_name)}</option>`).join("");
        if (!available.length) templateSelect.innerHTML = '<option value="">当前周期没有可用模板</option>';
      };
      periodSelect?.addEventListener("change", populateTemplates); populateTemplates();
      dialog.addEventListener("close", async () => {
        try {
          if (dialog.returnValue !== "confirm") return;
          const templateRef = templateSelect.value;
          const selectedPeriod = periodSelect?.value || periodKind;
          if (!templateRef) throw new Error("当前周期没有可用模板");
          const reportSource = conversationId
            ? { conversation_id: conversationId }
            : { ops_run_id: button.dataset.generateReportRun };
          const result = await KBotAIOpsAuth.request(`${api}/reports:generate`, {
            method: "POST",
            headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
            body: JSON.stringify({ ...reportSource, template_ref: templateRef, period_kind: selectedPeriod }),
          });
          location.href = `./report-detail.html?id=${encodeURIComponent(result.report_id)}`;
        } catch (error) { shell.toast(error.message); }
        finally { dialog.remove(); }
      }, { once: true });
      dialog.showModal();
    } catch (error) { shell.toast(error.message); button.disabled = false; }
  }

  function bindReportActions(root = document) {
    root.querySelectorAll("[data-generate-report-run], [data-generate-report-conversation]").forEach((button) => {
      button.onclick = () => openReportGenerator(button);
    });
  }

  function turnHtml(turn) {
    const messages = values(turn.messages);
    const user = messages.find((item) => item.message_type === "USER_MESSAGE");
    const assistant = messages.find((item) => item.message_type === "ASSISTANT_MESSAGE");
    const answerBlocks = values(turn.answer_blocks);
    const narrativeBlocks = answerBlocks.filter((block) => !["TABLE", "CHART", "EVIDENCE_REFERENCES"].includes(block.block_type));
    const blocks = narrativeBlocks.map((block) => answerBlockHtml(block, turn)).join("");
    const evidence = turnEvidenceHtml(answerBlocks, turn.evidence_gaps);
    const plan = investigationPlanHtml(turn.investigation_plan);
    const answer = assistant || blocks || evidence ? `<article class="ops-message agent"><div class="ops-avatar">AI</div><div class="ops-message-body ops-result-markdown"><div class="ops-message-content">${blocks || markdown.render(assistant?.payload?.text || "")}</div>${evidence}</div></article>` : "";
    const settled = ["COMPLETED", "PARTIAL", "CANCELLED"].includes(turn.status);
    const progress = settled && !turn.error_message ? "" : `<div class="ops-context-banner ops-progress" data-turn-progress="${esc(turn.turn_id)}">${esc(turn.error_message || `当前状态：${turn.status}`)}</div>`;
    const hasHtmlLinks = answerBlocks.some((block) => block.block_type === "HTML_REPORT_LINKS");
    const workloadReports = hasHtmlLinks ? "" : workloadReportActions(turn);
    return `${user ? messageHtml("USER", user.payload?.text || "", shell.fmt(user.created_at), imageAttachmentsHtml(turn.conversation_id, turn)) : ""}${plan}${progress}${answer}${workloadReports}`;
  }

  async function renderConversation(conversation, turns) {
    state.conversation = conversation;
    state.turns = turns;
    document.getElementById("conversation-title").textContent = conversation.title || "诊断对话";
    document.getElementById("conversation-context").textContent = conversation.source_type === "RUN" ? "这次对话继承自告警或巡检结果。" : "人工发起的智能运维。";
    const panel = document.getElementById("message-list");
    panel.innerHTML = conversation.source_run_id ? '<div class="ops-context-banner">已关联来源诊断；后续回答只会引用当前 Turn 明确关联的证据。</div>' : "";
    turns.forEach((turn) => panel.insertAdjacentHTML("beforeend", turnHtml(turn)));
    const hasCompletedResult = turns.some((turn) => turn.ops_run_id && ["COMPLETED", "PARTIAL"].includes(turn.status));
    const hasActiveTurn = turns.some((turn) => !["COMPLETED", "PARTIAL", "FAILED", "CANCELLED"].includes(turn.status));
    if (hasCompletedResult) {
      const action = reportAction({
        conversationId: conversation.conversation_id,
        sourceKind: "CHAT",
      });
      panel.insertAdjacentHTML("beforeend", hasActiveTurn
        ? '<div class="ops-context-banner">会话仍有进行中的 Turn；全部结束后可生成覆盖整个会话的正式报告。</div>'
        : action);
    }
    await hydrateConversationImages(panel);
    await Promise.all(turns.filter((turn) => turn.status === "WAITING_USER" && turn.ops_run_id).map(async (turn) => {
      const progress = panel.querySelector(`[data-turn-progress="${String(turn.turn_id)}"]`);
      if (!progress) return;
      try {
        const run = await KBotAIOpsAuth.request(
          `${api}/runs/${encodeURIComponent(turn.ops_run_id)}`,
        );
        if (!["WAITING_INPUT", "WAITING_APPROVAL"].includes(run?.status)) {
          return;
        }
        const pending = await KBotAIOpsAuth.request(`${api}/runs/${encodeURIComponent(turn.ops_run_id)}/pending-input`);
        const html = diagnosticQueryApprovalHtml(pending);
        if (!html) return;
        progress.insertAdjacentHTML("beforebegin", html);
        bindDiagnosticQueryActions(progress.previousElementSibling);
      } catch (_) {
        // 已处理或并发恢复的审批不再展示，Turn事件流会给出最终状态。
      }
    }));
    panel.scrollTop = panel.scrollHeight;
    document.querySelectorAll("[data-copy-code]").forEach((button) => { button.onclick = () => markdown.copyCode(button); });
    bindWorkloadReportActions(panel);
    bindImplementationRunbookActions(panel);
    bindReportActions(panel);
    resumeActiveTurns(conversation.conversation_id, turns);
  }

  async function loadConversation(id) {
    const conversation = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`);
    const turnRows = await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns?limit=200`);
    const turns = await Promise.all(turnRows.map((turn) => KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}/turns/${encodeURIComponent(turn.turn_id)}`)));
    document.getElementById("agent-select").value = conversation.agent_id;
    document.getElementById("target-select").value = conversation.target_id;
    renderAgentOptions(conversation.agent_id);
    syncAgentTargetContext();
    await renderConversation(conversation, turns);
    history.replaceState(null, "", `./chat.html?conversation=${encodeURIComponent(id)}`);
    document.querySelectorAll(".ops-workspace-item").forEach((button) => { button.setAttribute("aria-current", String(button.dataset.id === id)); });
  }

  function resetConversationView({ agentSelected = false } = {}) {
    state.conversation = null;
    syncAgentTargetContext();
    document.getElementById("conversation-title").textContent = agentSelected
      ? "开始一次数据库诊断"
      : "请先选择 Target 和 Agent";
    document.getElementById("conversation-context").textContent = agentSelected
      ? "可以选择历史会话，或在下方发起一次新诊断。"
      : "选择 Target 和 Agent 后，才会显示对应的会话历史。";
    document.getElementById("message-list").innerHTML = agentSelected
      ? starterHomeHtml()
      : '<div class="ops-empty">请选择 Target 和 Agent 以查看历史并开始诊断。</div>';
    if (agentSelected) bindStarterButtons(document.getElementById("message-list"));
  }

  const starterCategoryLabels = {
    RECOMMENDED: "常用功能",
    DIAGNOSTIC: "诊断检查",
    REPORT: "性能报告",
    RUNBOOK: "实施文档",
  };

  function starterCardHtml(item) {
    const unavailable = item.status === "UNAVAILABLE";
    const hint = item.status === "LIMITED"
      ? item.availability_reason || "执行时将再次核验能力"
      : item.input_schema?.length ? "需要填写参数" : "点击后立即执行";
    return `<button class="ops-starter-card" type="button" data-starter-id="${esc(item.starter_id)}"${unavailable ? " disabled" : ""}><strong>${esc(item.title)}</strong><span>${esc(item.description)}</span><small>${unavailable ? esc(item.availability_reason || "当前不可用") : esc(hint)}</small></button>`;
  }

  function starterSectionsHtml(items, { recommendedOnly = false } = {}) {
    const categories = recommendedOnly ? ["RECOMMENDED", "REPORT", "RUNBOOK"] : ["RECOMMENDED", "DIAGNOSTIC", "REPORT", "RUNBOOK"];
    return categories.map((category) => {
      let rows = values(items).filter((item) => item.category === category);
      if (recommendedOnly) rows = rows.slice(0, category === "RECOMMENDED" ? 4 : 3);
      if (!rows.length) return "";
      return `<section class="ops-starter-section"><h4>${esc(starterCategoryLabels[category] || category)}</h4><div class="ops-starter-grid">${rows.map(starterCardHtml).join("")}</div></section>`;
    }).join("");
  }

  function starterHomeHtml() {
    if (!state.starters.length) return state.starterCatalogLoaded
      ? '<div class="ops-empty">功能目录暂不可用，仍可在下方直接输入数据库运维问题。</div>'
      : '<div class="ops-empty">正在读取当前 Target 可用功能…</div>';
    return `<section class="ops-starter-home"><header><h3>今天要处理什么？</h3><p>选择一个功能直接执行，或在下方输入任何数据库运维问题。</p></header>${starterSectionsHtml(state.starters, { recommendedOnly: true })}<button class="ops-button ops-starter-more" type="button" data-open-all-starters>查看全部功能</button></section>`;
  }

  function bindStarterButtons(root) {
    root.querySelectorAll("[data-starter-id]").forEach((button) => {
      button.onclick = () => selectStarter(button.dataset.starterId);
    });
    root.querySelectorAll("[data-open-all-starters]").forEach((button) => {
      button.onclick = openStarterMenu;
    });
  }

  async function loadConversationStarters() {
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    state.starters = [];
    state.starterCatalogVersion = "";
    state.starterCatalogLoaded = false;
    if (!agentId || !targetId) return;
    try {
      const payload = await KBotAIOpsAuth.request(`${api}/conversation-starters?agent_id=${encodeURIComponent(agentId)}&target_id=${encodeURIComponent(targetId)}`);
      state.starters = values(payload.starters);
      state.starterCatalogVersion = payload.catalog_version || "";
    } catch (error) {
      shell.toast(`功能目录读取失败：${error.message}`);
    } finally {
      state.starterCatalogLoaded = true;
    }
    if (!state.conversation) resetConversationView({ agentSelected: true });
  }

  function openStarterMenu() {
    if (!document.getElementById("agent-select").value) return shell.toast("请先选择 Agent");
    const dialog = document.getElementById("starter-dialog");
    document.getElementById("starter-dialog-title").textContent = "全部功能";
    document.getElementById("starter-dialog-description").textContent = "功能会根据当前 Target 类型和连接状态自动筛选。";
    const content = document.getElementById("starter-dialog-content");
    content.innerHTML = starterSectionsHtml(state.starters)
      || '<div class="ops-empty">当前没有可展示的功能，请直接在聊天框中描述问题。</div>';
    bindStarterButtons(content);
    if (!dialog.open) dialog.showModal();
  }

  function starterParameterValue(field, initialParameters = {}) {
    let value;
    if (Object.prototype.hasOwnProperty.call(initialParameters, field.name)) value = initialParameters[field.name];
    else if (field.type === "timezone") value = Intl.DateTimeFormat().resolvedOptions().timeZone || field.default || "Asia/Shanghai";
    else value = field.default ?? "";
    if (field.type === "oracle_datetime" && value) return String(value).replace(" ", "T");
    return value;
  }

  function starterParameterHtml(field, initialParameters = {}) {
    const isDateTime = field.type === "datetime" || field.type === "oracle_datetime";
    const type = isDateTime ? "datetime-local" : field.type === "integer" ? "number" : "text";
    const value = starterParameterValue(field, initialParameters);
    const attributes = [
      field.required ? "required" : "",
      field.min !== undefined ? `min="${esc(field.min)}"` : "",
      field.max !== undefined ? `max="${esc(field.max)}"` : "",
      isDateTime ? 'step="1"' : "",
      field.pattern ? `pattern="${esc(field.pattern)}"` : "",
      field.placeholder ? `placeholder="${esc(field.placeholder)}"` : "",
    ].filter(Boolean).join(" ");
    let control = `<input name="${esc(field.name)}" type="${type}" value="${esc(value)}" ${attributes}>`;
    if (field.type === "select") {
      const emptyOption = field.required ? "" : `<option value="">自动获取或按默认策略</option>`;
      control = `<select name="${esc(field.name)}" ${field.required ? "required" : ""}>${emptyOption}${values(field.options).map((option) => `<option value="${esc(option.value)}"${String(option.value) === String(value) ? " selected" : ""}>${esc(option.label || option.value)}</option>`).join("")}</select>`;
    } else if (field.type === "identifier_list") {
      control = `<textarea name="${esc(field.name)}" rows="3" ${attributes}>${esc(value)}</textarea>`;
    }
    return `<label><span>${esc(field.label)}${field.required ? " *" : ""}</span>${control}${field.help ? `<small>${esc(field.help)}</small>` : ""}</label>`;
  }

  function starterParameterGroupsHtml(item, initialParameters = {}) {
    const groups = new Map();
    values(item.input_schema).forEach((field) => {
      const group = field.group || "参数";
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(field);
    });
    return [...groups.entries()].map(([group, fields]) => {
      const help = group === "高级参数"
        ? "仅在需要覆盖自动识别结果时填写。"
        : fields.every((field) => !field.required)
          ? "所有字段均可留空，系统会继续使用数据库事实、Target 运维事实和默认策略。"
          : "请填写带星号的必填参数后继续。";
      return `<section class="ops-starter-parameter-group"><header><h4>${esc(group)}</h4><p>${help}</p></header><div>${fields.map((field) => starterParameterHtml(field, initialParameters)).join("")}</div></section>`;
    }).join("");
  }

  function selectStarter(starterId, initialParameters = {}) {
    const item = state.starters.find((row) => row.starter_id === starterId);
    if (!item) return shell.toast("功能目录已经更新，请重新打开菜单");
    if (item.status === "UNAVAILABLE") return shell.toast(item.availability_reason || "当前功能不可用");
    if (!values(item.input_schema).length) {
      document.getElementById("starter-dialog").close();
      executeStarter(item, {}).catch((error) => shell.toast(error.message));
      return;
    }
    const dialog = document.getElementById("starter-dialog");
    document.getElementById("starter-dialog-title").textContent = item.title;
    document.getElementById("starter-dialog-description").textContent = item.description;
    const content = document.getElementById("starter-dialog-content");
    const isRunbook = item.execution_mode === "RUNBOOK";
    content.innerHTML = `<div class="ops-starter-parameters">${starterParameterGroupsHtml(item, initialParameters)}</div><div class="ops-starter-form-actions"><button type="button" data-starter-back>返回功能列表</button>${isRunbook ? '<button type="button" data-starter-auto>使用自动配置生成</button>' : ""}<button class="primary" type="button" data-starter-submit>${isRunbook ? "按当前参数生成" : "开始执行"}</button></div>`;
    content.querySelector("[data-starter-back]").onclick = openStarterMenu;
    if (isRunbook) content.querySelector("[data-starter-auto]").onclick = () => {
      dialog.close();
      executeStarter(item, {}).catch((error) => shell.toast(error.message));
    };
    content.querySelector("[data-starter-submit]").onclick = () => {
      const parameters = {};
      for (const field of item.input_schema) {
        const input = content.querySelector(`[name="${CSS.escape(field.name)}"]`);
        if (!input.reportValidity()) return;
        let value = input.value.trim();
        if (!value && !field.required) continue;
        if (field.type === "datetime") value = new Date(value).toISOString();
        if (field.type === "oracle_datetime") {
          if (value.length === 16) value += ":00";
          value = value.replace("T", " ");
        }
        if (field.type === "integer") value = Number(value);
        parameters[field.name] = value;
      }
      dialog.close();
      executeStarter(item, parameters).catch((error) => shell.toast(error.message));
    };
    if (!dialog.open) dialog.showModal();
  }

  function starterSubmittedText(item, parameters) {
    const labels = Object.fromEntries(values(item.input_schema).map((field) => [field.name, field.label]));
    return [`执行功能：${item.title}`, ...Object.entries(parameters).map(([key, value]) => `${labels[key] || key}：${value}`)].join("\n");
  }

  async function executeStarter(item, parameters) {
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    const path = state.conversation
      ? `${api}/conversations/${state.conversation.conversation_id}/turns`
      : `${api}/conversations`;
    const starter = {
      starter_id: item.starter_id,
      catalog_version: state.starterCatalogVersion,
      parameters,
    };
    const body = state.conversation
      ? { content: [], starter }
      : { agent_id: agentId, target_id: targetId, content: [], starter };
    const receipt = await KBotAIOpsAuth.request(path, {
      method: "POST",
      headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
      body: JSON.stringify(body),
    });
    const panel = document.getElementById("message-list");
    if (!state.conversation) panel.innerHTML = "";
    panel.insertAdjacentHTML("beforeend", messageHtml("USER", starterSubmittedText(item, parameters)));
    panel.insertAdjacentHTML("beforeend", '<section id="live-progress" class="ops-context-banner ops-progress" aria-live="polite">正在按所选功能建立确定性执行计划…</section>');
    await followTurn(receipt.conversation_id, receipt.turn_id, document.getElementById("live-progress"));
    await loadConversation(receipt.conversation_id);
    await loadConversationList();
  }

  function setComposerAvailability(enabled) {
    const form = document.getElementById("conversation-form");
    form.elements.message.disabled = !enabled;
    form.querySelector('button[type="submit"]').disabled = !enabled;
    document.getElementById("open-starter-menu").disabled = !enabled;
  }

  function clearConversationUrl() {
    history.replaceState(null, "", "./chat.html");
  }

  async function archiveConversation(id, title) {
    if (!confirm(`确认删除会话“${title || "未命名会话"}”吗？\n\n会话将从聊天历史中移除；关联的诊断、证据和变更审计记录仍会保留。`)) return;
    await KBotAIOpsAuth.request(`${api}/conversations/${encodeURIComponent(id)}`, {
      method: "DELETE",
    });
    if (String(state.conversation?.conversation_id) === String(id)) {
      clearConversationUrl();
      resetConversationView({ agentSelected: true });
    }
    shell.toast("会话已从历史中移除");
    await loadConversationList();
  }

  function renderConversationList(rows) {
    const list = document.getElementById("conversation-list");
    list.innerHTML = rows.length ? rows.map((item) => `<div class="ops-workspace-row"><button class="ops-workspace-item" type="button" data-id="${esc(item.conversation_id)}"><strong>${esc(item.title || "未命名会话")}</strong><small>${esc(item.source_type)} · ${esc(shell.fmt(item.updated_at))}</small></button><button class="ops-workspace-delete" type="button" data-delete-id="${esc(item.conversation_id)}" data-delete-title="${esc(item.title || "未命名会话")}" aria-label="删除会话 ${esc(item.title || "未命名会话")}" title="删除会话">删除</button></div>`).join("") : '<div class="ops-empty">当前 Agent 还没有诊断会话</div>';
    list.querySelectorAll(".ops-workspace-item").forEach((button) => {
      button.onclick = () => loadConversation(button.dataset.id).catch((error) => shell.toast(error.message));
    });
    list.querySelectorAll(".ops-workspace-delete").forEach((button) => {
      button.onclick = () => archiveConversation(button.dataset.deleteId, button.dataset.deleteTitle).catch((error) => shell.toast(error.message));
    });
  }

  async function loadConversationList(preferredId) {
    const select = document.getElementById("agent-select");
    const requestedId = preferredId || new URLSearchParams(location.search).get("conversation");
    if (!select.value && requestedId) await loadConversation(requestedId);
    const selectedAgent = select.value;
    const selectedTarget = document.getElementById("target-select").value;
    const list = document.getElementById("conversation-list");
    if (!selectedAgent || !selectedTarget) {
      list.innerHTML = '<div class="ops-empty">请先选择 Target 和 Agent 查看会话历史</div>';
      setComposerAvailability(false);
      resetConversationView();
      return;
    }
    setComposerAvailability(true);
    await loadConversationStarters();
    const rows = await KBotAIOpsAuth.request(`${api}/conversations?agent_id=${encodeURIComponent(selectedAgent)}&target_id=${encodeURIComponent(selectedTarget)}`);
    renderConversationList(rows);
    if (requestedId && String(state.conversation?.conversation_id) !== String(requestedId)) {
      await loadConversation(requestedId);
    }
  }

  function typingUnits(value) {
    const text = String(value || "");
    if (!text) return [];
    if (!graphemeSegmenter) return Array.from(text);
    return Array.from(graphemeSegmenter.segment(text), (item) => item.segment);
  }

  function typingBatchSize(pending) {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return pending.queue.length;
    if (pending.finalizing) return Math.max(1, Math.ceil(pending.queue.length / 16));
    if (pending.queue.length > 800) return 8;
    if (pending.queue.length > 240) return 4;
    if (pending.queue.length > 80) return 2;
    return 1;
  }

  function settleTyping(pending) {
    pending.timer = null;
    pending.message.classList.remove("is-typing");
    pending.message.setAttribute("aria-busy", "false");
    pending.waiters.splice(0).forEach((resolve) => resolve());
  }

  function typingTick(pending) {
    if (!pending.queue.length) return settleTyping(pending);
    pending.displayedMarkdown += pending.queue.splice(0, typingBatchSize(pending)).join("");
    pending.content.innerHTML = markdown.render(pending.displayedMarkdown);
    const panel = document.getElementById("message-list");
    panel.scrollTop = panel.scrollHeight;
    pending.timer = window.setTimeout(() => typingTick(pending), typingFrameMs);
  }

  function enqueueAnswerDelta(pending, delta) {
    pending.queue.push(...typingUnits(delta));
    pending.message.classList.add("is-typing");
    pending.message.setAttribute("aria-busy", "true");
    if (pending.timer === null) typingTick(pending);
  }

  function waitForTyping(pending) {
    if (!pending.queue.length && pending.timer === null) return Promise.resolve();
    return new Promise((resolve) => pending.waiters.push(resolve));
  }

  async function followTurn(conversationId, turnId, progress) {
    ensureProgressTimeline(progress);
    const elapsedTimer = window.setInterval(() => updateProgressElapsed(progress), 1000);
    let pending = null;
    let lastEventId = "";
    let completed = false;
    const path = `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}/events`;
    const onEvent = ({ event, data, id }) => {
      if (id) lastEventId = id;
      const payload = data?.payload || {};
      if ([
        "turn.created", "planning.started", "planning.route.selected", "turn.status",
        "input.analysis.started", "input.analysis.completed",
        "task.frame.completed", "investigation.planned",
        "playbook.completed", "tool.started", "tool.completed",
        "tool.gap", "evidence.added", "assessment.started", "assessment.completed",
        "investigation.replanned", "diagnostic.query_approval_required",
        "diagnostic.query_approved", "diagnostic.query_rejected",
      ].includes(event)) {
        appendProgress(progress, event, payload, id);
      }
      if (["investigation.planned", "investigation.replanned"].includes(event)) {
        showInvestigationPlan(progress, payload.plan);
      }
      if (event === "diagnostic.query_approval_required" && payload.hitl_id) {
        showDiagnosticQueryApproval(progress, payload.hitl_id).catch((error) => shell.toast(error.message));
      }
      if (event === "thinking.delta") {
        appendProgress(progress, event, {
          public_summary: payload.public_summary || payload.delta || "正在组织回答",
        }, id);
      }
      if (event === "answer.delta") {
        const delta = String(data?.payload?.delta || "");
        if (!pending) {
          progress.insertAdjacentHTML("afterend", messageHtml("AGENT", ""));
          const message = progress.nextElementSibling;
          pending = {
            message,
            content: message.querySelector(".ops-message-content"),
            displayedMarkdown: "",
            queue: [],
            timer: null,
            waiters: [],
            finalizing: false,
          };
        }
        enqueueAnswerDelta(pending, delta);
      }
      if (event === "answer.completed" && pending) pending.finalizing = true;
      if (event === "done") {
        completed = true;
        if (pending) pending.finalizing = true;
        appendProgress(progress, event, { public_summary: "诊断已完成，正在整理结论…" }, id);
      }
    };
    for (let attempt = 0; attempt < streamRecoveryAttempts && !completed; attempt += 1) {
      let streamFailed = false;
      try {
        await KBotAIOpsAuth.stream(path, onEvent, {
          headers: lastEventId ? { "Last-Event-ID": lastEventId } : {},
        });
      } catch (_) {
        streamFailed = true;
        // 临时断流统一由权威 Turn 状态与续传游标恢复。
      }
      if (completed) break;
      try {
        const turn = await KBotAIOpsAuth.request(
          `${api}/conversations/${encodeURIComponent(conversationId)}/turns/${encodeURIComponent(turnId)}`,
        );
        if (terminalTurnStatuses.has(turn.status)) {
          completed = true;
          break;
        }
      } catch (_) {
        // 状态回读也可能遇到同一次短暂网络抖动，下一轮继续恢复。
      }
      appendProgress(progress, "stream.recovery", {
        public_summary: streamFailed
          ? "事件流暂时中断，正在恢复诊断进度…"
          : "诊断仍在后台运行，正在继续获取进度…",
      }, "stream.recovery");
      await new Promise((resolve) => window.setTimeout(resolve, Math.min(5000, 1000 * (attempt + 1))));
    }
    window.clearInterval(elapsedTimer);
    updateProgressElapsed(progress);
    if (!completed) throw new Error("诊断仍在后台运行，请稍后刷新会话查看结果");
    if (pending) {
      pending.finalizing = true;
      await waitForTyping(pending);
    }
  }

  function resumeActiveTurns(conversationId, turns) {
    turns.filter((turn) => !terminalTurnStatuses.has(turn.status)).forEach((turn) => {
      const turnId = String(turn.turn_id);
      const followerKey = `${conversationId}:${turnId}`;
      const progress = document.querySelector(`[data-turn-progress="${turnId}"]`);
      if (!progress || activeTurnFollowers.has(followerKey)) return;
      activeTurnFollowers.add(followerKey);
      followTurn(conversationId, turnId, progress)
        .then(() => {
          if (String(state.conversation?.conversation_id) === String(conversationId)) {
            return loadConversation(conversationId);
          }
          return null;
        })
        .catch((error) => shell.toast(error.message))
        .finally(() => activeTurnFollowers.delete(followerKey));
    });
  }

  async function submitConversation(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const button = form.querySelector('button[type="submit"]');
    const agentId = document.getElementById("agent-select").value;
    const targetId = document.getElementById("target-select").value;
    const agent = state.agents.find((item) => String(item.agent_id) === String(agentId));
    const text = form.elements.message.value.trim();
    const selectedFiles = state.selectedFiles;
    if (!agentId) return shell.toast("请先选择 Agent");
    if (!targetId) return shell.toast("请先选择逻辑 Target");
    if (!values(agent?.target_ids).includes(targetId)) return shell.toast("当前 Agent 未绑定所选 Target");
    if (!text && !selectedFiles.length) return shell.toast("请输入问题或上传诊断材料");
    button.disabled = true;
    try {
      const path = state.conversation
        ? `${api}/conversations/${state.conversation.conversation_id}/turns`
        : `${api}/conversations`;
      const content = text ? [{ content_type: "TEXT", text }] : [];
      for (const selectedFile of selectedFiles) {
        document.getElementById("upload-preview").textContent =
          `正在上传诊断材料：${selectedFile.name}`;
        const uploaded = await KBotAIOpsAuth.request(`${api}/conversation-uploads`, {
          method: "POST",
          headers: {
            "Content-Type": uploadMediaType(selectedFile),
            "X-File-Name": encodeURIComponent(selectedFile.name),
          },
          body: selectedFile,
        });
        content.push({
          content_type: uploaded.media_type.startsWith("image/") ? "IMAGE" : "FILE",
          upload_id: uploaded.upload_id,
          media_type: uploaded.media_type,
        });
      }
      const body = state.conversation
        ? { content }
        : { agent_id: agentId, target_id: targetId, content };
      const receipt = await KBotAIOpsAuth.request(path, {
        method: "POST",
        headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() },
        body: JSON.stringify(body),
      });
      form.reset(); state.selectedFiles = []; document.getElementById("upload-preview").textContent = "";
      const panel = document.getElementById("message-list");
      const submittedText = [
        text,
        ...selectedFiles.map((file) => `已上传诊断材料：${file.name}`),
      ].filter(Boolean).join("\n\n");
      panel.insertAdjacentHTML("beforeend", messageHtml("USER", submittedText));
      panel.insertAdjacentHTML("beforeend", '<section id="live-progress" class="ops-context-banner ops-progress" aria-live="polite">正在建立诊断计划：先固定执行上下文，再理解问题并选择证据…</section>');
      const progress = document.getElementById("live-progress");
      await followTurn(receipt.conversation_id, receipt.turn_id, progress);
      await loadConversation(receipt.conversation_id);
      await loadConversationList();
    } catch (error) { shell.toast(error.message); } finally { button.disabled = false; }
  }

  async function initChat() {
    const select = document.getElementById("agent-select");
    await agents();
    const targetSelect = document.getElementById("target-select");
    targetSelect.innerHTML = '<option value="">选择逻辑 Target</option>' + state.targets.map((item) => `<option value="${esc(item.target_id)}">${esc(item.display_name)} · ${esc(item.db_type)}${item.readonly_connection_enabled ? "" : " · 仅监控"}</option>`).join("");
    const requestedTargetId = new URLSearchParams(location.search).get("target_id");
    if (requestedTargetId && state.targets.some((item) => String(item.target_id) === requestedTargetId)) {
      targetSelect.value = requestedTargetId;
      renderAgentOptions();
      syncAgentTargetContext();
    }
    targetSelect.onchange = () => {
      clearConversationUrl();
      state.conversation = null;
      renderAgentOptions();
      syncAgentTargetContext();
      resetConversationView();
      loadConversationList().catch((error) => shell.toast(error.message));
    };
    select.onchange = () => {
      clearConversationUrl();
      syncAgentTargetContext();
      resetConversationView({ agentSelected: Boolean(select.value) });
      loadConversationList().catch((error) => shell.toast(error.message));
    };
    document.getElementById("new-conversation").onclick = () => {
      if (!select.value) return shell.toast("请先选择 Agent");
      clearConversationUrl();
      resetConversationView({ agentSelected: true });
    };
    document.getElementById("conversation-form").onsubmit = submitConversation;
    document.getElementById("open-starter-menu").onclick = openStarterMenu;
    document.getElementById("evidence-file").onchange = (event) => {
      const files = Array.from(event.target.files || []);
      if (files.length > maxDiagnosticFiles) {
        event.target.value = "";
        state.selectedFiles = [];
        return shell.toast(`单次最多选择 ${maxDiagnosticFiles} 份诊断材料`);
      }
      if (files.some((file) => file.size > 20 * 1024 * 1024)) {
        event.target.value = "";
        state.selectedFiles = [];
        return shell.toast("每份诊断材料不能超过 20 MiB");
      }
      state.selectedFiles = files;
      document.getElementById("upload-preview").textContent = files.length
        ? `已选择 ${files.length} 份诊断材料：${files.map((file) => file.name).join("、")}（点击发送时上传）`
        : "";
    };
    await loadConversationList();
  }

  function renderAgentOptions(preferredId = "") {
    const targetId = document.getElementById("target-select").value;
    const select = document.getElementById("agent-select");
    const rows = state.agents.filter((item) => values(item.target_ids).includes(targetId));
    select.disabled = !targetId;
    select.innerHTML = '<option value="">选择 Agent</option>' + rows.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("");
    if (rows.some((item) => String(item.agent_id) === String(preferredId))) select.value = preferredId;
  }

  function continueForm(source, title) {
    const allowed = state.agents.filter((item) => values(item.target_ids).includes(String(source.target_id)));
    const inherited = source.source_run_id ? "自动诊断结果与证据" : "告警情境及其监控信号";
    const prompt = source.source_run_id
      ? `请基于“${title}”的自动诊断结果继续分析：`
      : `请基于告警情境“${title}”开始诊断，并核验关联监控证据：`;
    return `<div class="ops-inline-dialog"><h3>继续深入诊断</h3><p>系统会在服务端继承本次${inherited}；后续人工对话才可能按 Agent 权限产生审批待办。</p><form id="continue-form" class="ops-form"><div class="ops-field span-12"><label>Agent</label><select name="agent_id" required><option value="">请选择</option>${allowed.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.name || item.agent_key || shell.short(item.agent_id))}</option>`).join("")}</select></div><div class="ops-field span-12"><label>继续追问</label><textarea name="message" required>${esc(prompt)}</textarea></div><div class="ops-filter-actions"><button class="primary" type="submit">进入对话</button></div></form></div>`;
  }

  async function bindContinue(source) {
    const form = document.getElementById("continue-form");
    if (!form) return;
    form.onsubmit = async (event) => {
      event.preventDefault();
      const fields = Object.fromEntries(new FormData(form));
      const body = {
        agent_id: fields.agent_id,
        target_id: source.target_id,
        content: [{ content_type: "TEXT", text: fields.message }],
      };
      if (source.source_run_id) body.source_run_id = source.source_run_id;
      if (source.source_situation_id) body.source_situation_id = source.source_situation_id;
      try {
        const result = await KBotAIOpsAuth.request(`${api}/conversations`, { method: "POST", headers: { "Idempotency-Key": KBotAIOpsAuth.uuid() }, body: JSON.stringify(body) });
        location.href = `./chat.html?conversation=${encodeURIComponent(result.conversation_id)}`;
      } catch (error) { shell.toast(error.message); }
    };
  }

  function scheduleSituationRefresh(item, delayMs) {
    window.clearTimeout(situationRefreshTimer);
    situationRefreshTimer = window.setTimeout(() => {
      loadSituation(item, { background: true });
    }, delayMs);
  }

  async function loadSituation(item, { background = false } = {}) {
    const situationId = String(item.situation_id);
    if (activeSituationId !== situationId) return;
    try {
      const detail = await KBotAIOpsAuth.request(`${api}/situations/${encodeURIComponent(situationId)}`);
      if (activeSituationId !== situationId) return;
      document.getElementById("case-title").textContent = detail.title;
      const panel = document.getElementById("case-detail");
      const selectedAgentId = document.getElementById("case-filters")?.elements.agent_id.value || "";
      let run = null; let result = null;
      if (detail.run_ids.length) {
        const runs = await Promise.all(detail.run_ids.map((runId) => KBotAIOpsAuth.request(`${api}/runs/${runId}`)));
        run = runs.find((candidate) => !selectedAgentId || String(candidate.agent_id) === selectedAgentId) || runs[0];
        try {
          result = await KBotAIOpsAuth.request(`${api}/runs/${run.ops_run_id}/result`);
        } catch (error) {
          if (error.status !== 404 && error.status !== 409) throw error;
        }
      }
      if (activeSituationId !== situationId) return;
      const hasFinalResult = Boolean(result?.final_artifact);
      const source = hasFinalResult
        ? { target_id: run.target_id, source_run_id: run.ops_run_id }
        : { target_id: detail.target_id, source_situation_id: detail.situation_id };
      let diagnosis;
      if (hasFinalResult) {
        diagnosis = `<div class="ops-result-markdown">${conversationAnswerHtml(result)}${evidenceDetails(result)}</div>`;
      } else if (run && !terminalRunStatuses.has(run.status)) {
        diagnosis = `<div class="ops-empty">Agent 正在诊断，当前状态：${esc(run.status)}。页面会自动更新结果。</div>`;
      } else if (run) {
        const errorCode = run.error_code
          ? `<div><strong>错误码：</strong><code>${esc(run.error_code)}</code></div>`
          : "";
        const errorMessage = run.error_message
          ? `<div>${esc(run.error_message)}</div>`
          : '<div>本次自动诊断未形成可展示结果，请依据运行状态重试或联系管理员。</div>';
        diagnosis = `<div class="ops-error"><div><strong>自动诊断状态：</strong>${esc(run.status)}</div>${errorCode}${errorMessage}</div>`;
      } else {
        diagnosis = '<div class="ops-empty">告警已接收，正在等待 Agent 自动诊断任务启动。</div>';
      }
      const report = hasFinalResult ? reportAction({ runId: run.ops_run_id, sourceKind: "ALERT" }) : "";
      panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.severity)} ${shell.badge(detail.status)} · ${esc(situationStatusText(detail.status))} · 累计 ${esc(detail.event_count)} 次观测 · 最近观测 ${esc(shell.fmt(detail.last_observed_at))}</div>${monitoringSourceSummary(detail)}${situationAlertContent(detail)}${diagnosis}${report}${continueForm(source, detail.title)}`;
      await bindContinue(source);
      bindReportActions(panel);
      const runActive = !run || !terminalRunStatuses.has(run.status);
      if (runActive) scheduleSituationRefresh(item, 3000);
      else if (detail.status !== "RESOLVED") scheduleSituationRefresh(item, 15000);
    } catch (error) {
      if (!background) throw error;
      if (activeSituationId === situationId) {
        scheduleSituationRefresh(item, 5000);
      }
    }
  }

  async function showSituation(item) {
    activeSituationId = String(item.situation_id);
    window.clearTimeout(situationRefreshTimer);
    await loadSituation(item);
  }

  async function showInspection(item) {
    const detail = await KBotAIOpsAuth.request(`${api}/inspection-fires/${encodeURIComponent(item.fire_id)}`);
    document.getElementById("case-title").textContent = `巡检 ${shell.fmt(detail.scheduled_at)}`;
    const panel = document.getElementById("case-detail");
    const runId = detail.run_ids[0];
    let run = null; let result = null;
    if (runId) { run = await KBotAIOpsAuth.request(`${api}/runs/${runId}`); result = await KBotAIOpsAuth.request(`${api}/runs/${runId}/result`); }
    const source = run ? { target_id: run.target_id, source_run_id: run.ops_run_id } : null;
    const reportActionHtml = result?.final_artifact?.schema_version === "REPORT_CONTENT.v1"
      ? ""
      : reportAction({ runId: run?.ops_run_id, sourceKind: "INSPECTION", periodKind: "DAILY" });
    panel.innerHTML = `<div class="ops-context-banner">${shell.badge(detail.status)} · ${detail.completed_count}/${detail.target_count} 个目标完成 · ${detail.failed_count} 个失败</div>${result ? `<div class="ops-result-markdown">${inspectionAnswerHtml(result)}${evidenceDetails(result)}</div>${reportActionHtml}${continueForm(source, "本次日常巡检")}` : '<div class="ops-empty">本次巡检尚未形成可展示结果。</div>'}`;
    if (source) await bindContinue(source);
    bindReportActions(panel);
    bindWorkloadReportActions(panel);
    bindImplementationRunbookActions(panel);
  }

  async function initCases(page) {
    await agents();
    const requested = new URLSearchParams(location.search);
    const requestedTargetId = requested.get("target_id");
    const requestedSituationId = page === "situations" ? requested.get("situation") : "";
    state.caseTargets = page === "situations"
      ? values((await KBotAIOpsAuth.request(`${api}/targets?limit=200`))?.items)
      : state.targets;
    const list = document.getElementById("case-list");
    const filters = document.getElementById("case-filters");
    if (filters) {
      filters.elements.agent_id.innerHTML = '<option value="">全部 Agent</option>' + state.caseAgents.map((item) => `<option value="${esc(item.agent_id)}">${esc(item.display_name || item.agent_key || shell.short(item.agent_id))}</option>`).join("");
      filters.elements.target_id.innerHTML = '<option value="">全部 Target</option>' + state.caseTargets.map((item) => `<option value="${esc(item.target_id)}">${esc(item.display_name)} · L${esc(item.importance_level)}</option>`).join("");
      if (requestedTargetId && state.caseTargets.some((item) => String(item.target_id) === requestedTargetId)) {
        filters.elements.target_id.value = requestedTargetId;
      }
    }
    const loadRows = async () => {
      window.clearTimeout(situationRefreshTimer);
      const endpoint = page === "situations" ? "/situations" : "/inspection-fires";
      const query = new URLSearchParams({ limit: "100" });
      if (filters) {
        for (const name of ["agent_id", "severity", "target_id"]) {
          if (filters.elements[name].value) query.set(name, filters.elements[name].value);
        }
      }
      const payload = await KBotAIOpsAuth.request(`${api}${endpoint}?${query}`);
      const rows = payload.items || [];
      const targetNames = new Map(state.caseTargets.map((target) => [String(target.target_id), target.display_name]));
      list.innerHTML = rows.length ? rows.map((item) => `<button class="ops-case-row" data-id="${esc(item.situation_id || item.fire_id)}"><strong>${esc(item.title || `巡检 ${shell.fmt(item.scheduled_at)}`)}</strong>${shell.badge(item.severity || item.status)}<p>${esc(item.summary || targetNames.get(String(item.target_id)) || `${item.completed_count || 0}/${item.target_count || 0} 个目标已完成`)}</p></button>`).join("") : '<div class="ops-empty">当前筛选范围内暂无记录</div>';
      list.querySelectorAll("button").forEach((button, index) => { button.onclick = () => (page === "situations" ? showSituation(rows[index]) : showInspection(rows[index])).catch((error) => shell.toast(error.message)); });
      const selected = page === "situations" && requestedSituationId
        ? rows.find((item) => String(item.situation_id) === requestedSituationId) || rows[0]
        : rows[0];
      if (selected) await (page === "situations" ? showSituation(selected) : showInspection(selected));
      else {
        activeSituationId = null;
        document.getElementById("case-title").textContent = page === "situations" ? "没有匹配的告警事件" : "没有匹配的巡检";
        document.getElementById("case-detail").innerHTML = '<div class="ops-empty">请调整筛选条件后重试。</div>';
      }
    };
    if (filters) {
      filters.onsubmit = (event) => { event.preventDefault(); loadRows().catch((error) => shell.toast(error.message)); };
      filters.onreset = () => window.setTimeout(() => loadRows().catch((error) => shell.toast(error.message)), 0);
    }
    document.getElementById("refresh-workspace").onclick = () => loadRows().catch((error) => shell.toast(error.message));
    await loadRows();
  }

  addEventListener("click", (event) => {
    const copyButton = event.target.closest("[data-copy-code]");
    if (copyButton) markdown.copyCode(copyButton);
    const proposalButton = event.target.closest("[data-approve-proposal],[data-reject-proposal]");
    if (proposalButton) proposalAction(proposalButton);
    const manualButton = event.target.closest("[data-manual-proposal]");
    if (manualButton) submitManualResult(manualButton);
  });
  addEventListener("submit", (event) => {
    const form = event.target.closest("[data-confirm-target-fact]");
    if (!form) return;
    event.preventDefault();
    confirmTargetFact(form);
  });
  shell.ready.then((access) => {
    state.permissions = new Set(access?.permissions || []);
    const page = document.body.dataset.page;
    if (page === "chat") return initChat();
    if (["situations", "inspections"].includes(page)) return initCases(page);
    return null;
  }).catch((error) => shell.toast(error.message));
})();
