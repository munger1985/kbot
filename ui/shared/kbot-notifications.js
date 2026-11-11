(function () {
  "use strict";
  const isUuid = value => /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(String(value || ""));
  function href(item, domainId) {
    const params = item.route_params || {};
    const scope = String(domainId ?? "");
    if (!/^[1-9]\d*$/.test(scope) || String(item.domain_id) !== scope || String(params.domain_id) !== scope) return null;
    if (item.route_key === "aiops.run.detail") {
      if (!isUuid(params.run_id)) return null;
      const query = new URLSearchParams({ runId: params.run_id });
      for (const key of ["proposal_id", "hitl_id", "execution_id"]) {
        if (params[key]) { if (!isUuid(params[key])) return null; query.set(key, params[key]); }
      }
      return "./runs.html?" + query;
    }
    if (item.route_key === "aiops.situation.detail") {
      return isUuid(params.situation_id) ? "./situations.html?" + new URLSearchParams({ situationId: params.situation_id }) : null;
    }
    if (item.route_key === "aiops.report.detail" && isUuid(params.report_id)) {
      const query = new URLSearchParams({ id: params.report_id });
      if (params.job_id) { if (!isUuid(params.job_id)) return null; query.set("job_id", params.job_id); }
      if (params.report_version) {
        if (!/^[1-9]\d*$/.test(String(params.report_version))) return null;
        query.set("report_version", params.report_version);
      }
      return "./report-detail.html?" + query;
    }
    if (item.route_key !== "aiops.output_job.detail" || !isUuid(params.job_id)
      || !["CASE_EXTRACTION", "REPORT_GENERATION"].includes(params.job_type)) return null;
    const query = new URLSearchParams({ job_id: params.job_id });
    if (params.asset_id || params.asset_version_id) {
      if (params.job_type !== "CASE_EXTRACTION" || !isUuid(params.asset_id) || !isUuid(params.asset_version_id)) return null;
      query.set("asset_id", params.asset_id); query.set("asset_version_id", params.asset_version_id);
    }
    return (params.job_type === "CASE_EXTRACTION" ? "./cases.html?" : "./reports.html?") + query;
  }

  function mount(container, { request, domainId, toast = () => {}, navigate = url => location.assign(url) }) {
    const element = document.createElement("div");
    element.className = "kbot-notifications";
    element.innerHTML = '<button class="icon-button notification-trigger" type="button" aria-label="打开通知中心" aria-haspopup="dialog" aria-expanded="false"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4"/></svg><b class="notification-badge" data-notification-count hidden></b></button><dialog class="kbot-notification-panel" aria-label="通知中心"><header><div><p class="eyebrow">通知中心</p><h2>需要关注的事项</h2><p>重要结果和系统提醒集中在这里。</p></div><button type="button" class="icon-button" aria-label="关闭通知中心" data-notification-close autofocus>×</button></header><div class="notification-toolbar"><span>全部通知</span><button type="button" class="row-action" data-notification-refresh>刷新</button></div><p class="muted" data-notification-status role="status"></p><div data-notification-list></div><footer><a href="./dashboard.html">返回 AIOps 工作台</a></footer></dialog>';
    container.prepend(element);
    const toggle = element.querySelector("[aria-expanded]");
    const panel = element.querySelector("dialog");
    const status = element.querySelector("[data-notification-status]");
    const list = element.querySelector("[data-notification-list]");
    let controller, timer, sequence = 0, stopped = false;
    function schedule() {
      clearTimeout(timer);
      if (!stopped && !document.hidden) timer = setTimeout(refresh, 30000);
    }
    function close() { panel.close(); toggle.setAttribute("aria-expanded", "false"); toggle.focus(); }
    async function openItem(item) {
      try {
        const current = await request("/api/v1/notifications/" + encodeURIComponent(item.inbox_id));
        const destination = href(current, domainId);
        if (!destination) throw new Error("该通知的结果定位已不可用，请在对应页面核对生成记录。");
        if (!current.read_at) await request("/api/v1/notifications/" + encodeURIComponent(current.inbox_id), {
          method: "PATCH", body: JSON.stringify({ read: true, expected_row_version: current.row_version }),
        });
        navigate(destination);
      } catch (error) { toast(error.message || "无法打开通知"); }
    }
    async function refresh() {
      clearTimeout(timer); controller?.abort();
      controller = new AbortController(); const ticket = ++sequence;
      try {
        const summary = await request("/api/v1/notifications/summary", { signal: controller.signal });
        if (ticket !== sequence || stopped) return;
        const unread = Number(summary.unread) || 0;
        const badge = element.querySelector("[data-notification-count]");
        badge.hidden = unread <= 0; badge.textContent = unread > 99 ? "99+" : String(unread);
        if (panel.open) {
          const page = await request("/api/v1/notifications?limit=20", { signal: controller.signal });
          if (ticket !== sequence || stopped) return;
          list.replaceChildren();
          (page.items || []).forEach(item => {
            const article = document.createElement("article");
            article.className = "notification-card" + (!item.read_at ? " unread" : "");
            const title = document.createElement("strong"); title.textContent = item.title || "通知";
            const summaryText = document.createElement("p"); summaryText.textContent = item.summary || "";
            article.append(title, summaryText);
            if (["INFO", "WARNING", "HIGH", "CRITICAL"].includes(item.source_severity)) {
              const severity = document.createElement("small");
              severity.textContent = "原始告警级别：" + item.source_severity;
              article.append(severity);
            }
            if (href(item, domainId)) {
              const action = document.createElement("button"); action.type = "button"; action.className = "row-action";
              action.textContent = "查看结果"; action.onclick = () => openItem(item); article.append(action);
            }
            list.append(article);
          });
          status.textContent = page.items?.length ? "显示最近 20 条通知；生成阶段与失败重试请查看对应页面的生成记录。" : "暂无通知。";
        }
      } catch (error) {
        if (ticket !== sequence || stopped || error.name === "AbortError") return;
        list.replaceChildren(); status.textContent = "通知暂不可用，请刷新重试。";
        element.querySelector("[data-notification-count]").hidden = true;
      } finally { if (ticket === sequence) schedule(); }
    }
    toggle.onclick = () => {
      if (panel.open) { close(); return; }
      panel.showModal(); toggle.setAttribute("aria-expanded", "true");
      void refresh();
    };
    element.querySelector("[data-notification-close]").onclick = close;
    element.querySelector("[data-notification-refresh]").onclick = refresh;
    // 与 Ammolite 抽屉一致，只允许明确的关闭按钮，避免误触背景或 Esc 丢失上下文。
    panel.addEventListener("cancel", event => event.preventDefault());
    panel.addEventListener("keydown", event => {
      if (event.key !== "Tab") return;
      const fields = [...panel.querySelectorAll("button:not(:disabled), a[href]")].filter(node => node.getClientRects().length);
      const first = fields[0], last = fields.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    });
    const pause = () => { stopped = true; sequence += 1; clearTimeout(timer); controller?.abort(); };
    const resume = () => { stopped = false; if (!document.hidden) void refresh(); };
    const visibility = () => document.hidden ? pause() : resume();
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("pagehide", pause); window.addEventListener("pageshow", resume);
    void refresh();
    return { refresh, destroy() {
      pause(); if (panel.open) panel.close(); element.remove();
      document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("pagehide", pause); window.removeEventListener("pageshow", resume);
    } };
  }
  globalThis.KBotNotificationCenter = { mount, href };
})();
