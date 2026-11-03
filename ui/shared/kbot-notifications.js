(function () {
  "use strict";
  const isUuid = value => /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(String(value || ""));
  function href(item, domainId) {
    const params = item.route_params || {};
    if (String(item.domain_id) !== String(domainId) || String(params.domain_id) !== String(domainId)) return null;
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
    element.className = "account-menu kbot-notifications";
    element.innerHTML = '<button class="account-button" type="button" aria-label="通知中心" aria-expanded="false">通知 <span data-notification-count>—</span></button><section class="account-menu-popover kbot-notification-panel" aria-label="通知中心" hidden><header><strong>通知中心</strong><button type="button" class="row-action" data-notification-refresh>刷新</button></header><p class="muted" data-notification-status role="status"></p><div data-notification-list></div></section>';
    container.prepend(element);
    const toggle = element.querySelector("[aria-expanded]");
    const panel = element.querySelector("section");
    const status = element.querySelector("[data-notification-status]");
    const list = element.querySelector("[data-notification-list]");
    let controller, timer, sequence = 0, stopped = false;
    function schedule() {
      clearTimeout(timer);
      if (!stopped && !document.hidden) timer = setTimeout(refresh, 30000);
    }
    function close() { panel.hidden = true; toggle.setAttribute("aria-expanded", "false"); }
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
        element.querySelector("[data-notification-count]").textContent = String(summary.unread || 0);
        if (!panel.hidden) {
          const page = await request("/api/v1/notifications?limit=20", { signal: controller.signal });
          if (ticket !== sequence || stopped) return;
          list.replaceChildren();
          (page.items || []).forEach(item => {
            const article = document.createElement("article");
            const title = document.createElement("strong"); title.textContent = item.title || "通知";
            const summaryText = document.createElement("p"); summaryText.textContent = item.summary || "";
            article.append(title, summaryText);
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
        element.querySelector("[data-notification-count]").textContent = "—";
      } finally { if (ticket === sequence) schedule(); }
    }
    toggle.onclick = () => {
      panel.hidden = !panel.hidden; toggle.setAttribute("aria-expanded", String(!panel.hidden));
      if (!panel.hidden) void refresh();
    };
    element.querySelector("[data-notification-refresh]").onclick = refresh;
    const outside = event => { if (!element.contains(event.target)) close(); };
    const escape = event => { if (event.key === "Escape" && !panel.hidden) { close(); toggle.focus(); } };
    const pause = () => { stopped = true; sequence += 1; clearTimeout(timer); controller?.abort(); };
    const resume = () => { stopped = false; if (!document.hidden) void refresh(); };
    const visibility = () => document.hidden ? pause() : resume();
    document.addEventListener("pointerdown", outside); document.addEventListener("keydown", escape);
    document.addEventListener("visibilitychange", visibility);
    window.addEventListener("pagehide", pause); window.addEventListener("pageshow", resume);
    void refresh();
    return { refresh, destroy() {
      pause(); element.remove(); document.removeEventListener("pointerdown", outside);
      document.removeEventListener("keydown", escape); document.removeEventListener("visibilitychange", visibility);
      window.removeEventListener("pagehide", pause); window.removeEventListener("pageshow", resume);
    } };
  }
  globalThis.KBotNotificationCenter = { mount, href };
})();
