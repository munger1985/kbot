(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    const page = document.body.dataset.page;
    if (!access || !["notification-channels", "notification-delivery", "notification-settings"].includes(page)) return;
    const byId = id => document.getElementById(id);
    const status = byId("delivery-status");
    const request = (path, body, method = "POST") => KBotAIOpsAuth.request("/api/v1/notification-delivery" + path,
      body === undefined ? {} : { method, body: JSON.stringify(body) });
    const labels = { PENDING:"等待投递", SENDING:"正在发送", ACCEPTED:"平台已接受", RETRY_WAIT:"等待重试",
      UNKNOWN:"结果待核实", BLOCKED:"已阻断", DEAD:"投递失败", CANCELLED:"已取消", EXPIRED:"已过期" };
    const label = value => labels[value] || value || "—";
    const date = value => value && !Number.isNaN(new Date(value).getTime()) ? new Date(value).toLocaleString("zh-CN", { hour12:false }) : "—";
    const node = (tag, text, className) => { const element = document.createElement(tag); if (text != null) element.textContent = text; if (className) element.className = className; return element; };
    const option = (value, text) => { const element = node("option", text); element.value = value; return element; };
    let runtime = null;
    let caps = [], events = {}, connection = null, bindings = [], busy = false, loading = false, loadVersion = 0;
    let offset = 0, pageSize = 20, hasMore = false, applied = new URLSearchParams(), selected = null, detailVersion = 0;
    const can = key => caps.includes(key);
    const available = () => runtime?.enabled === true;
    const eventLabel = key => events[key]?.[0] || ({"connection.test":"测试消息","identity.verification":"身份验证"}[key] || key || "—");
    const personal = () => bindings.filter(item => item.connection_id === connection?.id && item.target_type === "open_id" && item.status !== "DISABLED");
    const verified = () => personal().filter(item => item.status === "VERIFIED");
    function button(text, handler, primary = false) {
      const element = node("button", text, primary ? "primary-button" : "row-action");
      element.type = "button"; element.onclick = handler; return element;
    }
    function facts(container, pairs) {
      const dl = node("dl", null, "delivery-facts");
      pairs.forEach(([title, text]) => { const line = node("div"); line.append(node("dt", title), node("dd", text ?? "—")); dl.append(line); });
      container.append(dl);
    }
    function controls() {
      document.querySelectorAll(".admin-drawer button").forEach(item => { item.disabled = busy; });
      byId("delivery-refresh").disabled = busy || loading;
      if (page === "notification-channels") {
        byId("delivery-edit").disabled = busy || loading || !available();
        byId("delivery-check").disabled = busy || loading || !available();
        byId("delivery-test-open").disabled = busy || loading || !available() || !connection?.enabled;
        byId("delivery-rule-open").disabled = busy || loading || !available() || !connection;
        document.querySelector('[form="delivery-test"]').disabled = busy || !available() || !verified().length;
      }
      if (page === "notification-delivery") {
        byId("delivery-prev").disabled = busy || loading || offset === 0;
        byId("delivery-next").disabled = busy || loading || !hasMore;
        byId("delivery-page-size").disabled = busy || loading;
        byId("delivery-page-label").textContent = "第 " + (Math.floor(offset / pageSize) + 1) + " 页";
      }
    }
    function close(id) {
      if (busy) return;
      KBotDrawer.close(id);
      if (id === "delivery-editor") byId("delivery-connection").elements.app_secret.value = "";
      if (id === "delivery-detail") { selected = null; ++detailVersion; }
    }
    document.querySelectorAll("[data-close]").forEach(item => { item.onclick = () => close(item.dataset.close); });
    async function action(task, message = "操作已保存") {
      if (busy) return;
      if (!available()) { status.textContent = runtime?.message || "外部通知部署状态尚未就绪，请刷新重试。"; return; }
      busy = true; controls(); status.textContent = "处理中…";
      try { await task(); status.textContent = message; await refresh(); }
      catch (error) { status.textContent = error.message || "操作未完成"; }
      finally { busy = false; controls(); }
    }
    function renderConnection() {
      const summary = byId("delivery-connection-summary"); summary.replaceChildren();
      const state = !connection ? "未配置" : !connection.enabled ? "已停用" : connection.health === "TOKEN_VALID" ? "连接正常" : connection.health === "UNCHECKED" ? "待检查" : "需要检查";
      byId("delivery-connection-state").textContent = state;
      byId("delivery-connection-state").className = "status-pill " + (connection?.enabled && connection?.health === "TOKEN_VALID" ? "status-success" : "status-muted");
      if (connection) facts(summary, [["连接名称",connection.name],["App ID",connection.app_id],["最近检查",date(connection.checked_at)], ...(connection.last_error_code ? [["检查错误码",connection.last_error_code]] : [])]);
      else summary.append(node("p","尚未配置飞书，配置后可检查连接并发送测试消息。","delivery-empty"));
      byId("delivery-edit").textContent = connection ? "编辑配置" : "配置飞书";
      byId("delivery-check").hidden = !connection; byId("delivery-test-open").hidden = !connection;
    }
    function renderPolicies(items) {
      const container = byId("delivery-policies"); container.replaceChildren();
      if (!items.length) container.append(node("p","暂无消息规则。配置渠道后，选择消息类型和接收用户。","delivery-empty"));
      items.forEach(item => {
        const article = node("article",null,"delivery-rule"), description = node("div");
        description.append(node("strong",item.event_types.map(eventLabel).join("、")),node("small","仅向业务已授权的配置目标发送；既有其他绑定本轮不调整"));
        article.append(description,node("span",item.enabled ? "启用" : "暂停","status-pill"),button(item.enabled ? "暂停" : "启用", async () => {
          if (busy || !available()) return;
          if (item.enabled && !await KBotAIOpsShell.confirmAction({title:"暂停消息规则",message:"停止生成新的外部消息；站内通知、业务任务和既有投递任务不受影响。",confirmLabel:"暂停规则",tone:"warning"})) return;
          await action(() => request("/policies/" + item.id,{expected_row_version:item.row_version,enabled:!item.enabled},"PUT"));
        }));
        container.append(article);
      });
    }
    async function loadChannels(current) {
      if (!can("channel")) { status.textContent = "当前账号没有通知渠道管理权限；个人身份请到“我的通知设置”验证。"; return; }
      const [targets, policies] = await Promise.all([request("/targets"),request("/policies")]);
      if (current !== loadVersion) return;
      connection = targets.connections.find(item => item.provider === "FEISHU") || null; bindings = targets.bindings;
      byId("delivery-channels-content").hidden = false;
      renderConnection(); renderPolicies(policies.items);
      document.querySelectorAll("#delivery-policies button").forEach(item => { item.disabled = !available(); });
      if (!available()) status.textContent = (runtime?.message || "外部通知部署状态尚未就绪。") + " 部署管理员需配置 external_delivery.enabled、public_origin 与独立凭据密钥；此状态不是账号权限不足。";
    }
    async function openDetail(item) {
      const current = ++detailVersion;
      selected = item;
      const summary = byId("delivery-detail-summary"), attempts = byId("delivery-attempts");
      summary.replaceChildren(node("span",label(item.status),"status-pill"),node("h3",eventLabel(item.event_type)));
      facts(summary,[["接收对象",item.target_type === "chat_id" ? "受控群（既有记录）" : item.recipient_actor_id],["投递 ID",item.id],["来源事件",item.source_event_id],["飞书回执",item.message_id],["安全错误码",item.last_error_code],["有效期至",date(item.expires_at)]]);
      if (item.entry_id) { const link = node("a","打开原任务","secondary-button"); link.href = "./notification-open.html?entry=" + encodeURIComponent(item.entry_id); summary.append(link); }
      byId("delivery-retry").hidden = !can("retry") || !["UNKNOWN","BLOCKED","DEAD"].includes(item.status);
      attempts.replaceChildren(node("p","正在读取尝试历史…")); KBotDrawer.open("delivery-detail");
      try {
        const result = await request("/jobs/" + item.id + "/attempts");
        if (current !== detailVersion) return;
        attempts.replaceChildren();
        if (!result.items.length) attempts.append(node("p","尚无发送尝试。"));
        result.items.forEach(row => {
          const article = node("article");
          article.append(node("strong","第 " + row.attempt_number + " 次 · " + label(row.status)),node("span",date(row.started_at) + " · " + (row.elapsed_ms == null ? "耗时未记录" : row.elapsed_ms + " ms")));
          if (row.error_code) article.append(node("small","安全错误码：" + row.error_code));
          if (row.message_id) article.append(node("small","回执：" + row.message_id));
          attempts.append(article);
        });
      } catch (error) { if (current === detailVersion) attempts.replaceChildren(node("p","尝试历史读取失败，请关闭后重试。")); }
    }
    function renderJobs(items) {
      const container = byId("delivery-jobs"), table = node("table"), head = node("thead"), header = node("tr"), body = node("tbody");
      ["消息","接收对象","投递状态","创建时间","操作"].forEach(text => header.append(node("th",text))); head.append(header);
      items.forEach(item => {
        const row = node("tr");
        [eventLabel(item.event_type),item.target_type === "chat_id" ? "受控群（既有记录）" : item.recipient_actor_id || "—",label(item.status) + " · 尝试 " + item.attempt_count + " 次",date(item.created_at)].forEach(text => row.append(node("td",text)));
        const cell = node("td"); cell.append(button("查看详情",() => openDetail(item))); row.append(cell); body.append(row);
      });
      if (!items.length) { const row = node("tr"), cell = node("td","没有符合筛选条件的投递记录。","empty-cell"); cell.colSpan = 5; row.append(cell); body.append(row); }
      table.append(head,body); container.replaceChildren(table);
    }
    async function loadJobs(current) {
      if (!can("read")) { status.textContent = "当前账号没有通知投递查看权限。"; return; }
      const query = new URLSearchParams(applied); query.set("limit",String(pageSize)); query.set("offset",String(offset));
      const result = await request("/jobs?" + query);
      if (current !== loadVersion) return;
      hasMore = result.has_more; renderJobs(result.items);
      try {
        const health = await request("/health");
        if (current !== loadVersion) return;
        const metrics = byId("delivery-metrics"); metrics.replaceChildren();
        const count = states => health.items.filter(item => states.includes(item.status)).reduce((sum,item) => sum + Number(item.count),0);
        [["等待投递",count(["PENDING","RETRY_WAIT"]),"全量队列"],["需要关注",count(["UNKNOWN","BLOCKED","DEAD"]),"待核实、阻断或失败"],["平台接受率",health.metrics.acceptance_rate == null ? "—" : (health.metrics.acceptance_rate * 100).toFixed(1) + "%","近 24 小时已结束尝试"]].forEach(([title,value,help]) => {
          const card = node("article"); card.append(node("span",title),node("strong",String(value)),node("small",help)); metrics.append(card);
        });
      } catch (error) { byId("delivery-metrics").replaceChildren(node("p","投递运行概览暂不可用。")); }
    }
    async function loadMine(current) {
      const result = await request("/me/bindings");
      if (current !== loadVersion) return;
      const container = byId("delivery-mine"); container.replaceChildren();
      const mine = result.items.filter(item => item.status !== "DISABLED");
      if (!mine.length) container.append(node("p","尚未绑定飞书。站内通知正常，如需飞书通知请联系管理员维护身份。","delivery-empty"));
      mine.forEach(item => {
        const article = node("article"); article.append(node("p",item.external_id + " · " + (item.status === "VERIFIED" ? "已验证" : "待本人验证")));
        if (item.status === "UNVERIFIED") {
          const form = node("form",null,"delivery-form"), field = node("label","飞书验证码","control-field"), code = node("input");
          code.inputMode = "numeric"; code.pattern = "[0-9]{6}"; code.maxLength = 6; code.required = true; code.autocomplete = "one-time-code";
          field.append(code);
          const actions = node("div",null,"toolbar-actions"), send = button("发送验证码",() => action(() => request("/me/bindings/" + item.id + "/challenge",{}),"验证码已入队，请在飞书中查看")), confirm = node("button","确认是本人","primary-button");
          actions.append(send,confirm); form.append(field,actions);
          form.onsubmit = event => { event.preventDefault(); void action(() => request("/bindings/" + item.id + "/confirm",{code:code.value,expected_row_version:item.row_version}),"飞书身份已验证"); };
          article.append(form);
        }
        container.append(article);
      });
    }
    async function refresh() {
      const current = ++loadVersion; loading = true; controls();
      try {
        if (page === "notification-channels") await loadChannels(current);
        else if (page === "notification-delivery") await loadJobs(current);
        else await loadMine(current);
      } catch (error) {
        if (current !== loadVersion) return;
        status.textContent = error.message || "读取失败，请刷新重试。";
        if (page === "notification-channels") byId("delivery-channels-content").hidden = true;
        if (page === "notification-delivery") { hasMore = false; byId("delivery-jobs").replaceChildren(node("p","投递记录读取失败，请刷新重试。","delivery-empty")); }
      } finally { if (current === loadVersion) { loading = false; controls(); } }
    }
    function applyFilters() {
      const form = byId("delivery-filters"), data = new FormData(form);
      applied = new URLSearchParams();
      ["status","event_type","recipient_actor_id"].forEach(key => { const value = String(data.get(key) || "").trim(); if (value) applied.set(key,value); });
      const range = data.get("range");
      if (range !== "all") {
        const days = range === "24h" ? 1 : range === "7d" ? 7 : 30;
        applied.set("created_from",new Date(Date.now() - days * 86400000).toISOString()); applied.set("created_to",new Date().toISOString());
      }
      offset = 0; void refresh();
    }
    if (page === "notification-channels") {
      byId("delivery-edit").onclick = () => {
        const form = byId("delivery-connection"); form.reset();
        form.elements.name.value = connection?.name || "飞书"; form.elements.app_id.value = connection?.app_id || "";
        form.elements.app_id.disabled = Boolean(connection); form.elements.enabled.checked = Boolean(connection?.enabled); form.elements.app_secret.required = !connection;
        byId("delivery-editor-title").textContent = connection ? "编辑飞书配置" : "配置飞书";
        byId("delivery-secret-label").textContent = connection ? "替换 Secret（留空不变）" : "App Secret"; KBotDrawer.open("delivery-editor");
      };
      byId("delivery-connection").onsubmit = event => {
        event.preventDefault(); const form = event.currentTarget;
        void action(async () => {
          await request(connection ? "/connections/" + connection.id : "/connections", connection ? {name:form.elements.name.value.trim(),enabled:form.elements.enabled.checked,expected_row_version:connection.row_version,...(form.elements.app_secret.value ? {app_secret:form.elements.app_secret.value} : {})}
            : {name:form.elements.name.value.trim(),provider:"FEISHU",app_id:form.elements.app_id.value.trim(),app_secret:form.elements.app_secret.value,enabled:form.elements.enabled.checked},connection ? "PUT" : "POST");
          form.elements.app_secret.value = ""; KBotDrawer.close("delivery-editor");
        },"飞书配置已保存，未发送测试消息");
      };
      byId("delivery-check").onclick = () => action(async () => {
        const result = await request("/connections/" + connection.id + "/check",{});
        if (result.health !== "TOKEN_VALID") { await refresh(); throw new Error("连接检查未通过，请核对应用凭据与权限。"); }
      },"连接检查通过；不代表消息已送达");
      byId("delivery-test-open").onclick = () => {
        byId("delivery-test-binding").replaceChildren(option("","选择已验证的用户"),...verified().map(item => option(item.id,item.external_id)));
        byId("delivery-test-note").textContent = verified().length ? "发送后请核对投递记录和飞书客户端。" : "暂无已验证的个人目标，请先绑定并由本人完成验证。";
        KBotDrawer.open("delivery-tester");
      };
      byId("delivery-test").onsubmit = async event => {
        event.preventDefault(); const bindingId = byId("delivery-test-binding").value;
        if (!verified().some(item => item.id === bindingId) || !await KBotAIOpsShell.confirmAction({title:"发送飞书测试消息",message:"将向选定用户发送一条真实消息，不会执行业务任务。",confirmLabel:"确认发送",tone:"warning"})) return;
        void action(async () => { await request("/connections/" + connection.id + "/test",{binding_id:bindingId}); KBotDrawer.close("delivery-tester"); },"测试消息已入队，请到通知投递页查看结果");
      };
      byId("delivery-rule-open").onclick = () => {
        byId("delivery-policy").reset();
        const eventOptions = byId("delivery-event-options"), bindingOptions = byId("delivery-binding-options");
        eventOptions.querySelectorAll("label").forEach(item => item.remove()); bindingOptions.querySelectorAll("label,.delivery-empty").forEach(item => item.remove());
        const add = (parent,name,value,text) => { const field = node("label",null,"delivery-choice"), input = node("input"); input.type = "checkbox"; input.name = name; input.value = value; field.append(input,node("span",text)); parent.append(field); };
        Object.entries(events).forEach(([key,value]) => add(eventOptions,"event_types",key,value[0]));
        personal().forEach(item => add(bindingOptions,"binding_ids",item.id,item.external_id + " · " + (item.status === "VERIFIED" ? "已验证" : "待本人验证")));
        if (!personal().length) bindingOptions.append(node("p","暂无个人绑定，请先在用户管理中维护。","delivery-empty"));
        KBotDrawer.open("delivery-rule-editor");
      };
      byId("delivery-policy").onsubmit = event => {
        event.preventDefault(); const form = event.currentTarget, data = new FormData(form);
        if (!data.getAll("event_types").length || !data.getAll("binding_ids").length) { status.textContent = "请选择消息类型和接收用户。"; return; }
        void action(async () => { await request("/policies",{connection_id:connection.id,event_types:data.getAll("event_types"),binding_ids:data.getAll("binding_ids"),actor_ids:[],enabled:form.elements.enabled.checked}); KBotDrawer.close("delivery-rule-editor"); },"规则已保存，不会补发历史消息");
      };
    }
    if (page === "notification-delivery") {
      Object.entries(labels).forEach(([key,value]) => byId("delivery-filters").elements.status.append(option(key,value)));
      byId("delivery-filters").onsubmit = event => { event.preventDefault(); applyFilters(); };
      byId("delivery-filters").onreset = () => { pageSize = 20; byId("delivery-page-size").value = "20"; queueMicrotask(applyFilters); };
      byId("delivery-page-size").onchange = event => { pageSize = Number(event.target.value); offset = 0; void refresh(); };
      byId("delivery-prev").onclick = () => { if (!loading && !busy) { offset = Math.max(0,offset-pageSize); void refresh(); } };
      byId("delivery-next").onclick = () => { if (!loading && !busy && hasMore) { offset += pageSize; void refresh(); } };
      byId("delivery-retry").onclick = async () => {
        const item = selected;
        if (!item || busy) return;
        const result = await KBotAIOpsShell.confirmActionWithOption({title:"重试外部通知",message:"仅重试此投递目标，不会重新执行业务任务。结果待核实时可能已送达，超出去重窗口的新代次可能产生重复消息。",option:{label:"我明确接受新发送代次可能产生重复消息"},confirmLabel:"重试通知",tone:"warning"});
        if (!result.confirmed) return;
        void action(async () => { await request("/jobs/" + item.id + "/retry",{expected_row_version:item.row_version,acknowledge_duplicate_risk:result.optionSelected}); KBotDrawer.close("delivery-detail"); selected = null; ++detailVersion; },"重试已入队，不会重新执行业务任务");
      };
      byId("delivery-audit-panel").ontoggle = async event => {
        if (!event.currentTarget.open || !can("read")) return;
        const container = byId("delivery-audit"); container.replaceChildren(node("p","正在读取…"));
        try {
          const result = await request("/audit?limit=20"); container.replaceChildren();
          if (!result.items.length) container.append(node("p","暂无操作记录。"));
          result.items.forEach(item => { const article = node("article",null,"delivery-rule"); article.append(node("strong",item.action),node("span",item.created_by + " · " + date(item.created_at)),node("small",item.outcome)); container.append(article); });
        } catch (error) { container.replaceChildren(node("p","操作记录读取失败，请关闭后重新展开重试。")); }
      };
    }
    byId("delivery-refresh").onclick = () => { if (page === "notification-delivery" && can("read")) applyFilters(); else void refresh(); };
    try {
      const capabilities = await request("/capabilities");
      caps = capabilities.capabilities; runtime = capabilities.runtime;
      document.querySelectorAll("[data-capability]").forEach(item => { item.hidden = !can(item.dataset.capability); });
      if ((page === "notification-channels" && can("channel")) || (page === "notification-delivery" && can("read"))) events = (await request("/catalog")).events;
      if (page === "notification-delivery" && can("read")) {
        const select = byId("delivery-filters").elements.event_type;
        Object.entries(events).forEach(([key,value]) => select.append(option(key,value[0]))); select.append(option("connection.test","测试消息")); applyFilters();
      } else await refresh();
    } catch (error) { status.textContent = error.message || "读取权限失败，请刷新重试。"; }
  });
})();
