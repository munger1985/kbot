(function () {
  "use strict";
  if (document.body.dataset.page === "diagnostic-source-detail") {
    const query = new URLSearchParams(location.search);if (query.has("id")) {query.set("sourceId",query.get("id"));query.delete("id");}location.replace("./diagnostic-sources.html?"+query);return;
  }
  const api = "/api/v1/apps/aiops/diagnostic-sources", shell = globalThis.KBotAIOpsShell, auth = globalThis.KBotAIOpsAuth;
  const esc = value => shell.escape(value ?? ""), rows = value => Array.isArray(value) ? value : value?.items || [];
  const drawer = document.getElementById("diagnostic-source-dialog"), form = document.getElementById("diagnostic-source-form"), webhookDrawer = document.getElementById("source-webhook-drawer");
  const state = {items:[],editing:null,loading:false,busy:false,search:"",sequence:0,opening:0,openingWebhook:"",poll:null,attempts:0,confirm:null,webhook:null,reveal:null,generating:false,webhookError:"",rotationKey:"",copied:{},copyTimers:[],manage:false};
  const request = (path = "",init) => auth.request(api + path,init), versionHeaders = item => ({"If-Match":`"rv-${item.row_version}"`});
  function showError(message = "") {const node = document.getElementById("source-error");node.textContent = message;node.hidden = !message;}
  function connectivity(item) {
    const status = item.connectivity_check_pending ? "CHECKING" : item.health_status ?? item.connectivity_status ?? "UNKNOWN";
    const labels = {HEALTHY:["已连通","success"],CONNECTED:["已连通","success"],DEGRADED:["受限","warning"],CHECKING:["未检测","muted"],UNREACHABLE:["不可达","danger"],MISCONFIGURED:["容器不匹配","danger"],DISCONNECTED:["未连接","danger"],UNHEALTHY:["异常","danger"],UNKNOWN:["未检测","muted"]};
    const [label,tone] = labels[status] || [status || "未检测","muted"];
    return {label,tone,usable:!item.connectivity_check_pending && ["HEALTHY","CONNECTED","DEGRADED"].includes(status)};
  }
  function button(label,action,source,extra = "",className = "row-action") {return `<button class="${className}" type="button" data-source-action="${action}" data-source-id="${esc(source.source_id)}"${state.busy || !state.manage ? " disabled" : ""}${extra}>${label}</button>`;}
  function renderCatalog() {
    const keyword = state.search.trim().toLocaleLowerCase();
    const items = state.items.filter(item => !keyword || [item.display_name,item.source_type,item.endpoint].some(value => String(value || "").toLocaleLowerCase().includes(keyword)));
    document.getElementById("source-catalog").innerHTML = `<button id="create-diagnostic-source" class="management-catalog-add-card" type="button"${state.loading || drawer.open || webhookDrawer.open || !state.manage ? " disabled" : ""}><span class="management-catalog-add-mark" aria-hidden="true">+</span><strong>新增监控源</strong><small>接入 Prometheus、Zabbix、OEM 等监控来源</small></button>` + items.map(item => {
      const active = ["ENABLED","ACTIVE"].includes(item.status), health = connectivity(item), enabled = item.status === "ENABLED";
      return `<article class="management-catalog-card monitor-source-card"><header><div><p class="monitor-source-card-updated">更新于 ${esc(shell.fmt(item.updated_at))}</p><h3 title="${esc(item.display_name)}">${esc(item.display_name)}</h3></div><div class="monitor-source-card-statuses"><span class="status-pill ${active ? "status-active" : "status-disabled"}">${active ? "已启用" : "已停用"}</span><span class="status-pill target-status-badge status-${health.tone}">${health.label}</span></div></header><dl><div><dt>类型</dt><dd>${esc(item.source_type || "—")}</dd></div><div><dt>服务地址</dt><dd title="${esc(item.endpoint)}">${esc(item.endpoint || "未配置")}</dd></div></dl><footer class="management-catalog-actions management-catalog-action-bar"><div class="management-catalog-primary-actions">${button("编辑","edit",item)}${button(item.connectivity_check_pending ? "检查中" : "测试连接","connectivity",item,item.connectivity_check_pending ? " disabled" : "")}${["ALERTMANAGER","ZABBIX"].includes(item.source_type) ? button(state.openingWebhook === item.source_id ? "打开中…" : "Webhook","webhook",item,` aria-label="Webhook 接入"${state.openingWebhook ? " disabled" : ""}`) : ""}</div><div class="management-catalog-overflow"><button class="row-action management-catalog-more" type="button" aria-haspopup="menu" aria-expanded="false"><span>更多</span><span aria-hidden="true">⌄</span></button><div class="management-catalog-menu" role="menu" hidden>${button(enabled ? "停用" : "启用",enabled ? "disable" : "enable",item,item.status === "DISABLED" && !health.usable ? ' disabled title="请先完成连接测试并确认通过"' : "",enabled ? "danger-row-action" : "row-action")}${button("删除","delete",item,item.status !== "DISABLED" ? ' disabled title="请先停用监控源"' : "","danger-row-action")}</div></div></footer></article>`;
    }).join("");
    const refresh = document.getElementById("refresh-sources");refresh.disabled = state.loading;refresh.textContent = state.loading ? "刷新中…" : "刷新目录";
  }
  function syncBusy() {
    form.querySelectorAll("input,select,textarea,button").forEach(node => {node.disabled = state.busy;});form.elements.source_type.disabled = state.busy || Boolean(state.editing);
    drawer.querySelector("[data-close-source]").disabled = state.busy;document.getElementById("save-diagnostic-source").textContent = state.busy ? "正在提交…" : state.editing ? "保存监控源" : "创建监控源";renderCatalog();
  }
  async function load() {
    const sequence = ++state.sequence;state.loading = true;showError();renderCatalog();
    try {
      const result = await request("?limit=100");if (sequence !== state.sequence) return;state.items = rows(result);clearTimeout(state.poll);state.poll = null;
      if (!state.items.some(item => item.connectivity_check_pending)) state.attempts = 0;
      else if (state.attempts < 6) {state.attempts++;state.poll = setTimeout(load,1500);}
    } catch (error) {if (sequence === state.sequence) showError(error.message);}
    finally {if (sequence === state.sequence) {state.loading = false;renderCatalog();}}
  }
  function renderType() {
    const type = form.elements.source_type.value, oem = type === "OEM";document.getElementById("source-token-field").hidden = oem;
    ["username","password","mappings"].forEach(key => {document.getElementById(`source-oem-${key}-field`).hidden = !oem;});document.getElementById("source-tenant-field").hidden = type !== "LOKI";
    document.getElementById("source-endpoint-help").textContent = oem ? "OEM 13.5 官方 API 地址，例如 https://oem.example.com/em/api；目标映射填写 32 位 Target GUID。" : "填写监控源的根地址，例如 https://monitor.example.com。";
  }
  function fillForm(item = null) {
    state.editing = item;form.reset();form.elements.display_name.value = item?.display_name || "";form.elements.source_type.value = item?.source_type || "PROMETHEUS";form.elements.endpoint.value = item?.endpoint || "";
    form.elements.loki_tenant_id.value = item?.config?.tenant_id || "";form.elements.oem_standard_mappings.value = JSON.stringify(item?.config?.standard_metric_mappings || {},null,2);
    document.getElementById("diagnostic-source-dialog-title").textContent = item ? "编辑监控源" : "新增监控源";renderType();syncBusy();if (!drawer.open) drawer.showModal();drawer.scrollTop = 0;renderCatalog();
  }
  function closeSource() {if (state.busy) return;++state.opening;if (drawer.open) drawer.close();state.editing = null;form.reset();showError();renderCatalog();}
  async function openEdit(sourceId) {
    if (state.busy || !state.manage) return;const sequence = ++state.opening;state.busy = true;syncBusy();showError();
    try {const item = await request(`/${encodeURIComponent(sourceId)}`);if (sequence === state.opening) fillForm(item);}
    catch (error) {if (sequence === state.opening) showError(error.message);}
    finally {if (sequence === state.opening) {state.busy = false;syncBusy();}}
  }
  function payload() {
    const value = name => form.elements[name].value.trim(), type = value("source_type");let credentials;
    if (type === "OEM") {const username = value("source_username"), password = form.elements.source_password.value;if (username || password) {if (!username || !password) throw new Error("OEM 用户名和密码需要同时填写；编辑时同时留空保持原凭据。");credentials = {username,password};}}
    else if (value("access_token")) credentials = {token:value("access_token")};
    let config = {};
    if (type === "OEM") {const mappings = JSON.parse(form.elements.oem_standard_mappings.value || "{}");if (!mappings || typeof mappings !== "object" || Array.isArray(mappings)) throw new Error("OEM标准指标映射必须为JSON对象");config = {standard_metric_mappings:mappings};}
    else if (type === "LOKI" && value("loki_tenant_id")) config = {tenant_id:value("loki_tenant_id")};
    return {display_name:value("display_name"),...(!state.editing ? {source_type:type} : {}),endpoint:value("endpoint") || undefined,...(credentials ? {credentials} : {}),config};
  }
  async function save(event) {
    event.preventDefault();if (state.busy || !state.manage) return;let body;try {body = payload();} catch (error) {showError(error.message);shell.toast(error.message);return;}
    state.busy = true;syncBusy();showError();
    try {const editing = state.editing;await request(editing ? `/${encodeURIComponent(editing.source_id)}` : "",{method:editing ? "PATCH" : "POST",headers:editing ? versionHeaders(editing) : {"Idempotency-Key":auth.uuid()},body:JSON.stringify(body)});state.busy = false;closeSource();await load();}
    catch (error) {showError(error.message);shell.toast(error.message);} finally {state.busy = false;syncBusy();}
  }
  function settleConfirmation(value) {const resolve = state.confirm;state.confirm = null;document.getElementById("source-confirm-dialog").close();resolve?.(value);}
  async function confirmAction(copy) {
    if (state.confirm) return false;const dialog = document.getElementById("source-confirm-dialog");dialog.className = `confirm-dialog confirm-dialog--${copy.tone}`;
    dialog.querySelector(".confirm-dialog-mark").textContent = copy.tone === "danger" ? "!" : "?";dialog.querySelector(".confirm-dialog-kicker").textContent = copy.tone === "danger" ? "请确认此操作" : "需要你的确认";
    document.getElementById("source-confirm-title").textContent = copy.title;document.getElementById("source-confirm-message").textContent = copy.message;const button = dialog.querySelector("[data-confirm-source]");button.textContent = copy.label;button.className = copy.tone === "danger" ? "danger-button" : "primary-button";
    dialog.showModal();dialog.querySelector("[data-cancel-source-confirm]").focus();return new Promise(resolve => {state.confirm = resolve;});
  }
  async function command(item,action) {
    if (state.busy || !state.manage) return;
    if (action !== "connectivity") {const copy = action === "delete" ? {title:"删除监控源？",message:"删除后将移除受管凭据。仍被 Agent、映射或监控信号历史引用的数据源不能删除。",label:"删除",tone:"danger"} : action === "enable" ? {title:"启用监控源？",message:"启用后 Agent 可以使用该数据源采集诊断证据。",label:"启用",tone:"warning"} : {title:"停用监控源？",message:"停用后 Agent 不会再使用该数据源。",label:"停用",tone:"danger"};if (!await confirmAction(copy)) return;}
    state.busy = true;syncBusy();showError();
    try {
      const result = await request(`/${encodeURIComponent(item.source_id)}${action === "delete" ? "" : action === "connectivity" ? "/connectivity-checks" : `/${action}`}`,{method:action === "delete" ? "DELETE" : "POST",headers:{...versionHeaders(item),"Idempotency-Key":auth.uuid()}});
      await load();if (action === "connectivity") shell.toast(`测试结果：${connectivity({...result,health_status:result.connectivity_status}).label}${result.last_error_code ? `（${result.last_error_code}）` : ""}`);else if (action === "delete") shell.toast("监控源已删除。");
    } catch (error) {showError(error.message);shell.toast(error.message);} finally {state.busy = false;syncBusy();}
  }
  function clearCopied() {state.copyTimers.forEach(clearTimeout);state.copyTimers = [];state.copied = {};}
  function closeWebhook() {if (state.generating) return;if (webhookDrawer.open) webhookDrawer.close();state.webhook = null;state.reveal = null;state.webhookError = "";state.rotationKey = "";clearCopied();document.getElementById("source-webhook-content").replaceChildren();renderCatalog();}
  async function openWebhook(item) {
    if (state.openingWebhook || state.busy || !state.manage) return;const sequence = ++state.opening;state.openingWebhook = item.source_id;renderCatalog();
    try {const detail = await request(`/${encodeURIComponent(item.source_id)}`);if (sequence !== state.opening) return;state.webhook = detail;state.reveal = null;state.webhookError = "";state.rotationKey = "";clearCopied();renderWebhook();webhookDrawer.showModal();webhookDrawer.scrollTop = 0;}
    catch (error) {shell.toast(error.message);} finally {state.openingWebhook = "";renderCatalog();}
  }
  function snippet() {const reveal = state.reveal;if (!reveal?.webhook_key) return "";const prefix = state.webhook.source_type === "ALERTMANAGER" ? "kbot_" : "";return `${prefix}webhook_key = ${reveal.webhook_key}\n${prefix}webhook_secret = ${reveal.webhook_secret}`;}
  function renderWebhook() {
    const source = state.webhook;if (!source) return;document.getElementById("webhook-source-name").textContent = `监控源 · ${source.display_name}`;webhookDrawer.querySelector("[data-close-webhook]").disabled = state.generating;
    const replacing = source.webhook_configured || source.webhook_secret?.configured, alertmanager = source.source_type === "ALERTMANAGER", content = document.getElementById("source-webhook-content");
    const heading = (index,title,description) => `<header><span>${index}</span><div><strong>${title}</strong><small>${description}</small></div></header>`;
    if (!state.reveal) {
      content.className = "webhook-credential-intro";
      content.innerHTML = `<section class="drawer-form-section">${heading("01","系统将生成两项独立凭据","无需自行编写随机值，也不要让两项使用相同内容。")}<dl class="webhook-credential-purpose"><div><dt>Webhook HMAC Secret</dt><dd>签名桥使用它计算 HMAC，KBot 使用同一个值验证告警正文。</dd></div><div><dt>Webhook Key</dt><dd>KBot 使用它在接收 URL 中定位当前监控源。</dd></div></dl></section><section class="drawer-form-section">${heading("02","生成后的配置位置","两项凭据只显示一次，生成后请立即复制。")}${alertmanager ? '<p class="field-help">把生成结果写入 <code>var/aiops-stack/aiops-stack.ini</code> 的 <code>[metrics]</code>：</p><dl class="webhook-config-map"><div><dt>HMAC Secret</dt><dd><code>kbot_webhook_secret</code></dd></div><div><dt>Webhook Key</dt><dd><code>kbot_webhook_key</code></dd></div></dl>' : '<p class="field-help">将 HMAC Secret 配置到 Webhook 发送端的签名凭据，将 Webhook Key 放入 KBot 事件接收 URL。</p>'}</section>${replacing ? '<div class="notice notice-warning"><strong>当前监控源已有 Webhook 凭据</strong><span>重新生成会替换 Secret 并轮换 Key；部署侧必须立即同步更新。</span></div>' : ""}<footer class="admin-drawer-footer"><button class="secondary-button" type="button" data-close-webhook${state.generating ? " disabled" : ""}>取消</button><button class="primary-button" type="button" data-generate-webhook${state.generating ? " disabled" : ""}>${state.generating ? "正在生成…" : replacing ? "重新生成两项凭据" : "生成两项凭据"}</button></footer>`;return;
    }
    const reveal = state.reveal, config = snippet(), copy = (key,label) => `<button class="secondary-button button-compact" type="button" data-copy-webhook="${key}">${state.copied[key] ? "已复制" : label}</button>`;
    content.className = "webhook-credential-result";
    content.innerHTML = `<div class="notice notice-warning"><strong>两项凭据只显示一次</strong><span>关闭抽屉后无法再次查看；遗失时必须重新生成并同步更新部署。</span></div>${state.webhookError ? `<p class="form-error">HMAC Secret 已更新，但 Webhook Key 生成失败：${esc(state.webhookError)}。请先复制 Secret，然后重试生成 Key。</p>` : ""}<div class="secret-reveal-panel"><strong>Webhook HMAC Secret</strong><div class="secret-reveal-value"><code>${esc(reveal.webhook_secret)}</code>${copy("secret","复制 Secret")}</div><span>配置到 ${alertmanager ? "kbot_webhook_secret" : "Webhook 发送端的 HMAC Secret"}。</span></div>${reveal.webhook_key ? `<div class="secret-reveal-panel"><strong>Webhook Key</strong><div class="secret-reveal-value"><code>${esc(reveal.webhook_key)}</code>${copy("key","复制 Key")}</div><span>${reveal.previous_key_expires_at ? `旧 Key 可使用到 ${esc(shell.fmt(reveal.previous_key_expires_at))}，请在此前完成部署切换。` : `配置到 ${alertmanager ? "kbot_webhook_key" : "KBot Webhook 接收 URL"}。`}</span></div>` : `<button class="secondary-button" type="button" data-retry-webhook${state.generating ? " disabled" : ""}>${state.generating ? "重试中…" : "重试生成 Webhook Key"}</button>`}${config ? `<section class="drawer-form-section">${heading("03","复制部署配置","以下内容包含敏感凭据，不要写入日志或提交到 Git。")}<pre class="webhook-config-snippet">${esc(config)}</pre><button class="secondary-button" type="button" data-copy-webhook="config">${state.copied.config ? "配置已复制" : "复制完整配置"}</button></section>` : ""}${alertmanager && reveal.webhook_key ? `<section class="drawer-form-section">${heading("04","下一步","按顺序完成部署与验证。")}<ol class="webhook-next-steps"><li>将上方两行写入 <code>var/aiops-stack/aiops-stack.ini</code> 的 <code>[metrics]</code>，并确认 <code>kbot_webhook_url</code> 是 Main API 根地址。</li><li>在部署主机执行无参数命令 <code>scripts/aiops-stack</code>，更新并重启 Webhook signer。</li><li>发送一条带 <code>target_key</code> 的测试告警，确认 KBot 能接收并关联到正确 Target。</li></ol></section>` : ""}<footer class="admin-drawer-footer">${reveal.created_at ? `<span>生成时间：${esc(shell.fmt(reveal.created_at))}</span>` : ""}<button class="primary-button" type="button" data-close-webhook${state.generating ? " disabled" : ""}>完成</button></footer>`;
  }
  async function rotateKey() {
    if (!state.rotationKey) state.rotationKey = auth.uuid();const result = await request(`/${encodeURIComponent(state.webhook.source_id)}/webhook-key:rotate`,{method:"POST",headers:{...versionHeaders(state.webhook),"Idempotency-Key":state.rotationKey}});
    state.reveal = {...state.reveal,...result};state.webhookError = "";await load();shell.toast("Webhook 接入凭据已生成，请立即复制并完成部署配置。");
  }
  async function generateWebhook(retry = false) {
    if (!state.webhook || state.generating || !state.manage || (retry && !state.reveal?.webhook_secret)) return;
    if (!retry && (state.webhook.webhook_configured || state.webhook.webhook_secret?.configured) && !await confirmAction({title:"重新生成 Webhook 接入凭据？",message:"此操作会同时替换 HMAC Secret 并轮换 Webhook Key。必须立即更新签名桥配置，否则新的告警将验签失败。",label:"重新生成",tone:"warning"})) return;
    state.generating = true;state.webhookError = "";renderWebhook();
    try {
      if (!retry) {
        if (!globalThis.crypto?.getRandomValues) throw new Error("当前浏览器不支持安全随机数生成，请更换现代浏览器。");
        const bytes = new Uint8Array(32);crypto.getRandomValues(bytes);const secret = Array.from(bytes,value => value.toString(16).padStart(2,"0")).join("");state.rotationKey = auth.uuid();
        state.webhook = await request(`/${encodeURIComponent(state.webhook.source_id)}`,{method:"PATCH",headers:versionHeaders(state.webhook),body:JSON.stringify({webhook_credentials:{value:secret}})});
        state.reveal = {webhook_secret:secret,webhook_key:""};clearCopied();renderWebhook();
      }
      try {await rotateKey();} catch (error) {state.webhookError = error.message;shell.toast(`HMAC Secret 已更新，但 Webhook Key 生成失败：${error.message}`);}
    } catch (error) {shell.toast(error.message);} finally {state.generating = false;renderWebhook();}
  }
  async function copyWebhook(key) {
    const value = key === "config" ? snippet() : state.reveal?.[key === "secret" ? "webhook_secret" : "webhook_key"];if (!value) return;
    try {
      if (navigator.clipboard?.writeText) await navigator.clipboard.writeText(value);
      else {const field = document.createElement("textarea");field.value = value;field.style.position = "fixed";field.style.opacity = "0";webhookDrawer.append(field);field.select();const copied = document.execCommand("copy");field.remove();if (!copied) throw new Error("自动复制失败，请手动复制。");}
      state.copied[key] = true;renderWebhook();state.copyTimers.push(setTimeout(() => {state.copied[key] = false;renderWebhook();},1800));
    } catch (error) {shell.toast(error.message);}
  }
  function closeMenus() {document.querySelectorAll(".management-catalog-menu").forEach(menu => {menu.hidden = true;});document.querySelectorAll(".management-catalog-more").forEach(button => button.setAttribute("aria-expanded","false"));}
  shell.ready.then(async access => {
    state.manage = rows(access.permissions).includes("aiops:diagnostic_source_manage");document.getElementById("source-search").oninput = event => {state.search = event.target.value;renderCatalog();};document.getElementById("refresh-sources").onclick = () => {state.attempts = 0;load();};form.onsubmit = save;form.elements.source_type.onchange = renderType;
    drawer.querySelectorAll("[data-close-source]").forEach(button => {button.onclick = closeSource;});drawer.addEventListener("cancel",event => {event.preventDefault();closeSource();});drawer.addEventListener("close",closeSource);
    document.getElementById("source-catalog").onclick = event => {
      if (event.target.closest("#create-diagnostic-source")) {if (!state.manage || state.busy) return;showError();fillForm();return;}
      const more = event.target.closest(".management-catalog-more");if (more) {const menu = more.nextElementSibling, wasOpen = !menu.hidden;closeMenus();menu.hidden = wasOpen;more.setAttribute("aria-expanded",String(!wasOpen));return;}
      const button = event.target.closest("[data-source-action]");if (!button || button.disabled) return;const item = state.items.find(row => row.source_id === button.dataset.sourceId);if (!item) return;closeMenus();if (button.dataset.sourceAction === "edit") openEdit(item.source_id);else if (button.dataset.sourceAction === "webhook") openWebhook(item);else command(item,button.dataset.sourceAction);
    };
    document.addEventListener("pointerdown",event => {if (!event.target.closest(".management-catalog-overflow")) closeMenus();});document.addEventListener("keydown",event => {if (event.key === "Escape") closeMenus();});
    webhookDrawer.onclick = event => {if (event.target.closest("[data-close-webhook]")) closeWebhook();else if (event.target.closest("[data-generate-webhook]")) generateWebhook();else if (event.target.closest("[data-retry-webhook]")) generateWebhook(true);else {const copy = event.target.closest("[data-copy-webhook]");if (copy) copyWebhook(copy.dataset.copyWebhook);}};webhookDrawer.addEventListener("cancel",event => {event.preventDefault();closeWebhook();});webhookDrawer.addEventListener("close",closeWebhook);
    const confirm = document.getElementById("source-confirm-dialog");confirm.querySelector("[data-cancel-source-confirm]").onclick = () => settleConfirmation(false);confirm.querySelector("[data-confirm-source]").onclick = () => settleConfirmation(true);confirm.addEventListener("cancel",event => {event.preventDefault();settleConfirmation(false);});
    window.addEventListener("pagehide",() => {clearTimeout(state.poll);++state.sequence;++state.opening;state.webhook = null;state.reveal = null;form.reset();document.getElementById("source-webhook-content").replaceChildren();clearCopied();});
    await load();const sourceId = new URLSearchParams(location.search).get("sourceId");if (sourceId) await openEdit(sourceId);
  });
  globalThis.KBotAIOpsConfigurations = {openEdit:(kind,id) => kind === "diagnostic-sources" ? openEdit(id) : undefined};
})();
