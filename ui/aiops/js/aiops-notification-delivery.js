(function () {
  "use strict";
  KBotAIOpsShell.ready.then(async access => {
    if (!access || document.body.dataset.page !== "notification-delivery") return;
    const status = document.getElementById("delivery-status");
    const domainId = String(access.domain_id);
    let capabilities = [], connections = [], bindings = [], busy = false;
    let jobStatus = "", jobOffset = 0, jobCount = 0;
    let loadVersion = 0;
    const can = key => capabilities.includes(key);
    const request = (path, body, method = "POST") => KBotAIOpsAuth.request("/api/v1/notification-delivery" + path + (path.includes("?") ? "&" : "?") + "domain_id=" + encodeURIComponent(domainId),
      body === undefined ? {} : {method, body: JSON.stringify(body)});
    const option = (value, label) => { const item = document.createElement("option"); item.value = value; item.textContent = label; return item; };
    function table(target, rows, columns, actions = () => []) {
      const container = document.getElementById(target), element = document.createElement("table");
      const header = document.createElement("tr");
      for (const [, label] of columns) { const cell = document.createElement("th"); cell.textContent = label; header.append(cell); }
      const head = document.createElement("thead"); head.append(header); element.append(head);
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const line = document.createElement("tr");
        for (const [key] of columns) { const cell = document.createElement("td"); cell.textContent = String(row[key] ?? "—"); line.append(cell); }
        const controls = document.createElement("td");
        actions(row).forEach(([label, handler]) => {
          const button = document.createElement("button"); button.type = "button"; button.className = "row-action"; button.textContent = label;
          button.onclick = handler; controls.append(button);
        }); line.append(controls); body.append(line);
      });
      element.append(body); container.replaceChildren(element);
      if (!rows.length) container.append(document.createTextNode("暂无记录"));
    }
    async function action(work, message = "操作已保存") {
      if (busy) return;
      busy = true; status.textContent = "处理中…";
      try { await work(); status.textContent = message; await refresh(); }
      catch (error) { status.textContent = error.message || "操作未完成"; }
      finally { busy = false; }
    }
    async function refresh() {
      const version = ++loadVersion;
      try {
        const list = async kind => (await request("/" + kind)).items;
        const targets = can("channel") || can("binding") ? await request("/targets") : {connections: [], bindings: []};
        const results = await Promise.all([targets.connections, targets.bindings,
          can("channel") ? list("policies") : [], can("read") ? list("jobs?offset=" + jobOffset + (jobStatus ? "&status=" + jobStatus : "")) : [], can("read") ? list("audit") : [], list("me/bindings").catch(() => [])]);
        const [newConnections, newBindings, policies, jobs, audit, mine] = results;
        if (version !== loadVersion) return;
        jobCount = jobs.length;
        connections = newConnections; bindings = newBindings;
        table("delivery-connections", connections, [["name","连接"],["app_id","App ID"],["enabled","启用"],["health","健康"],["last_error_code","安全错误码"]], row => [
          ["编辑／换密钥", () => { const form = document.getElementById("delivery-connection"); for (const key of ["id","name","app_id"]) form.elements[key].value = row[key]; form.elements.app_secret.value = ""; form.elements.enabled.checked = row.enabled; }],
          ["检查", () => action(() => request("/connections/" + row.id + "/check", {}), "令牌检查已完成，不代表消息已投递")],
          ["真实试发", () => {
            const bindingId = document.getElementById("delivery-test-binding").value;
            if (!bindingId) { status.textContent = "请先选择已验证试发目标"; return; }
            if (confirm("将向选定身份发送真实飞书测试消息，确认发送？")) void action(() => request("/connections/" + row.id + "/test", {binding_id: bindingId}), "试发已入队，请核对投递记录及飞书客户端");
          }],
        ]);
        document.querySelectorAll('select[name="connection_id"]').forEach(select => {
          const value = select.value; select.replaceChildren(option("", "选择连接"), ...connections.map(row => option(row.id, row.name))); select.value = value;
        });
        const testSelect = document.getElementById("delivery-test-binding"), previous = testSelect.value;
        testSelect.replaceChildren(option("", "选择已验证身份"), ...bindings.filter(row => row.status === "VERIFIED").map(row => option(row.id, (row.actor_id || "受控群") + " / " + row.external_id))); testSelect.value = previous;
        const bindingSelect = document.querySelector('select[name="binding_ids"]');
        const selected = new Set(Array.from(bindingSelect.selectedOptions, item => item.value));
        bindingSelect.replaceChildren(...bindings.filter(row => row.status !== "DISABLED").map(row => { const item = option(row.id, (row.actor_id || "群") + " / " + row.external_id); item.selected = selected.has(row.id); return item; }));
        table("delivery-bindings", bindings, [["actor_id","本地用户"],["external_id","飞书身份"],["status","验证状态"]], row => !can("binding") || row.status === "DISABLED" ? [] : [
          ...(row.target_type === "open_id" ? [["发送验证码", () => action(() => request("/bindings/" + row.id + "/challenge", {}), "验证消息已入队，请本人登录确认")]] : []),
          ["停用／解绑", () => { if (confirm("停用此身份绑定？待发送任务将被阻断。")) void action(() => request("/bindings/" + row.id + "/disable", {expected_row_version: row.row_version})); }],
        ]);
        const mineContainer = document.getElementById("delivery-mine"); mineContainer.replaceChildren();
        mine.forEach(row => {
          const form = document.createElement("form"), label = document.createElement("label"), code = document.createElement("input"), button = document.createElement("button");
          label.textContent = row.external_id + " · " + row.status; form.append(label);
          if (row.status === "UNVERIFIED") {
            code.inputMode = "numeric"; code.pattern = "[0-9]{6}"; code.maxLength = 6; code.required = true; code.autocomplete = "one-time-code";
            button.textContent = "确认是本人"; form.append(code, button);
            form.onsubmit = event => { event.preventDefault(); void action(() => request("/bindings/" + row.id + "/confirm", {expected_row_version: row.row_version, code: code.value})); };
          } mineContainer.append(form);
        });
        table("delivery-policies", policies, [["event_types","受控事件"],["enabled","启用"]], row => [
          [row.enabled ? "暂停" : "启用", () => action(() => request("/policies/" + row.id, {expected_row_version: row.row_version, enabled: !row.enabled}, "PUT"))],
        ]);
        table("delivery-jobs", jobs, [["event_type","事件"],["recipient_actor_id","本地目标"],["status","投递状态"],["attempt_count","尝试"],["message_id","接受回执"],["last_error_code","原因"]], row => [
          ["尝试历史", async () => { try { document.getElementById("delivery-attempts").textContent = JSON.stringify((await request("/jobs/" + row.id + "/attempts")).items, null, 2); } catch (error) { status.textContent = error.message; } }],
          ...(can("retry") && ["UNKNOWN","BLOCKED","DEAD"].includes(row.status) ? [["受控重试", () => {
            if (!confirm("仅重试该冻结目标；可能已被飞书接受，不会重新执行业务。继续？")) return;
            const acknowledge = confirm("若超出去重窗口，新代次可能产生重复消息。是否明确接受此风险？选择取消仍可尝试窗口内原 uuid 重试。");
            void action(() => request("/jobs/" + row.id + "/retry", {expected_row_version: row.row_version, acknowledge_duplicate_risk: acknowledge}));
          }]] : []),
        ]);
        table("delivery-audit", audit, [["action","操作"],["created_by","真实操作人"],["outcome","结果"],["created_at","时间"]]);
        if (can("read")) {
          let health = document.getElementById("delivery-health");
          if (!health) {
            const label = document.createElement("h2"); label.textContent = "队列健康与近24小时指标";
            health = document.createElement("pre"); health.id = "delivery-health";
            document.getElementById("delivery-audit").after(label, health);
          }
          health.textContent = JSON.stringify(await request("/health"), null, 2);
        }
      } catch (error) { status.textContent = error.message; }
    }
    document.getElementById("delivery-connection").onsubmit = event => {
      event.preventDefault(); const form = event.currentTarget, values = Object.fromEntries(new FormData(form));
      void action(async () => {
        const old = connections.find(row => row.id === values.id);
        await request(old ? "/connections/" + old.id : "/connections", old ? {name: values.name, enabled: form.elements.enabled.checked,
          expected_row_version: old.row_version, ...(values.app_secret ? {app_secret: values.app_secret} : {})} :
          {name: values.name, app_id: values.app_id, app_secret: values.app_secret, enabled: form.elements.enabled.checked}, old ? "PUT" : "POST");
        form.reset(); form.elements.id.value = "";
      });
    };
    document.getElementById("delivery-binding").onsubmit = event => {
      event.preventDefault(); const form = event.currentTarget, values = Object.fromEntries(new FormData(form));
      void action(() => request("/bindings", {...values, actor_id: values.target_type === "open_id" ? values.actor_id : null}));
    };
    document.getElementById("delivery-policy").onsubmit = event => {
      event.preventDefault(); const form = event.currentTarget, data = new FormData(form);
      void action(() => request("/policies", {connection_id: data.get("connection_id"), event_types: data.getAll("event_types"),
        binding_ids: data.getAll("binding_ids"), actor_ids: String(data.get("actor_ids") || "").split(",").map(value => value.trim()).filter(Boolean),
        enabled: form.elements.enabled.checked}));
    };
    document.getElementById("delivery-refresh").onclick = refresh;
    try {
      capabilities = (await request("/capabilities")).capabilities;
      const events = (await request("/catalog")).events;
      document.querySelectorAll("[data-capability]").forEach(element => { element.hidden = !can(element.dataset.capability); });
      if (can("read")) {
        const toolbar = document.createElement("div"), filter = document.createElement("select");
        const label = document.createElement("label"); label.textContent = "投递状态筛选";
        filter.append(option("", "全部状态"), ...["PENDING","SENDING","ACCEPTED","RETRY_WAIT","UNKNOWN","BLOCKED","DEAD","CANCELLED","EXPIRED"].map(value => option(value, value)));
        filter.onchange = () => { jobStatus = filter.value; jobOffset = 0; void refresh(); };
        label.append(filter); toolbar.append(label);
        [["上一页", -100], ["下一页", 100]].forEach(([title, delta]) => {
          const button = document.createElement("button"); button.textContent = title; button.type = "button";
          button.onclick = () => { if (delta > 0 && jobCount < 100) return; jobOffset = Math.max(0, jobOffset + delta); void refresh(); };
          toolbar.append(button);
        });
        document.getElementById("delivery-jobs").before(toolbar);
      }
      document.querySelector('select[name="event_types"]').replaceChildren(...Object.entries(events).map(([key, value]) => option(key, value[0])));
      await refresh();
    } catch (error) { status.textContent = error.message; }
  });
})();
