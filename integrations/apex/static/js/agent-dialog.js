(function () {
  "use strict";

  const state = { agent: null, models: [], resources: {} };
  const byId = (id) => document.getElementById(id);
  const value = (id) => String(byId(id)?.value || "").trim();
  const request = (path, options = {}) => KBotApi.request(path, {
    label: "Agent 请求",
    ...options
  });

  function items(payload) {
    if (Array.isArray(payload)) return payload;
    if (Array.isArray(payload?.items)) return payload.items;
    return [];
  }

  function showError(error) {
    KBotValidation.show(error, "保存 Agent 失败。");
  }

  function kind() {
    return document.querySelector('input[name="p76-agent-kind"]:checked')?.value || "knowledge";
  }

  function setKind(nextKind) {
    const input = document.querySelector(`input[name="p76-agent-kind"][value="${nextKind}"]`);
    if (input) input.checked = true;
    document.querySelectorAll('input[name="p76-agent-kind"]').forEach((item) => {
      item.disabled = Boolean(state.agent);
    });
    updateVisibility();
  }

  function updateVisibility() {
    const aiops = kind() === "aiops";
    byId("p76-knowledge-config").hidden = aiops;
    byId("p76-aiops-config").hidden = !aiops;
    byId("p76-knowledge-models").hidden = aiops;
    byId("p76-diagnosis-model").hidden = !aiops;
    const mode = value("p76-data-query-mode");
    byId("p76-data-profile-field").hidden = aiops || mode !== "MCP";
    byId("p76-semantic-help").hidden = aiops || mode !== "SEMANTIC";
    const creating = !state.agent;
    const knowledgeDraft = creating && !aiops;
    byId("p76-active-field").hidden = knowledgeDraft;
    byId("p76-active").disabled = knowledgeDraft;
    if (knowledgeDraft) byId("p76-active").checked = false;
    byId("p76-active-help").textContent = creating && aiops
      ? "未勾选时，AIOps Agent 将保存为草稿。"
      : "取消勾选后，已启用 Agent 将改为停用状态。";
  }

  function setOptions(id, rows, valueName, selected, optional) {
    const select = byId(id);
    select.replaceChildren(new Option(optional ? "- 不配置 -" : "- 请选择 -", ""));
    rows.forEach((row) => select.appendChild(new Option(
      `${row.display_name || row.name}（${row.status || "-"}）`,
      row[valueName]
    )));
    select.value = selected || "";
  }

  function setModelOptions(id, category, selected, optional) {
    const rows = state.models.filter((model) => {
      const status = String(model.status ?? "").toUpperCase();
      return Number(model.category) === category && ["ACTIVE", "1"].includes(status);
    });
    const select = byId(id);
    select.replaceChildren();
    if (optional) select.appendChild(new Option("- 不配置 -", ""));
    rows.forEach((model) => select.appendChild(new Option(
      `${model.display_name}（${model.served_model_name}）`,
      model.model_id
    )));
    select.value = selected || "";
  }

  function fillForm(agent) {
    byId("p76-display-name").value = agent?.display_name || "";
    byId("p76-description").value = agent?.description || "";
    byId("p76-instruction").value = agent?.instruction || "";
    byId("p76-active").checked = agent?.status === "ACTIVE";
    byId("p76-active-label").textContent = "是否启用？";
    byId("p76-resource-mode").value = agent?.config?.resource_mode || "managed_resources";
    byId("p76-data-query-mode").value = agent?.config?.data_query_mode || "";
    byId("p76-data-profile").value = agent?.config?.data_profile_name || "";
    byId("p76-rerank").checked = Boolean(agent?.do_rerank);
    setKind(agent?.agent_kind || value("P76_AGENT_KIND") || "knowledge");

    setModelOptions("p76-context-llm", 1, agent?.models?.context_llm, false);
    setModelOptions("p76-composer-llm", 1, agent?.models?.composer_llm, false);
    setModelOptions("p76-memory-llm", 1, agent?.models?.memory_llm, false);
    setModelOptions("p76-memory-embedding", 2, agent?.models?.memory_embedding, false);
    setModelOptions("p76-router-llm", 1, agent?.models?.router_llm, true);
    setModelOptions("p76-query-vlm", 5, agent?.models?.query_vlm, true);
    setModelOptions("p76-diagnosis-llm", 1, agent?.models?.diagnosis_llm, true);
    byId("p76-memory-embedding").disabled = Boolean(agent);

    setOptions("p76-monitor", state.resources.monitors, "source_id", agent?.monitor_source_id, false);
    setOptions("p76-policy", state.resources.policies, "policy_id", agent?.policy_id, false);
    setOptions("p76-target", state.resources.targets, "target_id", agent?.target_id, true);
    setOptions("p76-plan", state.resources.plans, "plan_id", agent?.inspection_plan_id, true);
    updateVisibility();
  }

  function formStatus() {
    if (byId("p76-active").checked) return "ACTIVE";
    if (!state.agent) return "DRAFT";
    return state.agent.status === "ACTIVE" ? "DISABLED" : state.agent.status;
  }

  function payload() {
    KBotValidation.clear(document.querySelector(".p76-form"));
    const agentKind = kind();
    const status = formStatus();
    const base = {
      display_name: value("p76-display-name"),
      description: value("p76-description") || null,
      instruction: value("p76-instruction") || null,
      status
    };
    if (!base.display_name) KBotValidation.fail("p76-display-name", "请填写 Agent 显示名称。");

    if (agentKind === "aiops") {
      const monitorSourceId = value("p76-monitor");
      const policyId = value("p76-policy");
      const targetId = value("p76-target") || null;
      const inspectionPlanId = value("p76-plan") || null;
      const diagnosisLlm = value("p76-diagnosis-llm");
      const missing = [
        !monitorSourceId && "p76-monitor",
        !policyId && "p76-policy",
        !diagnosisLlm && "p76-diagnosis-llm"
      ].filter(Boolean);
      if (missing.length) {
        KBotValidation.fail(missing, "AIOps Agent 必须配置监控源、Policy 和诊断 LLM。");
      }
      if (inspectionPlanId && !targetId) {
        KBotValidation.fail("p76-target", "配置巡检计划时必须同时选择 Target。");
      }
      return {
        ...base,
        monitor_source_id: monitorSourceId,
        policy_id: policyId,
        target_id: targetId,
        inspection_plan_id: inspectionPlanId,
        models: { diagnosis_llm: diagnosisLlm },
        image_capabilities: state.agent?.image_capabilities || {},
        config: state.agent?.config || {}
      };
    }

    const models = {
      context_llm: value("p76-context-llm"),
      composer_llm: value("p76-composer-llm"),
      memory_llm: value("p76-memory-llm"),
      memory_embedding: state.agent?.models?.memory_embedding || value("p76-memory-embedding")
    };
    const missingModels = Object.entries(models)
      .filter(([, modelId]) => !modelId)
      .map(([role]) => `p76-${role.replaceAll("_", "-")}`);
    if (missingModels.length) {
      KBotValidation.fail(missingModels, "请完整选择四个必选 Agent 模型。");
    }
    const router = value("p76-router-llm");
    const queryVlm = value("p76-query-vlm");
    if (router) models.router_llm = router;
    if (queryVlm) models.query_vlm = queryVlm;
    if (status === "ACTIVE" && !router) {
      KBotValidation.fail("p76-router-llm", "多能力 Agent 激活前必须配置 Router LLM。");
    }

    const dataQueryMode = value("p76-data-query-mode");
    const dataProfile = dataQueryMode === "MCP" ? value("p76-data-profile") : "";
    if (dataQueryMode && !["MCP", "SEMANTIC"].includes(dataQueryMode)) {
      KBotValidation.fail("p76-data-query-mode", "问数模式无效。");
    }
    if (dataQueryMode === "MCP" && !dataProfile) {
      KBotValidation.fail("p76-data-profile", "MCP 问数模式必须填写 Data Profile。");
    }
    if (!state.agent && status === "ACTIVE" && dataQueryMode === "SEMANTIC") {
      KBotValidation.fail(
        ["p76-data-query-mode", "p76-active"],
        "SEMANTIC Agent 请先保存为 DRAFT，完成语义模型与 Policy 绑定后再激活。"
      );
    }
    const config = { ...(state.agent?.config || {}) };
    config.resource_mode = value("p76-resource-mode");
    if (dataQueryMode) config.data_query_mode = dataQueryMode;
    else delete config.data_query_mode;
    if (dataProfile) config.data_profile_name = dataProfile;
    else delete config.data_profile_name;
    delete config.aiops_agent_id;
    delete config.aiops_target_id;
    return { ...base, models, do_rerank: byId("p76-rerank").checked, config };
  }

  async function save() {
    const spinner = apex.util.showSpinner(document.body);
    try {
      apex.message.clearErrors();
      const body = payload();
      const agentKind = kind();
      const basePath = agentKind === "aiops"
        ? "/api/v1/apps/aiops/agents"
        : "/api/v1/apps/knowledge-retrieval/agents";
      if (state.agent && agentKind === "knowledge" && body.status === "ACTIVE" && body.config.resource_mode === "managed_resources") {
        const bindings = await request(`/api/v1/apps/knowledge-retrieval/knowledge/agents/${encodeURIComponent(state.agent.agent_id)}/collection-bindings`);
        if (!(bindings?.bindings || []).some((binding) => binding.status === "ACTIVE")) {
          KBotValidation.fail(
            ["p76-resource-mode", "p76-active"],
            "托管资源模式的 Agent 必须先绑定至少一个有效 Collection 才能激活。"
          );
        }
      }
      const saved = state.agent
        ? await request(`${basePath}/${encodeURIComponent(state.agent.agent_id)}`, {
            method: "PATCH",
            body: JSON.stringify({ expected_row_version: state.agent.row_version, ...body })
          })
        : await request(basePath, { method: "POST", body: JSON.stringify(body) });
      apex.navigation.dialog.close(true, { agent_id: saved.agent_id });
    } catch (error) {
      showError(error);
    } finally {
      spinner.remove();
    }
  }

  async function init() {
    updateVisibility();
    try {
      const agentId = value("P76_AGENT_ID");
      const agentKind = value("P76_AGENT_KIND") || "knowledge";
      const results = await Promise.allSettled([
        request("/api/v1/model-catalog"),
        request("/api/v1/apps/aiops/monitor-sources?limit=200"),
        request("/api/v1/apps/aiops/policies?limit=200"),
        request("/api/v1/apps/aiops/targets?limit=200"),
        request("/api/v1/apps/aiops/inspection-plans?limit=200")
      ]);
      if (results[0].status !== "fulfilled") throw results[0].reason;
      state.models = items(results[0].value);
      state.resources = {
        monitors: results[1].status === "fulfilled" ? items(results[1].value) : [],
        policies: results[2].status === "fulfilled" ? items(results[2].value) : [],
        targets: results[3].status === "fulfilled" ? items(results[3].value) : [],
        plans: results[4].status === "fulfilled" ? items(results[4].value) : []
      };
      if (!state.models.length) throw new Error("模型目录中没有可用模型。");
      if (agentId) {
        const basePath = agentKind === "aiops"
          ? "/api/v1/apps/aiops/agents"
          : "/api/v1/apps/knowledge-retrieval/agents";
        state.agent = await request(`${basePath}/${encodeURIComponent(agentId)}`);
        state.agent.agent_kind = agentKind;
      }
      fillForm(state.agent);
    } catch (error) {
      showError(error);
    }
  }

  apex.jQuery(function () {
    byId("p76-save").addEventListener("click", save);
    byId("p76-cancel").addEventListener("click", () => apex.navigation.dialog.close(false));
    document.querySelectorAll('input[name="p76-agent-kind"]').forEach((input) => input.addEventListener("change", updateVisibility));
    byId("p76-data-query-mode").addEventListener("change", updateVisibility);
    init();
  });
})();
