(function () {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const state = {
    agents: [],
    targets: [],
    monitors: [],
    collections: [],
    selectedAgentId: "",
    activeRunId: "",
    activeHitl: null,
    activeProposal: null,
    streamController: null,
  };

  const MODEL_CATEGORY = { LLM: 1, TXT_EMBEDDING: 2 };

  function generatedKey(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${KBotUI.uuid()
      .replaceAll("-", "")
      .slice(0, 6)}`;
  }

  function optionalValue(value) {
    const normalized = String(value || "").trim();
    return normalized || null;
  }

  function delay(milliseconds) {
    return new Promise((resolve) => window.setTimeout(resolve, milliseconds));
  }

  function option(value, label, selected) {
    return `<option value="${KBotUI.escapeHtml(value)}"${
      selected ? " selected" : ""
    }>${KBotUI.escapeHtml(label)}</option>`;
  }

  function selectValue(selector) {
    return $(selector).value;
  }

  function modelLabel(model) {
    return `${model.display_name || model.served_model_name}（${
      model.served_model_name
    }）`;
  }

  async function refreshModels() {
    KBotUI.setStatus($("#agent-create-status"), "正在读取模型目录…");
    try {
      const models = await KBotUI.api("/api/v1/model-catalog");
      const llms = models.filter(
        (item) => Number(item.category) === MODEL_CATEGORY.LLM
      );
      const embeddings = models.filter(
        (item) => Number(item.category) === MODEL_CATEGORY.TXT_EMBEDDING
      );
      for (const name of [
        "diagnosisLlm",
        "contextLlm",
        "composerLlm",
        "memoryLlm",
      ]) {
        const select = $("#agent-form").elements[name];
        const current = select.value;
        select.innerHTML = llms
          .map((item) =>
            option(item.model_id, modelLabel(item), item.model_id === current)
          )
          .join("");
      }
      const embedding = $("#agent-form").elements.memoryEmbedding;
      const currentEmbedding = embedding.value;
      embedding.innerHTML = embeddings
        .map((item) =>
          option(
            item.model_id,
            modelLabel(item),
            item.model_id === currentEmbedding
          )
        )
        .join("");
      KBotUI.setStatus(
        $("#agent-create-status"),
        `已加载 ${llms.length} 个 LLM、${embeddings.length} 个文本 Embedding`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#agent-create-status"), error.message, "error");
    }
  }

  function renderAgents() {
    $("#agent-rows").innerHTML = state.agents
      .map(
        (item) => `
          <tr data-id="${KBotUI.escapeHtml(item.agent_id)}" class="${
            item.agent_id === state.selectedAgentId ? "selected" : ""
          }">
            <td><strong>${KBotUI.escapeHtml(
              item.display_name
            )}</strong></td>
            <td><span class="badge">${KBotUI.escapeHtml(item.status)}</span></td>
            <td><span class="badge ${
              item.models?.diagnosis_llm ? "" : "warning"
            }">${item.models?.diagnosis_llm ? "已配置" : "缺失"}</span></td>
            <td><span class="tracking-id">${KBotUI.escapeHtml(
              item.agent_id
            )}</span></td>
          </tr>`
      )
      .join("");
    $("#agent-rows")
      .querySelectorAll("tr")
      .forEach((row) =>
        row.addEventListener("click", () => {
          state.selectedAgentId = row.dataset.id;
          populateSelectedAgentModels();
          renderAgents();
          refreshBindingOverview();
        })
      );
  }

  function populateSelectedAgentModels() {
    const agent = state.agents.find(
      (item) => item.agent_id === state.selectedAgentId
    );
    if (!agent) return;
    const roleFields = {
      diagnosis_llm: "diagnosisLlm",
      context_llm: "contextLlm",
      composer_llm: "composerLlm",
      memory_llm: "memoryLlm",
      memory_embedding: "memoryEmbedding",
    };
    for (const [role, fieldName] of Object.entries(roleFields)) {
      const modelId = agent.models?.[role];
      if (modelId) $("#agent-form").elements[fieldName].value = modelId;
    }
    KBotUI.setStatus(
      $("#agent-create-status"),
      agent.models?.diagnosis_llm
        ? `已载入 ${agent.display_name} 的模型配置`
        : `${agent.display_name} 尚未配置 diagnosis_llm，请选择后点击更新`,
      agent.models?.diagnosis_llm ? "ok" : ""
    );
  }

  async function refreshAgents(preferredId) {
    KBotUI.setStatus($("#agent-list-status"), "正在读取 AIOps Agents…");
    try {
      const rows = await KBotUI.api("/api/v1/agents");
      state.agents = rows.filter((item) =>
        (item.enabled_capabilities || []).includes("aiops")
      );
      if (
        preferredId ||
        !state.agents.some((item) => item.agent_id === state.selectedAgentId)
      ) {
        state.selectedAgentId =
          preferredId || state.agents[0]?.agent_id || "";
      }
      renderAgents();
      KBotUI.setStatus(
        $("#agent-list-status"),
        `已读取 ${state.agents.length} 个 AIOps Agent`,
        state.agents.length ? "ok" : ""
      );
    } catch (error) {
      KBotUI.setStatus($("#agent-list-status"), error.message, "error");
    }
  }

  async function refreshSources(preferredTargetId, preferredMonitorId) {
    KBotUI.setStatus($("#source-list-status"), "正在读取数据源…");
    try {
      const [targetPage, monitorPage] = await Promise.all([
        KBotUI.api("/api/v1/ops/targets?limit=200"),
        KBotUI.api("/api/v1/ops/monitor-sources?limit=200"),
      ]);
      state.targets = targetPage.items || [];
      state.monitors = monitorPage.items || [];
      const targetSelect = $("#target-select");
      const monitorSelect = $("#monitor-select");
      const oldTarget = preferredTargetId || targetSelect.value;
      const oldMonitor = preferredMonitorId || monitorSelect.value;
      targetSelect.innerHTML = state.targets
        .map((item) =>
          option(
            item.target_id,
            `${item.display_name} · ${item.db_type} · ${item.status}`,
            item.target_id === oldTarget
          )
        )
        .join("");
      monitorSelect.innerHTML = [
        option("", "不绑定监控源", !oldMonitor),
        ...state.monitors.map((item) =>
          option(
            item.source_id,
            `${item.display_name} · ${item.source_type} · ${item.status} / ${item.health_status}`,
            item.source_id === oldMonitor
          )
        ),
      ].join("");
      KBotUI.setStatus(
        $("#source-list-status"),
        `数据库目标 ${state.targets.length} 个，监控源 ${state.monitors.length} 个`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#source-list-status"), error.message, "error");
    }
  }

  async function refreshCollections() {
    KBotUI.setStatus($("#binding-status"), "正在读取 Collections…");
    try {
      const payload = await KBotUI.api("/api/v1/knowledge/collections");
      state.collections = Array.isArray(payload)
        ? payload
        : payload.items || [];
      $("#collection-choices").innerHTML = state.collections.length
        ? state.collections
            .map(
              (item) => `
                <label class="selection-card">
                  <input type="checkbox" value="${KBotUI.escapeHtml(
                    item.collection_id
                  )}" />
                  <span><strong>${KBotUI.escapeHtml(
                    item.display_name
                  )}</strong><small>${KBotUI.escapeHtml(
                    item.collection_id
                  )} · ${KBotUI.escapeHtml(item.status)}</small></span>
                </label>`
            )
            .join("")
        : '<p class="muted">当前 Domain 暂无 Collection。</p>';
      KBotUI.setStatus(
        $("#binding-status"),
        `已读取 ${state.collections.length} 个 Collection`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#binding-status"), error.message, "error");
    }
  }

  async function refreshAll() {
    await Promise.all([
      refreshModels(),
      refreshAgents(),
      refreshSources(),
      refreshCollections(),
    ]);
    await refreshBindingOverview();
    await refreshReports();
  }

  KBotUI.bindAuthForm($("#auth-form"), refreshAll);
  $("#refresh-models").addEventListener("click", refreshModels);
  $("#refresh-agents").addEventListener("click", () => refreshAgents());
  $("#refresh-sources").addEventListener("click", () => refreshSources());
  $("#refresh-collections").addEventListener("click", refreshCollections);

  async function healthCheckAndEnable(sourceId) {
    const status = $("#monitor-create-status");
    if (!sourceId) {
      throw new Error("请先选择监控源");
    }
    let source = await KBotUI.api(
      `/api/v1/ops/monitor-sources/${sourceId}`
    );
    if (source.status === "ACTIVE" && source.health_status === "HEALTHY") {
      KBotUI.setStatus(status, "监控源已经处于健康启用状态", "ok");
      return source;
    }
    KBotUI.setStatus(status, "正在请求监控源健康检查…");
    await KBotUI.api(
      `/api/v1/ops/monitor-sources/${sourceId}/health-checks`,
      {
        method: "POST",
        headers: {
          "If-Match": `"rv-${source.row_version}"`,
          "Idempotency-Key": KBotUI.idempotency("monitor-health"),
        },
      }
    );
    for (let attempt = 0; attempt < 20; attempt += 1) {
      await delay(1000);
      source = await KBotUI.api(
        `/api/v1/ops/monitor-sources/${sourceId}`
      );
      if (!source.health_check_pending) break;
      KBotUI.setStatus(
        status,
        `监控健康检查执行中… ${attempt + 1}/20`
      );
    }
    if (source.health_check_pending) {
      throw new Error("监控健康检查超时，请确认 AIOps Worker 正在运行");
    }
    if (source.health_status !== "HEALTHY") {
      throw new Error(
        `监控源健康检查失败：${source.last_error_code || source.health_status}`
      );
    }
    if (source.status !== "ACTIVE") {
      source = await KBotUI.api(
        `/api/v1/ops/monitor-sources/${sourceId}/enable`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${source.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("monitor-enable"),
          },
        }
      );
    }
    KBotUI.setStatus(status, "监控源健康检查通过并已启用", "ok");
    return source;
  }

  $("#enable-monitor").addEventListener("click", async () => {
    const sourceId = selectValue("#monitor-select");
    try {
      await healthCheckAndEnable(sourceId);
      await refreshSources(null, sourceId);
    } catch (error) {
      KBotUI.setStatus($("#monitor-create-status"), error.message, "error");
    }
  });

  $("#target-form").elements.dbType.addEventListener("change", (event) => {
    const form = $("#target-form");
    const mysql = event.target.value === "MYSQL";
    form.elements.port.value = mysql ? "3306" : "1521";
    form.elements.versionCode.value = mysql ? "8.0" : "26ai";
    form.elements.databaseName.value = mysql ? "mysql" : "FREEPDB1";
    form.elements.capabilities.value = mysql
      ? '{"information_schema":true,"sys_schema":true,"replication_views":true}'
      : '{"dynamic_performance_views":true,"dba_catalog_views":true,"replication_views":true}';
  });

  $("#agent-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const status = $("#agent-create-status");
    KBotUI.setStatus(status, "正在创建 AIOps Agent…");
    try {
      const payload = await KBotUI.api("/api/v1/agents", {
        method: "POST",
        body: JSON.stringify({
          display_name: form.elements.displayName.value.trim(),
          description: "面向数据库监控、根因诊断、方案与报告生成的独立 Agent",
          enabled_capabilities: ["aiops"],
          models: {
            diagnosis_llm: form.elements.diagnosisLlm.value,
            context_llm: form.elements.contextLlm.value,
            composer_llm: form.elements.composerLlm.value,
            memory_llm: form.elements.memoryLlm.value,
            memory_embedding: form.elements.memoryEmbedding.value,
          },
          do_rerank: false,
          data_profile_name: null,
          instruction: optionalValue(form.elements.instruction.value),
          config: {},
          status: "ACTIVE",
        }),
      });
      await refreshAgents(payload.agent_id);
      KBotUI.setStatus(status, `已创建 ${payload.display_name}`, "ok");
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  });

  $("#update-agent-models").addEventListener("click", async () => {
    const agent = state.agents.find(
      (item) => item.agent_id === state.selectedAgentId
    );
    const form = $("#agent-form");
    const status = $("#agent-create-status");
    if (!agent) {
      KBotUI.setStatus(status, "请先在右侧选择要修改的 Agent", "error");
      return;
    }
    KBotUI.setStatus(status, `正在更新 ${agent.display_name} 的模型配置…`);
    try {
      const payload = await KBotUI.api(
        `/api/v1/agents/${agent.agent_id}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_row_version: agent.row_version,
            models: {
              ...agent.models,
              diagnosis_llm: form.elements.diagnosisLlm.value,
              context_llm: form.elements.contextLlm.value,
              composer_llm: form.elements.composerLlm.value,
              memory_llm: form.elements.memoryLlm.value,
              memory_embedding: form.elements.memoryEmbedding.value,
            },
          }),
        }
      );
      await refreshAgents(payload.agent_id);
      populateSelectedAgentModels();
      KBotUI.setStatus(
        status,
        `${payload.display_name} 的 diagnosis_llm 已保存`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  });

  $("#target-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const status = $("#target-create-status");
    const dbType = form.elements.dbType.value;
    const name = form.elements.databaseName.value.trim();
    KBotUI.setStatus(status, "正在创建数据库目标…");
    try {
      const capabilities = JSON.parse(form.elements.capabilities.value || "{}");
      const host = form.elements.host.value.trim();
      const endpoint = host
        ? {
            host,
            port: Number(form.elements.port.value),
            service: dbType === "ORACLE" ? name : null,
            database: dbType === "MYSQL" ? name : null,
            tls_enabled: false,
          }
        : null;
      const payload = await KBotUI.api("/api/v1/ops/targets", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("aiops-target"),
        },
        body: JSON.stringify({
          display_name: form.elements.displayName.value.trim(),
          db_type: dbType,
          version_code: form.elements.versionCode.value.trim(),
          environment: form.elements.environment.value,
          db_role: "UNKNOWN",
          endpoint,
          diagnostic_secret_ref: optionalValue(
            form.elements.diagnosticSecretRef.value
          ),
          execution_secret_ref: optionalValue(
            form.elements.executionSecretRef.value
          ),
          security_level: 1,
          capabilities,
        }),
      });
      await refreshSources(payload.target_id);
      KBotUI.setStatus(status, `已创建 ${payload.display_name}`, "ok");
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  });

  $("#monitor-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const status = $("#monitor-create-status");
    KBotUI.setStatus(status, "正在创建监控源…");
    try {
      const payload = await KBotUI.api("/api/v1/ops/monitor-sources", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("aiops-monitor"),
        },
        body: JSON.stringify({
          display_name: form.elements.displayName.value.trim(),
          source_type: form.elements.sourceType.value,
          endpoint: form.elements.endpoint.value.trim(),
          secret_ref: optionalValue(form.elements.secretRef.value),
          webhook_secret_ref: null,
          tls_profile_ref: null,
          capabilities: {},
        }),
      });
      await refreshSources(null, payload.source_id);
      await healthCheckAndEnable(payload.source_id);
      await refreshSources(null, payload.source_id);
    } catch (error) {
      KBotUI.setStatus(status, error.message, "error");
    }
  });

  function selectedCollections() {
    return Array.from(
      $("#collection-choices").querySelectorAll('input[type="checkbox"]:checked')
    ).map((input) => ({
      collectionId: input.value,
      collectionKey: input.dataset.key,
    }));
  }

  async function refreshBindingOverview() {
    const agentId = state.selectedAgentId;
    const targetId = selectValue("#target-select");
    const container = $("#binding-overview");
    if (!agentId || !targetId) {
      container.innerHTML =
        '<p class="muted">选择 Agent 和数据库目标后显示现有绑定。</p>';
      return;
    }
    container.innerHTML = '<p class="muted">正在读取既有绑定…</p>';
    try {
      const [agentBindings, monitorBindings, collectionPayload] =
        await Promise.all([
          KBotUI.api(
            `/api/v1/ops/targets/${targetId}/agent-bindings`
          ),
          KBotUI.api(
            `/api/v1/ops/targets/${targetId}/monitor-bindings`
          ),
          KBotUI.api(
            `/api/v1/knowledge/agents/${agentId}/collection-bindings`
          ),
        ]);
      const selectedAgentBindings = agentBindings.filter(
        (item) => item.agent_id === agentId
      );
      const collectionBindings = collectionPayload.bindings || [];
      const collectionName = (collectionId) =>
        state.collections.find(
          (item) => item.collection_id === collectionId
        )?.display_name || collectionId;
      const monitorName = (sourceId) =>
        state.monitors.find((item) => item.source_id === sourceId)
          ?.display_name || sourceId;
      const cards = [
        {
          title: "Agent → Target",
          rows: selectedAgentBindings.map(
            (item) =>
              `${item.status} · ${
                item.allow_mutation ? "允许审批后变更" : "仅建议"
              } · Policy ${
                item.policy_id || "无"
              }`
          ),
        },
        {
          title: "Target → Monitor",
          rows: monitorBindings.map(
            (item) =>
              `${monitorName(item.source_id)} · ${item.status} · ${
                item.external_target_key
              }`
          ),
        },
        {
          title: "Agent → Collection",
          rows: collectionBindings.map(
            (item) =>
              `${collectionName(item.collection_id)} · ${item.status}`
          ),
          empty: "未绑定（允许仅依赖 LLM 与实时证据）",
        },
      ];
      container.innerHTML = cards
        .map(
          (card) => `
            <article class="selection-card">
              <span><strong>${KBotUI.escapeHtml(card.title)}</strong>
              <small>${(card.rows.length
                ? card.rows
                : [card.empty || "未绑定"]
              )
                .map((row) => KBotUI.escapeHtml(row))
                .join("<br>")}</small></span>
            </article>`
        )
        .join("");
    } catch (error) {
      container.innerHTML = `<p class="status error">${KBotUI.escapeHtml(
        error.message
      )}</p>`;
    }
  }

  $("#refresh-binding-overview").addEventListener(
    "click",
    refreshBindingOverview
  );

  $("#bind-resources").addEventListener("click", async () => {
    const status = $("#binding-status");
    const agentId = state.selectedAgentId;
    const targetId = selectValue("#target-select");
    const monitorId = selectValue("#monitor-select");
    const collections = selectedCollections();
    if (!agentId || !targetId) {
      KBotUI.setStatus(status, "请先选择 Agent 和数据库目标", "error");
      return;
    }
    const selectedMonitor = state.monitors.find(
      (item) => item.source_id === monitorId
    );
    if (monitorId && selectedMonitor?.status !== "ACTIVE") {
      KBotUI.setStatus(
        status,
        "所选监控源尚未启用，请先点击“检查并启用”",
        "error"
      );
      return;
    }
    KBotUI.setStatus(status, "正在建立知识、策略和数据源绑定…");
    try {
      const kcBindings = await Promise.all(
        collections.map((item) =>
          KBotUI.api(
            `/api/v1/knowledge/agents/${agentId}/collections/${encodeURIComponent(
              item.collectionKey
            )}/binding`,
            {
              method: "PUT",
              body: JSON.stringify({
                note: "AIOps Agent 诊断知识绑定",
              }),
            }
          )
        )
      );
      const policy = await KBotUI.api("/api/v1/ops/policies", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("aiops-policy"),
        },
        body: JSON.stringify({
          policy_key: generatedKey("aiops-policy"),
          display_name: "AIOps 测试诊断策略",
          rules: {
            schema_version: "ops.policy.v1",
            allow_agent_execution: false,
            max_risk_level: "LOW",
            allowed_action_types: ["db.session.terminate"],
            auto_observe_min_severity: "CRITICAL",
            alert_cooldown_seconds: 900,
            aiops_collection_ids: collections.map(
              (item) => item.collectionId
            ),
          },
        }),
      });
      const activePolicy = await KBotUI.api(
        `/api/v1/ops/policies/${policy.policy_id}/activate`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${policy.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("activate-policy"),
          },
        }
      );
      const agentBinding = await KBotUI.api(
        `/api/v1/ops/targets/${targetId}/agent-bindings`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("agent-target-binding"),
          },
          body: JSON.stringify({
            agent_id: agentId,
            allow_mutation: $("#allow-mutation").checked,
            policy_id: activePolicy.policy_id,
            allowed_actions: ["db.session.terminate"],
            change_window: null,
            max_daily_executions: null,
          }),
        }
      );
      let monitorBinding = null;
      if (monitorId) {
        const metricCodes = JSON.parse($("#metric-codes").value || "[]");
        const mappingOverrides = JSON.parse(
          $("#metric-mapping").value || "{}"
        );
        const existingMonitorBindings = await KBotUI.api(
          `/api/v1/ops/targets/${targetId}/monitor-bindings`
        );
        const existingMonitorBinding = existingMonitorBindings.find(
          (item) => item.source_id === monitorId
        );
        const monitorPayload = {
          external_target_key: $("#external-target-key").value.trim(),
          role: "PRIMARY",
          priority: 100,
          metric_scope: { metric_codes: metricCodes },
          mapping_overrides: mappingOverrides,
        };
        monitorBinding = existingMonitorBinding
          ? await KBotUI.api(
              `/api/v1/ops/targets/${targetId}/monitor-bindings/${existingMonitorBinding.binding_id}`,
              {
                method: "PATCH",
                headers: {
                  "If-Match": `"rv-${existingMonitorBinding.row_version}"`,
                },
                body: JSON.stringify(monitorPayload),
              }
            )
          : await KBotUI.api(
              `/api/v1/ops/targets/${targetId}/monitor-bindings`,
              {
                method: "POST",
                headers: {
                  "Idempotency-Key": KBotUI.idempotency(
                    "monitor-target-binding"
                  ),
                },
                body: JSON.stringify({
                  source_id: monitorId,
                  ...monitorPayload,
                }),
              }
            );
      }
      const currentTarget = await KBotUI.api(
        `/api/v1/ops/targets/${targetId}`
      );
      const activeTarget =
        currentTarget.status === "MAINTENANCE"
          ? await KBotUI.api(`/api/v1/ops/targets/${targetId}/activate`, {
              method: "POST",
              headers: {
                "If-Match": `"rv-${currentTarget.row_version}"`,
                "Idempotency-Key": KBotUI.idempotency("activate-target"),
              },
            })
          : currentTarget;
      $("#binding-output").textContent = KBotUI.json({
        kc_bindings: kcBindings,
        policy: activePolicy,
        target_agent_binding: agentBinding,
        monitor_binding: monitorBinding,
        target: activeTarget,
      });
      await refreshSources(targetId, monitorId);
      await refreshBindingOverview();
      KBotUI.setStatus(status, "全部绑定已完成，可开始诊断", "ok");
    } catch (error) {
      $("#binding-output").textContent = KBotUI.json(
        error.payload || { error: error.message }
      );
      KBotUI.setStatus(status, error.message, "error");
    }
  });

  function thoughtForTask(payload) {
    const key = String(payload.task_key || "");
    if (key === "scope") return "确定目标数据库、时间窗口、权限与诊断范围";
    if (key.startsWith("observe:"))
      return "查询 Prometheus / Zabbix / OEM 监控指标并规范化观测数据";
    if (key.startsWith("diagnostic:db."))
      return `连接数据库执行只读诊断：${key.replace("diagnostic:", "")}`;
    if (key === "diagnosis:knowledge")
      return "检索已绑定 Knowledge Core，查找诊断依据与处置知识";
    if (key.includes(":draft")) return "调用诊断 LLM 生成当前轮根因假设";
    if (key.includes(":validate")) return "校验 LLM 提出的补证计划与查询边界";
    if (key.includes(":collect")) return "按补证计划继续采集监控或数据库证据";
    if (key.includes(":assess")) return "评估本轮证据是否足以确认根因";
    if (key.includes("root-cause")) return "综合监控、数据库和知识证据确认根因";
    if (key.includes("verify")) return "验证根因结论与证据引用是否一致";
    if (key.includes("solution")) return "生成可执行的解决思路、命令建议与验证方法";
    if (key.includes("action-plan")) return "整理变更步骤、风险、前置条件和回滚方案";
    if (key.includes("proposal")) return "生成需要单次人工审批的操作提案";
    if (key.includes("report")) return "生成结构化故障或性能分析报告";
    const fallback = {
      SCOPE: "正在确定诊断范围",
      OBSERVE: "正在采集监控指标",
      DIAGNOSE: "正在分析诊断证据",
      PROPOSE: "正在生成处理方案",
      REPORT: "正在生成报告",
      VERIFY: "正在验证处置结果",
    };
    return fallback[payload.task_type] || "正在执行 AIOps 任务";
  }

  function addThought(title, body, kind, meta) {
    const stream = $("#thought-stream");
    if (stream.querySelector(".muted")) stream.innerHTML = "";
    const article = document.createElement("article");
    article.className = `ops-thought ${kind || ""}`;
    article.innerHTML = `
      <div class="ops-thought-marker"></div>
      <div>
        <div class="ops-thought-head"><strong>${KBotUI.escapeHtml(
          title
        )}</strong><span>${KBotUI.escapeHtml(meta || "")}</span></div>
        <div class="ops-thought-body">${KBotUI.escapeHtml(body)}</div>
      </div>`;
    stream.appendChild(article);
    stream.scrollTop = stream.scrollHeight;
  }

  function addRawEvent(event) {
    const row = document.createElement("pre");
    row.textContent = `${event.id || "-"} · ${event.type}\n${KBotUI.json(
      event.json
    )}`;
    $("#raw-events").appendChild(row);
  }

  function answerSection(title, values) {
    const items = (Array.isArray(values) ? values : [])
      .filter(Boolean)
      .map((item) => `<li>${KBotUI.escapeHtml(String(item))}</li>`)
      .join("");
    return items ? `<section><h4>${KBotUI.escapeHtml(title)}</h4><ul>${items}</ul></section>` : "";
  }

  function renderRunResult(result) {
    const payload = result.payload || {};
    const root = payload.root_cause || {};
    const solution = payload.solution || {};
    const hypothesisDetails = payload.hypothesis_details || [];
    const primaryHypothesis = hypothesisDetails.find(
      (item) => item.hypothesis_key === root.primary_hypothesis_key
    );
    const facts = (payload.facts || []).map(
      (item) => item.fact_summary || item.summary || JSON.stringify(item)
    );
    const grade = result.root_cause_grade || root.effective_level || "未定级";
    const diagnosisState =
      grade === "INCONCLUSIVE"
        ? {
            title: "诊断未形成结论",
            detail:
              "流程已正常执行完成，但现有证据不足以确认根因。这不是系统报错，需要补充下列证据后继续诊断。",
            badge: "流程完成 · 证据不足（未确诊）",
          }
        : grade === "POSSIBLE"
          ? {
              title: "已形成初步判断",
              detail: "当前只有可能性结论，仍需补充直接证据。",
              badge: "流程完成 · 初步判断",
            }
          : {
              title: "已形成诊断结论",
              detail: `根因等级：${grade}`,
              badge: `流程完成 · ${grade}`,
            };
    const rejectedRequests = (payload.rejected_evidence_requests || []).map(
      (item) =>
        `${item.request_key || "-"}：${item.tool_id || "-"}（${
          item.reason_code || "REJECTED"
        }）`
    );
    const rootSummary =
      root.conclusion ||
      root.root_cause ||
      root.rationale_summary ||
      root.summary ||
      primaryHypothesis?.statement ||
      (root.primary_hypothesis_key
        ? `${root.effective_level || "未定级"}：${root.primary_hypothesis_key}`
        : "");
    const content = [
      `<section><h4>${KBotUI.escapeHtml(
        diagnosisState.title
      )}</h4><p>${KBotUI.escapeHtml(diagnosisState.detail)}</p></section>`,
      rootSummary
        ? `<section><h4>根因结论</h4><p>${KBotUI.escapeHtml(rootSummary)}</p>${
            primaryHypothesis?.mechanism
              ? `<p class="muted">${KBotUI.escapeHtml(
                  primaryHypothesis.mechanism
                )}</p>`
              : ""
          }</section>`
        : "",
      payload.diagnosis_rationale
        ? `<section><h4>诊断推理摘要</h4><p>${KBotUI.escapeHtml(
            payload.diagnosis_rationale
          )}</p></section>`
        : "",
      answerSection("关键证据", facts),
      answerSection("立即缓解措施", solution.immediate_mitigations),
      answerSection("长期改进建议", solution.long_term_remediations),
      answerSection("验证方法", solution.verification_plan),
      answerSection("风险", solution.risks),
      answerSection("当前限制", [
        ...(solution.limitations || []),
        ...(payload.gaps || []),
      ]),
      answerSection("未执行的取证请求", rejectedRequests),
    ].join("");

    $("#answer-output").innerHTML =
      content ||
      `<p>${KBotUI.escapeHtml(
        payload.summary || "最终产物已生成，请展开下方原始 Artifact 查看完整内容。"
      )}</p>`;
    $("#result-output").textContent = KBotUI.json(result);
    $("#result-badge").textContent = diagnosisState.badge;
  }

  async function loadRunResult() {
    if (!state.activeRunId) return;
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/runs/${state.activeRunId}/result`
      );
      renderRunResult(result);
      addThought(
        "AI 最终诊断已加载",
        `最终产物：${result.final_artifact?.schema_version || "无"}`,
        "done",
        result.final_artifact?.artifact_id || ""
      );
    } catch (error) {
      $("#result-output").textContent = KBotUI.json(
        error.payload || { error: error.message }
      );
      $("#result-badge").textContent = "读取失败";
      KBotUI.setStatus(
        $("#run-status"),
        `Run 已结束，但最终输出读取失败：${error.message}`,
        "error"
      );
    }
  }

  function firstQueryId(request) {
    const candidates = [
      ...(request?.queries || []),
      ...(request?.sql_requests || []),
      ...(request?.requests || []),
    ];
    return (
      candidates.find((item) => item?.query_id)?.query_id ||
      request?.query_id ||
      "manual-query-1"
    );
  }

  async function showHitl(hitlId) {
    try {
      const payload = await KBotUI.api(`/api/v1/ops/hitl/${hitlId}`);
      state.activeHitl = payload;
      $("#hitl-panel").hidden = false;
      $("#hitl-request").textContent = KBotUI.json(payload.request || payload);
      $("#hitl-form").elements.queryId.value = firstQueryId(payload.request);
      $("#hitl-panel").scrollIntoView({ behavior: "smooth", block: "center" });
    } catch (error) {
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
    }
  }

  async function showProposal(proposalId, advisory) {
    try {
      const payload = await KBotUI.api(
        `/api/v1/ops/proposals/${proposalId}`
      );
      state.activeProposal = payload;
      $("#proposal-panel").hidden = false;
      $("#proposal-output").textContent = KBotUI.json(payload);
      $("#approve-proposal").disabled = Boolean(advisory);
      $("#reject-proposal").disabled = Boolean(advisory);
      $("#proposal-hint").textContent = advisory
        ? "当前资源不具备自动变更能力：命令仅供人工评估和执行。"
        : "页面展示真实命令预览、风险与回滚方案；本次操作必须单独批准并留痕。";
      KBotUI.setStatus(
        $("#proposal-status"),
        advisory ? "建议方案已生成，请人工处理" : "等待单次人工审批",
        advisory ? "ok" : ""
      );
      $("#proposal-panel").scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
  }

  function handleEvent(event) {
    addRawEvent(event);
    const record = event.json || {};
    const payload = record.payload || record;
    if (event.type === "task.status") {
      const description = thoughtForTask(payload);
      const status = payload.status || "UNKNOWN";
      addThought(
        description,
        status === "RUNNING"
          ? "任务已被 Worker 领取并开始执行"
          : `任务状态更新为 ${status}`,
        status === "FAILED" ? "failed" : status === "SUCCEEDED" ? "done" : "",
        payload.task_key || payload.task_type
      );
    } else if (event.type === "run.status") {
      $("#run-badge").textContent = payload.status;
      addThought("诊断阶段切换", payload.status, "phase", record.occurred_at);
    } else if (event.type === "diagnostic.input_required") {
      addThought(
        "需要人工补充数据",
        "数据库不可直连或现有证据不足，已生成只读 SQL 等待用户执行",
        "attention",
        payload.hitl_id
      );
      showHitl(payload.hitl_id);
    } else if (event.type === "proposal.pending_approval") {
      addThought(
        "方案等待审批",
        "系统已生成实际操作建议；批准前不会执行任何命令",
        "attention",
        payload.proposal_id
      );
      showProposal(payload.proposal_id, false);
    } else if (event.type === "proposal.advisory_ready") {
      addThought(
        "建议命令已生成",
        "当前资源不具备自动变更能力，系统只展示命令、风险和回滚方案",
        "done",
        payload.proposal_id
      );
      showProposal(payload.proposal_id, true);
    } else if (event.type === "execution.status") {
      addThought(
        "操作执行状态",
        payload.status,
        payload.status === "FAILED" ? "failed" : "phase",
        payload.execution_id
      );
    } else if (event.type === "report.ready") {
      addThought(
        "报告已生成",
        `${payload.report_type} · ${payload.summary || ""}`,
        "done",
        payload.report_id
      );
      refreshReports(payload.report_id);
    } else if (event.type.startsWith("run.")) {
      const terminal = ["run.completed", "run.failed", "run.cancelled", "run.expired"].includes(
        event.type
      );
      $("#run-badge").textContent = payload.status || event.type;
      if (terminal) {
        KBotUI.setStatus(
          $("#run-status"),
          `诊断流程结束：${payload.status || event.type}`,
          event.type === "run.completed" ? "ok" : "error"
        );
        refreshRunSummary();
        if (event.type === "run.completed") loadRunResult();
      }
    }
  }

  async function refreshRunSummary() {
    if (!state.activeRunId) return;
    try {
      const summary = await KBotUI.api(
        `/api/v1/ops/runs/${state.activeRunId}`
      );
      $("#run-output").textContent = KBotUI.json(summary);
    } catch (error) {
      KBotUI.setStatus($("#run-status"), error.message, "error");
    }
  }

  async function listenRun(eventsUrl, cursor) {
    state.streamController?.abort();
    state.streamController = new AbortController();
    try {
      await KBotUI.streamSse(
        eventsUrl,
        {
          lastEventId: cursor || 0,
          onEvent: handleEvent,
        },
        state.streamController.signal
      );
    } catch (error) {
      if (error.name !== "AbortError") {
        KBotUI.setStatus($("#run-status"), error.message, "error");
      }
    }
  }

  const legacyQuestionForm = $("#question-form");
  if (legacyQuestionForm) {
    legacyQuestionForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const targetId = selectValue("#target-select");
    if (!state.selectedAgentId || !targetId) {
      KBotUI.setStatus(
        $("#run-status"),
        "请先选择 AIOps Agent 和数据库目标",
        "error"
      );
      return;
    }
    $("#thought-stream").innerHTML =
      '<p class="muted">Run 已创建，等待 Worker 领取任务…</p>';
    $("#raw-events").innerHTML = "";
    $("#answer-output").innerHTML =
      '<p class="muted">正在等待诊断流程生成最终输出…</p>';
    $("#result-output").textContent = "{}";
    $("#result-badge").textContent = "生成中";
    $("#hitl-panel").hidden = true;
    $("#proposal-panel").hidden = true;
    KBotUI.setStatus($("#run-status"), "正在创建对话式诊断 Run…");
    try {
      const receipt = await KBotUI.api("/api/v1/ops/runs", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("aiops-run"),
        },
        body: JSON.stringify({
          agent_id: state.selectedAgentId,
          target_id: targetId,
          input: form.elements.input.value.trim(),
          session_id: optionalValue(form.elements.sessionId.value),
          observation_start: null,
          observation_end: null,
          client_metadata: { source: "dev_console" },
        }),
      });
      state.activeRunId = receipt.ops_run_id;
      $("#run-output").textContent = KBotUI.json(receipt);
      $("#run-badge").textContent = receipt.status;
      KBotUI.setStatus(
        $("#run-status"),
        `Run ${receipt.ops_run_id} 已创建，正在监听`,
        "ok"
      );
      await listenRun(receipt.events_url, receipt.event_cursor);
    } catch (error) {
      $("#run-output").textContent = KBotUI.json(
        error.payload || { error: error.message }
      );
      KBotUI.setStatus($("#run-status"), error.message, "error");
    }
    });

    $("#stop-stream").addEventListener("click", () => {
    state.streamController?.abort();
    KBotUI.setStatus($("#run-status"), "已停止监听；Run 不会被取消");
    });

    $("#refresh-run-result").addEventListener("click", async () => {
    if (!state.activeRunId) {
      KBotUI.setStatus($("#run-status"), "当前页面还没有活动 Run", "error");
      return;
    }
    await refreshRunSummary();
    await loadRunResult();
    });

    $("#hitl-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.activeHitl) return;
    const form = event.currentTarget;
    KBotUI.setStatus($("#hitl-status"), "正在提交人工诊断结果…");
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/hitl/${state.activeHitl.hitl_id}/response`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("hitl-response"),
          },
          body: JSON.stringify({
            expected_row_version: state.activeHitl.row_version,
            responses: [
              {
                query_id: form.elements.queryId.value.trim(),
                status: "SUCCEEDED",
                raw_output: form.elements.inlineData.value,
                error: null,
              },
            ],
            note: optionalValue(form.elements.note.value),
          }),
        }
      );
      $("#hitl-panel").hidden = true;
      addThought(
        "人工数据已接收",
        "诊断流程将使用用户粘贴的查询结果继续分析",
        "done",
        result.hitl_id
      );
      KBotUI.setStatus($("#hitl-status"), "提交成功", "ok");
    } catch (error) {
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
    }
    });

    $("#skip-hitl").addEventListener("click", async () => {
    if (!state.activeHitl) return;
    try {
      await KBotUI.api(
        `/api/v1/ops/hitl/${state.activeHitl.hitl_id}/skip`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${state.activeHitl.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("hitl-skip"),
          },
        }
      );
      $("#hitl-panel").hidden = true;
      addThought("已跳过人工补数", "流程将记录证据缺口并继续", "attention");
    } catch (error) {
      KBotUI.setStatus($("#hitl-status"), error.message, "error");
    }
    });

    $("#approve-proposal").addEventListener("click", async () => {
    const proposal = state.activeProposal;
    if (!proposal) return;
    KBotUI.setStatus($("#proposal-status"), "正在提交单次审批…");
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/proposals/${proposal.proposal_id}/approve`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("proposal-approve"),
          },
          body: JSON.stringify({
            expected_row_version: proposal.row_version,
            expected_proposal_hash: proposal.proposal_hash,
            note: optionalValue($("#proposal-note").value),
          }),
        }
      );
      $("#proposal-output").textContent = KBotUI.json(result);
      KBotUI.setStatus($("#proposal-status"), "已批准，等待执行状态", "ok");
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
    });

    $("#reject-proposal").addEventListener("click", async () => {
    const proposal = state.activeProposal;
    if (!proposal) return;
    const reason = $("#proposal-note").value.trim();
    if (!reason) {
      KBotUI.setStatus($("#proposal-status"), "驳回时必须填写原因", "error");
      return;
    }
    try {
      const result = await KBotUI.api(
        `/api/v1/ops/proposals/${proposal.proposal_id}/reject`,
        {
          method: "POST",
          headers: {
            "Idempotency-Key": KBotUI.idempotency("proposal-reject"),
          },
          body: JSON.stringify({
            expected_row_version: proposal.row_version,
            reason,
          }),
        }
      );
      $("#proposal-output").textContent = KBotUI.json(result);
      KBotUI.setStatus($("#proposal-status"), "已驳回并留痕", "ok");
    } catch (error) {
      KBotUI.setStatus($("#proposal-status"), error.message, "error");
    }
    });
  }

  async function refreshReports(preferredReportId) {
    const targetId = selectValue("#target-select");
    const reportType = selectValue("#report-type");
    const params = new URLSearchParams({ limit: "100" });
    if (targetId) params.set("target_id", targetId);
    if (reportType) params.set("report_type", reportType);
    try {
      const page = await KBotUI.api(
        `/api/v1/ops/reports?${params.toString()}`
      );
      const items = page.items || [];
      $("#report-rows").innerHTML = items
        .map(
          (item) => `
            <tr data-id="${KBotUI.escapeHtml(item.report_id)}" class="${
              item.report_id === preferredReportId ? "selected" : ""
            }">
              <td><span class="badge">${KBotUI.escapeHtml(
                item.report_type
              )}</span></td>
              <td>${KBotUI.escapeHtml(item.summary)}</td>
              <td>${KBotUI.escapeHtml(item.status)}</td>
              <td>${KBotUI.escapeHtml(
                new Date(item.period_start).toLocaleString()
              )}<br />→ ${KBotUI.escapeHtml(
                new Date(item.period_end).toLocaleString()
              )}</td>
            </tr>`
        )
        .join("");
      $("#report-rows")
        .querySelectorAll("tr")
        .forEach((row) =>
          row.addEventListener("click", () => loadReport(row.dataset.id))
        );
      KBotUI.setStatus(
        $("#report-status"),
        `已读取 ${items.length} 份报告`,
        "ok"
      );
      if (preferredReportId) await loadReport(preferredReportId);
    } catch (error) {
      KBotUI.setStatus($("#report-status"), error.message, "error");
    }
  }

  async function loadReport(reportId) {
    try {
      const report = await KBotUI.api(`/api/v1/ops/reports/${reportId}`);
      $("#report-output").textContent = KBotUI.json(report);
      $("#report-rows .selected")?.classList.remove("selected");
      $(`#report-rows tr[data-id="${reportId}"]`)?.classList.add("selected");
    } catch (error) {
      KBotUI.setStatus($("#report-status"), error.message, "error");
    }
  }

  $("#refresh-reports").addEventListener("click", () => refreshReports());
  $("#report-type").addEventListener("change", () => refreshReports());
  $("#target-select").addEventListener("change", () => {
    refreshReports();
    refreshBindingOverview();
  });

  $("#inspection-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = event.currentTarget;
    const targetId = selectValue("#target-select");
    if (!targetId) {
      KBotUI.setStatus($("#inspection-status"), "请先选择数据库目标", "error");
      return;
    }
    KBotUI.setStatus($("#inspection-status"), "正在创建并启用巡检计划…");
    try {
      const plan = await KBotUI.api("/api/v1/ops/inspection-plans", {
        method: "POST",
        headers: {
          "Idempotency-Key": KBotUI.idempotency("inspection-plan"),
        },
        body: JSON.stringify({
          display_name: form.elements.displayName.value.trim(),
          schedule_type: form.elements.scheduleType.value,
          cron_expression: form.elements.cronExpression.value.trim(),
          timezone: form.elements.timezone.value.trim(),
          template_id: "database_daily",
          template_version: "1.0.0",
          timeout_seconds: 1800,
          overlap_policy: "SKIP",
          misfire_policy: "LATEST_ONLY",
          schedule_resolver_version: "1.0.0",
        }),
      });
      await KBotUI.api(
        `/api/v1/ops/inspection-plans/${plan.plan_id}/targets`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${plan.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("inspection-target"),
          },
          body: JSON.stringify({
            target_id: targetId,
            template_overrides: null,
          }),
        }
      );
      const current = await KBotUI.api(
        `/api/v1/ops/inspection-plans/${plan.plan_id}`
      );
      const active = await KBotUI.api(
        `/api/v1/ops/inspection-plans/${plan.plan_id}/activate`,
        {
          method: "POST",
          headers: {
            "If-Match": `"rv-${current.row_version}"`,
            "Idempotency-Key": KBotUI.idempotency("inspection-activate"),
          },
        }
      );
      $("#report-output").textContent = KBotUI.json(active);
      KBotUI.setStatus(
        $("#inspection-status"),
        `巡检计划已启用，下次执行：${active.next_run_at || "等待 Scheduler 计算"}`,
        "ok"
      );
    } catch (error) {
      KBotUI.setStatus($("#inspection-status"), error.message, "error");
    }
  });

  refreshAll();
})();
