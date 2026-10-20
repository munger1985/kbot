(function () {
  "use strict";
  const api = "/api/v1/apps/aiops/operations-knowledge/knowledge-bases";
  const assistantApi = "/api/v1/apps/aiops/operations-knowledge/knowledge-answering/binding";
  const shell = globalThis.KBotAIOpsShell;
  let rows = [];
  let editing = null;
  let assistantBinding = null;

  function render() {
    const body = document.getElementById("knowledge-base-rows");
    body.innerHTML = rows.length ? rows.map((item) => `<tr>
      <td><strong>${shell.escape(item.display_name)}</strong><small>${shell.escape(item.description || "未填写说明")}</small></td>
      <td>${shell.badge(item.status)}</td><td>${Number(item.usage?.assets || 0)}</td><td>诊断 ${Number(item.usage?.agent_versions || 0)} · 问答 ${Number(item.usage?.knowledge_assistants || 0)}</td><td>${shell.escape(shell.fmt(item.updated_at))}</td>
      <td><div class="ops-actions"><a href="./operations-knowledge.html?knowledge_base_id=${encodeURIComponent(item.knowledge_base_id)}">管理资料</a><button type="button" data-edit="${shell.escape(item.knowledge_base_id)}">编辑</button><button class="danger" type="button" data-delete="${shell.escape(item.knowledge_base_id)}">删除</button></div></td>
    </tr>`).join("") : '<tr><td colspan="6" class="ops-empty">当前 Domain 还没有运维知识库。</td></tr>';
  }
  async function load() { const page = await KBotAIOpsAuth.request(api); rows = page.items || []; render(); }
  async function loadAssistantBinding() {
    assistantBinding = await KBotAIOpsAuth.request(assistantApi);
  }
  function openAssistantBinding() {
    const selected = new Set(assistantBinding?.knowledge_base_ids || []);
    document.getElementById("knowledge-assistant-bases").innerHTML = rows.length
      ? rows.map((item) => `<label class="agent-switch-row"><input type="checkbox" name="knowledge_base_ids" value="${shell.escape(item.knowledge_base_id)}" ${selected.has(item.knowledge_base_id) ? "checked" : ""}><span><strong>${shell.escape(item.display_name)}</strong><small>${shell.escape(item.description || "未填写说明")}</small></span></label>`).join("")
      : '<p class="ops-empty">请先创建运维知识库。</p>';
    document.getElementById("knowledge-assistant-result").textContent = "";
    document.getElementById("knowledge-assistant-dialog").showModal();
  }
  async function saveAssistantBinding(event) {
    event.preventDefault();
    const knowledgeBaseIds = [...event.currentTarget.querySelectorAll('[name="knowledge_base_ids"]:checked')].map((item) => item.value);
    const result = document.getElementById("knowledge-assistant-result");
    if (!knowledgeBaseIds.length) { result.textContent = "请至少选择一个运维知识库"; return; }
    const button = document.getElementById("save-knowledge-assistant"); button.disabled = true;
    const payload = { knowledge_base_ids: knowledgeBaseIds };
    if (assistantBinding?.configured) payload.expected_row_version = assistantBinding.row_version;
    try {
      assistantBinding = await KBotAIOpsAuth.request(assistantApi, { method: "PUT", body: JSON.stringify(payload) });
      document.getElementById("knowledge-assistant-dialog").close(); shell.toast("知识问答 Agent 绑定已保存"); await load();
    } catch (error) { result.textContent = error.message; }
    finally { button.disabled = false; }
  }
  function open(row = null) {
    editing = row;
    const form = document.getElementById("knowledge-base-form"); form.reset();
    form.elements.display_name.value = row?.display_name || ""; form.elements.description.value = row?.description || "";
    document.getElementById("knowledge-base-dialog-title").textContent = row ? "编辑运维知识库" : "新建运维知识库";
    document.getElementById("save-knowledge-base").textContent = row ? "保存修改" : "创建知识库";
    document.getElementById("knowledge-base-result").textContent = "";
    document.getElementById("knowledge-base-dialog").showModal();
  }
  async function save(event) {
    event.preventDefault(); const form = event.currentTarget; const button = document.getElementById("save-knowledge-base"); button.disabled = true;
    const payload = { display_name: form.elements.display_name.value.trim(), description: form.elements.description.value.trim() || null, models: {} };
    if (editing) payload.expected_row_version = editing.row_version;
    try { await KBotAIOpsAuth.request(editing ? `${api}/${encodeURIComponent(editing.knowledge_base_id)}` : api, { method: editing ? "PUT" : "POST", body: JSON.stringify(payload) }); document.getElementById("knowledge-base-dialog").close(); shell.toast(editing ? "知识库已更新" : "知识库已创建"); await load(); }
    catch (error) { document.getElementById("knowledge-base-result").textContent = error.message; }
    finally { button.disabled = false; }
  }
  async function remove(row) {
    if (!confirm(`确认删除知识库“${row.display_name}”？只有无资料、无诊断 Agent 和无知识问答 Agent 引用的知识库可以删除。`)) return;
    try { await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(row.knowledge_base_id)}`, { method: "DELETE", headers: { "If-Match": `"rv-${row.row_version}"` } }); shell.toast("知识库已删除"); await load(); } catch (error) { shell.toast(error.message); }
  }
  shell.ready.then(async () => {
    document.getElementById("create-knowledge-base").onclick = () => open();
    document.getElementById("configure-knowledge-assistant").onclick = openAssistantBinding;
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("knowledge-base-form").onsubmit = save;
    document.getElementById("knowledge-assistant-form").onsubmit = saveAssistantBinding;
    document.getElementById("knowledge-base-rows").onclick = (event) => { const edit = event.target.closest("[data-edit]"); const deletion = event.target.closest("[data-delete]"); const id = edit?.dataset.edit || deletion?.dataset.delete; const row = rows.find((item) => item.knowledge_base_id === id); if (edit && row) open(row); if (deletion && row) void remove(row); };
    try { await Promise.all([load(), loadAssistantBinding()]); } catch (error) { shell.toast(error.message); }
  });
})();
