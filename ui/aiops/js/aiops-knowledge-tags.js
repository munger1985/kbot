(function () {
  "use strict";

  const api = "/api/v1/apps/aiops/operations-knowledge/tags";
  const shell = globalThis.KBotAIOpsShell;
  const state = { tags: [], editing: null, saving: false };
  const escape = (value) => shell.escape(value == null ? "" : value);
  const categoryLabel = (value) => ({ RESOURCE_CATEGORY: "资源分类", TECHNOLOGY_STACK: "技术栈", VENDOR: "厂商" }[value] || value || "—");

  function visibleTags() {
    const keyword = document.getElementById("tag-search").value.trim().toLowerCase();
    const category = document.getElementById("tag-category").value;
    const status = document.getElementById("tag-status").value;
    return state.tags.filter((item) => (!category || item.category === category) && (!status || item.status === status) && (!keyword || [item.display_name, item.description || "", categoryLabel(item.category)].some((value) => String(value).toLowerCase().includes(keyword))));
  }

  function render() {
    const items = visibleTags();
    document.getElementById("tag-count").textContent = `${items.length} 项`;
    document.getElementById("knowledge-tags").innerHTML = items.length ? items.map((item) => `<tr><td><strong>${escape(item.display_name)}</strong><small>${escape(item.tag_id)}</small></td><td>${escape(categoryLabel(item.category))}</td><td>${escape(item.description || "—")}</td><td>${escape(item.sort_order)}</td><td><span class="ops-badge ${item.status === "ACTIVE" ? "good" : ""}">${item.status === "ACTIVE" ? "启用" : "停用"}</span></td><td><button type="button" data-edit="${escape(item.tag_id)}">编辑</button><button class="${item.status === "ACTIVE" ? "danger" : ""}" type="button" data-toggle="${escape(item.tag_id)}">${item.status === "ACTIVE" ? "停用" : "启用"}</button></td></tr>`).join("") : '<tr><td colspan="6" class="ops-empty">当前条件下没有知识标签。</td></tr>';
  }

  async function load() {
    const result = await KBotAIOpsAuth.request(`${api}?include_disabled=true`);
    state.tags = result.items || [];
    render();
  }

  function openEditor(item = null) {
    state.editing = item;
    document.getElementById("tag-form").reset();
    document.getElementById("tag-drawer-title").textContent = item ? "编辑知识标签" : "新增知识标签";
    document.getElementById("editor-category").value = item?.category || "RESOURCE_CATEGORY";
    document.getElementById("editor-name").value = item?.display_name || "";
    document.getElementById("editor-description").value = item?.description || "";
    document.getElementById("editor-sort").value = String(item?.sort_order || 0);
    document.getElementById("tag-result").textContent = "";
    document.getElementById("tag-drawer").showModal();
  }

  async function save(event) {
    event.preventDefault();
    if (state.saving) return;
    const displayName = document.getElementById("editor-name").value.trim();
    if (!displayName) return;
    const payload = {
      category: document.getElementById("editor-category").value,
      display_name: displayName,
      description: document.getElementById("editor-description").value.trim() || null,
      sort_order: Number(document.getElementById("editor-sort").value || 0),
      expected_row_version: state.editing?.row_version || null,
    };
    state.saving = true;
    document.getElementById("save-tag").disabled = true;
    document.getElementById("tag-result").textContent = "正在保存…";
    try {
      await KBotAIOpsAuth.request(state.editing ? `${api}/${encodeURIComponent(state.editing.tag_id)}` : api, { method: state.editing ? "PUT" : "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      document.getElementById("tag-drawer").close();
      shell.toast(state.editing ? "知识标签已更新" : "知识标签已创建");
      await load();
    } finally {
      state.saving = false;
      document.getElementById("save-tag").disabled = false;
    }
  }

  async function toggle(item) {
    if (state.saving) return;
    const action = item.status === "ACTIVE" ? "disable" : "enable";
    state.saving = true;
    try {
      await KBotAIOpsAuth.request(`${api}/${encodeURIComponent(item.tag_id)}:${action}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ expected_row_version: Number(item.row_version) }) });
      shell.toast(action === "enable" ? "知识标签已启用" : "知识标签已停用；历史版本标签不受影响");
      await load();
    } finally {
      state.saving = false;
    }
  }

  shell.ready.then(async () => {
    document.querySelectorAll("[data-close]").forEach((button) => { button.onclick = () => button.closest("dialog").close(); });
    document.getElementById("create-tag").onclick = () => openEditor();
    document.getElementById("refresh-tags").onclick = () => load().catch((error) => shell.toast(error.message));
    ["tag-search", "tag-category", "tag-status"].forEach((id) => { document.getElementById(id).oninput = render; });
    document.getElementById("knowledge-tags").onclick = (event) => {
      const edit = event.target.closest("[data-edit]");
      const toggleButton = event.target.closest("[data-toggle]");
      const id = edit?.dataset.edit || toggleButton?.dataset.toggle;
      const item = state.tags.find((candidate) => candidate.tag_id === id);
      if (edit && item) openEditor(item);
      if (toggleButton && item) toggle(item).catch((error) => shell.toast(error.message));
    };
    document.getElementById("tag-form").onsubmit = (event) => save(event).catch((error) => { document.getElementById("tag-result").textContent = error.message; shell.toast(error.message); });
    try { await load(); } catch (error) { shell.toast(error.message); }
  });
})();
