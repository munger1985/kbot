(function () {
  "use strict";
  const layout = globalThis.KBotAIOpsTopologyLayout, view = globalThis.KBotAIOpsTopologyView;
  let state = view.createState(), input = null, root = null, version = 0, contextKey, selectedResourceId;
  function clearHost() {version++;state.hostContext = null;state.hostBusy = false;state.hostError = "";}
  function dispose(preserve = false) {if (root) {root.onclick = null;root.onchange = null;}root = null;if (!preserve) {clearHost();input = null;}}
  function render(value) {
    dispose(true);
    if (contextKey !== value.contextKey) {clearHost();state = view.createState();}
    contextKey = value.contextKey;
    input = value;
    if (selectedResourceId !== value.selectedResourceId && value.resources.some(resource => resource.draft_resource_id === value.selectedResourceId && layout.hostTypes.has(resource.resource_type_key))) state.hostId = value.selectedResourceId;
    selectedResourceId = value.selectedResourceId;
    return `<div class="business-system-topology">${view.renderTopology(value,state)}</div>`;
  }
  function refresh() {
    const old = root;if (!old || !input) return;
    const viewport = old.querySelector(".graph-viewport");
    const scroll = viewport ? [viewport.scrollLeft,viewport.scrollTop] : null;
    const focused = old.contains(document.activeElement) ? document.activeElement : null;
    const focusKey = focused && Object.keys(focused.dataset).find(key => key.startsWith("topology"));
    old.outerHTML = render(input);mount();
    if (scroll) {const current = root.querySelector(".graph-viewport");if (current) {current.scrollLeft = scroll[0];current.scrollTop = scroll[1];}}
    if (focusKey) Array.from(root.querySelectorAll("button,select")).find(element => element.dataset[focusKey] === focused.dataset[focusKey])?.focus({preventScroll:true});
  }
  async function click(event) {
    const button = event.target.closest("button");if (!button || button.disabled || !input) return;
    if (button.dataset.topologyResource || button.dataset.topologyRelation) return;
    if (!Object.keys(button.dataset).some(key => key.startsWith("topology"))) return;
    event.stopPropagation();
    if (button.dataset.topologyView) {clearHost();state.view = button.dataset.topologyView;}
    if (button.dataset.topologyZoom) {
      const current = state.hostContext ? layout.hostProjection(state.hostContext) : input;
      const model = layout.graphModel(current.resources,current.relations,state.view);
      const available = root.querySelector(".graph-viewport")?.clientWidth || model.width;
      state.zoom = button.dataset.topologyZoom === "fit" ? Math.min(1,Math.max(.25,(available - 4) / model.width)) : Math.min(1.5,Math.max(.25,state.zoom + (button.dataset.topologyZoom === "in" ? .1 : -.1)));
    }
    if (button.hasAttribute("data-topology-host-clear")) clearHost();
    if (button.hasAttribute("data-topology-host-load")) {
      const host = input.resources.find(resource => resource.draft_resource_id === state.hostId);
      if (!host?.resource_id || !input.loadHostTopology) return;
      clearHost();state.hostBusy = true;const ticket = version, load = input.loadHostTopology;refresh();
      try {const result = await load(host.resource_id);if (ticket === version) {state.hostContext = result;state.zoom = 1;}}
      catch (error) {if (ticket === version) state.hostError = error instanceof Error ? error.message : "读取主机承载失败，请重试。";}
      finally {if (ticket === version) {state.hostBusy = false;refresh();}}
      return;
    }
    refresh();
  }
  function mount() {
    root = document.querySelector(".business-system-topology");if (!root) return;
    root.onclick = click;
    root.onchange = event => {if (event.target.hasAttribute("data-topology-host")) {event.stopPropagation();clearHost();state.hostId = event.target.value;refresh();}};
  }
  globalThis.KBotAIOpsTopology = {render,mount,dispose};
})();
