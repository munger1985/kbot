(() => {
  const apps = globalThis.KBOT_UI_CONFIG?.enabledApps;
  if (!Array.isArray(apps)) return;

  const enabled = new Set(apps);
  const cards = [...document.querySelectorAll("[data-app]")];
  for (const card of cards) {
    card.hidden = !enabled.has(card.dataset.app);
  }
  document.getElementById("empty-apps").hidden = cards.some((card) => !card.hidden);
})();
