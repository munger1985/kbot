(function () {
  "use strict";
  const triggers = new WeakMap();
  const panel = id => document.getElementById(id);
  // 复用项目 admin-drawer 样式；只通过显式按钮或业务成功关闭。
  document.querySelectorAll('.admin-drawer-backdrop').forEach(backdrop => {
    ['pointerdown', 'mousedown', 'click'].forEach(name => backdrop.addEventListener(name, event => {
      if (event.target === backdrop) { event.preventDefault(); event.stopImmediatePropagation(); }
    }));
    backdrop.addEventListener('keydown', event => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopImmediatePropagation(); return; }
      if (event.key !== 'Tab') return;
      const drawer = backdrop.querySelector('[role="dialog"]');
      const fields = [...drawer.querySelectorAll('button:not(:disabled), input:not(:disabled), select:not(:disabled), textarea:not(:disabled), a[href], [tabindex="0"]')].filter(item => item.getClientRects().length);
      const first = fields[0], last = fields.at(-1);
      if (!first) { event.preventDefault(); drawer.focus(); }
      else if (event.shiftKey && (document.activeElement === first || document.activeElement === drawer)) { event.preventDefault(); last.focus(); }
      else if (!event.shiftKey && (document.activeElement === last || document.activeElement === drawer)) { event.preventDefault(); first.focus(); }
    });
  });
  window.KBotDrawer = Object.freeze({
    open(id) {
      const drawer = panel(id);
      triggers.set(drawer, document.activeElement);
      drawer.parentElement.hidden = false;
      (drawer.querySelector('[autofocus]') || drawer).focus();
    },
    close(id) {
      const drawer = panel(id);
      drawer.parentElement.hidden = true;
      drawer.dispatchEvent(new Event('drawer:closed'));
      const trigger = triggers.get(drawer);
      if (trigger?.isConnected) trigger.focus();
      triggers.delete(drawer);
    }
  });
})();
