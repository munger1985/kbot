const items = [
  {
    permission: "user_management", href: "./users.html", label: "用户管理",
    icon: ["M16 20v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2", "M9 10a4 4 0 1 0 0-8 4 4 0 0 0 0 8Z", "M16 11a4 4 0 0 0 0-8", "M22 20v-2a4 4 0 0 0-3-3.87"],
  },
  {
    permission: "system_management", href: "./models.html", label: "AI 模型管理",
    icon: ["M12 3 4.5 7.2v9.6L12 21l7.5-4.2V7.2z", "M4.8 7.4 12 12l7.2-4.6", "M12 12v9"],
  },
];

function renderIcon(paths) {
  const namespace = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(namespace, "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  for (const data of paths) {
    const path = document.createElementNS(namespace, "path");
    path.setAttribute("d", data);
    svg.append(path);
  }
  return svg;
}

export function mountPlatformMenu(entries, currentPermission, current = "page") {
  const main = document.getElementById("system-content");
  const header = document.querySelector(".system-management .media-topbar");
  const nav = document.createElement("nav");
  nav.id = "platform-navigation";
  nav.className = "platform-sidebar";
  nav.setAttribute("aria-label", "平台管理");
  const links = document.createElement("div");
  links.className = "platform-nav-items";

  for (const { permission, href, label, icon } of items) {
    if (entries[permission] !== true) continue;
    const link = document.createElement("a");
    link.className = "platform-nav-item";
    link.href = href;
    if (permission === currentPermission) link.setAttribute("aria-current", current);
    const mark = document.createElement("span");
    mark.className = "platform-nav-icon";
    mark.append(renderIcon(icon));
    link.append(mark, label);
    links.append(link);
  }
  if (!links.childElementCount) return;
  nav.append(links);

  const trigger = document.createElement("button");
  trigger.className = "platform-nav-trigger";
  trigger.type = "button";
  trigger.setAttribute("aria-label", "平台管理菜单");
  trigger.setAttribute("aria-controls", nav.id);
  trigger.append(renderIcon(["M5 7h14", "M5 12h14", "M5 17h14"]));

  const scrim = document.createElement("button");
  scrim.className = "platform-nav-scrim";
  scrim.type = "button";
  scrim.tabIndex = -1;
  scrim.setAttribute("aria-label", "关闭平台管理菜单");

  const narrow = matchMedia("(max-width: 860px)");
  function setOpen(open) {
    trigger.setAttribute("aria-expanded", String(open));
    nav.classList.toggle("is-open", open);
    nav.inert = narrow.matches && !open;
    scrim.hidden = !open;
    main.inert = open;
  }

  function closeMenu() {
    setOpen(false);
    trigger.focus();
  }

  trigger.addEventListener("click", () => {
    if (trigger.getAttribute("aria-expanded") === "true") return closeMenu();
    setOpen(true);
    (links.querySelector("[aria-current]") || links.firstElementChild).focus();
  });
  scrim.addEventListener("click", closeMenu);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && !scrim.hidden) {
      event.preventDefault();
      closeMenu();
    }
  });
  narrow.addEventListener("change", () => {
    const focusInMenu = nav.contains(document.activeElement);
    setOpen(false);
    if (narrow.matches && focusInMenu) trigger.focus();
  });

  setOpen(false);
  header.querySelector(".media-context").prepend(trigger);
  header.after(nav, scrim);
  document.body.classList.add("has-platform-menu");
}
