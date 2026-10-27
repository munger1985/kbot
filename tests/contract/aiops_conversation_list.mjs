import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { runInNewContext } from "node:vm";

const sourcePath = fileURLToPath(new URL("../../ui/aiops/js/aiops-workspaces.js", import.meta.url));
const source = readFileSync(sourcePath, "utf8");
const instrumented = source.replace(/\}\)\(\);\s*$/, "globalThis.sessionTest = { conversationGroups, renderConversationList, showConversationMenu, state };})();");
assert.notEqual(instrumented, source);
const listeners = {};
const list = { innerHTML: "", querySelectorAll: () => [] };
const createElement = () => ({
  attrs: {}, style: {}, children: [], removed: false,
  setAttribute(name, value) { this.attrs[name] = value; },
  append(child) { this.children.push(child); },
  remove() { this.removed = true; },
  contains(node) { return this === node || this.children.includes(node); },
  focus() { this.focused = true; },
});
const document = {
  addEventListener: (name, handler) => { listeners[name] = handler; },
  getElementById: () => list,
  createElement,
  body: { append(node) { this.last = node; } },
};
const escape = (value) => String(value ?? "").replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;");
const context = {
  document, window: { innerWidth: 500, innerHeight: 500, addEventListener() {} },
  KBotAIOpsShell: { escape, fmt: (value) => value || "—", ready: new Promise(() => {}), toast() {} },
  KBotMarkdown: {}, addEventListener() {}, confirm: () => false,
};
runInNewContext(instrumented, context, { filename: sourcePath });
const date = (day) => new Date(2026, 9, day, 12).toISOString();
const rows = [
  { conversation_id: "old", title: "旧会话", updated_at: date(1) },
  { conversation_id: "selected", title: '<SQL "测试">', updated_at: date(27) },
  { conversation_id: "yesterday", title: "昨日会话", updated_at: date(26) },
  { conversation_id: "unknown", title: "无时间", updated_at: null },
];
const groups = context.sessionTest.conversationGroups(rows, new Date(2026, 9, 27, 16));
assert.deepEqual(Array.from(groups, (group) => group.label), ["今天", "昨天", "更早", "时间未知"]);
assert.equal(groups[0].items[0].conversation_id, "selected");
context.sessionTest.state.conversation = { conversation_id: "selected" };
context.sessionTest.renderConversationList(rows);
assert.match(list.innerHTML, /class="ops-workspace-more"/);
assert.match(list.innerHTML, /aria-current="true"/);
assert.match(list.innerHTML, /&lt;SQL &quot;测试&quot;&gt;/);
assert.ok(!list.innerHTML.includes('class="ops-workspace-delete"'));
const trigger = createElement();
trigger.dataset = { deleteId: "selected", deleteTitle: "测试" };
trigger.getBoundingClientRect = () => ({ right: 250, top: 450, bottom: 480 });
context.sessionTest.showConversationMenu(trigger);
assert.equal(trigger.attrs["aria-expanded"], "true");
assert.equal(document.body.last.style.top, "392px");
assert.equal(document.body.last.children[0].textContent, "删除会话");
assert.equal(document.body.last.children[0].focused, true);
let prevented = false;
listeners.keydown({ key: "Escape", preventDefault() { prevented = true; } });
assert.equal(prevented, true);
assert.equal(trigger.attrs["aria-expanded"], "false");
assert.equal(trigger.focused, true);
assert.equal(document.body.last.removed, true);
