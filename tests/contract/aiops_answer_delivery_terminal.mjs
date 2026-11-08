import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

const source = readFileSync(new URL("../../ui/aiops/js/aiops-workspaces.js", import.meta.url), "utf8");
const sandbox = {
  addEventListener() {},
  KBotAIOpsShell: { escape: String, fmt: String, ready: new Promise(() => {}) },
};
sandbox.globalThis = sandbox;
vm.runInNewContext(source.replace(/\}\)\(\);\s*$/, "globalThis.delivery = { retainPublishedAnswer, publishedAnswers };})();"), sandbox);
const { retainPublishedAnswer, publishedAnswers } = sandbox.delivery;
const native = "\n## 分析\n已向用户输出的 SQL 诊断。\n";
publishedAnswers.set("turn", native);
for (const final of ["MODEL_DEADLINE_EXCEEDED", "", native.slice(0, 10), native, native + "\n后续结论。"]) {
  const evidence = { block_type: "TABLE", payload: { rows: [] } };
  const turn = { turn_id: "turn", status: "PARTIAL", answer_blocks: [
    { block_type: "MARKDOWN", payload: { markdown: final } }, evidence,
  ] };
  const delivered = retainPublishedAnswer(turn);
  assert.ok(delivered.answer_blocks[0].payload.markdown.startsWith(native));
  assert.equal(delivered.answer_blocks[1], evidence);
  if (!final.startsWith(native)) {
    assert.equal(delivered.answer_blocks[0].payload.raw_model_output, true);
    assert.equal(delivered.answer_blocks[0].payload.validation_status, "UNVERIFIED");
    assert.ok(delivered.answer_blocks[0].payload.answer_notice);
  } else {
    assert.equal(delivered, turn);
  }
  assert.equal(turn.answer_blocks[0].payload.markdown, final);
}
const other = { turn_id: "another-turn", answer_blocks: [] };
assert.equal(retainPublishedAnswer(other), other);
console.log("AIOps 已交付正文保护合同通过（错误终态、空正文、截断、追加及轮次隔离）");
