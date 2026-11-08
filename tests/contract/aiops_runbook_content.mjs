import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../../ui/aiops/js/aiops-workspaces.js', import.meta.url), 'utf8');
for (const page of ['chat', 'diagnosis']) {
  const context = { document: { body: { dataset: { page } } }, window: {
    KBotAIOpsShell: { escape: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;'),
      fmt: String, ready: new Promise(() => {}) },
  } };
  context.KBotAIOpsShell = context.window.KBotAIOpsShell;
  context.addEventListener = () => {};
  vm.createContext(context);
  vm.runInContext(source.replace(/\}\)\(\);\s*$/, 'globalThis.runbookTest = { answerBlockHtml };})();'), context);
  const note = { command_type: 'MANUAL', executor: 'MANUAL', title: '检查', content: '人工核对逻辑读' };
  for (const draft of [false, true]) {
    const payload = { profile: 'NONE', title: '诊断建议与验证方案', phases: [{ title: '补证与验证', steps: [{
      title: '核对计划', commands: draft ? [{ command_type: 'MANUAL', executor: 'PSQL', content: 'EXPLAIN SELECT 1' }] : [],
      verification_commands: [note], rollback: [note], verification_notes: ['核对索引覆盖'], rollback_conditions: ['延迟上升则停止'],
    }] }] };
    const html = context.runbookTest.answerBlockHtml({ block_type: 'IMPLEMENTATION_RUNBOOK', payload },
      { turn_id: 'turn', conversation_id: 'conversation', display_dictionary: {} });
    assert.match(html, /核对索引覆盖/);
    assert.match(html, /延迟上升则停止/);
    assert.match(html, /人工核对逻辑读/);
    assert.equal((html.match(/<pre>/g) || []).length, draft ? 1 : 0);
    assert.equal((html.match(/复制命令/g) || []).length, draft ? 1 : 0);
    if (draft) assert.match(html, /<code>EXPLAIN SELECT 1<\/code>/);
  }
}
console.log('通过：聊天与诊断页面的说明、验证、回退和真实SQL草案分组');
