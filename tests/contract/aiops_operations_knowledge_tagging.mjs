// 隔离模拟 Worker，验证受理后关闭抽屉和后台状态刷新。
import assert from 'node:assert/strict';

export async function verifyOperationsKnowledgeTagging(page) {
  await page.setViewportSize({width:1440,height:900});
  await page.evaluate(() => { globalThis.fixtureKnowledgeTagStatus = 'REVIEW_REQUIRED'; });
  const drawer = page.locator('.knowledge-detail-drawer');
  const releaseRefreshAfterClose = async () => {
    await page.waitForFunction(() => typeof globalThis.fixtureReleaseKnowledgeAssets === 'function');
    await drawer.waitFor({state:'hidden'});
    await page.evaluate(() => {
      globalThis.fixtureReleaseKnowledgeAssets();
      delete globalThis.fixtureReleaseKnowledgeAssets;
    });
    await page.getByText('AI 识别中',{exact:true}).waitFor();
  };
  await page.getByRole('button',{name:'审批资料',exact:true}).click();
  assert.equal(await drawer.getByRole('combobox',{name:'批准时选择解析方案'}).inputValue(),'AUTO');
  await drawer.getByText('暂无识别标签。',{exact:true}).waitFor();
  assert.equal(await drawer.locator('.knowledge-tag-group').count(),0);
  await page.evaluate(() => { globalThis.fixtureHoldKnowledgeAssets = true; });
  await drawer.getByRole('button',{name:'批准并开始 AI 识别',exact:true}).click();
  await releaseRefreshAfterClose();

  await page.getByRole('button',{name:'查看进度',exact:true}).click();
  await drawer.waitFor({state:'visible'});
  assert.equal(await drawer.getByRole('button',{name:'立即处理',exact:true}).count(),0);
  await drawer.locator('.admin-drawer-header .icon-button').click();
  await page.evaluate(() => { globalThis.fixtureKnowledgeTagStatus = 'TAGGING_FAILED'; });
  await page.getByRole('button',{name:'处理异常',exact:true}).click({timeout:15000});
  await drawer.getByText('隔离标签识别失败',{exact:true}).waitFor();
  await page.evaluate(() => { globalThis.fixtureHoldKnowledgeAssets = true; });
  await drawer.getByRole('button',{name:'重试标签识别',exact:true}).click();
  await releaseRefreshAfterClose();

  const commands = await page.evaluate(() => fixtureRequests
    .filter(row => row.method === 'POST' && /knowledge-version-1:(approve|retry|reconcile)$/.test(row.path))
    .map(row => [row.path.split(':').at(-1), JSON.parse(row.body)]));
  assert.deepEqual(commands,[
    ['approve',{expected_row_version:1,comment:null,parse_strategy:'AUTO'}],
    ['retry',{expected_row_version:2,comment:null}],
  ]);
  await page.evaluate(() => { globalThis.fixtureKnowledgeTagStatus = 'PUBLISHED'; });
  await page.getByText('解析完成',{exact:true}).waitFor({timeout:15000});
  await page.getByRole('button',{name:'查看',exact:true}).click();
  const tags = drawer.locator('.knowledge-tag-review');
  await tags.getByText('存储系统',{exact:true}).waitFor();
  assert.deepEqual(await tags.locator('legend').allTextContents(),['资源分类','厂商']);
  assert.deepEqual((await tags.locator('.knowledge-tag-option strong').allTextContents()).sort(),['Oracle','存储系统','物理服务器']);
  assert.equal(await tags.locator('input:checked:disabled').count(),3);
  assert.match(await tags.innerText(),/AI 识别 · 94%/);
  assert.doesNotMatch(await tags.textContent(),/exadata\.pdf|本章介绍 Exadata 存储架构。/);
  assert.doesNotMatch(await drawer.locator('.knowledge-summary-grid').innerText(),/受控标签|适用范围|RESOURCE_CATEGORY=|VENDOR=/);
  await drawer.locator('.admin-drawer-header .icon-button').click();
  await drawer.waitFor({state:'hidden'});
}
