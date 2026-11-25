// 在双端真实抽屉验证原文绑定、索引缺失和提炼结果展示。
import assert from 'node:assert/strict';

export async function verifyOperationsKnowledgeSource(page) {
  const drawer = page.locator('.knowledge-detail-drawer');
  const card = drawer.locator('.knowledge-source-card');
  for (const state of ['ready', 'historical']) {
    await page.evaluate(value => { globalThis.fixtureKnowledgeSource = value; }, state);
    await page.getByRole('button', { name:'查看', exact:true }).click();
    await drawer.getByRole('button', { name:'文件与解析' }).click();
    await card.getByText('数据库计算节点硬件规格说明.txt', { exact:true }).waitFor();
    assert.ok(await card.locator('strong').evaluate(node => node.clientWidth > 100 && node.scrollWidth <= node.clientWidth));
    await drawer.getByText('计算节点规格', { exact:true }).waitFor();
    await drawer.getByText('核对硬件', { exact:true }).click();
    await drawer.getByText('核对节点数量', { exact:true }).waitFor();
    assert.doesNotMatch(await drawer.innerText(), /OPS_MANUAL_PROFILE|procedure_key/);
    if (state === 'ready') {
      await drawer.getByText('索引已就绪', { exact:true }).waitFor();
      await card.getByRole('button', { name:'预览原文' }).click();
      const preview = card.locator('.knowledge-source-preview');
      await preview.locator('pre').waitFor();
      assert.match(await preview.locator('pre').textContent(), /示例原文\n<img/);
      assert.equal(await preview.locator('img').count(), 0);
      assert.equal(await page.evaluate(() => Boolean(globalThis.manualSourceInjected)), false);
      assert.match(await page.evaluate(() => fixtureRequests.filter(row => row.method === 'SOURCE').at(-1).path),
        /\/versions\/knowledge-version-1\/source\?document_version_id=doc-manual$/);
      await card.getByRole('button', { name:'收起预览' }).click();
      await preview.waitFor({ state:'hidden' });
    } else {
      await drawer.getByText(/有发布记录，但没有关联当前索引/).waitFor();
      assert.equal(await card.getByRole('button', { name:/原文/ }).count(), 0);
      assert.equal(await drawer.getByText('索引已就绪', { exact:true }).count(), 0);
    }
    await drawer.locator('.admin-drawer-header .icon-button').click();
    await drawer.waitFor({ state:'hidden' });
  }
  await page.evaluate(() => { delete globalThis.fixtureKnowledgeSource; });
}

export async function verifyOperationsKnowledgeRetirement(page) {
  const drawer = page.locator('.knowledge-detail-drawer');
  const row = page.getByRole('row').filter({ hasText:'asset-1' });
  const filter = page.locator('.resource-list-filter').filter({ hasText:'全部状态' }).getByRole('combobox');
  const portal = !new URL(page.url()).pathname.startsWith('/ui/');
  for (const status of ['', 'PUBLISHED']) {
    await page.evaluate(() => { globalThis.fixtureKnowledgeTagStatus = 'PUBLISHED'; });
    await filter.selectOption(status);
    await row.getByText('解析完成', { exact:true }).waitFor();
    await row.getByRole('button', { name:'查看', exact:true }).click();
    await page.evaluate(() => { globalThis.fixtureHoldKnowledgeAssets = true; });
    await drawer.getByRole('button', { name:'退役', exact:true }).click();
    if (portal) await page.getByRole('button', { name:'确认退役', exact:true }).click();
    // 在列表请求仍阻塞时确认抽屉已关闭、状态已由退役回执更新。
    await page.waitForFunction(() => typeof globalThis.fixtureReleaseKnowledgeAssets === 'function');
    await drawer.waitFor({ state:'hidden' });
    if (!status) await row.getByText('已退役', { exact:true }).waitFor();
    else await row.waitFor({ state:'hidden' });
    await page.evaluate(() => { globalThis.fixtureReleaseKnowledgeAssets(); delete globalThis.fixtureReleaseKnowledgeAssets; });
    if (status) await filter.selectOption('RETIRED');
    await row.getByText('已退役', { exact:true }).waitFor();
    await row.getByRole('button', { name:'查看', exact:true }).click();
    await drawer.getByText('已退役', { exact:true }).waitFor();
    await drawer.locator('.admin-drawer-header .icon-button').click();
    await drawer.waitFor({ state:'hidden' });
  }
  await filter.selectOption('');
}
