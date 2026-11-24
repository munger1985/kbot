// 双端使用隔离资料验证批量审批、方案交集和部分受理。
import assert from 'node:assert/strict';

async function verifyBatchDrawerLayout(page) {
  const originalViewport = page.viewportSize();
  await page.evaluate(() => {
    globalThis.fixtureKnowledgeBatch = Array.from({length:18}, (_, index) => ({
      key: `long${String.fromCharCode(97+index)}`, title: `运维手册 ${index+1} · ${'LongAssetName'.repeat(12)}`,
      status:'REVIEW_REQUIRED',revision:4,auto:true,
    }));
  });
  await page.getByRole('searchbox',{name:'搜索运维资料'}).clear();
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  await page.getByRole('checkbox',{name:/选择待审批资料：运维手册/}).last().waitFor();
  const all = page.getByRole('checkbox',{name:'全选当前列表的待审批资料'});
  const batch = page.getByRole('dialog',{name:/批量审批/});
  const body = batch.locator('.admin-drawer-body');
  const submit = batch.getByRole('button',{name:'确认批量审批',exact:true});
  try {
    for (const [width,height] of [[1440,880],[390,640],[320,520]]) {
      await page.setViewportSize({width,height});
      await all.check();
      await page.getByRole('button',{name:'批量审批',exact:true}).click();
      await batch.getByRole('combobox',{name:'批量审批解析模式'}).waitFor();
      const layout = await batch.evaluate(async node => {
        await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        await Promise.all(node.getAnimations().map(animation=>animation.finished.catch(()=>{})));
        const header = node.querySelector('.admin-drawer-header');
        const body = node.querySelector('.admin-drawer-body');
        const footer = node.querySelector('.admin-drawer-footer');
        return {
          panel:node.getBoundingClientRect().toJSON(),
          header:header.getBoundingClientRect().toJSON(),
          body:body.getBoundingClientRect().toJSON(),
          footer:footer.getBoundingClientRect().toJSON(),
          overflow:[node,body,footer].map(element=>element.scrollWidth-element.clientWidth),
          bodyScroll:body.scrollHeight-body.clientHeight,
          strategyInFooter:footer.contains(node.querySelector('select')),
        };
      });
      const context = page.url()+' @ '+width+'×'+height+' '+JSON.stringify(layout);
      assert.equal(Math.round(layout.panel.width),Math.min(820,width),context);
      assert.ok(Math.abs(layout.panel.right-width)<=1 && Math.abs(layout.panel.bottom-height)<=1,context);
      assert.ok(layout.header.bottom<=layout.body.top+1 && layout.body.bottom<=layout.footer.top+1,context);
      assert.ok(layout.strategyInFooter && layout.bodyScroll>0 && layout.overflow.every(value=>value<=1),context);
      assert.equal(await body.evaluate(node=>node.scrollTop),0,context);
      await body.evaluate(node=>{node.scrollTop=node.scrollHeight;});
      assert.ok(await body.evaluate(node=>node.scrollTop>0),context);
      assert.ok(await submit.evaluate(node=>{
        const box = node.getBoundingClientRect();
        return document.elementFromPoint(box.x+box.width/2,box.y+box.height/2)===node;
      }),context);
      await batch.getByRole('button',{name:'关闭',exact:true}).click();
      await batch.waitFor({state:'hidden'});
    }
  } finally {
    if (originalViewport) await page.setViewportSize(originalViewport);
  }
}

export async function verifyOperationsKnowledgeBatchReview(page) {
  await page.evaluate(() => {
    globalThis.fixtureKnowledgeBatch = [
      {key:'a',title:'批审 A',status:'REVIEW_REQUIRED',revision:4,auto:true},
      {key:'b',title:'批审 B',status:'REVIEW_REQUIRED',revision:7,auto:false},
      {key:'c',title:'批审 C',status:'REVIEW_REQUIRED',revision:2,auto:true,conflict:true},
      {key:'d',title:'已发布 D',status:'PUBLISHED',revision:5,auto:true},
    ];
  });
  await page.getByRole('button',{name:'刷新',exact:true}).click();
  const bulk = page.getByRole('button',{name:'批量审批',exact:true});
  const all = page.getByRole('checkbox',{name:'全选当前列表的待审批资料'});
  const batch = page.getByRole('dialog',{name:/批量审批/});
  const submit = batch.getByRole('button',{name:'确认批量审批',exact:true});
  const mode = batch.getByRole('combobox',{name:'批量审批解析模式'});
  const choose = (name) => page.getByRole('checkbox',{name:`选择待审批资料：${name}`});
  await choose('批审 A').waitFor();
  assert.equal(await choose('已发布 D').count(),0);
  assert.equal(await bulk.isDisabled(),true);
  await all.check();
  await page.getByText('已选 3 项',{exact:true}).waitFor();
  await bulk.click();
  await mode.waitFor();
  assert.equal(await mode.inputValue(),'');
  assert.deepEqual(await mode.locator('option').evaluateAll(options=>options.map(option=>option.value)),['','TEXT']);
  assert.equal(await submit.isDisabled(),true);
  await batch.getByRole('button',{name:'关闭',exact:true}).click();
  await batch.waitFor({state:'hidden'});

  await choose('批审 B').uncheck();
  assert.equal(await all.evaluate(input=>input.indeterminate),true);
  await bulk.click();
  await mode.waitFor();
  assert.equal(await mode.inputValue(),'AUTO');
  await submit.click();
  await batch.getByText('已受理 1 项，未确认受理 1 项。').waitFor();
  await batch.getByText('批审 C：隔离冲突，请刷新资料').waitFor();
  await batch.getByRole('button',{name:'关闭',exact:true}).click();
  await batch.waitFor({state:'hidden'});
  assert.equal(await choose('批审 C').isChecked(),true);
  await page.getByRole('searchbox',{name:'搜索运维资料'}).fill('批审 B');
  assert.equal(await bulk.isDisabled(),true);
  await all.check();
  await bulk.click();
  await mode.selectOption('TEXT');
  await submit.click();
  await batch.waitFor({state:'hidden'});
  await page.getByText('AI 识别中',{exact:true}).waitFor();

  const commands = await page.evaluate(() => fixtureRequests
    .filter(row => row.method === 'POST' && /version-batch-[a-z]+:approve$/.test(row.path))
    .map(row => [row.path.match(/version-batch-([a-z]+):approve$/)[1], JSON.parse(row.body)])
    .sort(([a],[b]) => a.localeCompare(b)));
  assert.deepEqual(commands,[
    ['a',{expected_row_version:4,comment:null,parse_strategy:'AUTO'}],
    ['b',{expected_row_version:7,comment:null,parse_strategy:'TEXT'}],
    ['c',{expected_row_version:2,comment:null,parse_strategy:'AUTO'}],
  ]);
  await verifyBatchDrawerLayout(page);
}
