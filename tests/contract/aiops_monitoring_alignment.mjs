// 实例导航与顶部观测设置的隔离交互验收，不访问真实监控源。
import assert from 'node:assert/strict';
import {resolve} from 'node:path';

async function verifyMonitoringContentRows(kbot, reference, artifacts) {
  const failures=[];
  for (const [label,page] of [['kbot',kbot],['ammolite',reference]]) {
    const originalCount=await page.locator('.monitoring-panel').count();
    await page.setViewportSize({width:1440,height:600});
    await page.evaluate(()=>{globalThis.fixtureMonitoringDensePanels=true;});
    await page.getByRole('button',{name:'刷新',exact:true}).click();
    await page.waitForFunction(()=>document.querySelectorAll('.monitoring-panel').length===24);
    try {
      for (const [width,height] of [[1920,768],[1440,600],[1024,600],[821,600]]) {
        await page.setViewportSize({width,height});
        for (const expanded of [false,true]) {
          await page.locator('.monitoring-dashboard-main').evaluate(node=>{node.scrollTop=0;});
          if (expanded) await page.getByRole('button',{name:/^数据质量/}).click();
          // 移开指针并等待通用卡片悬停动画结束，再测量布局间距。
          await page.mouse.move(0,0);
          await page.waitForFunction(()=>getComputedStyle(document.querySelector('.monitoring-observation-card')).transform==='none');
          const bounds=await page.evaluate(()=>{
            const rect=node=>{const box=node.getBoundingClientRect();return {top:box.top,bottom:box.bottom,left:box.left,right:box.right,height:box.height};};
            const main=document.querySelector('.monitoring-dashboard-main');
            return {card:rect(document.querySelector('.monitoring-observation-card')),toolbar:rect(document.querySelector('.monitoring-query-toolbar')),summary:rect(document.querySelector('.monitoring-summary-strip')),panels:rect(document.querySelector('.monitoring-panels')),fields:[...document.querySelectorAll('.monitoring-rail-fields select')].map(node=>({...rect(node),hit:document.elementFromPoint(node.getBoundingClientRect().x+node.clientWidth/2,node.getBoundingClientRect().y+node.clientHeight/2)===node})),scrollHeight:main.scrollHeight,clientHeight:main.clientHeight,gap:Number.parseFloat(getComputedStyle(main).rowGap)};
          });
          if (width===1920||width===1440) await page.screenshot({path:resolve(artifacts,`${label}-monitoring-dense-${width}x${height}-${expanded?'quality':'settings'}.png`)});
          assert.ok(bounds.toolbar.top>=bounds.card.top&&bounds.summary.bottom<=bounds.card.bottom+1,`${label} ${width}x${height}：多行指标不得压缩设置/状态卡片：${JSON.stringify(bounds)}`);
          assert.equal(bounds.fields.length,4);
          assert.ok(bounds.fields.every(field=>field.top>=bounds.card.top&&field.bottom<=bounds.card.bottom&&field.hit),'四个设置控件必须完整可见且可点击');
          assert.ok(Math.abs(bounds.panels.top-bounds.card.bottom-bounds.gap)<1,`${label} ${width}x${height}：多行指标仍保持统一间距：${JSON.stringify(bounds)}`);
          assert.ok(bounds.scrollHeight>bounds.clientHeight,'指标较多时由右侧工作区滚动，不得挤压顶部设置');
          const lastPanel=page.locator('.monitoring-panel').last();
          await lastPanel.scrollIntoViewIfNeeded();
          assert.ok(await lastPanel.locator('header').evaluate(node=>{const box=node.getBoundingClientRect(),main=document.querySelector('.monitoring-dashboard-main').getBoundingClientRect();return box.top>=main.top&&box.bottom<=main.bottom;}),'最后一行指标必须可以滚动到');
          if (expanded) {
            const detail=page.locator('.monitoring-quality-detail');
            await detail.scrollIntoViewIfNeeded();
            assert.ok(await detail.locator('header').evaluate(node=>{const box=node.getBoundingClientRect(),main=document.querySelector('.monitoring-dashboard-main').getBoundingClientRect();return box.top>=main.top&&box.bottom<=main.bottom;}),'数据质量展开后必须保留内容高度且可滚动到');
            await page.locator('.monitoring-dashboard-main').evaluate(node=>{node.scrollTop=0;});
            await page.getByRole('button',{name:/^数据质量/}).click();
          }
        }
      }
    } catch (error) {
      failures.push(error.message);
    } finally {
      await page.setViewportSize({width:1440,height:1000});
      await page.evaluate(()=>{globalThis.fixtureMonitoringDensePanels=false;});
      await page.getByRole('button',{name:'刷新',exact:true}).click();
      await page.waitForFunction(count=>document.querySelectorAll('.monitoring-panel').length===count,originalCount);
      await page.locator('.monitoring-dashboard-main').evaluate(node=>{node.scrollTop=0;});
    }
  }
  assert.deepEqual(failures,[],`多指标/短视口布局回归失败，截图：${artifacts}`);
}

export async function verifyMonitoringNavigation(kbot, reference, compareRegions, artifacts) {
  await verifyMonitoringContentRows(kbot,reference,artifacts);
  const pages = [kbot, reference];
  for (const page of pages) {
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(()=>{globalThis.fixtureMonitoringTree=true;});
    await page.getByRole('combobox',{name:'监控源',exact:true}).selectOption('source-1');
    await page.waitForFunction(()=>document.querySelectorAll('.monitoring-instance-option').length===4);
    assert.equal(await page.locator('.monitoring-system-group').count(),3,'同一业务系统的多个资源必须合并为一个分组');
    const mes=page.locator('.monitoring-system-group').filter({has:page.locator('.monitoring-system-toggle strong',{hasText:'MES 生产系统'})});
    assert.equal(await mes.getByRole('radio').count(),2);
    assert.equal(await page.locator('.monitoring-system-group').filter({hasText:'未归属业务系统'}).getByRole('radio').count(),1);
    assert.equal(await page.locator('.monitoring-instance-tree img,.monitoring-instance-tree script').count(),0);
    const requestCount=await page.evaluate(()=>fixtureRequests.length);
    await mes.getByRole('button').click();
    assert.equal(await mes.getByRole('button').getAttribute('aria-expanded'),'false');
    assert.equal(await mes.locator('.monitoring-system-instances').isVisible(),false);
    const search=page.getByRole('searchbox',{name:'搜索业务系统或实例'});
    await search.fill('MES');
    assert.equal(await mes.getByRole('button').getAttribute('aria-expanded'),'true','搜索应展开命中分组');
    await search.fill('ERP');
    assert.equal(await page.locator('.monitoring-system-group').count(),1);
    assert.equal(await page.getByRole('radio',{name:'选择 生产 PostgreSQL',exact:true}).isVisible(),true);
    await search.fill('不存在的实例');
    assert.equal(await page.getByText('没有匹配的业务系统或实例。',{exact:true}).isVisible(),true);
    await search.fill('');
    assert.equal(await mes.getByRole('button').getAttribute('aria-expanded'),'false');
    await mes.getByRole('button').click();
    assert.equal(await page.getByRole('radio',{name:'选择 生产 Oracle',exact:true}).isChecked(),true);
    assert.equal(await page.evaluate(()=>fixtureRequests.length),requestCount,'搜索与折叠不得重新查询或改变观测对象');
  }
  for (const width of [1440,1280,1181,1024,900,821,820,420,320]) {
    for (const page of pages) {
      await page.setViewportSize({width,height:1000});
      const bounds=await page.evaluate(()=>{
        const rect=selector=>{const box=document.querySelector(selector).getBoundingClientRect();return {x:box.x,y:box.y,right:box.right,bottom:box.bottom,width:box.width,height:box.height};};
        const observation=document.querySelector('.monitoring-observation-card');
        return {toolbar:rect('.monitoring-query-toolbar'),rail:rect('.monitoring-context-rail'),main:rect('.monitoring-dashboard-main'),card:rect('.monitoring-observation-card'),summary:rect('.monitoring-summary-strip'),panels:rect('.monitoring-panels'),gap:Number.parseFloat(getComputedStyle(document.querySelector('.monitoring-dashboard-shell')).columnGap),panelGap:Number.parseFloat(getComputedStyle(document.querySelector('.monitoring-panels')).gap),mainGap:Number.parseFloat(getComputedStyle(document.querySelector('.monitoring-dashboard-main')).rowGap),merged:observation.contains(document.querySelector('.monitoring-query-toolbar'))&&observation.contains(document.querySelector('.monitoring-summary-strip')),fields:[...document.querySelectorAll('.monitoring-rail-fields .control-field')].map(node=>{const box=node.getBoundingClientRect();return {x:box.x,y:box.y,right:box.right};}),viewport:innerWidth,treeOverflow:getComputedStyle(document.querySelector('.monitoring-instance-tree')).overflowY};
      });
      assert.ok(bounds.merged,'观测设置和运行状态必须合并在同一张右侧卡片内');
      assert.ok(bounds.toolbar.bottom<=bounds.summary.y+1,'观测设置不得与运行状态重叠');
      assert.equal(bounds.gap,bounds.panelGap,'左右栏与指标卡片使用统一间距');
      assert.equal(bounds.mainGap,bounds.panelGap,'右侧概览与指标卡片使用统一间距');
      assert.ok(Math.abs(bounds.card.x-bounds.panels.x)<1,'合并卡片与指标卡片左边界对齐');
      assert.ok(Math.abs(bounds.panels.y-bounds.card.bottom-bounds.gap)<1,'合并卡片与指标区的间距必须统一');
      assert.ok(bounds.fields.every(field=>field.right<=bounds.toolbar.right+1),'观测设置不得横向溢出');
      assert.ok(bounds.toolbar.right<=bounds.viewport+1,'监控工作区不得超出视口');
      if(width>=1440) assert.ok(bounds.fields.every(field=>Math.abs(field.y-bounds.fields[0].y)<1),'宽屏设置必须横排');
      if(width>820) {
        assert.ok(Math.abs(bounds.rail.y-bounds.card.y)<1,'实例列表必须向上延伸，与右侧合并卡片顶部对齐');
        assert.ok(Math.abs(bounds.card.x-bounds.rail.right-bounds.gap)<1,'设置区必须缩进到右侧，保持统一栏间距');
      } else assert.ok(Math.abs(bounds.main.y-bounds.rail.bottom-bounds.gap)<1,'窄屏实例导航与观测区保持统一间距且不重叠');
      assert.equal(bounds.treeOverflow,'auto','实例目录必须独立滚动');
    }
    await compareRegions(kbot,reference,['.monitoring-observation-card','.monitoring-summary-strip','.monitoring-query-toolbar','.monitoring-rail-fields','.monitoring-instance-heading','.monitoring-instance-search','.monitoring-instance-tree','.monitoring-system-toggle','.monitoring-instance-option'],{page:'monitoring',width,state:'instance-tree'});
    if([1440,420].includes(width)) for(const [label,page] of [['kbot',kbot],['ammolite',reference]]) await page.screenshot({path:resolve(artifacts,`${label}-monitoring-instance-tree-${width}.png`)});
  }
  for (const page of pages) {
    await page.setViewportSize({width:1440,height:1000});
    await page.evaluate(()=>{globalThis.fixtureMonitoringHoldProfile=true;});
    await page.getByRole('radio',{name:'选择 生产 Oracle 备库',exact:true}).check();
    await page.waitForFunction(()=>typeof fixtureReleaseMonitoringProfile==='function');
    await page.getByRole('radio',{name:'选择 生产 PostgreSQL',exact:true}).check();
    await page.locator('.monitoring-profile-context dd').getByText('总览 instance-3',{exact:true}).waitFor();
    await page.evaluate(()=>{fixtureMonitoringHoldProfile=false;fixtureReleaseMonitoringProfile();});
    await page.evaluate(()=>new Promise(done=>requestAnimationFrame(()=>requestAnimationFrame(done))));
    assert.equal(await page.getByRole('radio',{name:'选择 生产 PostgreSQL',exact:true}).isChecked(),true);
    assert.equal(await page.locator('.monitoring-profile-context dd').textContent(),'总览 instance-3','旧实例的迟到响应不得覆盖当前 Profile');
    const viewRequest=await page.evaluate(()=>fixtureRequests.filter(item=>item.path.includes('/monitoring/view?')).at(-1));
    assert.ok(viewRequest.path.includes('/targets/instance-3/monitoring/view?'));
    assert.ok(viewRequest.path.includes('profile_id=profile-instance-3'));
    assert.ok((await page.locator('.monitoring-rail-footer a').getAttribute('href')).includes('instance-3'));
    await page.getByRole('combobox',{name:'时间窗口',exact:true}).selectOption('6h');
    await page.getByRole('combobox',{name:'基线',exact:true}).selectOption('1m');
    await page.getByRole('button',{name:'刷新',exact:true}).click();
    const settingsRequest=await page.evaluate(()=>fixtureRequests.filter(item=>item.path.includes('/monitoring/view?')).at(-1));
    assert.ok(settingsRequest.path.includes('/targets/instance-3/monitoring/view?'));
    assert.ok(settingsRequest.path.includes('window=6h'));
    assert.ok(settingsRequest.path.includes('baseline=1m'));
    await page.evaluate(()=>{fixtureMonitoringFailure=true;});
    await page.getByRole('combobox',{name:'监控源',exact:true}).selectOption('source-2');
    await page.getByRole('button',{name:'重新加载实例',exact:true}).waitFor();
    assert.equal(await page.locator('.monitoring-instance-option').count(),0,'加载失败不得继续展示旧来源实例');
    await page.evaluate(()=>{fixtureMonitoringFailure=false;});
    await page.getByRole('button',{name:'重新加载实例',exact:true}).click();
    await page.getByRole('radio',{name:'选择 生产 PostgreSQL',exact:true}).waitFor();
    assert.equal(await page.locator('.monitoring-instance-option').count(),1,'切换来源只展示该源返回的授权实例');
    await page.evaluate(()=>{fixtureMonitoringEmpty=true;});
    const beforeEmpty=await page.evaluate(()=>fixtureRequests.filter(item=>item.path.includes('/monitoring/view?')).length);
    await page.getByRole('combobox',{name:'监控源',exact:true}).selectOption('source-1');
    await page.getByText('当前监控源下暂无可访问实例。',{exact:true}).waitFor();
    assert.equal(await page.locator('.monitoring-instance-option').count(),0);
    assert.equal(await page.evaluate(()=>fixtureRequests.filter(item=>item.path.includes('/monitoring/view?')).length),beforeEmpty,'无授权实例时不得发起指标查询');
    await page.evaluate(()=>{fixtureMonitoringEmpty=false;fixtureMonitoringTree=false;});
    await page.getByRole('combobox',{name:'监控源',exact:true}).selectOption('source-1');
    await page.getByRole('radio',{name:'选择 生产 Oracle',exact:true}).waitFor();
    await page.locator('.monitoring-profile-context dd').getByText('数据库总览',{exact:true}).waitFor();
    await page.getByRole('combobox',{name:'时间窗口',exact:true}).selectOption('1h');
    await page.getByRole('combobox',{name:'基线',exact:true}).selectOption('1w');
  }
}
