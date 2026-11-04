// 实例导航与顶部观测设置的隔离交互验收，不访问真实监控源。
import assert from 'node:assert/strict';
import {resolve} from 'node:path';

export async function verifyMonitoringNavigation(kbot, reference, compareRegions, artifacts) {
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
