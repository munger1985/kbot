// 通用时分控件浏览器回归；隔离数据，不调用业务 API。
// node tests/contract/time_picker.mjs
import assert from 'node:assert/strict';
import { readFileSync, mkdtempSync } from 'node:fs';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const portal = resolve(process.env.AMMOLITE_ROOT || resolve(root, '../ammolite_cube'), 'apps/portal-web');
const require = createRequire(resolve(portal, 'package.json'));
const { chromium } = require('@playwright/test');
const { createServer } = await import(require.resolve('vite'));
const vue = (await import(require.resolve('@vitejs/plugin-vue'))).default;
const artifacts = mkdtempSync(resolve(tmpdir(), 'time-picker-browser-'));
for (const name of ['time-picker.js', 'time-picker.css']) {
  assert.equal(readFileSync(resolve(root, 'ui/shared', name), 'utf8'), readFileSync(resolve(portal, 'src/shared/ui', name), 'utf8'), `${name} 两项目必须保持一致`);
}
const entry = `
import {createApp,h,nextTick,reactive} from 'vue';
import TimePicker from '/src/shared/ui/TimePicker.vue';
import DateTimePicker from '/src/shared/ui/DateTimePicker.vue';
import {mountTimePicker, normalizeTime,isTimeInRange} from '/static/time-picker.js';
import '/src/shared/ui/styles.css';
const mode=new URLSearchParams(location.search).get('mode');
const state=reactive({value:'02:00',required:true,disabled:false,readonly:false,min:'',max:''});
if(mode==='static') {
  document.querySelector('#control').innerHTML='<span class="time-picker"><input class="time-picker-input" type="text" aria-label="执行时间" maxlength="5" value="02:00" required><button type="button" class="icon-button time-picker-trigger" aria-label="选择时间">◷</button></span>';
  const input=document.querySelector('input');
  const picker=mountTimePicker(document.querySelector('.time-picker'));
  globalThis.fixtureSet=async(value,props={})=>{input.value=value;for(const [key,v] of Object.entries(props)) {if(typeof v==='boolean') input.toggleAttribute(key,v);else input.setAttribute(key,v);}picker.sync();};
  globalThis.fixtureValue=()=>input.value;
  globalThis.fixtureDestroy=()=>picker.destroy();
} else {
  const app=createApp({render:()=>h(mode==='facade'?DateTimePicker:TimePicker,{...state,modelValue:state.value,type:mode==='facade'?'time':undefined,'aria-label':'执行时间','onUpdate:modelValue':v=>state.value=v})});
  app.mount('#control');
  globalThis.fixtureSet=async(value,props={})=>{state.value=value;Object.assign(state,props);await nextTick();await nextTick();};
  globalThis.fixtureValue=()=>state.value;
  globalThis.fixtureDestroy=()=>app.unmount();
}
globalThis.fixtureParsers={normalizeTime,isTimeInRange};
document.querySelector('#drawer').showModal();
globalThis.fixtureReady=true;`;
const server = await createServer({root:portal,configFile:false,cacheDir:resolve(artifacts,'vite-cache'),
  plugins:[vue(),{name:'time-picker-isolated',enforce:'pre',resolveId(id){if(id==='/fixture-time.js')return '\0time-fixture';if(id==='/static/time-picker.js')return '\0time-engine';},load(id){if(id==='\0time-fixture')return entry;if(id==='\0time-engine')return readFileSync(resolve(root,'ui/shared/time-picker.js'),'utf8');},configureServer(server){server.middlewares.use((req,res,next)=>{
    if(req.url.startsWith('/static/')) {const name=req.url.split('/').at(-1).split('?')[0];if(!['time-picker.js','time-picker.css'].includes(name))return next();res.setHeader('Content-Type',name.endsWith('.js')?'text/javascript':'text/css');res.end(readFileSync(resolve(root,'ui/shared',name)));return;}
    if(!req.url.startsWith('/time-fixture.html'))return next();
    res.setHeader('Content-Type','text/html; charset=utf-8');res.end('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><link rel="stylesheet" href="/static/time-picker.css"></head><body class="business-app-shell"><dialog id="drawer" style="box-sizing:border-box;position:fixed;inset:12px 12px 12px auto;width:min(520px,calc(100vw - 24px));max-height:calc(100vh - 24px);padding:24px;overflow:auto;background:var(--surface);border:1px solid var(--line)"><form><div style="height:130px">通用时间控件</div><div id="control"></div><button id="outside" class="secondary-button" type="button">其他字段</button><button class="primary-button" type="submit">保存</button><div style="height:600px"></div></form></dialog><script type="module" src="/fixture-time.js"></script></body></html>');
  });}}],resolve:{alias:{'@':resolve(portal,'src')}},server:{host:'127.0.0.1',port:0},logLevel:'error'});
let browser;
try {
  await server.listen();
  browser=await chromium.launch({headless:true});
  const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
  for(const mode of ['static','vue','facade','fallback']) {
    const page=await browser.newPage({viewport:{width:1000,height:800}}),errors=[];
    page.setDefaultTimeout(15000);
    if(mode==='fallback') await page.addInitScript(()=>{HTMLElement.prototype.showPopover=undefined;HTMLElement.prototype.hidePopover=undefined;});
    page.on('pageerror',error=>errors.push(error.message));
    await page.goto(`${origin}/time-fixture.html?mode=${mode}`);
    await page.waitForFunction(()=>globalThis.fixtureReady);
    const input=page.locator('.time-picker-input'),trigger=page.getByRole('button',{name:'选择时间',exact:true}),panel=page.getByRole('dialog',{name:'选择时间',exact:true});
    const value=()=>page.evaluate(()=>fixtureValue());
    const set=(value,props={})=>page.evaluate(({value,props})=>fixtureSet(value,props),{value,props});
    async function select(hour,minute) {await trigger.click();await page.getByRole('listbox',{name:'小时',exact:true}).getByRole('option',{name:hour,exact:true}).click();await page.getByRole('listbox',{name:'分钟',exact:true}).getByRole('option',{name:minute,exact:true}).click();}
    await trigger.click();assert.equal(await page.getByRole('listbox',{name:'小时',exact:true}).getByRole('option').count(),24);assert.equal(await page.getByRole('listbox',{name:'分钟',exact:true}).getByRole('option').count(),60);
    await page.keyboard.press('Escape');assert.equal(await panel.isVisible(),false);assert.equal(await page.locator('#drawer').evaluate(n=>n.open),true);
    for(const time of ['00:00','23:59','04:17']) {await select(...time.split(':'));assert.notEqual(await value(),time);await panel.getByRole('button',{name:'确定',exact:true}).click();assert.equal(await value(),time);assert.equal(await panel.isVisible(),false);}
    await select('11','29');await panel.getByRole('button',{name:'取消',exact:true}).click();assert.equal(await value(),'04:17');
    await input.fill('2:31');await page.locator('#outside').click();assert.equal(await value(),'02:31');
    await input.fill('0407');await page.locator('#outside').click();assert.equal(await value(),'04:07');
    for(const invalid of ['24:00','02:60','12:3','aa:bb']) {await input.fill(invalid);assert.equal(await input.evaluate(n=>n.checkValidity()),false);}
    await input.fill('');assert.equal(await input.evaluate(n=>n.checkValidity()),false);await set('',{required:false});assert.equal(await input.evaluate(n=>n.checkValidity()),true);
    await set('09:05',{required:true});await input.focus();await page.keyboard.press('ArrowDown');await page.keyboard.press('End');await page.keyboard.press('ArrowRight');await page.keyboard.press('End');await page.keyboard.press('Enter');assert.equal(await value(),'23:59');
    await trigger.click();await set('04:17');assert.equal(await panel.locator('.time-picker-preview').innerText(),'04:17');await page.locator('#outside').click();assert.equal(await panel.isVisible(),false);
    await trigger.click();await panel.getByRole('button',{name:'确定',exact:true}).focus();await page.keyboard.press('Tab');assert.equal(await panel.isVisible(),false);
    await set('09:05',{min:'09:00',max:'10:30'});await trigger.click();assert.equal(await page.getByRole('listbox',{name:'小时',exact:true}).getByRole('option',{name:'08',exact:true}).getAttribute('aria-disabled'),'true');await page.getByRole('listbox',{name:'小时',exact:true}).getByRole('option',{name:'10',exact:true}).click();assert.equal(await page.getByRole('listbox',{name:'分钟',exact:true}).getByRole('option',{name:'31',exact:true}).getAttribute('aria-disabled'),'true');await page.keyboard.press('Escape');
    await set('23:59',{min:'22:00',max:'02:00'});assert.equal(await input.evaluate(n=>n.checkValidity()),true);await set('12:00');assert.equal(await input.evaluate(n=>n.checkValidity()),false);
    await set('04:17',{min:'',max:'',disabled:true});assert.equal(await trigger.isDisabled(),true);assert.equal(await input.isDisabled(),true);await set('04:17',{disabled:false,readonly:true});assert.equal(await trigger.isDisabled(),true);await set('04:17',{readonly:false});
    for(const size of [{width:1000,height:800},{width:320,height:400}]) {
      await page.setViewportSize(size);await trigger.click();const box=await panel.boundingBox();assert.ok(box.x>=0&&box.y>=0&&box.x+box.width<=size.width&&box.y+box.height<=size.height,`${mode} 浮层越界`);
      const centered=await page.getByRole('listbox',{name:'分钟',exact:true}).evaluate(list=>{const a=list.querySelector('[aria-selected="true"]').getBoundingClientRect(),b=list.getBoundingClientRect();return Math.abs(a.y+a.height/2-b.y-b.height/2)<3;});assert.equal(centered,true);
      if(size.width===1000) await page.screenshot({path:resolve(artifacts,`${mode}-popover.png`)});
      await page.keyboard.press('Escape');
    }
    await trigger.click();await page.locator('#drawer').evaluate(n=>n.close());assert.equal(await panel.isVisible(),false);await page.locator('#drawer').evaluate(n=>n.showModal());
    await trigger.click();await page.evaluate(()=>fixtureDestroy());assert.equal(await page.locator('.time-picker-panel').count(),0);
    assert.deepEqual(errors,[]);await page.close();console.log(`${mode}：时分选择、任意分钟、输入校验、键盘、范围、禁用和浮层生命周期通过`);
  }
  console.log(`截图：${artifacts}`);
} finally {await browser?.close();await server.close();}
