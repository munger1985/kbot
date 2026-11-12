// 使用真实父页面和 Exadata 子组件，验证合并控件、资源切换与普通观测设置；仅替换 API 数据。
import assert from 'node:assert/strict'
import {existsSync,readFileSync,mkdtempSync} from 'node:fs'
import {createRequire} from 'node:module'
import {createServer as createNetServer} from 'node:net'
import {resolve} from 'node:path'
import {fileURLToPath,pathToFileURL} from 'node:url'
const root=fileURLToPath(new URL('../../',import.meta.url)),kbot=existsSync(resolve(root,'ui/aiops/monitoring.html'))
const portal=resolve(kbot?process.env.AMMOLITE_ROOT||resolve(root,'../ammolite_cube'):root,'apps/portal-web')
const require=createRequire(resolve(portal,'package.json')),{chromium,expect}=require('@playwright/test')
const {createServer}=await import(pathToFileURL(resolve(portal,'node_modules/vite/dist/node/index.js')))
const read=path=>readFileSync(path,'utf8'),artifacts=mkdtempSync('/tmp/exadata-observation-')
const probe=createNetServer()
await new Promise(resolve=>probe.listen(0,'127.0.0.1',resolve))
const freePort=probe.address().port
await new Promise(resolve=>probe.close(resolve))
const server=await createServer({root:portal,configFile:resolve(portal,'vite.config.ts'),server:{host:'127.0.0.1',port:freePort},plugins:[{name:'exadata-observation-fixture',configureServer(server){
 server.middlewares.use(async(req,res,next)=>{
  const path=req.url?.split('?')[0]
  if(kbot&&/^\/__kbot\/(css|js)\/[\w.-]+$/.test(path)){
   res.setHeader('Content-Type',path.endsWith('.css')?'text/css':'text/javascript');res.end(read(resolve(root,'ui/aiops',path.slice('/__kbot/'.length))));return
  }
  if(path==='/__runtime-config.js'){res.end('window.KBOT_UI_CONFIG={mainApiBaseUrl:location.origin}');return}
  if(kbot&&path==='/vendor/echarts.min.js'){res.end(read(resolve(root,'ui/vendor/echarts.min.js')));return}
  if(path!=='/__observation')return next()
  const html=kbot?read(resolve(root,'ui/aiops/monitoring.html')).replace('<body data-page="monitoring">','<body data-page="monitoring" class="aiops-static-page">').replace('<head>','<head><base href="/__kbot/">').replace('../runtime-config.js','/__runtime-config.js').replace(/<script src="\.\/js\/aiops-shell.js[^>]*><\/script>/,'<script>window.KBotAIOpsShell={ready:Promise.resolve(),escape:value=>String(value??""),fmt:value=>value||"",toast:()=>{}};sessionStorage.setItem("kbot.aiops.session.v1",JSON.stringify({access_token:"fixture-token"}));</script>')
   :`<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body><div id="app"></div><script type="module">import {createApp,h} from 'vue';import {createRouter,createMemoryHistory,RouterView} from 'vue-router';import Component from '/src/features/aiops/business/AIOpsMonitoringView.vue';import '/src/shared/ui/styles.css';const router=createRouter({history:createMemoryHistory(),routes:[{path:'/monitoring/:domainCode',component:Component}]});await router.push('/monitoring/test?domainId=domain&targetId=exadata&businessSystemId=system');await router.isReady();createApp({render:()=>h(RouterView)}).use(router).mount('#app');</script></body></html>`
  res.setHeader('Content-Type','text/html; charset=utf-8');res.end(await server.transformIndexHtml(req.url,html))
 })
}}]})
await server.listen()
const origin=`http://127.0.0.1:${server.httpServer.address().port}`,browser=await chromium.launch({headless:true})
try{
 const page=await browser.newPage({viewport:{width:1440,height:1000}}),errors=[],requests=[]
 page.setDefaultTimeout(30000)
 page.on('console',message=>{if(message.type()==='error')console.log('浏览器错误：'+message.text().slice(0,400))})
 page.on('pageerror',error=>errors.push(error.message))
 await page.route('https://fonts.**',route=>route.abort())
 const source=(id,type)=>({source_id:id,display_name:id,source_type:type,monitoring_readiness:'READY',diagnostic_readiness:'READY',capability_gaps:[],diagnostic_gaps:[]})
 const oem=source('OEM','OEM'),prom=source('PROMETHEUS','PROMETHEUS')
 const resource=(id,type,sources)=>({instance_id:id,resource_id:id,display_name:id,resource_type_key:type,db_type:type==='DATABASE'?'ORACLE':null,business_system_id:'system',business_system_name:'测试业务系统',sources,status:'AVAILABLE',monitoring_readiness:'READY',diagnostic_readiness:'READY',capability_gaps:[],diagnostic_gaps:[]})
 const db=resource('普通数据库','DATABASE',[oem,prom]),exa=resource('exadata','EXADATA',[oem])
 const profile={profile_id:'database-overview',version:'1.0.0',display_name:'数据库总览',description:'',purpose:'OVERVIEW',is_default:true,metric_codes:[],supported_db_types:['ORACLE'],supported_database_versions:{},resolved_variant_ids:[],capacity_warning_percent:80,capacity_critical_percent:90}
 await page.route('**/api/v1/apps/aiops/**',async route=>{
  const request=route.request(),url=new URL(request.url()),body=request.postDataJSON(),path=url.pathname
  requests.push({path,body})
  let result
  if(path.endsWith('/monitoring/business-systems'))result=[{business_system_id:'system',display_name:'测试业务系统',resources:[exa,db]}]
  else if(path.includes('/exadata/')&&path.endsWith('/view'))result={resource_id:'exadata',binding_id:'binding',binding_version:1,source_version:1,catalog_sha256:'a'.repeat(64),query_mode:body.mode,members:[{target_id:'cell-a',target_type:'oracle_exadata',display_name:'测试 Cell'}],definitions:[],samples:[],gaps:[],total_queries:0,next_offset:null,catalog_version:'exadata-oem-v1',captured_at:'2026-11-12T09:00:00Z'}
  else if(path.endsWith('/metrics'))result={overview_panels:[],discovered_count:0,truncated:false}
  else if(path.endsWith('/profiles'))result=[profile]
  else if(path.endsWith('/monitoring/view'))result={schema_version:'aiops.public.v1',generated_at:'2026-11-12T09:00:00Z',refresh_after_seconds:30,source:prom,profile,window:{start:'2026-11-12T08:00:00Z',end:'2026-11-12T09:00:00Z',name:'1h'},instances:[db],panels:[],gaps:[],partial:false}
  else throw new Error('未配置的接口：'+path)
  await route.fulfill({json:result})
 })
 await page.goto(origin+'/__observation?target_id=exadata&businessSystemId=system',{waitUntil:'networkidle'})
 const observation=page.getByRole('region',{name:'观测设置与运行状态',exact:true})
 await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy','false',{timeout:30000}).catch(async error=>{console.log(JSON.stringify({errors,requests,text:(await page.locator('body').innerText()).slice(0,1500)}));throw error})
 await expect(observation.locator('select')).toHaveCount(kbot?7:5)
 for(const name of ['资源监控源','自动刷新','查询模式','设备范围','均值偏离'])await expect(observation.getByRole('combobox',{name,exact:true})).toBeVisible()
 const labelStyles=await observation.locator('.field-label').evaluateAll(nodes=>nodes.filter(node=>node.offsetParent!==null).map(node=>{const css=getComputedStyle(node);return {text:node.textContent,color:css.color,fontSize:css.fontSize,fontWeight:css.fontWeight}}))
 assert.ok(labelStyles.every(style=>style.color===labelStyles[0].color&&style.fontSize===labelStyles[0].fontSize),JSON.stringify(labelStyles))
 await expect(observation.getByRole('combobox',{name:'时间窗口',exact:true})).toBeHidden()
 await expect(page.locator('.exadata-heading,.exadata-directory')).toHaveCount(0)
 await expect(page.locator('.exadata-oem [data-mode]')).toHaveCount(0)
 await observation.locator('[data-mode]').selectOption('HISTORY')
 await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy','false')
 assert.equal(requests.at(-1).body.mode,'HISTORY')
 await observation.locator('[data-component]').selectOption('cell-a')
 await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy','false')
 assert.equal(requests.at(-1).body.component_id,'cell-a')
 for(const width of [1920,1440,1000,780,390]){
  await page.setViewportSize({width,height:1000})
  assert.ok(await observation.evaluate(node=>node.scrollWidth-node.clientWidth<=1),'合并后的观测设置不能横向溢出')
  await page.screenshot({path:resolve(artifacts,`${kbot?'kbot':'ammolite'}-${width}.png`),fullPage:true})
 }
 await page.setViewportSize({width:1440,height:1000})
 await page.getByRole('radio',{name:'选择 普通数据库',exact:true}).check()
 await expect(page.locator('.oem-overview')).toBeVisible()
 await expect(page.locator('[data-mode],[data-component],[data-baseline],[data-diagnose]')).toHaveCount(0)
 for(const name of ['资源监控源','自动刷新','时间窗口','基线'])await expect(observation.getByRole('combobox',{name,exact:true})).toBeVisible()
 await observation.getByRole('combobox',{name:'资源监控源',exact:true}).selectOption('PROMETHEUS')
 await expect(page.locator('.monitoring-no-panels')).toBeVisible()
 for(const name of ['资源监控源','自动刷新','时间窗口','基线'])await expect(observation.getByRole('combobox',{name,exact:true})).toBeVisible()
 await expect(observation.getByRole('combobox',{name:'基线',exact:true}).locator('option[value="1d"]')).toHaveCount(1)
 await page.getByRole('radio',{name:'选择 exadata',exact:true}).check()
 await expect(observation.locator('[data-mode]')).toHaveValue('LATEST_SNAPSHOT')
 await expect(observation.locator('[data-mode]')).toHaveCount(1)
 await expect(page.locator('.exadata-oem')).toHaveAttribute('aria-busy','false')
 assert.deepEqual(errors,[])
 console.log(JSON.stringify({ok:true,repository:kbot?'kbot':'ammolite',parent:'real',child:'real',switching:['EXADATA_OEM','DATABASE_OEM','DATABASE_PROMETHEUS','EXADATA_OEM'],artifacts}))
}finally{await browser.close();await server.close()}
