// 运行详情的隔离 DOM 与定时器；所有测试请求均在内存中处理。
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {runInNewContext} from 'node:vm';

export function createRunFixture(run,result) {
  const file=fileURLToPath(new URL('../../ui/aiops/js/aiops-runs.js',import.meta.url));
  const source=readFileSync(file,'utf8'), instrumented=source.replace(/\}\)\(\);\s*$/,'globalThis.runTest={state,open,close,renderDetail,cancelObservation,settleConfirmation,observationHtml};})();');
  assert.notEqual(source,instrumented);
  const nodes=new Map(),calls=[],timers=new Map();let timerId=0;
  const node=id=>{
    if(!nodes.has(id)) nodes.set(id,{innerHTML:'',textContent:'',open:false,hidden:false,disabled:false,scrollTop:0,showModal(){this.open=true;},close(){this.open=false;},focus(){},querySelector(){return node(`${id}-button`);}});
    return nodes.get(id);
  };
  const escape=value=>String(value??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
  const context={document:{body:{dataset:{page:'runs'}},getElementById:node},KBotAIOpsShell:{escape,fmt:String,toast(){},ready:new Promise(()=>{})},KBotAIOpsTurnResult:{html:()=>'<p>诊断结果</p>',mount(){}},KBotAIOpsAuth:{uuid:()=>`command-${calls.length}`,async request(path,init={}){calls.push({path,init});if(path.endsWith('/commands')) {run.status='CANCELLED';return {};};return path.endsWith('/result')?result:run;}},URLSearchParams,location:{search:'',href:''},setTimeout(callback){const id=++timerId;timers.set(id,callback);return id;},clearTimeout(id){timers.delete(id);}};
  context.window=context;runInNewContext(instrumented,context,{filename:file});
  return {...context.runTest,node,calls,timers,escape,async tick(){assert.equal(timers.size,1);const [id,callback]=timers.entries().next().value;timers.delete(id);await callback();}};
}
