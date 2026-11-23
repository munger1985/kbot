// 在真实双端页面验证上传合同；所有文件与回执均为隔离数据。
import assert from 'node:assert/strict';

export async function verifyOperationsKnowledgeUpload(page) {
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>{globalThis.fixtureKnowledgeUploads=true;});
  const drawer=page.locator('.knowledge-upload-drawer');
  const manual={name:'manual.md',mimeType:'text/markdown',buffer:Buffer.from('# Manual')};
  const appendix={name:'appendix.txt',mimeType:'text/plain',buffer:Buffer.from('Appendix')};
  await page.getByRole('button',{name:'上传资料',exact:true}).click();
  await drawer.waitFor({state:'visible'});
  assert.doesNotMatch(await drawer.innerText(),/Bundle|选择组织方式|多文件资料/);
  const picker=drawer.locator('input[type="file"]');
  assert.notEqual(await picker.getAttribute('multiple'),null);
  await picker.setInputFiles(manual);
  await picker.setInputFiles(appendix);
  assert.equal(await drawer.locator('.upload-file-row').count(),2);
  await drawer.getByLabel('资料名称',{exact:true}).first().fill('数据库运维资料');
  await page.evaluate(()=>{globalThis.fixtureRejectAppendix=true;});
  await drawer.getByRole('button',{name:'上传',exact:true}).click();
  await drawer.getByText('隔离拒绝').waitFor();
  await drawer.getByText('已接收，等待审批',{exact:true}).waitFor();
  await drawer.getByRole('button',{name:'上传',exact:true}).click();
  await drawer.waitFor({state:'hidden'});
  const payloads=await page.evaluate(async()=>Promise.all(
    fixtureRequests.filter(row=>row.path.endsWith('/operations-knowledge/uploads')).slice(-3).map(async request=>{
      const body=request.body, files=JSON.parse(body.get('files'));
      return {grouping:body.get('grouping_mode'),metadata:JSON.parse(body.get('metadata')),
        key:new Headers(request.headers).get('idempotency-key'),
        files:await Promise.all(files.map(async file=>({...file,content:await body.get(file.part_name).text()})))};
    })
  ));
  assert.equal(payloads.length,3);
  assert.deepEqual(payloads.map(row=>row.files.map(file=>file.content)),[['# Manual'],['Appendix'],['Appendix']]);
  assert.deepEqual(payloads.map(row=>row.metadata.display_name),['数据库运维资料','appendix','appendix']);
  assert.ok(payloads[0].key && payloads[1].key);
  assert.notEqual(payloads[0].key,payloads[1].key);
  assert.equal(payloads[1].key,payloads[2].key);
  for(const payload of payloads) {
    assert.equal(payload.grouping,null);
    assert.equal(payload.metadata.security_level,1);
    assert.match(payload.files[0].content_sha256,/^[a-f0-9]{64}$/);
    assert.equal('role' in payload.files[0],false);
  }
  assert.deepEqual(await page.evaluate(()=>fixtureFailures),[]);
}
