// 在真实双端页面验证上传合同；所有文件与回执均为隔离数据。
import assert from 'node:assert/strict';

export async function verifyOperationsKnowledgeUpload(page) {
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>{globalThis.fixtureKnowledgeUploads=true;});
  const drawer=page.locator('.knowledge-upload-drawer');
  const manual={name:'manual.md',mimeType:'text/markdown',buffer:Buffer.from('# Manual')};
  const appendix={name:'appendix.txt',mimeType:'text/plain',buffer:Buffer.from('Appendix')};
  for (const grouped of [true,false]) {
    await page.getByRole('button',{name:'上传资料',exact:true}).click();
    await drawer.waitFor({state:'visible'});
    assert.doesNotMatch(await drawer.innerText(),/Bundle|KC 解析/);
    assert.equal(await drawer.locator('input[value="EACH_FILE"]').isChecked(),true);
    if(grouped) await drawer.locator('input[value="SINGLE_ASSET"]').check();
    assert.equal(await drawer.locator('input[type="file"]').getAttribute('multiple')!==null,grouped);
    await drawer.locator('input[type="file"]').setInputFiles(manual);
    await drawer.locator('input[type="file"]').setInputFiles(appendix);
    assert.equal(await drawer.locator('.upload-file-row').count(),grouped?2:1);
    if(grouped) {
      await drawer.locator('.upload-file-row').first().locator('input[type="checkbox"]').check();
      await drawer.locator('.upload-file-row').nth(1).locator('select').selectOption('SUPPLEMENT');
    }
    await drawer.getByLabel('资料名称',{exact:true}).fill('数据库运维资料');
    await drawer.getByRole('button',{name:'上传',exact:true}).click();
    await drawer.waitFor({state:'hidden'});
    const payload=await page.evaluate(async()=>{
      const request=fixtureRequests.filter(row=>row.path.endsWith('/operations-knowledge/uploads')).at(-1);
      const body=request.body;
      const files=JSON.parse(body.get('files'));
      return {
        grouping:body.get('grouping_mode'),
        keys:[...body.keys()],
        metadata:JSON.parse(body.get('metadata')),
        headers:request.headers,
        files:await Promise.all(files.map(async file=>({
          ...file,content:await body.get(file.part_name).text(),
        }))),
      };
    });
    assert.equal(payload.grouping,grouped?'SINGLE_ASSET':'EACH_FILE');
    assert.equal(payload.metadata.display_name,'数据库运维资料');
    assert.equal(payload.metadata.security_level,1);
    assert.ok(payload.headers['Idempotency-Key']);
    assert.equal(payload.keys.includes('bundle'),false);
    assert.deepEqual(
      payload.files.map(file=>[file.display_name,file.ordinal,file.role,file.required_flag,file.content]),
      grouped?[
        ['manual.md',0,'CONTENT',true,'# Manual'],
        ['appendix.txt',1,'SUPPLEMENT',false,'Appendix'],
      ]:[['appendix.txt',0,'CONTENT',false,'Appendix']],
    );
    for(const file of payload.files) assert.match(file.content_sha256,/^[a-f0-9]{64}$/);
  }
  assert.deepEqual(await page.evaluate(()=>fixtureFailures),[]);
}
