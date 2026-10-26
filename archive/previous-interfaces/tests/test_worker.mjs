import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';
const source=readFileSync(new URL('../dist/server/index.js',import.meta.url),'utf8');
const {default:worker}=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const get=path=>worker.fetch(new Request('https://edition.test'+path));
assert.equal((await get('/')).status,200);assert.equal((await (await get('/api/health')).json()).backend,'Worker');
assert.equal((await get('/api/report?brand=invalid')).status,400);
for(const b of ['all','VG','Aftenposten','Bergens Tidende','E24']){const r=await (await get('/api/report?brand='+encodeURIComponent(b))).json();assert.ok(r.n>1000);assert.ok(r.policy_value.lower<r.policy_value.upper)}
const post=body=>worker.fetch(new Request('https://edition.test/api/decision',{method:'POST',body:JSON.stringify(body)}));
assert.equal((await post({articles:-1})).status,400);assert.equal((await post([])).status,400);
assert.equal((await post({premium_reads:30,articles:3})).status,400);
assert.equal((await post({active_days:2.5})).status,400);
assert.equal((await post({consent:'false'})).status,400);
const d=await(await post({subscribed:true})).json();assert.equal(d.action.id,'open');assert.ok(d.guardrails.length);
assert.equal((await worker.fetch(new Request('https://edition.test/api/decision',{method:'POST',body:'not json'}))).status,400);
assert.equal((await worker.fetch(new Request('https://edition.test/api/decision',{method:'POST',body:' '.repeat(17000)}))).status,413);
assert.equal((await worker.fetch(new Request('https://edition.test/api/decision',{method:'DELETE'}))).status,405);
console.log('Worker API checks passed: routes, brand reports, inference, validation, guards, size and method errors.');
