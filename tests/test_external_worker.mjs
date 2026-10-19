import assert from 'node:assert/strict';import {readFileSync} from 'node:fs';import {predictExternal} from '../backend/external_engine.mjs';
const a=JSON.parse(readFileSync(new URL('../artifacts/external_model.json',import.meta.url)));
for(const p of a.profiles)assert.ok(Math.abs(predictExternal(p.values,a.model)-p.prediction)<1e-10);
const src=readFileSync(new URL('../dist/server/index.js',import.meta.url),'utf8');const {default:w}=await import('data:text/javascript;base64,'+Buffer.from(src).toString('base64'));
const summary=await(await w.fetch(new Request('https://edition.test/api/benchmark'))).json();assert.equal(summary.original_rows,1000316);assert.equal(summary.model,undefined);assert.equal(summary.profiles.length,20);assert.ok(summary.profiles.every(p=>p.label===null));
const send=values=>w.fetch(new Request('https://edition.test/api/benchmark/score',{method:'POST',body:JSON.stringify({values})}));
const r=await(await send(a.profiles[0].values)).json();assert.ok(Math.abs(r.probability-a.profiles[0].prediction)<1e-10);assert.equal((await send({})).status,400);assert.equal((await send([])).status,400);assert.equal((await send({...a.profiles[0].values,num_unq:'x'})).status,400);
console.log('External Worker checks passed: 20 artificial-profile parity cases, summary, live scoring and schema errors.');
