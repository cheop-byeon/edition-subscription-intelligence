import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const html=readFileSync(new URL('../frontend/index.html',import.meta.url),'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.ok(scripts.length>0);
for(const [,source] of scripts)new vm.Script(source);
assert.ok(html.includes('/api/benchmark/score'));
console.log('Frontend JavaScript syntax and API integration entrypoint checked.');
