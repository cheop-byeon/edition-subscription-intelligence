import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
const html=readFileSync(new URL('../frontend/index.html',import.meta.url),'utf8');
const scripts=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];assert.ok(scripts.length>0);
for(const [,source] of scripts)new vm.Script(source);
assert.ok(html.includes('/api/benchmark/score'));
console.log('Frontend JavaScript syntax and API integration entrypoint checked.');

const commercial=readFileSync(new URL('../frontend/commercial.html',import.meta.url),'utf8');
for(const [,source] of commercial.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(source);
assert.ok(commercial.includes('/api/commercial/'));

const article=readFileSync(new URL('../frontend/article.html',import.meta.url),'utf8');
for(const [,source] of article.matchAll(/<script>([\s\S]*?)<\/script>/g))new vm.Script(source);
