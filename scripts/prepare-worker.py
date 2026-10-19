"""Bundle the frontend, portable inference and frozen model into the Worker entry."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
engine=(ROOT/'backend/engine.mjs').read_text().replace('export ','')
html=json.dumps((ROOT/'frontend/index.html').read_text(),ensure_ascii=False)
artifact=json.dumps(json.loads((ROOT/'artifacts/model.json').read_text()),separators=(',',':'))
research=json.dumps(json.loads((ROOT/'research/sources.json').read_text()),separators=(',',':'))
external_path=ROOT/'artifacts/external_model.json'
external=json.dumps(json.loads(external_path.read_text()),separators=(',',':')) if external_path.exists() else 'null'
external_engine=(ROOT/'backend/external_engine.mjs').read_text().replace('export ','')
handler=r'''
const json=(body,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
export default {async fetch(request){
 const u=new URL(request.url),p=u.pathname;
 if(request.method==='GET'){
  if(p==='/')return new Response(HTML,{headers:{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store','X-Content-Type-Options':'nosniff'}});
  if(p==='/favicon.ico')return new Response(null,{status:204});
  if(p==='/api/health')return json({status:'ok',backend:'Worker',model_version:MODEL.version,data_mode:'synthetic'});
  if(p==='/api/benchmark'){if(!EXTERNAL)return json(null);const {model,...summary}=EXTERNAL;return json(summary)}
  if(p==='/api/research')return json(RESEARCH);
  if(p==='/api/model'){const {reports,...model}=MODEL;return json(model)}
  if(p==='/api/report'){const b=u.searchParams.get('brand')||'all';if(!['all',...BRANDS].includes(b))return json({error:'Unsupported brand.'},400);return json(MODEL.reports[b])}
  return json({error:'Route not found.'},404);
 }
 if(request.method==='POST'&&['/api/decision','/api/benchmark/score'].includes(p)){
  if(Number(request.headers.get('Content-Length'))>16384)return json({error:'Request exceeds 16 KB.'},413);
  try{const body=await request.text();if(new TextEncoder().encode(body).length>16384)return json({error:'Request exceeds 16 KB.'},413);const data=JSON.parse(body);if(p==='/api/benchmark/score'){if(!EXTERNAL)return json({error:'External model is not trained.'},503);if(!data||typeof data!=='object'||Array.isArray(data)||Object.keys(data).length!==1||!Object.hasOwn(data,'values'))return json({error:'Expected an object containing values.'},400);return json(scoreExternal(data.values,EXTERNAL))}return json(decision(data,MODEL))}catch(e){return json({error:e.message},400)}
 }
 return json({error:'Method not allowed.'},405);
}};
'''
(ROOT/'worker/index.js').write_text(engine+'\n'+external_engine+'\nconst EXTERNAL='+external+';\nconst HTML='+html+';\nconst MODEL='+artifact+';\nconst RESEARCH='+research+';\n'+handler)
print('Bundled frontend, inference engine, model, and research sources.')
