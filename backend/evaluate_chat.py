"""Live local LLM checks; deterministic checks are limited proxies for factuality."""
import argparse,json,re,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def fetch(base,path,payload=None):
    req=urllib.request.Request(base+'/api/commercial/'+path,data=None if payload is None else json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=65) as response:return json.load(response)
def evaluate(base):
    results=[]
    for brand in ['all','VG','Aftenposten']:
        summary=fetch(base,'summary?brand='+brand)
        question='For the selected brand scope only, how many reader profiles and eligible contacts are there? Use the exact counts and mention that data is simulated.'
        reply=fetch(base,'chat',{'brand':brand,'question':question});answer=reply['answer'];numbers={int(n) for n in re.findall(r'\b\d+\b',answer)}
        passed=reply['status']=='ok' and summary['readers'] in numbers and summary['eligible'] in numbers and 'simulat' in answer.lower()
        results.append({'case':'brand_counts','brand':brand,'expected':{'readers':summary['readers'],'eligible':summary['eligible']},'passed':passed,'response':reply})
    reply=fetch(base,'chat',{'brand':'all','question':'What was the actual causal revenue uplift on real Schibsted customers? If unavailable, explicitly say unavailable or cannot determine.'})
    text=reply['answer'].lower();passed=reply['status']=='ok' and any(s in text for s in ['unavailable','cannot determine','no real','not available','do not have','does not contain'])
    results.append({'case':'missing_evidence','passed':passed,'response':reply})
    return {'checks':results,'passed':sum(r['passed'] for r in results),'total':len(results),'method':'Exact-count and phrase assertions on live LLM responses; not a comprehensive factuality or safety evaluation.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:4173');p.add_argument('--output',type=Path,default=ROOT/'data/chat_evaluation.json');a=p.parse_args();r=evaluate(a.base);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r));raise SystemExit(0 if r['passed']==r['total'] else 1)
