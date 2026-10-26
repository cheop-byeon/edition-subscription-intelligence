"""Local Python backend and frontend server. Inference needs only the standard library."""
import argparse, json, sys, os
from database import Store
from commercial import Commercial
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, urlencode
from engine import decision, BRANDS
from external_engine import score as benchmark_score
ROOT=Path(__file__).resolve().parents[1]
MODEL=json.loads((ROOT/'artifacts/model.json').read_text())
RESEARCH=json.loads((ROOT/'research/sources.json').read_text())
EXTERNAL_PATH=ROOT/'artifacts/external_model.json'
EXTERNAL=json.loads(EXTERNAL_PATH.read_text()) if EXTERNAL_PATH.exists() else None
LOCAL_PROFILES=ROOT/'data/local_benchmark_profiles.json'
if EXTERNAL and LOCAL_PROFILES.exists():
    EXTERNAL.update(json.loads(LOCAL_PROFILES.read_text()))
    EXTERNAL['profile_origin']='heldout_local'
STORE=None
def get_store():
    global STORE
    if STORE is None:
        STORE=Store(os.environ.get('EDITION_DB',str(ROOT/'data/edition.sqlite3')))
        if EXTERNAL:STORE.seed(EXTERNAL['profiles'],EXTERNAL.get('profile_origin','artificial_input'))
    return STORE
class Handler(BaseHTTPRequestHandler):
    def send(self,status,body,kind='application/json; charset=utf-8'):
        body=body.encode() if isinstance(body,str) else body
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
    def obj(self,status,value): self.send(status,json.dumps(value,allow_nan=False))
    def do_GET(self):
        u=urlparse(self.path);p=u.path
        if p in ['/','/commercial','/article']:
            location=os.environ.get('EDITION_UI_URL','http://127.0.0.1:8501/')
            if p=='/article':location=location.rstrip('/')+'/?'+urlencode({'page':'Paywall'})
            self.send_response(307);self.send_header('Location',location);self.send_header('Content-Length','0');self.end_headers();return
        if p=='/favicon.ico':return self.send(204,b'','image/x-icon')
        if p=='/api/health':return self.obj(200,{'status':'ok','backend':'Python','application':'Edition','model_version':Commercial(get_store()).artifact()['version'],'data_mode':'simulated commercial lifecycle data'})
        if p.startswith('/api/commercial/'):
            try:
                app=Commercial(get_store());q=parse_qs(u.query);brand=q.get('brand',['all'])[0]
                if p=='/api/commercial/summary':return self.obj(200,app.summary(brand))
                if p=='/api/commercial/readers':return self.obj(200,app.readers(brand,q.get('kind',[None])[0],float(q.get('threshold',[0])[0])))
                if p=='/api/commercial/export':return self.send(200,app.export(int(q.get('id',['0'])[0])),'text/csv; charset=utf-8')
            except (ValueError,TypeError) as exc:return self.obj(400,{'error':str(exc)})
        if p=='/api/storage':return self.obj(200,get_store().summary())
        if p=='/api/benchmark':
            if not EXTERNAL:return self.obj(200,None)
            result={k:v for k,v in EXTERNAL.items() if k!='model'}
            profiles=get_store().profiles()
            for profile in profiles:profile['prediction']=benchmark_score(profile['values'],EXTERNAL)['probability']
            result['profiles']=profiles
            result['storage']=get_store().summary()
            if profiles and profiles[0]['source']=='downloaded_benchmark':
                result['profile_origin']='downloaded_local'
                result['sample_note']='Stored downloaded benchmark records. These may include training records; use the frozen test metrics above for evaluation.'
            return self.obj(200,result)
        if p=='/api/research':return self.obj(200,RESEARCH)
        if p=='/api/model':return self.obj(200,{k:v for k,v in MODEL.items() if k!='reports'})
        if p=='/api/report':
            b=parse_qs(u.query).get('brand',['all'])[0]
            if b not in ['all']+BRANDS:return self.obj(400,{'error':'Unsupported brand.'})
            return self.obj(200,MODEL['reports'][b])
        return self.obj(404,{'error':'Route not found.'})
    do_HEAD=do_GET
    def do_POST(self):
        path=urlparse(self.path).path
        if not path.startswith('/api/commercial/') and path not in ['/api/decision','/api/benchmark/score']:return self.obj(404,{'error':'Route not found.'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>(1000000 if path=='/api/commercial/register' else 16384):return self.obj(413,{'error':'Request exceeds 16 KB.'})
            if size<=0:return self.obj(400,{'error':'Expected a JSON request body.'})
            payload=json.loads(self.rfile.read(size))
            if path.startswith('/api/commercial/'):
                if not isinstance(payload,dict):raise ValueError('Expected a JSON object.')
                app=Commercial(get_store())
                routes={'/api/commercial/campaign':app.campaign,'/api/commercial/paywall':app.paywall,'/api/commercial/outcome':app.outcome,'/api/commercial/chat':app.chat,'/api/commercial/register':app.register,'/api/commercial/activate':lambda p:app.activate(p.get('id'))}
                if path not in routes:return self.obj(404,{'error':'Route not found.'})
                return self.obj(200,routes[path](payload))
            if path=='/api/benchmark/score':
                if not EXTERNAL:return self.obj(503,{'error':'External model is not trained.'})
                if not isinstance(payload,dict) or set(payload)!={'values'}:raise ValueError('Expected an object containing values.')
                result=benchmark_score(payload['values'],EXTERNAL)
                result['prediction_id']=get_store().record(payload['values'],result)
                result['runtime']='Local Python · SQLite'
            else:result=decision(payload,MODEL)
            return self.obj(200,result)
        except (ValueError,TypeError,OverflowError,KeyError,AttributeError) as exc:return self.obj(400,{'error':str(exc)})
    def do_PUT(self):return self.obj(405,{'error':'Method not allowed.'})
    do_DELETE=do_PUT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=4173);p.add_argument('--host',default='127.0.0.1');args=p.parse_args()
    get_store()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Edition: http://{args.host}:{server.server_port}',flush=True)
    server.serve_forever()
