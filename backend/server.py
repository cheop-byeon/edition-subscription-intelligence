"""Local Python backend and frontend server. Inference needs only the standard library."""
import argparse, json, sys
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
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
class Handler(BaseHTTPRequestHandler):
    def send(self,status,body,kind='application/json; charset=utf-8'):
        body=body.encode() if isinstance(body,str) else body
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(body)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff');self.end_headers()
        if self.command!='HEAD':self.wfile.write(body)
    def obj(self,status,value): self.send(status,json.dumps(value,allow_nan=False))
    def do_GET(self):
        u=urlparse(self.path);p=u.path
        if p=='/':return self.send(200,(ROOT/'frontend/index.html').read_bytes(),'text/html; charset=utf-8')
        if p=='/favicon.ico':return self.send(204,b'','image/x-icon')
        if p=='/api/health':return self.obj(200,{'status':'ok','backend':'Python','model_version':MODEL['version'],'data_mode':'synthetic'})
        if p=='/api/benchmark':return self.obj(200,{k:v for k,v in EXTERNAL.items() if k!='model'} if EXTERNAL else None)
        if p=='/api/research':return self.obj(200,RESEARCH)
        if p=='/api/model':return self.obj(200,{k:v for k,v in MODEL.items() if k!='reports'})
        if p=='/api/report':
            b=parse_qs(u.query).get('brand',['all'])[0]
            if b not in ['all']+BRANDS:return self.obj(400,{'error':'Unsupported brand.'})
            return self.obj(200,MODEL['reports'][b])
        return self.obj(404,{'error':'Route not found.'})
    do_HEAD=do_GET
    def do_POST(self):
        if urlparse(self.path).path not in ['/api/decision','/api/benchmark/score']:return self.obj(404,{'error':'Route not found.'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if size>16384:return self.obj(413,{'error':'Request exceeds 16 KB.'})
            if size<=0:return self.obj(400,{'error':'Expected a JSON request body.'})
            payload=json.loads(self.rfile.read(size))
            if urlparse(self.path).path=='/api/benchmark/score':
                if not EXTERNAL:return self.obj(503,{'error':'External model is not trained.'})
                if not isinstance(payload,dict) or set(payload)!={'values'}:raise ValueError('Expected an object containing values.')
                result=benchmark_score(payload['values'],EXTERNAL)
            else:result=decision(payload,MODEL)
            return self.obj(200,result)
        except (ValueError,TypeError,OverflowError) as exc:return self.obj(400,{'error':str(exc)})
    def do_PUT(self):return self.obj(405,{'error':'Method not allowed.'})
    do_DELETE=do_PUT
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=4173);p.add_argument('--host',default='127.0.0.1');args=p.parse_args()
    server=ThreadingHTTPServer((args.host,args.port),Handler)
    print(f'Edition: http://{args.host}:{server.server_port}',flush=True)
    server.serve_forever()
