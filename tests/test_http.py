"""Exercise the actual HTTP server with an isolated ephemeral loopback port."""
import tempfile
import json, sys, threading, unittest, urllib.error, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
import server
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.previous_store=server.STORE;server.STORE=server.Store(Path(cls.temp.name)/'test.sqlite3')
        server.STORE.seed(server.EXTERNAL['profiles'],'test_fixture')
        cls.http=ThreadingHTTPServer(('127.0.0.1',0),server.Handler);cls.thread=threading.Thread(target=cls.http.serve_forever,daemon=True);cls.thread.start();cls.base=f'http://127.0.0.1:{cls.http.server_port}'
    @classmethod
    def tearDownClass(cls):cls.http.shutdown();cls.http.server_close();cls.thread.join(timeout=5);server.STORE=cls.previous_store;cls.temp.cleanup()
    def request(self,path,body=None,method=None):
        req=urllib.request.Request(self.base+path,data=None if body is None else json.dumps(body).encode(),headers={'Content-Type':'application/json'},method=method)
        try:
            with urllib.request.build_opener(NoRedirect).open(req,timeout=5) as response:return response.status,response.read(),response.headers
        except urllib.error.HTTPError as e:
            with e:return e.code,e.read(),e.headers
    def test_frontend_and_health(self):
        status,body,headers=self.request('/');self.assertEqual(status,307);self.assertEqual(headers['Location'],'http://127.0.0.1:8501/')
        status,body,_=self.request('/api/health');self.assertEqual(status,200);self.assertEqual(json.loads(body)['status'],'ok')
    def test_commercial_workflow(self):
        status,_,headers=self.request('/article');self.assertEqual(status,307);self.assertIn('page=Paywall',headers['Location'])
        status,_,headers=self.request('/commercial');self.assertEqual(status,307);self.assertEqual(headers['Location'],'http://127.0.0.1:8501/')
        status,body,_=self.request('/api/commercial/summary?brand=VG');self.assertEqual(status,200);self.assertEqual(json.loads(body)['readers'],60)
        status,body,_=self.request('/api/commercial/campaign',{'name':'HTTP test','brand':'VG','kind':'sales','threshold':0});self.assertEqual(status,200)
        campaign=json.loads(body);self.assertGreater(campaign['members'],0)
        self.assertEqual(self.request('/api/commercial/export?id='+str(campaign['id']))[0],200)
        self.assertEqual(self.request('/api/commercial/paywall',{'reader_id':'unknown'})[0],400)
    def test_head_supported(self):
        status,body,_=self.request('/',method='HEAD');self.assertEqual(status,307);self.assertEqual(body,b'')
    def test_live_external_inference(self):
        _,body,_=self.request('/api/benchmark');a=json.loads(body);self.assertNotIn('model',a);self.assertGreater(len(a['profiles']),0)
        p=a['profiles'][0];status,body,_=self.request('/api/benchmark/score',{'values':p['values']});self.assertEqual(status,200);self.assertAlmostEqual(json.loads(body)['probability'],p['prediction'],places=10)
    def test_errors_have_json_contract(self):
        for path,body,expected in [('/api/benchmark/score',{'values':{}},400),('/api/decision',{'articles':-1},400),('/unknown',None,404)]:
            status,data,_=self.request(path,body);self.assertEqual(status,expected);self.assertIn('error',json.loads(data))
    def test_guarded_reader(self):
        status,body,_=self.request('/api/decision',{'subscribed':True});self.assertEqual(status,200);self.assertEqual(json.loads(body)['action']['id'],'open')
    def test_methods_and_size_limits(self):
        self.assertEqual(self.request('/api/decision',method='DELETE')[0],405)
        self.assertEqual(self.request('/api/decision',{'x':'x'*17000})[0],413)
if __name__=='__main__':unittest.main()
