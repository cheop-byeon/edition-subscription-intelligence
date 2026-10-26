"""Validate a running deployment, with a bounded startup retry."""
import json, sys, time, urllib.request
base=sys.argv[1] if len(sys.argv)>1 else 'http://127.0.0.1:4173'
ui=(sys.argv[2] if len(sys.argv)>2 else 'http://127.0.0.1:8501').rstrip('/')
def get(path):
    with urllib.request.urlopen(base+path,timeout=5) as r:return r.read()
for attempt in range(30):
    try:
        assert json.loads(get('/api/health'))['status']=='ok';break
    except (OSError,AssertionError):
        if attempt==29:raise
        time.sleep(1)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None
for route in ['/','/commercial','/article']:
    try:urllib.request.build_opener(NoRedirect).open(base+route,timeout=5);raise AssertionError('Expected redirect')
    except urllib.error.HTTPError as exc:
        with exc:assert exc.code==307 and exc.headers['Location'].startswith(ui+'/')
with urllib.request.urlopen(ui+'/_stcore/health',timeout=5) as response:assert response.read()==b'ok'
a=json.loads(get('/api/benchmark'));assert a['original_rows']==1000316
assert a['profile_origin']=='artificial_input','Container must not contain downloaded benchmark records'
assert all(p['label'] is None for p in a['profiles'])
p=a['profiles'][0]
req=urllib.request.Request(base+'/api/benchmark/score',data=json.dumps({'values':p['values']}).encode(),headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=5) as r:d=json.load(r)
assert abs(d['probability']-p['prediction'])<1e-10
print('Container smoke test passed: single Streamlit UI, legacy redirects, API, model inference and public-safe examples.')
