#!/usr/bin/env python3
"""Start the one Edition UI and its compatibility API with a single command."""
import argparse,importlib.util,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ui-port',type=int,default=8501);parser.add_argument('--api-port',type=int,default=4173)
    parser.add_argument('--host',default='127.0.0.1');a=parser.parse_args()
    if not importlib.util.find_spec('streamlit'):
        import json
        local=ROOT/'config/runtime.local.json'
        runtime=json.loads(local.read_text()).get('python') if local.exists() else None
        if runtime and Path(runtime).is_file() and Path(runtime).absolute()!=Path(sys.executable).absolute():
            os.execv(runtime,[runtime,str(Path(__file__).resolve())]+sys.argv[1:])
        raise SystemExit('Install the UI dependencies first: python -m pip install -r requirements-ui.txt')
    environment={**os.environ,'EDITION_UI_URL':os.environ.get('EDITION_UI_URL',f'http://127.0.0.1:{a.ui_port}/')}
    commands=[ [sys.executable,str(ROOT/'backend/server.py'),'--host',a.host,'--port',str(a.api_port)],
        [sys.executable,'-m','streamlit','run',str(ROOT/'apps/edition.py'),'--server.address',a.host,'--server.port',str(a.ui_port),'--server.headless','true','--browser.gatherUsageStats','false'] ]
    children=[]
    def stop(*_):
        for child in children:
            if child.poll() is None:child.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        children.extend(subprocess.Popen(cmd,cwd=ROOT,env=environment) for cmd in commands)
        print(f'Edition: http://127.0.0.1:{a.ui_port}/ (old links redirect here)',flush=True)
        while all(child.poll() is None for child in children):time.sleep(.25)
    finally:
        stop()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
    return max(child.returncode or 0 for child in children)
if __name__=='__main__':raise SystemExit(main())
