"""Temporal lifecycle benchmark with validation calibration and optional MLflow tracking."""
import argparse,json,math,os
from pathlib import Path
import numpy as np
from lifecycle import generate,materialize,fingerprint
from train_commercial import FEATURES,BRANDS,ROOT

def fit(x,y,steps=900):
    b=np.zeros(x.shape[1])
    for _ in range(steps):
        p=1/(1+np.exp(-np.clip(x@b,-30,30)));b-=.15*(x.T@(p-y)/len(y)+.002*np.r_[0,b[1:]])
    return b

def validate_rows(rows):
    from datetime import date,timedelta
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Expected unique reader IDs and nonempty data.')
    for r in rows:
        if r['kind']!=('churn' if r['subscribed'] else 'sales') or r['label'] not in [0,1]:raise ValueError('Invalid lifecycle eligibility or label.')
        if set(r['values'])!=set(FEATURES) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in r['values'].values()):raise ValueError('Invalid features.')
        if date.fromisoformat(r['observed_until'])<date.fromisoformat(r['snapshot'])+timedelta(days=30):raise ValueError('Incomplete outcome window.')
    for partition in ['train','validation','test']:
        for kind in ['sales','churn']:
            if {r['label'] for r in rows if r['partition']==partition and r['kind']==kind}!={0,1}:raise ValueError('Every partition/cohort needs positive and negative examples.')
    for left,right in [('train','validation'),('validation','test')]:
        if max(r['observed_until'] for r in rows if r['partition']==left)>min(r['snapshot'] for r in rows if r['partition']==right):raise ValueError('Temporal outcome overlap.')
    return rows

def train(seed=42,output=None,tracking=False,features_file=None):
    raw=generate(seed) if features_file is None else None
    rows=validate_rows(materialize(raw) if raw is not None else [json.loads(line) for line in Path(features_file).read_text().splitlines() if line.strip()]);models={}
    for kind in ['sales','churn']:
        parts={p:[r for r in rows if r['kind']==kind and r['partition']==p] for p in ['train','validation','test']}
        x={p:np.array([[r['values'][k] for k in FEATURES] for r in rr]) for p,rr in parts.items()};y={p:np.array([r['label'] for r in rr]) for p,rr in parts.items()}
        mean=x['train'].mean(0);scale=np.maximum(x['train'].std(0),1e-8)
        design={p:np.column_stack([np.ones(len(v)),(v-mean)/scale]) for p,v in x.items()};b=fit(design['train'],y['train'])
        calibration=fit(np.column_stack([np.ones(len(x['validation'])),design['validation']@b]),y['validation'])
        rawtest=design['test']@b;p=1/(1+np.exp(-np.clip(calibration[0]+calibration[1]*rawtest,-30,30)));target=y['test'];pos=p[target==1];neg=p[target==0]
        auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
        prev=float(y['train'].mean());baseline=float(np.mean((prev-target)**2))
        bins=[]
        for lo in np.arange(0,1,.1):
            mask=(p>=lo)&(p<lo+.1)
            if mask.any():bins.append({'n':int(mask.sum()),'predicted':float(p[mask].mean()),'observed':float(target[mask].mean())})
        models[kind]={'intercept':float(calibration[0]+calibration[1]*b[0]),'weights':(calibration[1]*b[1:]).tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'target':kind+' within [snapshot, snapshot + 30 days)','metrics':{'auc':auc,'brier':float(np.mean((p-target)**2)),'prevalence':float(target.mean()),'constant_brier':baseline,'test_rows':len(target),'calibration':bins},'calibration':'Sigmoid fit on validation only'}
    base=json.loads((ROOT/'artifacts/commercial_model.json').read_text())
    base.update(version=f'lifecycle-temporal-seed{seed}',seed=seed,models=models,dataset_sha256=fingerprint(raw if raw is not None else rows),split={p:{'n':sum(r['partition']==p for r in rows),'start':min(r['snapshot'] for r in rows if r['partition']==p),'end':max(r['snapshot'] for r in rows if r['partition']==p)} for p in ['train','validation','test']},training='Dated simulated events; past-only features; separate eligible sales/churn cohorts; temporal partitions with 30-day outcome embargo; validation sigmoid calibration',data_contract={'features':'Events strictly before snapshot; visits/premium reads in last 30 days, failures in last 90 days. Tenure from start date. Recency capped at 90 days.','sales':'Non-subscribers at snapshot; subscribe in [snapshot, snapshot+30d).','churn':'Subscribers at snapshot; cancel in [snapshot, snapshot+30d).','censoring':'Reject records without a complete 30-day observation window.','identity':'Disjoint reader IDs across train/validation/test.'})
    base['data_source']='generated local lifecycle events' if features_file is None else 'imported feature snapshot'
    if output:
        output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(base,indent=2));folder=output.parent
        if raw is not None:(folder/'lifecycle_events.jsonl').write_text('\n'.join(json.dumps(r) for r in raw)+'\n')
        (folder/'lifecycle_features.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    if tracking:
        import mlflow
        from mlflow.tracking import MlflowClient
        uri='sqlite:///'+str(ROOT/'data/mlflow.sqlite3');mlflow.set_tracking_uri(uri)
        client=MlflowClient();experiment=client.get_experiment_by_name('Edition lifecycle')
        eid=experiment.experiment_id if experiment else client.create_experiment('Edition lifecycle',artifact_location=(ROOT/'data/mlflow-artifacts').as_uri())
        with mlflow.start_run(experiment_id=eid,run_name=base['version']) as run:
            mlflow.log_params({'seed':seed,'dataset_sha256':base['dataset_sha256'],'features':len(FEATURES),'label_window_days':30,'model':'L2 logistic + validation sigmoid','split':'temporal with outcome embargo','data_source':base['data_source']})
            mlflow.log_metrics({kind+'_'+k:v for kind,m in models.items() for k,v in m['metrics'].items() if isinstance(v,(int,float))})
            mlflow.log_dict(base,'commercial_model.json');mlflow.log_dict(base['data_contract'],'data_contract.json')
            if output:mlflow.log_artifact(str(output.parent/'lifecycle_features.jsonl'))
            base['mlflow_run_id']=run.info.run_id
        if output:output.write_text(json.dumps(base,indent=2))
    return base
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,default=ROOT/'data/training/candidate.json');p.add_argument('--mlflow',action='store_true');p.add_argument('--features-file',type=Path);a=p.parse_args();print(json.dumps({k:v['metrics'] for k,v in train(a.seed,a.output,a.mlflow,a.features_file)['models'].items()}))
