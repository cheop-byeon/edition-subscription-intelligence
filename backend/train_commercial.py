"""Reproducible, explicitly simulated sales/churn benchmark training."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['visits_30d','active_days_30d','premium_reads_30d','tenure_months','days_since_visit','payment_failures_90d']
BRANDS=['Aftenposten','VG','Bergens Tidende','E24']
def train(seed=42):
    rng=np.random.default_rng(seed);n=12000
    x=np.column_stack([rng.poisson(12,n),rng.integers(0,31,n),rng.poisson(4,n),rng.integers(0,61,n),rng.integers(0,31,n),rng.binomial(3,.08,n)]).astype(float)
    dates=np.arange(n);mean=x[:8000].mean(0);scale=x[:8000].std(0);z=(x-mean)/scale
    models={}
    for kind,w,b in [('sales',[.4,.65,.8,-.1,-.55,-.1],-2.3),('churn',[-.3,-.7,-.25,-.3,.9,.65],-1.7)]:
        y=(rng.random(n)<1/(1+np.exp(-(z@w+b)))).astype(float)
        design=np.column_stack([np.ones(n),z]);beta=np.zeros(7)
        for _ in range(800):
            prob=1/(1+np.exp(-np.clip(design[:8000]@beta,-30,30)))
            beta-=.15*(design[:8000].T@(prob-y[:8000])/8000+.002*np.r_[0,beta[1:]])
        p=1/(1+np.exp(-design[10000:]@beta));target=y[10000:];pos=p[target==1];neg=p[target==0]
        auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
        models[kind]={'intercept':float(beta[0]),'weights':beta[1:].tolist(),'mean':mean.tolist(),'scale':scale.tolist(),'metrics':{'auc':auc,'brier':float(np.mean((p-target)**2)),'prevalence':float(target.mean()),'test_rows':2000},'target':('subscription within 30 days' if kind=='sales' else 'cancellation within 30 days')}
    profiles=[]
    for i in range(240):
        j=10000+i
        profiles.append({'id':f'reader-{i+1:04}','brand':BRANDS[i%4],'subscribed':bool((i//4)%2),'consent':i%11!=0,'suppressed':i%17==0,'values':dict(zip(FEATURES,x[j].tolist()))})
    artifact={'version':f'commercial-logistic-seed{seed}','data_kind':'simulated commercial lifecycle data','seed':seed,'features':FEATURES,'brands':BRANDS,'models':models,'profiles':profiles,'split':{'train':8000,'validation_reserved':2000,'test':2000},'training':'L2 logistic regression, NumPy gradient descent; 30-day synthetic outcomes; ordered disjoint partitions','limitations':['All reader records and outcomes are simulated, not Schibsted customer data.','Metrics measure synthetic benchmark performance, not commercial effectiveness.','Paywall rules are a policy prototype, not learned causal offer effects.']}
    artifact['dataset_sha256']=hashlib.sha256(x.tobytes()).hexdigest()
    return artifact
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--output',type=Path,default=ROOT/'artifacts/commercial_model.json');a=p.parse_args();a.output.write_text(json.dumps(train(a.seed),indent=2));print(a.output)
