"""Reproducible multi-arm T-learner, held-out IPS/DR evaluation. No external data."""
import csv, json, math, sys
from pathlib import Path
from datetime import date, timedelta, datetime, timezone
import numpy as np
from engine import ACTIONS, BRANDS, FEATURES, features, blocked
ROOT = Path(__file__).resolve().parents[1]

def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))

def generate(n=42000, seed=42):
    rng=np.random.default_rng(seed)
    rows=[]
    for i in range(n):
        day=i*42//n
        d=int(rng.integers(0,29)); a=int(min(120,rng.poisson(d*2+2))); p=int(min(40,a,rng.binomial(a,.35)))
        r={'brand':BRANDS[int(rng.integers(4))], 'active_days':d,'articles':a,'premium_reads':p,'recency_hours':float(int(rng.integers(0,169))), 'dismissals':int(rng.choice(6,p=[.48,.24,.13,.08,.04,.03])), 'mobile':bool(rng.random()<.68), 'consent':bool(rng.random()<.92), 'subscribed':False,'public_interest':bool(rng.random()<.04)}
        x=np.array(features(r)); freq=d/28; prem=p/40; recent=r['recency_hours']/168
        # Explicit simulator, not inferred Schibsted behaviour. Hidden potential outcomes
        # are retained only for simulation diagnostics; fitting sees the assigned outcome.
        base=-4.8+3.1*freq+.9*a/120+.9*prem-.7*recent-.8*r['dismissals']/5+.15*r['mobile']+ .3*(r['brand']=='E24')
        logits=np.array([base,base+.9+1.2*freq-.9*recent,base+2.9-3.2*freq+1.2*prem,base+1.9-1.1*freq+.9*prem])
        probs=sigmoid(logits)
        # Business eligibility is enforced before assignment. Logged probabilities are
        # one for forced-open traffic, one quarter for eligible randomised traffic.
        eligible=not blocked(r)
        arm=int(rng.integers(4)) if eligible else 0
        propensity=.25 if eligible else 1.
        y=int(rng.random()<probs[arm])
        rows.append(dict(r, reader_id=f'sim-{i:06d}', date=str(date(2026,8,1)+timedelta(days=day)), day=day, action=arm, propensity=propensity, outcome=y))
    return rows

def fit(X,y,ridge):
    w=np.zeros(X.shape[1]); penalty=np.eye(len(w))*ridge;penalty[0,0]=0
    for _ in range(60):
        p=sigmoid(X@w); grad=X.T@(p-y)/len(y)+penalty@w
        hess=(X.T*(p*(1-p)))@X/len(y)+penalty+np.eye(len(w))*1e-9
        step=np.linalg.solve(hess,grad);w-=step
        if np.max(np.abs(step))<1e-7: break
    return w

def arrays(rows):
    return (np.array([features(r) for r in rows]),np.array([r['action'] for r in rows]),np.array([r['outcome'] for r in rows]),np.array([r['propensity'] for r in rows]),np.array([bool(blocked(r)) for r in rows]))

def logloss(y,p):
    p=np.clip(p,1e-9,1-1e-9)
    return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))

def auc(y,p):
    # Average ranks for tied scores (Mann-Whitney identity).
    order=np.argsort(p); ranks=np.empty(len(p),float); sorted_p=p[order]; start=0
    while start<len(p):
        end=start+1
        while end<len(p) and sorted_p[end]==sorted_p[start]:end+=1
        ranks[order[start:end]]=(start+1+end)/2;start=end
    n1=y.sum();n0=len(y)-n1
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)) if n1*n0 else None

def interval(z):
    value=float(np.mean(z));se=float(np.std(z,ddof=1)/math.sqrt(len(z)))
    return {'estimate':value,'lower':value-1.96*se,'upper':value+1.96*se,'se':se}

def policy(prob,blocked_mask,kind='learned',fixed=None):
    val=prob*np.array([a['margin'] for a in ACTIONS])-np.array([a['cost'] for a in ACTIONS])
    target=np.argmax(val,axis=1) if fixed is None else np.full(len(prob),fixed,int)
    target[blocked_mask]=0
    return target

def dr_scores(prob,arm,y,ps,target):
    margins=np.array([a['margin'] for a in ACTIONS]);costs=np.array([a['cost'] for a in ACTIONS]);idx=np.arange(len(y))
    q=prob*margins-costs; reward=y*margins[arm]-costs[arm]
    weight=(arm==target)/ps
    return q[idx,target]+weight*(reward-q[idx,arm]),weight*reward

def report(rows,W,baseline_arm):
    X,arm,y,ps,b=arrays(rows);prob=sigmoid(X@W.T);idx=np.arange(len(y));factual=prob[idx,arm]
    selected=policy(prob,b);baseline=policy(prob,b,fixed=baseline_arm);control=np.zeros(len(y),int)
    learned_dr,learned_ips=dr_scores(prob,arm,y,ps,selected)
    base_dr,base_ips=dr_scores(prob,arm,y,ps,baseline)
    control_dr,_=dr_scores(prob,arm,y,ps,control)
    cal=[]
    for low in np.linspace(0,.9,10):
        mask=(factual>=low)&(factual<low+.10000001)
        if mask.any():cal.append({'predicted':float(factual[mask].mean()),'observed':float(y[mask].mean()),'n':int(mask.sum())})
    treatment=[]
    for k,a in enumerate(ACTIONS):
        mask=arm==k;n=int(mask.sum());conv=float(y[mask].mean());se=math.sqrt(conv*(1-conv)/n)
        treatment.append({'id':a['id'],'name':a['name'],'n':n,'conversions':int(y[mask].sum()),'conversion':conv,'lower':max(0,conv-1.96*se),'upper':min(1,conv+1.96*se),'mean_contribution':float((y[mask]*a['margin']-a['cost']).mean())})
    # Randomised A/B results are restricted to eligible readers, so forced-open
    # traffic never contaminates comparisons between randomised arms.
    eligible=~b; randomised=[]
    for k,a in enumerate(ACTIONS):
        mask=eligible&(arm==k); n=int(mask.sum());v=y[mask];c=float(v.mean());se=math.sqrt(c*(1-c)/n)
        randomised.append({'name':a['name'],'id':a['id'],'n':n,'conversions':int(v.sum()),'conversion':c,'lower':max(0,c-1.96*se),'upper':min(1,c+1.96*se),'mean_contribution':float((v*a['margin']-a['cost']).mean())})
    control_y=y[eligible&(arm==0)]
    for k,item in enumerate(randomised):
        yy=y[eligible&(arm==k)];delta=float(yy.mean()-control_y.mean());se=math.sqrt(float(np.var(yy,ddof=1)/len(yy)+np.var(control_y,ddof=1)/len(control_y))) if k else 0
        item['difference_vs_control']={'estimate':delta,'lower':delta-1.96*se,'upper':delta+1.96*se}
    daily=[]
    for day in sorted({r['day'] for r in rows}):
        m=np.array([r['day']==day for r in rows]);daily.append({'date':rows[np.where(m)[0][0]]['date'],'learned':float(learned_dr[m].mean()),'baseline':float(base_dr[m].mean()),'n':int(m.sum())})
    counts=np.bincount(selected,minlength=4)
    return {'n':len(rows),'eligible_n':int(eligible.sum()),'blocked_n':int(b.sum()),'date_start':rows[0]['date'],'date_end':rows[-1]['date'],'auc':auc(y,factual),'brier':float(np.mean((factual-y)**2)),'logloss':logloss(y,factual),'ece':float(sum(c['n']*abs(c['predicted']-c['observed']) for c in cal)/len(rows)),'calibration':cal,'policy_value':interval(learned_dr),'baseline_value':interval(base_dr),'open_value':interval(control_dr),'incremental_value':interval(learned_dr-base_dr),'ips_value':interval(learned_ips),'baseline_arm':ACTIONS[baseline_arm]['name'],'policy_mix':[{'name':a['name'],'id':a['id'],'n':int(counts[k]),'share':float(counts[k]/len(rows))} for k,a in enumerate(ACTIONS)],'randomised_arms':randomised,'daily':daily,'overlap':{'min_logged_propensity':float(ps.min()),'max_weight':float(np.max((arm==selected)/ps)),'match_rate':float(np.mean(arm==selected))},'observed_conversion':float(y.mean())}

def main():
    rows=generate();train=[r for r in rows if r['day']<28]; val=[r for r in rows if 28<=r['day']<35];test=[r for r in rows if r['day']>=35]
    X,A,Y,_,_=arrays(train);V,va,vy,_,vb=arrays(val)
    candidates=[]
    for ridge in [.0001,.001,.01]:
        W=np.array([fit(X[A==a],Y[A==a],ridge) for a in range(4)])
        loss=logloss(vy,sigmoid(V@W.T)[np.arange(len(vy)),va]);candidates.append((loss,ridge,W))
    loss,ridge,W=min(candidates,key=lambda x:x[0]);valprob=sigmoid(V@W.T)
    # Choose a uniform non-control baseline using validation only.
    baseline_arm=max(range(1,4),key=lambda k: np.mean(dr_scores(valprob,va,vy,np.array([r['propensity'] for r in val]),policy(valprob,vb,fixed=k))[0]))
    reports={b:report([r for r in test if b=='all' or r['brand']==b],W,baseline_arm) for b in ['all']+BRANDS}
    for b,rep in reports.items():
        tr=[r for r in train if b=='all' or r['brand']==b]; tx,ta,ty,_,_=arrays(tr);tp=sigmoid(tx@W.T)[np.arange(len(ty)),ta]; rep['train_auc']=auc(ty,tp)
        trainfreq=np.histogram([r['active_days'] for r in tr],bins=[0,7,14,21,29])[0]/len(tr)
        testr=[r for r in test if b=='all' or r['brand']==b];testfreq=np.histogram([r['active_days'] for r in testr],bins=[0,7,14,21,29])[0]/len(testr)
        psi=float(np.sum((testfreq-trainfreq)*np.log((testfreq+1e-6)/(trainfreq+1e-6))))
        rep['drift']={'feature':'active_days','psi':psi,'bands':['0–6','7–13','14–20','21–28'],'train_share':trainfreq.tolist(),'test_share':testfreq.tolist(),'threshold':.2}
    artifact={'version':'tlearner-logit-v1-seed42','seed':42,'weights':W.tolist(),'feature_names':FEATURES,'actions':ACTIONS,'brands':BRANDS,'ridge':ridge,'validation_logloss':loss,'candidates':[{'ridge':c[1],'validation_logloss':c[0]} for c in candidates],'baseline_arm':baseline_arm,'training_rows':len(train),'validation_rows':len(val),'test_rows':len(test),'train_window':[train[0]['date'],train[-1]['date']],'validation_window':[val[0]['date'],val[-1]['date']],'test_window':[test[0]['date'],test[-1]['date']],'created_at':datetime.now(timezone.utc).isoformat(),'reports':reports,'data_mode':'synthetic'}
    (ROOT/'artifacts/model.json').write_text(json.dumps(artifact,indent=2,allow_nan=False))
    with (ROOT/'data/synthetic_experiment.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'training':len(train),'validation':len(val),'test':len(test),'ridge':ridge,'auc':reports['all']['auc'],'policy_value':reports['all']['policy_value'],'incremental_value':reports['all']['incremental_value']}))

if __name__=='__main__': main()
