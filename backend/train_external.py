"""Train a publisher-paper-linked subscription benchmark and export portable inference.
Usage: python backend/train_external.py --archive /path/to/kkbox-engagement.zip
"""
import argparse, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
ROOT=Path(__file__).resolve().parents[1]
RAW=['num_unq','num_100','num_25','num_50','num_75','num_985','total_secs']
ENRICH=['avg_num_unq','smooth_avg_num_unq','max_num_unq','min_num_unq','days_since_first_visit','visit_frequency_last7days','visit_frequency_last14days','visit_frequency_last28days','volume_total_secs_last7days','volume_total_secs_last14days','volume_total_secs_last28days','volume_num_100_last7days','volume_num_100_last28days']
FEATS=RAW+ENRICH

def clean(df,prep=None):
    X=df[FEATS].to_numpy(dtype=float);X[~np.isfinite(X)]=np.nan;X[X<0]=np.nan
    if prep is None:
        median=np.nanmedian(X,axis=0);cap=np.nanquantile(X,.995,axis=0);prep={'median':median.tolist(),'cap':cap.tolist()}
    X=np.where(np.isnan(X),np.array(prep['median']),X);X=np.minimum(X,np.array(prep['cap']));return np.log1p(X),prep

def metrics(y,p):
    pred=np.asarray(p);bins=np.minimum((pred*10).astype(int),9);cal=[]
    for i in range(10):
        m=bins==i
        if m.any():cal.append({'predicted':float(pred[m].mean()),'observed':float(np.mean(y[m])),'n':int(m.sum())})
    order=np.argsort(-pred);top=order[:max(1,int(len(y)*.1))]
    return {'auc':float(roc_auc_score(y,pred)),'average_precision':float(average_precision_score(y,pred)),'brier':float(brier_score_loss(y,pred)),'logloss':float(log_loss(y,pred)),'prevalence':float(np.mean(y)),'top_decile_rate':float(np.mean(y[top])),'top_decile_lift':float(np.mean(y[top])/np.mean(y)),'calibration':cal,'ece':float(sum(c['n']*abs(c['predicted']-c['observed']) for c in cal)/len(y))}

def tree_export(m):
    trees=[]
    for iteration in m._predictors:
        for predictor in iteration:
            n=predictor.nodes
            assert not np.any(n['is_categorical'])
            trees.append([{'leaf':bool(v['is_leaf']),'value':float(v['value']),'feature':int(v['feature_idx']),'threshold':float(v['num_threshold']),'left':int(v['left']),'right':int(v['right'])} for v in n])
    return {'base':float(m._baseline_prediction[0,0]),'trees':trees}

def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);args=p.parse_args();archive=Path(args.archive)
    with zipfile.ZipFile(archive) as z:df=pd.read_csv(z.open('kkbox_engagement_labeled_dataset.csv'))
    df.columns=[s.strip("'") for s in df.columns];original=len(df);dupes=int(df.duplicated().sum());neg=int((df[FEATS]<0).sum().sum())
    df=df.drop_duplicates().sort_values('date_converted',kind='stable').reset_index(drop=True)
    dates=df.date_converted;cut1=dates.iloc[int(len(df)*.65)];cut2=dates.iloc[int(len(df)*.78)];cut3=dates.iloc[int(len(df)*.88)]
    splits={'train':df[dates<=cut1],'selection':df[(dates>cut1)&(dates<=cut2)],'calibration':df[(dates>cut2)&(dates<=cut3)],'test':df[dates>cut3]}
    print({k:(len(v),str(v.date_converted.min()),str(v.date_converted.max()),float(v.subscription.mean())) for k,v in splits.items()},flush=True)
    X,prep=clean(splits['train']); xs={k:clean(v,prep)[0] for k,v in splits.items() if k!='train'}; ys={k:v.subscription.to_numpy() for k,v in splits.items()};tr=ys['train']
    scaler=StandardScaler().fit(X);lr=LogisticRegression(max_iter=250,C=.1).fit(scaler.transform(X),tr)
    rawidx=list(range(len(RAW)));rawscale=StandardScaler().fit(X[:,rawidx]);rawlr=LogisticRegression(max_iter=250,C=.1).fit(rawscale.transform(X[:,rawidx]),tr)
    candidates=[]
    for leaves in [7,15]:
        start=time.time();m=HistGradientBoostingClassifier(max_iter=90,max_leaf_nodes=leaves,min_samples_leaf=120,learning_rate=.07,l2_regularization=1,early_stopping=False,random_state=42).fit(X,tr)
        loss=log_loss(ys['selection'],m.predict_proba(xs['selection'])[:,1]);candidates.append((float(loss),leaves,m));print('candidate',leaves,'loss',loss,'seconds',time.time()-start,flush=True)
    _,leaves,model=min(candidates,key=lambda x:x[0]);probcal=model.predict_proba(xs['calibration'])[:,1];logit=np.log(np.clip(probcal,1e-9,1-1e-9)/np.clip(1-probcal,1e-9,1))
    calibrator=LogisticRegression(C=1e3,max_iter=200).fit(logit.reshape(-1,1),ys['calibration']);slope=float(calibrator.coef_[0,0]);intercept=float(calibrator.intercept_[0])
    prob=model.predict_proba(xs['test'])[:,1];testlogit=np.log(np.clip(prob,1e-9,1-1e-9)/np.clip(1-prob,1e-9,1));prob=calibrator.predict_proba(testlogit.reshape(-1,1))[:,1]
    ty=ys['test'];result=metrics(ty,prob);baselines={'constant':metrics(ty,np.full(len(ty),float(tr.mean()))),'raw_logistic':metrics(ty,rawlr.predict_proba(rawscale.transform(xs['test'][:,rawidx]))[:,1]),'enriched_logistic':metrics(ty,lr.predict_proba(scaler.transform(xs['test']))[:,1])}
    # Day-block resampling preserves within-day dependence. It cannot remedy the
    # absent user identifier or make this a user-disjoint validation.
    rng=np.random.default_rng(42);days=splits['test'].date_converted.to_numpy();uniq=np.unique(days);dayindexes=[np.where(days==d)[0] for d in uniq];scores=[]
    for _ in range(100):
        ix=np.concatenate([dayindexes[j] for j in rng.integers(0,len(uniq),len(uniq))]);scores.append(roc_auc_score(ty[ix],prob[ix]))
    result['auc_interval']=np.quantile(scores,[.025,.975]).tolist(); daily=[]
    for day in uniq:
        ix=days==day;daily.append({'date':str(day),'n':int(ix.sum()),'predicted':float(prob[ix].mean()),'observed':float(ty[ix].mean())})
    # Display a small reproducible held-out sample, including both label classes.
    # It is deliberately not used for threshold selection or model fitting.
    ix0=np.where(ty==0)[0];ix1=np.where(ty==1)[0];sample=np.concatenate([rng.choice(ix0,60,replace=False),rng.choice(ix1,20,replace=False)]);rng.shuffle(sample)
    profiles=[]
    for n,i in enumerate(sample):
        row=splits['test'].iloc[int(i)];values={k:float(row[k]) if row[k]>=0 else None for k in FEATS}
        profiles.append({'id':f'holdout-{n+1:03d}','date':str(row.date_converted),'label':int(row.subscription),'prediction':float(prob[i]),'values':values})
    modeljson=tree_export(model);modeljson.update({'feature_names':FEATS,'preprocess':prep,'calibration':{'slope':slope,'intercept':intercept}})
    # Numeric exported inference must agree with sklearn before it is served.
    from external_engine import predict_export
    for i in range(min(1000,len(xs['test']))):
        v={k:float(splits['test'].iloc[i][k]) for k in FEATS};p1=predict_export(v,modeljson)
        if abs(p1-prob[i])>1e-10:raise AssertionError(f'Export mismatch {i}: {p1} != {prob[i]}')
    payload={'version':'kkbox-hgb-v1-seed42','dataset':'KKBOX User Engagement Modeling Dataset','data_kind':'Published synthetic subscription benchmark based on KKBOX listening data','target':'Provided synthetic subscription label','source_url':'https://www.kaggle.com/datasets/pawelwmm/kkbox-user-engagement-modeling-dataset','paper_url':'https://doi.org/10.1007/978-3-030-95947-0_5','downloaded_at':datetime.now(timezone.utc).isoformat(),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'original_rows':original,'original_positives':int(df.subscription.sum()),'duplicate_rows_removed':dupes,'rows_after_deduplication':len(df),'negative_feature_cells':neg,'features':FEATS,'model_family':'Histogram gradient boosting + sigmoid calibration','selected_leaf_nodes':leaves,'candidates':[{'max_leaf_nodes':l,'selection_logloss':loss} for loss,l,_ in candidates],'splits':{k:{'n':len(v),'start':str(v.date_converted.min()),'end':str(v.date_converted.max()),'positives':int(v.subscription.sum())} for k,v in splits.items()},'metrics':result,'baselines':baselines,'daily':daily,'profiles':profiles,'sample_note':'Displayed 80-record sample includes 60 negatives and 20 positives; it is intentionally class-enriched and is not representative of prevalence. Metrics use the full chronological test set.','limitations':['Synthetic subscription labels: accuracy does not establish real newspaper conversion performance.','No user identifier is supplied: repeated users cannot be grouped or audited across splits.','A chronological date split is used; the precise original label-generation and observation horizon remain unverified.','No randomised offers or offer prices: this model estimates label propensity, not offer uplift or revenue.','Source metadata reports licence Unknown. Raw source data are kept local and excluded from deliverable archives and publication.','Negative measurements are imputed with training medians; feature caps are learned on training data only.'],'export_verified_profiles':1000,'model':modeljson,'library_versions':{}}
    import sklearn
    payload['library_versions']={'sklearn':sklearn.__version__,'numpy':np.__version__,'pandas':pd.__version__}
    (ROOT/'data/local_benchmark_profiles.json').write_text(json.dumps({'profiles':payload['profiles'],'sample_note':payload['sample_note']}))
    from public_profiles import make_profiles
    payload['profiles']=make_profiles(payload)
    payload['profile_origin']='artificial_input'
    payload['sample_note']='Public examples are artificial input profiles without observed labels. Test metrics use the full chronological holdout.'
    (ROOT/'artifacts/external_model.json').write_text(json.dumps(payload,allow_nan=False,separators=(',',':')))
    print('RESULT',json.dumps({'n':len(df),'test':len(ty),'metrics':result,'baseline_auc':{k:v['auc'] for k,v in baselines.items()}}),flush=True)
if __name__=='__main__':main()
