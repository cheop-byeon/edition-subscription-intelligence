"""Point-in-time features and 30-day labels from dated simulated reader events."""
from datetime import date, timedelta
import hashlib, json, random

def features(events, snapshot, started):
    cutoff=date.fromisoformat(snapshot)
    if date.fromisoformat(started)>cutoff:raise ValueError('Lifecycle starts after snapshot.')
    past=[e for e in events if date.fromisoformat(e['date'])<cutoff]
    recent=[e for e in past if (cutoff-date.fromisoformat(e['date'])).days<=30]
    visits=[e for e in recent if e['type']=='visit']
    all_visits=[e for e in past if e['type']=='visit']
    return {'visits_30d':len(visits),'active_days_30d':len({e['date'] for e in visits}),
        'premium_reads_30d':sum(e['type']=='premium_read' for e in recent),
        'tenure_months':(cutoff-date.fromisoformat(started)).days//30,
        'days_since_visit':min(90,min(((cutoff-date.fromisoformat(e['date'])).days for e in all_visits),default=90)),
        'payment_failures_90d':sum(e['type']=='payment_failure' and (cutoff-date.fromisoformat(e['date'])).days<=90 for e in past)}

def label(events,snapshot,kind,observed_until):
    cutoff=date.fromisoformat(snapshot);end=cutoff+timedelta(days=30)
    if date.fromisoformat(observed_until)<end:raise ValueError('Incomplete 30-day outcome window.')
    target={'sales':'subscribe','churn':'cancel'}[kind]
    return int(any(e['type']==target and cutoff<=date.fromisoformat(e['date'])<end for e in events))

def generate(seed=42,n=12000):
    rng=random.Random(seed);records=[]
    for i in range(n):
        partition='train' if i<8000 else 'validation' if i<10000 else 'test'
        start={'train':date(2024,1,1),'validation':date(2024,5,1),'test':date(2024,9,1)}[partition]
        snapshot=start+timedelta(days=i%60);subscribed=bool((i//4)%2)
        began=snapshot-timedelta(days=rng.randint(91,1800));events=[]
        for _ in range(rng.randint(0,35)):
            day=(snapshot-timedelta(days=rng.randint(1,40))).isoformat();events.append({'date':day,'type':'visit'})
            if rng.random()<.35:events.append({'date':day,'type':'premium_read'})
        for _ in range(rng.randint(0,2) if subscribed else 0):events.append({'date':(snapshot-timedelta(days=rng.randint(1,90))).isoformat(),'type':'payment_failure'})
        values=features(events,snapshot.isoformat(),began.isoformat())
        import math
        logit=(-3+.045*values['visits_30d']+.09*values['premium_reads_30d']-.035*values['days_since_visit']) if not subscribed else (-3-.05*values['active_days_30d']+.045*values['days_since_visit']+.6*values['payment_failures_90d'])
        if rng.random()<1/(1+math.exp(-logit)):events.append({'date':(snapshot+timedelta(days=rng.randint(0,29))).isoformat(),'type':'cancel' if subscribed else 'subscribe'})
        records.append({'id':f'lifecycle-{i:05}','brand':['Aftenposten','VG','Bergens Tidende','E24'][i%4],'subscribed':subscribed,'snapshot':snapshot.isoformat(),'started':began.isoformat(),'observed_until':(snapshot+timedelta(days=30)).isoformat(),'partition':partition,'events':events})
    return records

def materialize(records):
    rows=[]
    for r in records:
        kind='churn' if r['subscribed'] else 'sales'
        rows.append({**{k:v for k,v in r.items() if k!='events'},'values':features(r['events'],r['snapshot'],r['started']),'kind':kind,'label':label(r['events'],r['snapshot'],kind,r['observed_until'])})
    for left,right in [('train','validation'),('validation','test')]:
        if max(date.fromisoformat(r['observed_until']) for r in rows if r['partition']==left)>min(date.fromisoformat(r['snapshot']) for r in rows if r['partition']==right):raise ValueError('Outcome windows overlap later partition.')
    return rows

def fingerprint(records):return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
