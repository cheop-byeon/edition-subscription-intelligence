"""Optional Snowflake adapter. No connection is made unless explicitly requested."""
import argparse, json
from pathlib import Path
from lifecycle import generate, materialize, fingerprint
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['visits_30d','active_days_30d','premium_reads_30d','tenure_months','days_since_visit','payment_failures_90d']
def connect(name='edition'):
    import snowflake.connector
    return snowflake.connector.connect(connection_name=name)
def init(connection):
    with connection.cursor() as cursor:
        for name in ['snowflake_schema.sql','snowflake_features.sql']:
            sql=(ROOT/'sql'/name).read_text()
            for statement in sql.split(';'):
                if statement.strip():cursor.execute(statement)
def load_simulated(connection,seed=42):
    raw=generate(seed);materialize(raw);dataset=fingerprint(raw)
    # Transactional replacement of this dataset only makes reruns idempotent.
    with connection.cursor() as cursor:
        cursor.execute('BEGIN')
        try:
            for table in ['EDITION_EVENTS','EDITION_SNAPSHOTS']:
                cursor.execute('DELETE FROM '+table+' WHERE DATASET_ID = %s',(dataset,))
            snapshots=[(dataset,r['id'],r['brand'],r['subscribed'],r['snapshot'],r['started'],r['observed_until'],r['partition']) for r in raw]
            events=[(dataset,r['id'],e['date'],e['type']) for r in raw for e in r['events']]
            for table,values,slots in [('EDITION_SNAPSHOTS',snapshots,8),('EDITION_EVENTS',events,4)]:
                query='INSERT INTO '+table+' VALUES ('+','.join(['%s']*slots)+')'
                for start in range(0,len(values),2000):cursor.executemany(query,values[start:start+2000])
            cursor.execute('COMMIT')
        except Exception:cursor.execute('ROLLBACK');raise
    return dataset

def fetch_features(connection,dataset):
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM EDITION_FEATURES WHERE DATASET_ID = %s ORDER BY READER_ID',(dataset,))
        columns=[d[0].lower() for d in cursor.description];records=[dict(zip(columns,row)) for row in cursor.fetchall()]
    return decode_features(records)

def decode_features(records):
    rows=[]
    for r in records:
        rows.append({'id':r['reader_id'],'brand':r['brand'],'subscribed':bool(r['subscribed']),
            'snapshot':str(r['snapshot_date']),'started':str(r['started_date']),'observed_until':str(r['observed_until']),
            'partition':r['split_name'],'kind':r['kind'],'label':int(r['label']),
            'values':{k:float(r[k]) for k in FEATURES}})
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['plan','init','load','export']);p.add_argument('--connection',default='edition');p.add_argument('--seed',type=int,default=42);p.add_argument('--dataset');p.add_argument('--output',type=Path,default=ROOT/'data/training/snowflake_features.jsonl');a=p.parse_args()
    if a.command=='plan':
        print('No connection opened. init creates two tables and one feature view in the configured sandbox schema. load replaces only the seeded simulated dataset. export reads its feature view. A configured Snowflake account and warehouse are required.');raise SystemExit(0)
    if a.command=='export' and not a.dataset:p.error('export requires --dataset')
    with connect(a.connection) as connection:
        if a.command=='init':init(connection);print('Sandbox tables/view prepared.')
        elif a.command=='load':print(load_simulated(connection,a.seed))
        else:
            rows=fetch_features(connection,a.dataset)
            if not rows:raise ValueError('No features found for dataset.')
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text('\n'.join(json.dumps(r) for r in rows)+'\n');print('Exported',len(rows),'rows to',a.output)
