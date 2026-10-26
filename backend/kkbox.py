"""KKBOX billing features, SQL storage and portable inference (standard library only).

The original competition labels are kept intact. No sales/offer-effect label is
inferred from churn, and this namespace never replaces simulated commercial data.
"""
import csv
import gzip
import hashlib
import heapq
import io
import json
import math
import os
import sqlite3
import subprocess
import sys
import zipfile
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data'
SCHEMA = 'kkbox-billing-v1'
FEATURES = ['transactions_90d', 'paid_total_90d', 'paid_mean_90d',
            'last_plan_days', 'last_list_price', 'last_amount_paid',
            'last_auto_renew', 'last_cancel_flag', 'days_since_transaction',
            'days_to_expiry', 'history_days', 'last_discount']
TX_FIELDS = ['msno', 'transaction_date', 'membership_expire_date',
             'payment_plan_days', 'plan_list_price', 'actual_amount_paid',
             'is_auto_renew', 'is_cancel']


def now():
    return datetime.now(timezone.utc).isoformat()


class Connection(sqlite3.Connection):
    def __exit__(self, *args):
        try:
            return super().__exit__(*args)
        finally:
            self.close()


def connect(db_path):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(db_path, timeout=60, factory=Connection)
    db.row_factory = sqlite3.Row
    objects={row[0] for row in db.execute("SELECT name FROM sqlite_master")}
    required={'kkbox_datasets','kkbox_labels','kkbox_transactions','kkbox_tx_reader',
              'kkbox_features','kkbox_models','kkbox_predictions'}
    if required <= objects:
        return db
    db.executescript('''
      CREATE TABLE IF NOT EXISTS kkbox_datasets (
        id TEXT PRIMARY KEY, manifest TEXT NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS kkbox_labels (
        dataset_id TEXT NOT NULL, msno TEXT NOT NULL, is_churn INTEGER NOT NULL,
        split_name TEXT NOT NULL, snapshot_date TEXT NOT NULL,
        PRIMARY KEY(dataset_id, msno));
      CREATE TABLE IF NOT EXISTS kkbox_transactions (
        dataset_id TEXT NOT NULL, msno TEXT NOT NULL, transaction_date TEXT NOT NULL,
        membership_expire_date TEXT NOT NULL, payment_plan_days REAL NOT NULL,
        plan_list_price REAL NOT NULL, actual_amount_paid REAL NOT NULL,
        is_auto_renew INTEGER NOT NULL, is_cancel INTEGER NOT NULL,
        UNIQUE(dataset_id,msno,transaction_date,membership_expire_date,
               payment_plan_days,plan_list_price,actual_amount_paid,is_auto_renew,is_cancel));
      CREATE INDEX IF NOT EXISTS kkbox_tx_reader ON kkbox_transactions(dataset_id,msno,transaction_date);
      CREATE TABLE IF NOT EXISTS kkbox_features (
        dataset_id TEXT NOT NULL, msno TEXT NOT NULL, split_name TEXT NOT NULL,
        is_churn INTEGER NOT NULL, values_json TEXT NOT NULL,
        PRIMARY KEY(dataset_id,msno));
      CREATE TABLE IF NOT EXISTS kkbox_models (
        id INTEGER PRIMARY KEY, dataset_id TEXT NOT NULL, artifact TEXT NOT NULL,
        active INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS kkbox_predictions (
        id INTEGER PRIMARY KEY, model_id INTEGER NOT NULL, dataset_id TEXT NOT NULL,
        msno TEXT NOT NULL, probability REAL NOT NULL, created_at TEXT NOT NULL);
    ''')
    return db


def feature_sql(dialect='sqlite'):
    """Same point-in-time features for SQLite and the prepared Snowflake view."""
    if dialect not in ['sqlite', 'snowflake']:
        raise ValueError('Unsupported SQL dialect.')
    def days(first, last):
        return f'(julianday({last})-julianday({first}))' if dialect == 'sqlite' else f"DATEDIFF('day',{first},{last})"
    age = days('t.transaction_date', 'l.snapshot_date')
    recent = f'{age} <= 90'
    return f'''
      WITH ranked AS (
        SELECT t.*, ROW_NUMBER() OVER (
          PARTITION BY t.dataset_id,t.msno
          ORDER BY t.transaction_date DESC,t.membership_expire_date DESC,
                   t.is_cancel ASC,t.plan_list_price DESC,t.actual_amount_paid DESC,
                   t.payment_plan_days DESC,t.is_auto_renew DESC) AS rn
        FROM kkbox_transactions t JOIN kkbox_labels l
          ON t.dataset_id=l.dataset_id AND t.msno=l.msno
        WHERE t.transaction_date < l.snapshot_date
      ), history AS (
        SELECT t.dataset_id,t.msno,MIN(t.transaction_date) AS first_date,
          SUM(CASE WHEN {recent} THEN 1 ELSE 0 END) AS transactions_90d,
          SUM(CASE WHEN {recent} THEN t.actual_amount_paid ELSE 0 END) AS paid_total_90d,
          AVG(CASE WHEN {recent} THEN t.actual_amount_paid ELSE NULL END) AS paid_mean_90d
        FROM kkbox_transactions t JOIN kkbox_labels l
          ON t.dataset_id=l.dataset_id AND t.msno=l.msno
        WHERE t.transaction_date < l.snapshot_date GROUP BY t.dataset_id,t.msno
      )
      SELECT l.dataset_id,l.msno,l.split_name,l.is_churn,
        h.transactions_90d,h.paid_total_90d,COALESCE(h.paid_mean_90d,0) AS paid_mean_90d,
        r.payment_plan_days AS last_plan_days,r.plan_list_price AS last_list_price,
        r.actual_amount_paid AS last_amount_paid,r.is_auto_renew AS last_auto_renew,
        r.is_cancel AS last_cancel_flag,
        {days('r.transaction_date','l.snapshot_date')} AS days_since_transaction,
        {days('l.snapshot_date','r.membership_expire_date')} AS days_to_expiry,
        {days('h.first_date','l.snapshot_date')} AS history_days,
        r.plan_list_price-r.actual_amount_paid AS last_discount
      FROM kkbox_labels l JOIN ranked r
        ON l.dataset_id=r.dataset_id AND l.msno=r.msno AND r.rn=1
      JOIN history h ON l.dataset_id=h.dataset_id AND l.msno=h.msno
    '''


@contextmanager
def csv_source(path):
    """Stream a CSV, gzipped CSV, or single-CSV zip. Never extracts archives."""
    path = Path(path)
    if path.suffix == '.7z':
        raise ValueError(f'Extract {path.name} first with your archive utility, then select the CSV folder.')
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
            if len(names) != 1:
                raise ValueError('Extract multi-file archives first; each ZIP input must contain one CSV.')
            with archive.open(names[0]) as raw, io.TextIOWrapper(raw, encoding='utf-8-sig', newline='') as handle:
                yield csv.DictReader(handle)
    else:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as handle:
            yield csv.DictReader(handle)


def source_file(folder, stem):
    for suffix in ['.csv', '.csv.gz', '.csv.zip', '.zip']:
        candidate = folder / (stem + suffix)
        if candidate.is_file():
            return candidate
    if (folder / (stem + '.csv.7z')).exists():
        raise ValueError(f'Extract {stem}.csv.7z into this folder first.')
    return None


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(msno, seed=42):
    # Split by subscriber, independently of label, and never by transaction row.
    bucket = int(hashlib.sha256(f'{seed}:{msno}'.encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 60 else 'selection' if bucket < 75 else 'calibration' if bucket < 85 else 'test'


def iso_date(value):
    return date.fromisoformat(value) if '-' in value else datetime.strptime(value, '%Y%m%d').date()


def import_dataset(folder, db_path, limit=50000, seed=42, source_kind='official_kaggle', progress=print):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise ValueError('Choose a folder containing the extracted official KKBOX CSV files.')
    if source_kind not in ['official_kaggle', 'synthetic_test_fixture']:
        raise ValueError('Unrecognised data source kind.')
    if not 500 <= limit <= 500000:
        raise ValueError('Choose a subscriber limit between 500 and 500,000.')
    labels_path = source_file(folder, 'train_v2') or source_file(folder, 'train')
    history_path = source_file(folder, 'transactions')
    transactions = [p for p in [history_path, source_file(folder, 'transactions_v2')] if p]
    if labels_path is None or history_path is None:
        raise ValueError('Need train_v2.csv (or train.csv) and transactions.csv; transactions_v2.csv may be added. The engagement-labelled benchmark CSV is a different dataset.')
    cohort = '2017-03-01' if labels_path.name.startswith('train_v2.') else '2017-02-01'
    progress('Hashing source files for reproducible lineage…')
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': file_hash(p)} for p in [labels_path] + transactions]
    identity = {'files': files, 'limit': limit, 'seed': seed, 'schema': SCHEMA, 'snapshot': cohort, 'source_kind': source_kind}
    dataset_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    progress('Selecting a deterministic subscriber sample from the official labels…')
    seen = set()
    def labels():
        with csv_source(labels_path) as reader:
            if not {'msno', 'is_churn'}.issubset(reader.fieldnames or []):
                raise ValueError('Labels must contain msno and is_churn.')
            for row in reader:
                user = row['msno']
                if not user or user in seen or row['is_churn'] not in ['0', '1']:
                    raise ValueError('Invalid or duplicate subscriber/label in the labels file.')
                seen.add(user)
                rank = hashlib.sha256(('sample:' + str(seed) + ':' + user).encode()).hexdigest()
                yield (rank, user, int(row['is_churn']))
    selected = heapq.nsmallest(limit, labels())
    users = {user for _, user, _ in selected}
    db = connect(db_path)
    try:
        # A failed import rolls back without disturbing any earlier dataset/model.
        with db:
            for table in ['kkbox_features', 'kkbox_transactions', 'kkbox_labels']:
                db.execute(f'DELETE FROM {table} WHERE dataset_id=?', (dataset_id,))
            db.executemany('INSERT INTO kkbox_labels VALUES (?,?,?,?,?)',
                           [(dataset_id, user, target, split_for(user, seed), cohort) for _, user, target in selected])
            scanned = matched = discarded_future = 0
            for path in transactions:
                progress(f'Streaming {path.name}; only sampled subscribers and pre-cutoff rows are retained…')
                batch = []
                with csv_source(path) as reader:
                    if not set(TX_FIELDS).issubset(reader.fieldnames or []):
                        raise ValueError(f'{path.name} does not contain the official transaction columns.')
                    for row in reader:
                        scanned += 1
                        if row['msno'] not in users:
                            continue
                        tx_date = iso_date(row['transaction_date']).isoformat()
                        if tx_date >= cohort:
                            discarded_future += 1
                            continue
                        expiry = iso_date(row['membership_expire_date']).isoformat()
                        nums = [float(row[k]) for k in TX_FIELDS[3:]]
                        if any(not math.isfinite(x) or x < 0 for x in nums) or nums[-2] not in [0, 1] or nums[-1] not in [0, 1]:
                            raise ValueError('Invalid numeric transaction data.')
                        batch.append((dataset_id, row['msno'], tx_date, expiry, *nums))
                        matched += 1
                        if len(batch) >= 5000:
                            db.executemany('INSERT OR IGNORE INTO kkbox_transactions VALUES (?,?,?,?,?,?,?,?,?)', batch)
                            batch.clear()
                        if scanned % 1000000 == 0:
                            progress(f'Scanned {scanned:,} transactions; retained {matched:,} before duplicate removal.')
                    if batch:
                        db.executemany('INSERT OR IGNORE INTO kkbox_transactions VALUES (?,?,?,?,?,?,?,?,?)', batch)
            progress('Building point-in-time billing features in SQL…')
            rows = db.execute('SELECT * FROM (' + feature_sql() + ') WHERE dataset_id=?', (dataset_id,)).fetchall()
            if not rows:
                raise ValueError('No sampled subscribers have billing history before the cohort cutoff. Check that transactions.csv contains the original historical records.')
            db.executemany('INSERT INTO kkbox_features VALUES (?,?,?,?,?)',
                           [(dataset_id, r['msno'], r['split_name'], r['is_churn'], json.dumps({k: float(r[k]) for k in FEATURES}, allow_nan=False)) for r in rows])
            stored = db.execute('SELECT COUNT(*) FROM kkbox_transactions WHERE dataset_id=?', (dataset_id,)).fetchone()[0]
            manifest = {**identity, 'id': dataset_id, 'source_url': SOURCE_URL,
                        'label_file': labels_path.name, 'labelled_subscribers_in_file': len(seen),
                        'sampled_subscribers': len(selected), 'feature_rows': len(rows),
                        'excluded_without_prior_transaction': len(selected) - len(rows),
                        'scanned_transactions': scanned, 'stored_transactions': stored,
                        'exact_duplicates_removed': matched - stored,
                        'post_cutoff_transactions_excluded': discarded_future,
                        'target': 'Official is_churn: no valid renewal within 30 days after membership expiry; not a cancellation flag.',
                        'split_method': 'Subscriber-disjoint hash split within one labelled cohort: 60/15/10/15. Not a future-month evaluation.',
                        'feature_cutoff': 'Transaction date strictly before snapshot. No current/future is_churn is used as a feature.',
                        'same_day_order': 'Latest transaction day; largest expiry, non-cancel, list price, amount, plan days, auto-renew. Intraday ordering is unavailable.',
                        'feature_scope': 'Billing-only first baseline; no listening logs, demographics or inferred newspaper features.',
                        'created_at': now()}
            db.execute('INSERT OR REPLACE INTO kkbox_datasets VALUES (?,?,?)', (dataset_id, json.dumps(manifest, allow_nan=False), now()))
        return manifest
    finally:
        db.close()


def feature_rows(db_path, dataset_id):
    with connect(db_path) as db:
        return [dict(r) | {'values': json.loads(r['values_json'])} for r in db.execute(
            'SELECT * FROM kkbox_features WHERE dataset_id=? ORDER BY msno', (dataset_id,))]


def predict(values, artifact):
    if artifact.get('schema') != SCHEMA or set(values) != set(FEATURES):
        raise ValueError('Expected the KKBOX billing feature contract.')
    x = [values[k] for k in FEATURES]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in x):
        raise ValueError('KKBOX features must be finite numbers.')
    model = artifact['model']
    if model['type'] == 'logistic':
        raw = model['intercept'] + sum(w * v for w, v in zip(model['weights'], x))
    elif model['type'] == 'hist_gradient_boosting':
        raw = model['base']
        for nodes in model['trees']:
            j = 0
            while not nodes[j]['leaf']:
                node = nodes[j]
                j = node['left'] if x[node['feature']] <= node['threshold'] else node['right']
            raw += nodes[j]['value']
    else:
        raise ValueError('Unknown KKBOX model type.')
    z = artifact['calibration']['slope'] * raw + artifact['calibration']['intercept']
    return 1 / (1 + math.exp(-max(-40, min(40, z))))


def register(db_path, artifact):
    if artifact.get('schema') != SCHEMA or artifact.get('feature_names') != FEATURES:
        raise ValueError('Invalid KKBOX model contract.')
    with connect(db_path) as db:
        if not db.execute('SELECT 1 FROM kkbox_datasets WHERE id=?', (artifact['dataset_id'],)).fetchone():
            raise ValueError('Model dataset is not present in this database.')
        db.execute('UPDATE kkbox_models SET active=0')
        return db.execute('INSERT INTO kkbox_models(dataset_id,artifact,active,created_at) VALUES (?,?,1,?)',
                          (artifact['dataset_id'], json.dumps(artifact, allow_nan=False), now())).lastrowid


def status(db_path):
    with connect(db_path) as db:
        datasets = [json.loads(r[0]) for r in db.execute('SELECT manifest FROM kkbox_datasets ORDER BY created_at DESC')]
        row = db.execute('SELECT * FROM kkbox_models WHERE active=1 ORDER BY id DESC LIMIT 1').fetchone()
        model = None if row is None else dict(row) | {'artifact': json.loads(row['artifact'])}
    return {'datasets': datasets, 'model': model}


def holdout_rows(db_path, dataset_id, limit=200):
    with connect(db_path) as db:
        return [dict(r) | {'values': json.loads(r['values_json'])} for r in db.execute(
            'SELECT * FROM kkbox_features WHERE dataset_id=? AND split_name="test" ORDER BY msno LIMIT ?', (dataset_id, limit))]


def save_prediction(db_path, model_id, msno):
    with connect(db_path) as db:
        row = db.execute('SELECT dataset_id,artifact FROM kkbox_models WHERE id=?', (model_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown KKBOX model.')
        feature = db.execute('SELECT values_json FROM kkbox_features WHERE dataset_id=? AND msno=?', (row['dataset_id'], msno)).fetchone()
        if feature is None:
            raise ValueError('Subscriber does not belong to this model dataset.')
        probability = predict(json.loads(feature[0]), json.loads(row['artifact']))
        prediction_id = db.execute('INSERT INTO kkbox_predictions(model_id,dataset_id,msno,probability,created_at) VALUES (?,?,?,?,?)',
                                   (model_id, row['dataset_id'], msno, probability, now())).lastrowid
    return {'id': prediction_id, 'probability': probability}


def job_status(work_dir):
    path = Path(work_dir) / 'job.json'
    if not path.exists():
        return None
    try:
        job = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if job['status'] in ['running', 'starting']:
        try:
            os.kill(job['pid'], 0)
        except (OSError, KeyError):
            job = {**job, 'status': 'interrupted', 'message': 'The training process stopped. Start another run when ready.'}
    return job


def start_job(input_dir, db_path, work_dir, limit=50000):
    from importlib.util import find_spec
    if not all(find_spec(name) for name in ['numpy', 'sklearn', 'mlflow']):
        raise ValueError('Training dependencies are missing. Install requirements-kkbox.txt into the Python environment that starts Edition.')
    input_dir = Path(input_dir).expanduser().resolve()
    if not input_dir.is_dir():
        raise ValueError('The source folder does not exist.')
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    old_job = job_status(work_dir)
    if old_job and old_job['status'] in ['running', 'starting']:
        raise ValueError('A KKBOX training run is already active. Refresh its status below.')
    command = [sys.executable, str(ROOT / 'backend/train_kkbox.py'), '--input-dir', str(input_dir),
               '--db', str(Path(db_path).resolve()), '--work-dir', str(work_dir.resolve()), '--limit', str(int(limit))]
    with (work_dir / 'job.log').open('w') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True, env={**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2'})
    return process.pid
