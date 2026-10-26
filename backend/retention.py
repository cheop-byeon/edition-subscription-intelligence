"""Local research audiences from the actual KKBOX serving model."""
import json
import math
from kkbox import connect, holdout_rows, predict, status, now


def audience(db_path):
    model = status(db_path)['model']
    if not model:
        return None, []
    records = holdout_rows(db_path, model['dataset_id'], limit=500000)
    rows = [{'subscriber': r['msno'], 'risk': predict(r['values'], model['artifact'])}
            for r in records]
    return model, sorted(rows, key=lambda row: (-row['risk'], row['subscriber']))


def initialize(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='retention_drafts'").fetchone():
        db.execute('''CREATE TABLE retention_drafts(
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, model_id INTEGER NOT NULL,
            dataset_id TEXT NOT NULL, threshold REAL NOT NULL, members TEXT NOT NULL,
            created_at TEXT NOT NULL)''')


def drafts(db_path):
    with connect(db_path) as db:
        initialize(db)
        return [dict(r) | {'members': json.loads(r['members'])}
                for r in db.execute('SELECT * FROM retention_drafts ORDER BY id DESC')]


def save_draft(db_path, name, threshold):
    name = name.strip()
    if not name or len(name) > 120:
        raise ValueError('Enter a draft name of 1–120 characters.')
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between zero and one.')
    model, rows = audience(db_path)
    if not model:
        raise ValueError('Train a churn model first.')
    members = [r for r in rows if r['risk'] >= threshold]
    with connect(db_path) as db:
        initialize(db)
        return db.execute('INSERT INTO retention_drafts(name,model_id,dataset_id,threshold,members,created_at) VALUES (?,?,?,?,?,?)',
                          (name, model['id'], model['dataset_id'], threshold,
                           json.dumps(members, allow_nan=False), now())).lastrowid
