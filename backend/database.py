"""Laptop-local profile storage and an auditable inference history."""
from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {'profiles','predictions'} <= tables:
                return
            db.executescript('''
                CREATE TABLE IF NOT EXISTS profiles (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, date TEXT,
                    label INTEGER, features TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    model_version TEXT NOT NULL, probability REAL NOT NULL,
                    features TEXT NOT NULL);
            ''')
    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path, timeout=30)
        try:
            with db:yield db
        finally:db.close()
    def seed(self, profiles, source):
        with self.connect() as db:
            db.executemany('INSERT OR IGNORE INTO profiles VALUES (?,?,?,?,?)',
                ((p['id'], source, p.get('date'), p.get('label'), json.dumps(p['values'], allow_nan=False)) for p in profiles))
    def profiles(self, limit=80):
        with self.connect() as db:
            rows=db.execute('SELECT id,source,date,label,features FROM profiles ORDER BY CASE WHEN source="downloaded_benchmark" THEN 0 ELSE 1 END,id LIMIT ?', (limit,)).fetchall()
        return [dict(id=r[0],source=r[1],date=r[2],label=r[3],values=json.loads(r[4])) for r in rows]
    def record(self, values, result):
        with self.connect() as db:
            return db.execute('INSERT INTO predictions(model_version,probability,features) VALUES (?,?,?)',
                (result['model_version'],result['probability'],json.dumps(values,allow_nan=False))).lastrowid
    def summary(self):
        with self.connect() as db:
            return {'engine':'SQLite','profiles':db.execute('SELECT COUNT(*) FROM profiles').fetchone()[0],
                'predictions':db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],
                'sources':dict(db.execute('SELECT source,COUNT(*) FROM profiles GROUP BY source'))}
