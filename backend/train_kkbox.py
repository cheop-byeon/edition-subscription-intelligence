"""Train a local KKBOX churn baseline; track evidence with MLflow by default.

python backend/train_kkbox.py --input-dir /path/to/extracted/csvs --limit 50000
Only --fixture permits test data. Kaggle downloads/authentication are user-managed.
"""
import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from kkbox import (ROOT, SCHEMA, FEATURES, connect, feature_rows, import_dataset,
                   now, predict, register)

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')


def evaluate(y, p):
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind='stable')[:max(1, len(y) // 10)]
    calibration = []
    for bucket in range(10):
        selected = np.minimum((p * 10).astype(int), 9) == bucket
        if selected.any():
            calibration.append({'n': int(selected.sum()), 'predicted': float(p[selected].mean()), 'observed': float(y[selected].mean())})
    return {'auc': float(roc_auc_score(y, p)), 'average_precision': float(average_precision_score(y, p)),
            'logloss': float(log_loss(y, p, labels=[0, 1])), 'brier': float(brier_score_loss(y, p)),
            'prevalence': float(y.mean()), 'rows': len(y), 'positives': int(y.sum()),
            'top_decile_rate': float(y[order].mean()), 'top_decile_lift': float(y[order].mean() / y.mean()),
            'calibration': calibration}


def train(db_path, dataset_id, seed=42, tracking=True, work_dir=None, progress=print):
    import numpy as np
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss
    from train_external import tree_export

    rows = feature_rows(db_path, dataset_id)
    with connect(db_path) as db:
        found = db.execute('SELECT manifest FROM kkbox_datasets WHERE id=?', (dataset_id,)).fetchone()
    if found is None:
        raise ValueError('Unknown imported KKBOX dataset.')
    manifest = json.loads(found[0])
    parts = {part: [r for r in rows if r['split_name'] == part] for part in ['train', 'selection', 'calibration', 'test']}
    for part, records in parts.items():
        if len(records) < 20 or {r['is_churn'] for r in records} != {0, 1}:
            raise ValueError(f'{part} needs at least 20 subscribers and both outcomes. Increase the subscriber sample limit.')
    x = {part: np.array([[r['values'][k] for k in FEATURES] for r in records]) for part, records in parts.items()}
    y = {part: np.array([r['is_churn'] for r in records]) for part, records in parts.items()}
    if not all(np.isfinite(v).all() for v in x.values()):
        raise ValueError('Non-finite SQL feature.')
    # Selection and calibration have their own subscribers; the test partition
    # is never used for fitting, candidate selection, calibration or thresholds.
    candidates = {
        'logistic': make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=seed)),
        'hist_gradient_boosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
            min_samples_leaf=30, learning_rate=.07, l2_regularization=2, early_stopping=False, random_state=seed)}
    selection = {}
    for name, model in candidates.items():
        progress(f'Training {name} on {len(x["train"]):,} subscribers…')
        model.fit(x['train'], y['train'])
        selection[name] = float(log_loss(y['selection'], model.predict_proba(x['selection'])[:, 1]))
    winner = min(selection, key=selection.get)
    model = candidates[winner]
    calibrator = LogisticRegression(C=1000, max_iter=500).fit(model.decision_function(x['calibration']).reshape(-1, 1), y['calibration'])
    calibration = {'intercept': float(calibrator.intercept_[0]), 'slope': float(calibrator.coef_[0, 0])}
    if winner == 'logistic':
        scaler, logistic = model.steps[0][1], model.steps[1][1]
        weights = logistic.coef_[0] / scaler.scale_
        exported = {'type': winner, 'weights': weights.tolist(), 'intercept': float(logistic.intercept_[0] - weights @ scaler.mean_)}
    else:
        exported = {'type': winner, **tree_export(model)}
    probabilities = calibrator.predict_proba(model.decision_function(x['test']).reshape(-1, 1))[:, 1]
    baseline = np.full(len(y['test']), y['train'].mean())
    version = 'kkbox-billing-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    artifact = {'schema': SCHEMA, 'version': version, 'dataset_id': dataset_id,
                'source_kind': manifest['source_kind'], 'source_url': manifest['source_url'],
                'target': manifest['target'], 'snapshot': manifest['snapshot'], 'feature_names': FEATURES,
                'model': exported, 'selected_family': winner, 'calibration': calibration,
                'selection_logloss': selection, 'selection_rule': 'Lowest uncalibrated log loss on the selection partition.',
                'metrics': evaluate(y['test'], probabilities), 'constant_baseline': evaluate(y['test'], baseline),
                'uncalibrated_models': {name: evaluate(y['test'], candidate.predict_proba(x['test'])[:, 1]) for name, candidate in candidates.items()},
                'partitions': {name: {'n': len(target), 'positives': int(target.sum())} for name, target in y.items()},
                'validation': manifest['split_method'], 'manifest': manifest,
                'library_versions': {'sklearn': sklearn.__version__, 'numpy': np.__version__},
                'limitations': ['One historical cohort with subscriber holdouts; no claim of future-month generalisation.',
                    'Billing-only features; listening logs are not ingested in this first baseline.',
                    'Official churn labels are not sales conversion, cancellation flags or causal offer effects.',
                    'Same-day transaction order is approximated from dates and recorded fields.',
                    'Estimated churn risk does not establish which retention action would help.'],
                'created_at': now()}
    for record, expected in zip(parts['test'][:300], probabilities[:300]):
        if abs(predict(record['values'], artifact) - float(expected)) > 1e-9:
            raise AssertionError('Portable inference differs from the fitted sklearn model.')
    artifact['export_verified_rows'] = min(300, len(probabilities))
    if tracking:
        import mlflow
        from mlflow.tracking import MlflowClient
        tracking_path = Path(db_path).resolve().parent
        mlflow.set_tracking_uri('sqlite:///' + str(tracking_path / 'mlflow.sqlite3'))
        client = MlflowClient()
        experiment = client.get_experiment_by_name('Edition KKBOX churn')
        experiment_id = experiment.experiment_id if experiment else client.create_experiment(
            'Edition KKBOX churn', artifact_location=(tracking_path / 'mlflow-artifacts' / 'kkbox').as_uri())
        with mlflow.start_run(experiment_id=experiment_id, run_name=version) as run:
            mlflow.set_tags({'source_kind': manifest['source_kind'], 'task': 'subscription_churn', 'data_domain': 'KKBOX music', 'test_used_for_selection': 'false'})
            mlflow.log_params({'dataset_id': dataset_id, 'seed': seed, 'schema': SCHEMA,
                               'snapshot': manifest['snapshot'], 'selected_family': winner,
                               'feature_count': len(FEATURES), 'validation': 'subscriber_hash_60_15_10_15'})
            mlflow.log_metrics({'test_' + k: v for k, v in artifact['metrics'].items() if isinstance(v, (int, float))})
            mlflow.log_metrics({'selection_logloss_' + k: v for k, v in selection.items()})
            mlflow.log_metrics({'baseline_' + k: v for k, v in artifact['constant_baseline'].items() if isinstance(v, (int, float))})
            mlflow.log_dict(manifest, 'dataset_manifest.json')
            mlflow.log_dict(artifact, 'portable_model.json')
            mlflow.log_dict({'sqlite': (ROOT / 'sql/kkbox_features.sql').read_text(), 'snowflake': (ROOT / 'sql/kkbox_snowflake_features.sql').read_text()}, 'feature_sql.json')
            artifact['mlflow_run_id'] = run.info.run_id
            artifact['mlflow_experiment_id'] = str(experiment_id)
        artifact['mlflow_tracking_uri'] = 'sqlite:///' + str(tracking_path / 'mlflow.sqlite3')
    else:
        artifact['mlflow_run_id'] = None
    if work_dir:
        output = Path(work_dir) / 'runs' / version
        output.mkdir(parents=True, exist_ok=True)
        (output / 'model.json').write_text(json.dumps(artifact, indent=2, allow_nan=False))
    model_id = register(db_path, artifact)
    progress(f'Trained and registered model #{model_id}; held-out log loss {artifact["metrics"]["logloss"]:.4f}.')
    return {'model_id': model_id, 'artifact': artifact}


def write_job(path, payload):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload))
    temp.replace(path)


def pipeline(input_dir, db_path, work_dir, limit=50000, seed=42, tracking=True, fixture=False):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    status_path = work_dir / 'job.json'
    lock = work_dir / 'job.lock'
    if lock.exists():
        owner = lock.read_text().strip()
        if not owner:
            raise ValueError('A KKBOX pipeline is starting. Refresh its status shortly.')
        try:
            os.kill(int(owner), 0)
        except (OSError, ValueError):
            lock.unlink(missing_ok=True)
        else:
            raise ValueError('A KKBOX pipeline is already running in this working directory.')
    # Exclusive creation prevents two UI sessions from starting concurrent jobs.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid()))
    payload = {'pid': os.getpid(), 'status': 'running', 'started_at': now(), 'source_kind': 'synthetic_test_fixture' if fixture else 'official_kaggle'}
    def progress(message):
        payload['message'] = message
        payload['updated_at'] = now()
        write_job(status_path, payload)
        print(message, flush=True)
    try:
        progress('Import starting…')
        manifest = import_dataset(input_dir, db_path, limit, seed, payload['source_kind'], progress)
        payload['dataset_id'] = manifest['id']
        result = train(db_path, manifest['id'], seed, tracking, work_dir, progress)
        payload.update(status='complete', model_id=result['model_id'], message='KKBOX churn model ready.', finished_at=now())
        write_job(status_path, payload)
        return result
    except Exception as exc:
        payload.update(status='failed', message=str(exc), finished_at=now())
        write_job(status_path, payload)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--db', type=Path, default=Path(os.environ.get('EDITION_DB', ROOT / 'data/edition.sqlite3')))
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'data/kkbox')
    parser.add_argument('--limit', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-mlflow', action='store_true', help='Only for tests/development; normal runs track by default.')
    parser.add_argument('--fixture', action='store_true', help='Mark synthetic integration-test files honestly; never a real-data result.')
    args = parser.parse_args()
    try:
        result = pipeline(args.input_dir, args.db, args.work_dir, args.limit, args.seed, not args.no_mlflow, args.fixture)
        print(json.dumps({'model_id': result['model_id'], 'metrics': result['artifact']['metrics']}))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
