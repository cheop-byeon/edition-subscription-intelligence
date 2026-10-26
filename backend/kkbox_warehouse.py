"""Explicit Snowflake sandbox bridge for the same imported KKBOX billing data.

No cloud connection is attempted by plan or by opening the Edition UI.
"""
import argparse
import json
from pathlib import Path
from kkbox import ROOT, FEATURES, connect, feature_rows


def initialise(connection):
    with connection.cursor() as cursor:
        for filename in ['kkbox_snowflake_schema.sql', 'kkbox_snowflake_features.sql']:
            for statement in (ROOT / 'sql' / filename).read_text().split(';'):
                if statement.strip():
                    cursor.execute(statement)


def load(connection, db_path, dataset_id):
    with connect(db_path) as local:
        if not local.execute('SELECT 1 FROM kkbox_datasets WHERE id=?', (dataset_id,)).fetchone():
            raise ValueError('Unknown local dataset; import the official files first.')
        with connection.cursor() as cursor:
            cursor.execute('BEGIN')
            try:
                for table in ['kkbox_labels', 'kkbox_transactions']:
                    cursor.execute(f'DELETE FROM {table} WHERE dataset_id=%s', (dataset_id,))
                    # The first column is the dataset ID in both local tables.
                    source = local.execute(f'SELECT * FROM {table} WHERE dataset_id=?', (dataset_id,))
                    width = len(source.description)
                    while batch := source.fetchmany(2000):
                        cursor.executemany(f'INSERT INTO {table} VALUES (' + ','.join(['%s'] * width) + ')', [tuple(r) for r in batch])
                cursor.execute('COMMIT')
            except Exception:
                cursor.execute('ROLLBACK')
                raise


def verify(connection, db_path, dataset_id):
    """Compare every warehouse feature and label with the local SQL result."""
    expected = {r['msno']: r for r in feature_rows(db_path, dataset_id)}
    with connection.cursor() as cursor:
        cursor.execute('SELECT * FROM KKBOX_FEATURES WHERE DATASET_ID=%s ORDER BY MSNO', (dataset_id,))
        columns = [c[0].lower() for c in cursor.description]
        count = 0
        seen = set()
        for values in cursor:
            row = dict(zip(columns, values))
            user = row['msno']
            if user not in expected or user in seen:
                raise ValueError('Warehouse subscriber identity mismatch.')
            local = expected[user]
            if row['split_name'] != local['split_name'] or int(row['is_churn']) != local['is_churn']:
                raise ValueError('Warehouse label/split mismatch.')
            if any(abs(float(row[k]) - local['values'][k]) > 1e-7 for k in FEATURES):
                raise ValueError('Warehouse feature parity failed.')
            seen.add(user)
            count += 1
    if count != len(expected) or count == 0:
        raise ValueError('Warehouse feature row count mismatch.')
    return {'rows_verified': count, 'dataset_id': dataset_id}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'init', 'load', 'verify'])
    parser.add_argument('--connection', default='edition')
    parser.add_argument('--db', type=Path, default=ROOT / 'data/edition.sqlite3')
    parser.add_argument('--dataset')
    args = parser.parse_args()
    if args.command == 'plan':
        print('No connection opened. init creates KKBOX_LABELS/KKBOX_TRANSACTIONS and KKBOX_FEATURES in your configured sandbox. load uploads only the selected local dataset; verify compares SQL features. Requires a Snowflake account; cloud queries may consume credits.')
    else:
        if args.command in ['load', 'verify'] and not args.dataset:
            parser.error('--dataset is required')
        from warehouse import connect as cloud_connect
        with cloud_connect(args.connection) as connection:
            if args.command == 'init':
                initialise(connection)
                print('KKBOX sandbox schema prepared.')
            elif args.command == 'load':
                load(connection, args.db, args.dataset)
                print('Selected local KKBOX dataset uploaded.')
            else:
                print(json.dumps(verify(connection, args.db, args.dataset)))
