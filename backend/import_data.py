"""Import the author-published benchmark ZIP locally, without ML dependencies."""
import argparse, csv, io, json, zipfile
from pathlib import Path
from database import Store
from external_engine import validate_profile
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--database',type=Path,default=ROOT/'data/edition.sqlite3')
    args=parser.parse_args()
    model=json.loads((ROOT/'artifacts/external_model.json').read_text())['model']
    store=Store(args.database)
    with zipfile.ZipFile(args.archive) as archive:
        files=[n for n in archive.namelist() if n.endswith('.csv')]
        if len(files)!=1:raise ValueError('Expected exactly one CSV in archive.')
        with archive.open(files[0]) as raw:
            reader=csv.DictReader(io.TextIOWrapper(raw,encoding='utf-8-sig'),quotechar="'")
            batch=[]
            for i,row in enumerate(reader,1):
                values={k:float(row[k]) if row[k].strip() else None for k in model['feature_names']}
                validate_profile(values,model)
                batch.append(dict(id=f'benchmark-{i:07}',date=row['date_converted'],label=int(float(row['subscription'])),values=values))
                if len(batch)==5000:store.seed(batch,'downloaded_benchmark');batch=[]
            store.seed(batch,'downloaded_benchmark')
    print(json.dumps(store.summary()))
if __name__=='__main__':main()
