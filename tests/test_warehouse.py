import sys,unittest
from pathlib import Path
from decimal import Decimal
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from warehouse import decode_features,fetch_features,FEATURES
class Cursor:
    def __enter__(self):return self
    def __exit__(self,*a):pass
    def execute(self,sql,params):self.sql=sql;self.params=params
    description=[(k.upper(),) for k in ['reader_id','brand','subscribed','snapshot_date','started_date','observed_until','split_name','kind','label']+FEATURES]
    def fetchall(self):return [('r','VG',False,'2024-01-01','2023-01-01','2024-01-31','train','sales',1)+tuple(Decimal(2) for _ in FEATURES)]
class Connection:
    def __init__(self):self.c=Cursor()
    def cursor(self):return self.c
class WarehouseTests(unittest.TestCase):
    def test_bound_dataset_and_numeric_conversion(self):
        c=Connection();d=fetch_features(c,"sha' OR 1=1 --")
        self.assertNotIn("OR 1=1",c.c.sql);self.assertEqual(c.c.params,("sha' OR 1=1 --",))
        self.assertEqual(d[0]['values']['visits_30d'],2.0);self.assertEqual(d[0]['label'],1)
