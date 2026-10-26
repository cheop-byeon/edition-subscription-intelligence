# Original KKBOX churn workflow

This is the original WSDM 2018 KKBOX subscription-churn task, not Edition's older
KKBOX-derived engagement dataset with synthetic subscription labels. No original
KKBOX records or real-data accuracy results are bundled with the repository.

## Get the official records

1. Open https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data.
2. Sign in and review/accept the competition rules yourself. Credentials stay
   outside this project and are never requested in chat.
3. Download and extract `train_v2.csv` and `transactions.csv` into one directory.
   `transactions_v2.csv` can also be supplied. CSV, CSV.GZ and a single-CSV ZIP
   are supported; extract `.7z` and multi-file archives first.
4. In Edition, select **Models & data**, enter that directory, confirm the source
   and click **Import and train locally**. Refresh the run status for progress.

The first baseline deliberately uses billing history only. `user_logs.csv`
(very large listening histories) and `members_v3.csv` are not required. This makes
the first real-data experiment feasible without ingesting hundreds of millions
of listening events. These are KKBOX music subscriptions; no Schibsted customer
data or newspaper feature names are manufactured.

## JD stack

```sh
python -m pip install -r requirements-ui.txt -r requirements-kkbox.txt
python backend/train_kkbox.py --input-dir /absolute/path/to/kkbox --limit 50000
mlflow server --backend-store-uri sqlite:///data/mlflow.sqlite3 --host 127.0.0.1 --port 5001
python scripts/run_edition.py
```

- **Python:** bounded subscriber sampling, streaming CSV import, scikit-learn
  logistic regression and histogram gradient boosting, sigmoid calibration,
  evaluation and numeric JSON export. No GPU or LLM is needed for this model.
- **SQL:** SQLite stores imported billing records, computes point-in-time
  features and records dataset/model/prediction lineage. It uses the same
  `EDITION_DB` as the rest of Edition, with separate `kkbox_` tables.
- **MLflow:** enabled by default. Records source fingerprints, parameters,
  candidate-selection scores, test metrics, feature SQL and the exported model
  in the **Edition KKBOX churn** experiment. The SQLite model table controls
  local serving; this is not MLflow Model Registry deployment.
- **Streamlit:** import, progress, model comparisons, calibration, historical
  subscriber inspection and saved local predictions in the existing application.
- **Snowflake:** prepared sandbox tables/view, batched upload and a full feature
  parity verifier. No account is configured, so cloud execution remains unverified.

## Label and evaluation contract

The supplied `is_churn` is the outcome: failure to obtain a valid renewal within
30 days after membership expiry. A cancellation transaction alone is not churn.
Labels are not reconstructed from `is_cancel` and not used as input features.

`train_v2.csv` selects the March 2017 cohort with a **2017-03-01** feature cutoff;
fallback `train.csv` selects the earlier cohort with a **2017-02-01** cutoff.
Transactions on or after the cutoff are excluded. Dataset manifests report
excluded future rows, exact duplicates and labelled users without prior billing
history. Same-day ties are resolved deterministically using recorded fields;
the released daily timestamps cannot establish true intraday ordering.

The subscriber sample is chosen by a stable hash independently of the label.
Subscribers are then assigned to train/selection/calibration/test in
60%/15%/10%/15% hash buckets. This first experiment is **one historical cohort
with subscriber holdouts**, not a future-month validation. It avoids claiming
that all earlier-cohort outcomes would have been available at a later cutoff.

The two candidate classifiers are fitted only on training users. Selection uses
log loss on selection users; sigmoid calibration uses separate calibration users.
The test users are untouched until final evaluation and numeric-export parity
checking. The app reports ROC AUC, average precision, log loss, Brier score,
calibration bins and a constant training-prevalence baseline. No operating
threshold, pricing decision or causal retention benefit is inferred from these
results. This is not a Kaggle leaderboard submission or an exact reproduction
of the company's or another paper's model.

Raw records, models and test predictions stay local and are excluded from Git
and container images. The original benchmark and simulated offer decisions
are not overwritten. If source files are missing, the app explicitly says that
no real-data churn model has been trained.

## Optional Snowflake sandbox

Configure the existing private named connection `edition`, then run:

```sh
python -m pip install -r requirements-analytics.txt
python backend/kkbox_warehouse.py plan
python backend/kkbox_warehouse.py init --connection edition
python backend/kkbox_warehouse.py load --connection edition --dataset <dataset-id>
python backend/kkbox_warehouse.py verify --connection edition --dataset <dataset-id>
```

`plan` is offline. `load` sends the selected imported labels and pre-cutoff billing
records to the explicitly selected sandbox schema; it replaces only that dataset.
`verify` compares every local and warehouse feature, label and split. Account
configuration, actual Snowflake execution and SQL parity still need live validation.
SQLite is a local database, not a Snowflake emulator.

## Verification without restricted data

Tests construct small **synthetic integration fixtures in the official schema**.
They include future transactions that encode the outcome, so leakage checks must
exclude those rows. They also check idempotency, rollback, subscriber splits,
training/export parity, prediction persistence and the Streamlit workflow.
Fixture accuracy is never presented as real KKBOX benchmark accuracy.

```sh
python -m unittest discover -s tests -p 'test_kkbox.py' -v
python -m unittest discover -s tests -p 'test_streamlit_app.py' -v
```

Sources: [official competition](https://www.kaggle.com/c/kkbox-churn-prediction-challenge),
[WSDM challenge paper](https://sdlinpaper.csie.ntu.edu.tw/publications/Conference/Music%20Recommendation%20and%20Churn%20Prediction..pdf),
[Databricks point-in-time feature example](https://www.databricks.com/notebooks/churn/2-feature-engineering.html).
