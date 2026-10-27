# Edition · Subscription Intelligence

A subscription analytics project covering an end-to-end machine-learning workflow: **raw subscription records → SQL features → churn models → evaluation → retention audience analysis**.

The project combines Python, SQL and MLflow for reproducible subscription-churn modelling, with a Streamlit interface for exploring model evidence and retention audiences.

## What the project demonstrates

The main workflow uses the original **KKBOX subscription-churn competition data**. It predicts non-renewal from billing history, compares two machine-learning models, tracks experiments locally and uses the selected model to construct historical research audiences.

The project also includes simulated sales/churn propensity models, rule-based paywall decisions and randomized offer-experiment analysis. A local LLM assistant explains aggregate simulated commercial data. These exercises remain separate from the real KKBOX churn results.

| Component | Implementation | Status |
| --- | --- | --- |
| Data preparation | Streaming import, source fingerprints and bounded subscriber sampling | Working locally |
| Feature engineering | Point-in-time billing features in SQL; SQLite storage | Working locally |
| Churn modelling | Logistic regression and histogram gradient boosting; separate calibration | Trained on original KKBOX records locally |
| Evaluation | Subscriber holdouts, constant baseline, ROC AUC, average precision, log loss, Brier score and calibration | Working locally |
| Retention analysis | Score held-out subscribers, filter by risk and save audience snapshots with model/dataset lineage | Working locally; research drafts only |
| A/B experimentation | Sample-size planning, fixed-horizon simulation, chi-square test and confidence intervals | Working locally on synthetic outcomes; no measured commercial uplift |
| MLOps | MLflow experiment tracking, numeric JSON model export, SQLite serving registry and prediction audit | Working locally |
| Analysis interface | One Streamlit workspace | Local use; Deploy control hidden |
| Snowflake | Sandbox SQL, loader and local/warehouse feature parity verifier | Prepared; no account configured or live parity verified |
| LLM assistant | Local Ollama chat over aggregate simulated commercial data | Requires a separately installed local model |

## Run locally

Use Python **3.12 or 3.14**, the versions exercised by CI. Run these commands from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-ui.txt -r requirements-kkbox.txt
.venv/bin/python scripts/run_edition.py
```

Open [Edition on localhost](http://127.0.0.1:8501/). Stop the launcher with **Ctrl-C**. On macOS, `start.command` also starts the project once dependencies are installed; `EDITION_PYTHON` can select an existing Python environment.

The launcher starts Streamlit on port **8501** and the local API/compatibility service on **4173**. Both bind to `127.0.0.1` by default. Older interface URLs on port 4173 redirect to this workspace. The launcher does not stop other processes already using these ports.

A clean checkout includes simulated examples and reference artifacts, but **does not include the original KKBOX files, the locally trained KKBOX model or the local database**. The real-data workflow becomes available after importing and training.

## Obtain data and train

1. Visit the [official KKBOX competition data page](https://www.kaggle.com/c/kkbox-churn-prediction-challenge/data). Sign in and review/accept the rules yourself.
2. Download **`train_v2.csv`** and **`transactions.csv`**, then extract them into one local folder, for example `data/kkbox/raw/`. Optional `transactions_v2.csv` is also supported. Listening logs and member demographics are not required for this billing-history baseline.
3. Open **Models & data** in Edition. Enter the folder path, confirm the file source and select **Import and train locally**. The default sample limit is 50,000 subscribers. Refresh the run status to inspect progress.
4. Review model comparison and calibration, then open **Subscribers & retention** to explore the model's scores and save research audience drafts.

CSV, CSV.GZ and single-CSV ZIP files are supported. Extract `.7z` and multi-file archives before importing. Kaggle credentials are not used by the application.

The same pipeline can run from the terminal:

```sh
.venv/bin/python backend/train_kkbox.py --input-dir /absolute/path/to/kkbox --limit 50000
```

Training logs to local MLflow by default. To inspect the experiment UI, run this in another terminal:

```sh
.venv/bin/mlflow server --backend-store-uri sqlite:///data/mlflow.sqlite3 --host 127.0.0.1 --port 5001
```

Open [local MLflow](http://127.0.0.1:5001/). Its web server is not required for training to write to the local tracking database. The churn model runs on CPU; it needs neither a GPU nor an LLM. Local inference uses exported numeric JSON rather than loading a pickle file.

## Model and evaluation contract

The target is the competition's supplied **`is_churn`** label: failure to obtain a valid renewal within 30 days after membership expiry. A cancellation transaction alone is not the target.

For `train_v2.csv`, the pipeline uses a **2017-03-01 feature cutoff**. Only earlier transactions contribute features; labels are never inputs. The fallback `train.csv` uses the earlier cohort and a 2017-02-01 cutoff. Subscriber selection uses a stable hash independently of the label.

Subscribers are assigned to disjoint **training / selection / calibration / test** partitions in 60% / 15% / 10% / 15% hash buckets. Logistic regression and histogram gradient boosting train on the training partition. Selection uses selection-partition log loss, and sigmoid calibration uses a separate partition. Final metrics use held-out test subscribers.

This is a **single historical cohort with subscriber holdouts**, not a future-month evaluation, a Kaggle leaderboard submission or a reproduction of Schibsted's proprietary models. See [the KKBOX workflow](research/KKBOX_CHURN.md) for feature rules, timestamp limitations and verification details.

### Recorded local experiment

The completed local run used a 50,000-subscriber sample from the official files. It scanned 21,547,746 transaction rows and produced usable features for **49,850 subscribers**; 150 had no pre-cutoff billing history. The test partition contained **7,508 subscribers**, including 684 churn cases.

| Test metric | Selected gradient boosting model, calibrated | Constant training-prevalence baseline |
| --- | ---: | ---: |
| ROC AUC | 0.9000 | 0.5000 |
| Average precision | 0.6766 | 0.0911 |
| Log loss ↓ | 0.1719 | 0.3052 |
| Brier score ↓ | 0.04594 | 0.08281 |

These are results from this local experiment, not full-dataset or publisher performance claims. Saved-model predictions were independently checked against the recorded holdout metrics. Source records and generated artifacts remain outside Git; a fresh checkout must reproduce the run with its own permitted copy of the data.

## One analysis workspace

| Page | Purpose and data |
| --- | --- |
| Overview | Portfolio purpose, real churn-model summary, workflow and technology status |
| Subscribers & retention | Real KKBOX model scores on held-out subscribers; risk thresholds and locally saved research drafts |
| Models & data | KKBOX import, training, evaluation, calibration, individual inference and lineage; additional simulated registry exercises and warehouse setup |
| Readers & campaigns | Simulated prospects/subscribers, sales/churn scores, consent/suppression rules and draft audience exports |
| Paywall | Simulated article-access and subscription-offer decisions |
| Experiments | Prospective sample-size planning, reproducible simulated reader A/B test, chi-square test and confidence intervals; manual events kept descriptive |
| Assistant | Local LLM explanations of aggregate simulated commercial data |
| Research | Papers, code references and implementation evidence |

The brand selector applies to the simulated commercial exercises. KKBOX records have no publisher-brand mapping. Retention drafts store the model ID, dataset ID, threshold and scored audience snapshot; they do not send messages or establish permission to contact subscribers.

### What the data can and cannot support

- **Original KKBOX records:** support music-subscription non-renewal modelling. High risk does not establish who would benefit from a retention offer.
- **Simulated commercial records:** demonstrate sales/churn workflows across illustrative brands. Their metrics do not establish real commercial effectiveness.
- **Older KKBOX-derived reference benchmark:** uses synthetic subscription labels. It remains under Models & data for reference and does not drive the real KKBOX retention workflow.

KKBOX lacks newspaper article visits, paywall exposures, offer assignments and purchase responses. The paywall is therefore an explicit policy simulation, not a learned dynamic paywall model. Its rules separate **article entitlement** from **offer selection and timing**: deferring an offer never unlocks paid content. Prices and outcomes are illustrative.

Future-month generalisation, causal offer effects and live retention improvement remain unvalidated. These require additional data and evaluation, not reinterpretation of the existing churn scores.

## Simulated experiment analysis

The Experiments page adds a fixed-horizon paywall-offer study with blocked 1:1
randomization and one complete 30-day conversion outcome per fictional reader.
An approximate two-sided sample-size calculation uses the expected baseline,
absolute minimum detectable effect, alpha 0.05 and 80% or 90% target power.
The generated study reports Pearson chi-square (one degree of freedom, no
continuity correction), Wilson conversion-rate intervals and a Newcombe interval
for the absolute treatment-minus-control difference. Sparse expected counts
suppress the asymptotic test. Synthetic reader-level CSVs can be downloaded.

To explore the example, open **Experiments**, review the baseline conversion and
minimum detectable increase, select **Update sample-size plan**, then select
**Run simulated experiment**. Planning inputs and the saved simulation assumptions
are displayed separately. With the default 10% baseline, 1.5 percentage-point MDE,
80% power and alpha 0.05, the approximate requirement is **6,693 readers per arm**.
The default simulation uses 10,000 readers per arm and seed 42; its observed
difference is **+1.19 percentage points**, with a 95% interval of **+0.32 to +2.06**
and p = **0.0071**. These numbers describe a generated example whose effect was
specified in advance, not publisher results.

The seed, sample size and assumed effect make the study reproducible. The effect
is supplied to the simulator: results demonstrate statistical analysis, not real
paywall or retention uplift. No post-hoc power is reported. Manual paywall events
remain descriptive and are not pooled with this experiment. Real deployment would
require mature outcomes, validated assignment/exposure tracking and business
guardrails. Repeated seed selection or early stopping would invalidate the stated
fixed-horizon inference. Statistical methods follow the [NIST two-proportion test](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/diffprci.htm)
and [Newcombe interval reference](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html).

## Optional integrations

### Snowflake

Local SQLite remains the working database. To install the optional connector and inspect the offline plan:

```sh
.venv/bin/python -m pip install -r requirements-analytics.txt
.venv/bin/python backend/kkbox_warehouse.py plan
```

[The connection example](config/snowflake_connections.example.toml) describes a private named connection in `~/.snowflake/connections.toml`. After configuring your own sandbox account, explicit `init`, `load` and `verify` commands are documented in [the KKBOX guide](research/KKBOX_CHURN.md#optional-snowflake-sandbox). Loading sends the selected data to that account; queries may consume credits. Nothing connects or uploads automatically. The separate `backend/warehouse.py` adapter supports the simulated lifecycle exercise.

### Local LLM assistant

Install Ollama separately and make `llama3.2:latest` available, or set `EDITION_LLM_MODEL` to another installed model. The assistant uses aggregate **simulated** commercial context, has no campaign or model-changing tools, and reports unavailability if the local LLM cannot be reached. It does not currently analyse KKBOX retention drafts. `backend/evaluate_chat.py` provides limited live checks when the API and Ollama are running.

## Repository layout

```text
apps/                 Streamlit portfolio views
backend/              Import, SQL features, training, inference, retention and API
sql/                  Local and prepared Snowflake feature/schema SQL
artifacts/            Committed simulated/reference model artifacts
config/               Illustrative paywall policy and connection examples
research/             Dataset, model and job-role implementation notes
scripts/              Local launcher and container smoke checks
tests/               Unit, pipeline, inference and Streamlit integration tests
data/                Local databases, downloads and generated runs (ignored paths)
archive/              Historical interfaces; not served as alternate applications
```

The shared database defaults to `data/edition.sqlite3`; override it with `EDITION_DB`. Original KKBOX files, generated models and job logs live under ignored `data/kkbox/`. MLflow uses `data/mlflow.sqlite3` and local artifact storage. Keep source records, credentials and local outputs out of Git.

## Verification and CI

With the main dependencies installed:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Tests generate clearly marked synthetic fixtures to exercise the official KKBOX schema without redistributing restricted data. They cover pre-cutoff feature construction, import rollback/idempotency, subscriber splits, portable inference parity, prediction persistence, the integrated retention draft workflow, known-value statistical calculations and experiment-page rendering. Fixture accuracy is not real-data benchmark evidence. Optional warehouse tests require `requirements-analytics.txt`; live Snowflake parity requires an account and is separate from mocked checks.

GitHub Actions runs Python 3.12/3.14 checks, analytics/Streamlit tests and container smoke checks. The existing delivery workflow also publishes validated container images to GHCR on main/version-tag pushes; it does **not** deploy a website. Local portfolio use needs no container or deployment. Optional container instructions are in `compose.yaml`; its runtime does not bundle original KKBOX data, training dependencies, MLflow or Ollama services.

## Further reading

- [KKBOX data, evaluation and Snowflake workflow](research/KKBOX_CHURN.md)
- [Mapping the project to the subscription data science role](research/JD_ALIGNMENT.md)
- [Research references](research/RESEARCH.md)
- [Dataset options and constraints](research/DATASET_OPTIONS.md)
- [Older synthetic-label benchmark](research/EXTERNAL_MODEL.md)
