# Integrated external subscription model

Edition's default **Subscription model** view now uses a downloaded research benchmark and a trained, evaluated model. The original offer pages remain independent simulations.

## Data

- Source: [KKBox User Engagement Modeling Dataset](https://www.kaggle.com/datasets/pawelwmm/kkbox-user-engagement-modeling-dataset), pawelwmm, version 1.
- Associated paper: [Misiorek et al., Modeling User Engagement Profiles for Detection of Digital Subscription Propensity](https://doi.org/10.1007/978-3-030-95947-0_5).
- Downloaded archive: 68,600,737 bytes.
- SHA-256: `001a15a04d976180b61cdfd54d6155ff3196a6084369990d5f63110517789111`.
- Original CSV: 1,000,316 rows, 65 columns, 23,519 positive synthetic subscription labels.
- Removed 901 exact duplicate rows; retained 999,415 rows.
- 943 negative cells across the selected features are treated as missing.
- The archive is stored in the workspace's `work/kkbox-engagement.zip`; the full raw data are not added to the application source tree.

This is a published synthetic subscription benchmark based on real music listening records. It is not Schibsted data, newspaper purchase data, or a randomised subscription-offer experiment. The source metadata reports licence Unknown. No hosted upload or raw data redistribution has been performed. The local ignored data file includes 80 held-out example records for exploration. The Git-tracked artifact contains 20 artificial input examples with no outcome labels.

## Actual implementation

The model is trained here using scikit-learn's histogram gradient boosting. It is a transparent applied baseline associated with the paper's question, not a downloaded pretrained checkpoint or an exact reproduction of the original paper implementation.

Twenty numeric inputs include daily song counts, listening duration and engagement aggregates. Registration dates, registration method, the date column and the target are excluded. Negative measurements become missing, then training medians impute missing values. Training-set 99.5th percentile caps limit outliers. A log1p transform follows. All learned preprocessing uses training data only.

We fit raw-feature logistic regression and enriched-feature logistic regression baselines. Two gradient-boosting candidates (7 versus 15 leaves, 90 iterations) are compared on a separate selection period using log loss. A sigmoid calibrator is fitted on the following period. The test set is not used for model, hyperparameter or calibration fitting.

| Partition | Rows | Dates |
| --- | ---: | --- |
| Training | 650,748 | 2015-01-01–2016-07-04 |
| Model selection | 129,284 | 2016-07-05–2016-10-16 |
| Probability calibration | 99,652 | 2016-10-17–2016-12-28 |
| Final test | 119,731 | 2016-12-29–2017-03-31 |

## Measured test results

| Model | ROC AUC | Average precision | Brier score |
| --- | ---: | ---: | ---: |
| Constant training prevalence | 0.500 | 0.014 | 0.0142 |
| Raw-feature logistic regression | 0.807 | 0.071 | 0.0153 |
| Enriched-feature logistic regression | 0.933 | 0.245 | 0.0142 |
| Calibrated histogram gradient boosting | 0.946 | 0.344 | 0.0112 |

Test positive rate is 1.418%. The top-scoring decile's observed positive rate is 11.852%, approximately 8.357 times the full test rate. The model AUC day-block bootstrap interval is approximately [0.941, 0.951] using 100 resamples. This is a conditional benchmark interval, not a user-cluster or retraining interval. Probability calibration is imperfect, especially in high-score bins; the UI displays that curve.

The release does not supply user identifiers, so repeated-user leakage cannot be excluded or audited across date splits. The exact original synthetic label-generation mechanism and its observation horizon have not been independently reconstructed. These limitations restrict interpretation of the measured accuracy. No subscription price or intervention effect is learned from this dataset.

## Live integration

- `GET /api/benchmark`: provenance, full evaluation, baselines, splits and 80 example records. The sample intentionally contains 60 negative and 20 positive records; it does not represent the test prevalence.
- `POST /api/benchmark/score`: accepts `{"values": { ... all 20 required input fields ... }}` and computes a prediction from the trained trees on the server.
- The frontend does not compute the model score. It displays API predictions and a held-out label only when the local downloaded examples are available. Public examples have no observed label.
- The report download excludes all example records. Raw downloaded rows are excluded from Git and Docker builds.

The fitted model and evaluation metrics are identical for local and repository versions. The frozen model is exported to safe numeric JSON; serving needs Python standard-library code, not pickle or scikit-learn. During training, export predictions are checked against scikit-learn on 1,000 test records. Both Python and JavaScript inference agree with all 80 stored training-library example predictions to 1e-10.

## Reproduce

The included artifact lets you serve immediately:

```bash
python3 backend/server.py --port 4173
```

To retrain with a separately downloaded author archive:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-external.txt
python backend/train_external.py --archive /absolute/path/to/kkbox-engagement.zip
python -m unittest discover -s tests -p 'test_*.py'
bash scripts/build.sh
node tests/test_external_worker.mjs
```

Restart the local server after replacing the model artifact. Optional training dependencies are pinned to the verified environment; serving the checked-in artifact needs no ML libraries. The Worker build has matching inference and API checks but has not been deployed.
