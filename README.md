# Edition — Subscription Intelligence

A subscription modelling application with a trained gradient-boosting model, live prediction API, evaluation dashboard, and a separate offer-policy simulator.

Built around the [Schibsted Subscription Data Enablement role](https://schibsted.teamtailor.com/jobs/8549625-data-scientist-subscription-data-enablement).

## What runs

- **Subscription model:** provenance, baseline comparison, calibration, chronological holdout results, and live server-side scoring.
- **Offer simulator:** illustrative subscription offers, a multi-arm T-learner, guard rules, DR/IPS estimates and experiment views.
- **Research library:** paper/code links with explicit implementation status.

The external model was trained here on the published [KKBOX User Engagement Modeling Dataset](https://www.kaggle.com/datasets/pawelwmm/kkbox-user-engagement-modeling-dataset), associated with [Misiorek et al.](https://doi.org/10.1007/978-3-030-95947-0_5). Its subscription labels are **synthetic**. It does not establish real publisher conversion, offer uplift or revenue impact. This project is independent of Schibsted and Vend.

## Run locally

Python 3.12+ is sufficient to serve the included numeric model artifacts. No ML package installation is needed for inference.

```bash
python3 backend/server.py --port 4173
```

Open <http://127.0.0.1:4173>. Binding defaults to loopback. The public source includes 20 artificial input examples without observed labels; they can be scored by the trained model. The example inputs are not downloaded dataset rows.

To run the container:

```bash
docker compose up --build
```

It exposes the same UI on localhost port 4173. The container runs as a non-root user, without training libraries, with a read-only filesystem and dropped capabilities. Its development HTTP server is intended for this portfolio demo; a production deployment needs a suitable ingress/TLS/runtime configuration.

## Model results

The source dataset contains 1,000,316 records. After removing 901 exact duplicates, training, selection, calibration and testing use separate chronological windows. The final holdout has 119,731 records.

| Model | Test ROC AUC | Test average precision | Brier score |
| --- | ---: | ---: | ---: |
| Constant training prevalence | 0.500 | 0.014 | 0.0142 |
| Raw-feature logistic regression | 0.807 | 0.071 | 0.0153 |
| Enriched-feature logistic regression | 0.933 | 0.245 | 0.0142 |
| Calibrated histogram gradient boosting | 0.946 | 0.344 | 0.0112 |

Test positive-label prevalence is 1.418%. The top-scoring decile has 11.852% positives. These are synthetic-benchmark results. The release lacks user IDs, so cross-split repeated-user leakage cannot be ruled out. The original label-generation mechanism and observation horizon have not been independently reconstructed. Calibration is imperfect in high-score bins.

See [the external model card](research/EXTERNAL_MODEL.md), [dataset assessment](research/DATASET_OPTIONS.md), and [research assessment](research/RESEARCH.md).

## CI/CD

[GitHub Actions](https://github.com/cheop-byeon/edition-subscription-intelligence/actions) runs on pull requests, pushes to `main`, version tags (`v*`), and manual dispatch.

1. Run unit, inference-parity and real HTTP integration tests on Python 3.12 and 3.14.
2. Check frontend JavaScript, build the Worker, validate its API/inference, and upload the portable bundle as a 14-day Actions artifact.
3. Build the Docker image, start it with a restricted runtime, and smoke-test the frontend and live model API.
4. After all checks pass on a `main` or version-tag push, publish the image to `ghcr.io/cheop-byeon/edition-subscription-intelligence`.

Images are tagged `latest` for `main`, `sha-<commit>` for traceability, and the version tag for a `v*` push. Pull requests and manual CI runs do not publish images. Authentication uses GitHub's built-in `GITHUB_TOKEN`; no registry password is added to this repository. Registry access/visibility remains controlled by GitHub; the workflow does not make packages public.

Actions are pinned to verified commit hashes. Dependabot proposes weekly updates for Actions, the Docker base image, and Python requirements. Configure `Python 3.12 tests`, `Python 3.14 tests`, `Frontend and Worker build`, and `Container build and smoke test` as required branch checks if desired; branch protection is not changed by this repository.

This is **continuous delivery of a tested container**, not automatic deployment to a server. There is no configured production host, Kubernetes cluster, cloud account, or Sites publication step. Downloaded benchmark data and the local private hosting identity are excluded from Git and container builds.

## Reproduce checks

Node.js 22+ and Python 3.12+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
node scripts/check-frontend.mjs
bash scripts/build.sh
node tests/test_worker.mjs
node tests/test_external_worker.mjs
node scripts/validate-artifact.mjs
```

There are no npm dependencies. The Worker build uses the tracked `.openai/hosting.example.json` when a local `.openai/hosting.json` does not exist. Generated `dist/` and `worker/index.js` are ignored and regenerated in CI.

## Optional retraining

The normal CI workflow tests frozen artifacts; it does not download the source dataset or retrain models on each push.

To reproduce the synthetic offer simulation:

```bash
python backend/train.py
```

For the external benchmark, obtain the author archive separately, review its terms, then use an isolated environment:

```bash
python -m pip install -r requirements-external.txt
python backend/train_external.py --archive /absolute/path/to/kkbox-engagement.zip
```

That writes the fitted numeric model, evaluation report and public artificial input examples to `artifacts/external_model.json`. The 80 downloaded held-out example records go to ignored `data/local_benchmark_profiles.json`; the local Python server uses them when present. The published artifact and container use artificial examples. The full source archive/CSV is not included. Training exports are checked against scikit-learn on 1,000 records before saving. Python and JavaScript serving implementations are compared by tests.

Restart the local backend after retraining and rebuild any Worker bundle. Retraining dependencies are pinned to the verified environment; they are not needed by the runtime container.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/api/health` | Loaded service/model health |
| GET | `/api/benchmark` | External evaluation, provenance and example inputs |
| POST | `/api/benchmark/score` | Live trained-model prediction from all 20 required feature values |
| GET | `/api/model` | Offer simulation model metadata |
| GET | `/api/report?brand=all` | Frozen simulation report by brand |
| POST | `/api/decision` | Illustrative offer decision with rule overrides |
| GET | `/api/research` | Research references and implementation status |

The service does not persist submitted profiles or learn online. Existing subscribers, personalisation-off readers, public-interest stories and repeated offer dismissals bypass acquisition scoring in the offer simulator. No checkout, actual paywall enforcement, Snowflake connection, MLflow server or real reader collection is implemented.

## Repository layout

```text
.github/       CI, container delivery and dependency updates
backend/       Python training, numeric inference, API and JS inference parity
frontend/      Responsive UI and API client
artifacts/     Frozen model parameters and measured evaluation reports
research/      Paper links, provenance and model limitations
scripts/       Portable build, validation and container smoke checks
tests/         Unit, API, inference and serving tests
Dockerfile     Dependency-free Python inference runtime
compose.yaml   Local container setup
```
