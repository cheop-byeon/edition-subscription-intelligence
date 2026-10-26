# Alignment with the Subscription Data Enablement JD

Source checked 21 October 2026:
https://schibsted.teamtailor.com/jobs/8549625-data-scientist-subscription-data-enablement

The role concerns personalised subscription offers and their timing, commercial
experimentation and models taken through the full development lifecycle. It names
Snowflake, MLflow and Streamlit, plus Python or R and SQL. It covers propensity,
recommendation and offer optimisation; it does not explicitly require an LLM
chatbot. The existing chatbot remains an optional learning extension.

## Our interpretation and implemented boundaries

- Article access is determined by subscription entitlement, article classification
  and editorial exemptions. Our policy prototype uses a paid/free article flag.
- The offer engine separately decides whether to show a subscription offer now,
  and selects an eligible standard or introductory offer. Deferred offers do not
  grant access to paid articles.
- Consent disables propensity targeting, not the contextual standard offer or
  article gate. Marketing suppression stops commercial offers without unlocking
  the article.
- The current paywall-v2 experimental control receives standard offers, while the
  policy arm varies timing and offer. Older paywall-v1 results are not pooled with
  this changed design.
- These rules and brand prices are ours for learning, not a reconstruction of the
  company's undisclosed system. Optimising revenue needs valid action/outcome
  data and causal evaluation; subscription propensity alone does not supply it.

## Technology status

| Technology | Implementation | Verification |
| --- | --- | --- |
| Python + SQL | Training, APIs, SQLite registry/audit; Snowflake SQL features | Local unit/integration tests |
| MLflow | Local tracked temporal training, metrics and artifacts | Recorded run; activation/rollback in serving registry verified separately |
| Streamlit | Live analyst workbench using the same serving logic and registry | Unified app pages checked with AppTest and local browser |
| Snowflake | Named-connection adapter; sandbox tables; point-in-time feature view; batched synthetic loader; export into temporal model training | Adapter unit tests only; not connected or cloud-executed because no account is configured |

Snowflake is a remote warehouse service; SQLite is a local development source,
not an emulation of Snowflake. The SQL must still be run and compared against
Python feature fixtures in an actual sandbox account before claiming integration
validation. No credentials are stored in the repository. No warehouse, account
or cloud resource is provisioned.

## One application

All active interfaces are consolidated into apps/edition.py. The old HTML and
Worker interfaces are archived and not served. Legacy URLs redirect to the one
Streamlit workspace. Model/data references are pages within that application.

## Original KKBOX churn extension

The **KKBOX churn** page adds an original-data import and training workflow using
Python, SQL, MLflow and Streamlit. It keeps official churn labels and music-billing
features, separate from simulated news-brand scores and offer decisions. Real-data
training waits for user-obtained Kaggle files; test fixtures are labelled synthetic.
Snowflake tables, an equivalent feature view, selected-dataset upload and a full
local/warehouse parity verifier are prepared, but not cloud-executed. See
KKBOX_CHURN.md for the experiment contract and limitations.
