# Dataset and behavioural model options

Initial research checked on 2026-10-19. Update: the KKBOX-derived candidate is now downloaded and integrated through our own trained scikit-learn model. See [the external model card](EXTERNAL_MODEL.md) for actual schema, quality checks, splits and measured results. EB-NeRD and Criteo have not been integrated. The sections below preserve the original candidate assessment.

## Recommendation

Use **EB-NeRD for real publisher reading behaviour**, and maintain a separately labelled **causal benchmark** for offer targeting. Use the **Misiorek/Deep.BI KKBOX-derived synthetic benchmark** if reproducing subscription-propensity experiments is the priority. None of these verified releases supplies a complete real news-publisher experiment with our four subscription prices and conversion outcomes.

External synthetic data makes a study reproducible, but simulated subscription labels cannot establish accuracy on real reader subscription decisions. A meaningful backend needs an explicitly supported prediction target, a later holdout, baselines, calibration and uncertainty.

## 1. EB-NeRD — strongest verified publisher data/code pair

- Publisher: Ekstra Bladet, Denmark.
- [Official dataset description](https://recsys.eb.dk/dataset/).
- [Dataset access and terms](https://recsys.eb.dk/).
- [Dataset paper](https://arxiv.org/abs/2410.03432).
- [Publisher benchmark code](https://github.com/jppol-ai/ebnerd-benchmark).
- [Publisher schema constants](https://github.com/jppol-ai/ebnerd-benchmark/blob/main/src/ebrec/utils/_constants.py).

Real anonymised news interaction data, with impressions, displayed/clicked article IDs, timestamp, fixed prior click histories, read time, scroll, device and subscription status. Demo/small/large releases support experiments without beginning with the full dataset. The described collection spans six weeks in 2023. Dataset access has its own research terms; the benchmark repository's MIT licence does not replace those terms.

Confirmed released algorithms include NRMS, LSTUR, NPA, NAML and NRMSDocVec. Repository availability is verified; current installation and training have not been tested here.

### Behavioural models

- [NRMS: Neural News Recommendation with Multi-Head Self-Attention (2019)](https://aclanthology.org/D19-1671/): models news titles and browsing histories using attention. Fits next-article recommendation.
- [LSTUR: Neural News Recommendation with Long- and Short-term User Representations (2019)](https://aclanthology.org/P19-1033/): combines persistent preferences with recent reading. Fits changing interests and habitual consumption. Publisher benchmark implements it.

### Credible backend tasks

1. Next-article ranking on the actually displayed candidate set; compare recent-popularity and simple content-affinity baselines against NRMS/LSTUR. Use impression-level ranking metrics, NDCG@10, MRR and user-cluster uncertainty.
2. Future observed engagement, e.g. return within the next 48 hours, if IDs remain linkable and observation windows are sufficient. Define labels strictly from future events. Exclude right-censored decisions; preserve a gap for label maturity before evaluation.
3. Subscription-status association as descriptive analysis. Only attempt acquisition prediction if a label audit verifies reliable observed non-subscriber-to-subscriber transitions and enough positive events. A status snapshot is not a confirmed purchase event. Lack of a future logged visit is not evidence of no subscription or cancellation.

### Invalid claims to avoid

The release does not by itself identify the causal effects of showing a 49/129/249 kr offer. Displayed article candidates are not randomised paywall offers. Subscriber status is not the same thing as acquisition, renewal or churn. Next read-time/next scroll are future outcomes, not available features at the original decision time. Article lifetime aggregates can leak future popularity; build as-of-time features.

## 2. Published synthetic subscription-propensity benchmark

- [Misiorek, Warmuz, Kaczmarek & Ciesielczyk (2022), Modeling User Engagement Profiles for Detection of Digital Subscription Propensity](https://doi.org/10.1007/978-3-030-95947-0_5).
- [Author-linked KKBox User Engagement Modeling Dataset](https://www.kaggle.com/datasets/pawelwmm/kkbox-user-engagement-modeling-dataset).

The publisher's paper page confirms use of a synthetic benchmark based on public KKBOX music-service data and separate live paywall tests at a digital media website. It explicitly links the Kaggle dataset in note 8. This is unusually close to our engagement-to-subscription question, but the public benchmark is music-derived and synthetic, not the publisher's real conversion log.

**Verified:** paper identity, abstract, author affiliation (including Deep.BI) and author-linked dataset location. **Not verified:** downloaded file schema, licence, present unauthenticated download access, exact label-generation mechanics, and a matching public training-code repository. The Kaggle page did not expose file metadata to the research tool. Do not claim the release has been reproduced.

**Proposed implementation after inspecting files:** compare a last-activity baseline, logistic regression and gradient-boosted trees using time-aware engagement aggregates. These are proposed baselines, not a claim about uninspected source code. Split users and time carefully; assess whether the synthetic labeling rule makes the prediction task circular.

## 3. Criteo Uplift — separate real intervention benchmark

- [Official data/paper description](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).
- [Official Criteo mirror](https://huggingface.co/datasets/criteo/criteo-uplift).
- [EconML implemented meta-learners](https://github.com/py-why/EconML/blob/main/econml/metalearners/_metalearners.py).

Real advertising incrementality experiment data with anonymised features, treatment assignment and response outcomes. Useful for benchmarking T-/X-learners and causal evaluation on outcomes our own generator did not define. Its domain is advertising, not publisher subscription offers.

Use assignment (`treatment`), not realised ad exposure, as the randomised intervention. Do not use post-treatment exposure as an ordinary pre-treatment feature. The original official description documents non-uniform privacy subsampling; inspect the selected version's correction notes before treating observed response rates, treatment probabilities or uplift as population quantities. The original page and newer mirror describe different dataset versions, so record exact file, version and checksum.

Do not rename anonymous features into reader attributes, interpret ad conversions as news subscriptions, combine individuals with EB-NeRD, or treat a binary ad experiment as empirical evidence for four subscription prices. The data are separate validation domains, not joinable tables. Its source lists CC BY-NC-SA 4.0 terms.

## Proposed upgrade for Edition

- Add a visible dataset/task selector: **real publisher engagement**, **published synthetic subscription propensity**, **advertising uplift benchmark**, and the existing **illustrative offer simulation**.
- Keep each model's features, supported actions, reports and outcomes separate. No cross-dataset user joins or invented field mappings.
- On EB-NeRD, replace fabricated engagement signals with historical features computed from real logs. Serve next-article ranking and an audited future-engagement score.
- On the published synthetic benchmark, reproduce propensity prediction only after inspecting data construction; display its synthetic origin beside each result.
- Validate causal targeting separately on an external intervention dataset. Report holdout uplift, uncertainty and policy value in its native outcome units; economics remain explicit assumptions.
- Keep our exact offer-price preview as scenario analysis until actual randomised publisher offer data exists.

This yields a real evidence-based backend for supported behaviours, while preserving the reader-facing demo without overstating its commercial validation.
