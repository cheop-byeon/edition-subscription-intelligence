# Research assessment

Verified on 2026-10-19. The [target JD](https://schibsted.teamtailor.com/jobs/8549625-data-scientist-subscription-data-enablement) places dynamic paywalls and subscription offers at the centre of the work, with experimentation, causal methods and production delivery. That makes **incremental offer value** a closer project target than an article-ranking-only recommender.

## Company-linked research: strongest verified paper/code pair

**Eide, Leslie & Frigessi (2022), Dynamic slate recommendation with gated recurrent units and Thompson sampling.**

- [Published paper](https://doi.org/10.1007/s10618-022-00849-w)
- [FINN-owned repository](https://github.com/finn-no/recsys_slates_dataset)
- [Compatible historical release recommended by repository authors](https://github.com/finn-no/recsys_slates_dataset/tree/v1.0)
- [Companion FINN.no Slates Dataset paper](https://arxiv.org/abs/2111.03340)

This is FINN research from the Schibsted marketplace era, relevant to the marketplace business now associated with Vend. Its Bayesian recurrent recommender and Thompson sampling address sequential slate selection. The repository includes original research code in `code_eide_et_al21/` and exposure/click/no-click data.

**Important reproducibility finding:** the repository itself warns that its newer dataset is incompatible with the original research code and directs users to v1.0 for a compatible code/data pair. Presence of released code is verified; execution is not. The original FINN dataset models marketplace interactions, not subscription conversion or randomised paywall offer effects. Exposure/click logs must not be treated as causal subscription-offer data. Logged exposures alone do not establish the action-propensity support needed for off-policy evaluation.

**Use in this app:** a research-library reference and future engagement component. We do not claim a GRU/Thompson-sampling reproduction or train on FINN data. For a follow-on study, reproduce the pinned research release in an isolated environment and assess its maintenance and data access separately.

I did not verify a public Schibsted subscription-paywall paper with a matching released implementation in this search. That is a search limitation, not evidence of absence.

## Core model: meta-learners for heterogeneous effects

**Künzel, Sekhon, Bickel & Yu (2019), Meta-learners for estimating heterogeneous treatment effects using machine learning (PNAS; preprint 2017).**

- [Paper](https://arxiv.org/abs/1706.03461)
- [Authors' R implementation: causalToolbox](https://github.com/forestry-labs/causalToolbox)
- [Python implementation: EconML TLearner source](https://github.com/py-why/EconML/blob/main/econml/metalearners/_metalearners.py)

The framework separates outcome prediction from effect estimation. A T-learner fits response models per treatment and differences predictions. For subscription acquisition, this asks whether an offer changes conversion for similar readers, rather than targeting everyone likely to subscribe anyway.

**Implemented here:** an independent multi-arm T-learner with regularised logistic response models. Hyperparameters are chosen on validation log loss. The paper's X-learner is not implemented. The NumPy code is intentionally small and inspectable; the linked mature libraries are alternatives, not dependencies silently used here.

**Limits:** effects are identified by the synthetic randomisation, overlap and stable outcome assumptions. Per-arm models can be data hungry; effect estimates can be noisy even when factual prediction is good. Predictive AUC is not a causal quality metric. No individual causal truth is observable in production; reader-level predictions represent conditional average effects.

## Core evaluation: doubly robust policy value

**Dudík, Langford & Li (2011), Doubly robust policy evaluation and learning.**

- [Paper](https://arxiv.org/abs/1103.4601)
- [Third-party Python implementation: contextualbandits](https://github.com/david-cortes/contextualbandits)

DR combines a reward prediction with an inverse-propensity correction for observed outcomes. This supports evaluating a frozen targeting policy on a separate logged experiment without pretending to observe every reader's counterfactual outcome.

**Implemented here:** independent DR and IPS estimators for deterministic guarded policies. Propensities come from the actual simulator assignment mechanism, not fitted guesses. Predictions and target policies are learned without test outcomes. Normal intervals use independent reader scores; policy differences use paired scores. A small balanced fixture verifies both estimators against a known expected policy value, including a deliberately wrong reward model with correct propensities.

**Limits:** DR is not protection against arbitrary confounding, missing action support, interference, or the wrong reward definition. Repeated reader observations would require appropriate dependence handling. Current intervals condition on the learned policy and omit retraining uncertainty.

## Official runnable research framework for a future benchmark

**Saito, Aihara, Matsutani & Narita (2021), Open Bandit Dataset and Pipeline: Towards realistic and reproducible off-policy evaluation (NeurIPS Datasets and Benchmarks; preprint 2020).**

- [Paper](https://arxiv.org/abs/2008.07146)
- [Official Open Bandit Pipeline repository](https://github.com/st-tech/zr-obp)

This codebase implements policy learning and off-policy estimators including IPS and DR, alongside real logged fashion-commerce bandit data. It offers a stronger next benchmark than a simulator alone.

**Use here:** verified code reference and experimental-structure guidance. The repository and data are not installed or executed. It is not Schibsted/Vend research and its click outcomes are not news subscription outcomes. Before reuse, inspect Python/package constraints and pin a compatible environment; repository availability does not certify current installability.

## What to say in an interview

“I built a full-stack offer decision lab aligned with subscription acquisition. I implemented a transparent multi-arm T-learner and evaluated the frozen policy with DR and IPS on a later synthetic randomised holdout. The key distinction is conversion propensity versus incremental contribution. I found FINN's released Bayesian slate recommender, but kept it as a clearly labelled adjacent reference because its marketplace data cannot validate paywall offer effects. The next step is a governed real experiment, not extrapolating the simulator's gain.”

Only describe this work as your own contribution after you have reviewed, understood, and extended the implementation yourself.
