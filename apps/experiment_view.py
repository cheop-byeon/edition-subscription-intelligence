"""Reproducible paywall A/B analysis, separate from manually entered events."""
import csv
import io
import pandas as pd
import streamlit as st
from experiment_analysis import analyze, sample_size, simulate


def render():
    st.header('Experiment analysis · simulated paywall A/B test')
    st.warning('Synthetic experiment only. Outcomes are generated from explicit assumptions, not KKBOX records or real publisher traffic. Brand filters do not apply.')
    st.write('Hypothesis: changing the subscription offer changes the probability of subscribing within 30 days. Primary metric: subscriptions per randomized reader; null hypothesis: equal conversion rates.')
    st.caption('Design: blocked 1:1 randomization by unique fictional reader; standard offer versus alternative offer; one complete 30-day outcome per reader; intention-to-treat analysis at a fixed horizon. Article entitlement stays unchanged.')
    st.subheader('1 · Plan sample size')
    with st.form('experiment-plan'):
        baseline = st.number_input('Expected control conversion (%)', min_value=.1, max_value=95.0, value=10.0, step=.5)
        mde = st.number_input('Minimum detectable increase (percentage points)', min_value=.1, max_value=4.0, value=1.5, step=.1)
        power = st.selectbox('Target power', [.8, .9], format_func=lambda value: f'{value:.0%}')
        st.form_submit_button('Update sample-size plan')
    alpha = .05
    required = sample_size(baseline/100, mde/100, alpha, power)
    st.metric('Required readers per arm · approximate', f'{required:,}')
    st.caption(f'Total: {2*required:,} readers · two-sided α=0.05 · equal allocation · normal approximation. The MDE is an absolute increase, not relative lift. This is prospective planning, not observed post-hoc power.')
    st.subheader('2 · Generate and analyse a fixed-horizon experiment')
    with st.form('experiment-simulation'):
        size = st.number_input('Readers per arm', min_value=100, max_value=200000, value=10000, step=100)
        effect = st.number_input('Assumed treatment increase (percentage points)', min_value=-5.0, max_value=4.0, value=1.5, step=.1)
        seed = st.number_input('Simulation seed', min_value=0, max_value=1000000, value=42)
        run = st.form_submit_button('Run simulated experiment')
    if run:
        try:
            rows = simulate(int(size), baseline/100, effect/100, int(seed))
            st.session_state['ab-simulation'] = {'rows': rows, 'baseline': baseline, 'effect': effect, 'seed': int(seed)}
        except ValueError as exc:
            st.error(str(exc))
    experiment = st.session_state.get('ab-simulation')
    if not experiment:
        st.info('Run the simulation to see observed conversion rates, confidence intervals and the hypothesis test.')
        return
    rows = experiment['rows']
    counts = {arm: [r for r in rows if r['arm'] == arm] for arm in ['control', 'treatment']}
    nc, nt = len(counts['control']), len(counts['treatment'])
    xc, xt = (sum(r['converted_30d'] for r in counts[arm]) for arm in ['control', 'treatment'])
    result = analyze(xc, nc, xt, nt)
    st.caption(f"Saved simulation assumptions: control {experiment['baseline']:.1f}%, treatment effect {experiment['effect']:+.1f} pp, seed {experiment['seed']}. Editing planning inputs does not change this run until you run again.")
    st.dataframe(pd.DataFrame([{'Arm': arm, 'Readers': n, 'Subscriptions': x, 'Conversion': f'{x/n:.2%}', '95% Wilson interval': f'{ci[0]:.2%} to {ci[1]:.2%}'}
                               for arm, n, x, ci in [('Control', nc, xc, result['control_ci']), ('Treatment', nt, xt, result['treatment_ci'])]]), hide_index=True)
    low, high = result['difference_ci']
    st.metric('Treatment − control', f"{100*result['difference']:+.2f} percentage points")
    st.write(f'95% Newcombe confidence interval: **{100*low:+.2f} to {100*high:+.2f} percentage points**.')
    if result['test_valid']:
        st.write(f"Pearson χ²(1) = **{result['chi_square']:.3f}** · two-sided p-value = **{result['pvalue']:.4g}** · α = 0.05. No continuity correction.")
        st.info('Reject equal conversion rates in this synthetic experiment.' if result['significant'] else 'Insufficient evidence to reject equal conversion rates in this synthetic experiment. This does not establish equivalence or no effect.')
    else:
        st.warning('Expected cell count below 5: the asymptotic chi-square test is suppressed. Use an exact method for sparse data; confidence intervals remain approximate.')
    st.subheader('What this result means')
    st.write('This demonstrates randomized-experiment analysis under a known synthetic data-generating process. It estimates the simulated intention-to-treat difference in 30-day conversion. The confidence interval describes repeated-sampling uncertainty under the design; it is not a probability that the fixed true effect lies inside this particular interval.')
    st.write('It does not demonstrate real paywall uplift, revenue improvement, retention benefit or KKBOX treatment effects. The treatment effect was supplied to the simulator. A real study needs consent/eligibility rules, exposure and outcome instrumentation, mature follow-up, checks for assignment imbalance and missingness, and guardrails for revenue per reader, cancellations and engagement.')
    st.caption('One primary comparison, fixed sample size and no early stopping are assumed. Repeatedly trying seeds or thresholds and selecting a significant result invalidates the nominal error rate. The confidence interval and test use different approximations and may disagree near the decision boundary.')
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    st.download_button('Download synthetic experiment CSV', output.getvalue(), 'edition-simulated-ab.csv', 'text/csv')
    with st.expander('Statistical method references'):
        st.markdown('[NIST: test of two proportions](https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/diffprci.htm) · [Newcombe interval reference in statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.stats.proportion.confint_proportions_2indep.html)')
