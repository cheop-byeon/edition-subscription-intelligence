"""Fixed-horizon binary-outcome experiment planning and analysis, CPU/stdlib only."""
import math
import random
from statistics import NormalDist

NORMAL = NormalDist()


def sample_size(baseline, effect, alpha=.05, power=.8):
    """Equal-arm normal approximation, two-sided alpha; absolute risk difference."""
    if not (0 < baseline < 1 and 0 < baseline + effect < 1 and effect != 0
            and 0 < alpha < .5 and .5 < power < 1):
        raise ValueError('Invalid planning probabilities or effect.')
    treatment = baseline + effect
    pooled = (baseline + treatment) / 2
    numerator = (NORMAL.inv_cdf(1-alpha/2) * math.sqrt(2*pooled*(1-pooled))
                 + NORMAL.inv_cdf(power) * math.sqrt(baseline*(1-baseline)+treatment*(1-treatment)))
    return math.ceil((numerator / abs(effect))**2)


def wilson(successes, n, alpha=.05):
    z = NORMAL.inv_cdf(1-alpha/2)
    p = successes/n
    denominator = 1+z*z/n
    centre = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return max(0, centre-half), min(1, centre+half)


def analyze(control_successes, control_n, treatment_successes, treatment_n, alpha=.05):
    for count, n in [(control_successes, control_n), (treatment_successes, treatment_n)]:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= n:
            raise ValueError('Each arm needs positive integer observations and valid success counts.')
    if not 0 < alpha < .5:
        raise ValueError('Invalid alpha.')
    pc, pt = control_successes/control_n, treatment_successes/treatment_n
    lc, uc = wilson(control_successes, control_n, alpha)
    lt, ut = wilson(treatment_successes, treatment_n, alpha)
    difference = pt-pc
    # Newcombe hybrid-score interval for two independent proportions.
    ci = (difference-math.sqrt((pt-lt)**2+(uc-pc)**2),
          difference+math.sqrt((ut-pt)**2+(pc-lc)**2))
    pooled = (control_successes+treatment_successes)/(control_n+treatment_n)
    expected = [n*p for n in [control_n, treatment_n] for p in [pooled, 1-pooled]]
    valid = min(expected) >= 5
    variance = pooled*(1-pooled)*(1/control_n+1/treatment_n)
    z = difference/math.sqrt(variance) if variance else None
    # Pearson chi-square with one degree of freedom = pooled z squared; no continuity correction.
    pvalue = math.erfc(abs(z)/math.sqrt(2)) if valid else None
    return {'control_rate': pc, 'treatment_rate': pt, 'control_ci': (lc, uc),
            'treatment_ci': (lt, ut), 'difference': difference, 'difference_ci': ci,
            'relative_lift': difference/pc if pc else None,
            'chi_square': z*z if valid else None, 'pvalue': pvalue,
            'minimum_expected_cell': min(expected), 'test_valid': valid,
            'significant': pvalue < alpha if valid else None}


def simulate(n_per_arm=10000, baseline=.10, effect=.015, seed=42):
    """Blocked 1:1 random assignment, one mature outcome per fictional reader."""
    if not isinstance(n_per_arm, int) or isinstance(n_per_arm, bool) or not 1 <= n_per_arm <= 200000:
        raise ValueError('Invalid simulation size.')
    if not 0 <= baseline <= 1 or not 0 <= baseline+effect <= 1:
        raise ValueError('Invalid simulation rates.')
    rng = random.Random(seed)
    arms = ['control']*n_per_arm+['treatment']*n_per_arm
    rng.shuffle(arms)
    rows = []
    for index, arm in enumerate(arms):
        probability = baseline+(effect if arm == 'treatment' else 0)
        rows.append({'reader_id': f'sim-{index:07d}', 'arm': arm,
                     'converted_30d': int(rng.random() < probability), 'outcome_complete': True})
    return rows
