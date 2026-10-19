"""Portable inference. Python and the hosted Worker consume the same trained artifact."""
import math

BRANDS = ['Aftenposten', 'VG', 'Bergens Tidende', 'E24']
ACTIONS = [
    {'id': 'open', 'name': 'Keep reading', 'price': 249, 'margin': 199, 'cost': 0, 'description': 'No interruption. Organic subscriptions remain possible.'},
    {'id': 'standard', 'name': 'Digital subscription', 'price': 249, 'margin': 199, 'cost': 0.8, 'description': '249 kr for the first 30 days.'},
    {'id': 'trial', 'name': 'Introductory access', 'price': 49, 'margin': 39, 'cost': 0.8, 'description': '49 kr for the first 30 days.'},
    {'id': 'discount', 'name': 'Welcome offer', 'price': 129, 'margin': 103, 'cost': 0.8, 'description': '129 kr for the first 30 days.'},
]
DEFAULT = {'brand': 'Aftenposten', 'active_days': 12, 'articles': 24, 'premium_reads': 8, 'recency_hours': 8, 'dismissals': 0, 'mobile': True, 'consent': True, 'subscribed': False, 'public_interest': False}
LIMITS = {'active_days': (0, 28), 'articles': (0, 120), 'premium_reads': (0, 40), 'recency_hours': (0, 168), 'dismissals': (0, 5)}

def validate(payload):
    if not isinstance(payload, dict): raise ValueError('Expected a JSON object.')
    if set(payload) - set(DEFAULT): raise ValueError('Unknown reader fields.')
    reader = dict(DEFAULT, **payload)
    if reader['brand'] not in BRANDS: raise ValueError('Unsupported brand.')
    for key, (low, high) in LIMITS.items():
        value = reader[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{key} must be a finite number between {low} and {high}.')
        if key != 'recency_hours' and value != int(value): raise ValueError(f'{key} must be an integer.')
    for key in ['mobile', 'consent', 'subscribed', 'public_interest']:
        if not isinstance(reader[key], bool): raise ValueError(f'{key} must be boolean.')
    if reader['premium_reads'] > reader['articles']: raise ValueError('Premium reads cannot exceed total articles.')
    return reader

def features(r):
    d, a, p, h, f = r['active_days']/28, r['articles']/120, r['premium_reads']/40, r['recency_hours']/168, r['dismissals']/5
    return [1, d, a, p, h, f, float(r['mobile']), d*d, p*p, d*p, d*h, *[float(r['brand']==b) for b in BRANDS[1:]]]

FEATURES = ['Intercept', 'Reading frequency', 'Article depth', 'Premium interest', 'Time since visit', 'Offer dismissals', 'Mobile device', 'Reading frequency²', 'Premium interest²', 'Frequency × premium interest', 'Frequency × recency', 'Brand: VG', 'Brand: Bergens Tidende', 'Brand: E24']

def blocked(r):
    reasons = []
    if not r['consent']: reasons.append('Personalisation is off: no targeted offer.')
    if r['subscribed']: reasons.append('Existing subscriber: no acquisition offer.')
    if r['public_interest']: reasons.append('Public-interest story: keep access open.')
    if r['dismissals'] >= 3: reasons.append('Frequency cap: three or more recent dismissals.')
    return reasons

def decision(payload, artifact):
    r = validate(payload)
    reasons = blocked(r)
    if reasons:
        rows = [dict(a, probability=None, uplift=0 if i == 0 else None, value=None, eligible=i == 0) for i,a in enumerate(ACTIONS)]
        return {'reader':r, 'action':rows[0], 'alternatives':rows, 'incremental_value':0, 'guardrails':reasons, 'feature_contributions':[], 'model_version':artifact['version'], 'data_mode':'synthetic', 'objective':'Expected first-30-day contribution, NOK per reader', 'note':'Rule-based bypass: no model scoring or offer-effect estimates. The acquisition model is not applicable to existing subscribers; targeted actions lack experimental support for guarded readers.'}
    x = features(r)
    probs = [1/(1+math.exp(-max(-30,min(30,sum(a*b for a,b in zip(x,w)))))) for w in artifact['weights']]
    rows = [dict(a, eligible=True, probability=probs[i], uplift=probs[i]-probs[0], value=probs[i]*a['margin']-a['cost']) for i,a in enumerate(ACTIONS)]
    reasons = blocked(r)
    winner = 0 if reasons else max(range(4), key=lambda i: rows[i]['value'])
    base, choice = rows[0], rows[winner]
    terms = sorted([{'feature': FEATURES[i], 'log_odds_contribution': x[i]*artifact['weights'][winner][i]} for i in range(1,len(x))], key=lambda t: abs(t['log_odds_contribution']), reverse=True)[:4]
    return {'reader':r, 'action':choice, 'alternatives':rows, 'incremental_value':choice['value']-base['value'], 'guardrails':reasons, 'feature_contributions':terms, 'model_version':artifact['version'], 'data_mode':'synthetic', 'objective':'Expected first-30-day contribution, NOK per reader', 'note':'Predicted average effects for similar readers, not an observable individual causal effect. Future renewal value is excluded.'}
