"""Content entitlement and commercial offer timing are separate decisions."""
import hashlib, json
from pathlib import Path
POLICY=json.loads((Path(__file__).resolve().parents[1]/'config/paywall_policy.json').read_text())
def decide(reader,payload,policy=POLICY):
    for key in ['public_interest','premium']:
        if key in payload and not isinstance(payload[key],bool):raise ValueError(key+' must be boolean.')
    dismissals=payload.get('dismissals',0);articles=payload.get('articles_in_session',1)
    for key,value in [('dismissals',dismissals),('articles_in_session',articles)]:
        if isinstance(value,bool) or not isinstance(value,int) or not 0<=value<=100:raise ValueError('Invalid '+key+'.')
    access='full' if reader['subscribed'] or payload.get('public_interest',False) or not payload.get('premium',True) else 'teaser'
    assignment=int(hashlib.sha256((reader['id']+policy['version']).encode()).hexdigest()[:8],16)/2**32
    eligible=not reader['subscribed'] and reader['consent'] and not reader['suppressed'] and not payload.get('public_interest',False)
    arm=('control' if assignment<policy['control_fraction'] else 'policy') if eligible else 'excluded'
    if reader['subscribed']:action,reason='open','Subscriber entitlement grants full access.'
    elif payload.get('public_interest',False):action,reason='open','Editorial public-interest exemption grants full access.'
    elif reader['suppressed']:action,reason='no_offer','Commercial suppression: no offer; content entitlement is unchanged.'
    elif dismissals>=3:action,reason='no_offer','Offer fatigue: defer the offer; content entitlement is unchanged.'
    elif not reader['consent']:action,reason='standard_offer','No personalisation consent: show a contextual standard offer, without propensity targeting.'
    elif arm=='control':action,reason='standard_offer','Stable control receives the standard offer policy.'
    elif articles<policy['minimum_articles_in_session'] and reader['sales']<policy['high_intent_threshold']:action,reason='no_offer','Defer until the second article unless high intent; content entitlement is unchanged.'
    elif reader['sales']>=policy['high_intent_threshold']:action,reason='standard_offer','High-intent rule selects the standard offer.'
    elif reader['sales']>=policy['trial_threshold']:action,reason='trial_offer','Eligible reader receives an introductory offer.'
    else:action,reason='no_offer','Low-intent rule defers commercial interruption; content entitlement is unchanged.'
    prices=policy['brands'][reader['brand']]
    offer=None if action in ['open','no_offer'] else {'id':action,'currency':prices['currency'],'price':prices['trial_price'] if action=='trial_offer' else prices['standard_price'],'terms':'Illustrative first-month price; no real purchase.'}
    return {'reader_id':reader['id'],'brand':reader['brand'],'content_access':access,'action':action,'offer':offer,'reason':reason,'experiment':policy['version'],'arm':arm,'experiment_eligible':eligible,'articles_in_session':articles,'sales_probability':reader['sales'] if reader['consent'] and not reader['subscribed'] else None,'policy':'Handcrafted learning thresholds, not estimated causal uplift.'}
