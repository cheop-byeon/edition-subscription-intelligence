import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
from paywall import decide,POLICY
class PaywallTests(unittest.TestCase):
    def reader(self,**updates):
        r={'id':'prospect','brand':'VG','subscribed':False,'consent':True,'suppressed':False,'sales':.2};r.update(updates);return r
    def policy(self):
        p=copy.deepcopy(POLICY);p['control_fraction']=0;return p
    def test_offer_deferral_never_unlocks_paid_content(self):
        d=decide(self.reader(),{'articles_in_session':1},self.policy())
        self.assertEqual(d['action'],'no_offer');self.assertEqual(d['content_access'],'teaser');self.assertIsNone(d['offer'])
        d=decide(self.reader(),{'dismissals':3},self.policy());self.assertEqual(d['content_access'],'teaser')
    def test_consent_controls_targeting_not_entitlement(self):
        d=decide(self.reader(consent=False),{},self.policy())
        self.assertEqual(d['action'],'standard_offer');self.assertEqual(d['content_access'],'teaser');self.assertIsNone(d['sales_probability']);self.assertEqual(d['arm'],'excluded')
    def test_subscriber_and_editorial_access(self):
        for payload,r in [({},self.reader(subscribed=True)),({'public_interest':True},self.reader())]:
            d=decide(r,payload,self.policy());self.assertEqual(d['content_access'],'full');self.assertIsNone(d['offer'])
    def test_timing_and_brand_offer(self):
        d=decide(self.reader(),{'articles_in_session':2},self.policy())
        self.assertEqual(d['action'],'trial_offer');self.assertEqual(d['offer']['price'],29)
        d=decide(self.reader(sales=.8),{'articles_in_session':1},self.policy());self.assertEqual(d['action'],'standard_offer')
    def test_control_is_standard_not_free_access(self):
        p=self.policy();p['control_fraction']=1
        d=decide(self.reader(),{},p);self.assertEqual(d['arm'],'control');self.assertEqual(d['action'],'standard_offer');self.assertEqual(d['content_access'],'teaser')
    def test_bad_context(self):
        for context in [{'premium':'yes'},{'articles_in_session':-1},{'dismissals':True}]:
            with self.assertRaises(ValueError):decide(self.reader(),context)
