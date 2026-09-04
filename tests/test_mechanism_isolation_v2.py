import unittest
import numpy as np
from experiments.run_mechanism_isolation_v2 import make_control_panel
from sklearn.model_selection import GroupKFold, KFold
class MechanismIsolationTests(unittest.TestCase):
 def test_off_off_has_no_static_fingerprint(self):
  d=make_control_panel(False,False,7); self.assertNotIn('X_5',d); self.assertTrue((d.groupby('episode_id').size()==20).all())
  for c in [x for x in d if x.startswith('X_')]: self.assertTrue((d.groupby('episode_id')[c].nunique()>1).all())
 def test_independent_control_is_fixed_and_not_event_truncated(self):
  d=make_control_panel(False,False,8); self.assertTrue(d.fixed_length.all()); self.assertTrue(d.T_e.isna().all()); self.assertTrue((d.at_risk==1).all())
 def test_split_semantics(self):
  d=make_control_panel(False,False,9); X=d[['X_1']].values; y=d.Y.values; g=d.episode_id.values
  for a,b in GroupKFold(5).split(X,y,g): self.assertTrue(set(g[a]).isdisjoint(set(g[b])))
  a,b=next(KFold(5,shuffle=True,random_state=9).split(X,y)); self.assertTrue(bool(set(g[a])&set(g[b])))
