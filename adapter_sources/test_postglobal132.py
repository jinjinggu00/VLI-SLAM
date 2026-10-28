import importlib.util,unittest
import numpy as np
class PostGlobalHeldGate(unittest.TestCase):
 def implementation(self):
  spec=importlib.util.find_spec('held_gate132')
  self.assertIsNotNone(spec,'Final gate helper missing: post-global depth failure must block release')
  return __import__('held_gate132')
 def limits(self):
  return dict(held_median_px=3.,held_p90_px=6.,depth_median_m=.35,depth_p90_m=.8)
 def test_zero_pixel_error_cannot_hide_wrong_measured_3d(self):
  gate=self.implementation()
  self.assertIn('post_global_held_depth_3d',gate.failures(np.zeros(20),np.zeros(20),np.full(20,.81),self.limits()))
 def test_original_depth_and_pixel_boundaries_are_retained(self):
  gate=self.implementation()
  self.assertEqual(gate.failures(np.full(20,3.),np.full(20,3.),np.full(20,.35),self.limits()),[])
  self.assertIn('post_global_held_reprojection',gate.failures(np.full(20,3.01),np.zeros(20),np.zeros(20),self.limits()))
 def test_nonfinite_or_empty_evidence_never_passes(self):
  gate=self.implementation()
  self.assertTrue(gate.failures(np.zeros(20),np.zeros(20),np.full(20,np.nan),self.limits()))
  self.assertTrue(gate.failures(np.array([]),np.array([]),np.array([]),self.limits()))
 def test_failed_scene_must_return_nonzero_instead_of_success(self):
  gate=self.implementation()
  with self.assertRaises(ValueError):gate.require_complete(dict(scenes={'scene_0001':dict(passed=True,failures=[]),'scene_0002':dict(passed=False,failures=['post_global_held_depth_3d'])}))
  gate.require_complete(dict(scenes={'scene_0001':dict(passed=True,failures=[]),'scene_0002':dict(passed=True,failures=[])}))
if __name__=='__main__':unittest.main()
