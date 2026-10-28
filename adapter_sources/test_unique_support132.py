import importlib.util,unittest
import numpy as np
class UniqueMeasuredSupport(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('unique_support132'),'Deduplication must recheck original minimum support and spatial coverage')
  return __import__('unique_support132')
 def cfg(self):return dict(min_training_inliers=14,min_occupied_cells=4,depth_median_m=.35,depth_p90_m=.8)
 def spread(self,n):return np.array([[70+190*(i%4),60+150*((i//4)%3)] for i in range(n)],dtype=float)
 def test_prior_qualified_pair_with_only_twelve_remaining_features_fails(self):
  m=self.module();self.assertFalse(m.support(self.spread(12),self.spread(12),self.cfg()))
 def test_many_points_in_one_cell_are_not_geometric_support(self):
  m=self.module();self.assertFalse(m.support(np.full((24,2),100.),self.spread(24),self.cfg()))
 def test_actual_spread_support_passes_but_conflicting_measured_depth_fails(self):
  m=self.module();self.assertTrue(m.support(self.spread(24),self.spread(24),self.cfg()))
  self.assertFalse(m.depth_consistent(np.full(24,.81),self.cfg()))
  self.assertTrue(m.depth_consistent(np.full(24,.1),self.cfg()))
if __name__=='__main__':unittest.main()
