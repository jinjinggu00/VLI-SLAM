import importlib.util,unittest
class PostGlobalAdmission(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('admission132'),'Unchanged geometry gate must explicitly classify loop admission, not silently omit a failed record')
  return __import__('admission132')
 def cfg(self):return dict(held_median_px=3.,held_p90_px=6.,depth_median_m=.35,depth_p90_m=.8)
 def pair(self,i,j,failed=False):
  return dict(i=i,j=j,n=10,after_median_px=1.,after_reverse_median_px=1.,after_p90_px=7. if failed else 2.,depth_median_m=.1,depth_p90_m=.2,failures=['post_global_held_reprojection'] if failed else [])
 def bindings(self):
  return dict(loop_pairs=[dict(i=1,j=100,kind='loop'),dict(i=2,j=102,kind='loop'),dict(i=3,j=4,kind='temporal')],points=[dict(id=0,pair=[1,100],support=[[1,7],[100,9]]),dict(id=1,pair=[2,102],support=[[2,11],[102,13]]),dict(id=2,pair=[3,4],support=[[3,17],[4,19]])])
 def test_failed_loop_is_retained_as_rejected_and_unknown_pairs_are_not_added(self):
  r=self.module().approve(self.bindings(),[self.pair(1,100,True),self.pair(2,102),self.pair(3,4)],self.cfg())
  self.assertEqual([(x['i'],x['j']) for x in r['approved']],[(2,102),(3,4)])
  self.assertEqual(len(r['rejected']),1);self.assertEqual(r['rejected'][0]['i'],1)
  self.assertFalse(r['independent_validation_claimed']);self.assertTrue(r['frame_coverage_unchanged'])
 def test_missing_evidence_or_failed_temporal_gate_blocks(self):
  m=self.module();b=self.bindings()
  with self.assertRaises(ValueError):m.approve(b,[self.pair(1,100),self.pair(2,102)],self.cfg())
  with self.assertRaises(ValueError):m.approve(b,[self.pair(1,100),self.pair(2,102),self.pair(3,4,True)],self.cfg())
if __name__=='__main__':unittest.main()
