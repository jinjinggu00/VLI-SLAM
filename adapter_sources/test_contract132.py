import unittest
import numpy as np
from bridge_io132 import feature_contract, held_feature_mask, optical_rays

class Interfaces(unittest.TestCase):
 def test_empty_and_nonfinite_feature_contract(self):
  feature_contract(np.empty((0,2)),np.empty((0,128)),np.empty(0),[752,480])
  with self.assertRaises(ValueError):feature_contract(np.ones((1,2)),np.ones((1,256)),np.ones(1),[752,480])
  with self.assertRaises(ValueError):feature_contract(np.array([[np.nan,1.]]),np.ones((1,128)),np.ones(1),[752,480])
  with self.assertRaises(ValueError):feature_contract(np.array([[752.,1.]]),np.ones((1,128)),np.ones(1),[752,480])
 def test_held_identity_is_global_at_either_endpoint(self):
  idx=np.array([[0,1],[1,5],[4,9],[10,15],[11,12]])
  self.assertEqual(held_feature_mask(idx).tolist(),[True,True,False,True,False])
 def test_original_opencv_ray_matches_known_offcenter_points(self):
  import cv2
  K=np.array([[440.,0.,377.],[0.,450.,238.],[0.,0.,1.]])
  X=np.array([[.5,-.1,7.],[-.3,.4,8.],[0.,0.,9.]])
  uv=cv2.projectPoints(X,np.zeros(3),np.zeros(3),K,np.zeros(4))[0].reshape(-1,2)
  np.testing.assert_allclose(optical_rays(uv,K),X/X[:,2,None],atol=1e-7)
if __name__=='__main__':unittest.main(verbosity=2)
