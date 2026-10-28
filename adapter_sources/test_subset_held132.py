import importlib.util,unittest
import numpy as np,cv2
from scipy.spatial.transform import Rotation
class SelectedSubsetHeld(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('subset_held132'),'Held gates must validate the transform fitted to the actual final chosen training subset')
  return __import__('subset_held132')
 def sample(self):
  rng=np.random.default_rng(132);X=rng.uniform([-1.,-.7,4.],[1.,.7,8.],size=(20,3));T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.06,-.09,.03]).as_matrix();T[:3,3]=[.35,-.2,.13];Y=X@T[:3,:3].T+T[:3,3];K=np.array([[450.,0,375.5],[0,450.,239.5],[0,0,1.]])
  ui=cv2.projectPoints(X,np.zeros(3),np.zeros(3),K,np.zeros(4))[0].reshape(-1,2);uj=cv2.projectPoints(Y,np.zeros(3),np.zeros(3),K,np.zeros(4))[0].reshape(-1,2)
  return X,Y,ui,uj,K,T
 def cfg(self):return dict(held_median_px=3.,held_p90_px=6.,depth_median_m=.35,depth_p90_m=.8)
 def test_changed_chosen_fit_cannot_reuse_prior_good_qualification(self):
  m=self.module();X,Y,ui,uj,K,T=self.sample()
  self.assertTrue(m.check(X,Y,ui,uj,K,T,np.linalg.inv(T),self.cfg())['passed'])
  altered=T.copy();altered[:3,3]+=[.2,-.15,0]
  self.assertFalse(m.check(X,Y,ui,uj,K,altered,np.linalg.inv(altered),self.cfg())['passed'])
 def test_measured_3d_disagreement_is_checked_before_graph_admission(self):
  m=self.module();X,Y,ui,uj,K,T=self.sample();Y=Y+np.array([0.,0.,1.])
  r=m.check(X,Y,ui,uj,K,T,np.linalg.inv(T),self.cfg())
  self.assertIn('post_global_held_depth_3d',r['failures'])
if __name__=='__main__':unittest.main()
