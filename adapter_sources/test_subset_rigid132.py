import importlib.util,unittest
import numpy as np,cv2
from subset_held132 import check as separate_reverse_check
class OnePhysicalRegistration(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('subset_rigid132'),'Joint graph has one rigid relative transform, not two independently compensating poses')
  return __import__('subset_rigid132')
 def cfg(self):return dict(held_median_px=3.,held_p90_px=6.,depth_median_m=.35,depth_p90_m=.8)
 def inputs(self):
  rng=np.random.default_rng(132);X=rng.uniform([-1.,-.7,4.],[1.,.7,5.],size=(20,3));F=np.eye(4);F[:3,3]=[1.,0,0];Y=X+F[:3,3];K=np.array([[450.,0,375.5],[0,450.,239.5],[0,0,1.]])
  ui=cv2.projectPoints(X,np.zeros(3),np.zeros(3),K,np.zeros(4))[0].reshape(-1,2);uj=cv2.projectPoints(Y,np.zeros(3),np.zeros(3),K,np.zeros(4))[0].reshape(-1,2)
  return X,Y,ui,uj,K,F
 def test_separately_compensated_pose_pair_is_not_one_physical_alignment(self):
  X,Y,ui,uj,K,F=self.inputs();biased=Y*1.05;U=np.eye(4);U[:3,3]=[-1.05,0,0]
  self.assertTrue(separate_reverse_check(X,biased,ui,uj,K,F,U,self.cfg())['passed'])
  self.assertFalse(self.module().check(X,biased,ui,uj,K,F,U,self.cfg())['passed'])
 def test_true_single_rigid_registration_is_retained(self):
  X,Y,ui,uj,K,F=self.inputs()
  self.assertTrue(self.module().check(X,Y,ui,uj,K,F,np.linalg.inv(F),self.cfg())['passed'])
if __name__=='__main__':unittest.main()
