import importlib.util,unittest
import numpy as np
from scipy.spatial.transform import Rotation
class OriginalRigidRefine(unittest.TestCase):
 def module(self):
  self.assertIsNotNone(importlib.util.find_spec('measured_refine132'),'Original rigid3D refinement binding absent')
  return __import__('measured_refine132')
 def test_known_rigid_transform_and_outliers_without_scale_fitting(self):
  m=self.module();rng=np.random.default_rng(132);X=rng.uniform([-1.,-.7,4.],[1.,.7,8.],size=(64,3))
  T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.06,-.09,.03]).as_matrix();T[:3,3]=[.35,-.2,.13]
  Y=X@T[:3,:3].T+T[:3,3];Y[-16:]+=np.array([4.,-2.,3.])
  initial=T.copy();initial[:3,3]+=[.04,-.03,.02]
  Q,good=m.refine(X,Y,initial)
  self.assertLess(np.max(np.abs(Q-T)),1e-10)
  self.assertTrue(good[:48].all());self.assertFalse(good[-16:].any())
  self.assertAlmostEqual(np.linalg.det(Q[:3,:3]),1.,places=10)
 def test_empty_or_nonfinite_measurement_cannot_be_registered(self):
  m=self.module()
  for X in (np.zeros((0,3)),np.full((20,3),np.nan)):
   with self.assertRaises(ValueError):m.refine(X,X,np.eye(4))
if __name__=='__main__':unittest.main()
