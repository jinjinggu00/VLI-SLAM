import unittest
import torch
from depth_input132 import call_original_depth
class InputBinding(unittest.TestCase):
 def test_native_parameter_dtype_and_tensor_focal(self):
  marker=torch.zeros(1,dtype=torch.float16)
  class API:
   def parameters(self):return iter([marker])
   def infer(self,x,f_px=None):
    self.asserted=(x.dtype,x.device,f_px)
    if x.dtype!=marker.dtype or x.device!=marker.device or not torch.is_tensor(f_px) or f_px.ndim!=0:raise ValueError('Original tensor API requirement')
    return f_px
  api=API();out=call_original_depth(api,lambda _:torch.zeros(3,4,6,dtype=torch.float32),None,443.25)
  self.assertEqual(float(out),443.25)
 def test_no_focal_keeps_original_estimator_branch(self):
  marker=torch.zeros(1,dtype=torch.float16)
  class API:
   def parameters(self):return iter([marker])
   def infer(self,x,f_px=None):
    if x.dtype!=marker.dtype or f_px is not None:raise ValueError('Original unknown focal behavior changed')
    return 'original'
  self.assertEqual(call_original_depth(API(),lambda _:torch.zeros(3,4,6),None,None),'original')
if __name__=='__main__':unittest.main(verbosity=2)
