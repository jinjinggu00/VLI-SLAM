import numpy as np
from subset_held132 import check as frozen_check
def check(Xi,Xj,ui,uj,K,F,U,cfg):
 # Each tested motion is one physical SE3 with its own exact inverse.
 forward=frozen_check(Xi,Xj,ui,uj,K,F,np.linalg.inv(F),cfg)
 reverse=frozen_check(Xi,Xj,ui,uj,K,np.linalg.inv(U),U,cfg)
 bad=sorted(set(forward['failures']+reverse['failures']))
 return dict(passed=not bad,failures=bad,forward_physical=forward,reverse_physical=reverse,single_physical_transform=True,independent_reverse_fits_cannot_compensate_depth_bias=True)
