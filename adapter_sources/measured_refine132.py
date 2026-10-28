from pathlib import Path
import sys,numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'upstream/ffreereg'))
from Utils.pose import kabsch_refine
DISTANCE_M=.3
def refine(X,Y,initial):
 X,Y=np.asarray(X,dtype=float),np.asarray(Y,dtype=float);initial=np.asarray(initial,dtype=float)
 if X.ndim!=2 or X.shape[1]!=3 or X.shape!=Y.shape or len(X)<6 or initial.shape!=(4,4) or not all(np.isfinite(a).all() for a in [X,Y,initial]):raise ValueError('Actual finite measured3D correspondence contract')
 ids=np.column_stack([np.arange(len(X)),np.arange(len(X))]).astype(np.int32)
 # Exact author rigid refiner; known sensor metric scale is fixed1, not optimized.
 Q=kabsch_refine(False).run(X,Y,ids,1.,initial,thres=[2*DISTANCE_M,DISTANCE_M])
 if Q is None or not np.isfinite(Q).all():raise ValueError('Original measured3D refiner failed')
 e=np.linalg.norm(X@Q[:3,:3].T+Q[:3,3]-Y,axis=1)
 return Q,e<DISTANCE_M
