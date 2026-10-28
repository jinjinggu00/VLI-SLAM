"""Small original-library direction and keypoint/pixel indexing contracts."""
from pathlib import Path
import sys,json
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'upstream/ffreereg'))
from Utils.pose import kabsch_refine
from Utils.utils import dpt_3d_convert
from scipy.spatial.transform import Rotation
rng=np.random.default_rng(132);X=rng.uniform([-1.,-.7,4.],[1.,.7,8.],size=(64,3))
T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.06,-.09,.03]).as_matrix();T[:3,3]=[.35,-.2,.13]
Y=X@T[:3,:3].T+T[:3,3]
scale,Q=kabsch_refine(True).sse3_cal(X,Y)
if abs(scale-1)>1e-10 or np.max(np.abs(Q-T))>1e-10:raise ValueError('Original Kabsch direction contract')
K=np.array([[440.,0,377.],[0,450.,238.],[0,0,1.]])
selected=np.array([51,9,40,18,5,32,13,60])
uv,z=dpt_3d_convert().proj_3to2(X[selected],K,np.eye(4))
expected=np.column_stack([440*X[selected,0]/X[selected,2]+377,450*X[selected,1]/X[selected,2]+238])
if np.max(np.abs(uv-expected))>.002:raise ValueError('Selected keypoint uv mapping contract')
report=dict(original_kabsch_transform_error=float(np.max(np.abs(Q-T))),original_projector_known_pixel_error=float(np.max(np.abs(uv-expected))),upstream_demo_index_binding_issue='demo.py projects full image_pc but passes sampled img_kpts. Adapter projects sampled img_kpts using unchanged original projector.',source_numerical_algorithms_changed=False)
(H/'crossmodal_fixture.json').write_text(json.dumps(report,indent=2));print(report)
