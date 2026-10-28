from pathlib import Path
import ast,numpy as np,cv2
from held_gate132 import failures
H=Path(__file__).resolve().parent
tree=ast.parse((H/'verify132.py').read_text())
body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reproj']
scope=dict(np=np,cv2=cv2)
exec(compile(ast.Module(body=body,type_ignores=[]),str(H/'verify132.py'),'exec'),scope)
def check(Xi,Xj,ui,uj,K,F,U,cfg):
 e=scope['reproj'](Xi,uj,K,F);r=scope['reproj'](Xj,ui,K,U)
 if e is None or r is None:return dict(passed=False,failures=['held_cheirality_after_actual_subset_fit'])
 de=np.linalg.norm(np.asarray(Xi)@F[:3,:3].T+F[:3,3]-Xj,axis=1)
 bad=failures(e,r,de,cfg)
 return dict(passed=not bad,failures=bad,n=len(e),forward_median_px=float(np.median(e)),reverse_median_px=float(np.median(r)),p90_px=float(max(np.percentile(e,90),np.percentile(r,90))),depth_median_m=float(np.median(de)),depth_p90_m=float(np.percentile(de,90)))
