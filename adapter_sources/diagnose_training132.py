from pathlib import Path
import ast,json,numpy as np,cv2
from scipy.spatial.transform import Rotation
p=Path(__file__).resolve().parent
tree=ast.parse((p/'verify132.py').read_text());body=[r for r in tree.body if isinstance(r,ast.FunctionDef) and r.name in ['pnp','reproj','difference','cells']]
ns=dict(np=np,cv2=cv2,Rotation=Rotation,CONFIG=dict(min_training_inliers=14,min_inlier_ratio=.6,reprojection_px=3.,ransac_iterations=2000,confidence=.999));exec(compile(ast.Module(body=body,type_ignores=[]),'original_verify132','exec'),ns);cv2.setNumThreads(1)
out=[]
for scene in ['scene_0001','scene_0002']:
 cal=json.loads((p.parent/'v131_author_gba_appearance_lvi_20261009/prepared_r3'/scene/'calibration.json').read_text());fx,fy,cx,cy=cal['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
 ex=json.loads((p/'joint132'/scene/'native/feature_bindings.json').read_text());points=ex['points']
 for pair in ex['loop_pairs']:
  i,j=pair['i'],pair['j'];a=np.load(p/'association_full'/scene/(str(i)+'.npz'));b=np.load(p/'association_full'/scene/(str(j)+'.npz'))
  chosen=[t for t in points if t['pair']==[i,j]];ids=np.array([[r['support'][0][1],r['support'][1][1]] for r in chosen])
  Xi=a['xyz'][ids[:,0]];Xj=b['xyz'][ids[:,1]];ui=a['uv'][ids[:,0]];uj=b['uv'][ids[:,1]]
  T,fi=ns['pnp'](Xi,uj,K,130+i+j);U,ri=ns['pnp'](Xj,ui,K,260+i+j)
  if T is None or U is None:out.append(dict(scene=scene,i=i,j=j,failed='pnp'));continue
  common=sorted(set(map(int,fi))&set(map(int,ri)));depth=np.linalg.norm(Xi@T[:3,:3].T+T[:3,3]-Xj,axis=1);reciprocal=ns['difference'](np.eye(4),T@U)
  bad=len(common)<14 or min(len(fi),len(ri))/len(ids)<.6 or reciprocal['translation_m']>.3 or reciprocal['rotation_deg']>3 or np.median(depth)>.35 or np.percentile(depth,90)>.8
  rec=dict(scene=scene,i=i,j=j,n=len(ids),native_common=len(common),depth_median_m=float(np.median(depth)),depth_p90_m=float(np.percentile(depth,90)),reciprocal=reciprocal,training_rejected=bad)
  out.append(rec)
  if bad or [i,j] in [[45,1125],[115,425],[405,1125]]:print(rec,flush=True)
(p/'deduplicated_training132_diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
