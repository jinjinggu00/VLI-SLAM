from pathlib import Path
import sys,json,ast,resource
import numpy as np,cv2
from scipy.spatial.transform import Rotation
from bridge_io132 import load,save,sha
from source_binding132 import verified_workspace_source
from measured_refine132 import refine,DISTANCE_M
from unique_support132 import support
H=Path(__file__).resolve().parent;W=H.parents[3];OUT=H/'measured_registration_r5'
tree=ast.parse((H/'verify132.py').read_text())
body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['difference','reproj']]
CONFIG=load(H/'verification_full/config.json')
ns=dict(np=np,cv2=cv2,Rotation=Rotation)
exec(compile(ast.Module(body=body,type_ignores=[]),str(H/'verify132.py'),'exec'),ns)
cv2.setNumThreads(1)
def key(path):return '/workspace/'+Path(path).resolve().relative_to(W.resolve()).as_posix()
def derive(r,a,b,K):
 idx=np.array(r['index'],dtype=int)[r['optimization_correspondence_rows']]
 if np.any((idx[:,0]%5==0)|(idx[:,1]%5==0)):raise ValueError('Held feature in measured registration fitting')
 X=a['xyz'][idx[:,0]];Y=b['xyz'][idx[:,1]]
 Q,good=refine(X,Y,r['forward']);delta=ns['difference'](np.array(r['forward']),Q)
 de=np.linalg.norm(X@Q[:3,:3].T+Q[:3,3]-Y,axis=1)
 mask=idx[good];bad=[]
 if len(mask)<CONFIG['min_training_inliers'] or float(np.mean(good))<CONFIG['min_inlier_ratio'] or not support(a['uv'][mask[:,0]],b['uv'][mask[:,1]],CONFIG):bad.append('original_metric_inlier_support')
 if delta['translation_m']>CONFIG['reciprocal_m'] or delta['rotation_deg']>CONFIG['reciprocal_deg']:bad.append('original_2d_3d_pose_conflict')
 return dict(i=r['i'],j=r['j'],kind=r['kind'],initial_training_rows=len(idx),inlier_feature_pairs=mask.tolist(),transform=Q.tolist(),native3d_inliers=len(mask),fraction=float(np.mean(good)),pose_difference=delta,depth_median_m=float(np.median(de)),depth_p90_m=float(np.percentile(de,90)),accepted=not bad,failures=bad)
def main():
 OUT.mkdir(exist_ok=False)
 manifest={**load(H/'association_full/output_hashes.json'),**load(H/'verification_full/output_hashes.json')}
 for k,v in manifest.items():verified_workspace_source(W/Path(k).relative_to('/workspace'),W,manifest)
 pre={key(p):sha(p) for p in [Path(__file__),H/'measured_refine132.py',H/'unique_support132.py',H/'verify132.py',H/'upstream/ffreereg/Utils/pose.py',H/'upstream/ffreereg/Utils/utils.py',H/'association_full/output_hashes.json',H/'verification_full/output_hashes.json']}
 save(OUT/'pre_execution_hashes.json',pre)
 report=dict(scenes={},native_original='WHU-USI3DV/FreeReg FFreeReg kabsch_refine(False).run; fixed metric scale1; original distance0.3m',neural_direct_branch_enabled=False,held_identity_fitting=False,repeat_samples={},official_ate=None)
 for scene in ['scene_0001','scene_0002']:
  q=OUT/scene;q.mkdir()
  rows=[r for r in load(H/'verification_full'/scene/'pairs.json') if r['accepted']]
  cal=load(H.parent/'v131_author_gba_appearance_lvi_20261009/prepared_r3'/scene/'calibration.json');fx,fy,cx,cy=cal['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  records=[]
  for r in rows:
   a=np.load(H/'association_full'/scene/(str(r['i'])+'.npz'));b=np.load(H/'association_full'/scene/(str(r['j'])+'.npz'));records.append(derive(r,a,b,K))
  save(q/'pairs.json',records)
  repetitions=[]
  for k in [0,len(rows)//2,len(rows)-1]:
   r=rows[k];a=np.load(H/'association_full'/scene/(str(r['i'])+'.npz'));b=np.load(H/'association_full'/scene/(str(r['j'])+'.npz'));v=derive(r,a,b,K)
   if v!=records[k]:raise ValueError('Original metric registration fixed sample differs')
   repetitions.append(dict(i=r['i'],j=r['j'],bit_exact=True))
  report['repeat_samples'][scene]=repetitions
  report['scenes'][scene]=dict(pairs=len(records),accepted=sum(r['accepted'] for r in records),kept_train_matches=sum(r['native3d_inliers'] for r in records if r['accepted']),failures={b:sum(b in r['failures'] for r in records) for b in sorted({b for r in records for b in r['failures']})})
  print('AUTHOR MEASURED RIGID REGISTRATION',scene,report['scenes'][scene],flush=True)
 for k,v in pre.items():
  if sha(W/Path(k).relative_to('/workspace'))!=v:raise ValueError('Actual upstream source changed during registration')
 report['peak_child_rss_mib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 save(OUT/'report.json',report)
 save(OUT/'output_hashes.json',{key(q):sha(q) for q in OUT.rglob('*') if q.is_file() and q.name!='output_hashes.json'})
if __name__=='__main__':main()
