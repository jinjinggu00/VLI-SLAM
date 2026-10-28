"""Data binding around byte-identical existing OpenCV PnP functions; no new estimator."""
from pathlib import Path
import ast,json,hashlib
import numpy as np,cv2
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;O=H.parent/'v131_author_gba_appearance_lvi_20261009'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=O/'verify131.py';tree=ast.parse(source.read_text());selected=[]
 for x in tree.body:
  if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CONFIG' for t in x.targets):selected.append(x)
  elif isinstance(x,ast.FunctionDef) and x.name in ['pnp','reproj','difference','cells','original_fixture']:selected.append(x)
 scope=dict(np=np,cv2=cv2,Rotation=Rotation)
 exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
 fixture=scope['original_fixture']();cfg=scope['CONFIG'];cv2.setNumThreads(1)
 result={'original_numeric_source_sha256':sha(source),'original_fixture':fixture,'scene_tests':{},'accepted_graph_pairs':0,'scope':'Independent preliminary same-view check of original neural matches with unchanged original OpenCV solver and support/held gates. No graph factors from preliminary checks.'}
 pre={str(source):sha(source),str(Path(__file__)):sha(__file__)}
 for scene in ['scene_0001','scene_0002']:
  path=H/'crossmodal_probe'/scene/'raw.npz';pre[str(path)]=sha(path)
 (H/'crossmodal_pnp_pre_hashes.json').write_text(json.dumps(pre,indent=2)+'\n')
 for scene in ['scene_0001','scene_0002']:
  path=H/'crossmodal_probe'/scene/'raw.npz';a=np.load(path);idx=a['index'];uv=a['source_uv'][idx[:,0]];X=a['target_measured_xyz'][idx[:,1]];K=a['K']
  held=(idx[:,0]%5==0)|(idx[:,1]%5==0);train=~held;bad=[];rec={'raw_correspondences':len(idx),'training_count':int(train.sum()),'held_count':int(held.sum()),'ground_truth_input_to_fit':False}
  if len(idx)<cfg['min_correspondences'] or held.sum()<5 or train.sum()<cfg['min_training_inliers']+3:bad.append('independent_support')
  if scope['cells'](uv[train])<cfg['min_occupied_cells']:bad.append('spatial_support')
  if not bad:
   T,inliers=scope['pnp'](X[train],uv[train],K,1320 if scene=='scene_0001' else 1321)
   if T is None:bad.append('native_pnp_failed')
   else:
    rec['inliers']=len(inliers);rec['inlier_ratio']=len(inliers)/train.sum();rec['pose_measured_cloud_to_camera']=T.tolist();delta=scope['difference'](np.eye(4),T);rec['known_same_view_error']=delta
    if len(inliers)<cfg['min_training_inliers'] or rec['inlier_ratio']<cfg['min_inlier_ratio']:bad.append('pnp_support')
    if delta['translation_m']>cfg['reciprocal_m'] or delta['rotation_deg']>cfg['reciprocal_deg']:bad.append('known_same_view_extrinsic')
    err=scope['reproj'](X[held],uv[held],K,T)
    if err is None:bad.append('held_cheirality')
    else:
     rec['held_median_px']=float(np.median(err));rec['held_p90_px']=float(np.percentile(err,90))
     if rec['held_median_px']>cfg['held_median_px'] or rec['held_p90_px']>cfg['held_p90_px']:bad.append('held_reprojection')
  rec['failures']=bad;rec['passed']=not bad;result['scene_tests'][scene]=rec
 for path,value in pre.items():
  if sha(path)!=value:raise ValueError('Actual pre-run source changed')
 (H/'crossmodal_pnp132_report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2,allow_nan=False),flush=True)
if __name__=='__main__':main()
