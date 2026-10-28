"""Independent real held feature reprojection after full graph optimization."""
from common132 import *
import cv2
def projected(X,uv,K,T):
 Y=X@T[:3,:3].T+T[:3,3]
 if (Y[:,2]<=0).any():return None
 p=cv2.projectPoints(X,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,np.zeros(4))[0].reshape(-1,2)
 return np.linalg.norm(p-uv,axis=1)
def main():
 report={'scenes':{},'official_accuracy_verified':False,'held_feature_identity_rule':'either endpoint feature id divisible by 5; never in fitting or graph'}
 for scene in SCENES:
  root=H/'joint132'/scene;T=np.load(root/'native/poses.npy');B=np.load(O/'prepared_r3'/scene/'best.npy')
  bindings=load(root/'native/feature_bindings.json');used={(p['i'],p['j']) for p in bindings['loop_pairs']}
  images={r['node']:r for r in load(O/'prepared_r3'/scene/'images.json')};fx,fy,cx,cy=load(O/'prepared_r3'/scene/'calibration.json')['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  records=[];before=[];after=[];bad=[]
  for r in load(H/'verification_full'/scene/'pairs.json'):
   i,j=r['i'],r['j']
   if (i,j) not in used:continue
   a=np.load(H/'association_full'/scene/(str(i)+'.npz'));b=np.load(H/'association_full'/scene/(str(j)+'.npz'));idx=np.array(r['index'],dtype=int)[r['held_correspondence_rows']]
   if len(idx)<5 or not np.all((idx[:,0]%5==0)|(idx[:,1]%5==0)):raise ValueError('Actual held identity mismatch')
   A=np.array(images[i]['offset']);Z=np.array(images[j]['offset']);beforeT=np.linalg.inv(B[j]@Z)@(B[i]@A);afterT=np.linalg.inv(T[j]@Z)@(T[i]@A)
   e0=projected(a['xyz'][idx[:,0]],b['uv'][idx[:,1]],K,beforeT);e1=projected(a['xyz'][idx[:,0]],b['uv'][idx[:,1]],K,afterT)
   if e0 is None or e1 is None:bad.append('held_positive_depth_'+str(i));continue
   reverse=projected(b['xyz'][idx[:,1]],a['uv'][idx[:,0]],K,np.linalg.inv(afterT))
   if reverse is None:bad.append('held_reverse_positive_depth_'+str(i));continue
   # Re-check the same prospectively frozen held support gates after global solve.
   failures=[]
   if max(np.median(e1),np.median(reverse))>3. or max(np.percentile(e1,90),np.percentile(reverse,90))>6.:failures.append('post_global_held_reprojection')
   records.append(dict(i=i,j=j,n=len(idx),before_median_px=float(np.median(e0)),after_median_px=float(np.median(e1)),after_reverse_median_px=float(np.median(reverse)),after_p90_px=float(max(np.percentile(e1,90),np.percentile(reverse,90))),failures=failures))
   bad+=failures;before.extend(map(float,e0));after.extend(map(float,e1))
  if not records:raise ValueError('No independent real held observations')
  direct_records=[]
  direct_used={(p['i'],p['j']) for p in bindings['crossmodal_pairs']}
  for r in []:
   i,j=r['i'],r['j']
   if (i,j) not in direct_used:continue
   a=np.load(H/'crossmodal_r3'/scene/('%d_%d.npz'%(i,j)))
   held=a['held'];ids=a['source_ids'][held]
   if len(ids)<5 or not np.all(ids%5==0):raise ValueError('Direct held identity mismatch')
   Xi=a['source_xyz'][held];Xj=a['target_xyz'][held];uv=a['source_uv'][held]
   A=np.array(images[i]['offset']);Z=np.array(images[j]['offset'])
   original=np.linalg.inv(B[i]@A)@(B[j]@Z);actual=np.linalg.inv(T[i]@A)@(T[j]@Z)
   e0=projected(Xj,uv,K,original);e1=projected(Xj,uv,K,actual)
   failures=[]
   if e1 is None:failures.append('direct_held_positive_depth')
   else:
    de=np.linalg.norm(Xj@actual[:3,:3].T+actual[:3,3]-Xi,axis=1)
    if np.median(e1)>3. or np.percentile(e1,90)>6.:failures.append('direct_post_global_held_pixel')
    if np.median(de)>.35 or np.percentile(de,90)>.8:failures.append('direct_post_global_held_depth')
    direct_records.append(dict(i=i,j=j,n=len(ids),before_median_px=None if e0 is None else float(np.median(e0)),after_median_px=float(np.median(e1)),after_p90_px=float(np.percentile(e1,90)),depth_median_m=float(np.median(de)),depth_p90_m=float(np.percentile(de,90)),failures=failures))
   bad+=failures
  if direct_used and not direct_records:raise ValueError('Admitted direct factors lack independent genuine held support')
  if len(direct_records)!=len(direct_used):raise ValueError('Admitted direct factor/held record count differs')
  report['scenes'][scene]=dict(direct_crossmodal_pairs=direct_records,pairs=records,held_observations=len(after),before_median_px=float(np.median(before)),after_median_px=float(np.median(after)),before_p90_px=float(np.percentile(before,90)),after_p90_px=float(np.percentile(after,90)),failures=sorted(set(bad)),passed=not bad)
  print('HELD IMAGE GLOBAL',scene,{k:v for k,v in report['scenes'][scene].items() if k!='pairs'},flush=True)
 save(H/'held_global132_report.json',report)
if __name__=='__main__':main()
