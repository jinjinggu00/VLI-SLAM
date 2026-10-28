"""Original OpenCV PnP pose verification; only bindings and admission checks."""
from common132 import *
import csv,cv2,time
cv2.setNumThreads(1)
# Fixed prospectively before examining native real-data PnP results.
CONFIG=dict(min_correspondences=24,min_training_inliers=14,min_inlier_ratio=.60,reprojection_px=3.,held_median_px=3.,held_p90_px=6.,reciprocal_m=.30,reciprocal_deg=3.,depth_median_m=.35,depth_p90_m=.80,max_proposal_translation_m=1.5,max_proposal_rotation_deg=12.,min_occupied_cells=4,ransac_iterations=2000,confidence=.999)
def pnp(X,uv,K,seed):
 cv2.setRNGSeed(seed)
 ok,r,t,idx=cv2.solvePnPRansac(np.ascontiguousarray(X,dtype=np.float64),np.ascontiguousarray(uv,dtype=np.float64),K,np.zeros(4),iterationsCount=CONFIG['ransac_iterations'],reprojectionError=CONFIG['reprojection_px'],confidence=CONFIG['confidence'],flags=cv2.SOLVEPNP_EPNP)
 if not ok or idx is None:return None,np.array([],dtype=int)
 ids=idx.reshape(-1)
 if len(ids)>=6:r,t=cv2.solvePnPRefineLM(np.ascontiguousarray(X[ids]),np.ascontiguousarray(uv[ids]),K,np.zeros(4),r,t)
 T=np.eye(4);T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.reshape(3)
 if not np.isfinite(T).all():return None,np.array([],dtype=int)
 return T,ids
def reproj(X,uv,K,T):
 Y=X@T[:3,:3].T+T[:3,3]
 if (Y[:,2]<=0).any():return None
 q=cv2.projectPoints(X,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,np.zeros(4))[0].reshape(-1,2)
 return np.linalg.norm(q-uv,axis=1)
def difference(A,B):
 T=np.linalg.inv(A)@B
 return dict(translation_m=float(np.linalg.norm(T[:3,3])),rotation_deg=float(np.linalg.norm(Rotation.from_matrix(T[:3,:3]).as_rotvec())*180/np.pi))
def cells(uv):
 a=np.floor(uv/np.array([752/4,480/3])).astype(int)
 return len({(int(x),int(y)) for x,y in a if 0<=x<4 and 0<=y<3})
def original_fixture():
 # Nonidentity transform + independently withheld correspondences.
 rng=np.random.default_rng(130);X=rng.uniform([-1.,-.8,4.],[1.,.8,8.],size=(60,3));T=np.eye(4);T[:3,:3]=Rotation.from_euler('xyz',[.02,.07,.04]).as_matrix();T[:3,3]=[.3,-.1,.02]
 K=np.array([[450.,0,375.5],[0,450.,239.5],[0,0,1.]])
 uv=cv2.projectPoints(X,cv2.Rodrigues(T[:3,:3])[0],T[:3,3],K,np.zeros(4))[0].reshape(-1,2)
 train=np.arange(60)%4!=0;out,ids=pnp(X[train],uv[train],K,130);err=reproj(X[~train],uv[~train],K,out)
 if out is None or len(ids)!=45 or err is None or np.max(err)>1e-5 or difference(T,out)['translation_m']>1e-6:raise ValueError('Original PnP/transform-direction fixture failed')
 return dict(original_opencv_fixture_passed=True,held_max_px=float(np.max(err)),known_transform_error=difference(T,out))
def main():
 dest=H/'verification_full';dest.mkdir(exist_ok=False);save(dest/'config.json',dict(**CONFIG,appearance_loops_use_trajectory_admission=False,reason='Independent BoQ retrieval must allow true drift correction; all reciprocal PnP, held pixels/depth and spatial-support gates remain unchanged.'));save(dest/'fixture.json',original_fixture())
 verify(load(H/'association_full/output_hashes.json'));result={'scenes':{},'config':CONFIG,'official_ate':None}
 for scene in SCENES:
  out=dest/scene;out.mkdir();prep=O/'prepared_r3'/scene;images={r['node']:r for r in load(prep/'images.json')};cal=load(prep/'calibration.json');fx,fy,cx,cy=cal['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  records=[]
  for i,j in load(prep/'pairs.json'):
   a=np.load(H/'association_full'/scene/(str(i)+'.npz'));b=np.load(H/'association_full'/scene/(str(j)+'.npz'))
   with (H/'features_full'/scene/('match_%d_%d.csv'%(i,j))).open() as f:matches=list(csv.DictReader(f))
   index=np.array([[int(v['i']),int(v['j'])] for v in matches],dtype=int).reshape(-1,2)
   valid=a['valid'][index[:,0]]&b['valid'][index[:,1]]
   index=index[valid];rec=dict(i=i,j=j,kind='loop' if j-i>=200 else 'temporal',raw_matches=len(matches),both_lidar_depth=len(index),accepted=False,failures=[])
   bad=rec['failures']
   if len(index)<CONFIG['min_correspondences']:bad.append('too_few_true_depth_correspondences');records.append(rec);continue
   # Feature identities fixed before measurement; feature ids divisible by 5 in either view held globally; never fit or optimize.
   order=np.lexsort((index[:,1],index[:,0]));index=index[order];held=(index[:,0]%5==0)|(index[:,1]%5==0);train=~held
   if held.sum()<5 or train.sum()<CONFIG['min_training_inliers']+3:bad.append('independent_support');records.append(rec);continue
   Xi=a['xyz'][index[:,0]];Xj=b['xyz'][index[:,1]];ui=a['uv'][index[:,0]];uj=b['uv'][index[:,1]]
   if cells(ui[train])<CONFIG['min_occupied_cells'] or cells(uj[train])<CONFIG['min_occupied_cells']:bad.append('spatial_support')
   T,fi=pnp(Xi[train],uj[train],K,130+i+j);U,ri=pnp(Xj[train],ui[train],K,260+i+j)
   if T is None or U is None:bad.append('native_pnp_failed');records.append(rec);continue
   if len(fi)<CONFIG['min_training_inliers'] or len(ri)<CONFIG['min_training_inliers'] or min(len(fi),len(ri))/train.sum()<CONFIG['min_inlier_ratio']:bad.append('pnp_support')
   e=reproj(Xi[held],uj[held],K,T);r=reproj(Xj[held],ui[held],K,U)
   if e is None or r is None:bad.append('held_cheirality');records.append(rec);continue
   reciprocal=difference(np.eye(4),T@U);proposal=np.linalg.inv(np.array(images[j]['pose']))@np.array(images[i]['pose']);delta=difference(proposal,T)
   depth_error=np.linalg.norm((Xi[held]@T[:3,:3].T+T[:3,3])-Xj[held],axis=1)
   if reciprocal['translation_m']>CONFIG['reciprocal_m'] or reciprocal['rotation_deg']>CONFIG['reciprocal_deg']:bad.append('reciprocal_disagreement')
   if max(float(np.median(e)),float(np.median(r)))>CONFIG['held_median_px'] or max(float(np.percentile(e,90)),float(np.percentile(r,90)))>CONFIG['held_p90_px']:bad.append('held_reprojection')
   if np.median(depth_error)>CONFIG['depth_median_m'] or np.percentile(depth_error,90)>CONFIG['depth_p90_m']:bad.append('held_depth_3d_disagreement')
   if rec['kind']=='temporal' and (delta['translation_m']>CONFIG['max_proposal_translation_m'] or delta['rotation_deg']>CONFIG['max_proposal_rotation_deg']):bad.append('temporal_proposal_disagreement')
   train_ids=np.flatnonzero(train);common=sorted(set(map(int,fi))&set(map(int,ri)));accepted_ids=[int(train_ids[k]) for k in common]
   if len(accepted_ids)<CONFIG['min_training_inliers']:bad.append('reciprocal_inlier_intersection')
   rec.update(accepted=not bad,forward=T.tolist(),reverse=U.tolist(),reciprocal=reciprocal,proposal_difference=delta,held=dict(n=int(held.sum()),forward_median_px=float(np.median(e)),reverse_median_px=float(np.median(r)),p90_px=float(max(np.percentile(e,90),np.percentile(r,90))),depth_median_m=float(np.median(depth_error)),depth_p90_m=float(np.percentile(depth_error,90))),train_inliers=dict(forward=len(fi),reverse=len(ri)),index=index.tolist(),optimization_correspondence_rows=accepted_ids,held_correspondence_rows=list(map(int,np.flatnonzero(held))),shared_observations=len(accepted_ids) if not bad else 0)
   records.append(rec)
   if len(records)%25==0:print('LOOP VERIFY',scene,len(records),'accepted',sum(r['accepted'] for r in records),flush=True)
  save(out/'pairs.json',records)
  result['scenes'][scene]=dict(proposed=len(records),accepted=sum(r['accepted'] for r in records),accepted_loops=sum(r['accepted'] and r['kind']=='loop' for r in records),accepted_temporal=sum(r['accepted'] and r['kind']=='temporal' for r in records),actual_shared_correspondences=sum(r.get('shared_observations',0) for r in records),rejected_reasons={b:sum(b in r['failures'] for r in records) for b in sorted({b for r in records for b in r['failures']})})
  print('LOOP VERIFICATION COMPLETE',scene,result['scenes'][scene],flush=True)
 save(dest/'report.json',result);save(dest/'output_hashes.json',{str(p):sha(p) for p in dest.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
if __name__=='__main__':main()
