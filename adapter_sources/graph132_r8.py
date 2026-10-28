"""IO-only preserved best graph + original depth landmark observations."""
from common132 import *
import argparse,subprocess,time,resource,math,copy
from gba_binding132 import gba_motion,appearance_pose_factors
from verify132 import CONFIG,pnp,difference,cells
from unique_support132 import support,depth_consistent
from subset_held132 import check as selected_subset_held
def preserved(d):
 groups=native.groups(d,0,len(d['ns']));gravity=[];sources={}
 if d['scene']=='scene_0002':
  base=C/'v127_author_pointlio_robust_aux_20261008/scene_0002_graph_wx1nhjjr/short02_Cauchy_0p5'
  added=load(base/'added_factors.json');gravity=load(base/'gravity_factors.json')
  groups += [(e,2,len(d['edges'])+k) for k,e in enumerate(added)]
  for p in [base/'added_factors.json',base/'gravity_factors.json']:sources[str(p)]=sha(p)
 return groups,gravity,sources
def graph_lines(d,groups,gravity):
 lines=[f"{len(d['ns'])} {len(groups)} {len(gravity)}"]
 lines += [' '.join(format(float(x),'.17g') for x in vals(T)) for T in d['map']]
 lines += ['%d %d %d '%(e['i'],e['j'],kind)+' '.join(format(float(x),'.17g') for x in [*vals(np.asarray(e['T'])),*e['information'],e['huber_delta']]) for e,kind,_ in groups]
 lines += ['%d '%g['i']+' '.join(format(float(x),'.17g') for x in [*g['direction'],*g['body'],g['information']]) for g in gravity]
 return '\n'.join(lines)+'\n'
def landmarks(scene,d,B):
 images={r['node']:r for r in load(O/'prepared_r3'/scene/'images.json')};K=load(O/'prepared_r3'/scene/'calibration.json')['intrinsics']
 metric={ (r['i'],r['j']):r for r in load(H/'measured_registration_r5'/scene/'pairs.json') if r['accepted']}
 verify(load(H/'measured_registration_r5/output_hashes.json'))
 verify(load(H/'measured_registration_r5/pre_execution_hashes.json'))
 accepted=[r for r in load(H/'verification_full'/scene/'pairs.json') if r['accepted'] and (r['i'],r['j']) in metric]
 if not accepted:raise ValueError('No real verified feature registration: '+scene)
 cameras=[];points=[];obs=[];node_camera={};used=set();pair_receipts=[];subset_rejections=[]
 # Each image feature contributes once; no duplicate evidence or held identity.
 for r in sorted(accepted,key=lambda r:(-r['shared_observations'],r['i'],r['j'])):
  i,j=r['i'],r['j'];allowed={tuple(map(int,x)) for x in metric[(i,j)]['inlier_feature_pairs']};a=np.load(H/'association_full'/scene/(str(i)+'.npz'));b=np.load(H/'association_full'/scene/(str(j)+'.npz'));idx=np.array(r['index'],dtype=int)
  chosen=[]
  for row in r['optimization_correspondence_rows']:
   ia,ib=map(int,idx[row])
   if ia%5==0 or ib%5==0 or row in r['held_correspondence_rows']:raise ValueError('Held feature leaked into graph')
   if (ia,ib) not in allowed or (i,ia) in used or (j,ib) in used:continue
   if not a['valid'][ia] or not b['valid'][ib]:raise ValueError('False measured depth')
   chosen.append((ia,ib))
  ids=np.array(chosen,dtype=int).reshape(-1,2)
  if not support(a['uv'][ids[:,0]],b['uv'][ids[:,1]],CONFIG):
   subset_rejections.append(dict(i=i,j=j,reason='deduplicated_original_count_or_cells',remaining=len(chosen)));continue
  Xi=a['xyz'][ids[:,0]];Xj=b['xyz'][ids[:,1]];ui=a['uv'][ids[:,0]];uj=b['uv'][ids[:,1]]
  F,fi=pnp(Xi,uj,np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1.]]),130+i+j)
  U,ri=pnp(Xj,ui,np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1.]]),260+i+j)
  if F is None or U is None:
   subset_rejections.append(dict(i=i,j=j,reason='deduplicated_original_pnp',remaining=len(chosen)));continue
  common=sorted(set(map(int,fi))&set(map(int,ri)))
  reciprocal=difference(np.eye(4),F@U)
  de=np.linalg.norm(Xi@F[:3,:3].T+F[:3,3]-Xj,axis=1)
  if min(len(fi),len(ri))/len(chosen)<CONFIG['min_inlier_ratio'] or reciprocal['translation_m']>CONFIG['reciprocal_m'] or reciprocal['rotation_deg']>CONFIG['reciprocal_deg'] or not depth_consistent(de,CONFIG) or not support(ui[common],uj[common],CONFIG):
   subset_rejections.append(dict(i=i,j=j,reason='deduplicated_training_geometry',remaining=len(chosen),common=len(common),depth_median_m=float(np.median(de)),depth_p90_m=float(np.percentile(de,90)),reciprocal=reciprocal));continue
  # The qualified transform must be fitted to the final unique training representation.
  F,fi=pnp(Xi[common],uj[common],np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1.]]),130+i+j)
  U,ri=pnp(Xj[common],ui[common],np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1.]]),260+i+j)
  if F is None or U is None or min(len(fi),len(ri))<CONFIG['min_training_inliers'] or min(len(fi),len(ri))/len(common)<CONFIG['min_inlier_ratio']:
   subset_rejections.append(dict(i=i,j=j,reason='actual_final_subset_pnp_support'));continue
  reciprocal=difference(np.eye(4),F@U)
  if reciprocal['translation_m']>CONFIG['reciprocal_m'] or reciprocal['rotation_deg']>CONFIG['reciprocal_deg']:
   subset_rejections.append(dict(i=i,j=j,reason='actual_final_subset_reciprocal'));continue
  hi=idx[r['held_correspondence_rows']]
  validation=selected_subset_held(a['xyz'][hi[:,0]],b['xyz'][hi[:,1]],a['uv'][hi[:,0]],b['uv'][hi[:,1]],np.array([[K[0],0,K[2]],[0,K[1],K[3]],[0,0,1.]]),F,U,CONFIG)
  if not validation['passed']:
   subset_rejections.append(dict(i=i,j=j,reason='actual_final_subset_frozen_held_support',validation=validation));continue
  chosen=[chosen[k] for k in common]
  for node in (i,j):
   if node not in node_camera:
    cid=len(cameras);node_camera[node]=cid;cameras.append(dict(id=cid,node=node,offset=images[node]['offset'],intrinsics=K,capture_ns=images[node]['capture_ns'],header_ns=images[node]['header_ns']))
  for ia,ib in chosen:
   A=B[i]@np.asarray(images[i]['offset']);Z=B[j]@np.asarray(images[j]['offset'])
   X=(A[:3,:3]@a['xyz'][ia]+A[:3,3]+Z[:3,:3]@b['xyz'][ib]+Z[:3,3])/2.
   pid=len(points);points.append(dict(id=pid,world=X.tolist(),support=[[i,ia],[j,ib]],pair=[i,j]))
   for node,feature,f in [(i,ia,a),(j,ib,b)]:
    depth=float(f['depth'][feature]);z=[*map(float,f['uv'][feature]),depth]
    # Fixed heuristic uncertainty, not empirically calibrated sensor covariance.
    sigma_z=.10+.015*depth
    obs.append(dict(camera=node_camera[node],point=pid,measurement=z,information=[0.125,0.125,0.5/(sigma_z*sigma_z)],delta=2.45,feature_id=feature,depth_source='native LVI-SAM surface supported optical Z'))
    used.add((node,feature))
  pair_receipts.append(dict(i=i,j=j,kind=r['kind'],landmarks=len(chosen),forward_pnp=r['forward'],reverse_pnp=r['reverse'],held=r['held'],measured_rigid_training_registration=metric[(i,j)],actual_subset_held_validation=validation))
 if len(points)<20 or not pair_receipts:raise ValueError('Too little unique real feature fusion')
 return dict(cameras=cameras,points=points,observations=obs,loop_pairs=pair_receipts,measurement_sigmas='Original engineering2px/(0.10+0.015*z m) covariance scaled by1/0.5, conservative binding uncertainty; not calibrated sensor covariance',duplicate_feature_evidence=False,held_feature_ids_globally_excluded=True,pnp_pose_is_validation_only=True,feature_extractor='Original pretrained ALIKED-n16 + original ALIKED LightGlue',PLNet_core_unchanged=True,PLNet_landmarks_added=False,original_measured_rigid_refiner='Free-FreeReg kabsch_refine(False); measured optical3D; metric scale fixed1; training data only',deduplicated_subset_rejections=subset_rejections,deduplicated_subset_policy='Reapply original count, cells, bidirectional OpenCV PnP and measured3D thresholds using training rows only. Held identity rows are never used to choose residual factors.')

def crossmodal_landmarks(scene,B,extra):
 path=H/'crossmodal_receipt.json'
 if not path.is_file():raise ValueError('Direct upstream matching attempt not recorded')
 receipt=load(path)
 if not receipt.get('finished'):raise ValueError('Direct matching probe unfinished')
 # Unsupported direct registration is strictly absent; actual ALIKED/LiDAR landmarks still fuse both measured modalities.
 if receipt.get('accepted_pairs',0)!=0:raise ValueError('Positive direct registration requires its independent binding and held audit; do not silently use it')
 extra['crossmodal_pairs']=[];extra['direct_crossmodal_landmarks']=0
 extra['crossmodal_target_measurement_semantics']='No direct crossmodal model factor admitted; actual camera feature + measured LiDAR depth joint landmarks are distinct and present.'
 return extra

def observation_text(extra):
 cameras=extra.get('cameras',[]);points=extra.get('points',[]);observations=extra.get('observations',[])
 rows=['%d %d %d'%(len(cameras),len(points),len(observations))]
 for c in cameras:rows.append('%d %d '%(c['id'],c['node'])+' '.join(format(float(x),'.17g') for x in [*vals(np.asarray(c['offset'])),*c['intrinsics']]))
 for p in points:rows.append('%d '%p['id']+' '.join(format(float(x),'.17g') for x in p['world']))
 for z in observations:rows.append('%d %d '%(z['camera'],z['point'])+' '.join(format(float(x),'.17g') for x in [*z['measurement'],*z['information'],z['delta']]))
 return '\n'.join(rows)+'\n'
def audit(stats,groups,gravity,extra):
 old=stats['factor_audit'];index={}
 for r in old:index.setdefault(r['original_edge_index'],[]).append(r)
 if set(index)!=set(range(len(groups))) or stats['input_factors']!=len(groups):raise ValueError('Lost original pose factor')
 for k,(e,kind,_) in enumerate(groups):
  expected=['translation','rotation'] if kind==0 else ['shared']
  a=index[k]
  if sorted(r['component'] for r in a)!=sorted(expected):raise ValueError('Changed original factor components')
  for r in a:
   info=list(e['information']);kernel='Cauchy' if kind==2 else 'Huber';delta=.5 if kind==2 else e['huber_delta']
   if r['component']=='translation':info=info[:3]+[0.]*3
   elif r['component']=='rotation':info=[0.]*3+info[3:];kernel='Cauchy';delta=.5
   if r['kind']!=kind or r['i']!=e['i'] or r['j']!=e['j'] or r['kernel']!=kernel or r['delta']!=delta or r['information']!=info:raise ValueError('Changed original factor identity/kernel')
   if np.max(np.abs(np.array(r['measurement']).reshape(4,4)-np.asarray(e['T'])))>1e-12:raise ValueError('Changed original measurement')
 ga=stats['gravity_audit']
 if len(ga)!=len(gravity):raise ValueError('Changed original gravity count')
 for g,r in zip(gravity,ga):
  expected=np.concatenate([np.asarray(g['direction'])/np.linalg.norm(g['direction']),np.asarray(g['body'])/np.linalg.norm(g['body'])])
  if g['i']!=r['i'] or g['information']!=r['information'] or np.max(np.abs(expected-r['measurement']))>1e-12:raise ValueError('Changed original gravity factor')
 z=stats['feature_depth_audit']
 if len(z)!=len(extra.get('observations',[])) or stats['shared_landmarks']!=len(extra.get('points',[])) or stats['native_factors']!=len(old)+len(ga)+len(z):raise ValueError('New actual factor count')
 for actual,expected in zip(z,extra.get('observations',[])):
  if actual['camera']!=expected['camera'] or actual['landmark']!=expected['point'] or actual['measurement']!=expected['measurement'] or np.max(np.abs(np.array(actual['information'])-expected['information']))>1e-12 or actual['weight']<0:raise ValueError('New observation identity mismatch')
 total=math.fsum(x['native_cost'] for x in old)+math.fsum(x['chi2'] for x in ga)+math.fsum(x['cost'] for x in z)
 if abs(total-stats['final_cost'])>1e-7*max(1,total):raise ValueError('Actual objective audit failed')
def solve(d,groups,gravity,extra,out):
 out.mkdir(exist_ok=False);graph=out/'graph.txt';data=out/'observations.txt';graph.write_text(graph_lines(d,groups,gravity));data.write_text(observation_text(extra));save(out/'feature_bindings.json',extra)
 binary=O/'build/joint_r5/joint_graph';hashes={str(p):sha(p) for p in [graph,data,binary,H/'joint_bridge.cpp',Path(__file__),H/'gba_binding132.py',H/'common132.py']}
 save(out/'pre_execution_hashes.json',hashes);start=time.monotonic()
 cmd=[str(binary),str(graph),str(out/'poses.txt'),str(out/'optimizer.json'),'rotation_cauchy05','Cauchy','.5',str(data)]
 with (out/'native.log').open('x') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
 save(out/'execution.json',dict(command=cmd,exit_code=p.returncode,seconds=time.monotonic()-start,peak_child_rss_mib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024))
 verify(hashes)
 if p.returncode:raise ValueError('Original joint graph failed; numerical retry forbidden')
 a=np.loadtxt(out/'poses.txt')
 if a.shape!=(len(d['ns']),7) or not np.isfinite(a).all() or np.max(np.abs(np.linalg.norm(a[:,3:],axis=1)-1.))>1e-7:raise ValueError('Native complete output failed')
 T=np.repeat(np.eye(4)[None],len(a),axis=0);T[:,:3,3]=a[:,:3];T[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix()
 if np.max(np.abs(T[0]-d['map'][0]))>1e-10:raise ValueError('Fixed first body pose moved')
 stats=load(out/'optimizer.json');audit(stats,groups,gravity,extra);np.save(out/'poses.npy',T);m.b.write_trajectory(out/'trajectory.csv',d['ns'],T)
 return T,stats
def main(control=False):
 phase='control_r8' if control else 'joint_r8';root=H/phase;root.mkdir(exist_ok=False);report=dict(phase=phase,scenes={},official_score=None)
 pre={}
 for module in tuple(sys.modules.values()):
  path=getattr(module,'__file__',None)
  if path:
   p=Path(path).resolve()
   if str(p).startswith(str(R.resolve())+'/') and p.suffix=='.py' and p.is_file():pre[str(p)]=sha(p)
 for p in [Path(__file__),H/'gba_binding132.py',H/'joint_bridge.cpp',O/'build/joint_r5/joint_graph']:
  pre[str(p)]=sha(p)
 source=Path('/workspace/AirSLAM/catkin_ws/src/AirSLAM/src/g2o_optimization/g2o_optimization.cc');pre[str(source)]=sha(source)
 save(root/'pre_source_hashes.json',pre)

 for scene in SCENES:
  prep=O/'prepared_r3'/scene;d=m.b.restore(prep/'data.json');B=np.load(prep/'best.npy');groups,gravity,gh=preserved(d);verify(gh);folder=root/scene;folder.mkdir()
  if control:
   extra={};initial=d
  else:
   revised,coarse_gba,gba_report=gba_motion(scene,d);save(folder/'gba_bindings.json',gba_report)
   groups,gravity,gh=preserved(revised);verify(gh)
   groups += [(e,0,len(groups)+k) for k,e in enumerate(coarse_gba)]
   loops=appearance_pose_factors(scene);save(folder/'appearance_initializer_factors.json',loops)
   initializer_groups=groups+[(e,1,len(groups)+k) for k,e in enumerate(loops)]
   seed=copy.deepcopy(revised);seed['map']=B
   Q,_=solve(seed,initializer_groups,gravity,{},folder/'appearance_initializer')
   U0,_=solve(seed,initializer_groups,gravity,{},folder/'appearance_initializer_repeat')
   if np.max(np.abs(Q-U0))>1e-7:raise ValueError('Original coarse loop/GBA optimizer repeat differed')
   # PnP poses initialize only; final graph uses actual raw held-separated image/point observations.
   initial=copy.deepcopy(revised);initial['map']=Q
   extra=crossmodal_landmarks(scene,Q,landmarks(scene,d,Q))
  T,stats=solve(initial,groups,gravity,extra,folder/'native')
  if control:
   error=float(np.max(np.abs(T-B)));r=dict(max_abs_transform_error=error,original_pose_factors=len(groups),original_gravity=len(gravity),native_cost=stats['final_cost'],passed=error<=1e-7)
   if not r['passed']:raise ValueError('Zero-added native graph does not reproduce protected V127')
  else:
   # Independent rerun is serial, complete and uses identical frozen input.
   U,_=solve(initial,groups,gravity,extra,folder/'repeat');error=float(np.max(np.abs(U-T)))
   if error>1e-7:raise ValueError('Complete native graph repeat differed')
   ins,w,a=m.b.raw(scene);checks=[]
   for lo,hi in [(0,len(d['ns']))]+list(m.WINDOWS[scene]):
    gin=m.b.gravity_inputs(d['ns'][lo:hi],B[lo:hi],ins,w,a);q=m.b.compare(d,T[lo:hi],B[lo:hi],B[lo:hi],lo,hi,gin);q.update(start=lo,stop=hi);checks.append(q)
   fixed=m.b.fixed_pairs(d,T,folder/'native');failures=sorted(set(fixed['failures']+[x for q in checks for x in q['failures']]))
   r=dict(complete_vertices=len(T),input_pose_factors=len(groups),preserved_backbone_gravity_factors=len(gravity),actual_native_gba_replaced_vio_factors=gba_report['replaced_visual_motion'],actual_native_gba_coarse_factors=len(coarse_gba),appearance_initializer_loops=len(loops),direct_crossmodal_pairs=len(extra['crossmodal_pairs']),direct_crossmodal_landmarks=extra['direct_crossmodal_landmarks'],new_shared_landmarks=len(extra['points']),new_pixel_lidar_depth_factors=len(extra['observations']),verified_loop_pairs=sum(r['kind']=='loop' for r in extra['loop_pairs']),verified_temporal_pairs=sum(r['kind']=='temporal' for r in extra['loop_pairs']),checks=checks,fixed_pairs=fixed,failures=failures,passed=not failures,independent_repeat_transform_error=error,change_from_best_m=dict(median=float(np.median(np.linalg.norm(T[:,:3,3]-B[:,:3,3],axis=1))),maximum=float(np.max(np.linalg.norm(T[:,:3,3]-B[:,:3,3],axis=1)))),ate_improvement_verified=False)
  save(folder/'report.json',r);report['scenes'][scene]=r;save(root/'report.json',report);print('JOINT GRAPH',phase,scene, {k:v for k,v in r.items() if k not in ['checks','fixed_pairs']},flush=True)
 verify(pre);verify(load(O/'prepared_r3/receipt.json')['source_hashes']);save(root/'output_hashes.json',{str(p):sha(p) for p in root.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
 if any(not r['passed'] for r in report['scenes'].values()):raise ValueError('Complete candidate fails unchanged physical/held checks; not released')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--control',action='store_true');args=p.parse_args();main(args.control)
