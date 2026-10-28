"""Frame/timestamp/body binding only; GBA measurements come from actual native optimization."""
from common132 import *
import csv,copy
sys.path.insert(0,str(R/'pipeline_20261001/fusion'))
from long_inputs import AuthorMotion,scan_motion,choose_anchor
def actual_native_source(scene,rows):
 source=C/'v55_native_depth_consistency_20261005/scene_0001_20261005_035527_jy79xz57' if scene=='scene_0001' else Path(load(O/'scene2_map.json')['sequence'])
 report=load(source/'report.json')
 if report.get('functional_passed') is not True or report.get('failures'):raise ValueError('Actual complete map source not passed')
 with (source/'online_states.csv').open() as f:allstates=list(csv.DictReader(f))
 if len(allstates)!=report['frames_observed'] or [int(r['frame_index']) for r in allstates]!=list(range(len(allstates))):raise ValueError('Native complete frame history')
 states=[r for r in allstates if int(r['imu_init'])==1]
 stamps=np.array([int(rows[int(r['frame_index'])]['capture_stamp_ns']) for r in states],dtype=np.int64)
 poses=[]
 for r in states:
  T=np.eye(4);T[:3,3]=[float(r['imu_'+x]) for x in ['tx','ty','tz']];T[:3,:3]=Rotation.from_quat([float(r['imu_'+x]) for x in ['qx','qy','qz','qw']]).as_matrix();poses.append(T)
 source_map={int(r['frame_index']):T for r,T in zip(states,poses)}
 prep=Path(load(source/'run.json')['prepared'])
 with (prep/'mav0/imu0/data.csv').open() as f:imu=list(csv.reader(f))[1:]
 times=np.array([int(r[0]) for r in imu],dtype=np.int64);values=np.array([[float(v) for v in r[1:]] for r in imu])
 motion=AuthorMotion(times,values[:,:3],values[:,3:])
 def header_pose(stamp):
  k=choose_anchor(stamps,int(stamp));r=states[k]
  if int(stamp)-int(stamps[k])>150000000:raise ValueError('Actual native source/header gap')
  vel=np.array([float(r[x]) for x in ['vx','vy','vz']]);bg=np.array([float(r[x]) for x in ['bgx','bgy','bgz']]);ba=np.array([float(r[x]) for x in ['bax','bay','baz']])
  _,T=scan_motion(motion,int(stamps[k]),poses[k],vel,bg,ba,int(stamp),np.array([int(stamp),int(stamp)+1],dtype=np.int64))
  return T
 return source_map,header_pose,{str(source/n):sha(source/n) for n in ['online_states.csv','run.json','report.json','extrinsics.json']}

def gba_motion(scene,d):
 out=O/'gba_actual'/scene/'robust'
 rep=load(out/'report.json')
 if rep['exit_code']!=0 or sha(out/'poses.txt')!=rep['output_sha256']:raise ValueError('Genuine complete native GBA absent')
 A=np.loadtxt(out/'poses.txt');B=np.loadtxt(out/'poses.txt.before')
 if not np.array_equal(A[:,:2],B[:,:2]):raise ValueError('Native frame identities changed')
 prep=C/'v52_lidar_depth_long_20261004/input52_jqlf_xtr' if scene=='scene_0001' else O/'scene2_input'
 with (prep/'frames.csv').open() as f:rows={int(r['frame_index']):r for r in csv.DictReader(f)}
 ns=d['ns'];source_map,header_pose,source_hashes=actual_native_source(scene,rows)
 IC=np.asarray(load(O/'prepared_r3'/scene/'calibration.json')['T_imu_rect0'])
 mapped={};errors=[]
 for a,b in zip(A,B):
  fid=int(a[0])
  if fid not in rows:raise ValueError('GBA frame not in actual image manifest')
  t=int(rows[fid]['capture_stamp_ns'])
  if abs(int(round(a[1]*1e9))-t)>1000:raise ValueError('GBA timestamp not actual capture')
  if t<int(ns[0]) or t>int(ns[-1]):continue
  k=int(np.searchsorted(ns,t));ids=[q for q in [k-1,k] if 0<=q<len(ns)]
  node=min(ids,key=lambda q:abs(int(ns[q])-t))
  distance=abs(int(ns[node])-t)
  if distance>50000000:continue
  original=source_map[fid]
  before=b[2:].reshape(4,4)@np.linalg.inv(IC);after=a[2:].reshape(4,4)@np.linalg.inv(IC)
  delta=np.linalg.inv(original)@before
  error=dict(m=float(np.linalg.norm(delta[:3,3])),deg=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi))
  error.update(frame_index=fid,node=node,capture_ns=str(t),source_before=before.tolist(),source_vio=original.tolist());errors.append(error)
  offset=np.linalg.inv(header_pose(int(ns[node])))@original
  # Apply only actual native GBA AFTER*inverse(BEFORE) world correction
  # to the same online frame, preserving pre-existing online/map difference.
  correction=after@np.linalg.inv(before)
  refined_online=correction@original
  if np.max(np.abs((before@np.linalg.inv(before))@original-original))>1e-10:raise ValueError('Identity GBA correction changed online source')
  item=dict(node=node,frame_index=fid,capture_ns=str(t),header_ns=str(int(ns[node])),distance_ns=distance,pose=refined_online@np.linalg.inv(offset),local_gba_update=np.linalg.inv(header_pose(int(ns[node])))@refined_online@np.linalg.inv(offset),binding_error=error,actual_gba_world_correction=correction.tolist())
  if node not in mapped or distance<mapped[node]['distance_ns']:mapped[node]=item
 save(H/'gba_inputs'/(scene+'_active_factor_binding_raw_errors.json'),errors)
 if len(mapped)<200:raise ValueError('Insufficient actual GBA frame/header binding')
 # Actual source equivalence is checked; do not substitute an unrelated replay.
 if np.median([r['m'] for r in errors])>.10 or np.percentile([r['m'] for r in errors],95)>.5 or np.percentile([r['deg'] for r in errors],95)>5.:raise ValueError('GBA map/source pose identity disagreement')
 updated=copy.deepcopy(d);replaced=[];rejected=[]
 for k,e in enumerate(updated['edges']):
  i,j=e['i'],e['j']
  if e.get('kind')!='vio' or i not in mapped or j not in mapped:continue
  # Highest-scored source deliberately preserves selected historical VIO components.
  # Carry actual GBA node-local increments onto THAT active measurement, not a
  # different complete VIO estimate: E_new = C_i^-1 E_active C_j.
  Ci=mapped[i]['local_gba_update'];Cj=mapped[j]['local_gba_update']
  T=np.linalg.inv(Ci)@np.asarray(e['T'])@Cj
  delta=np.linalg.inv(np.asarray(e['T']))@T
  translation=float(np.linalg.norm(delta[:3,3]));degrees=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi)
  record=dict(edge=k,i=i,j=j,before=e['T'],after=T.tolist(),change_m=translation,change_deg=degrees,source_frames=[mapped[i]['frame_index'],mapped[j]['frame_index']])
  if translation>.2 or degrees>3.:rejected.append(record);continue
  e['T']=T.tolist();replaced.append(record)
 # Do not add coarse pose constraints derived from the SAME VIO/GBA estimates.
 # Those would duplicate correlated visual evidence. Qualified active VIO
 # measurements already carry every available actual native GBA increment.
 coarse=[]
 if not replaced:raise ValueError('Actual GBA did not update existing qualified visual motion')
 report=dict(mapped_headers=len(mapped),replaced_visual_motion=len(replaced),rejected_local_motion=len(rejected),additional_coarse_native_gba=len(coarse),replacement_records=replaced,rejected_records=rejected,coarse_factors=coarse,native_source_sha256=sha(out/'poses.txt'),actual_estimator_source_hashes=source_hashes,calibration_sha256=sha(O/'prepared_r3'/scene/'calibration.json'),binding_errors=dict(median_m=float(np.median([r['m'] for r in errors])),p95_m=float(np.percentile([r['m'] for r in errors],95)),p95_deg=float(np.percentile([r['deg'] for r in errors],95))),actual_gba_application='Actual native GBA AFTER*inverse(BEFORE) -> same online frame/header node-local increment -> C_i^-1 E_active C_j. Retain historical accepted measurement components; no duplicated coarse VIO priors. Identity update validated.',uncertainty='Prospective heuristic coarse sigma 0.10m/0.03rad; correlated with original VIO, not independent calibrated covariance.',integer_capture_timestamps_from_original_manifest=True)
 return updated,coarse,report

def appearance_pose_factors(scene):
 images={r['node']:r for r in load(O/'prepared_r3'/scene/'images.json')}
 result=[]
 for r in load(H/'verification_full'/scene/'pairs.json'):
  if not r['accepted'] or r['kind']!='loop':continue
  i,j=r['i'],r['j'];Oi=np.asarray(images[i]['offset']);Oj=np.asarray(images[j]['offset'])
  # Native PnP Xi -> Xj gives Cj<-Ci; g2o requires Bi<-Bj.
  T=Oi@np.linalg.inv(np.asarray(r['forward']))@np.linalg.inv(Oj)
  result.append(dict(i=i,j=j,T=T.tolist(),information=[25.]*3+[1/.05**2]*3,huber_delta=3.,kind='original_BoQ_PnP_appearance_loop',held=r['held'],source='original BoQ appearance + native pretrained ALIKED/LightGlue + native two-way PnP/LM'))
 return result
