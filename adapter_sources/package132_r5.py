"""Real public export only after actual upstream measurements and full graph verification."""
from common132 import *
import graph132_r5 as graph
from verify132 import CONFIG
from held_gate132 import failures as held_failures
import tempfile,shutil,datetime,math
EXPECTED=['README.md','trajectories/scene_0001.txt','trajectories/scene_0002.txt']
def read_complete(raw):
 rows=[r.split() for r in raw.decode('ascii').splitlines() if r.strip() and not r.startswith('#')]
 if not rows or any(len(r)!=8 for r in rows):raise ValueError('TUM eight-field contract')
 ns=np.array([int(Decimal(r[0])*10**9) for r in rows],dtype=np.int64)
 if any(Decimal(r[0])*10**9!=int(Decimal(r[0])*10**9) for r in rows) or np.any(np.diff(ns)<=0):raise ValueError('Exact ordered timestamp contract')
 a=np.array([[float(x) for x in row[1:]] for row in rows])
 if not np.isfinite(a).all() or np.max(np.abs(np.linalg.norm(a[:,3:],axis=1)-1))>1e-7:raise ValueError('Finite unit quaternions')
 T=np.repeat(np.eye(4)[None],len(a),axis=0);T[:,:3,3]=a[:,:3];T[:,:3,:3]=Rotation.from_quat(a[:,3:]).as_matrix();return ns,T
def qualifications():
 hashes={};state={}
 for phase in ['features_full','association_full','verification_full','measured_registration_r5','joint_r5']:
  q=H/phase/'output_hashes.json'
  if not q.is_file():raise ValueError('Unfinished actual phase '+phase)
  v=load(q);verify(v);hashes.update(v);hashes[str(q)]=sha(q)
 hashes.update(load(O/'prepared_r3/receipt.json')['source_hashes'])
 hashes.update(load(H/'association_full/receipt.json')['source_hashes'])
 hashes.update(load(H/'features_full/pre_source_hashes.json'));hashes.update(load(H/'joint132/pre_source_hashes.json'))
 repeat_depth=load(H/'association_repeat/report.json')
 for scene in SCENES:
  rows=load(H/'association_full'/scene/'receipt.json')
  chosen=[rows[k]['node'] for k in [0,len(rows)//2,len(rows)-1]]
  rec=repeat_depth['scenes'][scene]['samples']
  if [r['node'] for r in rec]!=chosen or any(not r['bit_exact'] or r['exit_code'] for r in rec):raise ValueError('Exact original depth sample repetition missing')
  for node in chosen:
   target=H/'association_repeat'/(scene+'_'+str(node)+'.f32')
   original=H/'association_full'/scene/(str(node)+'_depth.f32')
   if target.read_bytes()!=original.read_bytes():raise ValueError('Actual original depth repeat changed')
   hashes[str(target)]=sha(target)
 hashes[str(H/'association_repeat/report.json')]=sha(H/'association_repeat/report.json')
 fixture=load(H/'association_full/fixture.json')
 if fixture['exit_code'] or fixture['stdout'].strip()!='AUTHOR DEPTH CONTRACTS PASS: plane 8m; empty unknown; >3m cutoff; optical rays':raise ValueError('Original depth numerical fixture')
 if not load(H/'verification_full/fixture.json')['original_opencv_fixture_passed']:raise ValueError('Original PnP direction')
 model=load(H/'features_full/model_receipt.json')
 exact=load(H/'weight_binding132_receipt.json')
 if not exact['passed'] or not all(r['all_parameters_bit_exact'] for r in exact['models'].values()):raise ValueError('Author pretrained parameter value binding failed')
 hashes[str(H/'weight_binding132_receipt.json')]=sha(H/'weight_binding132_receipt.json')
 if not all(r['all_parameters_loaded'] for r in model.values()):raise ValueError('Unloaded actual neural parameters')
 boq=load(O/'appearance/report.json')
 if not boq['original_model_unchanged'] or boq['training'] or boq['spatial_gating']:raise ValueError('Changed original retrieval source')
 verify(boq['weights']);hashes.update(boq['weights'])
 control=load(O/'control_r4/report.json')
 held=load(H/'held_global132_r5_report.json');result=load(H/'joint132/report.json')
 if held.get('frozen_thresholds')!=CONFIG or not held.get('independence_limits'):raise ValueError('Final image and measured3D gates not explicitly frozen')
 for scene in SCENES:
  hr=held['scenes'][scene]
  if hr.get('whole_correspondence_independence_claimed') is not False or not hr.get('held_unique_feature_identities'):raise ValueError('Unsupported whole correspondence holdout claim')
  for pair in hr['pairs']:
   if pair.get('n',0)<5 or any(k not in pair for k in ['depth_median_m','depth_p90_m','after_median_px','after_reverse_median_px','after_p90_px']):raise ValueError('Incomplete actual final held pixel3D receipt')
   if pair['depth_median_m']>CONFIG['depth_median_m'] or pair['depth_p90_m']>CONFIG['depth_p90_m'] or max(pair['after_median_px'],pair['after_reverse_median_px'])>CONFIG['held_median_px'] or pair['after_p90_px']>CONFIG['held_p90_px'] or pair['failures']:raise ValueError('Actual final post-global held measurement failed')
 direct=load(H/'crossmodal_receipt.json')
 if not direct['finished'] or direct['accepted_pairs']!=0:raise ValueError('Unverified direct model branch cannot enter graph')
 for scene in SCENES:
  if not control['scenes'][scene]['passed'] or control['scenes'][scene]['max_abs_transform_error']>1e-7:raise ValueError('Original zero-added baseline not exact')
  r=result['scenes'][scene]
  if not r['passed'] or r['failures'] or r['independent_repeat_transform_error']>1e-7:raise ValueError('Full graph failed')
  if not held['scenes'][scene]['passed'] or held['scenes'][scene]['failures']:raise ValueError('Final genuine held features failed')
  for mode in ['robust','robust_repeat']:
   folder=O/'gba_actual'/scene/mode;g=load(folder/'report.json')
   if g['exit_code'] or sha(folder/'poses.txt')!=g['output_sha256']:raise ValueError('Native GBA changed')
   hashes.update(load(folder/'pre_execution_hashes.json'))
   for n in ['report.json','poses.txt','poses.txt.before']:hashes[str(folder/n)]=sha(folder/n)
  if load(O/'gba_actual'/scene/'robust_repeat/report.json')['repeat_max_abs']>1e-7:raise ValueError('Native GBA repeat')
  d=m.b.restore(O/'prepared_r3'/scene/'data.json');B=np.load(O/'prepared_r3'/scene/'best.npy')
  if np.max(np.abs(best(scene,d['ns'])-B))>1e-12:raise ValueError('Wrong best baseline')
  revised,coarse,binding=graph.gba_motion(scene,d)
  if binding!=load(H/'joint_r5'/scene/'gba_bindings.json'):raise ValueError('Genuine GBA/header/body evidence changed')
  groups,gravity,source=graph.preserved(revised);hashes.update(source)
  groups += [(e,0,len(groups)+k) for k,e in enumerate(coarse)]
  loops=graph.appearance_pose_factors(scene)
  if loops!=load(H/'joint_r5'/scene/'appearance_initializer_factors.json'):raise ValueError('Actual new appearance initializer changed')
  initializer_groups=groups+[(e,1,len(groups)+k) for k,e in enumerate(loops)]
  initializer_seed=dict(revised);initializer_seed['map']=B
  for kind in ['appearance_initializer','appearance_initializer_repeat']:
   folder=H/'joint_r5'/scene/kind
   if (folder/'graph.txt').read_bytes()!=graph.graph_lines(initializer_seed,initializer_groups,gravity).encode() or (folder/'observations.txt').read_bytes()!=graph.observation_text({}).encode():raise ValueError('Frozen actual initializer binding changed')
   if load(folder/'execution.json')['exit_code']:raise ValueError('Initializer original optimizer failed')
   graph.audit(load(folder/'optimizer.json'),initializer_groups,gravity,{})
   verify(load(folder/'pre_execution_hashes.json'));hashes.update(load(folder/'pre_execution_hashes.json'))
  Q=np.load(H/'joint_r5'/scene/'appearance_initializer/poses.npy')
  if np.max(np.abs(Q-np.load(H/'joint_r5'/scene/'appearance_initializer_repeat/poses.npy')))>1e-7:raise ValueError('Initializer repeat differs')
  seed=dict(revised);seed['map']=Q
  extra=graph.crossmodal_landmarks(scene,Q,graph.landmarks(scene,d,Q))
  if extra.get('PLNet_landmarks_added') is not False or extra['direct_crossmodal_landmarks']!=0:raise ValueError('Correlated or failed features admitted')
  for kind in ['native','repeat']:
   folder=H/'joint_r5'/scene/kind
   if (folder/'graph.txt').read_bytes()!=graph.graph_lines(seed,groups,gravity).encode() or (folder/'observations.txt').read_bytes()!=graph.observation_text(extra).encode():raise ValueError('Raw numeric binding changed')
   if load(folder/'feature_bindings.json')!=extra:raise ValueError('Actual original feature IDs changed')
   execution=load(folder/'execution.json')
   if execution['exit_code']:raise ValueError('Native actual solver failed')
   graph.audit(load(folder/'optimizer.json'),groups,gravity,extra)
   v=load(folder/'pre_execution_hashes.json');verify(v);hashes.update(v)
  T=np.load(H/'joint_r5'/scene/'native/poses.npy')
  if np.max(np.abs(T-np.load(H/'joint_r5'/scene/'repeat/poses.npy')))>1e-7:raise ValueError('Complete full repeat differs')
  native=np.loadtxt(H/'joint_r5'/scene/'native/poses.txt')
  if native.shape!=(len(T),7) or np.max(np.abs(native[:,:3]-T[:,:3,3]))>1e-12:raise ValueError('Native array/public pose provenance')
  if not np.isfinite(native).all() or np.max(np.abs(np.linalg.norm(native[:,3:],axis=1)-1.))>1e-7 or np.max(np.abs(Rotation.from_quat(native[:,3:]).as_matrix()-T[:,:3,:3]))>1e-12:raise ValueError('Native complete orientation/array provenance')
  state[scene]=dict(data=d,poses=T,report=r,extra=extra)
 for p in list(H.glob('*.py'))+list(H.glob('*.cpp'))+[H/'plan.md',H/'spec.md',H/'held_global132_r5_report.json',H/'crossmodal_receipt.json',H/'upstream_manifest.json',BEST,R/'official/config/lidar_visual_inertial.json']:
  if p.is_file():hashes[str(p)]=sha(p)
 verify(hashes);return state,hashes,result
def reproduction(out,state):
 dest=out/'reproduction';(dest/'src/v132').mkdir(parents=True)
 shutil.copyfile(H/'joint_bridge.cpp',dest/'src/v132/joint_bridge.cpp')
 up=dest/'src/v126_author_hdl_pointlio_gravity_20261008/upstream/include/g2o';up.mkdir(parents=True)
 original=C/'v126_author_hdl_pointlio_gravity_20261008/upstream/include/g2o/edge_se3_priorvec.hpp';shutil.copyfile(original,up/original.name)
 for scene,s in state.items():
  target=dest/'inputs'/scene;target.mkdir(parents=True)
  for n in ['graph.txt','observations.txt','feature_bindings.json','poses.txt','poses.txt.landmarks.txt']:
   shutil.copyfile(H/'joint_r5'/scene/'native'/n,target/n)
  (target/'headers.txt').write_text(''.join(str(int(t))+'\n' for t in s['data']['ns']))
  with zipfile.ZipFile(BEST) as z:prefix=b'' if scene=='scene_0001' else b''.join(z.read('trajectories/scene_0002.txt').splitlines(keepends=True)[:524])
  (target/'startup_prefix.txt').write_bytes(prefix)
  shutil.copyfile(out/'submission/trajectories'/(scene+'.txt'),target/'expected_public.txt')
  shutil.copyfile(H/'joint_r5'/scene/'report.json',target/'report.json')
 shutil.copyfile(H/'reproduce_joint132.py',dest/'reproduce_joint.py')
 (dest/'run_joint.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd -- "$(dirname -- "$0")"\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\npython3 -B reproduce_joint.py\n')
 adapter=dest/'adapter_sources';adapter.mkdir()
 for q in list(H.glob('*.py'))+list(H.glob('*.sh'))+[H/'spec.md',H/'plan.md',H/'方法与复现说明.md']:
  if q.is_file():shutil.copyfile(q,adapter/q.name)
 evidence=dest/'evidence';evidence.mkdir()
 for q in [H/'features_full/report.json',H/'features_full/model_receipt.json',H/'weight_binding132_receipt.json',H/'association_full/receipt.json',H/'association_full/fixture.json',H/'association_repeat/report.json',H/'crossmodal_fixture.json',H/'crossmodal_static_asset_hashes.json',H/'minkowski_header_compatibility.json',H/'runtime_compatibility132.json',H/'crossmodal_pnp132_report.json',H/'crossmodal_pnp_pre_hashes.json',H/'full_image_binding_audit.json',H/'appearance_best_conflict_audit.json',H/'measured_registration_r5/report.json',H/'measured_registration_r5/pre_execution_hashes.json',H/'final_review132.json',H/'verification_full/report.json',H/'held_global132_r5_report.json',H/'crossmodal_receipt.json',H/'upstream_manifest.json']:
  shutil.copyfile(q,evidence/(q.parent.name+'_'+q.name))
 # Include actual upstream feature numerical code; neural weights/raw sensors remain external, explicitly referenced.
 shutil.copytree(H/'measured_registration_r5',dest/'evidence/measured_registration_r5')
 for name in ['lightglue','ffreereg']:
  shutil.copytree(H/'upstream'/name,dest/'upstream'/name,ignore=shutil.ignore_patterns('.git','__pycache__','*.pyc','checkpoints','model','build','demo'))
 gba=adapter/'native_gba';gba.mkdir()
 for q in (O/'src').iterdir():
  if q.is_file():shutil.copyfile(q,gba/q.name)
 shutil.copyfile(Path('/workspace/AirSLAM/catkin_ws/src/AirSLAM/LICENSE.md'),dest/'AirSLAM_LICENSE.md')
 lic=dest/'licenses';lic.mkdir()
 for name,path in [('hdl_graph_slam',C/'v126_author_hdl_pointlio_gravity_20261008/upstream/LICENSE'),('LVI-SAM',C/'v49_lidar_depth_frontend_20261004/upstream/LICENSE'),('LightGlue',H/'upstream/lightglue/LICENSE'),('DepthPro',H/'upstream/ffreereg/tools/DepthPro/LICENSE')]:
  shutil.copyfile(path,lic/(name+'.txt'))
 (lic/'upstream_attribution.md').write_text('Numerical implementations are upstream AirSLAM/g2o, OpenCV, LVI-SAM DepthRegister, LightGlue ALIKED, BoQ and original protected V127 LiDAR backends. IO/launcher/observation binding is project integration code. Repository revisions and model hashes are in evidence/upstream_manifest.json. Free-FreeReg was tested independently; admission and failures are in its receipt.\n')
 libraries={str(q):sha(q) for q in [Path('/usr/local/lib/libg2o_types_slam3d.so'),Path('/usr/local/lib/libg2o_core.so'),Path('/usr/local/lib/libg2o_stuff.so')]}
 files={q.relative_to(dest).as_posix():sha(q) for q in dest.rglob('*') if q.is_file()}
 save(dest/'runtime.json',dict(image='sha256:9df819fdbb0ef0925adcd32f1f6d543e12272d4cc5c35118cb259597ebb508e7',libraries=libraries,input_hashes=files,standalone_scope='Recompile and repeat actual frozen two-scene original g2o joint graphs. Full raw extraction/DepthRegister/GBA additionally requires retained image/LiDAR/IMU/maps/weights; not embedded.'))
 archive=out/'V132_reproduction.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for q in sorted(dest.rglob('*')):
   if q.is_file():z.write(q,q.relative_to(dest).as_posix())
 with zipfile.ZipFile(archive) as z:
  if z.testzip():raise ValueError('Reproduction CRC failed')
 return archive
def package():
 state,hashes,result=qualifications()
 payloads={};bindings={};motion={}
 with zipfile.ZipFile(BEST) as z:
  if z.testzip() or z.namelist()!=EXPECTED:raise ValueError('Highest genuine ZIP contract')
  old={s:z.read('trajectories/'+s+'.txt') for s in SCENES}
 for scene,s in state.items():
  d=s['data'];T=s['poses'];raw=m.b.tum_bytes(d['ns'],T);start=0 if scene=='scene_0001' else 524
  prefix=b'' if not start else b''.join(old[scene].splitlines(keepends=True)[:start]);payload=prefix+raw
  ns,Q=read_complete(payload);oldns,_=read_complete(old[scene])
  if len(ns)!=dict(scene_0001=5044,scene_0002=4098)[scene] or not np.array_equal(ns,oldns) or not np.array_equal(ns[start:],d['ns']) or payload==old[scene]:raise ValueError('Full exact timestamp coverage or unchanged old payload')
  mm=m.b.motion(ns,Q)
  if mm['speed_mps']['max']>5 or mm['rotation_rate_degps']['max']>180:raise ValueError('Whole trajectory/startup seam motion bound')
  payloads[scene]=payload;motion[scene]=mm;bindings[scene]=dict(native_rows=len(T),prefix_rows=start,original_prefix_bit_exact=True,native_pose_sha256=sha(H/'joint_r5'/scene/'native/poses.txt'),public_payload_sha256=hashlib.sha256(payload).hexdigest())
 out=Path(tempfile.mkdtemp(prefix='v132exp_',dir=H/'candidates'));sub=out/'submission';(sub/'trajectories').mkdir(parents=True)
 for scene,b in payloads.items():(sub/'trajectories'/(scene+'.txt')).write_bytes(b)
 readme='# 云南大学 SLAM 团队 — V132 ALIKED / LightGlue / BoQ / native AirSLAM GBA / measured LiDAR joint graph\n\n'
 readme+='Unscored experimental candidate. Official highest remains V127 (64.211848498398). Both complete true IMU-origin trajectories5044/4098 timestamps, same genuine scene2 startup524-row prefix; no ground truth, artificial coverage or scale fitting.\n\n'
 readme+=(H/'方法与复现说明.md').read_text(encoding='utf-8')
 (sub/'README.md').write_text(readme,encoding='utf-8')
 counts=m.b.validate_public_submission(sub,R/'official/config/lidar_visual_inertial.json')
 if counts!={'scene_0001':5044,'scene_0002':4098}:raise ValueError('Official public format')
 archive=out/'YNU_SLAM_submission.zip'
 with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
  for name in EXPECTED:
   zi=zipfile.ZipInfo(name,(2026,10,9,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED;zi.external_attr=0o100644<<16;z.writestr(zi,(sub/name).read_bytes())
 with zipfile.ZipFile(archive) as z:
  if z.testzip() or z.namelist()!=EXPECTED or any(z.read(n)!=(sub/n).read_bytes() for n in EXPECTED):raise ValueError('ZIP actual payload/CRC contract')
 verify(hashes);save(out/'input_hashes.json',hashes)
 repro=reproduction(out,state)
 v=dict(version='V132',archive=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,counts=counts,native_payloads=bindings,motion=motion,official_score=None,best_official_version='V127',best_official_score=64.211848498398,experimental=True,normal_accuracy_promoted=False,source_binding_passed=True,real_joint_shared_landmarks=sum(s['report']['new_shared_landmarks'] for s in state.values()),real_pixel_lidar_depth_factors=sum(s['report']['new_pixel_lidar_depth_factors'] for s in state.values()),graph_loop_pairs={k:v['report']['verified_loop_pairs'] for k,v in state.items()},graph_temporal_pairs={k:v['report']['verified_temporal_pairs'] for k,v in state.items()},direct_model_graph_factors=0,reproduction_archive=str(repro),reproduction_archive_sha256=sha(repro),limitation='PLNet main network and weights unchanged, no retraining. ALIKED supplements actual extraction/matching. Direct Free-FreeReg attempt is not an active graph branch unless explicitly accepted; sensor checks are not official ATE.')
 save(out/'verification.json',v)
 save(out/'output_hashes.json',{str(q):sha(q) for q in out.rglob('*') if q.is_file() and q.name!='output_hashes.json'})
 save(H/'latest_candidate.json',dict(directory=str(out)))
 print('VERIFIED EXPERIMENTAL FULL ZIP',archive,sha(archive),flush=True);print('REPRODUCTION',repro,sha(repro),flush=True)
def audit():
 out=Path(load(H/'latest_candidate.json')['directory']);out.relative_to(H/'candidates')
 verify(load(out/'input_hashes.json'));verify(load(out/'output_hashes.json'))
 state,_,_=qualifications();v=load(out/'verification.json')
 if sha(v['archive'])!=v['archive_sha256'] or sha(v['reproduction_archive'])!=v['reproduction_archive_sha256']:raise ValueError('Archive changed')
 with zipfile.ZipFile(v['archive']) as z,zipfile.ZipFile(BEST) as old:
  if z.testzip() or z.namelist()!=EXPECTED:raise ValueError('Public ZIP changed')
  for scene,s in state.items():
   prefix=b'' if scene=='scene_0001' else b''.join(old.read('trajectories/scene_0002.txt').splitlines(keepends=True)[:524])
   if z.read('trajectories/'+scene+'.txt')!=prefix+m.b.tum_bytes(s['data']['ns'],s['poses']):raise ValueError('Public payload not actual original graph output')
 if m.b.validate_public_submission(out/'submission',R/'official/config/lidar_visual_inertial.json')!=v['counts']:raise ValueError('Public validation changed')
 save(H/'public_source_audit.json',dict(passed=True,archive=v['archive'],archive_sha256=v['archive_sha256'],official_score=None))
 print('PUBLIC/ACTUAL ORIGINAL NATIVE SOURCE AUDIT PASS')
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='audit':audit()
 else:package()
