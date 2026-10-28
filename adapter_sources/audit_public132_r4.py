"""Independent standard-library readback of the actual two-scene public ZIP."""
from pathlib import Path
from decimal import Decimal
import json,zipfile,hashlib,math
H=Path(__file__).resolve().parent
C=H.parent
BEST=C/'v127_author_pointlio_robust_aux_20261008/candidates/v127_at7_ojff/YNU_SLAM_submission.zip'
BEST_SHA='d5ee18c5f84613497396f95213f8ff20c282e3642d4c9c6c1902ef6365bd0e78'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def tum(raw):
 rows=[s.split() for s in raw.decode('ascii').splitlines() if s.strip()]
 if not rows or any(len(x)!=8 for x in rows):raise ValueError('Actual public TUM eight-field requirement')
 ns=[int(Decimal(x[0])*10**9) for x in rows]
 if any(Decimal(x[0])*10**9!=n for x,n in zip(rows,ns)) or any(x>=y for x,y in zip(ns,ns[1:])):raise ValueError('Actual nanosecond identity/order')
 poses=[[float(x) for x in r[1:]] for r in rows]
 if any(not all(math.isfinite(x) for x in r) or abs(math.sqrt(sum(x*x for x in r[3:]))-1)>1e-7 for r in poses):raise ValueError('Finite unit public poses')
 return ns,poses
def main():
 out=H/'candidates'/Path(load(H/'latest_candidate.json')['directory']).name
 v=load(out/'verification.json')
 zpath=out/'YNU_SLAM_submission.zip';repro=out/'V132_reproduction.zip'
 if sha(BEST)!=BEST_SHA or sha(zpath)!=v['archive_sha256'] or sha(repro)!=v['reproduction_archive_sha256']:raise ValueError('Protected or delivered archive changed')
 report={'passed':True,'archive_sha256':sha(zpath),'reproduction_sha256':sha(repro),'official_score':None,'scenes':{}}
 with zipfile.ZipFile(zpath) as z,zipfile.ZipFile(BEST) as old,zipfile.ZipFile(repro) as rz:
  if z.namelist()!=['README.md','trajectories/scene_0001.txt','trajectories/scene_0002.txt'] or z.testzip() or rz.testzip():raise ValueError('ZIP layout/CRC')
  for scene,n,start in [('scene_0001',5044,0),('scene_0002',4098,524)]:
   raw=z.read('trajectories/'+scene+'.txt');prior=old.read('trajectories/'+scene+'.txt')
   ns,poses=tum(raw);ons,_=tum(prior)
   if len(poses)!=n or ns!=ons or raw==prior:raise ValueError('Whole physical timestamps or unchanged payload')
   if start and raw.splitlines(keepends=True)[:start]!=prior.splitlines(keepends=True)[:start]:raise ValueError('Changed genuine 524-row startup prefix')
   native=H/'joint_r4'/scene/'native/poses.txt'
   if sha(native)!=v['native_payloads'][scene]['native_pose_sha256']:raise ValueError('Actual native source changed')
   a=[[float(x) for x in s.split()] for s in native.read_text().splitlines() if s.strip()]
   headers=[int(x) for x in rz.read('inputs/'+scene+'/headers.txt').decode('ascii').splitlines()]
   if headers!=ns[start:] or len(a)!=n-start or rz.read('inputs/'+scene+'/poses.txt')!=native.read_bytes():raise ValueError('Actual source/public/reproduction coverage binding')
   et=eq=0.
   for x,y in zip(a,poses[start:]):
    if len(x)!=7 or not all(math.isfinite(v) for v in x):raise ValueError('Actual native seven finite fields')
    et=max(et,max(abs(x[k]-y[k]) for k in range(3)))
    eq=max(eq,min(max(abs(x[k]-y[k]) for k in range(3,7)),max(abs(x[k]+y[k]) for k in range(3,7))))
   if et>1e-12 or eq>1e-12:raise ValueError('Translation or rotation differs from actual native solver')
   if rz.read('inputs/'+scene+'/expected_public.txt')!=raw:raise ValueError('Delivered archive and standalone expected full public bytes disagree')
   speeds=[];rotations=[]
   for i in range(1,n):
    dt=(ns[i]-ns[i-1])/1e9
    speeds.append(math.sqrt(sum((poses[i][k]-poses[i-1][k])**2 for k in range(3)))/dt)
    dot=abs(sum(poses[i][k]*poses[i-1][k] for k in range(3,7)))
    rotations.append(2*math.acos(min(1.,dot))*180/math.pi/dt)
   if max(speeds)>5 or max(rotations)>180:raise ValueError('Complete public trajectory motion bound')
   report['scenes'][scene]={'rows':n,'native_rows':len(a),'native_translation_max_error':et,'native_quaternion_sign_equivalent_error':eq,'nanosecond_headers_exact':True,'startup_prefix_byte_exact':True,'max_speed_mps':max(speeds),'max_rotation_rate_degps':max(rotations)}
 (H/'independent_public_readback.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
