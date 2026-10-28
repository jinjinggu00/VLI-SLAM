"""Standalone IO/build/repeat check of the frozen original native joint graph."""
from pathlib import Path
import subprocess,json,hashlib,datetime,math,sys
from decimal import Decimal
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pose_rows(p):
 rows=[[float(x) for x in l.split()] for l in p.read_text().splitlines() if l.strip()]
 if any(len(r)!=7 or not all(math.isfinite(x) for x in r) or abs(sum(x*x for x in r[3:])-1)>1e-7 for r in rows):raise ValueError('Native finite unit-quaternion output required')
 return rows
def main():
 r=json.loads((H/'runtime.json').read_text())
 for name,s in r['input_hashes'].items():
  if sha(H/name)!=s:raise ValueError('Frozen evidence changed: '+name)
 for name,s in r['libraries'].items():
  if sha(name)!=s:raise ValueError('Original g2o runtime changed. Use pinned author Noetic image: '+r['image'])
 out=H/'runs'/datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
 cmd=['c++','-std=c++17','-O2','-I/usr/local/include','-I/usr/include/eigen3',str(H/'src/v132/joint_bridge.cpp'),'-L/usr/local/lib','-Wl,-rpath,/usr/local/lib','-lg2o_types_slam3d','-lg2o_core','-lg2o_stuff','-lglog','-o',str(out/'joint_graph')]
 with (out/'compile.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=240)
 report=dict(passed=True,scope='Independent rebuilt two-scene full upstream joint graphs from frozen true measured observations; not raw feature/depth rerun or official scoring',scenes={})
 for scene in ['scene_0001','scene_0002']:
  data=H/'inputs'/scene;dest=out/scene;dest.mkdir()
  cmd=[str(out/'joint_graph'),str(data/'graph.txt'),str(dest/'poses.txt'),str(dest/'optimizer.json'),'rotation_cauchy05','Cauchy','.5',str(data/'observations.txt')]
  with (dest/'native.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=900)
  expected=pose_rows(data/'poses.txt');actual=pose_rows(dest/'poses.txt')
  if len(expected)!=len(actual):raise ValueError('Native complete row count changed')
  error=max(abs(x-y) for a,b in zip(expected,actual) for x,y in zip(a,b))
  # Quaternion sign equivalence is not accepted as bit reproduction here.
  if error>1e-7:raise ValueError('Rebuilt original graph differs from protected actual output')
  ns=[int(x) for x in (data/'headers.txt').read_text().splitlines()]
  if len(ns)!=len(actual) or any(x>=y for x,y in zip(ns,ns[1:])):raise ValueError('Frozen exact header coverage')
  poses=np.array(actual);rot=Rotation.from_quat(poses[:,3:]).as_matrix();q=Rotation.from_matrix(rot).as_quat()
  suffix=''.join('%d.%09d %s\n'%(t//10**9,t%10**9,' '.join(format(float(x),'.17g') for x in [*a[:3],*qi])) for t,a,qi in zip(ns,actual,q)).encode('ascii')
  prefix=(data/'startup_prefix.txt').read_bytes();public=prefix+suffix
  expected_public=(data/'expected_public.txt').read_bytes()
  if public!=expected_public:raise ValueError('Independently rebuilt complete public trajectory differs from delivered bytes')
  rows=[x.split() for x in public.decode('ascii').splitlines()]
  stamps=[int(Decimal(x[0])*10**9) for x in rows]
  if any(Decimal(x[0])*10**9!=t for x,t in zip(rows,stamps)) or any(x>=y for x,y in zip(stamps,stamps[1:])):raise ValueError('Public nanosecond precision/order')
  if len(rows)!=dict(scene_0001=5044,scene_0002=4098)[scene]:raise ValueError('Full actual public row count')
  if sha(data/'poses.txt.landmarks.txt')!=sha(dest/'poses.txt.landmarks.txt'):raise ValueError('Independently rebuilt shared landmarks differ')
  (dest/'public_trajectory.txt').write_bytes(public)
  report['scenes'][scene]=dict(native_rows=len(actual),public_rows=len(rows),max_pose_value_error=error,pose_file_bit_exact=sha(data/'poses.txt')==sha(dest/'poses.txt'),landmark_file_bit_exact=True,complete_public_bytes_exact=True,command=cmd)
  print(scene,report['scenes'][scene],flush=True)
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 print('STANDALONE TWO-SCENE JOINT REPRODUCTION PASSED',out,flush=True)
if __name__=='__main__':main()
