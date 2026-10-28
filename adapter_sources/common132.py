from pathlib import Path
import sys,json,hashlib,zipfile
from decimal import Decimal
import numpy as np
from scipy.spatial.transform import Rotation
H=Path(__file__).resolve().parent;C=H.parent;R=C.parent;O=C/'v131_author_gba_appearance_lvi_20261009'
sys.path.insert(0,str(C/'v93_best_submap_remeasurement_20261006'));import measure93 as m
sys.path.insert(0,str(C/'v127_author_pointlio_robust_aux_20261008'));import native127 as native
from registration_math import PoseSeries
BEST=C/'v127_author_pointlio_robust_aux_20261008/candidates/v127_at7_ojff/YNU_SLAM_submission.zip'
BEST_SHA='d5ee18c5f84613497396f95213f8ff20c282e3642d4c9c6c1902ef6365bd0e78'
SCENES=('scene_0001','scene_0002')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(4*1024*1024),b''):h.update(x)
 return h.hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def load(p):return json.loads(Path(p).read_text())
def verify(h):
 for p,s in h.items():
  if sha(p)!=s:raise ValueError('Changed protected input: '+p)
def best(scene,ns):
 if sha(BEST)!=BEST_SHA:raise ValueError('Historical highest changed')
 with zipfile.ZipFile(BEST) as z:
  if z.testzip():raise ValueError('Historical ZIP CRC')
  rows=[x.split() for x in z.read('trajectories/'+scene+'.txt').decode().splitlines()]
 states={}
 for v in rows:
  if len(v)!=8:raise ValueError('Highest format')
  t=int(Decimal(v[0])*10**9);r=np.asarray(v[1:],dtype=float);T=np.eye(4);T[:3,3]=r[:3];T[:3,:3]=Rotation.from_quat(r[3:]).as_matrix();states[t]=T
 if len(states)!=dict(scene_0001=5044,scene_0002=4098)[scene]:raise ValueError('Actual highest count')
 return np.array([states[int(x)] for x in ns])
def vals(T):return [*T[:3,3],*Rotation.from_matrix(T[:3,:3]).as_quat()]
def read_features(path):
 raw=Path(path).read_bytes();n=int(np.frombuffer(raw[:4],dtype='<u4')[0])
 if len(raw)!=4+259*n*4 or n<1 or n>5000:raise ValueError('Native features schema')
 a=np.frombuffer(raw[4:],dtype='<f4').reshape(n,259).copy()
 if not np.isfinite(a).all():raise ValueError('Native finite features')
 return a
