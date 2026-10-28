from pathlib import Path
import json,hashlib
import numpy as np
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for a in iter(lambda:f.read(4*1024*1024),b''):h.update(a)
 return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def feature_contract(uv,desc,scores,size):
 uv=np.asarray(uv);desc=np.asarray(desc);scores=np.asarray(scores);size=np.asarray(size)
 n=len(uv)
 if uv.shape!=(n,2) or desc.shape!=(n,128) or scores.shape!=(n,) or size.shape!=(2,) or np.any(size<=0):raise ValueError("Original ALIKED schema")
 if not all(np.isfinite(v).all() for v in [uv,desc,scores,size]):raise ValueError("Nonfinite features")
 if np.any(uv<-.5) or np.any(uv>=size):raise ValueError("Restored pixels outside original image")
def held_feature_mask(index):
 index=np.asarray(index,dtype=np.int64).reshape(-1,2)
 return (index[:,0]%5==0)|(index[:,1]%5==0)
def optical_rays(uv,K):
 import cv2
 xy=cv2.undistortPoints(np.asarray(uv,dtype=np.float64).reshape(-1,1,2),K,np.zeros(4)).reshape(-1,2)
 return np.column_stack([xy,np.ones(len(xy))]).astype("<f4")
