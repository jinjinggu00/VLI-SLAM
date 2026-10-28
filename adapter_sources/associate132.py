"""IO-only binding of original LVI-SAM DepthRegister to upstream ALIKED pixels."""
from common132 import *
from bridge_io132 import feature_contract,optical_rays
import cv2,subprocess,time
def main():
 out=H/'association_full';out.mkdir(exist_ok=False)
 verify(load(O/'prepared_r3/receipt.json')['source_hashes'])
 verify(load(H/'features_full/output_hashes.json'))
 binary=C/'v49_lidar_depth_frontend_20261004/author_depth'
 t=subprocess.run([str(binary),'--test'],capture_output=True,text=True,timeout=30)
 save(out/'fixture.json',dict(exit_code=t.returncode,stdout=t.stdout,stderr=t.stderr))
 if t.returncode or 'AUTHOR DEPTH CONTRACTS PASS' not in t.stdout:raise ValueError('Original DepthRegister fixture failed')
 receipt=dict(scenes={},source_hashes={str(binary):sha(binary),str(C/'v49_lidar_depth_frontend_20261004/author_depth.hpp'):sha(C/'v49_lidar_depth_frontend_20261004/author_depth.hpp')},depth_algorithm='Unmodified original LVI-SAM DepthRegister',intensity='Unused zero placeholder, not measured',descriptor_algorithm='Actual original pretrained ALIKED-n16')
 for scene in SCENES:
  dest=out/scene;dest.mkdir();prep=O/'prepared_r3'/scene
  cal=load(prep/'calibration.json');fx,fy,cx,cy=cal['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  d=m.b.restore(prep/'data.json');rows=[]
  for row in load(prep/'images.json'):
   i=row['node']
   if i in d['reserved']:raise ValueError('Independent scan used in association')
   f=np.load(H/'features_full'/scene/(str(i)+'.npz'));uv=f['keypoints'].astype(float)
   feature_contract(uv,f['descriptors'],f['keypoint_scores'],f['image_size'])
   rays=optical_rays(uv,K);cloud=O/'association_r3'/scene/(str(i)+'_cloud.f32')
   receipt['source_hashes'][str(cloud)]=sha(cloud)
   raypath=dest/(str(i)+'_rays.f32');depthpath=dest/(str(i)+'_depth.f32');rays.tofile(raypath)
   cmd=[str(binary),str(cloud),str(raypath),str(depthpath)];start=time.monotonic()
   with (dest/'native.log').open('a') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=30)
   if q.returncode:raise ValueError('Original association returned failure')
   depth=np.fromfile(depthpath,dtype='<f4')
   if depth.shape!=(len(uv),) or not np.isfinite(depth).all():raise ValueError('Native depth contract')
   valid=(depth>3.)&(depth<80.);np.savez_compressed(dest/(str(i)+'.npz'),uv=uv,rays=rays.astype(float),depth=depth,valid=valid,xyz=rays.astype(float)*depth[:,None])
   rows.append(dict(node=i,features=len(uv),valid=int(valid.sum()),command=cmd,exit_code=q.returncode,seconds=time.monotonic()-start,capture_ns=row['capture_ns'],header_ns=row['header_ns']))
   if len(rows)%20==0:print('ALIKED ORIGINAL DEPTH',scene,len(rows),'supported',sum(r['valid'] for r in rows),flush=True)
  save(dest/'receipt.json',rows)
  receipt['scenes'][scene]=dict(images=len(rows),features=sum(r['features'] for r in rows),lidar_supported=sum(r['valid'] for r in rows))
 verify(receipt['source_hashes']);save(out/'receipt.json',receipt)
 save(out/'output_hashes.json',{str(q):sha(q) for q in out.rglob('*') if q.is_file() and q.name!='output_hashes.json'})
if __name__=='__main__':main()
