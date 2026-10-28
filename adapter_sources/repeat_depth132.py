from common132 import *
import subprocess,time
def main():
 out=H/'association_repeat';out.mkdir(exist_ok=False);report=dict(scenes={})
 binary=C/'v49_lidar_depth_frontend_20261004/author_depth'
 for scene in SCENES:
  records=load(H/'association_full'/scene/'receipt.json')
  chosen=[records[k] for k in [0,len(records)//2,len(records)-1]];rows=[]
  for row in chosen:
   i=row['node'];src=H/'association_full'/scene;target=out/(scene+'_'+str(i)+'.f32')
   cmd=[str(binary),str(O/'association_r3'/scene/(str(i)+'_cloud.f32')),str(src/(str(i)+'_rays.f32')),str(target)]
   start=time.monotonic();q=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
   a=np.fromfile(target,dtype='<f4');b=np.fromfile(src/(str(i)+'_depth.f32'),dtype='<f4')
   if q.returncode or not np.array_equal(a,b):raise ValueError('Actual native ray-depth repeat differs')
   rows.append(dict(node=i,bit_exact=True,exit_code=0,seconds=time.monotonic()-start))
  report['scenes'][scene]=dict(samples=rows,scope='first/middle/last true images; does not claim full-extractor independent rerun')
 save(out/'report.json',report)
if __name__=='__main__':main()
