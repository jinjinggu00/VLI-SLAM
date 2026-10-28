from pathlib import Path
import sys,subprocess,datetime,json,time
H=Path(__file__).resolve().parent; W=H.parents[3]
LH='/workspace/semantic_map/lidar_challenge_2026/submission_campaign_20261002/'+H.name
D=r'C:\Users\k2230\AppData\Local\Programs\DockerDesktop\resources\bin\docker.exe'
IMAGE='sha256:9df819fdbb0ef0925adcd32f1f6d543e12272d4cc5c35118cb259597ebb508e7'
def main():
 phase=sys.argv[1]; command=' '.join(sys.argv[2:])
 if not command: raise ValueError('Explicit command required')
 active=H/'logs/feature_active.json'
 if active.is_file() and not json.loads(active.read_text()).get('finished',False):raise ValueError('Existing WSL neural experiment retained')
 names=subprocess.check_output([D,'ps','--format','{{.Names}}'],text=True).splitlines()
 if any('slam' in x.lower() for x in names): raise ValueError('Existing SLAM process retained '+repr(names))
 stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');log=H/'logs'/(phase+'_'+stamp+'.log')
 cmd='set -eo pipefail;cd '+LH+';source /opt/ros/noetic/setup.bash;'+command
 args=[D,'run','--rm','--name','slam-v132-'+phase,'--label','slam.heavy=true','--cpus','4','--memory','8g','--memory-swap','10g','--gpus','all',
 '--mount','type=bind,source='+str(W)+',target=/workspace,readonly','--mount','type=bind,source='+str(H)+',target='+LH,'--mount','type=bind,source='+str(W)+',target=/mnt/c/Users/k2230/Desktop/Study/Project/多传感器融合SLAM/workspace,readonly']
 for k,v in {'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','TMPDIR':LH+'/tmp','ROS_MASTER_URI':'http://127.0.0.1:11631','ROS_IP':'127.0.0.1','ROS_HOME':LH+'/tmp/ros_home','ROS_LOG_DIR':LH+'/tmp/ros_logs','CUDA_CACHE_PATH':LH+'/tmp/cuda_cache','TORCH_HOME':LH+'/tmp/torch_cache','HF_HOME':LH+'/tmp/hf_cache'}.items():args+=['-e',k+'='+v]
 (H/'tmp/ros_logs').mkdir(exist_ok=True)
 args+=['--entrypoint','/bin/bash',IMAGE,'-lc',cmd];t=time.monotonic()
 with log.open('x',encoding='utf-8') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 rec=dict(phase=phase,command=args,exit_code=p.returncode,seconds=time.monotonic()-t,log=str(log))
 (H/'logs'/(phase+'_'+stamp+'.json')).write_text(json.dumps(rec,indent=2))
 print(log.read_text(encoding='utf-8',errors='replace')[-9000:],flush=True);print('PHASE',phase,'EXIT',p.returncode,'SECONDS',rec['seconds'])
 raise SystemExit(p.returncode)
if __name__=='__main__': main()
