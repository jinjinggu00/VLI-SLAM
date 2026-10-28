from pathlib import Path
import sys,subprocess,time,json,datetime
H=Path(__file__).resolve().parent;LH='/mnt/c/'+H.as_posix()[3:];D=r'C:/Users/k2230/AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe'
phase=sys.argv[1]
if phase not in ['features_full','features_probe','ffreereg_build','ffreereg_probe']:raise ValueError('Unknown owned phase')
names=subprocess.check_output([D,'ps','--format','{{.Names}}'],text=True).splitlines()
if any('slam' in n.lower() for n in names):raise ValueError('Existing Docker task retained')
active=H/'logs/feature_active.json'
if active.exists() and not json.loads(active.read_text()).get('finished'):raise ValueError('Owned heavy task still active')
cmd={'features_full':['bash',LH+'/features132.sh'],'features_probe':['bash',LH+'/features132.sh','--probe'],'ffreereg_build':['bash',LH+'/build_crossmodal.sh'],'ffreereg_probe':['bash',LH+'/crossmodal132.sh']}[phase]
args=['wsl.exe','-d','Ubuntu-20.04','--',*cmd]
stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S');log=H/'logs'/(phase+'_'+stamp+'.log');t=time.monotonic()
rec=dict(phase=phase,command=args,log=str(log),finished=False,start_bjt=datetime.datetime.now().isoformat())
active.write_text(json.dumps(rec,indent=2))
try:
 with log.open('x',encoding='utf-8') as f:q=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT)
 rec.update(exit_code=q.returncode,seconds=time.monotonic()-t,finished=True,stop_bjt=datetime.datetime.now().isoformat())
finally:
 active.write_text(json.dumps(rec,indent=2));(H/'logs'/(phase+'_'+stamp+'.json')).write_text(json.dumps(rec,indent=2))
print(log.read_text(encoding='utf-8',errors='replace')[-5000:]);print('EXIT',rec.get('exit_code'),'SECONDS',rec.get('seconds'))
raise SystemExit(rec.get('exit_code',1))
