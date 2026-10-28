from pathlib import Path
import sys,json,hashlib,subprocess
H=Path(__file__).resolve().parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8*1024*1024),b''):h.update(a)
 return h.hexdigest()
src={}
for name in ['lightglue','ffreereg','minkowskiengine']:
 p=H/'upstream'/name
 src[name]={'commit':subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),'remote':subprocess.check_output(['git','-C',str(p),'remote','get-url','origin'],text=True).strip(),'source_hashes':{str(x):sha(x) for x in p.rglob('*.py') if '.git' not in x.parts}}
weights={}
for f in [(H/'upstream/ffreereg/tools/DepthPro/checkpoints/depth_pro.pt'),H/'upstream/ffreereg/tools/YOHO/model/Backbone/best_val_checkpoint.pth',H/'upstream/ffreereg/tools/YOHO/model/PartI_train/model_best.pth']:
 if f.stat().st_size<1000000:raise ValueError('Checkpoint pointer or partial file')
 weights[str(f)]={'bytes':f.stat().st_size,'sha256':sha(f)}
(H/'upstream_manifest.json').write_text(json.dumps({'sources':src,'weights':weights},indent=2))
print({k:v['commit'] for k,v in src.items()});print(weights)
