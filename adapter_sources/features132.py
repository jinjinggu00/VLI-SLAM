"""IO binding only: upstream ALIKED+LightGlue, pretrained, exact native restored pixels."""
from bridge_io132 import *
import os,sys,time,csv,resource,argparse
H=Path(__file__).resolve().parent;C=H.parent;W=H.parents[3];O=C/'v131_author_gba_appearance_lvi_20261009'
sys.path.insert(0,str(H/'upstream/lightglue'))
import torch,cv2
from lightglue import ALIKED,LightGlue
from lightglue.utils import load_image
cv2.setNumThreads(1);torch.set_num_threads(1)
def realpath(v):
 p=str(v)
 return W/p[len('/workspace/'):] if p.startswith('/workspace/') else Path(p)
def arrays(f):
 return {k:v.detach().cpu().numpy()[0].copy() for k,v in f.items() if torch.is_tensor(v)}
def validate_weights(model,path):
 s=torch.load(path,map_location='cpu')
 for i in range(9):
  s={k.replace('self_attn.%d'%i,'transformers.%d.self_attn'%i).replace('cross_attn.%d'%i,'transformers.%d.cross_attn'%i):v for k,v in s.items()}
 missing=[];shape=[]
 for k,v in model.named_parameters():
  if k not in s:missing.append(k)
  elif s[k].shape!=v.shape:shape.append(k)
 if missing or shape:raise ValueError('Uninitialized upstream parameters '+repr((missing,shape)))
 return dict(parameters=sum(v.numel() for v in model.parameters()),all_parameters_loaded=True,checkpoint_sha256=sha(path))
@torch.inference_mode()
def main(probe=False):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
 torch.manual_seed(132);torch.cuda.manual_seed_all(132);np.random.seed(132)
 out=H/('features_probe_r2' if probe else 'features_full');out.mkdir(exist_ok=False)
 config=dict(extractor='Original ALIKED-n16 pretrained',matcher='Original ALIKED LightGlue pretrained',max_keypoints=2048,resize_long_edge=1024,depth_confidence=-1,width_confidence=-1,flash=False,filter_threshold=.1,held_feature_identity='either endpoint native extraction index mod5 ==0 frozen before any matching',feature_dimension=128,unchanged_PLNet_core=True)
 save(out/'config.json',config)
 pre={str(p):sha(p) for p in [Path(__file__),H/'bridge_io132.py']+[p for p in (H/'upstream/lightglue/lightglue').rglob('*.py')]}
 save(out/'pre_source_hashes.json',pre)
 start=time.monotonic()
 extractor=ALIKED(model_name='aliked-n16',max_num_keypoints=config['max_keypoints']).eval().cuda()
 matcher=LightGlue(features='aliked',depth_confidence=-1,width_confidence=-1,flash=False,mp=False,filter_threshold=.1).eval().cuda()
 cache=Path(torch.hub.get_dir())/'checkpoints'
 evidence=dict(extractor=validate_weights(extractor,cache/'aliked-n16.pth'),matcher=validate_weights(matcher,cache/'aliked_lightglue_v0-1_arxiv.pth'))
 save(out/'model_receipt.json',evidence)
 summary={'scenes':{},'weights':evidence,'official_ate':None};repeat=[]
 for scene in ['scene_0001','scene_0002']:
  dest=out/scene;dest.mkdir();rows=load(O/'prepared_r3'/scene/'images.json');pairs=load(O/'prepared_r3'/scene/'pairs.json')
  if probe:
   pairs=list(dict.fromkeys(tuple(q) for q in [pairs[0],pairs[min(10,len(pairs)-1)],next(x for x in pairs if x[1]-x[0]>=200)]))
   ids={x for pair in pairs for x in pair};rows=[r for r in rows if r['node'] in ids]
  feats={};counts=[];sources={}
  for n,row in enumerate(rows):
   image_path=realpath(row['path']);image=load_image(image_path).cuda();f=extractor.extract(image);a=arrays(f)
   feature_contract(a['keypoints'],a['descriptors'],a['keypoint_scores'],a['image_size'])
   if len(a['keypoints'])<32:raise ValueError('Too little actual ALIKED image support')
   np.savez_compressed(dest/(str(row['node'])+'.npz'),**a)
   feats[row['node']]={k:v.detach() for k,v in f.items()}
   sources[str(image_path)]=sha(image_path);counts.append(len(a['keypoints']))
   if n<2 or n==len(rows)-1:
    b=arrays(extractor.extract(image));err=max(float(np.max(np.abs(a[k]-b[k]))) for k in a)
    if err>1e-5:raise ValueError('Original extractor repeat differs')
    repeat.append(dict(scene=scene,node=row['node'],max_abs=err,shape={k:list(v.shape) for k,v in a.items()}))
   if (n+1)%20==0:print('ALIKED',scene,n+1,len(rows),'points',sum(counts),flush=True)
  match_counts=[]
  for n,(i,j) in enumerate(pairs):
   result=matcher({'image0':feats[i],'image1':feats[j]})
   index=result['matches'][0].cpu().numpy();scores=result['scores'][0].cpu().numpy()
   if index.shape!=(len(scores),2) or np.any(index<0) or np.any(index[:,0]>=len(feats[i]['keypoints'][0])) or np.any(index[:,1]>=len(feats[j]['keypoints'][0])):raise ValueError('Original matcher IDs')
   np.savez_compressed(dest/('match_%d_%d.npz'%(i,j)),index=index,scores=scores)
   with (dest/('match_%d_%d.csv'%(i,j))).open('x',newline='') as f:
    writer=csv.writer(f);writer.writerow(['i','j','score']);writer.writerows([(int(x),int(y),float(s)) for (x,y),s in zip(index,scores)])
   if n<2 or n==len(pairs)-1:
    r2=matcher({'image0':feats[i],'image1':feats[j]});q=r2['matches'][0].cpu().numpy()
    if not np.array_equal(index,q) or np.max(np.abs(scores-r2['scores'][0].cpu().numpy()),initial=0)>1e-5:raise ValueError('Original matcher repeat differs')
   match_counts.append(len(index))
   if (n+1)%50==0:print('LIGHTGLUE',scene,n+1,len(pairs),'matches',sum(match_counts),flush=True)
  for p,s in sources.items():
   if sha(p)!=s:raise ValueError('Actual image changed')
  save(dest/'image_hashes.json',sources)
  summary['scenes'][scene]=dict(images=len(rows),pairs=len(pairs),features=sum(counts),matches=sum(match_counts),minimum_features=min(counts),maximum_features=max(counts))
  print('ALIKED SCENE COMPLETE',scene,summary['scenes'][scene],flush=True)
 for p,s in pre.items():
  if sha(p)!=s:raise ValueError('Runtime source changed')
 summary.update(seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,repeat=repeat)
 save(out/'report.json',summary);save(out/'output_hashes.json',{str(p):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='output_hashes.json'})
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--probe',action='store_true');main(a.parse_args().probe)
