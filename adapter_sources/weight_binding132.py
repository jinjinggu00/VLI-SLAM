"""Independent exact checkpoint parameter binding of unchanged upstream constructors."""
from pathlib import Path
import sys,os,json
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'upstream/lightglue'))
import torch
from lightglue import ALIKED,LightGlue
from bridge_io132 import sha,save,load
torch.set_num_threads(1)
def check(model,path):
 state=torch.load(path,map_location='cpu')
 for i in range(9):
  state={k.replace('self_attn.%d'%i,'transformers.%d.self_attn'%i).replace('cross_attn.%d'%i,'transformers.%d.cross_attn'%i):v for k,v in state.items()}
 checked=0
 for k,v in model.named_parameters():
  if k not in state or v.shape!=state[k].shape or not torch.equal(v.detach().cpu(),state[k]):raise ValueError('Actual upstream parameter does not equal checkpoint: '+k)
  checked+=v.numel()
 return dict(all_parameters_bit_exact=True,parameters=checked,checkpoint_sha256=sha(path),scope='Exact current unchanged original constructor/checkpoint binding; executed full feature source constructors unchanged')
def main():
 cache=H/'tmp/torch_cache/hub/checkpoints'
 constructors=[('extractor',ALIKED(model_name='aliked-n16',max_num_keypoints=2048).eval(),cache/'aliked-n16.pth'),('matcher',LightGlue(features='aliked',depth_confidence=-1,width_confidence=-1,flash=False,mp=False,filter_threshold=.1).eval(),cache/'aliked_lightglue_v0-1_arxiv.pth')]
 old=load(H/'features_full/model_receipt.json');pre=load(H/'features_full/pre_source_hashes.json')
 for path,value in pre.items():
  if sha(path)!=value:raise ValueError('Executed extraction numerical source changed')
 report=dict(passed=True,models={},official_accuracy_verified=False)
 for key,model,path in constructors:
  report['models'][key]=check(model,path)
  if report['models'][key]['checkpoint_sha256']!=old[key]['checkpoint_sha256'] or report['models'][key]['parameters']!=old[key]['parameters']:raise ValueError('Executed checkpoint identity/count changed')
 save(H/'weight_binding132_receipt.json',report);print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
