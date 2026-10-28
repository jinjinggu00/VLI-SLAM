"""Only orchestration/IO around original Free-FreeReg models and solver."""
from bridge_io132 import *
import sys,os,time,resource
from PIL import Image
from depth_input132 import call_original_depth
import runtime_compat132
from source_binding132 import verified_workspace_source
H=Path(__file__).resolve().parent;O=H.parent/'v131_author_gba_appearance_lvi_20261009';W=H.parents[3]
UP=H/'upstream/ffreereg'
sys.path.insert(0,str(UP));sys.path.insert(0,str(UP/'tools/DepthPro/src'))
os.chdir(UP);sys.argv=[sys.argv[0]]
import torch,cv2
import depth_pro
from depth_pro.depth_pro import DepthProConfig
from tools.YOHO.yoho_extract import yoho_extractor,nn_match
from demo import Pipe
from Utils.pose import kabschor
from Utils.utils import dpt_3d_convert,edge_filter
from scipy.spatial.transform import Rotation
torch.set_num_threads(1);cv2.setNumThreads(1)
class DepthIO:
 def __init__(self):
  cfg=DepthProConfig(patch_encoder_preset="dinov2l16_384",image_encoder_preset="dinov2l16_384",checkpoint_uri=str(UP/'tools/DepthPro/checkpoints/depth_pro.pt'),decoder_features=256,use_fov_head=True,fov_encoder_preset="dinov2l16_384")
  self.model,self.transform=depth_pro.create_model_and_transforms(config=cfg,device=torch.device('cuda'),precision=torch.float16)
  self.model.eval();self.K=None
 @torch.inference_mode()
 def __call__(self,image):
  a=Image.fromarray((image*255).clip(0,255).astype(np.uint8))
  pred=call_original_depth(self.model,self.transform,a,None if self.K is None else self.K[0,0])
  depth=pred['depth'].float().cpu().numpy()
  if self.K is None:
   f=float(pred['focallength_px']);h,w=depth.shape;K=np.array([[f,0,(w-1)/2],[0,f,(h-1)/2],[0,0,1.]])
  else:K=self.K.copy()
  if not np.isfinite(depth).all() or np.any(depth<=0):raise ValueError('Original neural predicted depth nonfinite')
  return depth,False,False,K
class OriginalPipe(Pipe):
 def _load_model(self):
  self.depthpro=DepthIO()
  self.yoho=yoho_extractor(fcgf_ckpt=str(UP/'tools/YOHO/model/Backbone/best_val_checkpoint.pth'),yoho_ckpt=str(UP/'tools/YOHO/model/PartI_train/model_best.pth'),device='cuda')
def infer(pipe,image,pcd,folder,seed,K=None):
 np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 pipe.vs=None;pipe.depthpro.K=K
 before=time.monotonic()
 ipc,cal=pipe._image_to_3d(image);ipc,pcd=pipe._coarse_align(ipc,pcd)
 pipe._determine_vs(ipc,pcd)
 sk,sf=pipe._extract_yoho(ipc,3000);tk,tf=pipe._extract_yoho(pcd,3000)
 # Original demo uses full ipc here; selected sk pixels are the correct one-to-one binding.
 uv,_=pipe.projector.proj_3to2(sk,cal,np.eye(4))
 pairs=pipe._match(sf,tf).astype(np.int32)
 if len(pairs)<24:raise ValueError('Upstream matching has insufficient actual correspondences')
 T=pipe._pose(uv,sk,tk,pairs,cal)
 if not np.isfinite(T).all() or abs(np.linalg.det(T[:3,:3])-1)>1e-4:raise ValueError('Original direct pose invalid')
 np.savez_compressed(folder/'raw.npz',source_uv=uv,source_predicted_xyz=sk,target_measured_xyz=tk,index=pairs,source_features=sf.cpu().numpy(),target_features=tf.cpu().numpy(),K=cal,pose_source_to_target=T)
 rec=dict(seconds=time.monotonic()-before,raw_matches=len(pairs),voxel_size=float(pipe.vs),estimated_camera_to_cloud=T.tolist(),image_depth='Original DepthPro prediction used to construct descriptor input; never measured depth',target_source='Actual input point samples',precision='Original DepthPro API float16; original YOHO float32',peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
 save(folder/'receipt.json',rec);return T,rec
@torch.inference_mode()
def main():
 out=H/'crossmodal_probe';out.mkdir(exist_ok=False)
 sources=load(H/'upstream_manifest.json')
 static=load(H/'crossmodal_static_asset_hashes.json')
 for path,value in static['assets'].items():
  if sha(path)!=value:raise ValueError('Original direct-model static asset changed')
 for repo in sources['sources'].values():
  for path,s in repo['source_hashes'].items():
   if sha(path)!=s:raise ValueError('Original source changed before model run')
 for path,r in sources['weights'].items():
  if sha(path)!=r['sha256']:raise ValueError('Original pretrained weight changed')
 pipe=OriginalPipe(nkpts=3000,w_2d=.5);report=dict(author_fixture={},same_view={},accepted_pairs=0,official_accuracy_verified=False)
 author=out/'upstream_outdoor';author.mkdir()
 image=pipe._load_img(str(UP/'demo/outdoor/000183.png'));pcd=pipe._load_pcd(str(UP/'demo/outdoor/000189.npy'))
 T,rec=infer(pipe,image,pcd,author,132)
 report['author_fixture']=dict(**rec,ground_truth_available=False,note='Original upstream outdoor asset executes; no author published ground-truth accuracy claim')
 for scene in ['scene_0001','scene_0002']:
  rows=load(O/'prepared_r3'/scene/'images.json');row=rows[len(rows)//2];i=row['node'];dest=out/scene;dest.mkdir()
  fx,fy,cx,cy=load(O/'prepared_r3'/scene/'calibration.json')['intrinsics'];K=np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]])
  path=W/row['path'][len('/workspace/'):]
  if sha(path)!=row['source_sha256']:raise ValueError('Real image source changed')
  image=np.array(Image.open(path).convert('RGB'))/255.
  cloud_path=O/'association_r3'/scene/(str(i)+'_cloud.f32')
  frozen=load(O/'association_r3/output_hashes.json')
  verified_workspace_source(cloud_path,W,frozen)
  raw=np.fromfile(cloud_path,dtype='<f4').reshape(-1,4)
  pcd=raw[:,[1,2,0]]*np.array([-1.,-1.,1.]);mask=(pcd[:,2]>3)&(pcd[:,2]<35)
  u=fx*pcd[:,0]/np.maximum(pcd[:,2],1e-6)+cx;v=fy*pcd[:,1]/np.maximum(pcd[:,2],1e-6)+cy;mask&=(u>2)&(u<750)&(v>2)&(v<478);pcd=pcd[mask]
  T,rec=infer(pipe,image,pcd,dest,132+i,K)
  error=dict(translation_m=float(np.linalg.norm(T[:3,3])),rotation_deg=float(np.linalg.norm(Rotation.from_matrix(T[:3,:3]).as_rotvec())*180/np.pi))
  passed=error['translation_m']<=.30 and error['rotation_deg']<=3.
  report['same_view'][scene]=dict(**rec,node=i,source_image_sha256=sha(path),source_cloud_sha256=sha(cloud_path),calibration_sha256=sha(O/'prepared_r3'/scene/'calibration.json'),capture_ns=row['capture_ns'],header_ns=row['header_ns'],expected_pose=np.eye(4).tolist(),actual_known_extrinsic_error=error,passed=passed,accepted_graph_factors=0)
  print('REAL DIRECT SAME VIEW',scene,error,'PASS',passed,flush=True);save(out/'report.json',report)
 save(out/'report.json',report)
 receipt=dict(finished=True,algorithm='Original WHU-USI3DV Free-FreeReg: DepthPro/YOHO/Kabsch',accepted_pairs=0,author_fixture_executed=True,same_view_passed={s:r['passed'] for s,r in report['same_view'].items()},scope='Same-view mandatory preliminary check; no cross-view graph factor yet',source_manifest_sha256=sha(H/'upstream_manifest.json'))
 save(H/'crossmodal_receipt.json',receipt)
 print('FREE FREEREG PROBE COMPLETE',receipt,flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  import traceback
  failure=dict(finished=True,algorithm='Original WHU-USI3DV Free-FreeReg: DepthPro/YOHO/Kabsch',accepted_pairs=0,status='failed_execution',error_type=type(error).__name__,error=str(error),source_manifest_sha256=sha(H/'upstream_manifest.json'),note='Execution failure is not successful registration; no direct graph factors. No numerical or OOM retry.')
  save(H/'crossmodal_receipt.json',failure)
  traceback.print_exc()
  raise
