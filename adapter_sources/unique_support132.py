import numpy as np
def support(source_uv,target_uv,cfg):
 a,b=(np.asarray(x,dtype=float) for x in (source_uv,target_uv))
 if a.ndim!=2 or a.shape[1]!=2 or a.shape!=b.shape or len(a)<cfg['min_training_inliers'] or not np.isfinite(a).all() or not np.isfinite(b).all():return False
 for uv in (a,b):
  v=np.floor(uv/np.array([752/4,480/3])).astype(int)
  cells={(int(x),int(y)) for x,y in v if 0<=x<4 and 0<=y<3}
  if len(cells)<cfg['min_occupied_cells']:return False
 return True
def depth_consistent(error,cfg):
 e=np.asarray(error,dtype=float)
 return bool(e.ndim==1 and len(e)>=cfg['min_training_inliers'] and np.isfinite(e).all() and not (e<0).any() and np.median(e)<=cfg['depth_median_m'] and np.percentile(e,90)<=cfg['depth_p90_m'])
