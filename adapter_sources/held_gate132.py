import numpy as np
def failures(forward_px,reverse_px,depth_m,limits):
 a,b,c=(np.asarray(x,dtype=float) for x in (forward_px,reverse_px,depth_m))
 if a.ndim!=1 or a.shape!=b.shape or a.shape!=c.shape or len(a)<5 or any(not np.isfinite(x).all() or (x<0).any() for x in (a,b,c)):return ['post_global_invalid_held_evidence']
 bad=[]
 if max(np.median(a),np.median(b))>limits['held_median_px'] or max(np.percentile(a,90),np.percentile(b,90))>limits['held_p90_px']:bad.append('post_global_held_reprojection')
 if np.median(c)>limits['depth_median_m'] or np.percentile(c,90)>limits['depth_p90_m']:bad.append('post_global_held_depth_3d')
 return bad
def require_complete(report):
 scenes=report.get('scenes',{})
 if set(scenes)!={'scene_0001','scene_0002'} or any(not s.get('passed') or s.get('failures') for s in scenes.values()):raise ValueError('Final original pixel/3D held gates failed; candidate not releasable')
