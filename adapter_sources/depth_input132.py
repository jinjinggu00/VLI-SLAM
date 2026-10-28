"""Device/focal input binding only for the unchanged original DepthPro API."""
import torch
def call_original_depth(model,transform,image,known_fx=None):
 parameter=next(model.parameters())
 x=transform(image).to(device=parameter.device,dtype=parameter.dtype)
 focal=None if known_fx is None else torch.tensor(float(known_fx),device=parameter.device,dtype=torch.float32)
 return model.infer(x,f_px=focal)
