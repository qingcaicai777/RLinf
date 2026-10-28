import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
from rlinf.models.embodiment.openpi_rlinf.modules import model as observation_model
from rlinf.models.embodiment.openpi_rlinf.pi0 import make_attn_mask

p=Path('artifacts/inference_images/session_1791641121079097907_1125482')
j=json.loads((p/'tick_000000_observation.json').read_text())
env={'states':np.array([j['states']],dtype=np.float32),'main_images':np.array([Image.open(p/'tick_000000_main.png').convert('RGB')]),'wrist_images':np.array([Image.open(p/'tick_000000_wrist.png').convert('RGB')]),'task_descriptions':[j['prompt']]}
m=load_stage1_model('examples/sft/config/flexiv_joint_stage1_inference.yaml','/home/qccc/checkpoints/myq/connecter_1010/full_weights.pt','cuda:0',norm_stats_path='/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json',num_flow_steps=10,use_cuda_graph=False,rtc_guidance_mode='approx',action_representation='joint_absolute')
m.requires_grad_(False)
results=[]
with torch.inference_mode():
 original=m.env_obs_to_observation(env)
 obs=observation_model.preprocess_observation(original,train=False)
 torch.manual_seed(0)
 noise=torch.randn(1,m.action_horizon,m.action_dim,device='cuda:0')
 for cast in (False,True):
  prepared=observation_model._observation_to_dtype(obs,m.embed_dtype) if cast else obs
  prefix,pm,pa=m.embed_prefix(prepared)
  _,mask,cache=m.build_prefix_cache(prepared)
  paths={}
  for method in ('full','cached'):
   x=noise.clone()
   for k in range(10):
    t=torch.full((1,),1-k/10,device='cuda:0',dtype=torch.float32)
    if method=='full':
     suffix,sm,sa,cond=m.embed_suffix(prepared,x,t)
     input_mask=torch.cat([pm,sm],dim=1)
     attention=make_attn_mask(input_mask,torch.cat([pa,sa],dim=0))
     _,so=m.llm([prefix,suffix],positions=torch.cumsum(input_mask.int(),dim=1)-1,mask=attention,adarms_cond=[None,cond])[0]
     v=m.velocity_from_suffix(so[:,-m.action_horizon:])
    else:
     v=m.velocity_from_suffix(m.run_suffix(prepared,x,t,cache,mask))
    x=x-.1*v
   paths[method]=m.decode_rlt_actions(x[:,:,:m.action_env_dim],original.state).float().cpu().numpy()
  diff=np.rad2deg(paths['full'][0,:,:7]-paths['cached'][0,:,:7])
  result={'cast_to_embed_dtype':cast,'full_cached_max_joint_diff_deg':float(np.abs(diff).max()),'full_cached_mean_joint_diff_deg':float(np.abs(diff).mean()),'full_action19':paths['full'][0,19].tolist(),'cached_action19':paths['cached'][0,19].tolist()}
  print(json.dumps(result),flush=True)
  results.append(result)
Path('artifacts/flexiv_joint_absolute_audit/live_forward_parity.json').write_text(json.dumps(results,indent=2)+'\n')
