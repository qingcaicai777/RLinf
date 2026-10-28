import io,json,time,yaml
from pathlib import Path
import numpy as np,torch,pyarrow.parquet as pq
from PIL import Image
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import sample_actions_with_rtc_guidance
from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext
root=Path('/home/qccc/projects/RLinf/artifacts/flexiv_inference_diagnosis')
t=pq.read_table('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data/episode_000000.parquet',columns=['observation.joint_state','observation.gripper_state','observation.image','observation.wrist_image','action']).to_pylist()
cfg=yaml.safe_load(Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml').read_text());initial=np.array(cfg['home']['left_joint_positions_rad']+t[0]['observation.gripper_state'],np.float32)
cap=Path('/home/qccc/projects/RLinf/artifacts/inference_images/session_1791623407637087412_1092819');live=[np.array(Image.open(cap/f'tick_000000_{k}.png').convert('RGB')) for k in ['main','wrist']]
m=load_stage1_model('examples/sft/config/flexiv_joint_stage1_inference.yaml','/home/qccc/checkpoints/myq/connecter_1010/full_weights.pt','cuda:0',norm_stats_path='/home/qccc/checkpoints/myq/connecter_1010/peg_insertion_joint_absolute_1008_v4/norm_stats.json',num_flow_steps=10,use_cuda_graph=False,rtc_guidance_mode='jacobian',action_representation='joint_absolute');m.requires_grad_(False)
paths={name:{'state':initial.copy(),'raw':None,'results':[]} for name in ['oracle_plain','oracle_jacobian','frozen_feedback_plain','frozen_feedback_jacobian']}
for frame in range(0,321,16):
 row=t[frame];imgs=[np.array(Image.open(io.BytesIO(row[k])).convert('RGB')) for k in ['observation.image','observation.wrist_image']];state=np.array(row['observation.joint_state']+row['observation.gripper_state'],np.float32)
 noise=torch.randn(1,32,32,device='cuda:0',generator=torch.Generator(device='cuda:0').manual_seed(42+frame))
 for name,path in paths.items():
  frozen=name.startswith('frozen');s=path['state'] if frozen else state;ims=live if frozen else imgs
  obs=m.env_obs_to_observation({'states':s[None],'main_images':ims[0][None],'wrist_images':ims[1][None],'task_descriptions':[cfg['recording']['task']]})
  if name.endswith('jacobian') and path['raw'] is not None:
   raw=sample_actions_with_rtc_guidance(m,obs,RTCGuidanceContext(prev_model_actions=path['raw'][:,16:],delay_steps=4),num_steps=10,noise=noise,guidance_clip=10,guidance_mode='jacobian')
  else:
   with torch.no_grad():raw=m.sample_actions(obs,num_steps=10,noise=noise)
  with torch.no_grad():out=m.decode_actions(raw,obs.state)[0].cpu().numpy()
  path['raw']=raw.detach().clone();path['state']=out[15].copy();path['state'][7]=np.clip(path['state'][7],.003,.06)
  truth=np.array([x['action'] for x in t[frame:frame+32]])
  path['results'].append({'frame':frame,'input_state':s.tolist(),'planned_action':out[4 if frame else 0].tolist(),'action15':out[15].tolist(),'chunk_q6_min':float(out[:,5].min()),'chunk_joint_MAE_vs_oracle_deg':float(abs(out[:,:7]-truth[:,:7]).mean()*180/np.pi) if not frozen else None})
 print('frame',frame,'q6_state/command15', {name:[round(p['results'][-1]['input_state'][5],4),round(p['results'][-1]['action15'][5],4)] for name,p in paths.items()},flush=True)
(root/'rtc_drift_experiment.json').write_text(json.dumps({'notes':['Oracle paths use recorded observations at every request.','Frozen feedback paths keep the old saved online image pair fixed and assume perfect command tracking. They are artificial feedback probes, not a physical rollout.','All paths use the same initial flow noise at each frame. RTC overlap is16 steps, delay4, weight10.'],'paths':{k:v['results'] for k,v in paths.items()}},indent=2))
for name,p in paths.items():
 vals=np.array([r['action15'][5] for r in p['results']]);print('SUMMARY',name,'q6_min',vals.min(),'q6_last',vals[-1],'MAE',np.mean([r['chunk_joint_MAE_vs_oracle_deg'] for r in p['results']]) if not name.startswith('frozen') else None,flush=True)
