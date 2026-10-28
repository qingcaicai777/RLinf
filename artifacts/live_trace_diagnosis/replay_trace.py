import json,io
from pathlib import Path
import numpy as np,torch,pyarrow.parquet as pq
from PIL import Image
from rlinf.algorithms.online_residual_rl.inference_trace import iter_trace
from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import sample_actions_with_rtc_guidance
from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext
p=Path('artifacts/live_inference_trace/session_95a6986a-cda0-4843-abfb-6e508c309b53');out=Path('artifacts/live_trace_diagnosis');cfg=json.loads((p/'manifest.json').read_text())['metadata']['inference_config'];lookup={e['request']['frame']:e for e in iter_trace(p) if e['request']['type']=='infer' and not e['request'].get('warmup')}
m=load_stage1_model(cfg['training_config'],cfg['checkpoint'],'cuda:0',norm_stats_path=cfg['norm_stats'],num_flow_steps=10,use_cuda_graph=False,rtc_guidance_mode='jacobian',action_representation='joint_absolute');m.requires_grad_(False);results=[];arrays={}
for frame in [0,16,32,48,64,128,256,512]:
 e=lookup[frame];r=e['request'];d=e['response']['diagnostics'];obs={k:np.asarray(v)[None] for k,v in r['observation'].items()};obs['task_descriptions']=[r['prompt']]
 torch.random.set_rng_state(torch.from_numpy(d['torch_cpu_rng_state']))
 torch.cuda.set_rng_state_all([torch.from_numpy(x) for x in d['torch_cuda_rng_states']])
 noise=torch.randn(1,32,32,device='cuda:0')
 with torch.inference_mode():plain,_=m.predict_action_batch(obs,noise=noise)
 committed=d['committed_raw_actions']
 if committed is not None:
  with torch.inference_mode(False):
   mapped=m.env_obs_to_observation(obs);raw=sample_actions_with_rtc_guidance(m,mapped,RTCGuidanceContext(prev_model_actions=torch.from_numpy(committed).to('cuda:0'),delay_steps=d['rtc_delay_steps']),num_steps=10,noise=noise,guidance_clip=d['rtc_guidance_clip'],guidance_mode='jacobian')
  with torch.inference_mode():guided=m.decode_rlt_actions(raw[:,:,:8],mapped.state)
 else:guided=plain
 plain=plain[0].cpu().float().numpy();guided=guided[0].cpu().float().numpy();online=d['full_actions_physical'][0];arrays[f'frame{frame}_online']=online;arrays[f'frame{frame}_replay']=guided;arrays[f'frame{frame}_plain']=plain
 result={'frame':frame,'replay_vs_online_max_joint_deg':float(abs(np.rad2deg(guided[:,:7]-online[:,:7])).max()),'rtc_vs_plain_mean_joint_deg':float(abs(np.rad2deg(guided[:,:7]-plain[:,:7])).mean()),'rtc_vs_plain_max_joint_deg':float(abs(np.rad2deg(guided[:,:7]-plain[:,:7])).max()),'plain_delta19_deg':np.rad2deg(plain[19,:7]-obs['states'][0,:7]).tolist()}
 if frame==0:
  row=pq.read_table('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008/data/episode_000000.parquet',columns=['observation.joint_state','observation.gripper_state','observation.image','observation.wrist_image']).slice(0,1).to_pylist()[0]
  ds=dict(obs);ds['states']=np.array([row['observation.joint_state']+row['observation.gripper_state']],np.float32);ds['main_images']=np.array([Image.open(io.BytesIO(row['observation.image'])).convert('RGB')]);ds['wrist_images']=np.array([Image.open(io.BytesIO(row['observation.wrist_image'])).convert('RGB')])
  with torch.inference_mode():dp,_=m.predict_action_batch(ds,noise=noise)
  arrays['dataset_initial_same_noise']=dp[0].cpu().float().numpy();result['dataset_same_noise_delta19_deg']=np.rad2deg(arrays['dataset_initial_same_noise'][19,:7]-ds['states'][0,:7]).tolist();result['initial_state_max_difference_deg']=float(abs(np.rad2deg(ds['states'][0,:7]-obs['states'][0,:7])).max())
 results.append(result);print(json.dumps(result),flush=True)
(out/'replay_results.json').write_text(json.dumps(results,indent=2));np.savez(out/'replay_actions.npz',**arrays)
