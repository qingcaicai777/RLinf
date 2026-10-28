"""Compare the 20 actions executed after a request at frame 16."""
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
import yaml

from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model
from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import sample_actions_with_rtc_guidance
from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext

root = Path('artifacts/flexiv_inference_diagnosis')
config = yaml.safe_load(Path('examples/embodiment/config/flexiv_joint_absolute_inference.yaml').read_text())
data = Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008')
start_frame, request_steps, boundary, execute_steps = 16, 16, 20, 20
parquet = pq.read_table(data / 'data/episode_000000.parquet', columns=['observation.joint_state', 'observation.gripper_state', 'observation.image', 'observation.wrist_image', 'action'])
rows = parquet.slice(start_frame, 1).to_pylist()
row = rows[0]
state = np.array(row['observation.joint_state'] + row['observation.gripper_state'], np.float32)
obs = {'states': state[None], 'main_images': np.array([Image.open(io.BytesIO(row['observation.image'])).convert('RGB')]), 'wrist_images': np.array([Image.open(io.BytesIO(row['observation.wrist_image'])).convert('RGB')]), 'task_descriptions': [json.loads((data / 'meta/tasks.jsonl').read_text().splitlines()[0])['task']]}
truth = np.array(parquet.slice(boundary, execute_steps).column('action').to_pylist(), dtype=np.float32)
model = load_stage1_model(config['training_config'], config['checkpoint'], 'cuda:0', norm_stats_path=config['norm_stats'], num_flow_steps=int(config.get('num_flow_steps', 10)), use_cuda_graph=False, rtc_guidance_mode='jacobian', action_representation='joint_absolute')
model.requires_grad_(False)
outputs = {'jacobian_rtc': [], 'no_rtc': []}
for seed in range(5):
    # Match the online RNG order: startup bootstrap first, request-at-16 second.
    torch.manual_seed(seed)
    with torch.inference_mode():
        _bootstrap_physical, bootstrap_meta = model.predict_action_batch(obs, mode='eval')
    committed = bootstrap_meta['model_actions'][:, request_steps:].detach()
    with torch.inference_mode(False):
        guided = sample_actions_with_rtc_guidance(model, model.env_obs_to_observation(obs), RTCGuidanceContext(prev_model_actions=committed, delay_steps=boundary-start_frame), num_steps=model.num_steps, guidance_clip=model.rtc_guidance_clip, guidance_mode='jacobian')
    with torch.inference_mode():
        guided_physical = model.decode_rlt_actions(guided[:, :, :model.action_env_dim], model.env_obs_to_observation(obs).state)[0].cpu().float().numpy()
        # Reset seed and consume the same bootstrap call before the independent request.
        torch.manual_seed(seed)
        model.predict_action_batch(obs, mode='eval')
        plain_physical, _ = model.predict_action_batch(obs, mode='eval')
    outputs['jacobian_rtc'].append(guided_physical[boundary-start_frame:boundary-start_frame+execute_steps])
    outputs['no_rtc'].append(plain_physical[0, boundary-start_frame:boundary-start_frame+execute_steps].cpu().float().numpy())
    print(f'completed seed {seed}', flush=True)
outputs = {name: np.stack(value) for name, value in outputs.items()}
error = {name: pred-truth[None] for name,pred in outputs.items()}
summary = {'episode': 0, 'request_frame': start_frame, 'planned_handoff_frame': boundary, 'request_prediction_indices_executed': [boundary-start_frame, boundary-start_frame+execute_steps-1], 'dataset_action_frames': [boundary, boundary+execute_steps-1], 'prompt': obs['task_descriptions'][0], 'rtc_guidance_mode': 'jacobian', 'rtc_delay_steps': boundary-start_frame, 'rtc_committed_raw_indices': [request_steps,31], 'seeds': 5, 'alignment': 'The robot accepts the returned coarse actions at frame 20. Its 20 executed actions are original prediction indices 4..23 and are compared with dataset action frames 20..39.', 'metrics': {}}
for name, pred in outputs.items():
    e = error[name]
    summary['metrics'][name] = {'joint_mae_deg': (np.abs(np.rad2deg(e[:,:,:7])).mean(axis=(0,1))).tolist(), 'joint_rmse_deg': np.sqrt(np.square(np.rad2deg(e[:,:,:7])).mean(axis=(0,1))).tolist(), 'gripper_mae_mm': float(np.abs(e[:,:,7]).mean()*1000), 'chunk_mae_deg': float(np.abs(np.rad2deg(e[:,:,:7])).mean()), 'per_seed_chunk_mae_deg': np.abs(np.rad2deg(e[:,:,:7])).mean(axis=(1,2)).tolist()}
np.savez_compressed(root/'executed_chunk_frame16.npz', actual=truth, **{f'{k}_predicted':v for k,v in outputs.items()}, **{f'{k}_error':v for k,v in error.items()})
(root/'executed_chunk_frame16.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels=[f'J{i}' for i in range(1,8)]+['gripper']
fig, axes=plt.subplots(8,1,figsize=(14,20),sharex=True)
x=np.arange(boundary,boundary+execute_steps)
for dim,ax in enumerate(axes):
    scale=180/np.pi if dim<7 else 1000
    unit='deg' if dim<7 else 'mm'
    ax.plot(x,truth[:,dim]*scale,color='black',linewidth=1.4,label='dataset action')
    for name,color in [('jacobian_rtc','C1'),('no_rtc','C0')]:
        pred=outputs[name][:,:,dim]*scale
        ax.plot(x,pred.mean(0),color=color,label=name)
        ax.fill_between(x,pred.min(0),pred.max(0),color=color,alpha=.18)
    ax.set_ylabel(f'{labels[dim]} ({unit})');ax.grid(alpha=.25)
    if dim==0: ax.legend(ncol=3)
axes[-1].set_xlabel('Dataset frame / executed tick (30 Hz)')
fig.suptitle('Request at frame 16, handoff at frame 20: executed actions vs dataset (mean and range over 5 seeds)')
fig.tight_layout()
fig.savefig(root/'executed_chunk_frame16.png',dpi=160)
print(json.dumps(summary,indent=2),flush=True)
