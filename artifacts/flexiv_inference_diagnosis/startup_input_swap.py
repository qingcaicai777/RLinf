"""Compare complete startup chunks with controlled image/state substitutions."""
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import torch
import yaml

from rlinf.models.embodiment.openpi_rlinf.inference import load_stage1_model

root = Path('artifacts/flexiv_inference_diagnosis')
config = yaml.safe_load(Path('examples/embodiment/config/flexiv_joint_absolute_inference.yaml').read_text())
data = Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008')
rows = pq.read_table(data / 'data/episode_000000.parquet', columns=['observation.joint_state', 'observation.gripper_state', 'observation.image', 'observation.wrist_image', 'action']).to_pylist()
cap = Path('artifacts/inference_images/session_1791641121079097907_1125482')
meta = json.loads((cap / 'tick_000000_observation.json').read_text())
record = rows[0]
ds = np.array(record['observation.joint_state'] + record['observation.gripper_state'], dtype=np.float32)
ls = np.array(meta['states'], dtype=np.float32)
di = [np.array(Image.open(io.BytesIO(record[k])).convert('RGB')) for k in ['observation.image', 'observation.wrist_image']]
li = [np.array(Image.open(cap / f'tick_000000_{k}.png').convert('RGB')) for k in ['main', 'wrist']]
# Six paired interventions identify main camera, wrist camera, and state effects.
cases = {
    'dataset': (ds, di),
    'live_state_only': (ls, di),
    'live_main_only': (ds, [li[0], di[1]]),
    'live_wrist_only': (ds, [di[0], li[1]]),
    'live_images_only': (ds, li),
    'live_all': (ls, li),
}
model = load_stage1_model(config['training_config'], config['checkpoint'], 'cuda:0', norm_stats_path=config['norm_stats'], num_flow_steps=10, use_cuda_graph=False, action_representation='joint_absolute')
model.requires_grad_(False)
predictions = {name: [] for name in cases}
with torch.inference_mode():
    for seed in range(5):
        noise = torch.randn(1, 32, 32, device='cuda:0', generator=torch.Generator(device='cuda:0').manual_seed(seed))
        for name, (state, images) in cases.items():
            obs = {'states': state[None], 'main_images': images[0][None], 'wrist_images': images[1][None], 'task_descriptions': [meta['prompt']]}
            action, _ = model.predict_action_batch(obs, noise=noise.clone())
            predictions[name].append(action[0].cpu().float().numpy())
        print(f'Completed paired seed {seed}', flush=True)
predictions = {name: np.stack(value) for name, value in predictions.items()}
truth = np.array([row['action'] for row in rows[:32]])
summary = {'capture': str(cap), 'checkpoint': config['checkpoint'], 'prompt': meta['prompt'], 'note': 'No RTC; five identical noise seeds per case. Dataset action is a reference demonstration, not a ground-truth action for changed live images.', 'live_minus_dataset_state': (ls-ds).tolist(), 'cases': {}}
for name, pred in predictions.items():
    state = cases[name][0]
    error = np.rad2deg(pred[:, :20, :7] - truth[None, :20, :7])
    summary['cases'][name] = {
        'first20_joint_mae_vs_reference_deg': float(np.abs(error).mean()),
        'action19_delta_from_input_deg_mean': np.rad2deg(pred[:, 19, :7]-state[:7]).mean(0).tolist(),
        'action19_delta_from_input_deg_std': np.rad2deg(pred[:, 19, :7]-state[:7]).std(0).tolist(),
        'action19_change_vs_dataset_input_deg_mean': np.rad2deg(pred[:,19,:7]-predictions['dataset'][:,19,:7]).mean(0).tolist(),
    }
summary['reference_action19_delta_deg'] = np.rad2deg(truth[19,:7]-ds[:7]).tolist()
# Context for why the full-episode first-action overlay looks deceptively close.
full = np.load('artifacts/flexiv_episode_compare/episode_000000_actions.npz')
state_all = np.array([r['observation.joint_state'] for r in rows])
gt = full['actual'][:,:7]
pred = full['predicted'][:,:7]
for label, sl in [('all', slice(None)), ('first20', slice(0,20)), ('first60', slice(0,60))]:
    summary[label] = {'model_mae_deg': float(np.abs(np.rad2deg(pred[sl]-gt[sl])).mean()), 'hold_current_state_mae_deg': float(np.abs(np.rad2deg(state_all[sl]-gt[sl])).mean()), 'model_delta_mae_deg': float(np.abs(np.rad2deg(pred[sl]-state_all[sl])).mean())}
np.savez_compressed(root / 'startup_input_swap.npz', **predictions, actual=truth, dataset_state=ds, live_state=ls)
(root / 'startup_input_swap.json').write_text(json.dumps(summary, indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(7, 1, figsize=(12, 17), sharex=True)
for i, ax in enumerate(axes):
    ax.plot(np.rad2deg(truth[:20,i]-ds[i]), 'k--', label='demonstration delta')
    for name, color in [('dataset','C0'), ('live_all','C1')]:
        delta = np.rad2deg(predictions[name][:,:20,i]-cases[name][0][i])
        ax.plot(delta.mean(0), color=color, label=name)
        ax.fill_between(range(20), delta.min(0), delta.max(0), color=color, alpha=.2)
    ax.set_ylabel(f'q{i+1} delta (deg)')
    ax.grid(alpha=.3)
axes[0].legend()
axes[-1].set_xlabel('Action index within first chunk (30 Hz)')
fig.suptitle('Startup motion relative to input state; bands: 5 noise seeds; RTC off')
fig.tight_layout()
fig.savefig(root / 'startup_input_swap.png', dpi=150)
print(json.dumps(summary, indent=2), flush=True)
