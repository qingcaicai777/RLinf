"""Run the online coarse RTC schedule and compare every executed action."""
import argparse
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

ROOT = Path('artifacts/flexiv_inference_diagnosis_1011_20000')
DATASET = Path('/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_joint_absolute_1008')
START_FRAME = 16
CHUNK_STEPS = 20
REQUEST_STEPS = 16
EXECUTED_AFTER_REQUEST = REQUEST_STEPS
SEEDS = (0, 1, 2)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--reuse-predictions",
    action="store_true",
    help="Regenerate framewise plots and metrics from the saved full-episode inference",
)
parser.add_argument("--prompt", default=None, help="Override the episode task prompt.")
args = parser.parse_args()
ROOT.mkdir(parents=True, exist_ok=True)

config = yaml.safe_load(Path('examples/embodiment/config/flexiv_joint_absolute_inference.yaml').read_text())
parquet = pq.ParquetFile(DATASET / 'data/episode_000000.parquet')
count = parquet.metadata.num_rows
with (DATASET / 'meta/tasks.jsonl').open() as stream:
    prompt = args.prompt or json.loads(stream.readline())['task']
request_frames = {0, *range(START_FRAME, count, REQUEST_STEPS)}
observations = {}
actual = np.empty((count, 8), dtype=np.float32)
row_index = 0
for batch in parquet.iter_batches(batch_size=64, columns=['observation.joint_state', 'observation.gripper_state', 'observation.image', 'observation.wrist_image', 'action']):
    for row in batch.to_pylist():
        actual[row_index] = row['action']
        if row_index in request_frames:
            state = np.asarray(row['observation.joint_state'] + row['observation.gripper_state'], dtype=np.float32)
            main = np.asarray(Image.open(io.BytesIO(row['observation.image'])).convert('RGB'))
            wrist = np.asarray(Image.open(io.BytesIO(row['observation.wrist_image'])).convert('RGB'))
            observations[row_index] = {'states': state[None], 'main_images': main[None], 'wrist_images': wrist[None], 'task_descriptions': [prompt]}
        row_index += 1

if args.reuse_predictions:
    cached = np.load(ROOT / 'full_episode_stride20.npz')
    if not np.array_equal(cached['actual'], actual):
        raise ValueError('Saved predictions belong to a different episode action sequence.')
    predictions = cached['predicted']
    if predictions.shape != (len(SEEDS), count, 8):
        raise ValueError(f'Unexpected saved prediction shape: {predictions.shape}')
else:
    model = load_stage1_model(config['training_config'], config['checkpoint'], 'cuda:0', norm_stats_path=config['norm_stats'], num_flow_steps=int(config.get('num_flow_steps', 10)), use_cuda_graph=False, rtc_guidance_mode='jacobian', action_representation='joint_absolute')
    model.requires_grad_(False)
    all_predictions = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        predicted = np.full_like(actual, np.nan)
        # Startup bootstrap supplies the first 20 executed actions without RTC.
        with torch.inference_mode():
            physical, metadata = model.predict_action_batch(observations[0], mode='eval')
        previous_raw = metadata['model_actions'].detach()
        predicted[:CHUNK_STEPS] = physical[0, :CHUNK_STEPS].float().cpu().numpy()
        previous_request = 0
        # Each request runs 20 model steps; after a 4-frame handoff wait,
        # indices 4..19 supply the 16 actions until the next request.
        for request_frame in range(START_FRAME, count, REQUEST_STEPS):
            boundary = request_frame + (CHUNK_STEPS - REQUEST_STEPS)
            committed_offset = request_frame - previous_request
            committed = previous_raw[:, committed_offset:].detach()
            if committed.shape[1] == 0:
                raise RuntimeError(f'No committed RTC overlap at frame {request_frame}.')
            observation = observations[request_frame]
            with torch.inference_mode(False):
                mapped = model.env_obs_to_observation(observation)
                raw = sample_actions_with_rtc_guidance(model, mapped, RTCGuidanceContext(prev_model_actions=committed, delay_steps=CHUNK_STEPS-REQUEST_STEPS), num_steps=model.num_steps, guidance_clip=model.rtc_guidance_clip, guidance_mode='jacobian')
            with torch.inference_mode():
                physical = model.decode_rlt_actions(raw[:, :, :model.action_env_dim], mapped.state)
            start = boundary
            stop = min(count, boundary + EXECUTED_AFTER_REQUEST)
            if start < count:
                predicted[start:stop] = physical[0, CHUNK_STEPS-REQUEST_STEPS:CHUNK_STEPS][:stop-start].float().cpu().numpy()
            previous_raw = raw.detach()
            previous_request = request_frame
            print(f'seed={seed} request={request_frame} executes={start}:{stop}', flush=True)
        if not np.isfinite(predicted).all():
            missing = np.flatnonzero(~np.isfinite(predicted[:, 0]))
            raise RuntimeError(f'Online schedule left uncovered episode frames: {missing[:10].tolist()}')
        all_predictions.append(predicted)
        print(f'completed episode for seed={seed}', flush=True)
    predictions = np.stack(all_predictions)

frames = np.arange(count, dtype=np.int64)
errors = predictions[:, frames] - actual[frames][None]
summary = {
    'episode': 0, 'episode_frames': count, 'comparison_frames': frames.tolist(),
    'comparison_stride_frames': 1, 'seeds': list(SEEDS),
    'request_schedule': {'startup_bootstrap_frame': 0, 'subsequent_request_first_frame': START_FRAME, 'request_interval_frames': REQUEST_STEPS, 'handoff_delay_frames': CHUNK_STEPS-REQUEST_STEPS, 'executed_actions_per_subsequent_prediction': EXECUTED_AFTER_REQUEST, 'executed_prediction_indices': [CHUNK_STEPS-REQUEST_STEPS, CHUNK_STEPS-1]},
    'prompt': prompt,
    'evaluation': 'Teacher-forced observations from the recorded dataset at each request; RTC uses the prior predicted raw action overlap. Robot dynamics and closed-loop image/state changes are not simulated.',
    'all_frame_metrics': {
        'joint_mae_deg': np.abs(np.rad2deg(errors[:, :, :7])).mean(axis=(0,1)).tolist(),
        'joint_rmse_deg': np.sqrt(np.square(np.rad2deg(errors[:, :, :7])).mean(axis=(0,1))).tolist(),
        'gripper_mae_mm': float(np.abs(errors[:, :, 7]).mean()*1000),
        'overall_joint_mae_deg': float(np.abs(np.rad2deg(errors[:, :, :7])).mean()),
    },
}
np.savez_compressed(ROOT/'full_episode_framewise_20k.npz', frames=frames, actual=actual, predicted=predictions)
(ROOT/'full_episode_framewise_20k.json').write_text(json.dumps(summary, indent=2))
import csv
with (ROOT/'full_episode_framewise_20k.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['frame', *[f'actual_q{i}' for i in range(1,8)], 'actual_gripper_m', *[f'pred_mean_q{i}' for i in range(1,8)], 'pred_mean_gripper_m', *[f'error_mean_q{i}' for i in range(1,8)], 'error_mean_gripper_m'])
    mean_prediction = predictions.mean(axis=0)
    for frame in range(count):
        writer.writerow([int(frame), *actual[frame].tolist(), *mean_prediction[frame].tolist(), *(mean_prediction[frame]-actual[frame]).tolist()])
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels = [f'J{i}' for i in range(1,8)] + ['gripper']
fig, axes = plt.subplots(8, 1, figsize=(15, 19), sharex=True)
for dim, axis in enumerate(axes):
    scale = 180/np.pi if dim < 7 else 1000
    unit = 'deg' if dim < 7 else 'mm'
    truth = actual[:, dim] * scale
    axis.plot(frames, truth, 'k-', label='dataset action', linewidth=.8)
    sample_prediction = predictions[:, :, dim] * scale
    axis.plot(frames, sample_prediction.mean(0), 'C0-', label='model mean (3 seeds)', linewidth=.8)
    axis.fill_between(frames, sample_prediction.min(0), sample_prediction.max(0), color='C0', alpha=.18, label='seed range')
    for boundary in range(CHUNK_STEPS, count, REQUEST_STEPS):
        axis.axvline(boundary, color='gray', alpha=.12, linewidth=.5)
    axis.set_ylabel(f'{labels[dim]} ({unit})')
    axis.grid(alpha=.25)
    if dim == 0:
        axis.legend(ncol=3)
axes[-1].set_xlabel('Executed episode frame (one action comparison per frame; vertical lines mark chunk handoffs)')
fig.suptitle('Full episode online-schedule inference; every executed action vs dataset; Jacobian RTC')
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(ROOT/'full_episode_framewise_20k.png', dpi=160)
print(json.dumps(summary, indent=2), flush=True)
