# Online residual RL contract

This package contains the shape, timing, loss, replay, and update-budget contract
for the Flexiv online residual policy described in `Online残差RL.md`.

The default physical action is a 12-frame block with seven values per frame:
six end-effector deltas `(dx, dy, dz, dRx, dRy, dRz)` followed by the gripper
command. `ActionRepresentation.JOINT_ABSOLUTE` switches the same contract to
seven absolute joint targets plus one gripper value. In both modes the first six
frames are the future committed queue snapshot and the last six frames are the new
execution block.

The intended runtime path is:

1. APXinf runs the PI0.5 base independently of committed actions, returning
   12x7 reference actions, the complete multimodal prefix and its validity mask.
2. `OnlineResidualController` runs the RLT token encoder and bounded residual
   Actor in PyTorch. Deployment loads only the Encoder, not the RLT Decoder.
   The combined adapter returns the full 12x7 action block. Optional committed
   input constructs Critic metadata but does not affect VLA/Actor predictions.
3. Human intervention replaces the final TCP delta and/or gripper command;
   final accepted command receipts are saved for replay. The next decision's
   committed prefix must be the commands actually accepted for that queue.
4. The collector must wait for the full 12-frame window and `t+12` successor.
   `TransitionAssembler` validates the resulting row. `EpisodeReplay` assigns
   terminal rewards and admits the entire episode after a confirmed human label.
5. `OnlineTraining` connects the Learner to finite G/D credit and independent
   TD, rank, relative-Q and BC sampling. It defaults to 50 initial episodes,
   100 Critic-only warmup updates, G/D=5/5 and `lambda_rank=50`.

`human_mask` covers valid intervention positions in the full window. By default,
`pair_mask` uses the same positions within the current execution half; historical
committed interventions participate in BC but never create duplicate rank pairs. The
Actor BC term reads the raw full 12-frame Actor output, while Critic candidates
always use the immutable committed prefix. The optional smoothness term applies
only to the six non-gripper dimensions and is disabled by default.

## Implementation boundaries

The modules are tested independently. The user owns real-robot asynchronous
execution; fixed-handoff integration is not required to call model inference.
The synchronous APX evaluation worker can run full-horizon model evaluation,
but is not the asynchronous robot collector or an online training runner.

- `scheduling.py` owns fixed-frame handoffs, request versions and late-result
  rejection. It is optional reference code for the external executor.
- `action_mapping.py` provides a shared differentiable physical projection.
  The Learner now requires an explicit mapping. Its physical bounds must match
  the robot controller and its statistics must match SFT.
- `replay.py` owns episode admission, deduplication and supervision pools.
- `training.py` owns update credit, warmup, complete local save/resume and a
  publication callback. Rollout Actor installation and periodic checkpoint
  scheduling still need integration.
- APX exposes prefix hidden states in source. Committed-conditioned flow
  sampling is explicitly out of scope and is not needed for the clarified design.
- Full-prefix RLT checkpoints require 1024 positional capacity and masked
  multimodal training. Previous image-only 512-position checkpoints are not
  interchangeable; retrain/export them using the corrected SFT configuration.
- APX Rust compilation, real checkpoint export parity, 200 ms inference timing,
  human terminal-label collection and physical execution remain unverified.

## Flexiv intervention and finalized transition records

The single-arm Flexiv action is a physical 6D TCP delta plus a separate 0–1
gripper command. `FlexivInterventionDevice` accepts the structural
`myq.intervention.InterventionSource.poll(now_ns)` interface. An `override`
command replaces the selected arm's six delta dimensions; a `residual` command
adds to the policy delta. An optional gripper value replaces only the gripper
dimension. It is interpreted as physical width in metres by default, matching
the myq collector; sources that publish 0–1 values set
`gripper_unit="normalized"` on the device. `leader`, `follower`, and `both`
selectors are supported, with the
device configured for the one arm it controls. An expired command has no effect.

Wrap the Flexiv environment in `TeleopIntervention` to retain both the physical
command receipt and the per-dimension masks. The external 30 Hz executor owns
the handoff and supplies the 12 consecutive step-info records for each decision:

```python
from rlinf.algorithms.online_residual_rl import TransitionAssembler
from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
from rlinf.envs.real.flexiv import FlexivEnv, FlexivInterventionDevice
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention

env = TeleopIntervention(FlexivEnv(flexiv_config), FlexivInterventionDevice(source))
assembler = TransitionAssembler(config)
replay = EpisodeReplay(config, storage_dir=episode_dir)

# At inference: feature.committed and proposal[:6] are the real normalized
# committed queue. Capture them before any intervention.
assembler.begin(feature, proposal)

# After all t..t+11 receipts are available, or a valid terminal prefix:
row = assembler.complete_from_receipts(
    feature.obs_id,
    step_infos=window_step_infos,
    encode_actions=apx_adapter.encode_receipts,
    terminal=terminal,
    next_feature=feature_at_t_plus_12,
)

# Commit the whole episode only after its human success/failure label is final.
replay.commit_episode(episode_rows, success=success, terminal_frame=last_frame)
```

Each step info must contain `frame_index` and the final `action_receipt`; the
Flexiv environment also provides `action_receipt_physical`, with gripper width
in metres instead of the controller's 0–1 command. The executor must likewise
convert a policy's metre-valued gripper target to the environment's 0–1 code
before `step`; the adapter does not alter PI0.5 output units. These receipts
describe final accepted commands, not measured gripper motion. The wrapper provides
`human_mask` and `pair_mask`. The assembler normalizes the physical receipt
with the frozen SFT statistics and verifies that the
first six execution receipts match the committed snapshot. It stores the sparse
pre-intervention proposal only at trustworthy pair-mask positions in the current
execution half. `EpisodeReplay` validates and publishes the labelled episode
atomically. When `storage_dir` is set, it also creates one `.pt` file per
episode with feature records by `obs_id`, transition rows, and sparse correction
records. No file is published for an invalid or unlabelled episode.

## Flexiv SFT data and statistics

The selected source is
`/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_ok`.
It stores absolute TCP targets. TCP delta computation is deferred until SFT is
enabled; no conversion is performed by the inference or statistics tools.

Inspect the source without loading OpenPI or writing files:

```bash
python toolkits/lerobot/calculate_flexiv_norm_stats.py --inspect
```

Compute statistics directly from the immutable absolute-target recording. The
tool applies the same 32-step training-time delta conversion in the shared TCP
control frame as the SFT loader:

```bash
python toolkits/lerobot/calculate_flexiv_norm_stats.py \
  --dataset /path/to/prepared_flexiv_delta \
  --output /path/to/flexiv_delta_assets
```

This follows `calculate_norm_stats.py`: `openpi.shared.normalize.RunningStats`
accumulates state/action moments and quantiles, and `normalize.save` writes the
standard `norm_stats.json`. The training action horizon is 32; deployment keeps
the first 12 actions and discards the remaining 20. The physical action is six
TCP deltas followed by gripper width; the generated OpenPI statistics pad this
7D physical action to the model's 32D action width with zeros. State is seven
joint angles followed by gripper width. Widths stay in metres.
Every recorded frame is counted once, including incomplete final read batches.
The script refuses absolute-target action statistics for delta deployment and
does not overwrite existing statistics. Any executor using a 0–1 gripper code
must explicitly convert to/from physical width at its boundary.
