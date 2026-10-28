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

The local `run_flexiv_online_residual.py` entry owns each labelled episode and
can keep one Learner alive across `--episodes N` episodes. `ResidualHandoffRuntime` sends one final target per 30 Hz tick and
accepts completed inference only at fixed six-frame handoffs. The synchronous
APX evaluation worker is separate from this real-robot timing path.

- `scheduling.py` owns fixed-frame handoffs, request versions and late-result
  rejection. It is optional reference code for the external executor.
- `action_mapping.py` provides a shared differentiable physical projection.
  The Learner now requires an explicit mapping. Its physical bounds must match
  the robot controller and its statistics must match SFT.
- `replay.py` owns episode admission, deduplication and supervision pools.
- `online.py` moves receipt assembly and learner updates off the control thread.
  It assembles receipts and stages episodes pending a terminal label.
  `learner_process.py` owns optimizer work, monitoring and checkpoints. The
  Actor mailbox installs a complete snapshot only on a later inference request;
  each accepted decision records its Actor version.
- APX exposes prefix hidden states in source. Committed-conditioned flow
  sampling is explicitly out of scope and is not needed for the clarified design.
- Full-prefix RLT checkpoints require 1024 positional capacity and masked
  multimodal training. Previous image-only 512-position checkpoints are not
  interchangeable; retrain/export them using the corrected SFT configuration.
- APX Rust compilation, real checkpoint export parity, 200 ms inference timing,
operator switches and physical execution remain unverified on hardware.

## Split Torch or APX inference and dual-arm Flexiv control

The split deployment keeps PI0.5, the residual Actor/Critic, episode
collection and the optional learner in RLinf's `.venv`. A separate process in
the Flexiv conda environment owns both RDK arms, the follower gripper, myq's
RealSense camera processes and the 30 Hz control loop. The two processes exchange
versioned NumPy/msgpack messages on a local TCP socket; no Flexiv SDK, Torch or
gym installation is needed in the other process.

The scheduling loop lives in `RemoteRobotRuntime` and depends only on the
`RobotAdapter` protocol. Each adapter returns `RobotObservation` with
policy-ready inputs and opaque action context, then maps each selected action
to a device command and returns a `RobotControlResult` from `control_tick`.
Before each handoff, `poll_control` returns a `RobotTickRequest`, so Space or a
terminal label takes effect before the runtime accepts a chunk or submits the
next inference. The final command receipt remains separate from that operator
request. The Flexiv adapter owns its
camera/state mapping, TCP-delta and gripper conversion, two-arm startup, leader
takeover and safe stop. Another robot can supply those adapter methods without
changing the fixed-rate scheduler; `remote_client.py` provides the same
dependency-light network client for that adapter's entry script. Wire version 3
carries the mapped observation, action context and inference phase, so update
both processes together.

The robot client reads `drag_collect.yaml` at startup. It uses the configured
RealSense serial numbers, 640x480 streams, main 320x320 crop (`-20,+90`) and
wrist 480x480 crop (`+20,0`) resized to 224x224. It first enables the left
gripper and selects its tool so Home uses the intended TCP frame. After both
arms reach Home, it initializes and opens the gripper, reads its measured width,
then samples the final force bias before policy control begins. Set
`gripper_control.initialize_on_start: false` in the myq config or pass
`--skip-gripper-init` to keep its current opening; tool selection and measured
width initialization still run. Before Home, the client samples an initial
force bias, configures both arms in Cartesian force mode and reduces the
right-arm impedance. When the myq Home
block contains `left_joint_positions_rad` and `right_joint_positions_rad`, it
uses myq's bounded joint Home return for both arms. The configured
`home_right_position_offset_m` is resolved using the saved right joints and
`right_tcp_pose_wxyz` through myq's existing IK helper. After Home, it restores
Cartesian control for the leader and switches the follower to joint control
when the inference action representation is `joint_absolute`. Legacy TCP Home
configurations retain the left Home plus right offset/yaw startup path with
the corresponding myq runtime. The server checks the post-crop image
shapes and backend during the handshake, before either robot is activated. The matching
single-process Flexiv config has the same serial/crop and speed values.

Set `control.home_left_position_offset_m: [0.0, -0.05, 0.0]` in the myq YAML
to shift the inference startup Home 5 cm along the left robot's world −Y axis,
keeping orientation fixed. The default is a zero offset. With a saved
`home.left_tcp_pose_wxyz`, the client solves the shifted Home before returning.
If only joint Home is saved, it first returns to that Home, captures the left
joint state and TCP pose together, and solves the offset using that reference.
Both moves use the same bounded joint Home trajectory. Each startup begins
from the saved baseline, so the offset does not accumulate. The client logs
`Home offset arm=left world_translation_m=...` when it resolves the target.
This setting is consumed by the split inference client; the collector's
existing Home behavior is unchanged.

From the RLinf root, start the trained Stage1 Torch policy on the inference host.
Torch is the default backend on both processes:

Both inference backends use 10 PI0.5 flow denoising steps by default. Pass
`--num-flow-steps N` to the inference server to change the count; the Stage1
SFT training recipe retains its own four-step setting.
At 30 Hz, a six-frame Actor handoff gives the next inference 200 ms. A coarse
base block of K frames with request offset R gives it (K-R)/30 seconds. If a
result is late, the robot holds its last target and continues to poll it. By
default, it continues holding until a usable result arrives; `--handoff-grace-ticks N`
sets how often the robot reports that it is still waiting. Elapsed actions are
skipped, never replayed. If the entire block expires, the server decision is
discarded and inference is requested again from a fresh observation while the
robot keeps holding. In the fine phase, a block received after its six-frame
handoff is discarded even if it still contains future actions. The robot holds
and requests a fresh block for the next six-frame boundary. Timing-invalid
receipts exclude affected windows from replay; healthy windows and the final
label remain usable.
Compare the server's `model_ms` with the robot's `response_ms` to locate model
or transport delay. `response_ms` measures robot-side request submission to the
socket response becoming available; `handoff_ms` measures submission to the
tick when the new block becomes active. The robot also prints `budget_ms`, the
time from request to planned handoff. For K=24 and R=18, that is only 200 ms at
30 Hz. If `model_ms`
already exceeds this budget, lower R to submit earlier; waiting longer keeps
the robot at its last target but cannot make the action arrive at the planned
boundary. Once the block has no actions left, it is discarded and a fresh
inference request is made from the current observation.

For an expired block, check whether the server printed `inference ... frame=0`
before the robot stopped. If it did not, the model call was still running or
the request had not reached the server; an accepted handoff's `response_ms`
is unavailable for that request.

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --serve --stage1-only \
  --checkpoint /path/to/stage1/model_state_dict/full_weights.pt \
  --norm-stats /path/to/flexiv_delta_assets/norm_stats.json
```

For Stage1-only inference, add `--base-chunk-steps K` to the server command to
set the handoff index within each 32-action prediction; the default is 22.
`--base-request-steps R` sets the request index; the default is 16. With these
defaults, the first prediction starts at tick 0, requests the next prediction
at tick 16, and executes rows 0–21 before switching at tick 22. The new
prediction was made at tick 16, so it enters at row 6. It executes rows 6–21,
requests again at its own row 16 (tick 32), and switches at its own row 22
(tick 38). Subsequent coarse handoffs repeat every 16 ticks. The server sends
only the rows that can execute before the next handoff. Torch RTC guidance
uses the previous accepted raw PI0.5 prediction starting at each new request
tick. Because Stage1 actions are deltas from their observation TCP, the server
rebases that remaining prediction to the new observation TCP before guidance;
the gripper and padded model dimensions keep their original values.

For a positive request index, the server requires `2*R >= K`. The new prediction
starts executing at its row `K-R`, so its row `R` must fall at or after its
handoff. When `2*R = K`, the robot accepts the new prediction and requests the
following one on the same tick. `K-R` can differ from six; it sets both the
entry row and the inference budget in ticks. The server rejects other positive
values before robot startup. `R=0` retains the fixed K-tick handoff schedule
used by short Stage1-only runs; that schedule requires `2*K <= 32`.

For a short Stage1 Torch comparison without cross-chunk RTC guidance, add
`--disable-rtc` to the server command. Each base handoff still executes the
actions aligned with its prediction origin. The flag only
stops passing the previous raw model actions into the next sample;
it cannot be combined with APX or Stage2. The flag works with the default
`jacobian` mode and skips Jacobian compilation and warmup while RTC is disabled.

Then start the robot process in its Flexiv environment:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --myq-config /home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml \
  --capture-dir /home/qccc/projects/RLinf/flexiv_vla_rgb \
  --capture-count 3 \
  --capture-interval-seconds 5
```

Add `--skip-gripper-init` to the robot command when the gripper is already
initialized and its current opening should be preserved. The option overrides
`gripper_control.initialize_on_start` for that run. After the first two regular
model completions, the server logs `completion_hz`, the rolling rate of completed
requests over the latest ten results; bootstrap is excluded. The robot logs the
same value as `completion_hz`. `model_hz` is `1000 / model_ms`, the reciprocal
of model call latency. `response_hz` is `1000 / response_ms`, the reciprocal of
robot-observed request/response latency, including transport and server
queueing. These reciprocal latency values describe how long one call takes;
they are not sustained throughput. Request and robot handoff cadence are
governed by the configured chunk schedule. With K=24 and R=18 at 30 Hz, the
robot requests every 600 ms after the initial request, regardless of the
model's faster call time.

On an RTX 4090, the default Torch deployment now captures the repeated PI0.5
action-expert step in a CUDA graph. With the Flexiv Stage1 checkpoint, two saved
camera frames and ten flow steps, the complete Torch call measured 89–92 ms
after warmup; fixed-seed actions matched eager inference exactly, including RTC
guidance. A separate-process loopback run completed K=4, R=0 chunks at 7.5 Hz
without late handoffs. K=3, R=0 reached 10 Hz request frequency, but the 100 ms
window was too short for the model and socket round trip, so it held at each
boundary and skipped stale actions. Use K=4, R=0 when Torch RTC guidance is
required and a faster handoff rate is desired:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --serve --stage1-only --backend torch \
  --checkpoint /path/to/stage1/model_state_dict/full_weights.pt \
  --norm-stats /path/to/flexiv_delta_assets/norm_stats.json \
  --num-flow-steps 10 --base-chunk-steps 4 --base-request-steps 0 \
  --rtc-max-guidance-weight 10 --rtc-delay-buffer-size 8 \
  --rtc-guidance-mode approx
```

The Torch server defaults to `jacobian` RTC. Its guided steps use compiled
autograd, while bootstrap and requests without RTC overlap use CUDA graph
replay. Set `--rtc-guidance-mode approx` for the lightweight correction and
CUDA graph replay during guided sampling as well. Approximate guidance scales
each post-Euler correction by the flow step size and caps its interpolation
coefficient at one, keeping the update between the unguided proposal and the
overlap target. The server runs a
shape-matched bootstrap and guided warmup
before opening the socket, so lazy compilation finishes before the robot client
connects and starts its response timeout.

The Torch server accepts `--rtc-max-guidance-weight W` (default `10`) and
`--rtc-delay-buffer-size N` (default `8`). It measures request response time on
the robot, converts that duration to 30 Hz ticks, and keeps the most recent N
accepted measurements; RTC uses at least the planned request-to-handoff interval
as its hard-prefix length, or the maximum retained delay when that is longer.
`--rtc-guidance-mode approx` keeps the lightweight correction. Set it to
`jacobian` to use a vector-Jacobian product through the PI0.5 clean-action
estimate at every flow step, matching OpenPI's VJP guidance semantics. This
mode differentiates each flow step through the compiled suffix; the
continuation warmup runs it once before robot motion to avoid charging compile
latency to the first handoff. APX does not apply RTC guidance.

The Flexiv config omits custom Cartesian velocity and acceleration values, so
`SendCartesianMotionForce` uses the RDK defaults. The local RDK 1.9 API declares
0.5 m/s linear velocity, 1.0 rad/s angular velocity, 2.0 m/s² linear
acceleration, and 5.0 rad/s² angular acceleration as those defaults. Hardware
safety limits remain controlled by Flexiv and its safety configuration.

For Stage1-only handoffs at 10 Hz, the matching APX export measured 44–46 ms per
call after warmup. A separate-process loopback run completed K=3, R=0 at 10 Hz
without late handoffs; start the robot client with `--backend apx` as well. APX
does not apply Torch's cross-chunk RTC guidance, so validate its action quality
separately. CUDA graph replay is enabled by default for Torch on CUDA; pass
`--disable-torch-cuda-graph` to use eager inference.

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --serve --stage1-only --backend apx \
  --checkpoint /path/to/stage1/model_state_dict/full_weights.pt \
  --apx-checkpoint /path/to/stage1/apxinf_stage1/model.safetensors \
  --norm-stats /path/to/flexiv_delta_assets/norm_stats.json \
  --num-flow-steps 10 --base-chunk-steps 3 --base-request-steps 0
```

Stage1-only Torch skips the unused RLT encoder both with and without RTC
guidance; RTC still uses the prefix KV cache needed for action sampling.
When Torch Stage2 does need RLT features, its prefix and normalization already
stay on the GPU; there is no APX-style host round trip to remove.
Stage1-only APX requests actions without returning prefix hidden states for
the absent Stage2 head. When Stage2 needs RLT features, the BF16 APX
prefix-output path also reuses a captured CUDA graph after its first request.
The CUDA-enabled binding exposes the raw BF16 prefix and RMSNorm scale as
device views; PyTorch normalizes them on the GPU before RLT encoding. Older
bindings and non-BF16 APX modes fall back to the CPU prefix API. Rebuild and
reinstall `apxinf_py` to remove the GPU→CPU→GPU transfer for BF16. The action
chunk still returns through
the host. For a single observation in eval mode,
both backends skip masked camera tokens in RLT attention while keeping the
original positional embeddings; training and unmasked inputs are unchanged.
These reductions can lower `model_ms`; the fine phase's six-tick handoff caps
request rate at 5 Hz. Time the first APX request separately because it
captures the graph.

When `--capture-dir` is set, the robot process saves three policy observation
pairs as RGB PNG files, with at least five seconds between captures. The first
available observation is saved immediately; later captures use the first policy
observation at or after each five-second interval. Each run creates a unique
subdirectory with files such as `tick_000000_main.png` and `tick_000150_wrist.png`.
These are the cropped and resized camera arrays sent to the inference server.
Each pair also includes `tick_000000_observation.json` with the same observation's
eight-dimensional state, TCP pose, prompt, action representation, and last joint
target. Use these matched inputs to reproduce a failed robot prediction offline
and compare the measured joints with the commanded target.
Image encoding runs on a background worker; a full queue drops a preview capture
rather than waiting in the control loop.

To record a complete split inference session for offline diagnosis, restart the
server with the updated code and add `--diagnostic-dir artifacts/live_inference_trace`
to the robot command. Each run writes a `session_<UUID>` directory containing:

- `manifest.json`: robot and inference configuration supplied to the client.
- `events.jsonl`: ordered requests/replies, transport timestamps, every control
  receipt, operator takeover state, and request-to-execution frame mappings.
- `exchange_*.npz`: lossless RGB observations, state, full physical action chunks,
  normalized actions and committed RTC overlap. Torch requests also capture CPU
  and initialized CUDA RNG states before sampling.
- `summary.json`: submitted/written/dropped/error counts after normal shutdown.

Inference observations are saved at each policy request, while measured joints,
gripper width, TCP pose, and commanded targets are recorded at every control tick.
The measured state is sampled immediately before sending that tick's command;
compare it with the preceding target when assessing tracking lag. Camera exposure
host timestamps and ages are attached to inference observations. Image arrays are
the processed RGB inputs sent over the wire; internal model resizing and
normalization are reproduced by the model's observation adapter.

Writes run in a bounded background queue. A full queue or write error is logged
and counted; a missing summary means the recording may have ended abruptly.
Recording adds array copies, diagnostic reply payloads, and one measured-state
read per control tick, so inspect deadline and transport logs during capture.
The trace keeps discarded episodes for diagnosis. Keep the same checkpoint for
replay; weights are not copied into the trace, and RNG capture alone does not
guarantee bitwise parity across CUDA graphs or different backends.

Load the requests and replies without pickle:

```python
from rlinf.algorithms.online_residual_rl.inference_trace import iter_trace

for event in iter_trace("artifacts/live_inference_trace/session_<UUID>"):
    request = event["request"]
    if request["type"] == "infer" and not request.get("warmup"):
        observation = request["observation"]
        prompt = request["prompt"]
        prediction = event["response"]["diagnostics"]["full_actions_physical"]
```

For APX, use the same two entry scripts and select `apx` on both sides. The
inference host also needs the matching APX Stage1 export:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --serve --backend apx --stage1-only \
  --checkpoint /home/qccc/checkpoints/myq/model_state_dict/full_weights.pt \
  --apx-checkpoint /home/qccc/checkpoints/myq/apxinf_stage1/model.safetensors \
  --norm-stats /home/qccc/checkpoints/myq/flexiv_delta_assets/norm_stats.json
```

After the server prints its listening address, start the robot process in a
separate terminal on the Flexiv host:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --backend apx \
  --myq-config /home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml
```

Both commands default to `127.0.0.1:18765`; set matching `--host` and `--port`
values if the processes run on different hosts. This wire format is not
authenticated or encrypted; use a trusted private network and never expose
the server to the public Internet. To run Stage2 with a trained Actor, omit
`--stage1-only` and add `--actor-checkpoint /path/to/trained_actor.pt`. To start
online training from the Stage1 policy, add `--online-train --episode-dir
/path/to/episodes --checkpoint-dir /path/to/stage2_checkpoints`; no Actor
checkpoint is needed. The residual head starts at zero, so the server collects
50 labelled base-only episodes by default, then completes the configured
Critic-only warmup before enabling the Actor. Set `--base-only-episodes N` to
change the collection count. `--stage1-only` is the permanent base-only
inference mode and cannot be combined with `--online-train`.

For one episode with base control during grasp and transport, launch the server
with `--actor-checkpoint /path/to/trained_actor.pt`, `--base-chunk-steps 10`,
and `--base-request-steps 7`, or replace the checkpoint option with
`--online-train` and its episode/checkpoint
directories. The current defaults start the episode in the coarse phase. The
Actor becomes active after the explicit phase switch.
On the robot console, press **I** when the grasped part reaches the pre-insertion
region. The request is latched until the next base handoff. The first six
actions of that accepted base chunk form a bridge while the first fine request
runs; the following handoff accepts the Actor's six-frame chunk. The switch is
one-way for the episode. During online training's initial base-only episodes,
the fine phase still uses the base action at six-frame boundaries and supplies
the existing 6+6 replay rows; coarse-phase decisions are excluded from Actor
training. An episode that never enters the fine phase supplies no such rows.

The first model call is a blocking warmup; then the robot thread queues each
request at its configured coarse request tick or at the start of a fine chunk,
and continues sending one target per absolute
30 Hz deadline. The network worker preserves call order across receipts,
handoffs, and inference requests so delayed inference cannot starve receipts.
With request step 0, the frame 0 prefetch reuses the bootstrap observation;
with the default request step 16, the first coarse request reads a new observation.
Camera readiness is timed after robot startup and
Home motion complete. On every steady-state tick, the robot polls operator
controls, accepts any ready handoff, submits the next inference if needed,
then sends one final target per arm and reports its receipt. A hold sends the
unchanged target once, without replaying a delta. In the coarse phase, both
backends produce a physical 32×7 prediction. With the default origin schedule,
the bootstrap block executes rows 0–21 and later accepted predictions execute
rows 6–21 before their own row-22 handoff. In the fine phase, the
existing 12×7 and 6+6 split remains in force.
Torch's first call has no RTC guidance. A second bootstrap call warms the
continuation path without replacing the first call's actions or RTC context.
Later requests use the most recently accepted raw prediction, sliced at the
request tick relative to the prediction's observation tick. Physical TCP
deltas and command receipts never become RTC guidance. APX retains its
unconditioned base sampling. At each active chunk boundary the robot thread polls
the result twice without waiting and piggybacks its acceptance on the next
inference request; the server registers that decision before computing the
next block. This avoids an extra network round trip. If the reply arrives
after a boundary, the robot holds the last TCP target and skips the elapsed
action indices. If the whole block expires, it discards that result and requests
a fresh block from the current observation while continuing to hold. The next
inference uses the original boundary observation when request step is 0; a later
configured request step reads a fresh observation when it becomes due. Torch
clears its guidance after a late acceptance. Each new session also resets its guidance.
Holds and missed control deadlines mark their receipts as scheduling-invalid.
The collector rejects affected windows and windows lacking a valid successor,
while retaining healthy windows and the final success/failure label. Each delta is composed exactly once against the inference
observation's TCP pose, and the final accepted target is converted back to a
Stage1-frame 7D delta receipt. The gripper action receipt records the accepted
target width in metres. The measured opening remains part of the next
observation and may lag while the fingers move. The collector also retains the
absolute final TCP target and re-expresses overlapping receipt windows relative
to each decision's own observation pose before normalizing them for training. Pressing
Space pauses model commands; R zeroes force bias and rebases the
right-leader/left-follower myq drag controller before enabling human control.
Press Space again to resume model control with a fresh prediction. The inference stream
continues in the background; each 30 Hz receipt records the current policy
proposal, final human-controlled command, and per-dimension intervention mask.
The transition stores this proposal trace and uses trustworthy corrected
execution frames for ranking and human BC. Press `S`/`F` to end with a human success/failure
label and stay paused in Stage2. Press Space to start a fresh coarse episode from the current pose, or
optionally press H and wait for Home and force zeroing first; wait for the automatic Fine trigger or press I at the fine stage.
Use `D` to discard the current unfinished episode and stay paused in Stage2, or `Q`
to stop and retain unlabelled valid windows for later labelling. Stage2 collection can save valid transitions with
`--no-online-train`. If the terminal window is invalid or unavailable, healthy
nonterminal fragments remain usable; no terminal reward is assigned to those
fragments. Saved metadata records `terminal_covered`.

The inference server owns its policy and collector thread. A separate spawn
process owns Replay, Actor/Critic targets and both optimizers. Its device defaults
to CPU; set `online_residual.learner_device` and `learner_cpu_threads` in the
Stage2 configuration to select its resources. The robot sends receipts
asynchronously and inference continues while the learner updates. Actor snapshots are installed only at a later inference boundary. This
keeps learner work off the 30 Hz robot control loop; inference latency still
needs to be measured on the target host.

During the first 50 labelled episodes and the Critic-only warmup, both backends
return the Stage1 reference without residual noise. Once warmup ends, online
rollout mode adds `rollout_noise_std` before `tanh` only to the new six-action
half. APX keeps its established deterministic Stage1 sampler and applies
exploration only in the PyTorch Actor sidecar. Standalone Stage2 evaluation and
`--stage1-only` remain deterministic.

The local tests cover both backend contracts, raw-action guidance, session
reset, fixed handoff, manual receipts and absolute control deadlines without a
GPU or robot. Before
the first live run, verify the two robot serials, camera serials/crops, saved
Home pose, clear workspace, correct gripper travel and Actor checkpoint. Torch
and APX live inference latency and robot-side command timing remain unverified
on hardware; the APX server also requires a CUDA-enabled binding.

## Select TCP delta or absolute joint inference

Use the same inference configuration on the server and robot client to select
Stage1 TCP deltas or absolute joint targets. The two supplied configurations
include checkpoint and normalization paths for this workstation; command-line
arguments override the server defaults. Run both commands from the RLinf root.

Start the TCP delta server in RLinf's environment:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --inference-config examples/embodiment/config/flexiv_tcp_delta_inference.yaml
```

Then start its robot client in the Flexiv environment:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --inference-config examples/embodiment/config/flexiv_tcp_delta_inference.yaml
```

These commands retain the 30 Hz TCP execution path, 22-step base handoff,
16-step request index, Jacobian RTC and existing drag intervention. Stop both
processes before selecting the other configuration.

Start the absolute joint server with the provided `joint_abs_1007` weights:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_inference.yaml
```

Then connect the matching robot client:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_inference.yaml
```

Set `disable_debug: true` in the shared inference YAML to suppress diagnostic
INFO logs on both processes and disable robot-side trace and processed-image
recording, even when `--diagnostic-dir` or `--capture-dir` is supplied. Startup
messages, the active prompt, inference frequency/latency, response latency,
handoff timing, Stage2 phase changes, Actor versions, episode collection
summaries, warnings and errors remain visible. These status records use
the `flexiv.status` logger; detailed joint arrays and prediction continuity
diagnostics stay suppressed. Both entrypoints also accept
`--disable-debug` and `--no-disable-debug` to override the YAML setting. To
record a diagnostic run again, pass `--no-disable-debug` on both processes and
add `--diagnostic-dir` on the robot. Vendor SDK console output is unaffected.

To run the same absolute-joint Stage1 recipe through APX, point the server at an
APX export of those joint-trained weights and select APX on both processes. APX
does not use Torch RTC guidance, so the server selects the approximate guidance
mode only to satisfy the deployment option contract:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_inference.yaml \
  --backend apx --rtc-guidance-mode approx \
  --apx-checkpoint /path/to/joint_abs_1007_apx/model.safetensors
```

After the server starts listening, connect the Flexiv process with the same
action representation and backend:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_inference.yaml \
  --backend apx
```

The APX model still returns normalized `[1,32,32]` actions. The selected
`pi05_flexiv_joint_absolute` output transform uses the joint checkpoint's
statistics to decode the first eight channels before the robot client receives
them. Do not reuse a TCP-delta APX export or its normalization file for this
command; the server validates the recipe and 8D statistics before loading APX.

The client verifies the action representation and width during the handshake,
before activating hardware. Both modes use 8 state values (7 measured joint
angles in radians and measured gripper width in metres), main RGB
`[320,320,3]`, wrist RGB `[224,224,3]` and the task prompt. PI0.5 resizes images
through its existing input transforms. The model predicts `[1,32,32]` raw
normalized actions; the physical outputs differ as follows:

| Configuration | Physical output | Decode and execution |
| --- | --- | --- |
| `flexiv_tcp_delta_inference.yaml` | `[1,32,7]` | Unnormalize 6 world-frame TCP deltas and gripper width; compose with the observation TCP pose; send Cartesian targets. |
| `flexiv_joint_absolute_inference.yaml` | `[1,32,8]` | Unnormalize 7 absolute joint angles and gripper width; send joint targets in radians. |

The joint recipe uses `pi05_flexiv_joint_absolute`, quantile normalization,
8 physical action channels and 24 padding channels. Its output transform
retains all 8 physical values. Absolute joint targets keep their coordinates
across observations, so RTC reuses raw overlap without TCP pose rebasing or
adding the measured joint state. The normalization validator rejects missing,
non-finite or degenerate physical statistics and mismatched action widths.

The joint client uses the existing dual-arm Home sequence, then switches the
left arm to `NRT_JOINT_POSITION`. To start the left arm at a demonstration's
measured joint position, set `home_dataset_path` in the shared inference YAML
to its dataset root, `data` directory, or an episode Parquet file. A directory
selects the first episode in sorted path order and reads its first frame's
`observation.joint_state`; `action` is not used for Home. The right arm retains
the myq Home and its offsets. Dataset Home bypasses the left Home offset and
does not modify the saved myq configuration. Gripper startup remains controlled
by myq's gripper settings; its tool is selected before Home, and its optional
initialization/opening runs after Home. VLA then receives fresh measured joints
and gripper width. Dataset loading and joint-vector validation happen
before hardware startup; the existing Home trajectory checks hardware joint
limits before sending movement commands. This option requires `joint_absolute`
and PyArrow in the robot environment. Omit the key to use the saved myq Home.

Each control tick validates an 8-value finite
action and checks the 7 joint angles against the connected follower's
`RobotInfo.q_min` and `q_max`. It calls `SendJointPosition` with zero terminal
joint velocities and the seven velocity/acceleration limits in `joint_control`
(rad/s and rad/s²). Set `max_joint_velocity_rad_s: null` or omit that key to
use the connected follower's `RobotInfo.dq_max`; an explicit seven-value list
overrides it. `max_joint_acceleration_rad_s2` requires seven positive finite
values. The Stage1 joint inference config uses 0.2 rad/s and 0.5 rad/s² for
autonomous policy targets. These are application limits, not reported hardware
maxima. Restart the robot client after changing these settings. It clamps
gripper width to the configured physical range.
An invalid joint target stops the episode before that command is sent; a late
prediction holds the last accepted joint target. Space stops sending policy
motion targets and calls SDK `Stop()` on both arms in the background. After
both stops complete, it captures their settled positions and restores hold
control. Wait for `Pause complete` before R, H, O/C or Space. Stopping receipts
are invalid and excluded from training. S/F/D can still end the episode while
stopping; Q waits for the pause worker before final hardware cleanup.
R performs stationary force zeroing before enabling manual control:
myq TCP admittance, leader/follower alignment and asynchronous
`myq.joint_control.JointTargetController` produce absolute joint commands for
both arms in `NRT_JOINT_IMPEDANCE`. This matches
`dual_robot_drag_collect.py --action-representation joint_absolute`. The right
arm uses `control.leader_joint_impedance_scale`; the follower uses nominal
joint stiffness. Manual velocity, acceleration and IK step/age limits come
from the myq configuration. Both IK results are validated before sending either
arm. The recorded action is the actual joint target sent to the follower,
followed by its accepted gripper width. Use S/F/D/Q for label/discard/quit.

To switch the gripper between fully closed and fully open, enable optional
binary control in the inference configuration used by both processes:

```yaml
gripper_policy:
  binary: true
  threshold: 0.5
  hold_arm_on_close: true
```

Restart both processes after changing this setting. You can also append
`--binary-gripper` to both launch commands, or `--no-binary-gripper` to select
continuous width control. Continuous control is the default. The handshake
checks that both processes use the same binary-control setting before hardware
activation.

Binary control compares the model's gripper action **before unnormalization**
directly with `threshold`: below it closes, above it opens, and equality holds
the current target. This value is not a metre width and is not rescaled to a
0–1 opening fraction. The server carries the normalized gripper channel with
each action block (channel 6 for TCP deltas, channel 7 for absolute joints),
sliced in the same way as the physical actions. The client selects the value
for the executing tick, including after a late handoff. For residual proposals,
the normalized Actor output takes precedence over the base model action.
Missing or non-finite normalized values stop binary execution before sending
that command.

The robot maps the binary decision to its configured
`robots.left.gripper_close_width_m` or `gripper_open_width_m`. Repeated decisions
for the same endpoint keep the command target without restarting the gripper
move. A stale-result hold keeps that target, and TCP manual takeover continues
to use the operator's selected width. Physical receipts record the accepted
endpoint width; the original continuous policy proposal remains available.

Set movement speed and contact-force limit in the robot's `--myq-config`
(default `/home/qccc/Flexiv/flexiv_rdk/example_py/myq/drag_collect.yaml`):

```yaml
gripper_control:
  velocity_m_s: 0.03
  force_limit_n: 25.0
```

`velocity_m_s` sets the speed in m/s. Opening, continuous-width commands and
manual O/C commands use `force_limit_n` in N. An autonomous binary close uses a
fixed 50 N contact-force limit. With `hold_arm_on_close: true` (the default),
at a close transition the client first holds
the measured arm pose and sends the closure off the control thread. It waits
for command delivery and the gripper's `is_moving` to become false, allowing at
least 0.1 seconds after delivery for telemetry to update. Then it automatically
requests fresh inference and resumes at the next handoff. Background inference
continues while the arm is held. Closure frames are scheduling-invalid and do
not enter training. Repeated closed predictions retain the accepted target;
an open prediction allows a later close to trigger another hold. A failed
command or a closure lasting over 30 seconds stops the episode rather than
resuming model motion. This behavior applies to TCP delta and absolute joint
inference. Set `hold_arm_on_close: false` to let model arm commands continue
while the gripper closes asynchronously, still with a 50 N contact-force limit.
This mode keeps normal scheduling and does not reset inference for closure.
The switch only affects autonomous binary closes. H still commands 0.06 m
without gripper initialization.

Validate decoding, RTC coordinates, action rejection and command receipts with:

```bash
.venv/bin/pytest tests/unit_tests/test_online_residual_remote.py -q
```

These tests use mock SDK command receivers and exercise both representations.
Real checkpoint inference can validate shapes and finite output with synthetic
camera frames; task success and motion tracking require a separate robot run.

## Train and save transitions with absolute joint commands

The joint Stage2 inference YAML enables a robot-side automatic Fine request.
The measured left-arm TCP must remain within a 15 mm XY radius of
`(480, 53) mm` and below `138 mm` in the robot base frame for three consecutive
30 Hz control ticks. The request follows the same valid handoff as I and fires
once per episode. Pauses, manual control and maintenance exclude automatic
requests; a new episode rearms the trigger. Set `auto_fine: null` for manual
switching, or adjust its `xy_center_mm`, `xy_radius_mm`, `z_below_mm` and
`consecutive_ticks` fields. I remains available for an earlier manual request.

Use the dedicated joint Stage2 configuration to run the residual Actor, collect
labelled episodes, and train from absolute joint command receipts. Both the
Stage1 base and Stage2 output use seven absolute joint angles in radians plus
one gripper width in metres. Stage2 adds its residual in the frozen model's
normalized action space, then decodes the resulting absolute targets. It does
not add the measured joint state or convert targets into TCP deltas.

Start the server from the RLinf root:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_stage2.yaml
```

Start the matching robot client in another terminal:

```bash
/home/qccc/miniconda3/envs/flexiv/bin/python \
  examples/embodiment/run_flexiv_online_residual_robot.py \
  --inference-config examples/embodiment/config/flexiv_joint_absolute_stage2.yaml
```

The Stage2 deployment matches `flexiv_joint_absolute_inference.yaml`: it loads
`ckpt_1013_12000`, retains the same normalization statistics and dataset Home,
uses a 20-step coarse chunk with requests every 16 frames, and preserves the
joint motion limits and binary gripper settings. It creates an 8D residual Actor. Its zero
initial residual reproduces the joint base policy. The robot starts in the
coarse base phase; the configured TCP region automatically requests the six-step
Stage2 phase at the next valid handoff, or press I to request it manually. Absolute-joint model control multiplies
the configured joint velocity and acceleration limits by
`joint_control.fine_motion_limit_scale` from that handoff onward. The joint Stage2
recipe sets this to `0.3333333333333333`, giving 0.02 rad/s and 0.05 rad/s²
for its 0.06 rad/s and 0.15 rad/s² coarse limits. The scale must be finite and
in `(0, 1]`; omitted settings retain the previous 0.5 scale.
Repeated I requests do not scale them again; a new coarse episode restores the
configured limits. Manual drag and Home retain their myq control parameters.
Only accepted, on-time fine decisions
supply the RLT features used to assemble transitions. Label the stopped episode
with S/F for success/failure; the client stays paused. D explicitly discards
the current unfinished episode and stays paused; Q stops and
stages valid windows for later labelling. Space pauses model control, R starts
manual joint drag after force zeroing, and O/C set the gripper target while
paused or dragging. Space resumes model control after a fresh inference
handoff. While paused, H returns both arms Home, commands a gripper width of
0.06 m with the myq speed and force settings, and zeroes force bias; control
remains paused until Space. Home, gripper opening and pause windows are excluded from
training.
The Stage1-only configuration remains available at
`flexiv_joint_absolute_inference.yaml`, where I does not select an Actor.

The default Stage2 configuration saves episodes under
`results/flexiv_joint_stage2_ckpt1013_12000/episodes` and learner checkpoints under
`results/flexiv_joint_stage2_ckpt1013_12000/checkpoints`. The learner uses the CPU
and starts with 50 labelled base-policy episodes followed by 100 Critic warmup
updates before Actor learning. Enter the fine phase to collect valid residual
transitions during this warmup as well. For inference with an already trained
Actor, add `--no-online-train --actor-checkpoint /path/to/actor.pt` to the server
command. Override `--episode-dir` and
`--checkpoint-dir` to choose different directories. The learner keeps fine
rollouts on the base policy for the first 50 admitted labelled episodes and
100 Critic warmup updates before Actor updates begin. Set
`--base-only-episodes` and `online_residual.critic_warmup_updates` in
`examples/embodiment/config/flexiv_joint_stage2.yaml` for a different warmup.

To collect transitions without optimizer updates, append `--no-online-train`
to the server command. Collection still uses the Stage2 adapter and its RLT
features. To evaluate an existing joint Actor, also pass
`--actor-checkpoint /path/to/joint_actor.pt`. Resume the complete joint learner
with `--resume-stage2 /path/to/latest.pt`. TCP Actors have a different input
and output width and are rejected; train a new joint Actor instead of reusing
their weights. Resuming also checks the normalization and gripper projection
against the current configuration. Flexiv Stage2 now uses the Stage1-normalized
8D proprio in both Torch and APX. Actor checkpoints and saved episodes declare
`proprio_normalization: stage1` and a `normalization_id` derived from the statistics
file. Older Actors trained with raw proprio and bare Actor state dictionaries
lack that contract and are rejected by this deployment; start a new Stage2 run.
Full learner checkpoints written by this version contain the required metadata.

Each finalized file uses transition schema 3 and explicitly records its action
representation, width and physical units:

| Field | Joint Stage2 meaning |
| --- | --- |
| `action_representation` | `joint_absolute` |
| `action_dim` | `8` |
| `physical_action_units` | Seven `rad` entries and one `m` entry |
| `features[*].base_ref` | `[12,8]` normalized absolute joint base reference |
| `features[*].committed` | `[6,8]` normalized final commands from the real committed prefix |
| `transitions[*].exec` | `[12,8]` final absolute commands encoded with the frozen joint statistics |
| `transitions[*].exec_physical` | `[12,8]` accepted absolute joint targets and gripper widths, with terminal padding masked by `valid_len` |
| `transitions[*].proposal_trace` | Normalized controller-projected proposals before human intervention, separate from final commands |

The collector keeps joint receipts unchanged across overlapping decisions,
even when observations include TCP poses. The robot validates joint targets
against SDK limits; the learner preserves those angles and projects only the
gripper. Optional binary gripper projection uses the last committed gripper
target for threshold ties. Both binary and continuous gripper modes apply the
same 1 mm command deadband sequentially from the committed target; changes
within 1e-8 m of the boundary are held to avoid float32 wire roundoff. This
projection applies to Actor, base-reference and target-Critic candidates.
Executed command targets are
stored separately from measured joint positions. Unknown receipts, invalid
windows are excluded. Unconfirmed terminal labels produce pending drafts,
which become Replay data only after explicit confirmation.
The store and replay reject mismatched action contracts; legacy schema 1/2
files remain readable only with matching action shapes.

Validate the joint Actor, collection, storage and resume paths in separate test
processes:

```bash
.venv/bin/pytest tests/unit_tests/test_online_residual_remote.py -q
.venv/bin/pytest tests/unit_tests/test_data.py -q -k 'online or residual or replay or joint_stage2 or joint_training'
.venv/bin/pytest tests/unit_tests/test_models.py -q -k 'stage1 or joint_stage2_actor or online_residual_actor'
```

The command receivers are mock SDK objects; actual robot tracking and insertion
success require a separate hardware run.

## Flexiv intervention and finalized transition records

The single-arm Flexiv action is a physical 6D TCP delta plus a gripper width
in metres. The adapter converts that width to the 0–1 Flexiv command at its
boundary. The command contract is copied locally into
`flexiv/intervention_command.py`; no myq import is needed at runtime.
`LeaderDragInterventionSource` adapts the main-arm admittance and -X/-Y/+Z
alignment from myq's drag collector. It reads the leader's measured pose,
maps it to a follower-world TCP delta, and requires an explicit switch edge
through `set_enabled(True/False)`; leader force alone cannot take over.
`FlexivInterventionDevice` consumes its local `poll(now_ns)` command. An `override`
command replaces the selected arm's six delta dimensions; a `residual` command
adds to the policy delta. An optional gripper value replaces only the gripper
dimension. It is interpreted as physical width in metres by default, matching
the myq collector; sources that publish 0–1 values set
`gripper_unit="normalized"` on the device. `leader`, `follower`, and `both`
selectors are supported, with the
device configured for the one arm it controls. An expired command has no effect.

For this Stage1 checkpoint, every predicted TCP delta is expressed in world
coordinates, metres and radian rotvec, relative to its inference observation's
TCP pose. The executor retains that pose for the entire six-frame execution
block. A leader drag delta is relative to the follower's current measured pose;
the realtime adapter composes its final target and records the equivalent delta
from the block's inference pose. The 7D receipt and Actor actions therefore
share the SFT action frame. The absolute TCP target is sent only to Flexiv.

The deployment entry performs intervention arbitration directly in its 30 Hz
realtime send path. For a separate synchronous Gym workflow, wrapping
`FlexivEnv` in `TeleopIntervention` exposes the same physical command receipt
and per-dimension masks. That workflow must supply 12 consecutive step-info
records for each decision:

```python
from rlinf.algorithms.online_residual_rl import TransitionAssembler
from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
from rlinf.envs.real.flexiv import FlexivEnv, FlexivInterventionDevice
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention

from rlinf.envs.real.flexiv import LeaderDragInterventionSource

follower = FlexivEnv(flexiv_config)
source = LeaderDragInterventionSource.connect(follower, leader_serial)
env = TeleopIntervention(follower, FlexivInterventionDevice(source))
env.reset()  # Configures leader impedance; keep stepping at control_hz.

# Wire the external, explicit button/switch events to these calls:
source.set_enabled(True)   # Take over; first frame re-anchors both TCP poses.
source.set_enabled(False)  # Hand back to policy on the next control frame.
assembler = TransitionAssembler(config)
replay = EpisodeReplay(config, storage_dir=episode_dir)

# At inference: capture the real normalized committed queue separately from
# the Actor's complete, unmodified proposal before any intervention.
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
# env.close() when execution stops; this also stops the leader robot.
```

The local executor now owns explicit console events, 30 Hz scheduling, gripper
control, and transition-window collection. It does not use the Gym wrapper's
sleeping `step()` path: it arbitrates the same intervention device before its
one realtime target send and emits the final physical command receipt there.
The scheduler records that normalized receipt in the active queue slot after
each send, while each inference request retains its immutable pre-intervention
snapshot for replay validation. Torch RTC guidance uses a separate raw PI0.5
model-action queue.
Leader RDK polling runs in a separate worker; the follower pose it reads is a
locked snapshot, not a second follower-RDK call. Type `on` or `off` followed by
Enter at the operator console to take or release control. Type `success` or
`failure` followed by Enter to stop and label the episode. Between episodes,
type `next` followed by Enter to re-arm the robot. `stop` retains valid
windows in the pending-label directory. `LeaderDragInterventionSource` does not connect itself through
the existing `teleop: spacemouse` configuration. Before using real hardware, confirm the
leader/follower serials and world-axis mapping, clear the arms, and validate
the command limits in dummy mode. The leader wrench bias is averaged for one
second by default at start, so keep hands off the arm during `env.reset()`.

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
records. Invalid windows are excluded; valid unlabelled episodes remain in
`pending/` until confirmation.

## Configure optional action smoothness

Keep `online_residual.smooth_weight: 0.0` to leave smoothness disabled. The
three order coefficients are configured separately:

```yaml
online_residual:
  smooth_weight: 0.0
  smooth_weights: [1.0, 1.0, 1.0]  # eta_1, eta_2, eta_3
```

When enabled, concatenate the frozen real committed prefix with the projected
normalized Actor execution. For each first-, second- and third-order difference,
sum squared differences across all seven joint coordinates (six coordinates in
TCP mode), then average valid time positions within each window and average
windows with valid positions. Gripper coordinates are excluded. A difference
is valid only when all of its contributing frames are valid; terminal padding
and gaps contribute nothing. The handoff boundary participates in the loss.

Smoothness uses the original frame-valid mask for relative-Q and BC samples.
BC's separate sampling mask cannot suppress smoothness on the relative-Q pool.
With weight zero, the smoothness calculation and its extra mask are skipped.
The new coefficients default to `(1, 1, 1)` when restoring older training or
Replay checkpoints; explicit incompatible coefficients are rejected on resume.

## Confirm stopped episodes and monitor training

Stop a task without a success/failure label to retain its trustworthy windows
under `<episode_dir>/pending/`. These drafts preserve features, receipts,
correction masks and the last executed frame. They receive no terminal reward
or training credit until confirmation. D is an explicit discard; a lost
connection with an unknown last executed frame cannot create a labelled episode.

List drafts, then confirm one episode using its displayed session ID:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/label_flexiv_episode.py \
  --episode-dir results/flexiv_joint_stage2/episodes
PYTHONPATH=. .venv/bin/python examples/embodiment/label_flexiv_episode.py \
  --episode-dir results/flexiv_joint_stage2/episodes \
  --episode-id SESSION_ID --label success
```

Use `--label failure` for a failed task or `--label discard` to delete a draft.
Confirmation calculates discounted returns and atomically publishes the final
file before deleting its draft. The running learner discovers confirmed files
between updates for submitted episodes; a resumed learner also scans them.
Episode IDs prevent duplicate admission and duplicate G=5 update credits.
Replay remains gated on an explicit boolean label.

Labelled episodes are serialized to bytes before entering the learner's
multiprocessing queue. This keeps shared-memory file descriptor usage from
growing with the number of tensor storages in a long episode. Serialization
errors propagate to the collector before submission. After S/F, check the
`Learner admitted episode` log or `episode/new_windows` metric to confirm that
the episode entered Replay.

Read the persistent journal while training:

```bash
tail -f results/flexiv_joint_stage2/checkpoints/metrics.jsonl
```

Each JSON record includes time, learner PID, Critic/Actor update counts, phase
and scalar metrics. Optimizer records include the existing loss/gradient norms,
Q1/Q2 and TD-target mean/std/min/max/p05/p50/p95. Actor records include
`actor/human_unreachable_fraction`: the fraction of valid human coordinates
whose normalized targets lie beyond the raw Actor's `base_ref ± residual_scale`
BC range. It is zero when the batch contains no human coordinates. Targets are
retained for BC even when unreachable.

Episode records include resident feature/window counts, terminal and relative-Q
windows, human/pair windows and corrected coordinate counts. Collection records
include received decisions/receipts, valid windows, invalid prefixes/features/
windows, decisions with no executed action, unresolved successors and pending
labels. Collection records are also written to the inference log. Only a new
confirmed transition grants update credit; the journal persists metrics from
each actual optimizer update, including Critic-only warmup.

The memory Replay stores observation IDs, successor IDs and correction IDs in
its main rows, with one feature table and one sparse correction table. Physical
command copies and full diagnostic proposal traces stay on disk rather than in
the resident Replay. Sampled
transitions share detached feature copies by ID. Capacity eviction removes
unreferenced records. Checkpoints preserve these tables and can also restore
legacy Replay checkpoints containing full transition objects. TD batch building
reads current z/p, executed commands and returns, then only nonterminal
successor fields. Rank reads current z/p, executed commands and sparse pairs;
Actor updates read reference/committed and BC masks separately.

## Train Stage1 from absolute joint demonstrations

Use a compact single-arm myq recording whose metadata declares
`action_representation: joint_absolute` and `action: [8]`. Record the accepted
7D joint command in radians and gripper command width in metres. The dataset
must also contain 7D measured joints, 1D measured gripper width, 6D measured TCP,
main/wrist RGB bytes, episode lengths and task prompts. The loader verifies the
representation and shapes, builds a 32-frame future chunk within each episode,
and repeats the terminal command for padding. Absolute joint actions pass
through unchanged; measured joints become proprio.

First generate fixed statistics from the exact training chunks:

```bash
PYTHONPATH=. .venv/bin/python toolkits/lerobot/calculate_flexiv_norm_stats.py \
  --dataset /path/to/joint_absolute_demonstrations \
  --output /path/to/flexiv_joint_assets \
  --action-horizon 32 --model-action-dim 32
```

The tool infers the action representation from metadata. Joint actions retain
all eight physical channels and receive 24 zero padding channels. State stays
8D. TCP recordings continue to use the existing current-pose delta conversion.
The output directory must be new so statistics cannot overwrite an existing
training/deployment contract.

Launch the existing RLinf SFT runner with the new joint recipe:

```bash
EMBODIED_PATH="$PWD/examples/sft" PYTHONPATH=. .venv/bin/python \
  examples/sft/train_vla_sft.py \
  --config-name flexiv_joint_rlt_stage1_sft_openpi_pi05 \
  data.train_data_paths=/path/to/joint_absolute_demonstrations \
  actor.model.model_path=/path/to/pi05_base_openpi_rlinf \
  actor.model.openpi_data.norm_stats_path=/path/to/flexiv_joint_assets/norm_stats.json
```

The recipe selects one GPU, seed 42, the 32-frame/8D joint layout and joint
flow-matching plus masked RL Token reconstruction. It uses the existing FSDP
worker, two-layer 2048D Token encoder and parallel-query decoder. Set placement
and batch sizes for your host. This entry writes reproducible RLinf checkpoints;
full training requires the base weights, actual demonstrations and a GPU.

For deployment, copy `flexiv_joint_absolute_stage2.yaml` and set `checkpoint` to
the trained `model_state_dict/full_weights.pt`, `norm_stats` to these assets and
`training_config` to the new SFT recipe. The lightweight
`flexiv_joint_stage1_inference.yaml` remains the architecture recipe for the
provided `joint_abs_1007` checkpoint. Verify the trained checkpoint before a
hardware rollout; synthetic dataset and config tests do not establish policy
performance.

## Flexiv SFT data and statistics

The selected source is
`/home/qccc/Flexiv/flexiv_rdk/example_py/myq/peg_insertion_ok`.
It stores absolute TCP targets. TCP delta computation is deferred until SFT is
enabled. The statistics tool applies that same conversion to future chunks;
inference receives already decoded model actions.

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

## Stage1 checkpoint deployment

The local inference entry loads the Stage1 SFT recipe, its RLinf
`full_weights.pt` checkpoint, and the fixed normalization statistics before
starting Flexiv. The Stage1 recipe in this repository trains an 8D state and
32-step, 7D TCP-delta action. The original `Online残差RL.md` joint-absolute
example uses 8D actions; deploying that representation requires a separately
trained matching checkpoint. The current command follows the actual Flexiv
Stage1 recipe.

Set the real robot serials, arm, camera serials, task prompt, units, frame, and
gripper ranges in `examples/embodiment/config/flexiv_pi05_rtc.yaml`. Set the
Stage1 `openpi_data.norm_stats_path` to the statistics used for training. The
entry first looks for `flexiv_delta_assets/norm_stats.json` beside the checkpoint's
`model_state_dict` directory, then uses the Stage1 recipe path. Pass
`--norm-stats /path/to/norm_stats.json` to select another file explicitly. A
dummy robot exercises the same scheduler without sending hardware commands:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --backend torch \
  --checkpoint /path/to/global_step_N \
  --num-flow-steps 4 --dummy-robot --max-ticks 12
```

Pass `--actor-checkpoint /path/to/actor.pt` to load a trained residual Actor.
When omitted without `--online-train`, the entry uses Stage1-only inference.
The deployment default is `--num-flow-steps 10`, while the Flexiv Stage1 SFT
recipe sets `actor.model.openpi.num_steps: 4`. For a latency-sensitive Torch
run, explicitly pass `--num-flow-steps 4` after validating its action quality
and timing with the trained checkpoint. Fewer steps change the sampled action;
they are not a numerically equivalent acceleration. Torch RTC sampling reuses
the Stage1 prefix cache and avoids a per-step GPU-to-CPU time read, but a
10-step request may still exceed the 200 ms six-tick handoff budget on the
target GPU. The displayed handoff frequency is capped at 5 Hz by the 30 Hz
control loop and six-frame horizon, even if a request finishes faster. It is
not the model's standalone inference throughput. Measure a dummy rollout
before enabling the real robot.
The Torch backend applies the remaining previous PI0.5 sampled raw normalized
model actions, including the 32D padded action width, through RTC overlap guidance
while sampling the next Stage1 reference. Its prefix cache also feeds the RLT
Encoder. Final command receipts are normalized separately for replay. The APX L1
binding currently accepts image, text, and noise but no committed-action
guidance; APX therefore samples an unconditioned reference. The collector
replaces the stored committed prefix with final accepted command receipts
before constructing a Critic row; when the Actor is active, both backends
execute only its new six-action half. Do not treat APX and Torch base samples
as numerically identical.
For APX, export the **same Stage1 base weights** to an APX PI0.5 checkpoint,
using `sft_to_apxinf` in the OpenPI checkpoint converter, then run with
`--backend apx --apx-checkpoint /path/to/model.safetensors`.
The RLinf `full_weights.pt` path remains `--checkpoint`; it supplies the RLT
Encoder and Stage1 transform metadata. The export is a weight-layout conversion;
numeric parity with Torch RTC guidance is not implied because APX does not
apply that guidance.
The in-process APX path needs the CUDA `apxinf_py` binding built for the target
GPU. From the RLinf root, install Rust and `maturin`, then build the vendored
binding with `--features cuda`; `APXinf-robo/README.md` lists the build and
import checks. Installing only the `apxinf` Python frontend is insufficient.
The native APX model is thread-bound. The entry loads it on the dedicated
inference worker, performs bootstrap and later requests on that same worker,
and closes it there after the robot has stopped. Normal control ticks do not
wait for APX inference; only startup and shutdown may wait.
APX omits the masked third camera for faster VLA inference. Before passing its
prefix to the Stage1 RLT Encoder, the adapter restores that masked camera slot
so the language-token positional indices match the two-camera SFT layout.
The Flexiv Actor receives the Stage1-normalized first eight state dimensions
in both backends. Its normalized output is zero-padded to the model's 32D
action width before the OpenPI output transform, matching the Torch path.

The older in-process `ResidualHandoffRuntime` stops safely if a six-frame
handoff is missed. The split robot runtime described above instead holds and
requests recovery; use that path for manual coarse-to-fine switching and joint
control. Both collectors retain healthy windows and final labels.

At control frame `t=0`, startup inference supplies an accepted six-command
queue. The control thread snapshots observation and that committed queue, then
submits one background request. It sends one projected TCP delta and gripper
command on each absolute 30 Hz deadline. At `t+6`, it installs a completed
result and submits the next request. Early results wait for that frame. If
a result misses the boundary, the in-process executor discards it and stops
robot motion. The inference thread never reads or commands Flexiv. It
also decodes, projects, and normalizes the next queue before
completion, so the control thread does no GPU readback at the handoff. With
`--online-train`, each receipt and accepted decision is queued for
`EpisodeCollector`. It normalizes final accepted commands using frozen Stage1
statistics, waits for the `t+12` successor and submits only a fully labelled
episode to `OnlineLearnerThread`. The Learner performs finite G/D updates and
publishes Actor snapshots through a latest-only mailbox. The control thread
never loads weights or waits for the Learner. Torch RTC guidance uses the prior
PI0.5 raw sample, not the planned physical committed commands, so later human
corrections do not invalidate the row. APX base inference is unconditioned and
also retains the corrected row.

Run attended episodes with `--episodes N`, or resume Replay, optimizers, credit,
RNG and Actor version from `latest.pt` in a later invocation:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --backend apx --checkpoint /path/to/stage1/global_step_N \
  --apx-checkpoint /path/to/apx/model.safetensors \
  --online-train --episode-dir /path/to/episodes \
  --checkpoint-dir /path/to/stage2 --operator-console --episodes 50 \
  --leader-serial REPLACE_WITH_MAIN_ARM_SERIAL

PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --backend apx --checkpoint /path/to/stage1/global_step_N \
  --apx-checkpoint /path/to/apx/model.safetensors \
  --online-train --episode-dir /path/to/episodes \
  --checkpoint-dir /path/to/stage2 --resume-stage2 /path/to/stage2/latest.pt \
  --operator-console --leader-serial REPLACE_WITH_MAIN_ARM_SERIAL
```

For a no-hardware smoke test add `--dummy-robot --max-ticks 18
--terminal-label success` and omit `--leader-serial --operator-console`. The
entry does not compare APX and Torch numerically. APX accelerates only the PI0.5 VLA; its flow sampling
does not condition on committed actions. RLT and Actor remain PyTorch, and the
two Critics run in the Learner thread.

With the local Stage1 checkpoint and its APX export, the concrete dummy run is:

```bash
PYTHONPATH=. .venv/bin/python examples/embodiment/run_flexiv_online_residual.py \
  --backend apx \
  --checkpoint /home/qccc/checkpoints/myq/model_state_dict/full_weights.pt \
  --apx-checkpoint /home/qccc/checkpoints/myq/apxinf_stage1/model.safetensors \
  --dummy-robot --max-ticks 18 --terminal-label success
```

This checks loading, the APX→RLT→Actor pipeline and timed dummy commands. It
does not establish numeric equivalence to Torch RTC or safe real-robot timing.

### Diagnose Periodic Joint Motion

Use the split server and robot logs to distinguish late inference from a target
change at an on-time handoff. With `base_chunk_steps: 22`,
`base_request_steps: 18`, and 30 Hz control, the first handoff is at tick 22;
later handoffs are 18 ticks (0.6 seconds) apart. Each request has four ticks
(133 ms) before its handoff. `response_ms` measures when the result arrives;
`handoff_ms` includes waiting for the planned boundary.

For absolute joint control, `joint handoff` reports `late_ticks`, the largest
commanded joint step in degrees (`max_target_step_deg`), and the change from the
previous step (`max_target_step_change_deg`). These describe command continuity,
not measured robot velocity or acceleration. Large values with `late_ticks=0`
indicate an on-time switch with a discontinuous target or target slope.
`joint gripper target` reports width changes so gripper reversals can be compared
with motion at the grasp point.

Compare slow and fast inference under the same schedule. A response that misses
the boundary changes execution: the runtime holds the last target and enters
the new prediction at a later index. For example, at 30 Hz with 22/18 scheduling,
a 300 ms response can arrive at request index 9 rather than the planned index 4,
after about five held ticks. `coarse chunk entry` reports the actual
`first_prediction_index` and `held_ticks`. The hold gives the joint controller
time to approach its last target; whether it reduces physical motion at the
next switch depends on actual tracking. A smoother slow run does not establish
that the fast run's inference or transport is incorrect.

For a joint adapter that returns `joint_target_rad`, `joint request tracking`
compares the last sent target with the measured joint state supplied to the next
inference request. `max_command_tracking_error_deg` reports their largest
absolute difference. This diagnostic uses the existing observation and adds no
SDK read. Compare it with the handoff target step to investigate whether the
model replans from a measured state that trails the previous commanded plan.

On the server, `inference RTC` reports whether the request has prior actions
(`conditioned`) and their remaining length (`overlap_steps`). For coarse absolute
joint actions, a late acceptance retains the raw prediction and its original frame;
the next request selects its remaining tail using the new observation frame.
`RTC joint context retained after late handoff` reports this case. TCP delta
actions and fine-phase actions still clear context after a late acceptance and emit
`RTC context cleared after late handoff`. Even with context, RTC guidance
does not enforce exact equality at the handoff or constrain joint derivatives.
The runtime continues sampling and sending targets until an operator stop or
the tick limit, including while the robot is near a grasp pose.

To distinguish a changed prediction from normal motion, inspect the server's
`joint prediction continuity` log. `max_same_time_change_deg` compares the old
and new predictions for the same planned handoff frame, rather than two
consecutive motion frames. At K=22 and R=18, it compares old index 22 with new
index 4. A large value on an on-time request identifies a disagreement between
plans; RTC's soft overlap guidance does not guarantee equality at that index.
The command targets pass through unchanged, so this diagnostic exposes the
disagreement without adding filtering or blending.

Keep `num_flow_steps: 10` explicit in the joint inference YAML when comparing
latency. CUDA graph replay for unconditioned requests prevents a late handoff
from switching subsequent inference to the slower eager suffix. The planned
K=22, R=18 handoff still waits four control ticks from the request; reducing model
latency increases that deadline's margin, rather than changing the action frame
to which the prediction belongs. Use `--disable-torch-cuda-graph` for an eager
comparison of the unconditioned path; Jacobian-guided steps continue to use
compiled autograd.

For a shorter open-loop execution window, compare the current schedule with
`--base-chunk-steps 16 --base-request-steps 12` on the server. Both schedules
provide four ticks for inference, while 16/12 switches predictions every
0.4 seconds instead of 0.6 seconds. The robot receives these indices in the
server handshake. This changes prediction freshness and the RTC overlap; it
does not interpolate targets. On three recorded observation times and three
fixed seeds with the joint checkpoint, the median handoff target step decreased
from 1.38 degrees at 22/18 to 0.77 degrees at 16/12. Validate the complete task
before adopting the shorter schedule; this offline comparison does not measure
actual robot motion or task success.

The OpenPI espresso server and station client provide a useful comparison for
chunk alignment. Their RTC mode installs a ready result immediately and skips
the number of actions actually consumed since the request, including actions
consumed during response transport. Their VLSH mode stages the result until a
planned boundary and consumes the unused old tail if inference is late. This
runtime stages results until its planned boundary and skips expired actions on
a late switch, but holds the last target while awaiting that result. Normal
22/18 switches use old index 21 followed by new index 4: these represent adjacent
global frames, rather than a repeated or omitted action. Advancing the handoff
without changing the returned slice would send future actions too early.

The reference RTC sampler also uses exponential overlap weights, while this
sampler uses its existing soft mask. A controlled comparison with the same
joint checkpoint, recorded inputs, old predictions, and seeds reduced the median
22/18 target step only from 1.38 to 1.36 degrees; the joint prediction error
increased from 0.242 to 0.260 degrees. That comparison used exponential weights
only where a real old prediction exists. The reference implementation also
guides zero-padded positions beyond that tail. The weight change was not retained:
this evidence does not establish it as a remedy for physical jitter.


### Human supervision and timeout recovery

`human_mask` marks only dimensions owned by the human that differ from the
controller-projected policy proposal: TCP translation above 0.1 mm, TCP rotation
or joint angles above 0.001 rad, and gripper changes above 1 mm. Gripper ownership
persists after O/C until the episode ends. Merely entering manual mode, unchanged
coordinates, binary projection and controller limits do not create human labels.
`pair_mask` is a subset of `human_mask` with reliable pre-intervention proposals;
the assembler clears its historical committed half and terminal padding. Human BC
can use valid historical corrections, whereas rank only uses new execution
corrections. An absent pair mask does not implicitly create preference labels.

The loss retains separately normalized ordinary BC and human BC with coefficients
5 and 25. Human samples supervise only `human_mask` positions; no BC is applied
to their uncorrected coordinates. Rank reconstructs its comparison by replacing
only trustworthy `pair_mask` coordinates of the real executed commands. Missing,
nonfinite or malformed proposal coordinates clear the corresponding pair while
preserving valid TD and human BC. Invalid physical receipts or features still
exclude their windows.

Set `--task-timeout-seconds 120` on the robot client, or `task_timeout_seconds: 120`
in the shared inference YAML, to stop an episode and label failure at the task
time limit. This clock starts after Home and inference warmup. `--max-ticks`
remains a debugging stop without an inferred label. S/F/D/Q controls remain
explicit. In both action representations, Space pauses or resumes model
control, and R enables manual control after stationary force zeroing.
The task timeout can be left unset for operator-labelled runs.

Coarse scheduling is unchanged until the configured automatic trigger or I
requests the switch, then the next valid coarse handoff bridges into six-frame
fine requests and handoffs. RTC committed actions remain the original normalized
Stage1 prediction; real committed receipts used by training are stored separately.

The manual controller reads `control.follower_position_scale` (three positive
axis scales), `control.xy_motion_direction`, and optional
`control.follower_orientation_max_deviation_deg` from the myq YAML. Orientation
limits are measured relative to the first manual takeover pose and reset after
Home. `recording.force_bias_before_manual_seconds` controls R-key zeroing and
falls back to `recording.force_bias_after_home_seconds` (default 0.5 seconds).
O/C use the configured follower widths, speed and force through the asynchronous
gripper worker.

RL transition storage keeps frozen features, normalized proprio, base/committed
predictions, final command receipts, rewards and sparse human correction data;
it does not store images. Decision and receipt frame indices define temporal
alignment. Receipts must form a consecutive window from the decision, and the
successor must be at the configured TD horizon. Missing, reordered or late
receipts invalidate the affected window. No camera-time interpolation is used
for training actions. Robot receipts also expose monotonic command start/return
timestamps for timing diagnostics; these are not cross-host wall-clock times
and are not additional training inputs.


Stage2 supports consecutive labelled episodes over one robot connection.
After S/F/D, the robot holds its measured pose while the server finalizes the
current episode asynchronously. Coarse base inference continues on fresh observations while the robot stays
paused, without executing those predictions or adding receipts or transitions
to the closed episode. These paused previews use the configured coarse request
cadence. Press Space to start a new episode from the current pose. Optionally,
press H first to return both arms Home, command the gripper to 0.06 m and zero
force bias; wait for that sequence to complete before pressing Space. The
new episode gets a distinct ID, resets its frame index to zero, clears old
Actor proposals and RTC context, and resumes the configured coarse schedule.
Wait for the automatic trigger or press I to enable fine requests and RL transition collection. Replay,
Learner warmup counts, Actor weights and pending training updates persist across
episodes. The task timer restarts when the new bootstrap is ready; `--max-ticks`
counts control cycles across the whole client run. D discards only the current
episode without saving or staging transitions and adds no Learner training credit.
It supports the same H/Space restart flow as S/F. Q stops the client; already
labelled episodes remain saved. Stage1-only S/F/D behavior remains an exit.
Restart both processes once to enable the new handshake capability.


An I request arriving while the next episode's start reply or bootstrap is
pending is retained and applied at the first valid coarse handoff. The client
reports that the request was queued and later prints `robot entered fine phase`.
I pressed while paused or dragging reports that it was not applied; after S/F/D,
press Space to start the next episode before requesting Fine collection.
An episode with zero valid windows prints `No RL transitions saved` and does
not count toward the initial episode threshold. Its scheduling summary reports
coarse/Fine requests, accepted Fine decisions and discarded results, so missing
Fine entry can be distinguished from inference that repeatedly misses the budget.

After startup gripper initialization with opening enabled, the robot client
waits for measured width within 1 mm of the configured open target and stopped
motion. A timeout aborts startup before model actions execute. Explicitly
skipping initialization or disabling open_on_start retains its existing behavior.

Resuming with Space after H returns the active episode to the coarse base
phase on both the robot and inference server, restores configured joint motion
limits and clears old actions and RTC context. The episode ID and collected
data remain intact. Fine can be requested again manually or by the configured
automatic trigger. A plain Space pause/resume retains the current phase.

Labelled episodes submitted to the learner process are atomically persisted by
collection before enqueueing. A temporary Replay assigns terminal rewards using
the same admission contract; it retains no parent-side training data. The worker
services incoming events between individual optimizer steps, so the initial
training-credit backlog does not starve subsequent episodes. Existing disk scans
and queued redeliveries remain deduplicated and add update credit only once.

The learner saves full `latest.pt` state on episode admission, approximately every
10 seconds at update boundaries while training progresses, and on normal close.
Close retains the existing behavior of draining accepted finite update credit.
Worker errors and unexpected exits are reported immediately and appended to
`learner_error.log`, including PID and exit code when available. Atomic episode
files survive a learner failure even if their queue events were not consumed.
A forced exit can still lose optimizer updates since the last checkpoint.

Resume the server with `--resume-stage2 <full-checkpoint.pt>` and the same episode
and checkpoint directories, then launch the robot client with the same inference
config. Full resume restores optimizers, Replay, update credit and warmup state;
`--actor-checkpoint` alone restores only Actor weights. Choose the newest valid
full checkpoint when recovering an older run whose `latest.pt` is stale. The
robot starts coarse and uses the restored residual Actor only in Fine. Deleting
an episode file does not retract samples already held by Replay or a checkpoint.
