Flexiv PI0.5 SFT and RTC Inference
==================================

This page explains how to train the 2-camera, 8D-state Flexiv PI0.5 contract in RLinf and deploy a standard OpenPI/JAX SFT checkpoint with asynchronous RTC inference. The deployment path covers ordinary speed-field guidance only; it does not invoke APXinf, RL Token, a residual Actor, a Critic, or online training.

Prepare the robot host
----------------------

The RLinf installer creates the OpenPI environment and installs the RealSense dependency. Set ``OPENPI_RTC_PATH`` to the audited OpenPI checkout that implements ``return_raw_actions`` and ordinary speed-field sampling; the published ``rlinf-openpi==0.1.1`` wheel does not contain those APIs. Flexiv RDK remains a vendor host dependency, so ``import flexivrdk`` must work in the selected environment before the robot is activated.

.. code-block:: bash

   export OPENPI_RTC_PATH=/absolute/path/to/rtc-compatible/openpi
   bash requirements/install.sh embodied --venv openpi --model openpi --env flexiv
   source switch_env openpi
   python -c "import flexivrdk, pyrealsense2"

Train the standard SFT contract
-------------------------------

The training config consumes compact Flexiv demonstrations, builds 12-step chunks, converts stored absolute TCP targets to world-frame deltas from the sample's measured TCP pose, and trains 7D outputs without RL Token.

.. code-block:: bash

   bash examples/sft/run_vla_sft.sh flexiv_sft_openpi_pi05

Before launching, replace ``data.train_data_paths``, ``actor.model.model_path``, and ``actor.model.openpi_data.norm_stats_path`` in ``examples/sft/config/flexiv_sft_openpi_pi05.yaml``. The RLinf FSDP trainer writes PyTorch training checkpoints. The RTC entry below intentionally accepts only a standard OpenPI/JAX checkpoint containing ``params/`` and matching ``assets/``; exporting an RLinf FSDP checkpoint to that format is a separate step and is not performed by the entry script.

Configure deployment semantics
------------------------------

Copy ``examples/embodiment/config/flexiv_pi05_rtc.yaml`` and replace the checkpoint, robot serial, arm side, two camera serials, prompt, control frequency, units, reference frame, limits, and gripper ranges. The compact dataset stores gripper state and commands as opening width in metres, so the example uses ``[0.003, 0.060]`` for both ranges. ``policy.camera_keys`` maps the Flexiv ``main`` and ``wrist`` frames to the exact keys used by training. The policy, horizon, output dimension, and physical semantics are validated before ``FlexivEnv`` opens hardware resources.

Run inference with the completed config:

.. code-block:: bash

   python examples/embodiment/run_flexiv_pi05_rtc.py \
       --config /absolute/path/to/flexiv_pi05_rtc.yaml

Control and inference timing
----------------------------

The first inference blocks before steady control and carries no previous-action guidance. During steady control, the owner thread polls a completed inference, snapshots cached camera frames and robot state, polls again, submits at most one immutable request, polls once more, selects one physical delta, sends the final TCP target and independent gripper command, commits the tick, and waits for an absolute monotonic deadline. The extra non-blocking polls let a result completed during observation capture or submission switch in during the same tick.

The inference worker never calls the robot and never changes the cursor. A request records its session, generation, observation tick, submit tick, committed tick, and old cursor. At acceptance, ``stale_prefix`` is the number of control ticks committed since submission. Those entries are discarded before the new chunk is installed. The next RTC request uses the unexecuted tail of the previous normalized raw chunk; only the postprocessed physical chunk reaches the robot.

Each physical delta is relative to the measured TCP pose stored with that chunk's observation. The adapter converts the selected delta into a final TCP target and overwrites the previous target, so later rows in a chunk are not accumulated. A hold resends the final target without applying any delta again. Flexiv NRT arm commands and the latest-command gripper worker do not wait in the control loop, while background camera threads keep the latest complete RGB frame for each view. A normal control tick therefore performs no policy wait, gripper ``Move`` wait, camera ``wait_for_frames``, or relative-period sleep.

Validate without hardware
-------------------------

The focused tests use a gated fake policy, fake robot, and controllable clock to cover RTC guidance, zero/one/multi-tick inference delay, exact chunk handoff, one in-flight request, generation reset, hold behavior, absolute deadlines, TCP composition, gripper independence, and idempotent cleanup.

.. code-block:: bash

   pytest -q tests/unit_tests/test_openpi_rtc.py
   pytest -q tests/unit_tests/test_real_env.py -k flexiv

The tests do not validate a GPU checkpoint, camera serials, the selected arm, gripper calibration, or robot workspace limits. Confirm those values at low speed before the first hardware run.
