# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Datasets, batching, and the on-the-wire shapes they produce."""

import copy
import dataclasses
import json
import random
import time
from unittest import mock

import pytest
import torch
from omegaconf import DictConfig

from rlinf.algorithms.online_residual_rl import (
    FeatureRecord,
    OnlineResidualRLConfig,
    TransitionAssembler,
    UpdateBudget,
)
from rlinf.algorithms.online_residual_rl.learner import OnlineResidualLearner
from rlinf.data.datasets.reasoning.dataset import ReasoningDataset
from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.data.storage.lerobot import add_frame_to_dataset, episode_boundaries
from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
from rlinf.utils.nested_dict_process import split_dict_to_chunk
from rlinf.utils.obs_compression import (
    _CODEC_KEY,
    compress_obs,
    decompress_obs,
    infer_obs_batch_size,
    is_compressed_image,
    is_compression_enabled,
)


def _online_feature(
    config: OnlineResidualRLConfig, obs_id: str, frame_index: int
) -> FeatureRecord:
    return FeatureRecord(
        obs_id=obs_id,
        z=torch.zeros(config.z_dim),
        proprio=torch.zeros(config.proprio_dim),
        base_reference=torch.zeros(config.actor_horizon, config.action_dim),
        committed=torch.zeros(config.inference_horizon, config.action_dim),
        episode_id="episode-0",
        frame_index=frame_index,
        model_version=0,
        config_version="test",
    )


def test_online_transition_human_bc_does_not_imply_trustworthy_rank_pair():
    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    assembler = TransitionAssembler(config)
    feature = _online_feature(config, "obs-0", 0)
    next_feature = _online_feature(config, "obs-12", config.td_horizon)
    proposal = torch.zeros(config.actor_horizon, config.action_dim)
    executed = proposal.clone()
    human_mask = torch.zeros_like(proposal, dtype=torch.bool)
    human_mask[7, :6] = True
    executed[human_mask] = 0.25
    assembler.begin(feature, proposal)

    transition = assembler.complete(
        feature.obs_id,
        executed=executed,
        rewards=torch.zeros(config.td_horizon),
        terminal=False,
        valid_len=config.td_horizon,
        next_feature=next_feature,
        human_mask=human_mask,
    )

    assert transition.correction is not None
    assert torch.equal(transition.correction.human_mask, human_mask)
    assert not transition.correction.pair_mask.any()
    torch.testing.assert_close(transition.correction.proposal(executed), executed)
    assert transition.relative_q_eligible


def test_online_receipts_persist_real_action_and_sparse_intervention(tmp_path):
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    feature = _online_feature(config, "obs-0", 0)
    feature = dataclasses.replace(
        feature,
        committed=torch.full_like(feature.committed, 0.2),
    )
    proposal = torch.full((config.actor_horizon, config.action_dim), 0.3)
    assembler = TransitionAssembler(config)
    assembler.begin(feature, proposal)
    infos = []
    for frame in range(config.td_horizon):
        physical = torch.full((config.action_dim,), 0.1 if frame < 6 else 0.15)
        mask = torch.zeros(config.action_dim, dtype=torch.bool)
        if frame == 7:
            physical[:6] = 0.25
            mask[:6] = True
        infos.append(
            {
                "frame_index": frame,
                "action_receipt": torch.full((config.action_dim,), -99.0),
                "action_receipt_physical": physical,
                "policy_proposal_physical": torch.full((config.action_dim,), 0.4),
                "policy_proposal_valid": True,
                "human_mask": mask,
                "pair_mask": mask.clone(),
            }
        )
    row = assembler.complete_from_receipts(
        feature.obs_id,
        step_infos=infos,
        encode_actions=lambda actions: actions * 2,
        terminal=True,
        next_feature=None,
    )
    torch.testing.assert_close(row.executed[:6], feature.committed)
    torch.testing.assert_close(row.executed[7, :6], torch.full((6,), 0.5))
    assert row.correction.human_mask[7, :6].all()
    assert not row.correction.pair_mask[:6].any()
    torch.testing.assert_close(row.correction.proposal_values, torch.full((6,), 0.8))
    torch.testing.assert_close(
        row.proposal_trace,
        torch.full((config.td_horizon, config.action_dim), 0.8),
    )

    replay = EpisodeReplay(config, storage_dir=tmp_path)
    assert replay.commit_episode([row], success=True, terminal_frame=11) == 1
    saved = replay.store.load_episode(feature.episode_id)
    assert set(saved["features"]) == {feature.obs_id}
    assert saved["transitions"][0]["obs_id"] == feature.obs_id
    assert saved["transitions"][0]["correction_id"] == feature.obs_id
    assert saved["schema_version"] == 3
    torch.testing.assert_close(saved["transitions"][0]["exec"], row.executed)
    torch.testing.assert_close(
        saved["transitions"][0]["proposal_trace"], row.proposal_trace
    )
    torch.testing.assert_close(
        saved["corrections"][feature.obs_id]["proposal_values"],
        row.correction.proposal_values,
    )
    assert saved["transitions"][0]["return"].item() == pytest.approx(0.99**11)
    assert len(list(tmp_path.glob("*.pt"))) == 1


def test_online_label_preserves_healthy_fragment_without_inventing_terminal_reward():
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    feature = _online_feature(config, "obs-0", 0)
    assembler = TransitionAssembler(config)
    assembler.begin(feature, torch.zeros(config.actor_horizon, config.action_dim))
    row = assembler.complete(
        feature.obs_id,
        executed=torch.zeros(config.td_horizon, config.action_dim),
        rewards=torch.zeros(config.td_horizon),
        terminal=False,
        valid_len=config.td_horizon,
        next_feature=_online_feature(config, "obs-12", 12),
    )
    replay = EpisodeReplay(config)
    assert replay.commit_episode([row], success=True, terminal_frame=20) == 1
    saved = replay.transitions()[0]
    assert not saved.terminal and not saved.rewards.any()
    assert len(replay) == 1


def test_online_receipts_reject_missing_frame_or_committed_mismatch():
    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    feature = _online_feature(config, "obs-0", 0)
    proposal = torch.zeros(config.actor_horizon, config.action_dim)
    assembler = TransitionAssembler(config)
    assembler.begin(feature, proposal)
    infos = [
        {"frame_index": i, "action_receipt": torch.zeros(config.action_dim)}
        for i in range(config.td_horizon)
    ]
    infos[4]["frame_index"] = 5
    with pytest.raises(ValueError, match="consecutive"):
        assembler.complete_from_receipts(
            feature.obs_id,
            step_infos=infos,
            encode_actions=lambda values: values,
            terminal=True,
            next_feature=None,
        )
    infos[4]["frame_index"] = 4
    infos[0]["action_receipt_physical"] = torch.ones(config.action_dim)
    with pytest.raises(ValueError, match="committed prefix"):
        assembler.complete_from_receipts(
            feature.obs_id,
            step_infos=infos,
            encode_actions=lambda values: values,
            terminal=True,
            next_feature=None,
        )


def test_online_update_budget_stops_without_new_transitions():
    budget = UpdateBudget(critic_per_transition=5, critic_per_actor=5)
    budget.add_transitions(2)

    critic_updates = 0
    actor_updates = 0
    while budget.take_critic():
        critic_updates += 1
        actor_updates += int(budget.take_actor())

    assert (critic_updates, actor_updates) == (10, 2)
    assert not budget.take_critic()


def test_online_training_keeps_base_policy_through_episode_and_critic_warmup():
    from types import SimpleNamespace

    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    training = OnlineTraining(
        SimpleNamespace(config=config),
        EpisodeReplay(config),
        initial_episodes=2,
        warmup_updates=3,
    )
    assert training.rollout_mode == "base"
    training.episodes = 2
    assert training.rollout_mode == "base"
    training.warmup_remaining = 0
    assert training.rollout_mode == "rollout"


def test_online_replay_rejects_invalid_schedule_and_unexecuted_decision():
    config = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    assembler = TransitionAssembler(config)
    feature = _online_feature(config, "pending", 0)
    action = torch.zeros(config.actor_horizon, config.action_dim)
    assembler.begin(feature, action)
    kwargs = {
        "executed": action,
        "rewards": torch.zeros(config.td_horizon),
        "terminal": True,
        "next_feature": None,
    }
    with pytest.raises(ValueError, match="Scheduling-invalid"):
        assembler.complete("pending", valid_len=12, schedule_valid=False, **kwargs)
    with pytest.raises(ValueError, match="execution must have started"):
        assembler.complete("pending", valid_len=6, **kwargs)
    row = assembler.complete(
        "pending",
        valid_len=8,
        human_mask=torch.ones_like(action, dtype=torch.bool),
        **kwargs,
    )
    assert row.correction.human_mask[:8].all()
    assert not row.correction.human_mask[8:].any()
    assert not row.correction.pair_mask[:6].any()
    assert not row.correction.pair_mask.any()


def test_online_bc_averages_samples_not_intervention_elements():
    from rlinf.algorithms.online_residual_rl.losses import behavior_cloning_losses

    actions = torch.zeros(2, 12, 7)
    targets = torch.ones_like(actions)
    targets[1] = 3
    human = torch.zeros_like(actions, dtype=torch.bool)
    human[0, 6, 0] = True
    human[1, 6:, :] = True
    ordinary, corrected = behavior_cloning_losses(
        actions, actions, targets, valid_mask=torch.ones_like(human), human_mask=human
    )
    assert ordinary.item() == 0
    assert corrected.item() == 5  # Mean of per-sample MSEs 1 and 9.


def test_online_residual_learner_updates_actor_and_critics():
    config = OnlineResidualRLConfig(
        z_dim=4,
        hidden_dim=8,
        inference_horizon=2,
        actor_horizon=4,
        td_horizon=4,
        critic_updates_per_transition=1,
        critic_updates_per_actor_update=1,
    )
    learner = OnlineResidualLearner(config, action_projector=lambda value: value)
    feature = _online_feature(config, "row", 0)
    assembler = TransitionAssembler(config)
    proposal = torch.zeros(config.actor_horizon, config.action_dim)
    assembler.begin(feature, proposal)
    transition = assembler.complete(
        feature.obs_id,
        executed=torch.zeros_like(proposal),
        rewards=torch.ones(config.td_horizon),
        terminal=True,
        valid_len=config.td_horizon,
        next_feature=None,
        human_mask=torch.zeros_like(proposal, dtype=torch.bool),
        pair_mask=torch.zeros_like(proposal, dtype=torch.bool),
    )
    critic_metrics = learner.critic_step([transition])
    actor_metrics = learner.actor_step([transition])
    assert {"critic/td", "critic/rank"} <= set(critic_metrics.values)
    assert {"actor/total", "actor/ordinary_bc"} <= set(actor_metrics.values)
    assert all(torch.isfinite(value).all() for value in learner.actor.parameters())


def test_online_replay_training_credit_and_checkpoint(tmp_path):
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    assembler = TransitionAssembler(cfg)
    feature = _online_feature(cfg, "terminal", 0)
    actions = torch.zeros(cfg.actor_horizon, cfg.action_dim)
    assembler.begin(feature, actions)
    row = assembler.complete(
        feature.obs_id,
        executed=actions,
        rewards=torch.zeros(cfg.td_horizon),
        terminal=True,
        valid_len=12,
        next_feature=None,
    )
    replay = EpisodeReplay(cfg)
    learner = OnlineResidualLearner(cfg, action_projector=lambda value: value)
    loop = OnlineTraining(
        learner,
        replay,
        initial_episodes=1,
        warmup_updates=0,
        td_batch_size=2,
        supervision_batch_size=2,
    )
    assert loop.commit_episode([row], success=True, terminal_frame=11) == 1
    assert loop.commit_episode([row], success=True, terminal_frame=11) == 0
    saved_row = replay.sample(1)[0]
    assert saved_row.rewards.sum().item() == 1
    torch.testing.assert_close(saved_row.discounted_return, torch.tensor([0.99**11]))
    # A terminal-only batch must not invoke either successor network.
    with mock.patch.object(
        learner.target_actor, "forward", side_effect=AssertionError("bootstrap")
    ):
        for _ in range(5):
            assert "critic/td" in loop.step()
    assert loop.actor_updates == 1
    assert loop.step() == {}
    checkpoint = tmp_path / "resume.pt"
    loop.save(checkpoint)
    restored = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=lambda value: value),
        EpisodeReplay(cfg),
        initial_episodes=1,
        warmup_updates=0,
    )
    restored.load(checkpoint)
    assert restored.actor_updates == 1
    assert restored.step() == {}
    assert restored.commit_episode([row], success=True, terminal_frame=11) == 0


def test_online_fixed_handoff_rejects_early_switch_and_late_result():
    from rlinf.algorithms.online_residual_rl.scheduling import HandoffSchedule

    config = OnlineResidualRLConfig()
    schedule = HandoffSchedule(config)
    queue = torch.zeros(6, 7)
    schedule.recover(queue)
    first = schedule.begin(0, 4)
    queue.fill_(99)
    assert first.committed.count_nonzero() == 0
    assert schedule.submit(first.sequence, torch.ones(6, 7), completed_frame=3)
    with pytest.raises(ValueError, match="Early"):
        schedule.handoff(3)
    torch.testing.assert_close(schedule.handoff(6), torch.ones(6, 7))
    second = schedule.begin(6, 5)
    assert not schedule.submit(second.sequence, torch.zeros(6, 7), completed_frame=13)
    assert schedule.handoff(12) is None
    assert schedule.recovering
    assert not schedule.submit(first.sequence, torch.ones(6, 7), completed_frame=12)


@pytest.mark.parametrize("release_early,expected_code", [(True, 0), (False, 1)])
def test_online_residual_executor_handoffs_without_waiting_for_inference(
    release_early, expected_code
):
    import threading

    import numpy as np

    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )
    from rlinf.models.embodiment.openpi.rtc.types import PolicyObservation

    gate = threading.Event()
    owner_thread = threading.get_ident()

    class Policy:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def predict_action_batch(self, obs, *, mode, committed):
            assert mode == "eval" and obs["states"].shape == (1, 8)
            self.calls += 1
            if committed is not None:
                assert threading.get_ident() != owner_thread
                assert committed.shape == (1, 6, 7)
                gate.wait(0.5)
            values = torch.arange(12, dtype=torch.float32).view(1, 12, 1)
            return values.expand(1, 12, 7).clone(), {}

        def encode_receipts(self, physical):
            return physical

        def close(self):
            self.closed = True

    class Robot:
        def __init__(self):
            self.sent = []
            self.stopped = False
            self.closed = False

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            assert threading.get_ident() == owner_thread
            del tick
            return PolicyObservation.snapshot(
                np.zeros(8),
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                },
                prompt,
                np.array([0, 0, 0, 1, 0, 0, 0]),
            )

        def read_tcp_pose(self):
            return np.array([0, 0, 0, 1, 0, 0, 0])

        def project_policy_actions(self, actions):
            raise AssertionError("Inference thread must not call the robot adapter")

        def apply_policy_action(self, step):
            assert threading.get_ident() == owner_thread
            self.sent.append((step.chunk_index, float(step.tcp_delta[0])))
            if release_early and len(self.sent) == 2:
                gate.set()

        def stop(self):
            self.stopped = True
            gate.set()

        def close(self):
            self.closed = True

    policy, robot = Policy(), Robot()
    runtime = ResidualHandoffRuntime(
        robot,
        policy,
        OnlineResidualRLConfig(),
        ResidualDeploymentConfig(prompt="task", control_hz=100, max_ticks=12),
        project_actions=lambda actions: actions.copy(),
    )
    assert runtime.run() == expected_code
    assert [value for _, value in robot.sent] == list(range(12 if release_early else 6))
    assert robot.stopped and robot.closed and policy.closed
    runtime.close()


def test_online_residual_executor_uses_absolute_control_deadlines():
    import threading

    import numpy as np

    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )
    from rlinf.models.embodiment.openpi.rtc.types import PolicyObservation

    gate = threading.Event()

    class Clock:
        now = 0.0

        def sleep(self, duration):
            self.now += duration

    class Policy:
        def predict_action_batch(self, obs, *, mode, committed):
            del obs, mode
            if committed is not None:
                gate.wait(0.5)
            return torch.zeros(1, 12, 7), {}

        def encode_receipts(self, commands):
            return commands

        def close(self):
            pass

    class Robot:
        def __init__(self, clock):
            self.clock = clock
            self.tick_ends = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            del tick
            return PolicyObservation.snapshot(
                np.zeros(8),
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                },
                prompt,
                np.array([0, 0, 0, 1, 0, 0, 0]),
            )

        def read_tcp_pose(self):
            return np.array([0, 0, 0, 1, 0, 0, 0])

        def project_policy_actions(self, commands):
            raise AssertionError("Inference thread must not call the robot adapter")

        def apply_policy_action(self, step):
            del step
            self.clock.now += 0.004 * (len(self.tick_ends) + 1)
            self.tick_ends.append(self.clock.now)

        def stop(self):
            gate.set()

        def close(self):
            pass

    clock = Clock()
    runtime = ResidualHandoffRuntime(
        Robot(clock),
        Policy(),
        OnlineResidualRLConfig(),
        ResidualDeploymentConfig(prompt="task", control_hz=30, max_ticks=3),
        project_actions=lambda commands: commands.copy(),
        monotonic=lambda: clock.now,
        sleep=clock.sleep,
    )
    assert runtime.run() == 0
    assert clock.now == pytest.approx(3 / 30)


def test_online_residual_external_worker_preserves_apx_thread_affinity():
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )
    from rlinf.models.embodiment.openpi.rtc.types import PolicyObservation

    class Policy:
        def __init__(self):
            self.owner = threading.get_ident()
            self.calls = 0

        def predict_action_batch(self, obs, *, mode, committed):
            assert threading.get_ident() == self.owner
            self.calls += 1
            return torch.zeros(1, 12, 7), {}

        def encode_receipts(self, physical):
            return physical

        def close(self):
            assert threading.get_ident() == self.owner

    class Robot:
        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            return PolicyObservation.snapshot(
                np.zeros(8),
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                },
                prompt,
                np.array([0, 0, 0, 1, 0, 0, 0]),
            )

        def read_tcp_pose(self):
            return np.array([0, 0, 0, 1, 0, 0, 0])

        def apply_policy_action(self, step):
            pass

        def stop(self):
            pass

        def close(self):
            pass

    with ThreadPoolExecutor(max_workers=1) as worker:
        policy = worker.submit(Policy).result()
        runtime = ResidualHandoffRuntime(
            Robot(),
            policy,
            OnlineResidualRLConfig(),
            ResidualDeploymentConfig(prompt="task", control_hz=30, max_ticks=1),
            project_actions=lambda values: values.copy(),
            own_policy=False,
            inference_executor=worker,
        )
        assert runtime.run() == 0
        assert policy.calls >= 1
        worker.submit(policy.close).result()


def test_online_residual_executor_closes_after_robot_start_failure():
    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )

    class Robot:
        stopped = False
        closed = False

        def start(self):
            raise RuntimeError("startup failed")

        def stop(self):
            self.stopped = True

        def close(self):
            self.closed = True

    class Policy:
        closed = False

        def close(self):
            self.closed = True

    robot, policy = Robot(), Policy()
    runtime = ResidualHandoffRuntime(
        robot,
        policy,
        OnlineResidualRLConfig(),
        ResidualDeploymentConfig(prompt="task"),
        project_actions=lambda commands: commands.copy(),
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        runtime.run()
    runtime.close()
    assert robot.stopped and robot.closed and policy.closed


def test_online_action_projection_uses_physical_limits():
    from rlinf.algorithms.online_residual_rl.action_mapping import ActionMapping

    mapping = ActionMapping([0.0, 0.5], [0.1, 0.5], [-0.02, 0.0], [0.02, 1.0])
    actions = torch.tensor([[0.5, 3.0]], requires_grad=True)
    projected = mapping(actions)
    torch.testing.assert_close(mapping.decode(projected), torch.tensor([[0.02, 1.0]]))
    projected.sum().backward()
    assert actions.grad is not None


def test_flexiv_norm_batches_apply_training_delta_and_preserve_gripper(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toolkits.lerobot.calculate_flexiv_norm_stats import iter_numeric_batches

    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    metadata = {
        "recorded_arm_names": ["left"],
        "action_representation": "eef_absolute",
        "total_frames": 3,
        "features": {
            "observation.eef_state": {"shape": [6]},
            "observation.joint_state": {"shape": [7]},
            "observation.gripper_state": {"shape": [1]},
            "action": {"shape": [7]},
        },
    }
    info = tmp_path / "meta" / "info.json"
    info.write_text(json.dumps(metadata))
    pq.write_table(
        pa.table(
            {
                "observation.eef_state": [
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                "observation.joint_state": [[0.0] * 7] * 3,
                "observation.gripper_state": [[0.060]] * 3,
                "action": [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003],
                    [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.004],
                    [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.005],
                ],
            }
        ),
        tmp_path / "data" / "episode_000000.parquet",
    )
    batches = list(iter_numeric_batches(tmp_path, batch_size=2, action_horizon=2))
    assert [len(batch["state"]) for batch in batches] == [2, 1]
    assert batches[0]["state"].shape == (2, 8)
    assert batches[0]["actions"].shape == (2, 2, 32)
    assert batches[0]["state"][0, -1] == 0.060
    assert batches[0]["actions"][0, 0, 6] == 0.003
    np.testing.assert_allclose(batches[0]["actions"][:, :, 0], [[1.0, 2.0], [1.0, 2.0]])
    np.testing.assert_allclose(batches[1]["actions"][:, :, 0], [[1.0, 1.0]])
    np.testing.assert_allclose(batches[0]["actions"][..., 7:], 0.0)

    metadata["action_representation"] = "eef_delta"
    info.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="absolute TCP or absolute joint"):
        list(iter_numeric_batches(tmp_path))


def test_flexiv_tcp_delta_is_world_frame_and_keeps_absolute_gripper():
    import numpy as np

    from rlinf.data.datasets.openpi_rlinf.flexiv.flexiv_sft_dataset import (
        absolute_tcp_chunk_to_world_delta,
        future_action_indices,
    )

    reference = np.array([1.0, 2.0, 3.0, 0.0, 0.0, np.pi / 2])
    targets = np.array(
        [
            [2.0, 4.0, 6.0, 0.0, 0.0, np.pi, 0.06],
            [0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.01],
        ]
    )
    delta = absolute_tcp_chunk_to_world_delta(reference, targets)

    np.testing.assert_allclose(delta[:, :3], [[1.0, 2.0, 3.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        delta[:, 3:6],
        [[0.0, 0.0, np.pi / 2], [0.0, 0.0, -np.pi / 2]],
    )
    np.testing.assert_allclose(delta[:, 6], [0.06, 0.01])
    np.testing.assert_array_equal(
        future_action_indices(3, 4),
        [[0, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2]],
    )


class TestMathDatasetMultithread:
    """Tests for ReasoningDataset multithread processing consistency."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = mock.Mock()
        tokenizer.is_fast = True
        tokenizer.eos_token_id = 2

        def apply_chat_template_side_effect(
            prompts, tokenize=False, add_generation_prompt=True
        ):
            """Mock apply_chat_template that handles generator input."""
            # Convert generator to list if needed
            prompts_list = list(prompts) if not isinstance(prompts, list) else prompts
            return [
                f"<|user|>\n{prompt}\n<|assistant|>\n"
                if isinstance(prompt, str)
                else prompt
                for prompt in prompts_list
            ]

        tokenizer.apply_chat_template = mock.Mock(
            side_effect=apply_chat_template_side_effect
        )
        tokenizer.batch_encode_plus = mock.Mock(
            side_effect=lambda texts: {
                "input_ids": [[1] * len(text.split()) for text in texts]
            }
        )
        tokenizer.encode = mock.Mock(side_effect=lambda text: [1] * len(text.split()))
        return tokenizer

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for testing."""
        config = DictConfig(
            {
                "data": {
                    "max_prompt_length": 1000,
                    "prompt_key": "question",
                    "answer_key": "answer",
                    "apply_chat_template": True,
                    "filter_prompt_by_length": False,
                    "process_workers": 4,
                    "process_batch_size": 32,
                }
            }
        )
        return config

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing (at least 10000 items)."""
        # Generate at least 10000 math problems
        data = []
        operations = [
            ("+", lambda a, b: a + b),
            ("-", lambda a, b: a - b),
            ("*", lambda a, b: a * b),
            ("/", lambda a, b: a // b if b != 0 else 0),
        ]

        for i in range(10000):
            op_symbol, op_func = random.choice(operations)
            a = random.randint(1, 1000)
            b = random.randint(1, 1000) if op_symbol != "/" else random.randint(1, 100)
            if op_symbol == "/" and b == 0:
                b = 1
            result = op_func(a, b)
            question = f"What is {a} {op_symbol} {b}?"
            data.append({"question": question, "answer": str(result)})

        return data

    def test_multithread_vs_singlethread_consistency(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test that multithread processing produces identical results to single-thread processing.

        This test verifies that:
        1. Results from multi-worker processing match single-worker processing
        2. All keys are the same
        3. All values are the same
        """
        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        # We need to reload the raw data to avoid double processing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        time_start = time.time()
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )
        time_elapse_multithread = time.time() - time_start

        # Test with single thread
        time_start = time.time()
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)
        time_elapse_singlethread = time.time() - time_start

        # Verify lengths are equal
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify all items have the same keys and values
        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            keys_mt, keys_st = item_mt.keys(), item_st.keys()
            assert keys_mt == keys_st, (
                f"Keys mismatch at index {idx}: "
                f"multithread={list(keys_mt)}, singlethread={list(keys_st)}"
            )

            # Check all values are equal
            unequal_keys = [key for key in keys_mt if item_mt[key] != item_st[key]]
            assert len(unequal_keys) == 0, (
                f"Values mismatch at index {idx} for keys: {unequal_keys}"
            )

        # Log timing information (for debugging)
        print(
            f"Data count: {len(data_multithread)}, "
            f"Multithread processing time: {time_elapse_multithread:.2f}s, "
            f"Singlethread processing time: {time_elapse_singlethread:.2f}s"
        )

    def test_multithread_consistency_with_filter(
        self, mock_tokenizer, mock_config, sample_data, tmp_path
    ):
        """
        Test multithread processing consistency when filter_prompt_by_length is enabled.
        """
        # Update config to enable filtering
        mock_config.data.filter_prompt_by_length = True
        mock_config.data.max_prompt_length = 50  # Reasonable limit to test filtering

        # Create a temporary JSON file with sample data
        data_file = tmp_path / "test_data.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f)

        # Create ReasoningDataset instance to get the configuration
        dataset = ReasoningDataset(
            data_paths=str(data_file),
            config=mock_config,
            tokenizer=mock_tokenizer,
        )

        # Use original raw data (before processing) for testing
        raw_data = dataset._load_data()

        # Deep copy to avoid modifying the original
        raw_data_multithread = copy.deepcopy(raw_data)
        raw_data_singlethread = copy.deepcopy(raw_data)

        # Test with multithread parameters
        data_multithread = dataset.load_post_process(
            raw_data_multithread, dataset.process_workers, dataset.process_batch_size
        )

        # Test with single thread
        data_singlethread = dataset.load_post_process(raw_data_singlethread, 1, 1)

        # Verify consistency
        assert len(data_multithread) == len(data_singlethread), (
            f"Length mismatch: multithread={len(data_multithread)}, singlethread={len(data_singlethread)}"
        )

        # Verify that some data was filtered (not all data passed)
        assert len(data_multithread) <= len(raw_data), (
            f"Filtering should reduce data size, but got {len(data_multithread)} >= {len(raw_data)}"
        )

        for idx, (item_mt, item_st) in enumerate(
            zip(data_multithread, data_singlethread)
        ):
            assert item_mt.keys() == item_st.keys(), f"Keys mismatch at index {idx}"
            for key in item_mt.keys():
                assert item_mt[key] == item_st[key], (
                    f"Mismatch at index {idx}, key {key}"
                )

        print(
            f"Filtered data count: {len(data_multithread)}/{len(raw_data)} "
            f"(max_prompt_length={mock_config.data.max_prompt_length})"
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def test_split_dict_to_chunk_keeps_mixed_fields_aligned():
    batch = {
        "values": torch.arange(10),
        "sample_ids": list(range(10)),
        "nested": {"values": torch.arange(10) + 100},
    }

    chunks = split_dict_to_chunk(batch, 3)

    assert [chunk["values"].tolist() for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["sample_ids"] for chunk in chunks] == [
        [0, 1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    assert [chunk["nested"]["values"].tolist() for chunk in chunks] == [
        [100, 101, 102, 103],
        [104, 105, 106],
        [107, 108, 109],
    ]


def test_split_dict_to_chunk_returns_requested_number_of_chunks():
    batch = {"values": torch.arange(2), "sample_ids": ["a", "b"]}

    chunks = split_dict_to_chunk(batch, 4)

    assert len(chunks) == 4
    assert [chunk["values"].tolist() for chunk in chunks] == [[0], [1], [], []]
    assert [chunk["sample_ids"] for chunk in chunks] == [["a"], ["b"], [], []]


def _make_trajectory_builder(batch_size: int) -> EmbodiedTrajectoryBuilder:
    sample_ids = torch.arange(batch_size)
    builder = EmbodiedTrajectoryBuilder()
    builder.curr_obs.append({"sample_ids": sample_ids})
    builder.actions.append(sample_ids[:, None])
    builder.rewards.append(sample_ids)
    return builder


def test_trajectory_chunks_keep_top_level_fields_aligned():
    trajectories = _make_trajectory_builder(10).to_splited_trajectories(3)

    expected_ids = [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert [
        trajectory.curr_obs["sample_ids"][0].tolist() for trajectory in trajectories
    ] == expected_ids
    assert [
        trajectory.actions[0, :, 0].tolist() for trajectory in trajectories
    ] == expected_ids
    assert [
        trajectory.rewards[0].tolist() for trajectory in trajectories
    ] == expected_ids


def test_trajectory_split_returns_requested_number_of_chunks():
    trajectories = _make_trajectory_builder(2).to_splited_trajectories(4)

    assert len(trajectories) == 4
    assert [trajectory.actions.shape[1] for trajectory in trajectories] == [1, 1, 0, 0]
    assert [
        trajectory.curr_obs["sample_ids"].shape[1] for trajectory in trajectories
    ] == [1, 1, 0, 0]


class _LegacyDataset:
    """Mimics lerobot < 0.2: the task lives inside the frame dict."""

    def __init__(self):
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append(frame)

    def save_episode(self):
        self.saved_episodes += 1


class _PostRevertDataset(_LegacyDataset):
    """Mimics lerobot >= 0.4: back to ``add_frame(frame)``, but it pops the task.

    0.4 reverted the 0.3.x signature, so a version-number check would dispatch
    this one wrongly. It also mutates the caller's dict.
    """

    def add_frame(self, frame):
        if "task" not in frame:
            raise ValueError("Missing features: {'task'}")
        self.frames.append({**frame, "task": frame.pop("task")})


class _CurrentDataset:
    """Mimics lerobot >= 0.3: the task is a separate argument.

    Like the real implementation, a ``task`` key inside *frame* is rejected
    because it is not part of the feature schema.
    """

    def __init__(self):
        self.frames = []
        self.tasks = []
        self.saved_episodes = 0

    def add_frame(self, frame, task, timestamp=None):
        if "task" in frame:
            raise ValueError("Extra features: {'task'}")
        self.frames.append(frame)
        self.tasks.append(task)

    def save_episode(self):
        self.saved_episodes += 1


def _make_writer(dataset):
    # ``create()`` needs a real lerobot install, so attach the dataset the way
    # ``create()`` would.
    writer = LeRobotDatasetWriter()
    writer.dataset = dataset
    return writer


def _episode(n=2):
    return [{"state": i, "actions": i, "task": "pick up the cube"} for i in range(n)]


def test_legacy_dataset_keeps_task_in_frame():
    dataset = _LegacyDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


def test_current_dataset_gets_task_as_argument():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode(_episode())

    assert dataset.tasks == ["pick up the cube"] * 2
    assert all("task" not in f for f in dataset.frames)
    assert dataset.frames[0]["state"] == 0
    assert dataset.saved_episodes == 1


ALL_SHAPES = [_LegacyDataset, _CurrentDataset, _PostRevertDataset]


def test_post_revert_dataset_keeps_task_in_frame():
    # lerobot >= 0.4 took the 0.3.x signature back out again.
    dataset = _PostRevertDataset()
    _make_writer(dataset).add_episode(_episode())

    assert [f["task"] for f in dataset.frames] == ["pick up the cube"] * 2
    assert dataset.saved_episodes == 1


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_caller_frames_are_not_mutated(dataset_cls):
    # The DAgger worker shares these dicts with the in-memory training store,
    # and lerobot >= 0.4 pops "task" out of whatever frame it is handed.
    episode = _episode()
    before = [dict(f) for f in episode]
    _make_writer(dataset_cls()).add_episode(episode)

    assert episode == before


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_frame_without_task_is_rejected(dataset_cls):
    with pytest.raises(ValueError, match="missing the required 'task' field"):
        add_frame_to_dataset(dataset_cls(), {"state": 0, "actions": 0})


@pytest.mark.parametrize("dataset_cls", ALL_SHAPES)
def test_add_frame_to_dataset_is_usable_standalone(dataset_cls):
    # The toolkit collectors drive LeRobotDataset directly, without the writer.
    dataset = dataset_cls()
    add_frame_to_dataset(dataset, {"state": 0, "task": "wipe the table"})

    assert len(dataset.frames) == 1


def test_empty_episode_is_skipped():
    dataset = _CurrentDataset()
    _make_writer(dataset).add_episode([])

    assert dataset.frames == []
    assert dataset.saved_episodes == 0


# --------------------------------------------------------------------------
# episode_boundaries: dataset format v2.1 vs v3.0
# --------------------------------------------------------------------------


class _V21Dataset:
    """Dataset format v2.1: a dict of two tensors on the dataset itself."""

    def __init__(self, starts, ends):
        self.episode_data_index = {
            "from": torch.tensor(starts),
            "to": torch.tensor(ends),
        }


class _V30Meta:
    def __init__(self, starts, ends):
        self.episodes = {"dataset_from_index": starts, "dataset_to_index": ends}


class _V30Dataset:
    """Dataset format v3.0 (lerobot >= 0.4): columns on ``meta.episodes``."""

    def __init__(self, starts, ends):
        self.episode_data_index = None
        self.meta = _V30Meta(starts, ends)


@pytest.mark.parametrize("dataset_cls", [_V21Dataset, _V30Dataset])
def test_episode_boundaries_agree_across_formats(dataset_cls):
    starts, ends = episode_boundaries(dataset_cls([0, 3, 7], [3, 7, 9]))

    assert starts == [0, 3, 7]
    assert ends == [3, 7, 9]
    assert all(isinstance(x, int) for x in starts + ends)


def test_episode_boundaries_reports_an_unknown_layout():
    class _Alien:
        episode_data_index = None
        meta = None

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Alien())


def test_episode_boundaries_rejects_v30_meta_without_the_columns():
    class _Partial:
        episode_data_index = None
        meta = _V30Meta([0], [1])

    _Partial.meta.episodes = {"length": [1]}

    with pytest.raises(RuntimeError, match="Cannot determine episode boundaries"):
        episode_boundaries(_Partial())


# Skip codec round-trip tests when the optional backends are not installed.
_CODECS = []
try:
    import lz4.frame  # noqa: F401

    _CODECS.append("lz4")
except ImportError:
    pass
try:
    import zstandard  # noqa: F401

    _CODECS.append("zstd")
except ImportError:
    pass

requires_codec = pytest.mark.skipif(
    not _CODECS, reason="no observation compression codec (lz4/zstd) installed"
)


def _make_payload(num_envs: int = 4) -> dict:
    """A payload shaped like EnvWorker._build_rollout_input_data output."""
    obs = {
        "main_images": torch.randint(0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8),
        "extra_view_images": torch.randint(
            0, 256, (num_envs, 6, 6, 3), dtype=torch.uint8
        ),
        "states": torch.randn(num_envs, 7, dtype=torch.float32),
        "task_descriptions": ["put carrot on plate"] * num_envs,
    }
    return {
        "obs": obs,
        "final_obs": {
            "main_images": torch.randint(
                0, 256, (num_envs, 8, 8, 3), dtype=torch.uint8
            ),
            "states": torch.randn(num_envs, 7, dtype=torch.float32),
        },
        "rlt_switch_flags": None,
    }


def _assert_payload_equal(a: dict, b: dict) -> None:
    assert a.keys() == b.keys()
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, dict):
            _assert_payload_equal(va, vb)
        elif isinstance(va, torch.Tensor):
            assert torch.equal(va, vb), f"tensor mismatch for {key!r}"
        else:
            assert va == vb, f"value mismatch for {key!r}"


def _cfg(**overrides):
    # A plain dict is sufficient: the codec only calls ``config.get(...)``,
    # which both ``dict`` and OmegaConf's ``DictConfig`` support identically.
    base = {"enable": True, "codec": "lz4", "level": 1, "xor_delta": True}
    base.update(overrides)
    return base


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
@pytest.mark.parametrize("xor_delta", [True, False])
def test_compress_decompress_is_lossless(codec, xor_delta):
    payload = _make_payload()
    config = _cfg(codec=codec, xor_delta=xor_delta)

    compressed = compress_obs(payload, config)
    # Image tensors are replaced by self-describing marker dicts...
    assert _CODEC_KEY in compressed["obs"]["main_images"]
    assert _CODEC_KEY in compressed["obs"]["extra_view_images"]
    # ...while non-image fields are passed through untouched.
    assert isinstance(compressed["obs"]["states"], torch.Tensor)
    assert compressed["obs"]["task_descriptions"] == payload["obs"]["task_descriptions"]

    restored = decompress_obs(compressed)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_single_env_batch_roundtrip():
    # XOR-delta is skipped when there is only one frame; must still be lossless.
    payload = _make_payload(num_envs=1)
    restored = decompress_obs(compress_obs(payload, _cfg(xor_delta=True)))
    _assert_payload_equal(payload, restored)


def test_disabled_config_is_passthrough():
    payload = _make_payload()
    assert compress_obs(payload, _cfg(enable=False)) is payload
    assert compress_obs(payload, None) is payload
    assert not is_compression_enabled(None)
    assert not is_compression_enabled(_cfg(enable=False))
    assert is_compression_enabled(_cfg(enable=True))


def test_decompress_on_uncompressed_payload_is_noop():
    # The rollout worker always routes received data through decompress_obs, so
    # it must be a no-op on payloads sent without compression.
    payload = _make_payload()
    restored = decompress_obs(payload)
    _assert_payload_equal(payload, restored)


@requires_codec
def test_only_uint8_images_are_compressed():
    # A float image-shaped tensor is not a uint8 observation and must be left
    # untouched, as must low-rank uint8 tensors (e.g. flags).
    payload = {
        "obs": {
            "float_map": torch.randn(4, 8, 8, 3),
            "uint8_flags": torch.ones(4, dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert isinstance(compressed["obs"]["float_map"], torch.Tensor)
    assert isinstance(compressed["obs"]["uint8_flags"], torch.Tensor)


def test_unknown_codec_raises():
    payload = _make_payload()
    with pytest.raises(ValueError, match="Unknown observation compression codec"):
        compress_obs(payload, _cfg(codec="bogus"))


@requires_codec
@pytest.mark.parametrize("codec", _CODECS)
def test_routing_split_then_compress_roundtrip(codec):
    """Compression must be compatible with the Env->Rollout channel routing.

    The env worker installs compression as a ``split_fn`` so it runs *after*
    the scheduler splits the batch: ``infer_batch_size`` and ``split_batch``
    see plain tensors, and each shard is compressed independently. This test
    reproduces that flow with the real routing helpers and asserts the payload
    survives split -> compress -> decompress -> merge unchanged.
    """
    routing = pytest.importorskip("rlinf.scheduler.worker.routing")

    payload = _make_payload(num_envs=6)
    # The scheduler infers the batch size from the *uncompressed* payload.
    assert routing.infer_batch_size(payload) == 6

    # split_fn = split_batch first, then compress each shard (env send path).
    split_sizes = [2, 1, 3]
    shards = routing.split_batch(payload, split_sizes)
    compressed_shards = [compress_obs(shard, _cfg(codec=codec)) for shard in shards]

    # Rollout side: decompress each shard, then merge (merge_obs path).
    restored_shards = [decompress_obs(shard) for shard in compressed_shards]
    merged = routing.merge_batches(restored_shards)
    _assert_payload_equal(payload, merged)


def test_infer_obs_batch_size_uncompressed():
    payload = _make_payload(num_envs=5)
    assert infer_obs_batch_size(payload) == 5
    # Also accepts a bare obs dict (no "obs" wrapper).
    assert infer_obs_batch_size(payload["obs"]) == 5


@requires_codec
def test_infer_obs_batch_size_with_compressed_images():
    # The rollout worker infers batch size on the receive path, before
    # decompression, so a compressed image must still report its batch size.
    payload = _make_payload(num_envs=5)
    compressed = compress_obs(payload, _cfg())
    assert is_compressed_image(compressed["obs"]["main_images"])
    assert infer_obs_batch_size(compressed) == 5


@requires_codec
def test_infer_obs_batch_size_images_only():
    # Regression: a batch whose only batched field is a (compressed) image,
    # with no states/task_descriptions, must not break batch-size inference.
    payload = {
        "obs": {
            "main_images": torch.randint(0, 256, (3, 8, 8, 3), dtype=torch.uint8),
        }
    }
    compressed = compress_obs(payload, _cfg())
    assert infer_obs_batch_size(compressed) == 3


def test_infer_obs_batch_size_raises_when_unbatched():
    with pytest.raises(ValueError, match="Cannot infer batch size"):
        infer_obs_batch_size({"obs": {}})


def test_vlm_trend_batch_video_metadata_stays_nested_per_sample():
    pytest.importorskip("transformers.video_utils")
    from transformers.video_utils import VideoMetadata

    from rlinf.data.datasets.vlm.vlm_trend_reward import VLMTrendRewardSFTDataset

    captured = {}

    class _Processor:
        video_token = "<|video_pad|>"

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, **kwargs):
            captured["videos_kwargs"] = kwargs["videos_kwargs"]
            batch = len(kwargs["text"])
            return {
                "input_ids": torch.zeros(batch, 4, dtype=torch.long),
                "attention_mask": torch.ones(batch, 4, dtype=torch.long),
            }

    VLMTrendRewardSFTDataset.process_inputs(
        processor=_Processor(),
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[["task a"], ["task b"]],
        videos=[
            [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            [[10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
        ],
        answer_text=None,
        video_fps=24.0,
    )

    metadata = captured["videos_kwargs"]["video_metadata"]
    assert len(metadata) == 2
    assert all(len(sample) == 2 for sample in metadata)
    assert all(
        isinstance(item, VideoMetadata) for sample in metadata for item in sample
    )
    assert metadata[0][0].total_num_frames == 5
    assert metadata[0][0].frames_indices == [0, 1, 2, 3, 4]
    assert metadata[1][1].total_num_frames == 5


def test_online_collector_uses_final_receipts_and_terminal_label():
    import numpy as np

    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    replay = EpisodeReplay(cfg)

    def metadata():
        return {
            "forward_inputs": {
                "z_rl": torch.zeros(1, cfg.z_dim),
                "proprio": torch.zeros(1, cfg.proprio_dim),
                "base_ref": torch.zeros(1, 12, 7),
                "actor_output": torch.zeros(1, 12, 7),
            }
        }

    collector = EpisodeCollector(
        cfg,
        lambda physical: physical,
        lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
        episode_id="receipt-episode",
    )
    for tick in range(18):
        mask = np.zeros(7, dtype=bool)
        values = np.zeros(7, dtype=np.float32)
        if tick == 7:
            values[0] = 0.01
            mask[0] = True
        collector.receipt(
            tick,
            {
                "action_receipt_physical": values,
                "policy_proposal_physical": np.zeros(7, dtype=np.float32),
                "human_mask": mask,
                "pair_mask": mask,
            },
        )
        if tick in {5, 11, 17}:
            collector.decision(
                tick - 5,
                metadata(),
                0,
            )
    collector.finish(17, True)
    assert len(replay) == 2
    rows = sorted(replay.transitions(), key=lambda row: row.feature.frame_index)
    assert rows[0].next_feature.frame_index == 12
    assert rows[0].correction.pair_mask[7, 0]
    assert rows[1].feature.committed[1, 0] == pytest.approx(0.01)
    assert not rows[1].correction.pair_mask[:6].any()
    assert rows[1].terminal and rows[1].rewards.sum() == 1
    collector.finish(17, True)


def test_online_collector_rebases_overlapping_tcp_receipts_per_decision():
    import numpy as np

    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    committed = []

    def metadata():
        return {
            "forward_inputs": {
                "z_rl": torch.zeros(1, cfg.z_dim),
                "proprio": torch.zeros(1, cfg.proprio_dim),
                "base_ref": torch.zeros(1, 12, 7),
                "actor_output": torch.zeros(1, 12, 7),
            }
        }

    collector = EpisodeCollector(
        cfg,
        lambda physical: physical,
        lambda rows, success, terminal: committed.extend(rows),
        episode_id="rebase-episode",
    )
    for frame in range(18):
        target = np.array([frame * 0.01, 0, 0, 1, 0, 0, 0], float)
        collector.receipt(
            frame,
            {
                "action_receipt_physical": np.zeros(7, np.float32),
                "tcp_target_pose": target,
            },
        )
        if frame in {5, 11, 17}:
            start = frame - 5
            reference = np.array([start * 0.01, 0, 0, 1, 0, 0, 0], float)
            collector.decision(
                start,
                metadata(),
                0,
                tcp_reference_pose=reference,
            )
    collector.finish(17, True)
    rows = sorted(committed, key=lambda row: row.feature.frame_index)
    assert len(rows) == 2
    assert rows[0].executed[6, 0] == pytest.approx(0.06)
    assert rows[1].feature.committed[0, 0] == pytest.approx(0.0)
    assert rows[1].executed[6, 0] == pytest.approx(0.06)


def test_online_collector_preserves_valid_fragment_after_manual_outlives_last_window():
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    submitted = []
    metadata = {
        "forward_inputs": {
            "z_rl": torch.zeros(1, cfg.z_dim),
            "proprio": torch.zeros(1, cfg.proprio_dim),
            "base_ref": torch.zeros(1, 12, 7),
            "actor_output": torch.zeros(1, 12, 7),
        }
    }
    collector = EpisodeCollector(
        cfg,
        lambda physical: physical,
        lambda rows, success, terminal: submitted.append((rows, success, terminal)),
        episode_id="long-manual",
    )
    for frame in range(31):
        collector.receipt(
            frame,
            {
                "action_receipt_physical": torch.zeros(7),
                "human_mask": torch.ones(7, dtype=torch.bool),
            },
        )
        if frame in {5, 11, 17}:
            collector.decision(
                frame - 5,
                metadata,
                0,
            )
    collector.finish(30, True)
    assert len(submitted) == 1
    assert submitted[0][1:] == (True, 30)
    assert all(not row.terminal for row in submitted[0][0])


def test_online_actor_mailbox_installs_only_newest_version():
    from rlinf.algorithms.online_residual_rl.online import ActorMailbox

    mailbox = ActorMailbox()
    state = {"weight": torch.ones(2)}
    mailbox.publish(1, state)
    state["weight"].zero_()
    mailbox.publish(2, state)
    version, snapshot = mailbox.take()
    assert version == 2
    torch.testing.assert_close(snapshot["weight"], torch.zeros(2))
    assert mailbox.take() is None
    with pytest.raises(ValueError, match="increase"):
        mailbox.publish(2, state)


def test_flexiv_delta_mapping_matches_vector_limits_and_gripper():
    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivDeltaActionMapping,
    )

    mapping = FlexivDeltaActionMapping(
        torch.zeros(7),
        torch.ones(7),
        translation_limit=0.02,
        rotation_limit=0.1,
        gripper_range=(0.003, 0.06),
    )
    value = torch.tensor([[0.03, 0.04, 0.0, 0.0, 0.0, 0.2, 0.08]])
    projected = mapping(value)
    torch.testing.assert_close(projected[0, :3], torch.tensor([0.012, 0.016, 0.0]))
    assert projected[0, 5] == pytest.approx(0.1)
    assert projected[0, 6] == pytest.approx(0.06)


def test_online_runtime_collects_receipts_without_inference_thread_robot_calls():
    import threading

    import numpy as np

    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )
    from rlinf.algorithms.online_residual_rl.online import (
        ActorMailbox,
        EpisodeCollector,
    )
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.models.embodiment.openpi.rtc.types import PolicyObservation

    owner = threading.get_ident()
    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    replay = EpisodeReplay(cfg)
    mailbox = ActorMailbox()
    mailbox.publish(1, {"weight": torch.ones(1)})

    class Policy:
        conditions_on_committed = False

        def __init__(self):
            self.installs = []

        def predict_action_batch(self, obs, *, mode, committed):
            assert mode == "eval" and obs["states"].shape == (1, 8)
            del committed
            return torch.zeros(1, 12, 7), {
                "forward_inputs": {
                    "z_rl": torch.zeros(1, 4),
                    "proprio": torch.zeros(1, 8),
                    "base_ref": torch.zeros(1, 12, 7),
                    "actor_output": torch.zeros(1, 12, 7),
                }
            }

        def encode_receipts(self, physical):
            return physical

        def install_actor(self, state, version):
            assert threading.get_ident() != owner
            self.installs.append(version)

        def close(self):
            pass

    class Robot:
        def __init__(self):
            self.sent = 0
            self.stopped = False
            self.references = []

        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            assert threading.get_ident() == owner
            return PolicyObservation.snapshot(
                np.zeros(8),
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                },
                prompt,
                np.array([tick * 0.001, 0, 0, 1, 0, 0, 0]),
            )

        def read_tcp_pose(self):
            assert threading.get_ident() == owner
            return np.array([0, 0, 0, 1, 0, 0, 0])

        def apply_policy_action(self, step):
            assert threading.get_ident() == owner
            self.references.append(float(step.tcp_reference_pose[0]))
            value = np.concatenate((step.tcp_delta, [step.gripper])).astype(np.float32)
            mask = np.zeros(7, dtype=bool)
            if self.sent == 7:
                value[0] = 0.01
                mask[0] = True
            self.sent += 1
            target = step.tcp_reference_pose.copy()
            target[:3] += value[:3]
            return {
                "action_receipt_physical": value,
                "tcp_target_pose": target,
                "human_mask": mask,
                "pair_mask": mask,
            }

        def stop(self):
            self.stopped = True

        def close(self):
            pass

    policy, robot = Policy(), Robot()
    collector = EpisodeCollector(
        cfg,
        policy.encode_receipts,
        lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
        episode_id="runtime-episode",
    )
    runtime = ResidualHandoffRuntime(
        robot,
        policy,
        cfg,
        ResidualDeploymentConfig(prompt="test", control_hz=100, max_ticks=18),
        project_actions=lambda values: values.copy(),
        collector=collector,
        actor_mailbox=mailbox,
        terminal_label=True,
    )
    assert runtime.run() == 0
    assert robot.sent == 18 and robot.stopped
    np.testing.assert_allclose(robot.references, [0.0] * 12 + [0.006] * 6)
    assert policy.installs == [1]
    assert len(replay) == 2
    rows = sorted(replay.transitions(), key=lambda row: row.feature.frame_index)
    assert {row.feature.model_version for row in rows} == {1}
    assert rows[0].executed[7, 0] == pytest.approx(0.01)
    assert rows[1].feature.committed[1, 0] == pytest.approx(0.004)


def test_online_torch_rtc_uses_base_raw_actions_for_guidance():
    import numpy as np

    from rlinf.algorithms.online_residual_rl.deployment import (
        ResidualDeploymentConfig,
        ResidualHandoffRuntime,
    )
    from rlinf.models.embodiment.openpi.rtc.types import PolicyObservation

    class Policy:
        conditions_on_committed = True

        def __init__(self):
            self.guidance = []

        def predict_action_batch(self, obs, *, mode, committed):
            del obs, mode
            self.guidance.append(None if committed is None else committed.clone())
            return torch.zeros(1, 12, 7), {
                "raw_model_actions": torch.full((1, 32, 32), float(len(self.guidance))),
                "normalized_actions": torch.zeros(1, 12, 7),
            }

        def encode_receipts(self, physical):
            return physical

        def close(self):
            pass

    class Robot:
        def start(self):
            pass

        def read_observation(self, *, tick, prompt):
            del tick
            return PolicyObservation.snapshot(
                np.zeros(8),
                {
                    "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
                    "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8),
                },
                prompt,
                np.array([0, 0, 0, 1, 0, 0, 0]),
            )

        def read_tcp_pose(self):
            return np.array([0, 0, 0, 1, 0, 0, 0])

        def apply_policy_action(self, step):
            return {
                "action_receipt_physical": np.r_[step.tcp_delta, step.gripper],
                "human_mask": np.zeros(7, dtype=bool),
                "pair_mask": np.zeros(7, dtype=bool),
            }

        def stop(self):
            pass

        def close(self):
            pass

    class Collector:
        def __init__(self):
            self.decisions = []

        def decision(
            self,
            frame,
            metadata,
            version,
            *,
            tcp_reference_pose=None,
        ):
            del metadata, version, tcp_reference_pose
            self.decisions.append(frame)

        def receipt(self, tick, receipt):
            del tick, receipt

        def finish(self, tick, label):
            del tick, label

    policy = Policy()
    collector = Collector()
    runtime = ResidualHandoffRuntime(
        Robot(),
        policy,
        OnlineResidualRLConfig(),
        ResidualDeploymentConfig(prompt="task", control_hz=100, max_ticks=12),
        project_actions=lambda actions: actions.copy(),
        collector=collector,
    )
    assert runtime.run() == 0
    assert policy.guidance[0] is None
    torch.testing.assert_close(policy.guidance[1], torch.full((1, 32, 32), 1.0))
    torch.testing.assert_close(policy.guidance[2], torch.full((1, 26, 32), 2.0))
    assert collector.decisions == [0, 6]


def test_online_learner_thread_saves_and_publishes_after_label(tmp_path):
    from rlinf.algorithms.online_residual_rl.online import (
        ActorMailbox,
        OnlineLearnerThread,
    )
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    assembler = TransitionAssembler(cfg)
    feature = _online_feature(cfg, "one", 0)
    actions = torch.zeros(12, 7)
    assembler.begin(feature, actions)
    row = assembler.complete(
        feature.obs_id,
        executed=actions,
        rewards=torch.zeros(12),
        terminal=True,
        valid_len=12,
        next_feature=None,
    )
    mailbox = ActorMailbox()
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=lambda value: value),
        EpisodeReplay(cfg),
        initial_episodes=1,
        warmup_updates=0,
        td_batch_size=2,
        supervision_batch_size=2,
        publish_interval=1,
        publish=mailbox.publish,
    )
    worker = OnlineLearnerThread(training, checkpoint_dir=tmp_path)
    worker.submit([row], True, 11)
    worker.close()
    assert (tmp_path / "latest.pt").is_file()
    assert training.actor_updates == 1
    assert mailbox.take()[0] == 1


def test_joint_stage2_collects_absolute_commands_saves_and_resumes_training(tmp_path):
    import numpy as np

    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivJointActionMapping,
    )
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.storage import TransitionStore
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    offset = torch.tensor([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    scale = torch.tensor([0.1] * 7 + [0.02])
    mapping = FlexivJointActionMapping(offset, scale, gripper_range=(0.003, 0.06))
    replay = EpisodeReplay(cfg, storage_dir=tmp_path / "episodes")
    collector = EpisodeCollector(
        cfg,
        mapping.encode,
        lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
        episode_id="joint-episode",
    )
    commands = []
    for frame in range(30):
        physical = offset.clone()
        physical[:7] += frame * 0.001
        mask = np.zeros(8, dtype=bool)
        if frame == 7:
            physical[6] += 0.05
            mask[6] = True
        commands.append(physical)
        collector.receipt(
            frame,
            {
                "action_representation": "joint_absolute",
                "action_receipt_physical": physical.numpy(),
                "policy_proposal_physical": offset.numpy(),
                "human_mask": mask,
                "pair_mask": mask,
                "receipt_valid": True,
            },
        )
        if frame % 6 == 0:
            collector.decision(
                frame,
                {
                    "forward_inputs": {
                        "z_rl": torch.zeros(1, 4),
                        "proprio": torch.zeros(1, 8),
                        "base_ref": torch.zeros(1, 12, 8),
                        "actor_output": torch.zeros(1, 12, 8),
                    }
                },
                0,
                tcp_reference_pose=np.array([100 + frame, 200, 300, 1, 0, 0, 0]),
            )
    collector.finish(29, True)
    assert len(replay) == 4
    rows = sorted(replay.transitions(), key=lambda row: row.feature.frame_index)
    for row in rows:
        start = row.feature.frame_index
        physical = torch.stack(commands[start : start + 12])
        assert row.executed_physical is None
        torch.testing.assert_close(mapping.decode(row.executed), physical)
        torch.testing.assert_close(row.executed, mapping.encode(physical))
        assert row.feature.config_version == "flexiv-joint-absolute-v1"
    assert rows[0].correction.pair_mask[7, 6]
    assert not rows[0].correction.pair_mask[7, 7]
    saved = replay.store.load_episode("joint-episode")
    assert saved["schema_version"] == 3
    assert saved["action_representation"] == "joint_absolute"
    assert saved["action_dim"] == 8
    assert saved["physical_action_units"] == ["rad"] * 7 + ["m"]
    torch.testing.assert_close(
        saved["transitions"][0]["exec_physical"], torch.stack(commands[:12])
    )
    with pytest.raises(ValueError, match="representation"):
        TransitionStore(
            tmp_path / "episodes", OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
        ).load_episode("joint-episode")
    with pytest.raises(ValueError, match="contract"):
        EpisodeReplay(OnlineResidualRLConfig(z_dim=4, hidden_dim=8)).load_state_dict(
            replay.state_dict()
        )

    learner = OnlineResidualLearner(cfg, action_projector=mapping)
    training = OnlineTraining(
        learner,
        replay,
        initial_episodes=0,
        warmup_updates=0,
        td_batch_size=2,
        supervision_batch_size=2,
    )
    training.budget.add_transitions(len(replay))
    for _ in range(5):
        assert "critic/td" in training.step()
    assert training.actor_updates == 1
    checkpoint = tmp_path / "joint_stage2.pt"
    training.save(checkpoint)
    restored_mapping = FlexivJointActionMapping(
        offset, scale, gripper_range=(0.003, 0.06)
    )
    restored = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=restored_mapping),
        EpisodeReplay(cfg),
    )
    restored.load(checkpoint)
    assert restored.actor_updates == 1 and len(restored.replay) == 4
    with pytest.raises(ValueError, match="normalization"):
        OnlineTraining(
            OnlineResidualLearner(
                cfg,
                action_projector=FlexivJointActionMapping(
                    offset + 0.01, scale, gripper_range=(0.003, 0.06)
                ),
            ),
            EpisodeReplay(cfg),
        ).load(checkpoint)


def test_joint_training_projection_keeps_absolute_angles_and_binary_gripper_history():
    from rlinf.algorithms.online_residual_rl.action_mapping import (
        FlexivJointActionMapping,
    )

    offset = torch.tensor([-0.2, -0.4, 0.2, 2.3, -0.3, 1.2, 0.3, 0.04])
    scale = torch.tensor([0.1] * 7 + [0.02])
    mapping = FlexivJointActionMapping(
        offset, scale, gripper_range=(0.003, 0.06), binary_gripper=True
    )
    normalized = torch.full((1, 6, 8), 0.1, requires_grad=True)
    values = normalized.clone()
    values[0, :, 7] = torch.tensor([0.5, 0.4, 0.5, 0.6, 0.5, 0.4])
    committed = mapping.encode(offset.expand(1, 6, 8))
    projected = mapping(values, committed=committed)
    physical = mapping.decode(projected)
    torch.testing.assert_close(physical[..., :7], mapping.decode(values)[..., :7])
    torch.testing.assert_close(
        physical[0, :, 7], torch.tensor([0.04, 0.003, 0.003, 0.06, 0.06, 0.003])
    )
    projected[..., :7].sum().backward()
    torch.testing.assert_close(
        normalized.grad[..., :7], torch.ones_like(normalized.grad[..., :7])
    )


@pytest.mark.parametrize(
    "corruption", ["nan", "shape", "missing", "invalid", "malformed"]
)
def test_online_bad_preference_degrades_to_td_and_human_bc(corruption):
    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    feature = _online_feature(cfg, "bad-pair", 0)
    assembler = TransitionAssembler(cfg)
    assembler.begin(feature, torch.zeros(12, 8))
    infos = []
    for frame in range(12):
        value = torch.zeros(8)
        mask = torch.zeros(8, dtype=torch.bool)
        if frame == 7:
            value[2] = 0.1
            mask[2] = True
        infos.append(
            {
                "frame_index": frame,
                "action_receipt_physical": value,
                "human_mask": mask,
                "pair_mask": mask,
                "policy_proposal_physical": torch.zeros(8),
            }
        )
    if corruption == "nan":
        infos[7]["policy_proposal_physical"][2] = float("nan")
    elif corruption == "shape":
        infos[7]["policy_proposal_physical"] = torch.zeros(7)
    elif corruption == "missing":
        del infos[7]["policy_proposal_physical"]
    elif corruption == "malformed":
        infos[7]["policy_proposal_physical"] = ["unknown"] * 8
    else:
        infos[7]["policy_proposal_valid"] = False
    row = assembler.complete_from_receipts(
        feature.obs_id,
        step_infos=infos,
        encode_actions=lambda x: x,
        terminal=False,
        next_feature=_online_feature(cfg, "next", 12),
    )
    assert row.correction.human_mask[7, 2]
    assert not row.correction.pair_mask.any()
    assert row.relative_q_eligible and torch.isfinite(row.executed).all()
    assert row.proposal_trace is None or torch.isfinite(row.proposal_trace).all()
    from rlinf.algorithms.online_residual_rl.losses import behavior_cloning_losses

    ordinary, human = behavior_cloning_losses(
        torch.zeros(1, 12, 8),
        torch.ones(1, 12, 8),
        row.executed[None],
        valid_mask=torch.ones(1, 12, 8, dtype=torch.bool),
        human_mask=row.correction.human_mask[None],
    )
    assert ordinary == 0 and human == pytest.approx(0.01)


@pytest.mark.parametrize("bad_receipt", [False, True])
def test_online_collector_drops_only_windows_touching_timeout(tmp_path, bad_receipt):
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    replay = EpisodeReplay(cfg, storage_dir=tmp_path)
    metadata = {
        "forward_inputs": {
            "z_rl": torch.zeros(1, 4),
            "proprio": torch.zeros(1, 8),
            "base_ref": torch.zeros(1, 12, 7),
            "actor_output": torch.zeros(1, 12, 7),
        }
    }
    collector = EpisodeCollector(
        cfg,
        lambda x: x,
        lambda rows, success, terminal: replay.commit_episode(
            rows, success=success, terminal_frame=terminal
        ),
        episode_id="timeout-recovery",
    )
    for frame in range(48):
        collector.receipt(
            frame,
            {
                "action_receipt_physical": torch.full((7,), float("nan"))
                if bad_receipt and frame == 20
                else torch.zeros(7),
                "schedule_valid": frame != 20 if not bad_receipt else True,
            },
        )
        if frame % 6 == 5:
            collector.decision(frame - 5, metadata, 0)
    collector.finish(47, True)
    rows = replay.transitions()
    frames = sorted(row.feature.frame_index for row in rows)
    assert frames == [0, 24, 30, 36]
    assert any(row.terminal and row.rewards[-1] == 1 for row in rows)
    assert all(
        not (row.feature.frame_index <= 20 < row.feature.frame_index + row.valid_len)
        for row in rows
    )


def test_online_actor_rejects_old_proprio_normalization(tmp_path):
    from dataclasses import asdict

    from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state
    from rlinf.algorithms.online_residual_rl.networks import ResidualActor

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute",
        z_dim=4,
        hidden_dim=8,
        proprio_normalization="stage1",
        normalization_id="joint-stats",
    )
    actor = ResidualActor(cfg)
    path = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), path)
    with pytest.raises(ValueError, match="proprio normalization"):
        load_actor_state(path, cfg)
    torch.save({"actor": actor.state_dict(), "config": asdict(cfg)}, path)
    assert load_actor_state(path, cfg).keys() == actor.state_dict().keys()


def test_online_tcp_corrupt_preference_pose_keeps_real_receipt_and_bc():
    import numpy as np

    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    saved = []
    collector = EpisodeCollector(
        cfg,
        lambda x: x,
        lambda rows, success, terminal: saved.extend(rows),
        episode_id="tcp-bad-pair",
    )
    reference = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)
    metadata = {
        "forward_inputs": {
            "z_rl": torch.zeros(1, 4),
            "proprio": torch.zeros(1, 8),
            "base_ref": torch.zeros(1, 12, 7),
            "actor_output": torch.zeros(1, 12, 7),
        }
    }
    for frame in range(18):
        target = reference.copy()
        mask = np.zeros(7, dtype=bool)
        if frame == 7:
            target[0] = 0.01
            mask[0] = True
        collector.receipt(
            frame,
            {
                "action_receipt_physical": np.zeros(7),
                "tcp_target_pose": target,
                "policy_proposal_tcp_target_pose": np.full(7, np.nan)
                if frame == 7
                else reference,
                "policy_proposal_gripper_width_m": 0.0,
                "human_mask": mask,
                "pair_mask": mask,
            },
        )
        if frame % 6 == 5:
            collector.decision(frame - 5, metadata, 0, tcp_reference_pose=reference)
    collector.finish(17, True)
    assert len(saved) == 2
    assert saved[0].correction.human_mask[7, 0]
    assert not saved[0].correction.pair_mask.any()
    assert saved[0].executed[7, 0] == pytest.approx(0.01)


def test_online_pending_label_survives_restart_and_assigns_terminal_reward(tmp_path):
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector
    from rlinf.algorithms.online_residual_rl.storage import (
        PendingEpisodeStore,
        TransitionStore,
    )

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    pending = PendingEpisodeStore(tmp_path, cfg)
    submitted, metrics = [], []
    collector = EpisodeCollector(
        cfg,
        lambda values: values,
        lambda *args: submitted.append(args),
        episode_id="needs-label",
        pending_store=pending,
        record_metrics=metrics.append,
    )
    inputs = {
        "z_rl": torch.zeros(1, 4),
        "proprio": torch.zeros(1, 8),
        "base_ref": torch.zeros(1, 12, 8),
        "actor_output": torch.zeros(1, 12, 8),
    }
    collector.decision(0, {"forward_inputs": inputs}, 0)
    for frame in range(12):
        collector.receipt(
            frame,
            {
                "action_receipt_physical": torch.zeros(8),
                "action_representation": "joint_absolute",
            },
        )
    collector.finish(11, None)
    assert not submitted and not list(tmp_path.glob("*.pt"))
    assert metrics[0]["collection/valid_windows"] == 1
    draft = pending.pending.load_episode("needs-label")
    assert draft["success"] is None and not draft["transitions"][0]["rewards"].any()
    # A new owner can confirm the draft after the inference server exits.
    PendingEpisodeStore(tmp_path, cfg).label("needs-label", True)
    saved = TransitionStore(tmp_path, cfg).load_episode("needs-label")
    assert saved["success"] is True
    assert saved["transitions"][0]["rewards"].sum() == 1
    assert saved["transitions"][0]["return"] == pytest.approx(cfg.gamma**11)
    assert not list((tmp_path / "pending").glob("*.pt"))


@pytest.mark.parametrize("success", [True, False])
def test_online_process_ingests_offline_labels_once_and_records_metrics(
    tmp_path, success
):
    import os

    from rlinf.algorithms.online_residual_rl.learner_process import OnlineLearnerProcess
    from rlinf.algorithms.online_residual_rl.online import ActorMailbox
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.storage import PendingEpisodeStore
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    feature = _online_feature(cfg, "one", 0)
    assembler = TransitionAssembler(cfg)
    actions = torch.zeros(12, 8)
    human = torch.zeros_like(actions, dtype=torch.bool)
    human[7, 3] = True
    actions[7, 3] = 0.2
    assembler.begin(feature, torch.zeros_like(actions))
    row = assembler.complete(
        feature.obs_id,
        executed=actions,
        rewards=torch.zeros(12),
        terminal=True,
        valid_len=12,
        next_feature=None,
        human_mask=human,
        pair_mask=human,
    )
    episodes, checkpoints = tmp_path / "episodes", tmp_path / "checkpoints"
    PendingEpisodeStore(episodes, cfg).stage([row], 11)
    mailbox = ActorMailbox()
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=torch.nn.Identity()),
        EpisodeReplay(cfg, storage_dir=episodes),
        initial_episodes=1,
        warmup_updates=0,
        td_batch_size=2,
        supervision_batch_size=2,
        publish_interval=1,
        publish=mailbox.publish,
    )
    worker = OnlineLearnerProcess(training, checkpoint_dir=checkpoints)
    assert worker.pid != os.getpid()
    worker.record_metrics(
        {"collection/valid_windows": 1}, episode_id=feature.episode_id
    )
    try:
        PendingEpisodeStore(episodes, cfg).label(feature.episode_id, success)
        # Redelivery of this episode must not create more update credit.
        worker.submit([row], success, 11)
    finally:
        worker.close()
    state = torch.load(checkpoints / "latest.pt", weights_only=False)
    assert state["episodes"] == 1 and state["critic_updates"] == 5
    assert state["actor_updates"] == 1 and mailbox.take()[0] == 1
    journal = [
        json.loads(line)
        for line in (checkpoints / "metrics.jsonl").read_text().splitlines()
    ]
    assert all(entry["learner_pid"] == worker.pid for entry in journal)
    assert any(entry["metrics"].get("replay/pair_windows") == 1 for entry in journal)
    assert any("critic/q1/p95" in entry["metrics"] for entry in journal)
    assert any(
        entry["metrics"].get("actor/human_unreachable_fraction") == 1
        for entry in journal
    )
    assert training.actor_updates == 0  # Parent optimizer was never mutated.
    training.load(checkpoints / "latest.pt")
    resumed = OnlineLearnerProcess(training, checkpoint_dir=checkpoints)
    resumed.close()
    restored = torch.load(checkpoints / "latest.pt", weights_only=False)
    assert restored["critic_updates"] == 5 and restored["episodes"] == 1
    assert restored["budget"]["pending_critic"] == 0


@pytest.mark.parametrize("success", [True, False])
def test_online_process_submits_episode_with_limited_file_descriptors(
    tmp_path, success
):
    import os

    resource = pytest.importorskip("resource")

    from rlinf.algorithms.online_residual_rl.learner_process import OnlineLearnerProcess
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.storage import TransitionStore
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    features = [
        dataclasses.replace(
            _online_feature(cfg, f"obs-{i}", i * 12),
            committed=torch.full((6, 8), i / 100.0),
        )
        for i in range(48)
    ]
    assembler = TransitionAssembler(cfg)
    rows = []
    for index, feature in enumerate(features):
        actions = torch.full((12, 8), index / 100.0)
        assembler.begin(feature, actions)
        rows.append(
            assembler.complete(
                feature.obs_id,
                executed=actions,
                rewards=torch.zeros(12),
                terminal=index == len(features) - 1,
                valid_len=12,
                next_feature=features[index + 1] if index + 1 < len(features) else None,
            )
        )
    original_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    limited = min(256, original_limit[0])
    if len(os.listdir("/proc/self/fd")) + 32 >= limited:
        pytest.skip("Insufficient file descriptor headroom for the learner process")
    episodes, checkpoints = tmp_path / "episodes", tmp_path / "checkpoints"
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=torch.nn.Identity()),
        EpisodeReplay(cfg, storage_dir=episodes),
    )
    worker = OnlineLearnerProcess(training, checkpoint_dir=checkpoints)
    terminal_frame = features[-1].frame_index + 11
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (limited, original_limit[1]))
        try:
            worker.submit(rows, success, terminal_frame)
        finally:
            worker.close()
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, original_limit)

    state = torch.load(checkpoints / "latest.pt", weights_only=False)
    assert state["episodes"] == 1
    assert state["budget"]["pending_critic"] == len(rows) * 5
    store = TransitionStore(episodes, cfg)
    saved = store.load_episode(features[0].episode_id)
    assert saved["success"] is success
    restored = store.rows_from_payload(saved)
    assert len(restored) == len(rows)
    for expected, actual in zip(rows, restored, strict=True):
        torch.testing.assert_close(actual.executed, expected.executed)
        assert actual.feature.obs_id == expected.feature.obs_id
    assert restored[-1].rewards[-1].item() == float(success)


def test_online_replay_interns_successor_features_and_restores_ids():
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    assembler = TransitionAssembler(cfg)
    features = [_online_feature(cfg, f"obs-{frame}", frame) for frame in (0, 12)]
    rows = []
    for index, feature in enumerate(features):
        actions = torch.zeros(12, 7)
        assembler.begin(feature, actions)
        rows.append(
            assembler.complete(
                feature.obs_id,
                executed=actions,
                rewards=torch.zeros(12),
                terminal=index == 1,
                valid_len=12,
                next_feature=features[1] if index == 0 else None,
            )
        )
    replay = EpisodeReplay(cfg)
    replay.commit_episode(rows, success=False, terminal_frame=23)
    state = replay.state_dict()
    assert len(state["features"]) == 2
    assert state["rows"][0].next_obs_id == state["rows"][1].obs_id
    restored = EpisodeReplay(cfg)
    restored.load_state_dict(state)
    snapshot = restored.transitions()
    assert snapshot[0].next_feature is snapshot[1].feature
    assert restored.statistics()["replay/windows"] == 2
    legacy = EpisodeReplay(cfg)
    legacy.load_state_dict({**state, "schema_version": 1, "rows": snapshot})
    assert legacy.statistics()["replay/features"] == 2
    limited = EpisodeReplay(cfg, capacity=1)
    limited.commit_episode(rows, success=False, terminal_frame=23)
    assert limited.statistics()["replay/features"] == 1


def test_online_critic_only_does_not_read_current_reference_or_bc_labels():
    from rlinf.algorithms.online_residual_rl.transition import SparseCorrection

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    feature = _online_feature(cfg, "terminal", 0)
    assembler = TransitionAssembler(cfg)
    actions = torch.zeros(12, 7)
    assembler.begin(feature, actions)
    row = assembler.complete(
        feature.obs_id,
        executed=actions,
        rewards=torch.zeros(12),
        terminal=True,
        valid_len=12,
        next_feature=None,
    )

    class UnreadableBC:
        @property
        def human_mask(self):
            raise AssertionError("Critic-only updates must not access BC labels")

    # Omitted BC-only fields remain legal inputs to the TD update contract.
    row = dataclasses.replace(
        row,
        feature=dataclasses.replace(feature, base_reference=None),
        correction=UnreadableBC(),
    )
    learner = OnlineResidualLearner(cfg, action_projector=torch.nn.Identity())
    metrics = learner.critic_step([row], corrections=[])
    assert "critic/q1/mean" in metrics.values
    # Rank also skips both current reference and successor features.
    pair = torch.zeros_like(actions, dtype=torch.bool)
    pair[7, 0] = True
    ranked = dataclasses.replace(
        row, correction=SparseCorrection(pair, pair, torch.tensor([0.1]))
    )
    assert "critic/rank" in learner.critic_step([row], corrections=[ranked]).values


def test_flexiv_joint_stage1_loader_and_statistics_keep_absolute_commands(tmp_path):
    import io

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image

    from rlinf.data.datasets.openpi_rlinf.flexiv.flexiv_sft_dataset import (
        FlexivSftDataset,
    )
    from toolkits.lerobot.calculate_flexiv_norm_stats import iter_numeric_batches

    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    metadata = {
        "recorded_arm_names": ["left"],
        "action_representation": "joint_absolute",
        "total_frames": 3,
        "total_episodes": 2,
        "image_keys": ["observation.image", "observation.wrist_image"],
        "features": {
            "observation.eef_state": {"shape": [6]},
            "observation.joint_state": {"shape": [7]},
            "observation.gripper_state": {"shape": [1]},
            "action": {"shape": [8]},
        },
    }
    (tmp_path / "meta" / "info.json").write_text(json.dumps(metadata))
    (tmp_path / "meta" / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps({"episode_index": i, "length": length})
            for i, length in enumerate((2, 1))
        )
    )
    (tmp_path / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": "insert"})
    )
    stream = io.BytesIO()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(stream, format="PNG")
    commands = [[float(i)] * 7 + [0.01 * i] for i in (1, 2, 3)]
    for episode, (start, length) in enumerate(((0, 2), (2, 1))):
        pq.write_table(
            pa.table(
                {
                    "observation.eef_state": [[0.0] * 6] * length,
                    "observation.joint_state": [[0.7] * 7] * length,
                    "observation.gripper_state": [[0.06]] * length,
                    "action": commands[start : start + length],
                    "task_index": [0] * length,
                    "observation.image": [stream.getvalue()] * length,
                    "observation.wrist_image": [stream.getvalue()] * length,
                }
            ),
            tmp_path / "data" / f"episode_{episode:06d}.parquet",
        )
    dataset = FlexivSftDataset(
        data_path=str(tmp_path),
        action_horizon=4,
        action_representation="joint_absolute",
    )
    np.testing.assert_allclose(
        dataset[0]["actions"], [commands[0], commands[1], commands[1], commands[1]]
    )
    np.testing.assert_allclose(dataset[2]["actions"], [commands[2]] * 4)
    assert dataset[0]["observation/state"].shape == (8,)
    assert dataset[0]["observation/image"].shape == (8, 8, 3)
    batches = list(iter_numeric_batches(tmp_path, action_horizon=4))
    np.testing.assert_allclose(batches[0]["actions"][0, :, :8], dataset[0]["actions"])
    np.testing.assert_allclose(batches[1]["actions"][0, :, :8], dataset[2]["actions"])
    assert not batches[0]["actions"][..., 8:].any()
    from toolkits.lerobot.calculate_flexiv_norm_stats import compute_stats

    assets = tmp_path / "assets"
    compute_stats(tmp_path, assets, action_horizon=4)
    stats = json.loads((assets / "norm_stats.json").read_text())["norm_stats"]
    assert len(stats["state"]["mean"]) == 8
    assert len(stats["actions"]["mean"]) == 32
    assert stats["actions"]["mean"][7] > 0
    assert stats["actions"]["mean"][8:] == [0.0] * 24

    from rlinf.data.datasets.openpi_rlinf.flexiv.flexiv_sft_data_loader import (
        create_flexiv_sft_data_loader,
    )

    loader = create_flexiv_sft_data_loader(
        data_path=str(tmp_path),
        model_path=str(tmp_path),
        config_name="pi05_flexiv_joint_absolute",
        action_dim=32,
        action_horizon=32,
        max_token_len=200,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        seed=42,
        dist_rank=0,
        dist_world_size=1,
        data_kwargs={
            "repo_id": "flexiv_joint_absolute",
            "norm_stats_path": str(assets / "norm_stats.json"),
        },
    )
    observation, targets = next(iter(loader))
    assert observation.state.shape == (1, 32)
    assert not observation.state[:, 8:].any()
    assert targets.shape == (1, 32, 32)
    low, high = (np.asarray(stats["actions"][key])[:8] for key in ("q01", "q99"))
    decoded = (targets[0, 0, :8].numpy() + 1) / 2 * (high - low + 1e-6) + low
    np.testing.assert_allclose(decoded, commands[0], atol=1e-5)
    with pytest.raises(ValueError, match="action"):
        FlexivSftDataset(data_path=str(tmp_path), action_horizon=4)


def test_online_smoothness_sums_joint_dimensions_and_freezes_prefix_and_gripper():
    from rlinf.algorithms.online_residual_rl.losses import direct_delta_smoothness_loss

    committed = torch.zeros(1, 6, 8, requires_grad=True)
    execution = (
        torch.arange(1, 7, dtype=torch.float32)[None, :, None]
        .expand(1, 6, 8)
        .clone()
        .requires_grad_()
    )
    valid = torch.ones(1, 12, 8, dtype=torch.bool)
    loss = direct_delta_smoothness_loss(
        committed, execution, valid_mask=valid, smooth_dims=7, weights=(0.5, 2.0, 3.0)
    )
    # Seven identical ramps: six first differences, one second difference,
    # and two third differences at the prefix/execution boundary.
    assert loss.item() == pytest.approx(0.5 * 42 / 11 + 2 * 7 / 10 + 3 * 14 / 9)
    loss.backward()
    assert committed.grad is None
    assert execution.grad[:, :, :7].abs().sum() > 0
    assert not execution.grad[:, :, 7].any()


def test_online_smoothness_masks_padding_and_averages_each_window_equally():
    from rlinf.algorithms.online_residual_rl.losses import direct_delta_smoothness_loss

    prefix = torch.zeros(2, 6, 8)
    execution = torch.zeros(2, 6, 8)
    execution[0, :, :7] = torch.arange(1, 7, dtype=torch.float32)[:, None]
    execution[1, :2, :7] = torch.tensor([2.0, 4.0])[:, None]
    execution[1, 2:, :7] = 1e6  # Invalid terminal padding must not contribute.
    mask = torch.ones(2, 12, 8, dtype=torch.bool)
    mask[1, 8:] = False
    loss = direct_delta_smoothness_loss(
        prefix, execution, valid_mask=mask, smooth_dims=7, weights=(1.0, 0.0, 0.0)
    )
    assert loss.item() == pytest.approx((42 / 11 + 56 / 7) / 2)
    # A difference is valid only if every involved frame is valid.
    mask.zero_()
    assert (
        direct_delta_smoothness_loss(
            prefix, execution, valid_mask=mask, smooth_dims=7
        ).item()
        == 0
    )


def test_online_smoothness_uses_frame_mask_independent_of_bc_sampling():
    cfg = OnlineResidualRLConfig(
        action_representation="joint_absolute",
        z_dim=4,
        hidden_dim=8,
        smooth_weight=0.1,
        smooth_weights=[0.5, 2.0, 3.0],
        relative_q_weight=0.0,
        ordinary_bc_weight=0.0,
    )
    assert cfg.smooth_weights == (0.5, 2.0, 3.0)
    assembler = TransitionAssembler(cfg)
    rows = []
    for index in range(2):
        feature = _online_feature(cfg, f"smooth-{index}", index * 24)
        reference = feature.base_reference.clone()
        if index == 0:
            reference[6:, :7] = torch.arange(1, 7, dtype=torch.float32)[:, None] * 0.01
        feature = dataclasses.replace(feature, base_reference=reference)
        assembler.begin(feature, reference)
        rows.append(
            assembler.complete(
                feature.obs_id,
                executed=torch.zeros(12, 8),
                rewards=torch.zeros(12),
                terminal=False,
                valid_len=12,
                next_feature=_online_feature(cfg, f"next-{index}", index * 24 + 12),
            )
        )
    learner = OnlineResidualLearner(cfg, action_projector=torch.nn.Identity())
    metrics = learner.actor_step([rows[0]], ordinary=[rows[1]], human=[]).values
    expected = (0.5 * 42 / 11 + 2 * 7 / 10 + 3 * 14 / 9) * 0.01**2 / 2
    assert metrics["actor/smooth"] == pytest.approx(expected)
    assert metrics["actor/total"] == pytest.approx(0.1 * expected)
    assert metrics["actor/grad_norm"] > 0


def test_online_smoothness_disabled_needs_no_projection_or_extra_mask():
    from rlinf.algorithms.online_residual_rl.losses import actor_loss

    cfg = OnlineResidualRLConfig(action_representation="joint_absolute")
    actions = torch.ones(1, 12, 8)
    zeros = (torch.zeros(1, 1), torch.zeros(1, 1))
    loss = actor_loss(
        actions,
        torch.zeros_like(actions),
        actions,
        actions[:, :6],
        q_actor=zeros,
        q_reference=zeros,
        q_proposal=None,
        valid_mask=torch.ones_like(actions, dtype=torch.bool),
        human_mask=torch.zeros_like(actions, dtype=torch.bool),
        corrected=torch.zeros(1, dtype=torch.bool),
        relative_q_eligible=torch.zeros(1, dtype=torch.bool),
        config=cfg,
    )
    assert cfg.smooth_weight == 0 and loss.smooth.item() == 0
    assert loss.total.item() == 5  # Existing ordinary BC coefficient.


@pytest.mark.parametrize(
    "weights",
    [(1, 2), (1, 2, 3, 4), (1, -1, 1), (1, float("nan"), 1), (1, float("inf"), 1)],
)
def test_online_smoothness_rejects_invalid_order_coefficients(weights):
    with pytest.raises(ValueError, match="smooth_weights"):
        OnlineResidualRLConfig(smooth_weights=weights)


def test_online_smoothness_defaults_restore_older_checkpoint_contract(tmp_path):
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=torch.nn.Identity()),
        EpisodeReplay(cfg),
    )
    checkpoint = tmp_path / "legacy.pt"
    training.save(checkpoint)
    state = torch.load(checkpoint, weights_only=False)
    state["config"].pop("smooth_weights")
    state["replay"]["config"].pop("smooth_weights")
    torch.save(state, checkpoint)
    training.load(checkpoint)
    assert training.learner.config.smooth_weights == (1.0, 1.0, 1.0)
    different = dataclasses.replace(cfg, smooth_weights=(2.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="contract"):
        OnlineTraining(
            OnlineResidualLearner(different, action_projector=torch.nn.Identity()),
            EpisodeReplay(different),
        ).load(checkpoint)


def test_online_empty_episode_reports_no_transitions_saved(caplog):
    from rlinf.algorithms.online_residual_rl.online import EpisodeCollector

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    saved = []
    collector = EpisodeCollector(
        cfg,
        lambda actions: actions,
        lambda *args: saved.append(args),
        episode_id="no-fine",
    )
    for frame in range(18):
        collector.receipt(frame, {"action_receipt_physical": torch.zeros(7)})
    collector.finish(17, True)
    assert saved == []
    assert "No RL transitions saved for episode no-fine" in caplog.text
    assert "decisions=0 receipts=18" in caplog.text


def test_online_learner_services_new_episodes_and_checkpoints_during_backlog(
    tmp_path, monkeypatch
):
    import queue
    from types import SimpleNamespace

    from rlinf.algorithms.online_residual_rl import learner_process
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    rows = []
    for name in ("first", "second"):
        feature = dataclasses.replace(_online_feature(cfg, name, 0), episode_id=name)
        assembler = TransitionAssembler(cfg)
        actions = torch.zeros(12, 7)
        assembler.begin(feature, actions)
        rows.append(
            assembler.complete(
                feature.obs_id,
                executed=actions,
                rewards=torch.zeros(12),
                terminal=True,
                valid_len=12,
                next_feature=None,
            )
        )
    events = queue.Queue()
    events.put(("episode", learner_process._serialize(([rows[0]], True, 11))))
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=torch.nn.Identity()),
        EpisodeReplay(cfg, storage_dir=tmp_path / "episodes"),
        initial_episodes=1,
        warmup_updates=0,
        td_batch_size=2,
        supervision_batch_size=2,
    )
    checkpoints = tmp_path / "checkpoints"
    critic_step = training.learner.critic_step
    calls = []

    def step(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            events.put(("episode", learner_process._serialize(([rows[1]], False, 11))))
            events.put(None)
        if len(calls) == 2:
            saved = torch.load(checkpoints / "latest.pt", weights_only=False)
            assert saved["episodes"] == 2 and saved["critic_updates"] == 1
        if len(calls) == 3:
            saved = torch.load(checkpoints / "latest.pt", weights_only=False)
            assert saved["critic_updates"] == 2
        return critic_step(*args, **kwargs)

    monkeypatch.setattr(training.learner, "critic_step", step)
    ticks = iter(range(0, 10000, 11))
    monkeypatch.setattr(
        learner_process,
        "time",
        SimpleNamespace(
            monotonic=lambda: next(ticks),
            time=time.time,
        ),
    )
    learner_process.run_learner(training, events, checkpoints, 250)
    journal = [
        json.loads(line)
        for line in (checkpoints / "metrics.jsonl").read_text().splitlines()
    ]
    second = next(row for row in journal if row["episode_id"] == "second")
    assert second["critic_updates"] == 1
    saved = torch.load(checkpoints / "latest.pt", weights_only=False)
    assert saved["episodes"] == 2 and saved["critic_updates"] == 10
    assert saved["budget"]["pending_critic"] == 0


def test_online_process_failure_preserves_submitted_episode_and_reports_exit(tmp_path):
    import os
    import signal

    from rlinf.algorithms.online_residual_rl.learner_process import OnlineLearnerProcess
    from rlinf.algorithms.online_residual_rl.replay import EpisodeReplay
    from rlinf.algorithms.online_residual_rl.training import OnlineTraining

    cfg = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    feature = _online_feature(cfg, "one", 0)
    assembler = TransitionAssembler(cfg)
    actions = torch.zeros(12, 7)
    assembler.begin(feature, actions)
    row = assembler.complete(
        feature.obs_id,
        executed=actions,
        rewards=torch.zeros(12),
        terminal=True,
        valid_len=12,
        next_feature=None,
    )
    episodes, checkpoints = tmp_path / "episodes", tmp_path / "checkpoints"
    training = OnlineTraining(
        OnlineResidualLearner(cfg, action_projector=torch.nn.Identity()),
        EpisodeReplay(cfg, storage_dir=episodes),
    )
    worker = OnlineLearnerProcess(training, checkpoint_dir=checkpoints)
    os.kill(worker.pid, signal.SIGTERM)
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            try:
                worker.rollout_mode
            except RuntimeError:
                break
            time.sleep(0.02)
        else:
            pytest.fail("Learner exit was not reported")
        with pytest.raises(RuntimeError, match="Online learner failed"):
            worker.submit([row], True, 11)
        saved = training.replay.store.load_episode(feature.episode_id)
        assert saved["success"] is True
        assert saved["transitions"][0]["rewards"].sum() == 1
        error = (checkpoints / "learner_error.log").read_text()
        assert str(worker.pid) in error and "exitcode=-15" in error
    finally:
        with pytest.raises(RuntimeError, match="Online learner failed"):
            worker.close()
