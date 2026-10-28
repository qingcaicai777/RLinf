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


def test_online_transition_defaults_pair_mask_to_human_intervention():
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
    assert torch.equal(transition.correction.pair_mask, human_mask)
    torch.testing.assert_close(transition.correction.proposal(executed), proposal)
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
    proposal[: config.inference_horizon] = feature.committed
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
    torch.testing.assert_close(row.correction.proposal_values, torch.full((6,), 0.3))

    replay = EpisodeReplay(config, storage_dir=tmp_path)
    assert replay.commit_episode([row], success=True, terminal_frame=11) == 1
    saved = replay.store.load_episode(feature.episode_id)
    assert set(saved["features"]) == {feature.obs_id}
    assert saved["transitions"][0]["obs_id"] == feature.obs_id
    assert saved["transitions"][0]["correction_id"] == feature.obs_id
    torch.testing.assert_close(saved["transitions"][0]["exec"], row.executed)
    torch.testing.assert_close(
        saved["corrections"][feature.obs_id]["proposal_values"],
        row.correction.proposal_values,
    )
    assert saved["transitions"][0]["return"].item() == pytest.approx(0.99**11)
    assert len(list(tmp_path.glob("*.pt"))) == 1


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
    assert row.correction.pair_mask[6:8].all()


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


def test_online_action_projection_uses_physical_limits():
    from rlinf.algorithms.online_residual_rl.action_mapping import ActionMapping

    mapping = ActionMapping([0.0, 0.5], [0.1, 0.5], [-0.02, 0.0], [0.02, 1.0])
    actions = torch.tensor([[0.5, 3.0]], requires_grad=True)
    projected = mapping(actions)
    torch.testing.assert_close(mapping.decode(projected), torch.tensor([[0.02, 1.0]]))
    projected.sum().backward()
    assert actions.grad is not None


def test_flexiv_norm_batches_use_future_absolute_actions(tmp_path):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toolkits.lerobot.calculate_flexiv_norm_stats import iter_numeric_batches

    (tmp_path / "meta").mkdir()
    (tmp_path / "data").mkdir()
    metadata = {
        "recorded_arm_names": ["left"],
        "total_frames": 3,
        "features": {
            "observation.joint_state": {"shape": [7]},
            "observation.gripper_state": {"shape": [1]},
            "action": {"shape": [8]},
        },
    }
    info = tmp_path / "meta" / "info.json"
    info.write_text(json.dumps(metadata))
    pq.write_table(
        pa.table(
            {
                "observation.joint_state": [
                    [0.0] * 7,
                    [1.0] * 7,
                    [2.0] * 7,
                ],
                "observation.gripper_state": [[0.060], [0.004], [0.005]],
                "action": [
                    [0.0] * 7 + [0.060],
                    [1.0] * 7 + [0.004],
                    [2.0] * 7 + [0.005],
                ],
            }
        ),
        tmp_path / "data" / "episode_000000.parquet",
    )
    batches = list(iter_numeric_batches(tmp_path, batch_size=2, action_horizon=2))
    assert [len(batch["state"]) for batch in batches] == [2, 1]
    assert batches[0]["state"].shape == (2, 8)
    assert batches[0]["state"][0, 7] == 0.060
    assert batches[0]["actions"].shape == (2, 2, 8)
    np.testing.assert_allclose(batches[0]["actions"][:, :, 0], [[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(batches[1]["actions"][:, :, 0], [[2.0, 2.0]])
    np.testing.assert_allclose(
        batches[0]["actions"][:, :, 7], [[0.060, 0.004], [0.004, 0.005]]
    )


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
