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

"""Model registration, embeddings, and the reward-model helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.losses import compute_ppo_critic_loss
from rlinf.algorithms.online_residual_rl import (
    ActionRepresentation,
    OnlineResidualController,
    OnlineResidualRLConfig,
    ResidualActor,
    TwinQCritic,
)
from rlinf.config import SupportedModel
from rlinf.hybrid_engines.fsdp.utils import get_fsdp_wrap_policy
from rlinf.models import get_model, register_model
from rlinf.models.embodiment.modules.rlt_token_transformer import (
    RLTTokenTransformer,
)
from rlinf.models.embodiment.openpi.apxinf_adapter import OpenPIApxInfAdapter
from rlinf.scheduler import Worker
from rlinf.utils.env_helpers import HistoryManager
from rlinf.utils.env_helpers.delay_sampler import (
    ConstantDelaySampler,
    DelaySampler,
    ExponentialDelaySampler,
    GaussianDelaySampler,
    UniformDelaySampler,
)


class _DummyModel:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class _DummyBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)


class _DummyFSDPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = _DummyBlock()
        self.head = torch.nn.Linear(4, 2)
        self.head._fsdp_wrap_name = "custom_head"


def test_custom_model_registration_smoke():
    model_type = f"custom_model_smoke_{int(time.time() * 1000)}"
    received = {"torch_dtype": None}

    def _builder(cfg, torch_dtype):
        received["torch_dtype"] = torch_dtype
        return _DummyModel()

    register_model(model_type, _builder, category="embodied")

    supported_model = SupportedModel(model_type)
    assert supported_model.value == model_type

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    model = get_model(cfg)

    assert isinstance(model, _DummyModel)
    assert received["torch_dtype"] == torch.float32


def test_custom_model_registration_with_fsdp_wrap_policy():
    model_type = f"custom_model_fsdp_{int(time.time() * 1000)}"

    def _builder(cfg, torch_dtype):
        return _DummyFSDPModel()

    register_model(
        model_type,
        _builder,
        category="embodied",
    )

    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "fp32",
            "is_lora": False,
        }
    )
    fsdp_cfg = OmegaConf.create(
        {
            "wrap_policy": {
                "transformer_layer_cls_to_wrap": ["_DummyBlock"],
                "module_classes_to_wrap": ["_DummyBlock"],
                "no_split_names": ["custom_head"],
            },
            "use_orig_params": True,
        }
    )
    model = get_model(cfg)
    wrap_policy = get_fsdp_wrap_policy(
        module=model,
        config=fsdp_cfg,
        is_lora=False,
        model_type=model_type,
    )

    assert wrap_policy is not None
    assert wrap_policy(module=model.block, recurse=False, nonwrapped_numel=0)
    assert wrap_policy(module=model.head, recurse=False, nonwrapped_numel=0)


def _make_model(
    *, prefix_seq_len: int = 5, decoder_type: str = "autoregressive"
) -> RLTTokenTransformer:
    torch.manual_seed(0)
    return RLTTokenTransformer(
        input_dim=8,
        embed_dim=8,
        prefix_seq_len=prefix_seq_len,
        num_layers=1,
        num_heads=2,
        dropout_rate=0.0,
        decoder_type=decoder_type,
    )


def test_decoder_causal_mask_blocks_future_teacher_targets():
    model = _make_model()
    model.eval()
    rl_tokens = torch.randn(1, 1, model.embed_dim)
    targets = torch.randn(1, model.prefix_seq_len, model.input_dim)

    changed_targets = targets.clone()
    changed_targets[:, 2:] += 100.0

    original_output = model.decode(rl_tokens, targets)
    changed_output = model.decode(rl_tokens, changed_targets)

    # target[2:] enters decoder positions 3+, so positions 0..2 must not
    # change when causal attention prevents access to future positions.
    torch.testing.assert_close(
        original_output[:, :3],
        changed_output[:, :3],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.allclose(original_output[:, 3:], changed_output[:, 3:])


def test_loss_masks_trailing_padding():
    model = _make_model(prefix_seq_len=4)
    model.eval()
    prefix_embs = torch.randn(2, 4, model.input_dim)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ]
    )

    loss, _ = model.loss(prefix_embs, mask)
    reconstructed, _ = model.reconstruct(prefix_embs, mask)
    valid = mask.unsqueeze(-1).to(dtype=torch.float32)
    expected_loss = (
        torch.square(reconstructed.float() - prefix_embs.float()) * valid
    ).sum() / (valid.sum() * model.input_dim)
    torch.testing.assert_close(loss, expected_loss)

    changed_padding = prefix_embs.clone()
    changed_padding[~mask] += 1000.0
    changed_loss, _ = model.loss(changed_padding, mask)
    torch.testing.assert_close(loss, changed_loss, rtol=1e-5, atol=1e-5)


def test_reconstruct_output_shape_matches_prefix_embeddings():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(3, 4, model.input_dim)

    reconstructed, _ = model.reconstruct(prefix_embs)

    assert reconstructed.shape == prefix_embs.shape


def test_parallel_query_decoder_does_not_read_teacher_target_values():
    model = _make_model(prefix_seq_len=4, decoder_type="parallel_query")
    model.eval()
    rl_tokens = torch.randn(2, 1, model.embed_dim)
    targets = torch.randn(2, 4, model.input_dim)

    original = model.decode(rl_tokens, targets)
    changed = model.decode(rl_tokens, targets + 1000.0)

    torch.testing.assert_close(original, changed)
    assert original.shape == targets.shape


def test_online_residual_shapes_follow_selected_action_representation():
    eef = OnlineResidualRLConfig()
    joints = OnlineResidualRLConfig(
        action_representation=ActionRepresentation.JOINT_ABSOLUTE
    )

    assert (eef.action_dim, eef.actor_input_dim, eef.actor_output_dim) == (7, 2140, 84)
    assert (joints.action_dim, joints.actor_input_dim, joints.actor_output_dim) == (
        8,
        2152,
        96,
    )
    assert eef.rank_weight == 50.0


def test_online_residual_actor_starts_at_base_and_uses_committed_prefix():
    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    actor = ResidualActor(config)
    batch_size = 2
    z = torch.randn(batch_size, config.z_dim)
    proprio = torch.randn(batch_size, config.proprio_dim)
    reference = torch.randn(batch_size, config.actor_horizon, config.action_dim)
    committed = torch.randn(batch_size, config.inference_horizon, config.action_dim)

    actions = actor(z, proprio, reference)
    candidate = actor.candidate(committed, actions)

    torch.testing.assert_close(actions, reference)
    torch.testing.assert_close(candidate[:, : config.inference_horizon], committed)
    torch.testing.assert_close(
        candidate[:, config.inference_horizon :],
        reference[:, config.inference_horizon :],
    )


def test_online_residual_actor_bound_and_independent_critics():
    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    actor = ResidualActor(config)
    critic = TwinQCritic(config)
    batch_size = 2
    z = torch.randn(batch_size, config.z_dim)
    proprio = torch.randn(batch_size, config.proprio_dim)
    reference = torch.randn(batch_size, config.actor_horizon, config.action_dim)
    noise = torch.full_like(reference, 1000.0)

    actions = actor(z, proprio, reference, pre_tanh_noise=noise)
    q1, q2 = critic(z, proprio, actions)

    assert torch.all((actions - reference).abs() <= config.residual_scale + 1e-6)
    assert q1.shape == q2.shape == (batch_size, 1)
    assert critic.q1.mlp[0].weight.data_ptr() != critic.q2.mlp[0].weight.data_ptr()


def test_online_residual_controller_uses_final_receipts_as_next_committed_prefix():
    class Encoder(torch.nn.Module):
        def encode_flat(self, prefix):
            return prefix.mean(dim=1)

    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    actor = ResidualActor(config)
    controller = OnlineResidualController(
        config,
        rlt_encoder=Encoder(),
        actor=actor,
        action_decoder=lambda value: value * 2,
        action_encoder=lambda value: value / 2,
    )
    base = torch.zeros(1, config.actor_horizon, config.action_dim)
    prefix = torch.ones(1, 5, config.z_dim)
    proprio = torch.zeros(1, config.proprio_dim)

    first = controller.decide(
        prefix_hidden_states=prefix,
        proprio=proprio,
        base_reference=base,
        deterministic=True,
        committed=base[:, : config.inference_horizon],
    )
    physical_receipts = torch.full_like(first.decoded_execution, 0.4)
    second = controller.decide(
        prefix_hidden_states=prefix,
        proprio=proprio,
        base_reference=base,
        deterministic=True,
        committed=physical_receipts / 2,
    )

    torch.testing.assert_close(second.committed, physical_receipts / 2)
    assert controller.pending_proposal is not None


def test_online_residual_controller_applies_rollout_noise_only_to_new_actions():
    class Encoder(torch.nn.Module):
        def encode_flat(self, prefix):
            return prefix.mean(dim=1)

    class RecordingActor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.noise = None

        def forward(self, z, proprio, base_reference, *, pre_tanh_noise=None):
            del z, proprio
            self.noise = pre_tanh_noise
            return (
                base_reference
                if pre_tanh_noise is None
                else base_reference + pre_tanh_noise
            )

    config = OnlineResidualRLConfig(z_dim=4, proprio_dim=3, hidden_dim=16)
    actor = RecordingActor()
    controller = OnlineResidualController(
        config,
        rlt_encoder=Encoder(),
        actor=actor,
    )
    reference = torch.zeros(1, config.actor_horizon, config.action_dim)
    prefix = torch.ones(1, 5, config.z_dim)
    proprio = torch.zeros(1, config.proprio_dim)
    torch.manual_seed(17)
    decision = controller.decide(
        prefix_hidden_states=prefix,
        proprio=proprio,
        base_reference=reference,
        deterministic=False,
    )

    assert actor.noise is not None
    assert not actor.noise[:, : config.inference_horizon].any()
    assert actor.noise[:, config.inference_horizon :].any()
    assert decision.actor_output[:, : config.inference_horizon].count_nonzero() == 0

    base_decision = controller.decide(
        prefix_hidden_states=prefix,
        proprio=proprio,
        base_reference=reference,
        deterministic=False,
        use_actor=False,
    )
    torch.testing.assert_close(base_decision.actor_output, reference)


def test_apxinf_adapter_exposes_normalized_actions_and_prefix_for_sidecar(monkeypatch):
    class Model:
        action_horizon = 12
        action_dim = 32
        num_views = 2
        image_size = 224

        def infer_rgb(self, rgb, layout, tokens, noise=None):
            assert rgb.shape == (2, 224, 224, 3)
            assert layout == "nhwc"
            assert tokens.tolist() == [1, 2]
            assert noise is None
            return np.zeros((12, 32), dtype=np.float32)

        def infer_rgb_with_prefix(self, rgb, layout, tokens, noise=None):
            assert rgb.shape == (2, 224, 224, 3)
            assert layout == "nhwc"
            assert tokens.tolist() == [1, 2]
            assert noise is None
            return (
                np.zeros((12, 32), dtype=np.float32),
                np.ones((514, 2048), dtype=np.float32),
            )

    class Processor:
        def preprocess_batch(self, env_obs, *, num_views, image_size):
            del env_obs
            assert (num_views, image_size) == (2, 224)
            return [
                {
                    "rgb_u8": np.zeros((2, 224, 224, 3), dtype=np.uint8),
                    "token_ids": np.array([1, 2], dtype=np.uint32),
                    "state": np.zeros(8, dtype=np.float32),
                }
            ]

        def postprocess_batch(self, normalized, prepared, *, chunk_len=None):
            del prepared
            length = 12 if chunk_len is None else chunk_len
            return torch.from_numpy(normalized[:, :length, :7])

    cfg = {
        "action_dim": 7,
        "num_action_chunks": 12,
        "openpi": {
            "config_name": "pi05_flexiv_online_residual",
            "num_steps": 4,
            "noise_method": "flow_ode",
        },
        "apxinf": {
            "action_horizon": 12,
            "num_flow_steps": 4,
            "return_prefix_hidden_states": True,
        },
    }
    adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=Model(),
        processor=Processor(),
    )

    actions, result = adapter.predict_action_batch({}, mode="train")

    assert actions.shape == (1, 12, 7)
    assert result["normalized_actions"].shape == (1, 12, 32)
    assert result["prefix_hidden_states"].shape == (1, 514, 2048)
    adapter.close()
    adapter.close()

    class CudaModel(Model):
        def infer_rgb_with_prefix_cuda(self, rgb, layout, tokens, noise=None):
            assert noise is None
            return np.zeros((12, 32), dtype=np.float32), object(), object(), 1e-6

    monkeypatch.setattr(
        "rlinf.models.embodiment.openpi.apxinf_adapter._normalize_cuda_prefix",
        lambda prefix, scale, eps: torch.ones(514, 2048),
    )
    cuda_adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=CudaModel(),
        processor=Processor(),
    )
    _, cuda_result = cuda_adapter.predict_action_batch({})
    torch.testing.assert_close(
        cuda_result["prefix_hidden_states"], result["prefix_hidden_states"]
    )
    cuda_adapter.close()

    class NonBf16Model(CudaModel):
        def infer_rgb_with_prefix_cuda(self, *args, **kwargs):
            raise AssertionError("non-BF16 checkpoints must use the host prefix API")

    cfg["apxinf"]["precision"] = "int8"
    fallback_adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=NonBf16Model(),
        processor=Processor(),
    )
    _, fallback_result = fallback_adapter.predict_action_batch({})
    torch.testing.assert_close(
        fallback_result["prefix_hidden_states"], result["prefix_hidden_states"]
    )
    fallback_adapter.close()
    cfg["apxinf"].pop("precision")

    cfg["apxinf"].pop("num_flow_steps")
    cfg["openpi"].pop("num_steps")
    default_adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=Model(),
        processor=Processor(),
    )
    assert default_adapter.num_flow_steps == 10
    default_adapter.close()

    cfg["openpi"]["num_steps"] = 4
    cfg["apxinf"]["num_flow_steps"] = 10
    override_adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=Model(),
        processor=Processor(),
    )
    assert override_adapter.num_flow_steps == 10
    override_adapter.close()

    cfg["apxinf"]["return_prefix_hidden_states"] = False
    stage1_adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        model=Model(),
        processor=Processor(),
    )
    stage1_actions, stage1_result = stage1_adapter.predict_action_batch({})
    assert stage1_actions.shape == (1, 12, 7)
    assert "prefix_hidden_states" not in stage1_result
    stage1_adapter.close()


def test_apxinf_adapter_disables_splitkv_when_configured(monkeypatch):
    monkeypatch.delenv("APXINF_DISABLE_FA2_SPLITKV", raising=False)

    class Model:
        action_horizon = 12
        action_dim = 32
        num_views = 2
        image_size = 224

    class Processor:
        pass

    def load_model(*args, **kwargs):
        assert os.environ["APXINF_DISABLE_FA2_SPLITKV"] == "1"
        assert kwargs["num_flow_steps"] == 10
        return Model()

    monkeypatch.setitem(
        sys.modules,
        "apxinf_py",
        SimpleNamespace(Model=SimpleNamespace(load=load_model)),
    )
    cfg = {
        "model_path": "/tmp/openpi",
        "action_dim": 7,
        "num_action_chunks": 12,
        "openpi": {
            "config_name": "pi05_flexiv_online_residual",
            "num_steps": 4,
        },
        "apxinf": {
            "checkpoint": "/tmp/apx/model.safetensors",
            "tactics": "/tmp/tactics.json",
            "action_horizon": 12,
            "num_flow_steps": 10,
            "num_views": 2,
            "disable_fa2_splitkv": True,
        },
    }
    adapter = OpenPIApxInfAdapter(
        cfg,
        "cpu",
        processor=Processor(),
    )
    adapter.close()


def test_rlt_encoder_packs_masked_prefix_without_shifting_positions():
    from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder

    encoder = RLTTokenEncoder(
        input_dim=8, embed_dim=8, num_heads=2, prefix_seq_len=16
    ).eval()
    prefix = torch.randn(1, 10, 8)
    mask = torch.tensor(
        [[True, True, False, False, True, True, False, True, True, True]]
    )

    reference = encoder.encode_flat(prefix, mask)
    with torch.inference_mode():
        packed = encoder.encode_flat(prefix, mask)

    torch.testing.assert_close(packed, reference, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("normalization", ["raw", "stage1"])
def test_online_residual_apx_predicts_without_committed_or_decoder(
    tmp_path, normalization
):
    from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenEncoder
    from rlinf.models.embodiment.openpi.apxinf_adapter import (
        OnlineResidualApxInfAdapter,
    )

    encoder = RLTTokenEncoder(input_dim=8, embed_dim=8, num_heads=2, prefix_seq_len=16)
    checkpoint = tmp_path / "stage1"
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(
        {
            **{
                f"rlt_module.encoder.{key}": value
                for key, value in encoder.state_dict().items()
            },
            "img.encoder.layers.2.attn.weight": torch.ones(1),
        },
        weights,
    )

    class Processor:
        def postprocess_batch(self, normalized, prepared, *, chunk_len):
            assert chunk_len == 12
            assert normalized.shape == (1, 12, 32)
            assert not normalized[..., 7:].any()
            return torch.from_numpy(normalized[:, :chunk_len, :7])

    class Base:
        return_prefix_hidden_states = True
        metadata = {"backend": "test"}
        processor = Processor()
        model = SimpleNamespace(num_views=2, image_size=2, patch_size=1)

        def predict_action_batch(self, obs, *, mode):
            # No committed argument is supported by the VLA boundary.
            assert mode == "eval"
            return torch.zeros(1, 32, 7), {
                "normalized_actions": torch.zeros(1, 12, 32),
                "prefix_hidden_states": torch.ones(1, 10, 8),
                "prefix_mask": torch.ones(1, 10, dtype=torch.bool),
                "prepared": [{"state": np.full(8, -0.5, dtype=np.float32)}],
                "apxinf_timing": [],
            }

    adapter = OnlineResidualApxInfAdapter(
        {
            "num_action_chunks": 12,
            "action_dim": 7,
            "online_residual": {
                "z_dim": 8,
                "hidden_dim": 8,
                "proprio_normalization": normalization,
            },
            "rlt": {
                "checkpoint": str(checkpoint),
                "num_heads": 2,
                "prefix_seq_len": 16,
            },
        },
        "cpu",
        base=Base(),
    )
    seen = []
    encode_flat = adapter.rlt_module.encode_flat

    def capture_prefix(prefix, mask):
        seen.append((prefix.detach().clone(), mask.detach().clone()))
        return encode_flat(prefix, mask)

    adapter.rlt_module.encode_flat = capture_prefix
    observation = {"states": np.full((1, 8), 0.3, dtype=np.float32)}
    coarse, _ = adapter.predict_base_action_batch(observation, horizon=10)
    assert coarse.shape == (1, 32, 7)
    actions, info = adapter.predict_action_batch(observation)
    changed, candidate = adapter.predict_action_batch(
        observation, committed=torch.ones(1, 6, 7)
    )
    assert actions.shape == (1, 12, 7)
    torch.testing.assert_close(changed, actions)
    assert info["forward_inputs"]["policy_proposal"] is None
    torch.testing.assert_close(
        info["forward_inputs"]["proprio"],
        torch.full((1, 8), -0.5 if normalization == "stage1" else 0.3),
    )
    assert seen[0][0].shape == (1, 14, 8)
    assert seen[0][1][0].tolist() == [True] * 8 + [False] * 4 + [True] * 2
    if normalization == "raw":
        with pytest.raises(ValueError, match="finite"):
            adapter.predict_action_batch({"states": np.full((1, 8), np.nan)})
    torch.testing.assert_close(
        candidate["forward_inputs"]["policy_proposal"][:, :6], torch.ones(1, 6, 7)
    )
    assert not hasattr(adapter.rlt_module, "decoder")


def test_openpi_rlinf_stage1_adapter_returns_full_horizon_with_raw_guidance(
    monkeypatch,
):
    from rlinf.models.embodiment.openpi_rlinf.inference import (
        OpenPIRLInfResidualAdapter,
        OpenPIRLInfStage1Adapter,
    )
    from rlinf.models.embodiment.openpi_rlinf.sampling import rtc_guidance

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.rlt_cfg = SimpleNamespace(use_rlt=True, rlt_normalized_actions=True)
            self.action_horizon = 32
            self.action_dim = 32
            self.action_env_dim = 7
            self.num_steps = 10
            self.rtc_guidance_clip = 5.0
            self.contexts = []

        @property
        def device(self):
            return self.weight.device

        def extract_rlt_obs(self, obs, *, rtc_context=None):
            del obs
            self.contexts.append(rtc_context)
            if rtc_context is not None:
                assert rtc_context.delay_steps == 6
                assert rtc_context.get_prev_remaining().shape == (1, 26, 32)
                torch.testing.assert_close(
                    rtc_context.get_prev_remaining()[0, :, :7], torch.ones(26, 7)
                )
            return {
                "z_rl": torch.zeros(1, 4),
                "proprio": torch.zeros(1, 8),
                "ref_chunk": torch.zeros(1, 32, 7),
                "raw_model_actions": torch.arange(32 * 32).reshape(1, 32, 32).float(),
                "model_state": torch.zeros(1, 8),
            }

        def decode_rlt_actions(self, actions, state):
            assert state.shape == (1, 8)
            return actions + 0.25

        def predict_action_batch(self, obs, mode="eval"):
            assert mode == "eval"
            assert obs["states"].shape == (1, 8)
            return torch.full((1, 32, 7), 0.25), {
                "model_actions": torch.zeros(1, 32, 32)
            }

        def env_obs_to_observation(self, obs):
            assert obs["states"].shape == (1, 8)
            return SimpleNamespace(state=torch.zeros(1, 8))

        def normalize_rlt_actions(self, actions):
            return actions - 0.25

    contract = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    model = Model()
    adapter = OpenPIRLInfResidualAdapter(model, contract)
    observation = {"states": np.zeros((1, 8), np.float32)}
    adapter.predict_action_batch(observation)
    physical, metadata = adapter.predict_action_batch(
        observation, committed=torch.ones(1, 26, 32)
    )
    assert model.contexts[0] is None and model.contexts[1] is not None
    assert physical.shape == (1, 12, 7)
    torch.testing.assert_close(physical, torch.full_like(physical, 0.25))
    assert metadata["forward_inputs"]["policy_proposal"] is None
    assert metadata["raw_model_actions"].shape == (1, 32, 32)
    torch.testing.assert_close(
        adapter.encode_receipts(physical), torch.zeros_like(physical)
    )

    def guided_actions(
        model, observation, context, *, num_steps, guidance_clip, guidance_mode
    ):
        assert model is not None
        assert observation.state.shape == (1, 8)
        assert context.delay_steps == 6
        assert context.get_prev_remaining().shape == (1, 26, 32)
        assert num_steps == 10 and guidance_clip == 5.0
        assert guidance_mode == "approx"
        return torch.zeros(1, 32, 32)

    monkeypatch.setattr(
        rtc_guidance, "sample_actions_with_rtc_guidance", guided_actions
    )
    stage1 = OpenPIRLInfStage1Adapter(model, inference_horizon=6)
    first, first_metadata = stage1.predict_action_batch(observation)
    second, second_metadata = stage1.predict_action_batch(
        observation, committed=torch.ones(1, 26, 32)
    )
    assert first.shape == second.shape == (1, 32, 7)
    assert first_metadata["raw_model_actions"].shape == (1, 32, 32)
    assert second_metadata["raw_model_actions"].shape == (1, 32, 32)
    assert len(model.contexts) == 2
    torch.testing.assert_close(stage1.encode_receipts(first), torch.zeros_like(first))

    without_rtc = OpenPIRLInfStage1Adapter(model, inference_horizon=6, use_rtc=False)
    assert not without_rtc.conditions_on_committed
    no_rtc_actions, _ = without_rtc.predict_action_batch(
        observation, committed=torch.ones(1, 26, 32)
    )
    assert len(model.contexts) == 2
    torch.testing.assert_close(no_rtc_actions, first)

    longer = OpenPIRLInfStage1Adapter(model, inference_horizon=10, use_rtc=False)
    longer_actions, _ = longer.predict_action_batch(observation)
    assert longer_actions.shape == (1, 32, 7)

    def guided_long(
        _model, _observation, context, *, num_steps, guidance_clip, guidance_mode
    ):
        assert context.delay_steps == 10
        assert context.get_prev_remaining().shape == (1, 22, 32)
        return torch.zeros(1, 32, 32)

    monkeypatch.setattr(rtc_guidance, "sample_actions_with_rtc_guidance", guided_long)
    longer_rtc = OpenPIRLInfStage1Adapter(model, inference_horizon=10)
    guided, _ = longer_rtc.predict_action_batch(
        observation, committed=torch.ones(1, 22, 32)
    )
    assert guided.shape == (1, 32, 7)
    staged = OpenPIRLInfResidualAdapter(model, contract, base_horizon=10)
    coarse, _ = staged.predict_base_action_batch(observation, horizon=10)
    assert coarse.shape == (1, 32, 7)

    def guided_short(
        _model, _observation, context, *, num_steps, guidance_clip, guidance_mode
    ):
        assert context.delay_steps == 0
        assert context.get_prev_remaining().shape == (1, 1, 32)
        return torch.zeros(1, 32, 32)

    monkeypatch.setattr(rtc_guidance, "sample_actions_with_rtc_guidance", guided_short)
    nearly_full = OpenPIRLInfStage1Adapter(model, inference_horizon=31)
    short_overlap, _ = nearly_full.predict_base_action_batch(
        observation,
        horizon=31,
        committed=torch.ones(1, 1, 32),
        rtc_delay_steps=0,
    )
    assert short_overlap.shape == (1, 32, 7)
    OpenPIRLInfStage1Adapter(model, inference_horizon=32)
    with pytest.raises(ValueError, match="fit in the model horizon"):
        OpenPIRLInfStage1Adapter(model, inference_horizon=33)


def test_stage1_apx_export_maps_base_weights_without_rlt_or_unused_expert_head():
    from rlinf.utils.ckpt_convertor.openpi.sft_to_apxinf import prepare_apx_weights

    bare = {
        "img.stem.weight": torch.ones(2, 3),
        "llm.embedder.embedding.weight": torch.ones(4, 2),
        "action_in_proj.weight": torch.ones(2, 7),
        "action_out_proj.weight": torch.ones(7, 2),
        "time_mlp_in.weight": torch.ones(2, 2),
        "time_mlp_out.weight": torch.ones(2, 2),
        "rlt_module.encoder.prefix_pos_enc": torch.ones(1, 2),
    }
    converted = prepare_apx_weights(bare)
    assert all(not key.startswith("rlt_module.") for key in converted)
    assert "paligemma_with_expert.gemma_expert.lm_head.weight" not in converted
    assert "paligemma_with_expert.paligemma.lm_head.weight" in converted
    assert all(tensor.dtype == torch.bfloat16 for tensor in converted.values())

    with pytest.raises(ValueError, match="missing APX PI0.5 weights"):
        prepare_apx_weights({"llm.embedder.embedding.weight": torch.ones(4, 2)})


@pytest.mark.parametrize(
    "representation,width", [("eef_delta", 7), ("joint_absolute", 8)]
)
def test_openpi_rlinf_stage1_loader_uses_sft_shape(
    monkeypatch, tmp_path, representation, width
):
    from rlinf.models.embodiment.openpi_rlinf import inference

    captured = {}

    class Model:
        def to(self, device):
            captured["device"] = device
            return self

        def eval(self):
            return self

    def fake_get_model(config):
        captured["config"] = config
        return Model()

    monkeypatch.setattr(inference, "get_model", fake_get_model)
    import json

    recipe = Path(
        "examples/sft/config/flexiv_joint_stage1_inference.yaml"
        if width == 8
        else "examples/sft/config/flexiv_rlt_stage1_sft_openpi_pi05.yaml"
    )
    norm_stats = tmp_path / "norm_stats.json"
    stats = {
        name: {
            "mean": [0.0] * size,
            "std": [1.0] * size,
            "q01": [-1.0] * size,
            "q99": [1.0] * size,
        }
        for name, size in (("state", 8), ("actions", width))
    }
    norm_stats.write_text(json.dumps({"norm_stats": stats}), encoding="utf-8")
    inference.load_stage1_model(
        recipe,
        "/tmp/stage1",
        "cpu",
        norm_stats_path=norm_stats,
        action_representation=representation,
    )
    cfg = captured["config"]
    assert cfg.model_path == "/tmp/stage1"
    assert cfg.openpi.task == "eval"
    assert cfg.openpi.action_horizon == 32
    assert cfg.openpi.action_chunk == 32
    assert cfg.num_steps == cfg.openpi.num_steps == 10
    assert cfg.openpi.rlt_prefix_seq_len == 1024
    assert cfg.action_dim == cfg.openpi.action_env_dim == width
    assert cfg.openpi_data.norm_stats_path == str(norm_stats)
    assert captured["device"] == "cpu"
    inference.load_stage1_model(
        recipe,
        "/tmp/stage1",
        "cpu",
        norm_stats_path=norm_stats,
        num_flow_steps=6,
        action_representation=representation,
    )
    assert captured["config"].num_steps == captured["config"].openpi.num_steps == 6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph needs a GPU")
def test_openpi_cuda_graph_suffix_uses_current_actions_and_prefix_cache():
    """Graph replay must read new request inputs, not captured values."""
    from rlinf.models.embodiment.openpi_rlinf.sampling.cuda_graph import (
        CudaGraphedSuffix,
    )

    class Model:
        pi05 = True
        device = torch.device("cuda:0")

        def run_suffix(self, _observation, actions, timestep, cache, prefix_mask):
            key, value = cache[0]
            return (
                actions
                + timestep[:, None, None]
                + key.mean()
                + value.mean()
                + prefix_mask.float().mean()
            )

        def velocity_from_suffix(self, suffix):
            return suffix * 2

    model = Model()
    graph = CudaGraphedSuffix(model)
    with torch.inference_mode():
        for offset in (0.0, 1.0, 2.0):
            actions = torch.full((1, 2, 3), offset, device="cuda")
            timestep = torch.tensor([offset / 10], device="cuda")
            cache = (
                (
                    torch.full((1, 2, 3), offset + 1, device="cuda"),
                    torch.full((1, 2, 3), offset + 2, device="cuda"),
                ),
            )
            prefix_mask = torch.tensor(
                [[True, offset > 0]], device="cuda", dtype=torch.bool
            )
            expected = model.velocity_from_suffix(
                model.run_suffix(None, actions, timestep, cache, prefix_mask)
            )
            actual = graph(None, actions, timestep, cache, prefix_mask)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # A guided step uses autograd independently of the replay buffers.
            with torch.inference_mode(False), torch.enable_grad():
                guided_actions = actions.clone().requires_grad_(True)
                guided = model.velocity_from_suffix(
                    model.run_suffix(None, guided_actions, timestep, cache, prefix_mask)
                )
                gradient = torch.autograd.grad(guided.sum(), guided_actions)[0]
                torch.testing.assert_close(gradient, torch.full_like(gradient, 2))


@pytest.mark.parametrize(
    "asset_dir,recipe_name",
    [
        ("flexiv_delta_assets", "flexiv_rlt_stage1_sft_openpi_pi05.yaml"),
        ("flexiv_joint_assets", "flexiv_joint_stage1_inference.yaml"),
    ],
)
def test_stage1_norm_stats_defaults_to_checkpoint_sibling(
    tmp_path, asset_dir, recipe_name
):
    from rlinf.models.embodiment.openpi_rlinf.inference import (
        resolve_stage1_norm_stats,
    )

    weights = tmp_path / "model_state_dict" / "full_weights.pt"
    weights.parent.mkdir()
    weights.touch()
    stats = tmp_path / asset_dir / "norm_stats.json"
    stats.parent.mkdir()
    stats.write_text("{}", encoding="utf-8")
    recipe = Path("examples/sft/config") / recipe_name
    assert resolve_stage1_norm_stats(recipe, weights) == stats.resolve()


def test_online_residual_actor_loads_published_and_training_weights(tmp_path):
    from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state

    actor = ResidualActor(OnlineResidualRLConfig(z_dim=4, hidden_dim=8))
    published = tmp_path / "actor.pt"
    torch.save(actor.state_dict(), published)
    assert load_actor_state(published).keys() == actor.state_dict().keys()

    training = tmp_path / "training.pt"
    learner = {
        **{f"actor.{key}": value for key, value in actor.state_dict().items()},
        "target_actor.extra": torch.tensor(1),
    }
    torch.save({"learner": learner, "config": {}}, training)
    assert load_actor_state(training).keys() == actor.state_dict().keys()


def test_openpi_rlinf_rtc_sampler_reuses_stage1_prefix_cache():
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        sample_actions_with_rtc_guidance,
    )

    class Model:
        action_horizon = 12
        action_dim = 32

        def __init__(self, velocity=0.0):
            self.velocity = velocity

        def build_prefix_cache(self, observation):
            raise AssertionError("Prefix cache must be reused")

        def run_suffix(self, observation, x_t, t, kv_cache, prefix_mask):
            del observation, t, kv_cache, prefix_mask
            return torch.full_like(x_t, self.velocity)

        def velocity_from_suffix(self, suffix):
            return suffix

    observation = SimpleNamespace(state=torch.zeros(1, 8))
    result = sample_actions_with_rtc_guidance(
        Model(),
        observation,
        None,
        num_steps=1,
        noise=torch.zeros(1, 12, 32),
        prefix_cache=(torch.ones(1, 1, dtype=torch.bool), ()),
    )
    torch.testing.assert_close(result, torch.zeros(1, 12, 32))

    noise = torch.linspace(-0.5, 0.5, 12 * 32).reshape(1, 12, 32)
    legacy_rng = torch.Generator().manual_seed(41)
    for _ in range(3):
        torch.randn(noise.shape, generator=legacy_rng)
    optimized_rng = torch.Generator().manual_seed(41)
    actual = sample_actions_with_rtc_guidance(
        Model(0.125),
        observation,
        None,
        num_steps=3,
        noise=noise,
        rng=optimized_rng,
        prefix_cache=(torch.ones(1, 1, dtype=torch.bool), ()),
    )
    torch.testing.assert_close(actual, noise - 0.125, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        torch.randn(4, generator=optimized_rng),
        torch.randn(4, generator=legacy_rng),
    )


@pytest.mark.parametrize("num_steps", (1, 3, 10, 20))
@pytest.mark.parametrize("guidance_clip", (1.0, 10.0, 100.0))
def test_openpi_rlinf_rtc_approx_stays_within_overlap_target(num_steps, guidance_clip):
    from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        sample_actions_with_rtc_guidance,
    )

    class ConstantModel:
        action_horizon = 3
        action_dim = 1

        def __init__(self):
            self.inputs = []

        def sample_velocity(self, observation, x_t, timestep, kv_cache, prefix_mask):
            self.inputs.append(x_t.clone())
            return torch.zeros_like(x_t)

    model = ConstantModel()
    rng = torch.Generator().manual_seed(41)
    result = sample_actions_with_rtc_guidance(
        model,
        SimpleNamespace(state=torch.zeros(1, 1)),
        RTCGuidanceContext(prev_model_actions=torch.zeros(1, 2, 1), delay_steps=1),
        num_steps=num_steps,
        noise=torch.ones(1, 3, 1),
        rng=rng,
        guidance_clip=guidance_clip,
        guidance_mode="approx",
        prefix_cache=(torch.ones(1, 1, dtype=torch.bool), ()),
    )
    trajectory = torch.stack([*model.inputs, result])
    assert bool(((trajectory >= 0) & (trajectory <= 1)).all())
    assert bool((trajectory[1:] <= trajectory[:-1]).all())
    assert result[0, 0, 0] <= result[0, 1, 0] < 1
    torch.testing.assert_close(trajectory[:, 0, 2, 0], torch.ones(num_steps + 1))
    first_change = trajectory[0, 0, 0, 0] - trajectory[1, 0, 0, 0]
    assert first_change <= min(guidance_clip / num_steps, 1.0) + 1e-6

    expected_rng = torch.Generator().manual_seed(41)
    for _ in range(num_steps):
        torch.randn(result.shape, generator=expected_rng)
    torch.testing.assert_close(
        torch.randn(4, generator=rng), torch.randn(4, generator=expected_rng)
    )


def test_openpi_rlinf_rtc_jacobian_mode_uses_clean_action_vjp():
    from rlinf.models.embodiment.openpi.rtc_guidance import (
        RTCGuidanceContext,
        build_rtc_target_and_mask,
    )
    from rlinf.models.embodiment.openpi_rlinf.sampling.rtc_guidance import (
        sample_actions_with_rtc_guidance,
    )

    class DifferentiableModel:
        action_horizon = 2
        action_dim = 1

        def sample_velocity(self, observation, x_t, timestep, kv_cache, prefix_mask):
            del observation, timestep, kv_cache, prefix_mask
            return x_t * 0.0 + 0.125

    observation = SimpleNamespace(state=torch.zeros(1, 1))
    noise = torch.tensor([[[0.1], [0.2]]])
    context = RTCGuidanceContext(
        prev_model_actions=torch.full_like(noise, 0.25), delay_steps=1
    )
    result = sample_actions_with_rtc_guidance(
        DifferentiableModel(),
        observation,
        context,
        num_steps=1,
        noise=noise,
        guidance_clip=5.0,
        guidance_mode="jacobian",
        prefix_cache=(torch.ones(1, 1, dtype=torch.bool), ()),
    )
    target, mask = build_rtc_target_and_mask(
        context.get_prev_remaining(), 2, 1, 1, noise.device, noise.dtype
    )
    mean = noise - 0.125
    expected = mean + 5.0 * (target - mean) * mask
    torch.testing.assert_close(result, expected)


def test_apxinf_rollout_forwards_real_committed_prefix_only_for_online_policy():
    from rlinf.workers.rollout.apxinf.apxinf_worker import ApxInfRolloutWorker

    class Adapter:
        def __init__(self):
            self.received = None

        def predict_action_batch(self, obs, *, mode, **kwargs):
            self.received = kwargs
            return torch.zeros(1, 12, 7), {}

        def close(self):
            pass

    worker = ApxInfRolloutWorker.__new__(ApxInfRolloutWorker)
    worker.apxinf_adapter = Adapter()
    worker.model_cfg = {"online_residual": {"enabled": True}}
    committed = torch.ones(1, 6, 7)
    assert worker.predict({"committed": committed}).shape == (1, 12, 7)
    torch.testing.assert_close(worker.apxinf_adapter.received["committed"], committed)

    worker.model_cfg = {"online_residual": {"enabled": False}}
    worker.predict({"committed": committed})
    assert worker.apxinf_adapter.received == {}


def test_flexiv_input_transform_preserves_both_camera_views():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.policies.franka_policy import FrankaEEInputs

    main = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.full_like(main, 37)
    observation = {
        "observation/image": main,
        "observation/wrist_image": wrist,
        "observation/state": np.zeros(8),
    }
    transformed = FrankaEEInputs(action_dim=32, pad_state=False, use_wrist_image=True)(
        observation
    )
    assert sum(bool(value) for value in transformed["image_mask"].values()) == 2
    np.testing.assert_array_equal(transformed["image"]["left_wrist_0_rgb"], wrist)
    assert transformed["state"].shape == (8,)


def test_reconstruct_detaches_targets_but_trains_encoder_and_decoder():
    model = _make_model(prefix_seq_len=4)
    prefix_embs = torch.randn(2, 4, model.input_dim, requires_grad=True)

    loss, _ = model.loss(prefix_embs)
    loss.backward()

    assert prefix_embs.grad is None
    encoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    decoder_grad_norm = sum(
        parameter.grad.abs().sum().item()
        for parameter in model.decoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad_norm > 0
    assert decoder_grad_norm > 0


class _FakeValueExpert:
    def __init__(self, image_emb, lang_emb):
        self.image_emb = image_emb
        self.lang_emb = lang_emb

    def embed_image(self, image):
        return self.image_emb.to(device=image.device)

    def embed_language_tokens(self, tokens):
        return self.lang_emb.to(device=tokens.device)


def _load_value_critic_model(monkeypatch):
    value_model_dir = (
        Path(__file__).resolve().parents[2]
        / "rlinf/models/embodiment/value_model/recap"
    )
    package_name = "value_model_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(value_model_dir)]
    monkeypatch.setitem(sys.modules, package_name, package)

    module_name = f"{package_name}.modeling_critic"
    spec = importlib.util.spec_from_file_location(
        module_name,
        value_model_dir / "modeling_critic.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module.ValueCriticModel


def test_value_model_does_not_rescale_gemma3_language_embeddings(monkeypatch):
    """Gemma3 embed_tokens already applies sqrt(hidden_size) internally."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("transformers.Gemma3ForCausalLM")

    ValueCriticModel = _load_value_critic_model(monkeypatch)

    hidden_size = 4
    image_emb = torch.zeros(1, 2, hidden_size)
    lang_emb = torch.arange(12, dtype=torch.float32).reshape(1, 3, hidden_size)

    model = SimpleNamespace(
        gradient_checkpointing_enabled=False,
        training=False,
        value_expert=_FakeValueExpert(image_emb=image_emb, lang_emb=lang_emb),
        _apply_checkpoint=lambda func, *args: func(*args),
    )

    prefix_embs, prefix_pad_masks = ValueCriticModel.embed_prefix(
        model,
        images=[torch.empty(1, 3, 8, 8)],
        img_masks=[torch.tensor([True])],
        lang_tokens=torch.tensor([[1, 2, 3]]),
        lang_masks=torch.tensor([[True, True, False]]),
    )

    torch.testing.assert_close(prefix_embs[:, 2:], lang_emb)
    torch.testing.assert_close(
        prefix_pad_masks,
        torch.tensor([[True, True, True, True, False]]),
    )


_STARVLA_UTILS_DIR = (
    Path(__file__).resolve().parents[2] / "rlinf/models/embodiment/starvla/utils"
)
_FRANKA_ACTION_STATS = {
    "q01": [-0.5] * 7,
    "q99": [0.5] * 7,
    "min": [-1.0] * 7,
    "max": [1.0] * 7,
    "mask": [True] * 6 + [False],
}


def _load_starvla_util(name: str) -> ModuleType:
    # The starvla package __init__ imports starVLA, which only its venv has.
    spec = importlib.util.spec_from_file_location(
        f"starvla_{name}_under_test", _STARVLA_UTILS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("source", "bound"), [("q01q99", 0.5), ("minmax", 1.0)])
def test_starvla_action_stats_follow_the_configured_source(source, bound):
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    stats = action_space.resolve_action_norm_stats(
        model, "franka", action_dim=7, action_stats_source=source
    )

    np.testing.assert_array_equal(stats["q99"], [bound] * 7)
    np.testing.assert_array_equal(stats["q01"], [-bound] * 7)
    np.testing.assert_array_equal(stats["mask"], [True] * 6 + [False])


def test_starvla_action_stats_name_the_available_keys_for_an_unknown_key():
    action_space = _load_starvla_util("action_space")
    model = SimpleNamespace(norm_stats={"franka": {"action": _FRANKA_ACTION_STATS}})

    with pytest.raises(RuntimeError, match=r"available keys: \['franka'\]"):
        action_space.resolve_action_norm_stats(model, "libero_spatial", action_dim=7)


def test_starvla_action_stats_require_a_norm_stats_mapping():
    action_space = _load_starvla_util("action_space")

    with pytest.raises(RuntimeError, match="no usable 'norm_stats' mapping"):
        action_space.resolve_action_norm_stats(
            SimpleNamespace(norm_stats=None), "franka", action_dim=7
        )


def test_starvla_env_actions_keep_their_shape_and_map_the_libero_gripper(monkeypatch):
    action_space = _load_starvla_util("action_space")
    received_shapes = []

    def unnormalize_actions(actions, action_norm_stats):
        received_shapes.append(actions.shape)
        return actions

    tools = ModuleType("starVLA.model.tools")
    tools.FrameworkTools = SimpleNamespace(unnormalize_actions=unnormalize_actions)
    monkeypatch.setitem(sys.modules, "starVLA.model.tools", tools)

    normalized = np.zeros((2, 3, 7), dtype=np.float32)
    normalized[..., 0] = 0.25
    normalized[0, :, 6] = 1.0
    stats = {"q99": np.ones(7), "q01": -np.ones(7), "mask": np.ones(7, dtype=bool)}

    env_actions = action_space.unnormalize_actions_for_env(
        normalized, stats, policy_setup="libero"
    )

    # starVLA unnormalizes [T, action_dim]; the chunk layout comes back intact.
    assert received_shapes == [(6, 7)]
    assert env_actions.shape == (2, 3, 7)
    np.testing.assert_array_equal(env_actions[..., 0], 0.25)
    # LIBERO wants the 0/1 gripper as -1 (open) / +1 (closed).
    np.testing.assert_array_equal(env_actions[0, :, 6], -1.0)
    np.testing.assert_array_equal(env_actions[1, :, 6], 1.0)


def test_starvla_autocast_targets_the_worker_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    # CPU stands in for a non-CUDA accelerator such as an Ascend NPU.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu")

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert torch.is_autocast_enabled("cpu")
        assert torch.get_autocast_dtype("cpu") == torch.bfloat16


def test_starvla_autocast_is_a_noop_without_an_accelerator(monkeypatch):
    accelerator = _load_starvla_util("accelerator")
    monkeypatch.setattr(Worker, "torch_device_type", None)

    with accelerator.accelerator_autocast(torch.bfloat16):
        assert not torch.is_autocast_enabled("cpu")
        assert not torch.is_autocast_enabled("cuda")


def test_starvla_gaussian_is_float32_and_keeps_the_gradient_path():
    accelerator = _load_starvla_util("accelerator")
    mean = torch.zeros(2, 3, dtype=torch.bfloat16, requires_grad=True)
    log_std = torch.nn.Parameter(torch.zeros(3))

    dist = accelerator.build_gaussian(mean, log_std.exp())
    sample = dist.rsample()

    assert dist.loc.dtype == dist.scale.dtype == sample.dtype == torch.float32
    dist.log_prob(sample.detach()).sum().backward()
    assert mean.grad is not None and mean.grad.dtype == torch.bfloat16
    assert log_std.grad is not None


class _QwenVisionPatchEmbed(torch.nn.Module):
    """Shape contract of Qwen2.5-VL PatchEmbed: Conv3d kernel == stride."""

    def __init__(self, in_channels=3, temporal=2, patch=4, embed_dim=8):
        super().__init__()
        self.in_channels = in_channels
        self.temporal_patch_size = temporal
        self.patch_size = patch
        kernel = (temporal, patch, patch)
        self.proj = torch.nn.Conv3d(
            in_channels, embed_dim, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(
            -1,
            self.in_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        hidden_states = self.proj(hidden_states.to(self.proj.weight.dtype))
        return hidden_states.view(-1, self.proj.out_channels)


def test_qwen_vl_linear_patch_embed_matches_conv3d_and_backprops():
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
    )

    torch.manual_seed(0)
    module = _QwenVisionPatchEmbed()
    patches = torch.randn(5, 3 * 2 * 4 * 4, requires_grad=True)

    conv_out = module(patches)
    linear_out = _linear_patch_embed_forward(module, patches)
    torch.testing.assert_close(linear_out, conv_out, rtol=1e-5, atol=1e-5)

    linear_out.sum().backward()
    assert module.proj.weight.grad is not None
    assert patches.grad is not None


def test_qwen_vl_linear_patch_embed_is_rebound_on_npu(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        _linear_patch_embed_forward,
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())
    original_forward = model[0].forward

    assert patch_vision_patch_embed(model) == 1
    assert model[0].forward.__func__ is _linear_patch_embed_forward
    assert original_forward.__func__ is not _linear_patch_embed_forward


def test_qwen_vl_linear_patch_embed_is_left_alone_on_nvidia(monkeypatch):
    from rlinf.models.embodiment.qwen_vl_linear_patch_embed import (
        patch_vision_patch_embed,
    )
    from rlinf.scheduler import AcceleratorType

    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    model = torch.nn.Sequential(_QwenVisionPatchEmbed())

    assert patch_vision_patch_embed(model) == 0
    assert model[0].forward.__func__ is _QwenVisionPatchEmbed.forward


_WAN_NPU_PATCHES = "rlinf.envs.sim.world_model.backend.npu_patches"


@pytest.fixture
def wan_dit(monkeypatch):
    """diffsynth's Wan DiT module, whose operators the NPU patches rebind."""
    from rlinf.utils.patcher import Patcher

    def flash_attention(q, k, v, num_heads, compatibility_mode=False):
        return q

    def rope_apply(x, freqs, num_heads):
        return x

    class RMSNorm(torch.nn.Module):
        def forward(self, x):
            return x

    dit = ModuleType("diffsynth.models.wan_video_dit")
    dit.flash_attention, dit.rope_apply, dit.RMSNorm = (
        flash_attention,
        rope_apply,
        RMSNorm,
    )
    for name in ("diffsynth", "diffsynth.models"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, dit.__name__, dit)
    yield dit
    Patcher.clear()


def _import_wan_npu_patches(monkeypatch, *, mindiesd: bool) -> ModuleType:
    """Import the Wan NPU patches on an Ascend stack with or without MindIE-SD."""
    vendor = {"torch_npu": ModuleType("torch_npu"), "mindiesd": None}
    if mindiesd:
        names = (
            "mindiesd",
            "mindiesd.layers",
            "mindiesd.layers.flash_attn",
            "mindiesd.layers.flash_attn.attention_forward",
        )
        vendor.update({name: ModuleType(name) for name in names})
        vendor["mindiesd"].rotary_position_embedding = lambda *args, **kwargs: None
        vendor[names[-1]].attention_forward = lambda *args, **kwargs: None
    for name, module in vendor.items():
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.find_spec(_WAN_NPU_PATCHES)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, _WAN_NPU_PATCHES, module)
    spec.loader.exec_module(module)
    return module


def _patch_like_wan_backend(npu_patches: ModuleType) -> None:
    """Run the patch sequence ``WanBackend._build_pipeline`` runs."""
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    npu_patches.apply_npu_patches(Patcher)
    Patcher.apply()


def _wan_operators(dit: ModuleType) -> tuple:
    return dit.flash_attention, dit.rope_apply, dit.RMSNorm.forward


def test_wan_npu_patches_leave_diffsynth_alone_on_nvidia(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NV_GPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators


def test_wan_npu_patches_log_why_mindiesd_is_unavailable(wan_dit, monkeypatch, caplog):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=False)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    operators = _wan_operators(wan_dit)

    _patch_like_wan_backend(npu_patches)
    assert _wan_operators(wan_dit) == operators
    assert "import of mindiesd halted" in caplog.text


def test_wan_npu_patches_rebind_the_dit_operators_on_every_build(wan_dit, monkeypatch):
    from rlinf.scheduler import AcceleratorType

    npu_patches = _import_wan_npu_patches(monkeypatch, mindiesd=True)
    monkeypatch.setattr(Worker, "accelerator_type", AcceleratorType.NPU)
    kernels = (
        npu_patches.npu_flash_attention,
        npu_patches.npu_rope_apply,
        npu_patches.npu_rmsnorm_forward,
    )

    # Every WanBackend in the process repeats the sequence on the patched module.
    for _ in range(2):
        _patch_like_wan_backend(npu_patches)
        assert _wan_operators(wan_dit) == kernels


def _history_cfg():
    return OmegaConf.create(
        {
            "model": {
                "history_buffers": {
                    "main": {
                        "history_size": 2,
                        "min_history_size": 1,
                        "input_interval": 3,
                        "history_keys": ["main_images"],
                        "input_on_done": True,
                    }
                }
            }
        }
    )


def _append_step(manager: HistoryManager, value: int) -> None:
    manager.append_to_history_entries(
        {"main_images": torch.tensor([[value], [value + 10]])}
    )


def test_build_history_input_skips_between_interval_ticks():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_input == {}
    assert history_length == {}
    assert manager.history_counts == [2, 2]


def test_build_history_input_emits_on_interval_tick():
    manager = HistoryManager(_history_cfg(), num_envs=2)
    _append_step(manager, 1)
    _append_step(manager, 2)
    _append_step(manager, 3)

    history_input, history_length = manager.build_history_input(
        torch.tensor([False, False])
    )

    assert history_length == {"main": [2, 2]}
    assert history_input["main"]["main_images"][0] == [
        torch.tensor([2]),
        torch.tensor([3]),
    ]
    assert history_input["main"]["main_images"][1] == [
        torch.tensor([12]),
        torch.tensor([13]),
    ]


def _success_potential_state_machine():
    from rlinf.models.embodiment.reward.vlm_reward_model import (
        ShapedVLMRewardModel,
    )

    model = ShapedVLMRewardModel.__new__(ShapedVLMRewardModel)
    model.potential_gamma = 1.0
    model.potential_scale = 1.0
    model.potential_ema_alpha = 0.5
    model.potential_clip = 0.0
    model.success_threshold = 0.5
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model.gt_success_bonus = 0.0
    model.infer_micro_batch_size = 0
    model._previous_potentials = None
    model._success_fired = None
    model._success_streak = None
    return model


def test_empty_history_input_still_resets_shaping_state_on_done():
    model = _success_potential_state_machine()
    model._previous_potentials = torch.tensor([0.4, 0.8])
    model._success_fired = torch.tensor([True, True])
    model._success_streak = torch.tensor([3, 1], dtype=torch.int32)

    rewards = model.compute_reward(
        {
            "history_input": {},
            "dones": torch.tensor([True, False]),
        }
    )

    assert rewards.tolist() == pytest.approx([0.0, 0.0])
    assert torch.isnan(model._previous_potentials[0])
    assert float(model._previous_potentials[1]) == pytest.approx(0.8)
    assert model._success_fired.tolist() == [False, True]
    assert model._success_streak.tolist() == [0, 1]


VALUE_CLIP = 0.2
HUBER_DELTA = 10.0


def _critic_metrics(values, prev_values, returns, loss_mask=None):
    _, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=loss_mask,
    )
    return metrics


def test_value_clip_ratio_is_zero_when_no_update_is_clipped():
    prev_values = torch.zeros(4, 8)
    values = torch.full((4, 8), VALUE_CLIP / 2)
    returns = torch.zeros(4, 8)

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_clip_ratio_reports_the_fraction_of_clipped_updates():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    # Half of the entries move outside the trust region, half stay inside.
    values = torch.full((4, 8), VALUE_CLIP / 2)
    values[:, :4] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_grows_with_the_size_of_the_value_update():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)

    ratios = [
        float(
            _critic_metrics(torch.full((4, 8), scale), prev_values, returns)[
                "critic/value_clip_ratio"
            ]
        )
        for scale in (0.5 * VALUE_CLIP, 2 * VALUE_CLIP)
    ]

    assert ratios == [pytest.approx(0.0), pytest.approx(1.0)]


def test_value_clip_ratio_ignores_masked_out_entries():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    loss_mask[:, :2] = True

    # Every valid entry is clipped; every padded entry is not.
    values = torch.zeros(4, 8)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(1.0)


def test_value_clip_ratio_broadcasts_a_narrower_loss_mask():
    prev_values = torch.zeros(4, 8, 3)
    returns = torch.zeros(4, 8, 3)
    loss_mask = torch.zeros(4, 8, 1, dtype=torch.bool)
    loss_mask[:, :4] = True

    values = torch.zeros(4, 8, 3)
    values[:, :2] = 10 * VALUE_CLIP

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    # 2 of the 4 unmasked steps are clipped.
    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.5)


def test_value_clip_ratio_is_zero_when_every_entry_is_masked_out():
    prev_values = torch.zeros(4, 8)
    returns = torch.zeros(4, 8)
    loss_mask = torch.zeros(4, 8, dtype=torch.bool)
    values = torch.full((4, 8), 10 * VALUE_CLIP)

    metrics = _critic_metrics(values, prev_values, returns, loss_mask=loss_mask)

    assert float(metrics["critic/value_clip_ratio"]) == pytest.approx(0.0)


def test_value_loss_is_unchanged_by_the_metric_computation():
    torch.manual_seed(0)
    prev_values = torch.randn(4, 8)
    values = torch.randn(4, 8, requires_grad=True)
    returns = torch.randn(4, 8)

    loss, metrics = compute_ppo_critic_loss(
        values=values,
        returns=returns,
        prev_values=prev_values,
        value_clip=VALUE_CLIP,
        huber_delta=HUBER_DELTA,
        loss_mask=None,
    )

    value_pred_clipped = prev_values + (values - prev_values).clamp(
        -VALUE_CLIP, VALUE_CLIP
    )
    expected = torch.max(
        torch.nn.functional.huber_loss(
            values, returns, delta=HUBER_DELTA, reduction="none"
        ),
        torch.nn.functional.huber_loss(
            value_pred_clipped, returns, delta=HUBER_DELTA, reduction="none"
        ),
    ).mean()

    assert float(loss.detach()) == pytest.approx(float(expected.detach()), abs=1e-6)
    assert loss.requires_grad
    assert not metrics["critic/value_clip_ratio"].requires_grad


def test_create_builds_expected_sampler_types():
    constant = DelaySampler.create(
        OmegaConf.create({"type": "constant", "delay": 0.12})
    )
    uniform = DelaySampler.create(
        OmegaConf.create({"type": "uniform", "min_delay": 0.03, "max_delay": 0.08})
    )
    exponential = DelaySampler.create(
        OmegaConf.create({"type": "exponential", "rate": 0.5})
    )
    gaussian = DelaySampler.create(
        OmegaConf.create({"type": "gaussian", "mean": 0.20, "stddev": 0.03})
    )

    assert isinstance(constant, ConstantDelaySampler)
    assert isinstance(uniform, UniformDelaySampler)
    assert isinstance(exponential, ExponentialDelaySampler)
    assert isinstance(gaussian, GaussianDelaySampler)


def test_create_accepts_none():
    assert DelaySampler.create(None) is None


def test_same_seed_produces_same_sequence_per_sampler():
    first = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)
    second = UniformDelaySampler(min_delay=0.1, max_delay=0.2, seed=2026)

    assert first.sample(8) == second.sample(8)


def test_constant_sampler_uses_seconds_helpers():
    sampler = ConstantDelaySampler(delay=0.25)

    assert sampler.sample(3) == [0.25, 0.25, 0.25]
    assert sampler.sample_one() == 0.25


def test_gaussian_sampler_never_returns_negative_seconds():
    sampler = GaussianDelaySampler(mean=0, stddev=0.1, seed=0)

    assert all(delay >= 0 for delay in sampler.sample(100))


def test_invalid_ranges_raise_clear_errors():
    with pytest.raises(ValueError, match="min_delay must be <="):
        UniformDelaySampler(min_delay=0.2, max_delay=0.1)

    with pytest.raises(ValueError, match="rate must be > 0"):
        ExponentialDelaySampler(rate=0)


def test_num_samples_must_be_non_negative_int():
    sampler = ConstantDelaySampler(delay=1)

    with pytest.raises(TypeError, match="num_samples must be an int"):
        sampler.sample(1.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_samples must be >= 0"):
        sampler.sample(-1)


class _FakeEnv:
    """Minimal non-gym env exposing the chunk_step/reset surface."""

    def chunk_step(self, *args, **kwargs):
        return "stepped"

    def reset(self, *args, **kwargs):
        return "obs", {}


# Mock gymnasium and its transitive imports for unit-test environments that
# do not install the embodied extras. A minimal gym.Wrapper shim is enough
# because InsertDelay only delegates to self.env.


class _FakeGymEnv:
    pass


class _FakeGymWrapper:
    def __init__(self, env):
        self.env = env


_fake_gym = MagicMock()
_fake_gym.Env = _FakeGymEnv
_fake_gym.Wrapper = _FakeGymWrapper

if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = _fake_gym
if "imageio" not in sys.modules:
    sys.modules["imageio"] = MagicMock()


def _delayed_env(delay: float):
    from rlinf.envs.wrappers import InsertDelay

    return InsertDelay(
        _FakeEnv(), OmegaConf.create({"type": "constant", "delay": delay})
    )


def test_chunk_step_does_not_block_the_caller():
    env = _delayed_env(0.5)

    start = time.monotonic()
    assert env.chunk_step() == "stepped"
    elapsed = time.monotonic() - start

    # The delay is sampled, not slept: blocking here would stall the event loop.
    assert elapsed < 0.05


def test_wait_delay_waits_out_the_accumulated_delay():
    env = _delayed_env(0.05)
    env.chunk_step()
    env.chunk_step()

    start = time.monotonic()
    asyncio.run(env.wait_delay())
    elapsed = time.monotonic() - start

    # Both sampled delays are paid, never dropped.
    assert elapsed == pytest.approx(0.1, abs=0.03)


def test_wait_delay_yields_to_other_coroutines():
    env = _delayed_env(0.2)
    env.chunk_step()
    progressed = []

    async def main():
        async def ticker():
            for _ in range(4):
                await asyncio.sleep(0.01)
                progressed.append(1)

        await asyncio.gather(env.wait_delay(), ticker())

    asyncio.run(main())
    # A blocking sleep would have starved the ticker entirely.
    assert len(progressed) == 4


def test_wait_delay_is_a_noop_when_nothing_is_pending():
    env = _delayed_env(0.5)

    start = time.monotonic()
    asyncio.run(env.wait_delay())

    assert time.monotonic() - start < 0.05


def test_delay_metrics_report_every_sample():
    env = _delayed_env(0.03)
    env.chunk_step()
    env.reset()

    metrics = env.insert_delay_metrics()

    assert metrics.tolist() == pytest.approx([0.03, 0.03])
    assert env.insert_delay_metrics().numel() == 0


def test_joint_stage2_actor_checkpoint_rejects_tcp_weights(tmp_path):
    from rlinf.algorithms.online_residual_rl.checkpoint import load_actor_state

    joint = OnlineResidualRLConfig(
        action_representation="joint_absolute", z_dim=4, hidden_dim=8
    )
    tcp = OnlineResidualRLConfig(z_dim=4, hidden_dim=8)
    path = tmp_path / "actor.pt"
    actor = ResidualActor(joint)
    torch.save(actor.state_dict(), path)
    restored = load_actor_state(path, joint)
    target = ResidualActor(joint)
    target.load_state_dict(restored)
    output = target(torch.zeros(1, 4), torch.zeros(1, 8), torch.zeros(1, 12, 8))
    assert output.shape == (1, 12, 8)
    with pytest.raises(ValueError, match="shape"):
        load_actor_state(path, tcp)
    torch.save(ResidualActor(tcp).state_dict(), path)
    with pytest.raises(ValueError, match="shape"):
        load_actor_state(path, joint)


@pytest.mark.parametrize(
    "name,expected",
    [("pi05_flexiv_joint_absolute", -0.5), ("pi05_flexiv", -0.5), ("pi05_libero", 0.3)],
)
def test_stage1_rlt_proprio_uses_flexiv_model_input_normalization(
    monkeypatch, name, expected
):
    from rlinf.models.embodiment.openpi_rlinf.tasks import eval as eval_module

    observation = SimpleNamespace(
        state=torch.full((1, 32), -0.5), tokenized_prompt=None
    )
    monkeypatch.setattr(eval_module, "preprocess_observation", lambda obs, train: obs)
    model = SimpleNamespace(
        config_name=name,
        _require_rlt=lambda: None,
        env_obs_to_observation=lambda obs: observation,
        build_prefix_cache=lambda obs: (
            torch.zeros(1, 2, 4),
            torch.ones(1, 2, dtype=torch.bool),
            None,
        ),
        _select_rlt_prefix_embeddings=lambda prefix, mask, prompt: (prefix, mask),
        _encode_rlt_flat=lambda prefix, mask: torch.zeros(1, 4),
        _sample_actions_from_prefix_cache=lambda *args: torch.zeros(1, 32, 32),
        _select_configured_state=lambda states: states,
        rlt_cfg=SimpleNamespace(rlt_normalized_actions=True),
        action_env_dim=8,
        action_chunk=12,
    )
    result = eval_module.Pi0Eval.extract_rlt_obs(
        model, {"states": torch.full((1, 8), 0.3)}
    )
    torch.testing.assert_close(result["proprio"], torch.full((1, 8), expected))
