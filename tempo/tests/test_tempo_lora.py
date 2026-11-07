"""覆盖动作空间、历史因果性与 PEFT 参数冻结的必要回归。"""

import numpy as np
import pytest
import torch

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.policies.yam_dual_policy import YamInputs
from openpi.tempo_lora.data import action_window, delta_mask, history_buckets
from openpi.tempo_lora.inference import OnlinePolicy
from openpi.tempo_lora.model import inject_lora, optimizer_groups, set_precision
from openpi.tempo_lora.train import ResumeBatchSampler


@pytest.mark.parametrize("dim", [7, 14])
def test_feedback_window_and_roundtrip(dim):
    state = np.arange(6 * dim, dtype=np.float32).reshape(6, dim) / 100
    action = state[np.minimum(np.arange(6) + 1, 5)]
    target = action_window(action, state, 2, 4)
    mask = np.asarray(delta_mask(dim))
    np.testing.assert_allclose(target[0, mask], state[3, mask] - state[2, mask])
    np.testing.assert_array_equal(target[:, ~mask], action[2:6, ~mask])
    stats = {"actions": transforms.NormStats(mean=np.zeros(dim), std=np.ones(dim),
                                            q01=-np.ones(dim), q99=2 * np.ones(dim))}
    normalized = transforms.Normalize(stats, use_quantiles=True)({"actions": target.copy()})
    restored = transforms.Unnormalize(stats, use_quantiles=True)(normalized)
    restored["state"] = state[2]
    result = transforms.AbsoluteActions(delta_mask(dim))(restored)["actions"]
    np.testing.assert_allclose(result, action[2:6], atol=2e-7)


def test_history_is_causal_and_startup_padding():
    action = np.arange(30 * 7, dtype=np.float32).reshape(30, 7)
    first, pad = history_buckets(action, 0, 3, 4)
    assert pad.all() and not first.any()
    before, pad = history_buckets(action, 5, 3, 4)
    action[5:] = 9999
    after, _ = history_buckets(action, 5, 3, 4)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_allclose(before[-1], action[1:5].mean(0))
    assert pad.tolist() == [True, False, False]


def test_single_arm_missing_camera_mask():
    from openpi.models.model import ModelType
    result = YamInputs(ModelType.PI05)({"state": np.zeros(7), "images": {
        "head": np.zeros((3, 224, 224, 3), np.uint8),
        "left_wrist": np.zeros((3, 224, 224, 3), np.uint8)}})
    assert not result["image_mask"]["right_wrist_0_rgb"]
    assert result["image"]["right_wrist_0_rgb"].shape == (3, 224, 224, 3)
    assert result["state"].shape == (7,)


def tiny_model():
    model = torch.nn.Module()
    paths = ["paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj",
             "paligemma_with_expert.gemma_expert.model.layers.0.mlp.up_proj",
             "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.3.self_attn.out_proj"]
    for path in paths:
        parent = model
        for name in path.split(".")[:-1]:
            if name not in parent._modules:
                parent.add_module(name, torch.nn.Module())
            parent = getattr(parent, name)
        parent.add_module(path.split(".")[-1], torch.nn.Linear(8, 8))
    model.action_in_proj = torch.nn.Linear(8, 8)
    return model, paths


def test_peft_identity_freeze_and_fp32_update():
    model, paths = tiny_model()
    before = model.get_submodule(paths[0])(torch.ones(2, 8)).detach()
    config = Pi0Config(lora_enabled=True, lora_vlm_rank=2, lora_expert_rank=2, lora_siglip_rank=2)
    matched = inject_lora(model, config)
    assert all(len(values) == 1 for values in matched.values())
    torch.testing.assert_close(model.get_submodule(paths[0])(torch.ones(2, 8)), before, rtol=0, atol=0)
    set_precision(model, "float32")
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == ("lora_" in name or name.startswith("action_in_proj"))
    optimizer = torch.optim.AdamW(optimizer_groups(model, {"lora_lr": 5e-5, "ut_lr": 5e-5, "tempo_lr": 1e-4}))
    frozen = model.get_submodule(paths[0]).weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad()
        model.get_submodule(paths[0])(torch.ones(2, 8)).sum().backward()
        optimizer.step()
    torch.testing.assert_close(model.get_submodule(paths[0]).weight, frozen, rtol=0, atol=0)
    assert model.get_submodule(paths[0]).lora_B["default"].weight.abs().sum() > 0
    for values in optimizer.state.values():
        assert values["exp_avg"].dtype == values["exp_avg_sq"].dtype == torch.float32


def test_resume_sampler_same_future_batches():
    full = list(ResumeBatchSampler(25, 4, 42, 0, 30))
    resumed = list(ResumeBatchSampler(25, 4, 42, 13, 30))
    assert full[13:] == resumed


def test_legacy_configuration_and_complete_config_discovery():
    from openpi.training.config import get_config
    assert Pi0Config().predict_ut is False
    assert get_config("pi05_aloha").model.lora_enabled is False
    complete = get_config("pi05_piperx_tempo_ut_lora")
    assert complete.model.predict_ut and complete.model.lora_enabled
    assert complete.model.action_dim == 32 and complete.model.ut_dim == 64
    assert complete.data.action_dim == complete.model.action_history_dim == 7


def test_cache_manifest_rejects_incompatible_contract(tmp_path, monkeypatch):
    import openpi.tempo_lora.data as data_module
    monkeypatch.setattr(data_module, "contract", lambda cfg: cfg)
    monkeypatch.setattr(data_module, "split_data", lambda cfg: {"train_episodes": [0], "val_episodes": []})
    data_module.write_manifest({"version": 1}, tmp_path, "norm")
    with pytest.raises(ValueError, match="契约"):
        data_module.check_manifest({"version": 2}, tmp_path, "norm")


def test_mvae_video_mean_and_native_7d_decoder():
    from tools.ut.mvae import MVAEConfig, PoEMVAE
    from openpi.models_pytorch.ut_decoder import UtActionDecoder
    cfg = MVAEConfig(window=4, action_dim=7, sam_dim=256, sam_grid=1,
                     hidden_dim=32, latent_dim=8, nhead=4, enc_layers=1, dec_layers=1, beta=0)
    model = PoEMVAE(cfg)
    latent, action = torch.randn(2, 4, 256, 1, 1), torch.randn(2, 4, 7)
    result = model(latent, action)
    result["loss"].backward()
    assert torch.isfinite(result["loss"])
    assert result["action_recon"].shape == (2, 4, 7)
    model.eval()
    mu, _ = model.video_enc(latent)
    decoder = UtActionDecoder(latent_dim=8, hidden_dim=32, window=4, action_dim=7, n_layers=1, nhead=4)
    decoder.load_state_dict(model.action_dec.state_dict())
    assert decoder(mu).shape == (2, 4, 7)


def test_online_feedback_matches_offline_and_resets():
    class Sam:
        def reset(self):
            self.count = 0
        def step(self, rgb):
            self.count += 1
            return np.zeros((256, 8, 8), np.float32)
    class Policy:
        metadata = {}
        def infer(self, observation, **kwargs):
            return {"actions": np.zeros((16, 7))}
    cfg = {"model": {"action_history_steps": 10, "obs_history": 3, "obs_history_stride_s": 1},
           "data": {"fps": 30, "action_history_seconds": 10, "action_dim": 7,
                    "cameras": ["head", "left_wrist"], "prompt": "grab the rotating red square"}}
    online = OnlinePolicy(Policy(), cfg, sam_stream=Sam())
    states = np.arange(45 * 7, dtype=np.float32).reshape(45, 7)
    images = {cam: np.zeros((224, 224, 3), np.uint8) for cam in cfg["data"]["cameras"]}
    for t in range(45):
        online.infer({"episode_id": "a", "timestamp": t / 30, "state": states[t], "images": images})
        history, padding = history_buckets(states[1:], t)
        np.testing.assert_allclose(online.inputs()["action_history"], history)
        np.testing.assert_array_equal(online.inputs()["action_history_is_pad"], padding)
    online.infer({"episode_id": "b", "timestamp": 10., "state": states[0], "images": images})
    assert online.frame == 0 and online.sam.count == 1 and online.inputs()["action_history_is_pad"].all()
    with pytest.raises(ValueError, match="连续"):
        online.infer({"episode_id": "b", "timestamp": 10.5, "state": states[1], "images": images})
