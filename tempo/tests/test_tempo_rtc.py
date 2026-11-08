"""RTC潜变量引导和原生绝对动作前缀的兼容性回归。"""

import numpy as np
import pytest
import torch

from openpi.shared.normalize import NormStats
from openpi.tempo_lora.inference import OnlinePolicy
from openpi.tempo_lora.rtc import guided_velocity, prefix_weights


def test_decoded_vjp_matches_analytic_gradient():
    value = torch.tensor([[[.3, -.4]]], dtype=torch.float32)
    matrix = torch.tensor([[1., 2.], [-1., .5]])
    time = torch.tensor(.5)
    prefix = torch.tensor([[[.2, .1]]])
    weights = torch.tensor([[[1., .2]]])
    velocity = lambda x: .3 * x
    decode = lambda x: x @ matrix
    result = guided_velocity(velocity, decode, value, time, prefix, weights, 10)
    estimate = decode(value - time * velocity(value))
    correction = ((prefix - estimate) * weights) @ matrix.T * (1 - .3 * time)
    torch.testing.assert_close(result, velocity(value) - 2 * correction)


def test_zero_mask_preserves_ordinary_velocity():
    value = torch.randn(1, 1, 4)
    result = guided_velocity(lambda x: x.square(), lambda x: x, value, torch.tensor(.2),
                             torch.ones_like(value), torch.zeros_like(value), 10)
    torch.testing.assert_close(result, value.square(), rtol=0, atol=0)


def test_guidance_reduces_decoded_linear_prefix_error():
    value = torch.ones(1, 1, 2)
    target = torch.zeros_like(value)
    velocity = guided_velocity(lambda x: x * 0, lambda x: x, value, torch.tensor(.5),
                               target, torch.ones_like(target), 10)
    updated = value - .1 * velocity
    assert torch.linalg.vector_norm(updated - target) < torch.linalg.vector_norm(value - target)


def test_soft_prefix_mask():
    weights = prefix_weights(2, 5, 8)
    np.testing.assert_array_equal(weights[:2], 1)
    assert 1 > weights[2] > weights[3] > weights[4] > 0
    np.testing.assert_array_equal(weights[5:], 0)
    with pytest.raises(ValueError):
        prefix_weights(6, 5, 8)


class Sam:
    def reset(self):
        self.count = 0

    def step(self, rgb):
        self.count += 1
        return np.zeros((256, 8, 8), np.float32)


class Policy:
    metadata = {}
    tempo_norm_stats = {"actions": NormStats(mean=np.zeros(7), std=np.ones(7),
                                           q01=-np.ones(7), q99=np.ones(7))}

    def infer(self, obs, **kwargs):
        self.kwargs = kwargs
        return {"actions": np.zeros((16, 7), np.float32)}


def online():
    cfg = {"model": {"action_history_steps": 10, "obs_history": 3, "obs_history_stride_s": 1,
                     "action_horizon": 16, "action_dim": 32},
           "data": {"fps": 30, "action_history_seconds": 10, "action_dim": 7,
                    "cameras": ["head", "left_wrist"], "prompt": "grab the rotating red square"}}
    return OnlinePolicy(Policy(), cfg, sam_stream=Sam())


def observation():
    return {"episode_id": "test", "timestamp": 0., "state": np.full(7, .2, np.float32),
            "images": {name: np.zeros((224, 224, 3), np.uint8) for name in ("head", "left_wrist")}}


def test_absolute_prefix_is_reanchored_and_normalized():
    policy = online()
    obs = {**observation(), "mode": "rtc", "rtc_prefix": np.full((4, 7), .3, np.float32), "rtc_delay": 2}
    result = policy.infer(obs)
    kwargs = policy.policy.kwargs["sample_kwargs"]
    np.testing.assert_allclose(kwargs["rtc_prefix"][0, :4, :6], .1, atol=1e-6)
    np.testing.assert_allclose(kwargs["rtc_prefix"][0, :4, 6], .3, atol=1e-6)
    assert not kwargs["rtc_weights"][:, :, 7:].any()
    assert result["mode"] == "rtc" and result["rtc_prefix_length"] == 4


@pytest.mark.parametrize("change", [{"mode": "invalid"}, {"prompt": "another task"},
                                    {"mode": "rtc", "rtc_prefix": np.zeros((17, 7))},
                                    {"mode": "rtc", "rtc_prefix": np.zeros((4, 7)), "rtc_delay": 5}])
def test_invalid_request_does_not_mutate_episode(change):
    policy = online()
    with pytest.raises(ValueError):
        policy.infer({**observation(), **change})
    assert policy.frame == -1 and policy.sam.count == 0


def test_parallel_sam_resolves_current_batch_and_drains_on_policy_error():
    policy = online()
    policy.parallel_sam = True
    resolved = []
    images = []
    def encode_async(batch):
        images.extend(batch)
        def resolve():
            resolved.append(True)
            return torch.full((1, 64, 256), .75)
        return resolve
    policy.sam.encode_async = encode_async
    result = policy.infer(observation())
    assert len(images) == 1 and len(resolved) == 1
    np.testing.assert_array_equal(policy.latest_tokens, .75)
    assert "sam2_tokens_provider" in policy.policy.kwargs["sample_kwargs"]
    assert result["feedback_ingest_ms"] >= 0
    def fail(*args, **kwargs):
        raise RuntimeError("合成策略错误")
    policy.policy.infer = fail
    with pytest.raises(RuntimeError, match="策略错误"):
        policy.infer({**observation(), "timestamp": 1 / 30})
    assert len(resolved) == 2


def test_visual_encoding_finishes_before_sam_wait_and_fusion():
    from types import SimpleNamespace
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(lora_enabled=False, compact_prefix=False)
    model.use_sam2_fusion, model.use_action_history = True, False
    events = []
    def image(value, num_frames):
        events.append("vision")
        return torch.ones(1, 2, 4) * len(events)
    def language(value):
        events.append("language")
        return torch.zeros(1, 3, 4)
    model.paligemma_with_expert = SimpleNamespace(embed_image=image, embed_language_tokens=language)
    model._apply_checkpoint = lambda fn, *args: fn(*args)
    def resolve():
        events.append("sam_wait")
        return torch.full((1, 64, 256), 2.)
    def fusion(head, sam):
        events.append("fusion")
        return head + sam[0, 0, 0]
    model.sam2_fusion = fusion
    embs, _, _ = model.embed_prefix([torch.zeros(1, 3, 4, 4)] * 2, [torch.ones(1, dtype=torch.bool)] * 2,
                                     torch.zeros(1, 3, dtype=torch.long), torch.ones(1, 3, dtype=torch.bool),
                                     sam2_tokens_provider=resolve)
    assert events == ["vision", "vision", "sam_wait", "fusion", "language"]
    torch.testing.assert_close(embs[:, :2], torch.full((1, 2, 4), 3.))
    torch.testing.assert_close(embs[:, 2:4], torch.full((1, 2, 4), 2.))
