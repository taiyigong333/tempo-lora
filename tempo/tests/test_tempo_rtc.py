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
