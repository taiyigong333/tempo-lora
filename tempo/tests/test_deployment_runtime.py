"""部署参数、完整预热和去噪循环的局部回归。"""
from types import SimpleNamespace
from unittest.mock import patch
import time

import numpy as np
import pytest
import torch

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.shared.normalize import NormStats
from openpi.tempo_full.inference import validate_runtime, warmup


@pytest.mark.parametrize('change', [
    {'cpu_threads': 0}, {'cpu_threads': True}, {'warmup_repeats': -1},
    {'warmup_batches': []}, {'warmup_batches': [True]}, {'warmup_batches': [301]},
    {'warmup_batches': '1,4'}, {'denoise_compile_mode': 'unknown'},
])
def test_invalid_runtime_rejected(change):
    with pytest.raises(ValueError):
        validate_runtime(change)


def test_old_runtime_defaults_and_explicit_options():
    options = {}
    validate_runtime(options)
    assert options['cpu_threads'] == 8
    assert options['warmup_batches'] == [1, 4, 8]
    for mode in ('default', 'reduce-overhead', 'max-autotune-no-cudagraphs'):
        explicit = dict(cpu_threads=2, warmup_repeats=2, warmup_batches=[1, 6], denoise_compile_mode=mode)
        validate_runtime(explicit)
        assert explicit['cpu_threads'] == 2


class WarmPolicy:
    cfg = {'model': {'action_horizon': 30}}

    def __init__(self):
        self.requests = []
        self.reset_count = 0
        self.last_timestamp = None

    def reset(self):
        self.reset_count += 1
        self.last_timestamp = None

    def infer(self, request):
        stamps = [frame['timestamp'] for frame in request['feedback']]
        if self.last_timestamp is not None:
            assert stamps[0] == pytest.approx(self.last_timestamp + 1 / 30)
        self.last_timestamp = stamps[-1]
        self.requests.append(request)
        return {'actions': np.zeros((30, 7), np.float32), 'end_to_end_ms': 1.0}


def test_warmup_covers_batches_and_clears_history():
    policy = WarmPolicy()
    norm = {'state': NormStats(mean=np.zeros(7), std=np.ones(7))}
    with patch('torch.cuda.synchronize'):
        rows = warmup(policy, norm, {'warmup_batches': [1, 4, 8], 'warmup_repeats': 2})
    assert len(rows) == 12
    assert policy.reset_count == 3
    assert policy.last_timestamp is None
    assert [len(r['feedback']) for r in policy.requests] == [1, 4, 8] * 4
    for request in policy.requests[6:]:
        assert request['mode'] == 'rtc' and request['rtc_prefix'].shape == (30, 7)


def test_failed_warmup_clears_history():
    policy = WarmPolicy()
    def fail(request):
        raise RuntimeError('合成预热失败')
    policy.infer = fail
    norm = {'state': NormStats(mean=np.zeros(7), std=np.ones(7))}
    with pytest.raises(RuntimeError, match='预热失败'):
        warmup(policy, norm)
    assert policy.reset_count == 2


@pytest.mark.parametrize('steps', [1, 3, 5, 10, 100])
def test_cpu_counted_loop_preserves_time_and_result(steps):
    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    model.predict_ut = False
    model.flow_positions, model.flow_dim = 1, 2
    model._preprocess_observation = lambda *args, **kwargs: (None,) * 8
    model.embed_prefix = lambda *args, **kwargs: (None, torch.ones((1, 1), dtype=torch.bool), torch.ones((1, 1)))
    model._prepare_attention_masks_4d = lambda value: value
    model.paligemma_with_expert = SimpleNamespace(
        paligemma=SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace())),
        forward=lambda **kwargs: (None, None))
    observed = []
    def velocity(state, masks, cache, value, timestep, **kwargs):
        observed.append(timestep.clone())
        return value * .3 + timestep[..., None, None]
    model.denoise_step = velocity
    noise = torch.tensor([[[.2, -.4]]])
    observation = SimpleNamespace(state=torch.zeros((1, 2)))
    actual = model.sample_actions('cpu', observation, noise=noise.clone(), num_steps=steps)
    time = torch.tensor(1., dtype=torch.float32)
    dt = torch.tensor(-1. / steps, dtype=torch.float32)
    expected = noise.clone()
    times = []
    while time >= -dt / 2:
        times.append(time.clone())
        expected = expected + dt * (expected * .3 + time)
        time += dt
    assert len(observed) == len(times) == steps
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.stack(observed).flatten(), torch.stack(times), rtol=0, atol=0)


@pytest.mark.parametrize('steps', [0, -1, True, 1.5])
def test_invalid_denoise_count_rejected(steps):
    model = PI0Pytorch.__new__(PI0Pytorch)
    torch.nn.Module.__init__(model)
    with pytest.raises(ValueError, match='num_steps'):
        model.sample_actions('cpu', object(), num_steps=steps)


def test_policy_timing_includes_device_transfer():
    from openpi.policies.policy import Policy
    class SlowActions:
        def __getitem__(self, key):
            return self

        def detach(self):
            return self

        def cpu(self):
            # 模拟等待CUDA及回传，旧计时会完全漏掉这段耗时。
            time.sleep(.02)
            return np.zeros((30, 7), np.float32)

    model = torch.nn.Linear(1, 1)
    model.sample_actions = lambda *args, **kwargs: SlowActions()
    policy = Policy(model, is_pytorch=True, pytorch_device='cpu')
    with patch('openpi.policies.policy._model.Observation.from_dict', return_value=object()):
        result = policy.infer({'state': np.zeros(7, np.float32)})
    assert result['policy_timing']['infer_ms'] >= 20
    assert result['policy_timing']['total_ms'] >= result['policy_timing']['infer_ms']
