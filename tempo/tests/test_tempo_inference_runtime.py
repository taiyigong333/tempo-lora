"""推理局部编译保持默认兼容、10次采样与开放端口前预热。"""
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from openpi.tempo_lora import inference


def prepare(monkeypatch):
    cfg = {"name": "runtime_test", "data": {"action_dim": 7, "fps": 30, "prompt": "task", "cameras": ["head", "left_wrist"]},
           "model": {"action_horizon": 16}, "inference": {"denoise_steps": 10, "execute_steps": 1, "host": "localhost", "port": 8000}}
    model = Mock()
    model.denoise_step = Mock()
    model.predict_ut = False
    compiled = Mock(return_value=Mock())
    factory = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(metadata=kwargs["metadata"], settings=kwargs))
    monkeypatch.setattr(inference, "load_policy_checkpoint", lambda path: {"config": cfg, "trainable": {}, "norm_stats": {}, "step": 10000})
    monkeypatch.setattr(inference, "checkpoint_file", lambda path: path)
    monkeypatch.setattr(inference, "configure_environment", lambda cfg: None)
    monkeypatch.setattr(inference, "build_model", lambda *args, **kwargs: (model, {}))
    dc = SimpleNamespace(data_transforms=SimpleNamespace(inputs=[], outputs=[]), model_transforms=SimpleNamespace(inputs=[], outputs=[]))
    monkeypatch.setattr(inference, "make_train_config", lambda cfg: SimpleNamespace(data=SimpleNamespace(create=lambda *args: dc)))
    monkeypatch.setattr(inference, "artifact_root", lambda cfg: "test")
    monkeypatch.setattr(inference, "model_config", lambda cfg: None)
    monkeypatch.setattr(inference, "deserialize_json", lambda value: {})
    monkeypatch.setattr(inference, "Policy", factory)
    monkeypatch.setattr(inference.torch, "compile", compiled)
    return cfg, model, compiled


def test_default_runtime_preserves_eager_and_ten_steps(monkeypatch):
    _, model, compiled = prepare(monkeypatch)
    policy = inference.create_policy("checkpoint")
    compiled.assert_not_called()
    model.requires_grad_.assert_not_called()
    assert policy.settings["sample_kwargs"] == {"num_steps": 10}
    assert policy.metadata["compile_inference"] is False
    assert policy.metadata["denoise_steps"] == 10


def test_enabled_runtime_compiles_only_denoise(monkeypatch):
    _, model, compiled = prepare(monkeypatch)
    original = model.denoise_step
    policy = inference.create_policy("checkpoint", compile_inference=True)
    compiled.assert_called_once_with(original, dynamic=True, mode="default")
    model.requires_grad_.assert_called_once_with(False)
    assert model.denoise_step is compiled.return_value
    assert policy.settings["sample_kwargs"] == {"num_steps": 10}
    assert policy.metadata["compile_inference"] is True


@pytest.mark.parametrize("invalid", ["false", 1, None])
def test_invalid_compile_setting_is_rejected(invalid):
    with pytest.raises(ValueError, match="布尔"):
        inference.create_policy("unused", compile_inference=invalid)


def test_native_and_rtc_warmup_finish_before_listening(monkeypatch, tmp_path):
    cfg, _, _ = prepare(monkeypatch)
    events = []
    base = SimpleNamespace(tempo_norm_stats={"state": SimpleNamespace(mean=np.zeros(7))})
    create = Mock(return_value=base)
    monkeypatch.setattr(inference, "create_policy", create)
    class Online:
        def __init__(self, policy, cfg, **kwargs):
            assert kwargs["compile_inference"] is True
            self.policy, self.metadata = policy, {}
        def infer(self, observation):
            events.append(observation.get("mode", "tempo"))
            return {"actions": np.zeros((16, 7), np.float32)}
        def reset(self):
            events.append("reset")
    monkeypatch.setattr(inference, "OnlinePolicy", Online)
    from openpi.serving import websocket_policy_server
    class Server:
        def __init__(self, *args, **kwargs):
            assert kwargs["single_client"] is True
        def serve_forever(self):
            events.append("listen")
    monkeypatch.setattr(websocket_policy_server, "WebsocketPolicyServer", Server)
    inference.serve({"inference": {"compile_inference": True}}, checkpoint=tmp_path / "checkpoint")
    create.assert_called_once_with(tmp_path / "checkpoint", compile_inference=True)
    assert events == ["tempo", "rtc", "reset", "listen"]


def test_ut_decoder_is_compiled_with_denoise(monkeypatch):
    _, model, compiled = prepare(monkeypatch)
    model.predict_ut = True
    inference.create_policy("checkpoint", compile_inference=True)
    assert compiled.call_count == 2
    assert compiled.call_args_list[1].kwargs == {"dynamic": False, "mode": "default"}
