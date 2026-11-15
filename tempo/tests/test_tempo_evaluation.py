"""评估通过正常入口保留逐步误差，并单独统计TE使用的第6步及以后动作。"""

from types import SimpleNamespace

import numpy as np
import pytest

from openpi.tempo_lora import inference


@pytest.mark.parametrize("horizon", [16, 4])
def test_evaluate_keeps_legacy_metrics_and_reports_te_tail(tmp_path, monkeypatch, horizon):
    cfg = {"model": {"action_horizon": horizon, "obs_history": 3, "obs_history_stride_s": 1.,
                     "action_history_steps": 10, "predict_ut": True, "ut_dim": 64, "action_dim": 32},
           "data": {"fps": 30, "action_history_seconds": 10, "cameras": ["head", "left_wrist"], "prompt": "task"},
           "paths": {"run": str(tmp_path)}}
    dataset = SimpleNamespace(index=[(0, 0)], rows={0: {"observation.state": np.zeros((16, 7)),
                                                      "action": np.zeros((16, 7))}})

    class Dataset:
        index, rows = dataset.index, dataset.rows

        def __len__(self):
            return 1

    prediction = np.repeat(np.arange(1, horizon + 1, dtype=np.float32)[:, None], 7, axis=1)
    policy = SimpleNamespace(infer=lambda observation, noise: {"actions": prediction})
    monkeypatch.setattr(inference, "checkpoint_file", lambda path: tmp_path / "fake.pt")
    monkeypatch.setattr(inference, "create_policy", lambda path: policy)
    monkeypatch.setattr(inference, "load_policy_checkpoint", lambda path: {"config": cfg})
    monkeypatch.setattr(inference, "split_data", lambda config: {"train_episodes": [0], "val_episodes": []})
    monkeypatch.setattr(inference, "WindowDataset", lambda *args: Dataset())
    monkeypatch.setattr(inference, "artifact_root", lambda config: tmp_path)
    monkeypatch.setattr(inference.np, "load", lambda path, **kwargs:
                        np.zeros((16, 256, 8, 8), np.float32) if "latent" in str(path)
                        else np.zeros((16, 224, 224, 3), np.uint8))
    result = inference.evaluate(cfg, limit=1)
    np.testing.assert_allclose(result["native_action_mse_per_dim"], np.mean(prediction ** 2, axis=0))
    np.testing.assert_allclose(result["native_action_mse_by_prediction_step"], prediction ** 2)
    if horizon >= 6:
        np.testing.assert_allclose(result["te_native_action_mse_per_dim"], np.mean(prediction[5:] ** 2, axis=0))
    else:
        assert result["te_native_action_mse_per_dim"] is None
    np.testing.assert_allclose(result["prediction_step_target_offset_s"], np.arange(1, horizon + 1) / 30)
    assert result["te_prediction_step_start"] == 6
    assert np.shape(result["native_action_mse_per_dim"]) == (7,)
