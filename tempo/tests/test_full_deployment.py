"""覆盖30步契约、历史帧索引及训练数据变换的一致性。"""
import dataclasses
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models_pytorch.ut_decoder import UtActionDecoder
from openpi.policies.yam_dual_policy import YamInputs, YamOutputs
from openpi.shared.normalize import NormStats, serialize_json
from openpi.tempo_full.inference import FullPolicy, read_deployment
from openpi.tempo_lora.inference import OnlinePolicy


class FakeSam:
    def reset(self):
        pass


class FullDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = Pi0Config(pi05=True, predict_ut=True, use_sam2_fusion=True,
                               use_action_history=True, action_history_dim=7,
                               obs_history=3, obs_history_stride_s=.333,
                               action_horizon=30, ut_window=30)
        for name in ('model.safetensors', 'tokenizer', 'sam2_checkpoint'):
            (self.root/name).touch()
        self.write_config(self.model)
        norm = {key: NormStats(mean=np.zeros(7), std=np.ones(7), q01=-np.ones(7), q99=np.ones(7))
                for key in ('state', 'actions')}
        (self.root/'norm_stats.json').write_text(serialize_json(norm))
        self.save_decoder(30)
        settings = dict(name='test_full', step=5000, checkpoint='.', decoder='decoder.pt',
                        tokenizer='tokenizer', sam2_checkpoint='sam2_checkpoint', sam2_repo='.',
                        prompt='grab the rotating red square')
        self.path = self.root/'deployment.json'
        self.path.write_text(json.dumps(settings))

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, config):
        (self.root/'model_config.json').write_text(json.dumps(dataclasses.asdict(config)))

    def save_decoder(self, window):
        decoder = UtActionDecoder(hidden_dim=16, window=window, action_dim=7, n_layers=1, nhead=4)
        torch.save(dict(act_dec=decoder.state_dict(), action_space='policy_norm',
                        model_cfg=dict(latent_dim=64, hidden_dim=16, window=window,
                                       action_dim=7, dec_layers=1, nhead=4)), self.root/'decoder.pt')

    def test_bfloat16_actions_convert_before_numpy_boundary(self):
        from unittest.mock import Mock
        policy = FullPolicy.__new__(FullPolicy)
        policy._model = Mock()
        policy._model.sample_actions.return_value = torch.zeros((1, 30, 32), dtype=torch.bfloat16)
        actual = policy.sample_float32('cuda', object())
        self.assertEqual(actual.dtype, torch.float32)
        self.assertEqual(np.asarray(actual).shape, (1, 30, 32))

    def test_optimization_flags_enable_explicitly(self):
        settings = json.loads(self.path.read_text())
        settings.update(skip_missing_cameras=True, latest_sam_features_only=True)
        self.path.write_text(json.dumps(settings))
        settings, _, model, _ = read_deployment(self.path)
        self.assertTrue(model.skip_missing_cameras)
        self.assertTrue(settings['latest_sam_features_only'])

    def test_optimization_flags_reject_string(self):
        settings = json.loads(self.path.read_text())
        settings['skip_missing_cameras'] = 'false'
        self.path.write_text(json.dumps(settings))
        with self.assertRaisesRegex(ValueError, '必须为布尔值'):
            read_deployment(self.path)

    def test_paper_guidance_reaches_online_policy(self):
        from unittest.mock import patch
        from openpi.tempo_full.inference import main
        settings = json.loads(self.path.read_text())
        settings.update(inference_profile='rtc_paper', denoise_steps=5, rtc_max_guidance=5.,
                        host='127.0.0.1', port=8002)
        self.path.write_text(json.dumps(settings))
        with patch('sys.argv', ['inference', '--config', str(self.path)]), \
             patch('openpi.tempo_full.inference.create_policy'), \
             patch('openpi.tempo_full.inference.warmup'), \
             patch('openpi.tempo_full.inference.WebsocketPolicyServer'), \
             patch('openpi.tempo_full.inference.OnlinePolicy') as online:
            main()
            self.assertEqual(online.call_args.kwargs['rtc_max_guidance'], 5.)

    def test_invalid_sampling_parameters_rejected(self):
        for key, value in (('denoise_steps', 0), ('denoise_steps', True),
                           ('rtc_max_guidance', -1), ('rtc_max_guidance', float('nan'))):
            with self.subTest(key=key, value=value):
                settings = json.loads(self.path.read_text())
                settings[key] = value
                self.path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    read_deployment(self.path)
                settings.pop(key)
                self.path.write_text(json.dumps(settings))

    def test_full_contract_and_decoder(self):
        settings, cfg, model, norm = read_deployment(self.path)
        self.assertEqual(model.action_horizon, 30)
        self.assertEqual((model.flow_positions, model.flow_dim), (1, 64))
        self.assertFalse(model.lora_enabled or model.compile_sample_actions or model.compact_prefix)
        self.assertEqual(cfg['data']['fps'], 30)
        self.assertEqual(set(norm), {'state', 'actions'})

    def test_mismatched_decoder_rejected(self):
        self.save_decoder(16)
        with self.assertRaisesRegex(ValueError, '窗口或维度'):
            read_deployment(self.path)

    def test_incomplete_download_rejected(self):
        (self.root/'model.safetensors').unlink()
        (self.root/'model.safetensors.incomplete').touch()
        with self.assertRaises(FileNotFoundError):
            read_deployment(self.path)

    def test_wrong_image_history_rejected(self):
        self.write_config(dataclasses.replace(self.model, obs_history_stride_s=1.0))
        with self.assertRaisesRegex(ValueError, '图像历史'):
            read_deployment(self.path)

    def test_online_history_matches_training_indices_and_zero_padding(self):
        _, cfg, _, _ = read_deployment(self.path)
        online = OnlinePolicy(object(), cfg, sam_stream=FakeSam())
        states = []
        for t in range(331):
            state = np.full(7, t / 1000, np.float32)
            states.append(state)
            images = {key: np.full((224, 224, 3), t % 256, np.uint8) for key in cfg['data']['cameras']}
            online.ingest(dict(timestamp=t/30, state=state, images=images), encode_sam=False)
            if t not in (0, 15, 20, 300, 330):
                continue
            actual = online.inputs()
            offsets = np.maximum(0, np.array([t-20, t-10, t]))
            np.testing.assert_array_equal(actual['images']['head'][:, 0, 0, 0], offsets % 256)
            # action[i] = state[i+1]，历史仅可使用已经到达的实测状态。
            indices = np.arange(t-300, t)
            valid = indices >= 0
            history = np.zeros((300, 7), np.float32)
            history[valid] = np.asarray(states)[indices[valid] + 1]
            np.testing.assert_allclose(actual['action_history'], history.reshape(10,30,7).mean(1))
            np.testing.assert_array_equal(actual['action_history_is_pad'], (~valid).reshape(10,30).all(1))

    def test_action_roundtrip_keeps_gripper_absolute_and_30_steps(self):
        _, _, _, norm = read_deployment(self.path)
        mask = transforms.make_bool_mask(6, -1)
        state = np.arange(7, dtype=np.float32) / 10
        actions = np.tile(state, (30,1)) + np.arange(30)[:,None] / 1000
        sample = {'state': state.copy(), 'actions': actions.copy()}
        sample = transforms.DeltaActions(mask)(sample)
        sample = transforms.Normalize(norm, use_quantiles=True)(sample)
        sample = transforms.PadStatesAndActions(32)(sample)
        sample = transforms.Unnormalize(norm, use_quantiles=True)(sample)
        sample = transforms.AbsoluteActions(mask)(sample)
        actual = YamOutputs(action_dim=7)(sample)['actions']
        np.testing.assert_allclose(actual, actions, atol=1e-6)
        self.assertEqual(actual.shape, (30,7))


if __name__ == '__main__':
    unittest.main()
