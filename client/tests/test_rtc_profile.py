"""验证论文延迟窗口、旧配置兼容和推理参数握手。"""
import copy
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from tempo_client.cli import DEFAULT_CONFIG, load_config
from tempo_client.delay import DelayEstimator
from tempo_client.settings import select_runtime
from tempo_client.transport import Client


class RTCProfileTests(unittest.TestCase):
    def test_window_max_forgets_expired_peak(self):
        estimate = DelayEstimator({'rtc_delay_estimator': 'window_max',
                                   'rtc_delay_buffer_size': 10, 'rtc_initial_delay_steps': 8}, 30)
        self.assertAlmostEqual(estimate.seconds, 8 / 30)
        estimate.update(.4)
        for _ in range(9):
            estimate.update(.1)
        self.assertEqual(estimate.seconds, .4)
        estimate.update(.1)
        self.assertEqual(estimate.seconds, .1)
        self.assertEqual(len(estimate.samples), 10)

    def test_legacy_peak_matches_previous_formula(self):
        estimate = DelayEstimator({}, 30)
        expected = 0.
        for value in (.2, .1, .1, .3, .2):
            expected = max(value, expected * .95)
            estimate.update(value)
            self.assertAlmostEqual(estimate.seconds, expected)

    def test_invalid_delay_configuration(self):
        for config in ({'rtc_delay_estimator': 'mean'}, {'rtc_delay_buffer_size': 0},
                       {'rtc_delay_buffer_size': True}, {'rtc_initial_delay_steps': -1}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DelayEstimator(config, 30)
        estimate = DelayEstimator({}, 30)
        for value in (float('nan'), float('inf'), -.1):
            with self.assertRaises(ValueError):
                estimate.update(value)

    def test_paper_config_loads_through_normal_entry(self):
        cfg = load_config(DEFAULT_CONFIG.parent / 'piperx_full_rtc_paper.yaml')
        selected = select_runtime(cfg)
        self.assertEqual(selected['inference']['mode'], 'rtc')
        self.assertEqual(selected['inference']['rtc_delay_buffer_size'], 10)
        self.assertEqual(selected['model']['denoise_steps'], 5)
        self.assertEqual(selected['model']['rtc_max_guidance'], 5.)
        self.assertEqual(selected['model']['action_horizon'], 30)

    def test_handshake_rejects_wrong_sampling_profile(self):
        cfg = load_config(DEFAULT_CONFIG.parent / 'piperx_full_rtc_paper.yaml')
        metadata = {**cfg['model'], 'native_action_dim': 7, 'feedback_fps': 30,
                    'denoise_steps': 10}
        raw, websocket = Mock(), Mock()
        raw.getsockname.return_value = ('192.168.50.1', 32100)
        with patch('tempo_client.transport.socket.create_connection', return_value=raw), \
             patch('tempo_client.transport.connect', return_value=websocket), \
             patch.object(Client, '_receive', return_value=metadata):
            with self.assertRaisesRegex(ValueError, '推理参数不匹配 denoise_steps'):
                Client(cfg)
        websocket.close.assert_called_once()

    def test_bad_window_rejected_before_runtime(self):
        cfg = load_config(DEFAULT_CONFIG)
        cfg['inference']['rtc_delay_buffer_size'] = 0
        with self.assertRaisesRegex(ValueError, '窗口'):
            select_runtime(cfg)


if __name__ == '__main__':
    unittest.main()
