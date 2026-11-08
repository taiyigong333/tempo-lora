"""校验反馈时序、实测插值和断线状态，防止在线历史被污染。"""

from collections import deque
import unittest
from unittest.mock import Mock

import numpy as np

from tempo_client.live import interpolate, letterbox
from tempo_client.transport import Client, validate_frames


def frame(stamp=0):
    return {"timestamp": stamp, "state": np.zeros(7, np.float32),
            "images": {name: np.zeros((224, 224, 3), np.uint8) for name in ("head", "left_wrist")}}


class ClientTests(unittest.TestCase):
    def test_continuous_feedback(self):
        self.assertAlmostEqual(validate_frames([frame(0), frame(1 / 30)], 30), 1 / 30)
        self.assertAlmostEqual(validate_frames([frame(2 / 30)], 30, 1 / 30), 2 / 30)

    def test_rejects_missing_feedback(self):
        with self.assertRaises(ValueError):
            validate_frames([frame(0), frame(2 / 30)], 30)
        with self.assertRaises(ValueError):
            validate_frames([], 30)

    def test_rejects_wrong_units_container_or_image(self):
        value = frame()
        value["state"] = np.zeros(7, np.float64)
        with self.assertRaises(ValueError):
            validate_frames([value], 30)
        value = frame()
        value["images"]["head"] = np.zeros((224, 224, 3), np.float32)
        with self.assertRaises(ValueError):
            validate_frames([value], 30)

    def test_interpolates_real_samples(self):
        samples = deque([(1.0, np.array([0.])), (1.02, np.array([2.]))])
        np.testing.assert_allclose(interpolate(samples, 1.01, .1), [1.])
        with self.assertRaises(LookupError):
            interpolate(samples, 1.03, .1)
        with self.assertRaises(RuntimeError):
            interpolate(samples, 1.01, .01)

    def test_letterbox_preserves_rgb_and_black_padding(self):
        rgb = np.full((480, 640, 3), [255, 0, 0], np.uint8)
        result = letterbox(rgb)
        self.assertEqual(result.shape, (224, 224, 3))
        np.testing.assert_array_equal(result[112, 112], [255, 0, 0])
        self.assertFalse(result[:28].any())
        self.assertFalse(result[-28:].any())

    def test_failed_rpc_closes_without_retry(self):
        client = Client.__new__(Client)
        client.cfg = {"model": {"fps": 30, "action_horizon": 16, "action_dim": 7}}
        client.previous, client.episode = None, "test"
        client.packer = Mock()
        client.ws = Mock()
        client._receive = Mock(side_effect=TimeoutError("timeout"))
        with self.assertRaises(TimeoutError):
            client.infer([frame()])
        client.ws.send.assert_called_once()
        client.ws.close.assert_called_once()
        self.assertIsNone(client.previous)

    def test_invalid_reply_closes_without_advancing_history(self):
        client = Client.__new__(Client)
        client.cfg = {"model": {"fps": 30, "action_horizon": 16, "action_dim": 7}}
        client.previous, client.episode = None, "test"
        client.packer, client.ws = Mock(), Mock()
        client._receive = Mock(return_value={"actions": np.zeros((16, 7), np.float64)})
        with self.assertRaises(ValueError):
            client.infer([frame()])
        client.ws.close.assert_called_once()
        self.assertIsNone(client.previous)


if __name__ == "__main__":
    unittest.main()
