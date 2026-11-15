"""RTC延迟估计，支持论文的有限窗口最大值和已有衰减峰值。"""

from collections import deque
import math


class DelayEstimator:
    @staticmethod
    def validate(config):
        method = config.get('rtc_delay_estimator', 'decayed_peak')
        if method not in ('decayed_peak', 'window_max'):
            raise ValueError('RTC延迟估计方式必须为decayed_peak或window_max')
        size = config.get('rtc_delay_buffer_size', 10)
        if type(size) is not int or size < 1:
            raise ValueError('RTC延迟历史窗口必须为正整数')
        initial = config.get('rtc_initial_delay_steps', 0)
        if type(initial) is not int or initial < 0:
            raise ValueError('RTC初始延迟必须为非负整数步')

    def __init__(self, config, fps):
        self.validate(config)
        self.method = config.get('rtc_delay_estimator', 'decayed_peak')
        size = config.get('rtc_delay_buffer_size', 10) if self.method == 'window_max' else 1
        # 首次引导前的非引导请求通常更快，初始化估计覆盖引导路径的额外耗时。
        self.samples = deque([config.get('rtc_initial_delay_steps', 0) / fps], maxlen=size)

    @property
    def seconds(self):
        return max(self.samples)

    def update(self, seconds):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('实测推理延迟必须为有限非负秒数')
        if self.method == 'decayed_peak':
            seconds = max(seconds, self.samples[-1] * .95)
        self.samples.append(float(seconds))
