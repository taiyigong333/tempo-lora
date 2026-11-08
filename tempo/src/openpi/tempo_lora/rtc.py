"""对解码动作执行RTC软前缀VJP，使64维潜变量模型也能约束动作重叠段。"""

import math

import numpy as np
import torch


def prefix_weights(delay, overlap, horizon):
    if type(delay) is not int or not 0 <= delay <= overlap <= horizon:
        raise ValueError("RTC延迟和重叠长度不合法")
    index = np.arange(horizon, dtype=np.float32)
    fraction = (overlap - index) / max(overlap - delay + 1, 1)
    soft = fraction * np.expm1(fraction) / math.expm1(1)
    return np.where(index < delay, 1., np.where(index < overlap, soft, 0.)).astype(np.float32)


def guided_velocity(velocity_fn, decode_fn, x, time, prefix, weights, max_guidance):
    # 当前flow使用t=1噪声到t=0数据，故引导速度符号与正向积分相反。
    with torch.enable_grad():
        value = x.detach().requires_grad_(True)
        velocity = velocity_fn(value)
        estimate = decode_fn(value - time * velocity)
        error = ((prefix - estimate) * weights).detach()
        correction = torch.autograd.grad(estimate, value, grad_outputs=error)[0]
    t = float(time)
    gain = min(max_guidance, (t * t + (1 - t) ** 2) / max(t * (1 - t), 1e-8))
    result = velocity.detach() - gain * correction.detach()
    if not torch.isfinite(result).all():
        raise RuntimeError("RTC引导产生非有限速度")
    return result
