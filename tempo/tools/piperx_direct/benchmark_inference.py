"""固定录制帧和噪声，交替比较零残差模型的原路径和省算路径。"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from openpi.tempo_full.inference import can_skip_sam2, create_policy, read_deployment
from openpi.tempo_lora.inference import OnlinePolicy


def load_cases(path):
    cases = []
    index = 0
    with np.load(path, allow_pickle=False) as source:
        states = source['states']
        for case, count in enumerate((1, 4, 8, 2, 8, 4, 1, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8)):
            frames = []
            for timestamp in range(index, index + count):
                frame = timestamp % len(states)
                frames.append({'timestamp': timestamp / 30, 'state': states[frame],
                               'images': {camera: source[camera][frame] for camera in ('head', 'left_wrist')}})
            mode = ('tempo', 'rtc', 'vlash')[case % 3]
            request = {'episode_id': 'fixed', 'mode': mode, 'feedback': frames}
            if mode == 'rtc':
                request.update(rtc_prefix=np.tile(frames[-1]['state'], (30, 1)), rtc_delay=4)
            elif mode == 'vlash':
                request.update(future_state=frames[-1]['state'], future_state_offset_steps=3)
            cases.append(request)
            index += count
    return cases


def run(online, cases, noise_shape):
    online.reset()
    actions, times = [], []
    for case, request in enumerate(cases):
        noise = np.random.default_rng(410 + case).standard_normal(noise_shape, dtype=np.float32)
        result = online.infer(request, noise=noise)
        output = result['actions']
        if output.shape != (30, 7) or not np.isfinite(output).all():
            raise ValueError('动作输出不符合 [30, 7] 有限数约定')
        actions.append(output.copy())
        times.append(result['end_to_end_ms'])
    torch.cuda.synchronize()
    return np.stack(actions), times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    settings, cfg, config, norm = read_deployment(args.config)
    torch.set_num_threads(settings['cpu_threads'])
    base = create_policy(settings, cfg, config, norm)
    if not can_skip_sam2(base._model):
        raise ValueError('权重未满足冻结且 SAM2 残差为零的条件')
    cases = load_cases(args.replay)
    paths = {
        name: OnlinePolicy(base, cfg, parallel_sam=settings.get('parallel_sam', True),
                           compile_inference=settings.get('compile_inference', False),
                           latest_sam_features_only=settings.get('latest_sam_features_only', False),
                           sam2_full_pointer_span=settings.get('sam2_full_pointer_span', False),
                           rtc_max_guidance=settings['rtc_max_guidance'], skip_sam2=skip)
        for name, skip in (('baseline', False), ('optimized', True))
    }
    noise_shape = (config.flow_positions, config.flow_dim)
    print('权重零残差检查已通过，正在预热和配对测试', flush=True)
    reference, _ = run(paths['baseline'], cases, noise_shape)
    run(paths['optimized'], cases, noise_shape)
    timings = {name: [] for name in paths}
    for repeat in range(4):
        order = list(paths) if not repeat % 2 else list(reversed(paths))
        for name in order:
            actual, times = run(paths[name], cases, noise_shape)
            # 按字节比较浮点值，同样核对正负零。
            if actual.tobytes() != reference.tobytes():
                raise AssertionError(f'{name} 动作不同，最大绝对差 {np.max(np.abs(actual - reference))}')
            timings[name].append(times)
    report = {'checkpoint': settings['checkpoint'], 'step': settings['step'],
              'predict_ut': config.predict_ut, 'rounds': 4, 'cases_per_round': len(cases),
              'feedback_frames_per_round': sum(len(case['feedback']) for case in cases),
              'actions_bitwise_equal': True, 'denoise_steps': settings['denoise_steps'],
              'rtc_max_guidance': settings['rtc_max_guidance'], 'robot_commands': 0, 'timings': {}}
    for name, rows in timings.items():
        samples = np.asarray(rows)
        report['timings'][name] = {'p50_ms': float(np.median(samples)),
                                   'p95_ms': float(np.quantile(samples, .95)),
                                   'rtc_p95_ms': float(np.quantile(samples[:, 1::3], .95))}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    paths['baseline'].sam.executor.shutdown()


if __name__ == '__main__':
    main()
