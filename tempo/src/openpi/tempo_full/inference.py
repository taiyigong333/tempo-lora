"""严格加载完整策略权重，复用已验证的在线历史、RTC 与会话协议。"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import safetensors.torch
import torch

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.models_pytorch.ut_decoder import UtActionDecoder
from openpi.policies.policy import Policy
from openpi.policies.yam_dual_policy import YamInputs, YamOutputs
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.shared.normalize import deserialize_json
from openpi.tempo_lora.config import configure_environment
from openpi.tempo_lora.inference import OnlinePolicy
from openpi.tempo_lora.model import require_patch


def read_deployment(path, *, overrides=None):
    path = Path(path).resolve()
    settings = json.loads(path.read_text())
    settings.update({key: value for key, value in (overrides or {}).items() if value is not None})
    validate_runtime(settings)
    settings.setdefault('inference_profile', 'default')
    settings.setdefault('denoise_steps', 10)
    settings.setdefault('rtc_max_guidance', 10.0)
    if type(settings.get('denoise_steps')) is not int or settings['denoise_steps'] < 1:
        raise ValueError('denoise_steps必须为正整数')
    guidance = settings['rtc_max_guidance']
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or not np.isfinite(guidance) or guidance <= 0:
        raise ValueError('rtc_max_guidance必须为有限正数')
    for key in ('compile_inference', 'parallel_sam', 'skip_missing_cameras', 'latest_sam_features_only'):
        if type(settings.get(key, False)) is not bool:
            raise ValueError(f'{key}必须为布尔值')
    # 相对路径以部署配置为基准，便于独立搬移 trained_model。
    for key in ('checkpoint', 'decoder', 'tokenizer', 'sam2_repo', 'sam2_checkpoint'):
        settings[key] = str((path.parent / settings[key]).resolve())
    directory = Path(settings['checkpoint'])
    required = [directory / name for name in ('model.safetensors', 'model_config.json', 'norm_stats.json')]
    required.extend(Path(settings[key]) for key in ('decoder', 'tokenizer', 'sam2_checkpoint'))
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f'部署文件尚未就绪 {missing}')
    values = json.loads((directory / 'model_config.json').read_text())
    allowed = {field.name for field in dataclasses.fields(Pi0Config)}
    if set(values) - allowed:
        raise ValueError(f'模型配置包含当前源码不支持的字段 {set(values) - allowed}')
    # 完整权重加载不经过旧 LoRA 的基础权重、adapter 注入和精度重写。
    values.update(ut_decoder_path=settings['decoder'], compile_sample_actions=False,
                  compact_prefix=False, lora_enabled=False,
                  skip_missing_cameras=settings.get("skip_missing_cameras", False))
    config = Pi0Config(**values)
    if not (config.pi05 and config.predict_ut and config.use_sam2_fusion and config.use_action_history):
        raise ValueError('当前入口要求完整 π0.5 TEMPO 潜变量策略')
    if (config.action_dim, config.ut_dim, config.action_history_dim) != (32, 64, 7):
        raise ValueError('模型维度与 PiperX 部署契约不一致')
    if (config.obs_history, round(config.obs_history_stride_s * 30)) != (3, 10):
        raise ValueError('当前训练数据的图像历史必须对应 t-20、t-10、t')
    if config.action_history_steps != 10 or (config.sam2_num_tokens, config.sam2_token_dim) != (64, 256):
        raise ValueError('历史反馈桶或 SAM2 特征维度不匹配')
    decoder = UtActionDecoder.from_pretrained(settings['decoder'])
    if (decoder.window, decoder.latent_dim, decoder.action_dim) != (config.action_horizon, config.ut_dim, 7):
        raise ValueError('策略与独立动作解码器的窗口或维度不匹配')
    norm = deserialize_json((directory / 'norm_stats.json').read_text())
    for name in ('state', 'actions'):
        stat = norm[name]
        if any(np.asarray(getattr(stat, key)).shape != (7,) for key in ('mean', 'std', 'q01', 'q99')):
            raise ValueError(f'{name} 归一化统计必须为7维')
        if not all(np.isfinite(getattr(stat, key)).all() for key in ('mean', 'std', 'q01', 'q99')):
            raise ValueError(f'{name} 归一化统计含非有限值')
    cfg = {'name': settings['name'], 'model': dataclasses.asdict(config),
           'data': {'fps': 30, 'action_dim': 7, 'cameras': ['head', 'left_wrist'],
                    'action_history_seconds': 10, 'prompt': settings['prompt']},
           'training': {}, 'paths': {key: settings[key] for key in ('tokenizer', 'sam2_repo', 'sam2_checkpoint')}}
    return settings, cfg, config, norm


class FullPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sample_actions = self.sample_float32

    def sample_float32(self, device, observation, **kwargs):
        # 与全量训练评测入口一致，保留原始混合参数精度并使用 BF16 autocast。
        # 在进入 Policy 的 NumPy 边界前转换，NumPy 无法直接接收 BF16 张量。
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return self._model.sample_actions(device, observation, **kwargs).float()


def create_policy(settings, cfg, config, norm):
    configure_environment(cfg)
    require_patch()
    model = PI0Pytorch(config)
    safetensors.torch.load_model(model, Path(settings['checkpoint']) / 'model.safetensors', strict=True)
    model.eval().requires_grad_(False)
    model._ut_decoder.eval().requires_grad_(False)
    if settings.get('compile_inference', False):
        model.denoise_step = torch.compile(model.denoise_step, dynamic=True, mode=settings['denoise_compile_mode'])
        model._ut_decoder.forward = torch.compile(model._ut_decoder.forward, dynamic=False, mode=settings['denoise_compile_mode'])
    mask = transforms.make_bool_mask(6, -1)
    def native_inputs(result):
        # 训练数据入口在归一化后统一为 float32，在线保留同样边界。
        return {**result, 'state': np.asarray(result['state'], dtype=np.float32)}
    def native_actions(result):
        return {**result, 'actions': np.asarray(result['actions'], dtype=np.float32)}
    policy = FullPolicy(model, is_pytorch=True, pytorch_device='cuda',
                        transforms=[YamInputs(model_type=config.model_type), transforms.DeltaActions(mask),
                                    transforms.Normalize(norm, use_quantiles=True),
                                    transforms.TokenizePrompt(PaligemmaTokenizer(config.max_token_len),
                                                              discrete_state_input=config.discrete_state_input),
                                    transforms.PadStatesAndActions(config.action_dim), native_inputs],
                        output_transforms=[transforms.Unnormalize(norm, use_quantiles=True),
                                           transforms.AbsoluteActions(mask), YamOutputs(action_dim=7), native_actions],
                        sample_kwargs={'num_steps': settings['denoise_steps']},
                        metadata={'config': settings['name'], 'step': settings['step'],
                                  'checkpoint': str(settings['checkpoint']), 'training_method': 'full_finetune',
                                  'native_action_dim': 7, 'action_horizon': config.action_horizon,
                                  'action_space': 'absolute_joint_rad_gripper_m', 'feedback_fps': 30,
                                  'execute_steps': 1, 'modes': ['tempo', 'temporal_ensemble', 'rtc'],
                                  'rtc_algorithm': 'decoded_action_vjp_soft_prefix',
                                  'task_prompts': [settings['prompt']], 'protocol_version': 2,
                                  'denoise_steps': settings['denoise_steps'], 'ema_enabled': False,
                                  'inference_profile': settings['inference_profile'],
                                  'compile_inference': settings.get('compile_inference', False),
                                  'skip_missing_cameras': settings.get('skip_missing_cameras', False),
                                  'latest_sam_features_only': settings.get('latest_sam_features_only', False),
                                  'cpu_threads': torch.get_num_threads(),
                                  'denoise_compile_mode': settings['denoise_compile_mode'],
                                  'warmup_batches': settings['warmup_batches'],
                                  'warmup_repeats': settings['warmup_repeats'],
                                  'observation_offsets_frames': [-20, -10, 0]})
    policy.tempo_norm_stats = norm
    return policy


def validate_runtime(settings):
    # 旧配置缺省沿用8线程，新的部署选项在模型加载前统一核验。
    settings.setdefault('cpu_threads', 8)
    settings.setdefault('denoise_compile_mode', 'default')
    settings.setdefault('warmup_batches', [1, 4, 8])
    settings.setdefault('warmup_repeats', 3)
    for key in ('cpu_threads', 'warmup_repeats'):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f'{key}必须为正整数')
    if settings['denoise_compile_mode'] not in ('default', 'reduce-overhead', 'max-autotune-no-cudagraphs'):
        raise ValueError('不支持的denoise_compile_mode')
    batches = settings['warmup_batches']
    if not isinstance(batches, list) or not batches or any(type(n) is not int or not 1 <= n <= 300 for n in batches):
        raise ValueError('warmup_batches必须为包含1..300整数的非空列表')


def warmup(policy, norm, settings=None):
    options = dict(settings or {})
    validate_runtime(options)
    state = np.asarray(norm['state'].mean, np.float32)
    images = {camera: np.zeros((224, 224, 3), np.uint8) for camera in ('head', 'left_wrist')}
    expected = (policy.cfg['model']['action_horizon'], 7)
    timings = []
    # 覆盖首帧、连续记忆和不同积压帧数，并在开放端口前完成RTC反向图编译。
    try:
        for mode in ('tempo', 'rtc'):
            policy.reset()
            index = 0
            previous = np.tile(state, (expected[0], 1))
            for repeat in range(options['warmup_repeats']):
                for count in options['warmup_batches']:
                    frames = [{'timestamp': i / 30, 'state': state, 'images': images}
                              for i in range(index, index + count)]
                    request = {'episode_id': '__warmup__', 'mode': mode, 'feedback': frames}
                    if mode == 'rtc':
                        request.update(rtc_prefix=previous, rtc_delay=1)
                    result = policy.infer(request)
                    previous = result['actions']
                    if previous.shape != expected or not np.isfinite(previous).all():
                        raise ValueError('预热动作形状或数值异常')
                    timings.append({'mode': mode, 'repeat': repeat, 'feedback_frames': count,
                                    'end_to_end_ms': result['end_to_end_ms']})
                    index += count
        torch.cuda.synchronize()
    finally:
        # 预热历史必须丢弃，首个真实episode从干净状态开始。
        policy.reset()
    print(json.dumps({'warmup': 'passed', 'shape': expected, 'samples': timings}), flush=True)
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host')
    parser.add_argument('--port', type=int)
    parser.add_argument('--check-files', action='store_true')
    parser.add_argument('--warmup-only', action='store_true')
    parser.add_argument('--cpu-threads', type=int)
    parser.add_argument('--denoise-steps', type=int)
    parser.add_argument('--rtc-max-guidance', type=float)
    parser.add_argument('--compile-mode', dest='denoise_compile_mode')
    parser.add_argument('--compile-inference', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--parallel-sam', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--warmup-batches', type=int, nargs='+')
    parser.add_argument('--warmup-repeats', type=int)
    args = parser.parse_args()
    keys = ('cpu_threads', 'denoise_steps', 'rtc_max_guidance', 'denoise_compile_mode',
            'compile_inference', 'parallel_sam', 'warmup_batches', 'warmup_repeats')
    settings, cfg, config, norm = read_deployment(
        args.config, overrides={key: getattr(args, key) for key in keys})
    if args.check_files:
        from safetensors import safe_open
        with safe_open(str(Path(settings['checkpoint']) / 'model.safetensors'), framework='pt') as weights:
            count = len(weights.keys())
        print(json.dumps({'files': 'ready', 'tensors': count, 'action_horizon': config.action_horizon,
                          'observation_offsets_frames': [-20, -10, 0],
                          'inference_profile': settings['inference_profile'],
                          'denoise_steps': settings['denoise_steps'],
                          'rtc_max_guidance': settings['rtc_max_guidance'],
                          'cpu_threads': settings['cpu_threads'],
                          'compile_inference': settings.get('compile_inference', False),
                          'parallel_sam': settings.get('parallel_sam', True),
                          'denoise_compile_mode': settings['denoise_compile_mode'],
                          'warmup_batches': settings['warmup_batches'],
                          'warmup_repeats': settings['warmup_repeats']}), flush=True)
        return
    torch.set_num_threads(settings['cpu_threads'])
    startup = time.monotonic()
    base = create_policy(settings, cfg, config, norm)
    online = OnlinePolicy(base, cfg, rtc_max_guidance=settings['rtc_max_guidance'],
                          parallel_sam=settings.get('parallel_sam', True),
                          compile_inference=settings.get('compile_inference', False),
                          latest_sam_features_only=settings.get('latest_sam_features_only', False))
    warmup(online, norm, settings)
    if args.warmup_only:
        print(json.dumps({'warmup_only': 'passed', 'metadata': online.metadata}), flush=True)
        return
    print(json.dumps({'service_ready': True, 'startup_seconds': time.monotonic() - startup,
                      'metadata': online.metadata}), flush=True)
    WebsocketPolicyServer(online, host=args.host or settings['host'],
                          port=args.port if args.port is not None else settings['port'],
                          metadata=online.metadata, single_client=True).serve_forever()


if __name__ == '__main__':
    main()
