"""隔离验证模型服务身份、占用保护和资产检查，不启动真实服务。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

path = Path(__file__).resolve().parents[2]/'trained_model/scripts/manage_model_service.py'
spec = importlib.util.spec_from_file_location('model_service_control', path)
manager = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manager)


def test_status_distinguishes_three_full_checkpoints(monkeypatch, tmp_path):
    catalog = manager.services(tmp_path/'tempo')
    for model_id in ('clockwise4000', 'clockwise5000', 'all3935000'):
        entry = catalog[model_id]
        active = {'args': f'python -m openpi.tempo_full.inference --config ignored --config {entry["config"]}'}
        assert manager.loaded_id(entry, active, catalog) == model_id


def test_unknown_process_identity_is_rejected(monkeypatch, tmp_path):
    entry = manager.services(tmp_path/'tempo')['clockwise4000']
    responses = iter([SimpleNamespace(returncode=0), SimpleNamespace(stdout='123'), SimpleNamespace(stdout='python unrelated.py')])
    monkeypatch.setattr(manager, 'command', lambda *args: next(responses))
    with pytest.raises(RuntimeError, match='身份不匹配'):
        manager.process(entry)


def test_switch_checks_assets_before_stopping_any_model(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'status', lambda *args: {'selected_running': False, 'running_models': []})
    stop = Mock()
    monkeypatch.setattr(manager, 'stop', stop)
    with pytest.raises(FileNotFoundError):
        manager.manage('start', 'clockwise4000', tmp_path/'tempo')
    stop.assert_not_called()


def test_connected_model_prevents_any_switch_mutation(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'status', lambda *args: {'selected_running': False, 'running_models': [{'connections': 1, 'port': 8002}]})
    monkeypatch.setattr(manager, 'validate_assets', lambda *args: None)
    stop = Mock()
    monkeypatch.setattr(manager, 'stop', stop)
    with pytest.raises(RuntimeError, match='客户端连接'):
        manager.manage('start', 'clockwise4000', tmp_path/'tempo')
    stop.assert_not_called()


def test_already_running_start_is_idempotent(monkeypatch, tmp_path):
    state = {'selected_running': True, 'running_models': [{'connections': 1, 'port': 8002}]}
    monkeypatch.setattr(manager, 'status', lambda *args: state)
    stop = Mock()
    monkeypatch.setattr(manager, 'stop', stop)
    result = manager.manage('start', 'clockwise5000', tmp_path/'tempo')
    assert result['selected_running']
    stop.assert_not_called()


def test_stop_terminates_only_verified_pid(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'process', Mock(side_effect=[{'pid': 123, 'args': 'verified'}, RuntimeError('不应在退出过程中重复核验进程命令')]))
    monkeypatch.setattr(manager, 'connections', lambda *args: 0)
    monkeypatch.setattr(manager, 'command', lambda *args: SimpleNamespace(stdout='', returncode=1))
    kill = Mock()
    monkeypatch.setattr(manager.os, 'kill', kill)
    manager.stop(manager.services(tmp_path/'tempo')['clockwise4000'])
    kill.assert_called_once_with(123, manager.signal.SIGTERM)
    assert manager.process.call_count == 1


def test_unmanaged_target_port_is_checked_before_stopping_current_model(monkeypatch, tmp_path):
    monkeypatch.setattr(manager, 'status', lambda *args: {'selected_running': False,
                        'running_models': [{'connections': 0, 'port': 8000}]})
    monkeypatch.setattr(manager, 'validate_assets', lambda *args: None)
    monkeypatch.setattr(manager, 'command', lambda *args: SimpleNamespace(stdout='LISTEN'))
    stop = Mock()
    monkeypatch.setattr(manager, 'stop', stop)
    with pytest.raises(RuntimeError, match='端口'):
        manager.manage('start', 'clockwise4000', tmp_path/'tempo')
    stop.assert_not_called()
