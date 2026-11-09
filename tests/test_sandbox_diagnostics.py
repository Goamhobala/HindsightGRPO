import errno
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hindsight_grpo.rewards import RewardRunner, SandboxUnavailable


def test_namespace_failure_reports_resource_evidence(monkeypatch):
    runner = RewardRunner(require_slurm=False)
    monkeypatch.setenv('WANDB_API_KEY', 'private-test-value')
    def start(command, **kwargs):
        kwargs['stderr'].write(b'bwrap: Creating new namespace failed: Resource temporarily unavailable\n')
        return SimpleNamespace(wait=lambda **wait_kwargs: 1)
    evidence = {'launcher_nproc_soft': 4096, 'visible_uid_threads': 120,
                'cgroup_pids': [{'current': '100', 'max': '100'}]}
    with patch.object(runner, '_command', return_value=['fake-bwrap']), \
            patch('hindsight_grpo.rewards.subprocess.Popen', side_effect=start), \
            patch('hindsight_grpo.rewards.sandbox_diagnostics', return_value=evidence):
        with pytest.raises(SandboxUnavailable, match='Resource diagnostics') as error:
            runner._run(Path('not-executed.json'))
    message = str(error.value)
    assert 'visible_uid_threads' in message and 'cgroup_pids' in message
    assert 'inside the worker' in message
    assert 'private-test-value' not in message


def test_account_fork_failure_is_reported_as_infrastructure_failure():
    runner = RewardRunner(require_slurm=False)
    with patch.object(runner, '_command', return_value=['fake-bwrap']), \
            patch('hindsight_grpo.rewards.subprocess.Popen',
                  side_effect=OSError(errno.EAGAIN, 'Resource temporarily unavailable')), \
            patch('hindsight_grpo.rewards.sandbox_diagnostics', return_value={'uid': 1001}):
        with pytest.raises(SandboxUnavailable, match='Cannot launch the reward sandbox'):
            runner._run(Path('not-executed.json'))


def probe_module():
    path = Path(__file__).parents[1] / 'scripts/check_sandbox.py'
    spec = importlib.util.spec_from_file_location('check_sandbox_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_standalone_probe_requires_compute_allocation(monkeypatch):
    probe = probe_module()
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with patch('sys.argv', ['check_sandbox.py']), patch.object(probe, 'RewardRunner') as runner:
        with pytest.raises(SystemExit, match='SLURM compute allocation'):
            probe.main()
    runner.assert_not_called()


def test_probe_saves_failure_diagnostics_for_the_user(tmp_path, monkeypatch):
    probe = probe_module()
    output = tmp_path / 'sandbox.json'
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    runner = SimpleNamespace(check_available=lambda: (_ for _ in ()).throw(
        SandboxUnavailable('namespace task limit reached')))
    with patch('sys.argv', ['check_sandbox.py', '--output', str(output)]), \
            patch.object(probe, 'RewardRunner', return_value=runner), \
            patch.object(probe, 'sandbox_diagnostics', return_value={'visible_uid_threads': 120}):
        with pytest.raises(SystemExit, match='namespace task limit reached'):
            probe.main()
    report = json.loads(output.read_text())
    assert report['status'] == 'failed'
    assert report['resources']['visible_uid_threads'] == 120
    assert report['slurm_job_id'] == '123'
