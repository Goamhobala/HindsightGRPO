"""Run submission/wrapper scripts with fake cluster commands, never real jobs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from test_slurm_stop import wait_for


@pytest.fixture
def cluster(tmp_path):
    home = tmp_path / 'home/user'
    project = tmp_path / 'scratch/yhxjin001/HindsightGRPO'
    home.mkdir(parents=True)
    for name in ('scripts', 'slurm', '.venv/bin', 'runtime/logs', 'runtime/preflight', 'runtime/metadata'):
        (project / name).mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    for name in ('scripts/submit.sh', 'slurm/run.sh'):
        (project / name).write_bytes((root / name).read_bytes())
    (project / 'scripts/env.sh').write_text('export HINDSIGHT_TIME_LIMIT=2-00:00:00\n')
    (project / 'runtime/metadata/inputs_manifest.json').write_text('{}')
    fake_program = f'#!{sys.executable}\n' + '''
import json
import os
from pathlib import Path
import signal
import sys
import time
name = Path(sys.argv[0]).name
args = sys.argv[1:]
with Path('commands.jsonl').open('a') as stream:
    stream.write(json.dumps({'command': name, 'args': args}) + '\\n')
if name == 'sbatch':
    print('999')
elif args == ['scripts/time_limit.py']:
    print('2-00:00:00')
elif 'hindsight_grpo.train' in args and os.environ.get('TEST_WAIT_FOR_SIGNAL'):
    signal.signal(signal.SIGUSR1, lambda *args: sys.exit(0))
    Path('training_ready').write_text('ready')
    while True:
        time.sleep(.01)
'''
    commands = tmp_path / 'bin'
    commands.mkdir()
    for name in ('sbatch', 'sinfo', 'scontrol', 'sacctmgr'):
        path = commands / name
        path.write_text(fake_program)
        path.chmod(0o755)
    python = project / '.venv/bin/python'
    python.write_text(fake_program)
    python.chmod(0o755)
    env = {**os.environ, 'HOME': str(home), 'PATH': str(commands) + os.pathsep + os.environ['PATH'],
           'WANDB_API_KEY': 'synthetic-not-real', 'EXPERIMENT_GROUP': 'existing-group',
           'SLURM_JOB_ID': '123'}
    return project, env


def calls(project):
    path = project / 'commands.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


@pytest.mark.parametrize('args,arms,mode', [
    ([], ['root', 'entropy_cf', 'r_cf', 'gradient_ce_cf'], []),
    (['--arm', 'root', '--evaluate-only'], ['root'], ['--evaluate-only']),
])
def test_submit_selects_arms_without_duplicating_other_jobs(cluster, args, arms, mode):
    project, env = cluster
    result = subprocess.run(['bash', 'scripts/submit.sh', *args], cwd=project,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    submitted = [c for c in calls(project) if c['command'] == 'sbatch']
    assert len(submitted) == len(arms)
    for call, arm in zip(submitted, arms):
        assert call['args'][-(3 + len(mode)):] == ['slurm/run.sh', arm, 'existing-group', *mode]


def test_evaluation_only_submission_requires_an_explicit_arm(cluster):
    project, env = cluster
    result = subprocess.run(['bash', 'scripts/submit.sh', '--evaluate-only'], cwd=project,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert not calls(project)


@pytest.mark.parametrize('mode', [[], ['--evaluate-only']])
def test_wrapper_forwards_mode_to_python_without_empty_arguments(cluster, mode):
    project, env = cluster
    result = subprocess.run(['bash', 'slurm/run.sh', 'root', 'existing-group', *mode], cwd=project,
                            env=env, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    training = [c for c in calls(project) if 'hindsight_grpo.train' in c['args']]
    assert len(training) == 1
    assert training[0]['args'] == ['-u', '-m', 'hindsight_grpo.train', '--arm', 'root',
                                   '--group', 'existing-group', *mode]


def test_successor_preserves_evaluation_only_mode(cluster):
    project, env = cluster
    process = subprocess.Popen(['bash', 'slurm/run.sh', 'root', 'existing-group', '--evaluate-only'],
        cwd=project, env={**env, 'TEST_WAIT_FOR_SIGNAL': '1'}, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        wait_for(project / 'training_ready', 'ready', process)
        process.send_signal(signal.SIGUSR1)
        output = process.communicate(timeout=10)
        assert process.returncode == 75, output
        submitted = [c for c in calls(project) if c['command'] == 'sbatch']
        assert len(submitted) == 1
        assert submitted[0]['args'][-4:] == ['slurm/run.sh', 'root', 'existing-group', '--evaluate-only']
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
