"""Exercise the real wrapper's traps using a harmless child and fake sbatch."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def wait_for(path, value, process):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if path.exists() and value in path.read_text():
            return
        if process.poll() is not None:
            raise AssertionError(process.communicate())
        time.sleep(.01)
    raise AssertionError(f'Timed out waiting for {value}')


def test_term_allows_child_cleanup_and_suppresses_later_successor_signal(tmp_path):
    home = tmp_path / 'home/user'
    project = tmp_path / 'scratch/yhxjin001/HindsightGRPO'
    home.mkdir(parents=True)
    (project / 'scripts').mkdir(parents=True)
    (project / '.venv/bin').mkdir(parents=True)
    (project / 'scripts/env.sh').write_text('export HINDSIGHT_TIME_LIMIT=2-00:00:00\n')
    child = project / '.venv/bin/python'
    child.write_text(f'#!{sys.executable}\n' + '''
from pathlib import Path
import signal
import sys
import time
events = Path('events')
def stop(number, frame):
    with events.open('a') as stream:
        stream.write('term received\\n')
    time.sleep(.6)
    with events.open('a') as stream:
        stream.write('cleanup completed\\n')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
with events.open('a') as stream:
    stream.write('child started\\n')
while True:
    time.sleep(.05)
''')
    child.chmod(0o755)
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    sbatch = fake_bin / 'sbatch'
    sbatch.write_text('#!/bin/sh\nprintf "unexpected sbatch\\n" >> events\nprintf "999\\n"\n')
    sbatch.chmod(0o755)
    wrapper = Path(__file__).resolve().parents[1] / 'slurm/run.sh'
    env = {**os.environ, 'HOME': str(home), 'WANDB_API_KEY': 'synthetic-not-real',
           'SLURM_JOB_ID': '123', 'PATH': str(fake_bin) + os.pathsep + os.environ['PATH']}
    process = subprocess.Popen(['bash', str(wrapper), 'root', 'group'], cwd=tmp_path,
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    events = project / 'events'
    try:
        wait_for(events, 'child started', process)
        process.send_signal(signal.SIGTERM)
        wait_for(events, 'term received', process)
        process.send_signal(signal.SIGUSR1)
        output = process.communicate(timeout=10)
        assert process.returncode == 75, output
        recorded = events.read_text()
        assert 'cleanup completed' in recorded
        assert recorded.count('child started') == 1
        assert 'unexpected sbatch' not in recorded
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
