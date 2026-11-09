"""Keep user-wide process accounting out of bubblewrap namespace startup.

All limit-setting calls are mocked. These tests do not alter the test runner's
resource limits, create namespaces, or execute candidate code.
"""
from unittest.mock import patch

import pytest

from hindsight_grpo import _reward_worker as worker
from hindsight_grpo.rewards import RewardRunner


def test_launcher_keeps_account_process_limit_for_namespace_creation():
    runner = RewardRunner(timeout_seconds=5, memory_mb=512, require_slurm=False)
    with patch("hindsight_grpo.rewards.resource.setrlimit") as setlimit, \
            patch("hindsight_grpo.rewards.os.umask"):
        runner._limit_child()
    resources = {call.args[0]: call.args[1] for call in setlimit.call_args_list}
    # NPROC counts the real UID's existing Linux threads, including unrelated
    # Torch/W&B processes. The launcher must retain its inherited account limit.
    assert worker.resource.RLIMIT_NPROC not in resources
    assert resources[worker.resource.RLIMIT_CPU] == (5, 6)
    assert resources[worker.resource.RLIMIT_AS] == (512 * 1024 * 1024,) * 2
    assert resources[worker.resource.RLIMIT_FSIZE] == (65536, 65536)
    assert resources[worker.resource.RLIMIT_NOFILE] == (64, 64)
    assert resources[worker.resource.RLIMIT_CORE] == (0, 0)


@pytest.mark.parametrize("inherited,expected", [
    ((worker.resource.RLIM_INFINITY, worker.resource.RLIM_INFINITY), 32),
    ((1024, 4096), 32),
    ((16, 64), 16),
    ((8, 8), 8),
    ((worker.resource.RLIM_INFINITY, 12), 12),
    ((0, 32), 0),
])
def test_worker_process_limit_never_raises_inherited_limits(inherited, expected):
    with patch.object(worker.resource, "getrlimit", return_value=inherited) as getlimit, \
            patch.object(worker.resource, "setrlimit") as setlimit:
        result = worker.limit_processes()
    getlimit.assert_called_once_with(worker.resource.RLIMIT_NPROC)
    setlimit.assert_called_once_with(worker.resource.RLIMIT_NPROC, (expected, expected))
    assert result == expected


def test_worker_installs_process_limit_before_reading_candidate_payload():
    events = []

    class StopBeforePayload(Exception):
        pass

    def install():
        events.append("limit")
        return 32

    def stop(*args, **kwargs):
        events.append("payload")
        raise StopBeforePayload()

    with patch.object(worker, "limit_processes", side_effect=install), \
            patch.object(worker.sys, "argv", ["worker.py", "/input.json"]), \
            patch("builtins.open", side_effect=stop):
        with pytest.raises(StopBeforePayload):
            worker.main()
    assert events == ["limit", "payload"]


def test_worker_limit_failure_does_not_read_or_execute_candidate():
    with patch.object(worker, "limit_processes", side_effect=OSError("limit rejected")), \
            patch("builtins.open") as read_payload:
        with pytest.raises(OSError, match="limit rejected"):
            worker.main()
    read_payload.assert_not_called()
