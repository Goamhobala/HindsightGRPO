"""Fail-closed, namespace-isolated MBPP test execution on allocated compute nodes."""
from __future__ import annotations

import ast
import json
import math
import os
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .data import extract_code


@dataclass(frozen=True)
class RewardResult:
    reward: float
    partial: float
    syntax_valid: bool
    error: str | None = None
    timeout: bool = False
    tests_passed: int = 0
    tests_total: int = 0
    execution_seconds: float = 0.0
    test_errors: tuple[str | None, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SandboxUnavailable(RuntimeError):
    pass


REQUIRED_BWRAP_OPTIONS = (
    "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent",
    "--new-session", "--cap-drop", "--clearenv", "--setenv", "--ro-bind",
    "--proc", "--dev", "--size", "--tmpfs", "--chdir", "--remount-ro",
)


def find_bubblewrap() -> str | None:
    """Prefer an explicit binary, then the scratch-local build, then the site tool."""
    override = os.environ.get("HINDSIGHT_BWRAP")
    if override:
        return str(Path(override).expanduser().resolve()) if "/" in override else (shutil.which(override) or override)
    local = Path(__file__).resolve().parents[2] / "runtime" / "bin" / "bwrap"
    return str(local) if local.is_file() else shutil.which("bwrap")


def check_bubblewrap_capabilities(executable: str | None = None) -> dict[str, Any]:
    """Inspect command support without running any generated code or namespace probe.

    A system binary can exist yet be too old for the required policy. Fail here
    with the exact missing features, before creating a candidate worker.
    """
    executable = executable or find_bubblewrap()
    remedy = "Run .venv/bin/python scripts/install_bwrap.py on the login node to build the pinned scratch-local binary."
    if platform.system() != "Linux" or not executable:
        raise SandboxUnavailable(f"Linux bubblewrap is required; host execution is prohibited. {remedy}")
    environment = {"PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    try:
        help_result = subprocess.run([executable, "--help"], capture_output=True, text=True,
                                     errors="replace", timeout=10, check=False, env=environment)
        version_result = subprocess.run([executable, "--version"], capture_output=True, text=True,
                                        errors="replace", timeout=10, check=False, env=environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise SandboxUnavailable(f"Cannot inspect bubblewrap at {executable}: {error}. {remedy}") from error
    help_text = help_result.stdout + help_result.stderr
    version = (version_result.stdout + version_result.stderr).strip()[:256] or "unknown version"
    if help_result.returncode:
        raise SandboxUnavailable(f"Bubblewrap help failed at {executable} ({version}): {help_text[:1024]}. {remedy}")
    missing = [option for option in REQUIRED_BWRAP_OPTIONS
               if not re.search(r"(?<![\w-])" + re.escape(option) + r"(?![\w-])", help_text)]
    if missing:
        raise SandboxUnavailable(f"Bubblewrap at {executable} ({version}) is missing required options: "
                                 f"{', '.join(missing)}. {remedy} Sandbox protections remain required.")
    return {"binary": executable, "version": version, "required_options": list(REQUIRED_BWRAP_OPTIONS)}


class RewardRunner:
    """Execute only inside Linux bubblewrap; never fall back to host execution.

    A user namespace with no network, separate PID/IPC/UTS namespaces, a small
    tmpfs, read-only runtime mounts, and process resource limits surrounds the
    worker. No home, scratch, model cache, credentials, or job directory is bound.
    """
    def __init__(self, timeout_seconds: float = 5.0, memory_mb: int = 512,
                 require_slurm: bool = True, temporary_dir: str | Path | None = None):
        self.timeout_seconds = float(timeout_seconds)
        self.memory_mb = int(memory_mb)
        if self.timeout_seconds <= 0 or self.memory_mb < 64:
            raise ValueError("Reward sandbox needs a positive timeout and at least 64 MiB")
        self.require_slurm = require_slurm
        self.temporary_dir = Path(temporary_dir) if temporary_dir else None
        self._verified = False
        self.bwrap = find_bubblewrap()
        self._capabilities: dict[str, Any] | None = None
        self.python = Path(getattr(sys, "_base_executable", sys.executable)).resolve()

    def _limit_child(self) -> None:
        resource.setrlimit(resource.RLIMIT_CPU,
                           (math.ceil(self.timeout_seconds), math.ceil(self.timeout_seconds) + 1))
        memory = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (65536, 65536))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.umask(0o077)

    def _command(self, input_path: Path) -> list[str]:
        if platform.system() != "Linux" or self.bwrap is None:
            raise SandboxUnavailable("Linux bubblewrap is required; host execution is prohibited")
        command = [self.bwrap, "--unshare-all", "--unshare-user", "--disable-userns",
                   "--die-with-parent", "--new-session",
                   "--cap-drop", "ALL", "--clearenv", "--setenv", "HOME", "/tmp",
                   "--setenv", "PATH", "/usr/bin:/bin", "--setenv", "PYTHONHASHSEED", "0"]
        # Runtime mounts intentionally omit /etc, /home, and /scratch. Python's
        # base prefix may be a uv-managed installation on scratch: bind that
        # installation alone, never the containing scratch or project tree.
        mounted: list[Path] = []
        for path in [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"),
                     Path(sys.base_prefix).resolve()]:
            if not path.exists() or any(path == p or p in path.parents for p in mounted):
                continue
            if path == Path("/"):
                raise SandboxUnavailable("Refusing to expose root as a Python runtime mount")
            command.extend(["--ro-bind", str(path), str(path)])
            mounted.append(path)
        if not any(self.python == p or p in self.python.parents for p in mounted):
            raise SandboxUnavailable(f"Python interpreter is outside approved runtime mounts: {self.python}")
        worker = Path(__file__).with_name("_reward_worker.py").resolve()
        command.extend(["--proc", "/proc", "--dev", "/dev", "--size", str(16 * 1024 * 1024),
                        "--tmpfs", "/tmp", "--chdir", "/tmp",
                        "--ro-bind", str(worker), "/worker.py",
                        "--ro-bind", str(input_path), "/input.json",
                        "--remount-ro", "/dev", "--remount-ro", "/proc", "--remount-ro", "/",
                        str(self.python), "-I", "-S", "-B", "/worker.py", "/input.json"])
        return command

    def check_available(self) -> dict[str, Any]:
        """Exercise the actual namespace/runtime setup using a fixed harmless task."""
        self._capabilities = check_bubblewrap_capabilities(self.bwrap)
        if self.temporary_dir:
            self.temporary_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hindsight-sandbox-probe-", dir=self.temporary_dir) as name:
            path = Path(name) / "input.json"
            path.write_text(json.dumps({"code": "def fixed_probe():\n    return 7\n",
                                        "tests": ["assert fixed_probe() == 7"], "setup": ""}))
            result = self._run(path)
        if result.get("tests_passed") != 1:
            raise SandboxUnavailable(f"Bubblewrap preflight failed: {result}")
        self._verified = True
        return {"backend": "bubblewrap", "network": "unshared", "memory_mb": self.memory_mb,
                "wall_seconds": self.timeout_seconds, "temporary_bytes": 16 * 1024 * 1024,
                **self._capabilities}

    def _run(self, input_path: Path) -> dict[str, Any]:
        command = self._command(input_path)
        # Regular files bound output size; capture_output pipes do not honour
        # RLIMIT_FSIZE and could consume unbounded memory before timeout.
        with tempfile.TemporaryFile(mode="w+b") as output, tempfile.TemporaryFile(mode="w+b") as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=errors,
                                       env={"PATH": "/usr/bin:/bin"}, start_new_session=True,
                                       preexec_fn=self._limit_child)
            try:
                returncode = process.wait(timeout=self.timeout_seconds + 1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                return {"error": "wall_timeout", "timeout": True, "tests_passed": 0}
            output.seek(0)
            raw = output.read(65536)
            errors.seek(0)
            stderr = errors.read(4096).decode(errors="replace")
            if returncode != 0:
                if stderr.lstrip().startswith("bwrap:"):
                    raise SandboxUnavailable(f"Bubblewrap infrastructure failed: {stderr}")
                timeout = returncode in (-signal.SIGXCPU, -signal.SIGKILL, 128 + signal.SIGXCPU)
                return {"error": f"sandbox_exit_{returncode}: {stderr}", "timeout": timeout,
                        "tests_passed": 0}
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return {"error": "invalid_worker_result", "timeout": False, "tests_passed": 0}

    def evaluate(self, completion: str, row: Mapping[str, Any]) -> RewardResult:
        if self.require_slurm and not os.environ.get("SLURM_JOB_ID"):
            raise SandboxUnavailable("Generated code rewards require a SLURM compute allocation")
        if not self._verified:
            self.check_available()
        started = time.monotonic()
        code = extract_code(completion)
        tests = list(row.get("test_list", []))
        if not tests:
            raise ValueError("MBPP task has no required tests")
        try:
            ast.parse(code)
            syntax = bool(code.strip())
        except (SyntaxError, ValueError, MemoryError, RecursionError):
            syntax = False
        if not syntax:
            return RewardResult(0.0, 0.0, False, "syntax_error", False, 0, len(tests),
                                time.monotonic() - started)
        if self.temporary_dir:
            self.temporary_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="hindsight-reward-", dir=self.temporary_dir) as name:
            path = Path(name) / "input.json"
            path.write_text(json.dumps({"code": code, "setup": row.get("test_setup_code", ""),
                                        "tests": tests}))
            result = self._run(path)
        passed = int(result.get("tests_passed", 0))
        if not 0 <= passed <= len(tests):
            raise RuntimeError("Reward worker returned invalid test count")
        timeout = bool(result.get("timeout", False))
        error = result.get("error")
        errors = tuple(result.get("test_errors", ()))
        return RewardResult(float(passed == len(tests) and not error and not timeout),
                            passed / len(tests), syntax, error, timeout, passed, len(tests),
                            time.monotonic() - started, errors)
