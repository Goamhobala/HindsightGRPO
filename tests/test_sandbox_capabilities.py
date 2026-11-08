import hashlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hindsight_grpo.rewards import (
    REQUIRED_BWRAP_OPTIONS, RewardRunner, SandboxUnavailable,
    check_bubblewrap_capabilities, find_bubblewrap,
)


def results(help_text):
    return [SimpleNamespace(returncode=0, stdout=help_text, stderr=""),
            SimpleNamespace(returncode=0, stdout="bubblewrap 0.13.0\n", stderr="")]


def test_old_bubblewrap_reports_exact_missing_options_before_worker(tmp_path):
    text = "\n".join(option for option in REQUIRED_BWRAP_OPTIONS if option != "--disable-userns")
    runner = RewardRunner(require_slurm=False, temporary_dir=tmp_path)
    runner.bwrap = "/usr/bin/bwrap"
    with patch("hindsight_grpo.rewards.platform.system", return_value="Linux"), \
            patch("hindsight_grpo.rewards.subprocess.run", side_effect=results(text)), \
            patch.object(runner, "_run") as worker:
        with pytest.raises(SandboxUnavailable, match="missing required options: --disable-userns") as error:
            runner.check_available()
    assert "scripts/install_bwrap.py" in str(error.value)
    worker.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_capability_check_returns_binary_version_without_executing_code(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "private-test-value")
    with patch("hindsight_grpo.rewards.platform.system", return_value="Linux"), \
            patch("hindsight_grpo.rewards.subprocess.run", side_effect=results("\n".join(REQUIRED_BWRAP_OPTIONS))) as run:
        report = check_bubblewrap_capabilities("/scratch/runtime/bin/bwrap")
    assert report["version"] == "bubblewrap 0.13.0"
    assert report["binary"] == "/scratch/runtime/bin/bwrap"
    assert run.call_count == 2
    assert [call.args[0][-1] for call in run.call_args_list] == ["--help", "--version"]
    assert all("WANDB_API_KEY" not in call.kwargs["env"] for call in run.call_args_list)


def test_required_option_is_matched_as_a_complete_option():
    text = "\n".join(REQUIRED_BWRAP_OPTIONS).replace("--disable-userns", "--disable-userns-extra")
    with patch("hindsight_grpo.rewards.platform.system", return_value="Linux"), \
            patch("hindsight_grpo.rewards.subprocess.run", side_effect=results(text)):
        with pytest.raises(SandboxUnavailable, match="--disable-userns"):
            check_bubblewrap_capabilities("/usr/bin/bwrap")


def test_explicit_binary_override_is_respected(monkeypatch, tmp_path):
    binary = tmp_path / "bwrap"
    monkeypatch.setenv("HINDSIGHT_BWRAP", str(binary))
    assert find_bubblewrap() == str(binary)


def test_scratch_local_binary_precedes_old_system_binary(monkeypatch):
    monkeypatch.delenv("HINDSIGHT_BWRAP", raising=False)
    expected = str(Path(__file__).resolve().parents[1] / "runtime/bin/bwrap")
    with patch("hindsight_grpo.rewards.Path.is_file", return_value=True), \
            patch("hindsight_grpo.rewards.shutil.which", return_value="/usr/bin/bwrap") as system:
        assert find_bubblewrap() == expected
    system.assert_not_called()


def test_missing_executable_has_install_remedy():
    with patch("hindsight_grpo.rewards.platform.system", return_value="Linux"), \
            patch("hindsight_grpo.rewards.subprocess.run", side_effect=FileNotFoundError("missing")):
        with pytest.raises(SandboxUnavailable, match="scripts/install_bwrap.py"):
            check_bubblewrap_capabilities("/not-installed/bwrap")


SPEC = importlib.util.spec_from_file_location("install_bwrap", Path(__file__).parents[1] / "scripts/install_bwrap.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


def test_bad_archive_digest_is_rejected_without_installing(tmp_path):
    destination = tmp_path / "source.tar.xz"
    with patch.object(installer.urllib.request, "urlopen", return_value=io.BytesIO(b"wrong-source")):
        with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
            installer.verified_download("https://example.invalid/source", destination, "0" * 64)
    assert not destination.exists()
    assert not list(tmp_path.iterdir())


def test_verified_archive_is_reused_without_network(tmp_path):
    archive = tmp_path / "source.tar.xz"
    archive.write_bytes(b"verified-source")
    with patch.object(installer.urllib.request, "urlopen") as download:
        assert installer.verified_download("https://example.invalid/source", archive,
            hashlib.sha256(archive.read_bytes()).hexdigest()) == archive
    download.assert_not_called()


def test_source_archive_rejects_parent_traversal_before_extraction(tmp_path):
    archive = tmp_path / "bad.tar.xz"
    with tarfile.open(archive, "w:xz") as output:
        member = tarfile.TarInfo("../outside")
        member.size = 1
        output.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(RuntimeError, match="Unexpected source archive path"):
        installer.extract_source(archive, tmp_path / "work", "bubblewrap-0.13.0")
    assert not (tmp_path / "outside").exists()


def test_installer_builds_static_libcap_and_keeps_user_namespace_policy(tmp_path):
    runtime = tmp_path / "runtime"
    programs = {name: f"/tools/{name}" for name in ("uv", "cc", "make", "ar", "ranlib", "pkg-config")}
    calls = []
    def source(archive, working, name):
        path = working / name
        (path / "libcap").mkdir(parents=True)
        return path
    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0].endswith("/meson") and command[1] == "install":
            binary = runtime / "opt/bubblewrap-0.13.0/bin/bwrap"
            binary.parent.mkdir(parents=True)
            binary.write_bytes(b"synthetic-compiled-binary")
    report = {"binary": str(runtime / "opt/bubblewrap-0.13.0/bin/bwrap"),
              "version": "bubblewrap 0.13.0", "required_options": list(REQUIRED_BWRAP_OPTIONS)}
    with patch.object(installer.platform, "system", return_value="Linux"), \
            patch.object(installer, "build_prerequisites", return_value=programs), \
            patch.object(installer, "verified_download", return_value=tmp_path / "source.tar.xz"), \
            patch.object(installer, "extract_source", side_effect=source), \
            patch.object(installer.subprocess, "run", side_effect=run), \
            patch.object(installer, "check_bubblewrap_capabilities", return_value=report):
        result = installer.install(runtime)
    assert (runtime / "bin/bwrap").is_file()
    assert result["libcap_linkage"] == "static"
    assert not result["setuid"]
    assert (runtime / "metadata/bubblewrap.json").is_file()
    venv = next(command for command, _ in calls if command[:2] == [programs["uv"], "venv"])
    assert "--allow-existing" in venv
    make_calls = [command for command, _ in calls if command[0] == programs["make"]]
    assert all("SHARED=no" in command and "PTHREADS=no" in command for command in make_calls)
    assert make_calls[-1][-1] == "install-static-cap"
    setup, environment = next((command, kwargs["env"]) for command, kwargs in calls
                              if command[0].endswith("/meson") and command[1] == "setup")
    assert "-Dprefer_static=true" in setup
    assert environment["PKG_CONFIG_PATH"].startswith(str(runtime / "opt/libcap-2.75/lib/pkgconfig"))
    assert not any(command[0] in ("git", "sudo") for command, _ in calls)
