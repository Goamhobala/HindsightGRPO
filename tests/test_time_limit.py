"""SLURM time syntax and checkpoint-warning boundaries, using recorded evidence."""
import importlib.util
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("hindsight_time_limit", Path(__file__).parents[1] / "scripts/time_limit.py")
time_limit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(time_limit)


@pytest.mark.parametrize("value, expected", [
    ("5", 300), ("05:30", 330), ("01:05:30", 3930),
    ("1-02:30:04", 95404), ("1-02", 93600), ("1-02:30", 95400),
    ("UNLIMITED", None), ("INFINITE", None), ("N/A", None), ("", None),
])
def test_slurm_time_formats(value, expected):
    assert time_limit.seconds(value) == expected


def write_evidence(tmp_path, partition="2-00:00:00", qos="1-00:00:00"):
    path = tmp_path / "runtime/preflight"
    path.mkdir(parents=True)
    (path / "partition-detail.txt").write_text(f"PartitionName=l40s MaxTime={partition}\n")
    (path / "qos.txt").write_text(f"l40sfree|{qos}|\n")


def test_effective_limit_uses_most_restrictive_evidence(tmp_path, monkeypatch, capsys):
    write_evidence(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HINDSIGHT_TIME_LIMIT", "12:00:00")
    time_limit.main()
    assert capsys.readouterr().out.strip() == "0-12:00:00"


def test_unlimited_evidence_uses_user_48_hour_maximum(tmp_path, monkeypatch, capsys):
    write_evidence(tmp_path, "UNLIMITED", "")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HINDSIGHT_TIME_LIMIT", raising=False)
    time_limit.main()
    assert capsys.readouterr().out.strip() == "2-00:00:00"


@pytest.mark.parametrize("partition,qos,override,expected", [
    ("3-00:00:00", "3-00:00:00", None, "2-00:00:00"),
    ("3-00:00:00", "3-00:00:00", "72:00:00", "2-00:00:00"),
    ("3-00:00:00", "1-00:00:00", None, "1-00:00:00"),
    ("3-00:00:00", "3-00:00:00", "12:00:00", "0-12:00:00"),
])
def test_user_maximum_scheduler_and_override_limits(tmp_path, monkeypatch, capsys,
                                                    partition, qos, override, expected):
    write_evidence(tmp_path, partition, qos)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HINDSIGHT_TIME_LIMIT", raising=False)
    if override:
        monkeypatch.setenv("HINDSIGHT_TIME_LIMIT", override)
    time_limit.main()
    assert capsys.readouterr().out.strip() == expected


def test_too_short_limit_rejected_before_submission(tmp_path, monkeypatch):
    write_evidence(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HINDSIGHT_TIME_LIMIT", "10")
    with pytest.raises(SystemExit, match="too short"):
        time_limit.main()
