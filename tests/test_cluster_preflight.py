"""Scheduler checks use fake evidence and never require or submit SLURM jobs."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("hindsight_cluster_preflight", Path(__file__).parents[1] / "scripts/cluster_preflight.py")
cluster_preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cluster_preflight)


def evidence(tmp_path, association="l40sfree|normal,l40sfree||\n", extra="", state="UP", gres="gpu:l40s:4"):
    (tmp_path / "partition-detail.txt").write_text(
        f"PartitionName=l40s State={state} AllowAccounts=ALL AllowQos=ALL {extra}\n")
    (tmp_path / "association.txt").write_text(association)
    (tmp_path / "qos.txt").write_text("l40sfree|1-00:00:00|\n")
    (tmp_path / "partition.txt").write_text(f"l40s up {gres}\n")


def test_effective_permission_accepts_all_partition_association(tmp_path, capsys):
    evidence(tmp_path)
    cluster_preflight.main(tmp_path)
    assert "effective l40sfree" in capsys.readouterr().out


@pytest.mark.parametrize("association", ["l40sfree|normal||\n", "l40sfree|l40sfree|other|\n", "other|l40sfree||\n", "l40sfree|l40sfree,-l40sfree||\n"])
def test_wrong_association_qos_or_partition_rejected(tmp_path, association):
    evidence(tmp_path, association=association)
    with pytest.raises(SystemExit, match="account association"):
        cluster_preflight.main(tmp_path)


@pytest.mark.parametrize("field", ["AllowAccounts=other", "AllowQos=normal", "DenyAccounts=l40sfree", "DenyQos=l40sfree"])
def test_partition_allow_and_deny_rules_enforced(tmp_path, field):
    evidence(tmp_path)
    path = tmp_path / "partition-detail.txt"
    text = path.read_text()
    key = field.split("=")[0]
    if key.startswith("Allow"):
        text = text.replace(key + "=ALL", field)
    else:
        text += field + "\n"
    path.write_text(text)
    with pytest.raises(SystemExit, match="does not permit|explicitly denies"):
        cluster_preflight.main(tmp_path)


def test_non_gpu_or_inactive_partition_rejected(tmp_path):
    evidence(tmp_path, state="DOWN")
    with pytest.raises(SystemExit, match="active"):
        cluster_preflight.main(tmp_path)
    evidence(tmp_path, gres="(null)")
    with pytest.raises(SystemExit, match="GPU resources"):
        cluster_preflight.main(tmp_path)
