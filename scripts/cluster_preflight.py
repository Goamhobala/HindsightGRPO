"""Validate effective account, QoS, partition and GPU scheduler evidence."""
from pathlib import Path
import re


def _tokens(value):
    return {item.strip() for item in value.split(",") if item.strip()}


def main(evidence=Path("runtime/preflight")):
    partition = (evidence / "partition-detail.txt").read_text()
    association = (evidence / "association.txt").read_text()
    qos = (evidence / "qos.txt").read_text()
    gres = (evidence / "partition.txt").read_text()
    if not re.search(r"\bPartitionName=l40s\b", partition) or not re.search(r"\bState=UP\b", partition):
        raise SystemExit("Expected active l40s partition; inspect runtime/preflight/partition-detail.txt")
    if "l40s" not in gres.lower() or "gpu:" not in gres.lower():
        raise SystemExit("Expected GPU resources in l40s partition")
    valid_association = False
    for line in association.splitlines():
        fields = [field.strip() for field in line.split("|")]
        if len(fields) < 3 or fields[0] != "l40sfree" or fields[2] not in {"", "l40s"}:
            continue
        available_qos = _tokens(fields[1])
        if available_qos & {"l40sfree", "+l40sfree", "ALL"} and "-l40sfree" not in available_qos:
            valid_association = True
    if not valid_association:
        raise SystemExit("No l40sfree account association granting QoS l40sfree on partition l40s")
    if not any(line.startswith("l40sfree|") for line in qos.splitlines()):
        raise SystemExit("QoS l40sfree not available")
    for key, expected in [("Accounts", "l40sfree"), ("Qos", "l40sfree")]:
        allow = re.search(rf"\bAllow{key}=(\S+)", partition, flags=re.IGNORECASE)
        deny = re.search(rf"\bDeny{key}=(\S+)", partition, flags=re.IGNORECASE)
        if allow and allow[1].upper() != "ALL" and expected not in _tokens(allow[1]):
            raise SystemExit(f"l40s partition does not permit {key}={expected}")
        if deny and (deny[1].upper() == "ALL" or expected in _tokens(deny[1])):
            raise SystemExit(f"l40s partition explicitly denies {key}={expected}")
    print("Verified l40s GPU partition and effective l40sfree account/QoS permission.")


if __name__ == "__main__":
    main()
