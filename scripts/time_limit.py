"""Respect the user's 48-hour ceiling, stricter scheduler limits, and shorter overrides."""
from pathlib import Path
import os
import re

MAX_JOB_SECONDS = 48 * 60 * 60


def seconds(value):
    """Parse sbatch minutes[:seconds], hours:minutes:seconds, or days-hours[:minutes[:seconds]]."""
    if not value or value.upper() in {"UNLIMITED", "INFINITE", "N/A", "NONE"}:
        return None
    has_day = "-" in value
    day, clock = value.split("-", 1) if has_day else ("0", value)
    if not day.isdigit() or not re.fullmatch(r"\d+(?::\d+){0,2}", clock):
        raise ValueError(f"Invalid SLURM duration: {value!r}")
    parts = list(map(int, clock.split(":")))
    if has_day:
        # Once a day prefix appears, the first component always means hours.
        parts += [0] * (3 - len(parts))
    elif len(parts) == 1:
        return parts[0] * 60
    else:
        # Without a day prefix, two components mean minutes:seconds.
        parts = [0] * (3 - len(parts)) + parts
    return int(day) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def main():
    partition = Path("runtime/preflight/partition-detail.txt").read_text()
    match = re.search(r"\bMaxTime=(\S+)", partition)
    qos = Path("runtime/preflight/qos.txt").read_text().strip().splitlines()
    limits = [MAX_JOB_SECONDS, seconds(match.group(1)) if match else None]
    for line in qos:
        fields = line.split("|")
        if fields[0] == "l40sfree":
            limits.append(seconds(fields[1]))
    if os.environ.get("HINDSIGHT_TIME_LIMIT"):
        limits.append(seconds(os.environ["HINDSIGHT_TIME_LIMIT"]))
    finite = [value for value in limits if value is not None]
    result = min(finite)
    if result <= 600:
        raise SystemExit("The allocation limit is too short for the 300-second checkpoint warning.")
    days, remainder = divmod(result, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    print(f"{days}-{hours:02}:{minutes:02}:{secs:02}")


if __name__ == "__main__":
    main()
