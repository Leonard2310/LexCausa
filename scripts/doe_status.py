#!/usr/bin/env python
"""Progress of a running multi-DoE campaign: completed runs, rate, ETA.

Counts the raw per-run JSONs (written as each run finishes) — metrics.csv only
appears when a shard ends, so it is useless while the campaign is in flight.

    poetry run python scripts/doe_status.py
    poetry run python scripts/doe_status.py "experiments/multi_doe/runs/or_doe600_pcB_*_g*"
"""
import datetime
import glob
import os
import subprocess
import sys
import time

pattern = (
    sys.argv[1] if len(sys.argv) > 1 else "experiments/multi_doe/runs/or_doe600_pc*_g*"
)
dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
if not dirs:
    raise SystemExit(f"nessuna directory per: {pattern}")

per_shard = 75
files = [f for d in dirs for f in glob.glob(f"{d}/runs/*.json")]
n, target = len(files), per_shard * len(dirs)

# run_matrix.csv is written once at launch -> reliable start time
starts = [
    os.path.getmtime(f"{d}/run_matrix.csv")
    for d in dirs
    if os.path.exists(f"{d}/run_matrix.csv")
]
start = min(starts) if starts else min(os.path.getctime(d) for d in dirs)
el_h = max((time.time() - start) / 3600, 1e-9)
rate = n / el_h
eta_h = (target - n) / rate if rate > 0 else float("inf")

bar = int(n / target * 40)
print(f"  {datetime.datetime.now():%H:%M:%S}   {n}/{target} run  ({n/target*100:.1f}%)")
print(f"  [{'#'*bar}{'.'*(40-bar)}]")
for d in dirs:
    print(
        f"    {d.split('_')[-1]}: {len(glob.glob(f'{d}/runs/*.json')):3d}/{per_shard}"
    )
if eta_h == float("inf"):
    print(f"  ritmo -- | trascorso {el_h:.2f} h | ETA n/d")
else:
    end = datetime.datetime.now() + datetime.timedelta(hours=eta_h)
    print(
        f"  ritmo {rate:.1f} run/h | trascorso {el_h:.2f} h | ETA {eta_h:.1f} h -> {end:%H:%M di %d/%m}"
    )

alive = len(
    subprocess.run(
        ["pgrep", "-f", "run_multi_doe.py"], capture_output=True, text=True
    ).stdout.split()
)


def swap_used() -> str:
    """Swap in use — macOS via sysctl, Linux via /proc/meminfo."""
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True
        ).stdout
        return out.split("used = ")[1].split()[0] if "used = " in out else "n/d"
    try:
        info = dict(
            (k.rstrip(":"), int(v))
            for k, v, *_ in (ln.split() for ln in open("/proc/meminfo"))
        )
        used_kb = info.get("SwapTotal", 0) - info.get("SwapFree", 0)
        return f"{used_kb / 1024:.0f}M"
    except Exception:
        return "n/d"


print(f"  shard vivi {alive} | swap {swap_used()}")
