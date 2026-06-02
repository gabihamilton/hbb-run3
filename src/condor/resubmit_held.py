#!/usr/bin/env python3
"""
resubmit_held.py
----------------
Find held condor jobs, diagnose the hold reason, and resubmit memory-held
jobs with a higher RequestMemory value.

The JDL files are expected to live at:
    condor/<tag>/<year>_<subsample>_<jobid>.jdl

and the condor UserLog attribute points to:
    condor/<tag>/logs/<year>_<subsample>_<jobid>.log

so the JDL path is inferred by stripping /logs/ and swapping the extension.

Usage
-----
    # Dry run — see what would happen
    python src/condor/resubmit_held.py --tag 26May18 --dry-run

    # Actually resubmit memory-held jobs with 2× memory
    python src/condor/resubmit_held.py --tag 26May18

    # Use 3× memory instead
    python src/condor/resubmit_held.py --tag 26May18 --factor 3

    # Resubmit ALL held jobs regardless of reason
    python src/condor/resubmit_held.py --tag 26May18 --all-held
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

# Hold reasons that indicate a memory problem
MEMORY_KEYWORDS = ["memory", "ram", "mem limit", "oom", "exceed", "over memory"]


# ---------------------------------------------------------------------------
# Condor queries
# ---------------------------------------------------------------------------

def get_held_jobs() -> list[dict]:
    """
    Return a list of held jobs as dicts with keys:
        job_id, hold_reason, log_path
    """
    # Use -af (autoformat) to get ClassAd values.
    # UserLog gives us the .log file path → we derive the .jdl from it.
    result = subprocess.run(
        [
            "condor_q", "-held",
            "-af", "ClusterId", "ProcId", "HoldReason", "UserLog",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[ERROR] condor_q failed:\n{result.stderr}")
        return []

    jobs = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        # Split into at most 4 parts: ClusterId ProcId <HoldReason possibly with spaces> UserLog
        # HoldReason may contain spaces so we can't just split — use a known-width approach.
        # Safer: split first 2 tokens, last token, middle is HoldReason.
        parts = line.split()
        if len(parts) < 4:
            continue
        cluster  = parts[0]
        proc     = parts[1]
        log_path = parts[-1]   # UserLog is last
        reason   = " ".join(parts[2:-1])

        jobs.append({
            "job_id":     f"{cluster}.{proc}",
            "hold_reason": reason,
            "log_path":    log_path,
        })
    return jobs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_memory_hold(reason: str) -> bool:
    rl = reason.lower()
    return any(kw in rl for kw in MEMORY_KEYWORDS)


def jdl_from_log(log_path: str) -> Path | None:
    """Derive the .jdl path from the UserLog path.

    Log lives at: condor/<tag>/logs/<prefix>.log
    JDL lives at: condor/<tag>/<prefix>.jdl
    """
    p = Path(log_path)
    # Remove the 'logs/' component and swap extension
    jdl = p.parent.parent / p.with_suffix(".jdl").name
    return jdl if jdl.exists() else None


def bump_memory(jdl: Path, factor: float) -> tuple[int, int] | None:
    """Multiply request_memory in the JDL by factor. Returns (old, new) or None."""
    text = jdl.read_text()
    m = re.search(r"(request_memory\s*=\s*)(\d+)", text, re.IGNORECASE)
    if not m:
        return None
    old_mem = int(m.group(2))
    new_mem = int(old_mem * factor)
    new_text = re.sub(
        r"(request_memory\s*=\s*)\d+",
        f"\\g<1>{new_mem}",
        text, flags=re.IGNORECASE,
    )
    jdl.write_text(new_text)
    return old_mem, new_mem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resubmit memory-held condor jobs with more RAM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag", required=True,
                        help="Condor submission tag, e.g. 26May18")
    parser.add_argument("--factor", type=float, default=2.0,
                        help="Memory multiplier (default: 2 → double the memory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would happen without removing/resubmitting")
    parser.add_argument("--all-held", action="store_true",
                        help="Resubmit ALL held jobs, not just memory-related ones")
    args = parser.parse_args()

    jobs = get_held_jobs()
    if not jobs:
        print("No held jobs found.")
        return

    print(f"Found {len(jobs)} held job(s).\n")

    # Bucket by reason type
    mem_jobs   = [j for j in jobs if is_memory_hold(j["hold_reason"])]
    other_jobs = [j for j in jobs if not is_memory_hold(j["hold_reason"])]

    print(f"  Memory-related : {len(mem_jobs)}")
    print(f"  Other          : {len(other_jobs)}")

    if other_jobs:
        print("\nNon-memory holds (skipping unless --all-held):")
        for j in other_jobs:
            print(f"  {j['job_id']:>12}  {j['hold_reason'][:90]}")

    to_process = jobs if args.all_held else mem_jobs
    if not to_process:
        print("\nNothing to resubmit.")
        return

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Processing {len(to_process)} job(s) "
          f"with {args.factor}× memory:\n")

    n_ok = n_missing_jdl = n_err = 0

    for job in to_process:
        jdl = jdl_from_log(job["log_path"])
        job_id = job["job_id"]

        print(f"  {job_id}  |  {job['hold_reason'][:70]}")

        if jdl is None:
            print(f"    [WARN] JDL not found (expected near {job['log_path']})")
            n_missing_jdl += 1
            continue

        print(f"    JDL: {jdl}")

        if args.dry_run:
            # Just show current memory
            text = jdl.read_text()
            m = re.search(r"request_memory\s*=\s*(\d+)", text, re.IGNORECASE)
            cur = int(m.group(1)) if m else "?"
            new = int(cur * args.factor) if isinstance(cur, int) else "?"
            print(f"    [DRY RUN] Would bump memory {cur} MB → {new} MB and resubmit")
            continue

        # 1. Bump memory in JDL
        result = bump_memory(jdl, args.factor)
        if result is None:
            print(f"    [WARN] No request_memory line found in {jdl}")
            n_err += 1
            continue
        old_mem, new_mem = result
        print(f"    Memory: {old_mem} MB → {new_mem} MB")

        # 2. Remove old held job
        rm = subprocess.run(["condor_rm", job_id], capture_output=True, text=True)
        if rm.returncode != 0:
            print(f"    [ERROR] condor_rm failed: {rm.stderr.strip()}")
            n_err += 1
            continue
        print(f"    Removed {job_id}")

        # 3. Resubmit
        sub = subprocess.run(["condor_submit", str(jdl)],
                              capture_output=True, text=True)
        if sub.returncode == 0:
            # Extract new cluster ID from output
            new_id = re.search(r"cluster (\d+)", sub.stdout)
            print(f"    Resubmitted → cluster {new_id.group(1) if new_id else '?'}")
            n_ok += 1
        else:
            print(f"    [ERROR] condor_submit failed: {sub.stderr.strip()}")
            n_err += 1

    if not args.dry_run:
        print(f"\nSummary: {n_ok} resubmitted | {n_missing_jdl} JDL not found | {n_err} errors")


if __name__ == "__main__":
    main()
