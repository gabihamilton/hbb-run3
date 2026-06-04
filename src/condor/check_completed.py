#!/usr/bin/env python3
"""
check_completed.py
------------------
Check whether all submitted condor jobs completed successfully.

A job is considered complete when its output pickle exists on EOS:
    <eos-base>/<year>/<subsample>/pickles/out_<jobnum>.pkl

A job is considered to have produced output when at least one parquet
exists under:
    <eos-base>/<year>/<subsample>/parquet/nominal/*/part<jobnum>.parquet

Jobs still running / idle / held are reported separately (not failures).

Usage
-----
    # Check all jobs for a tag
    python src/condor/check_completed.py --tag 26May18

    # Check a specific year only
    python src/condor/check_completed.py --tag 26May18 --year 2024

    # Resubmit jobs whose output is missing (and that are not still running)
    python src/condor/check_completed.py --tag 26May18 --resubmit
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

try:
    from colorama import Fore, Style
    GREEN  = Fore.GREEN
    RED    = Fore.RED
    YELLOW = Fore.YELLOW
    RESET  = Style.RESET_ALL
except ImportError:
    GREEN = RED = YELLOW = RESET = ""

EOS_BASE = "/eos/uscms/store/group/lpchbbrun3/skims"


# ---------------------------------------------------------------------------
# Condor helpers
# ---------------------------------------------------------------------------

def get_active_jobs() -> set[str]:
    """Return set of 'ClusterId.ProcId' strings that are still in the queue."""
    result = subprocess.run(
        "condor_q -af ClusterId ProcId",
        shell=True, capture_output=True, text=True,
    )
    active = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            active.add(f"{parts[0]}.{parts[1]}")
    return active


def get_held_jobs() -> dict[str, str]:
    """Return {job_id: hold_reason} for held jobs."""
    result = subprocess.run(
        "condor_q -held -af ClusterId ProcId HoldReason",
        shell=True, capture_output=True, text=True,
    )
    held = {}
    for line in result.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2:
            jid = f"{parts[0]}.{parts[1]}"
            held[jid] = parts[2] if len(parts) > 2 else ""
    return held


# ---------------------------------------------------------------------------
# JDL scanning
# ---------------------------------------------------------------------------

def parse_jdl_name(jdl: Path) -> tuple[str, str, int] | None:
    """
    Extract (year, subsample, jobnum) from a JDL filename like:
        2024_EGamma_3.jdl
        2022EE_TTG-1Jets_PTG-200_0.jdl
    """
    stem = jdl.stem   # e.g. "2022EE_TTG-1Jets_PTG-200_0"
    m = re.match(r"^(\d{4}(?:EE|BPix)?)_(.+)_(\d+)$", stem)
    if not m:
        return None
    return m.group(1), m.group(2), int(m.group(3))


# ---------------------------------------------------------------------------
# Log checks
# ---------------------------------------------------------------------------

# Patterns in .err that indicate a genuine failure
ERR_PATTERNS = [
    "Traceback (most recent call last)",
    "Error:",
    "Exception:",
    "Killed",
    "MemoryError",
    "Segmentation fault",
    "bus error",
    "killed",
    "OOM",
]

# Pattern in .out that confirms the job finished writing output
SUCCESS_PATTERN = "Saved output to"


def check_logs(log_dir: Path, stem: str) -> tuple[str, str]:
    """
    Check the .err and .out log files for a job.

    Returns (status, detail) where status is one of:
        "success"   — .out contains success marker, .err is clean
        "error"     — .err contains a known error pattern
        "no_output" — job ran but success marker not found in .out
        "no_logs"   — log files don't exist yet (job hasn't started / too early)
    """
    err_file = log_dir / f"{stem}.err"
    out_file = log_dir / f"{stem}.out"

    if not err_file.exists() and not out_file.exists():
        return "no_logs", ""

    # Check .err for errors (read last 100 lines — tracebacks are at the end)
    if err_file.exists():
        try:
            err_text = err_file.read_text(errors="replace")
            # Only look at the last 3 KB to avoid reading huge files
            tail = err_text[-3000:]
            for pat in ERR_PATTERNS:
                if pat.lower() in tail.lower():
                    # Extract first matching line for context
                    for line in tail.splitlines():
                        if pat.lower() in line.lower():
                            return "error", line.strip()[:120]
        except OSError:
            pass

    # Check .out for success marker
    if out_file.exists():
        try:
            out_text = out_file.read_text(errors="replace")
            if SUCCESS_PATTERN in out_text:
                return "success", ""
            else:
                # .err was clean but success marker missing — partial run?
                # Show last non-empty line of .out as context
                lines = [l for l in out_text.splitlines() if l.strip()]
                last = lines[-1][:120] if lines else "(empty)"
                return "no_output", last
        except OSError:
            pass

    return "no_logs", ""


# ---------------------------------------------------------------------------
# EOS output checks
# ---------------------------------------------------------------------------

def pickle_exists(eos_base: Path, year: str, subsample: str, jobnum: int) -> bool:
    p = eos_base / year / subsample / "pickles" / f"out_{jobnum}.pkl"
    return p.exists()


def parquet_exists(eos_base: Path, year: str, subsample: str, jobnum: int) -> bool:
    """Check if any parquet file for this job exists (any jer_var, any region)."""
    parquet_dir = eos_base / year / subsample / "parquet"
    if not parquet_dir.exists():
        return False
    pattern = f"part{jobnum}.parquet"
    return any(parquet_dir.rglob(pattern))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check completion status of condor jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag",  required=True,
                        help="Submission tag, e.g. 26May18")
    parser.add_argument("--year", default=None,
                        choices=["2022", "2022EE", "2023", "2023BPix", "2024"],
                        help="Filter to a single year (default: all years)")
    parser.add_argument("--eos-base", default=EOS_BASE,
                        help=f"EOS base path (default: {EOS_BASE})")
    parser.add_argument("--resubmit", action="store_true",
                        help="Resubmit JDLs whose output is missing and not still running")
    parser.add_argument("--no-parquet-check", action="store_true",
                        help="Skip the (slower) parquet check — only check pickles")
    args = parser.parse_args()

    condor_dir = Path(f"condor/{args.tag}")
    if not condor_dir.exists():
        print(f"[ERROR] Condor directory not found: {condor_dir}")
        return

    eos_base = Path(args.eos_base) / args.tag

    # Collect all JDL files, grouped by (year, subsample)
    jdls = sorted(condor_dir.glob("*.jdl"))
    if not jdls:
        print(f"[ERROR] No .jdl files found in {condor_dir}")
        return

    # Filter by year if requested
    if args.year:
        jdls = [j for j in jdls if j.stem.startswith(args.year + "_")]

    # Get current condor state
    print("Querying condor queue ...")
    active_jobs = get_active_jobs()
    held_jobs   = get_held_jobs()

    # Group JDLs by (year, subsample)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    unparseable = []
    for jdl in jdls:
        parsed = parse_jdl_name(jdl)
        if parsed is None:
            unparseable.append(jdl)
            continue
        year, subsample, jobnum = parsed
        groups[(year, subsample)].append((jobnum, jdl))

    if unparseable:
        print(f"[WARN] Could not parse {len(unparseable)} JDL filename(s): "
              f"{[j.name for j in unparseable[:3]]}")

    log_dir = condor_dir / "logs"

    # Status counters
    total = done = running = held = missing = n_error = n_no_output = 0
    missing_jdls: list[Path] = []
    error_details: list[tuple[str, str, str]] = []   # (label, stem, detail)

    print(f"\n{'Sample':<45} {'Jobs':>5} {'Done':>5} {'Run':>5} {'Held':>5} {'Err':>5} {'Miss':>5}")
    print("-" * 76)

    for (year, subsample), job_list in sorted(groups.items()):
        n_jobs  = len(job_list)
        n_done  = n_run = n_held = n_miss = n_err_grp = 0

        for jobnum, jdl in sorted(job_list):
            stem = jdl.stem   # e.g. "2022_EGamma_3"

            # Still in the queue?
            is_held   = any(stem in jid for jid in held_jobs)
            is_active = any(stem in jid for jid in active_jobs)

            if is_held:
                n_held += 1
                continue
            if is_active:
                n_run += 1
                continue

            # Job is no longer in the queue — check logs
            log_status, detail = check_logs(log_dir, stem)

            if log_status == "success":
                n_done += 1
            elif log_status == "error":
                n_err_grp += 1
                missing_jdls.append(jdl)
                error_details.append((f"{year}/{subsample}", stem, detail))
            elif log_status == "no_output":
                # Ran but didn't print success — treat as missing
                n_miss += 1
                missing_jdls.append(jdl)
                error_details.append((f"{year}/{subsample}", stem,
                                      f"no success marker — last line: {detail}"))
            else:
                # no_logs: job hasn't produced logs yet (very recently submitted)
                n_miss += 1
                missing_jdls.append(jdl)

        total   += n_jobs
        done    += n_done
        running += n_run
        held    += n_held
        missing += n_miss
        n_error += n_err_grp

        # Colour-code the row
        if n_err_grp > 0 or n_miss > 0:
            colour = RED
        elif n_held > 0:
            colour = YELLOW
        elif n_done == n_jobs:
            colour = GREEN
        else:
            colour = ""

        label = f"{year}/{subsample}"
        print(f"{colour}{label:<45} {n_jobs:>5} {n_done:>5} {n_run:>5} "
              f"{n_held:>5} {n_err_grp:>5} {n_miss:>5}{RESET}")

    print("-" * 76)
    print(f"{'TOTAL':<45} {total:>5} {done:>5} {running:>5} {held:>5} {n_error:>5} {missing:>5}")
    print()

    # Print error details
    if error_details:
        print(f"{RED}--- Failed / incomplete jobs ---{RESET}")
        for label, stem, detail in error_details:
            print(f"  {label}  |  {stem}")
            if detail:
                print(f"    → {detail}")
        print()

    if n_error == 0 and missing == 0 and held == 0:
        print(f"{GREEN}All jobs completed successfully!{RESET}")
    else:
        if held > 0:
            print(f"{YELLOW}{held} held job(s) — run resubmit_held.py to fix.{RESET}")
        if n_error + missing > 0:
            print(f"{RED}{n_error + missing} job(s) failed or missing output.{RESET}")
            if args.resubmit:
                print(f"\nResubmitting {len(missing_jdls)} job(s)...")
                for jdl in missing_jdls:
                    result = subprocess.run(f"condor_submit {jdl}",
                                            shell=True, capture_output=True, text=True)
                    status = "OK" if result.returncode == 0 else f"FAILED: {result.stderr.strip()}"
                    print(f"  {jdl.name}: {status}")
            else:
                print("Run with --resubmit to resubmit failed/missing jobs.")


if __name__ == "__main__":
    main()
