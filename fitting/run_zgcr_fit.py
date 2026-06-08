#!/usr/bin/env python3
"""
run_zgcr_fit.py
---------------
Full pipeline for the Z+gamma control region JMS/JMR calibration fit.

Steps:
  1. Build datacards for each year using the recommended polynomial orders
     from the F-test (hardcoded below).
  2. Build RooFit workspaces (text2workspace).
  3. Combine all years into a single combined workspace.
  4. Run the fit (FitDiagnostics).
  5. Summarize JMS/JMR results.

Usage:
    python run_zgcr_fit.py --tag 26May18 [--outdir results/zgcr_final] [--skip-datacards] [--skip-workspace] [--skip-fit]

"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# F-test recommended polynomial orders per year
# ---------------------------------------------------------------------------
POLY_ORDERS = {
    "2022":     {"mc_pt": 0, "mc_rho": 1, "res_pt": 0, "res_rho": 0},
    "2022EE":   {"mc_pt": 0, "mc_rho": 2, "res_pt": 0, "res_rho": 0},
    "2023":     {"mc_pt": 0, "mc_rho": 0, "res_pt": 0, "res_rho": 0},
    "2023BPix": {"mc_pt": 0, "mc_rho": 0, "res_pt": 0, "res_rho": 0},
    "2024":     {"mc_pt": 0, "mc_rho": 1, "res_pt": 0, "res_rho": 0},
}

YEARS = list(POLY_ORDERS.keys())
ANALYSIS = "zgcr"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, dry_run: bool = False) -> None:
    print(f"\n[RUN] {cmd}")
    if dry_run:
        return
    ret = subprocess.call(cmd, shell=True)
    if ret != 0:
        print(f"[ERROR] Command failed (exit {ret}): {cmd}", file=sys.stderr)
        sys.exit(ret)


def model_dir(outdir: Path, tag: str, year: str) -> Path:
    return outdir / tag / year / "datacards" / f"{ANALYSIS}Model_{year}"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_datacards(tag: str, indir: str, outdir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("STEP 1: Build datacards")
    print("=" * 60)
    for year, orders in POLY_ORDERS.items():
        cmd = (
            f"python make_datacards.py"
            f" --year {year}"
            f" --tag {tag}"
            f" --analysis {ANALYSIS}"
            f" --indir {indir}"
            f" --outdir {outdir}"
            f" --mc-pt-order {orders['mc_pt']}"
            f" --mc-rho-order {orders['mc_rho']}"
            f" --res-pt-order {orders['res_pt']}"
            f" --res-rho-order {orders['res_rho']}"
        )
        run(cmd, dry_run)


def step_workspace(tag: str, outdir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Build workspaces")
    print("=" * 60)
    for year in YEARS:
        mdir = model_dir(outdir, tag, year)
        run(f"cd {mdir} && bash build.sh", dry_run)


def step_combine_cards(tag: str, outdir: Path, combined_dir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("STEP 3: Combine cards across years")
    print("=" * 60)
    combined_dir.mkdir(parents=True, exist_ok=True)

    # Build combineCards command — use absolute paths so cd doesn't break relative refs
    # Prefix year with 'y' to avoid Combine rejecting bin names starting with a digit
    card_args = ""
    for year in YEARS:
        mdir = model_dir(outdir.resolve(), tag, year)
        card_args += f" y{year}={mdir}/model_combined.txt"

    run(
        f"cd {combined_dir.resolve()} && combineCards.py {card_args} > model_combined.txt",
        dry_run,
    )

    # Build physics model config (multiSignalModel for r_bb and r_cc)
    t2w_cmd = (
        f"cd {combined_dir.resolve()} && text2workspace.py"
        f" -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel"
        f" --PO verbose"
        f" --PO 'map=.*/zgammabb:r_bb[1,0,10]'"
        f" --PO 'map=.*/zgammacc:r_cc[1,0,10]'"
        f" model_combined.txt"
        f" -o workspace_combined.root"
    )
    run(t2w_cmd, dry_run)


def step_fit(combined_dir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Run FitDiagnostics")
    print("=" * 60)
    # Limit JMS/JMR nuisance parameters to ±3 to prevent morphing extrapolation
    # into unphysical regions during minimization (templates are only reliable
    # within ±1; beyond that the affine morphing can produce near-zero bins).
    jmsr_ranges = ":".join(
        f"CMS_jms_{y}=-3,3:CMS_jmr_{y}=-3,3"
        for y in ["2022", "2022EE", "2023", "2023BPix", "2024"]
    )
    run(
        f"cd {combined_dir.resolve()} && combine"
        f" -M FitDiagnostics"
        f" workspace_combined.root"
        f" --saveShapes"
        f" --saveWithUncertainties"
        f" --saveNormalizations"
        f" -n _zgcr"
        f" --robustFit 1"
        f" --setParameterRanges {jmsr_ranges}"
        f" -v 1",
        dry_run,
    )


def step_summarize(combined_dir: Path, dry_run: bool) -> None:
    print("\n" + "=" * 60)
    print("STEP 5: Summarize JMS/JMR results")
    print("=" * 60)
    fitfile = combined_dir / "fitDiagnostics_zgcr.root"
    run(
        f"python summarize_jmsr.py"
        f" --fitfile {fitfile}"
        f" --setup setup_zgcr.json"
        f" --tag zgcr_combined"
        f" --output {combined_dir}/jmsr_results.md",
        dry_run,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full Z+gamma JMS/JMR calibration fit pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--tag",      required=True,  help="Production tag, e.g. 26May18")
    parser.add_argument("--indir",    default=None,   help="Input ROOT files directory (default: results/<tag>)")
    parser.add_argument("--outdir",   default="results/zgcr_final", help="Output directory for datacards")
    parser.add_argument("--skip-datacards",  action="store_true", help="Skip datacard building")
    parser.add_argument("--skip-workspace",  action="store_true", help="Skip workspace building")
    parser.add_argument("--skip-combine",    action="store_true", help="Skip combining cards")
    parser.add_argument("--skip-fit",        action="store_true", help="Skip FitDiagnostics")
    parser.add_argument("--skip-summarize",  action="store_true", help="Skip JMS/JMR summary")
    parser.add_argument("--dry-run",         action="store_true", help="Print commands without running them")
    args = parser.parse_args()

    indir = args.indir or f"results/{args.tag}"
    outdir = Path(args.outdir)
    combined_dir = outdir / args.tag / "combined"

    print(f"\nZ+gamma JMS/JMR Fit Pipeline")
    print(f"  tag:          {args.tag}")
    print(f"  indir:        {indir}")
    print(f"  outdir:       {outdir}")
    print(f"  combined_dir: {combined_dir}")
    print(f"\nPolynomial orders:")
    for year, orders in POLY_ORDERS.items():
        print(f"  {year}: MC=({orders['mc_pt']},{orders['mc_rho']}), Res=({orders['res_pt']},{orders['res_rho']})")

    if not args.skip_datacards:
        step_datacards(args.tag, indir, outdir, args.dry_run)
    if not args.skip_workspace:
        step_workspace(args.tag, outdir, args.dry_run)
    if not args.skip_combine:
        step_combine_cards(args.tag, outdir, combined_dir, args.dry_run)
    if not args.skip_fit:
        step_fit(combined_dir, args.dry_run)
    if not args.skip_summarize:
        step_summarize(combined_dir, args.dry_run)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
