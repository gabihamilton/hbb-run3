#!/usr/bin/env python3
"""
run_ftest_pipeline.py
---------------------
Full F-test pipeline for testing QCD transfer function polynomial orders.

Runs:
  1. make_datacards.py  (null model)
  2. make_datacards.py  (alt model)
  3. combineCards.py + text2workspace.py for both
  4. run_ftest.py       (snapshots, GoF, toys)
  5. plot_ftest.py      (F-statistic plot + p-value)

The null model should be the SIMPLER (lower order) polynomial.
The alt model should be the MORE COMPLEX (higher order) polynomial.
A p-value > 0.05 means the added complexity is NOT justified → keep null.

Polynomial orders:
  --mc-pt-order / --mc-rho-order : MC template transfer factor (default: 0, 1)
  --res-pt-order / --res-rho-order : data residual transfer factor (default: 0, 0)

Typical comparisons to run:
  # Test residual rho order: (0,0) vs (0,1)
  python run_ftest_pipeline.py --year 2024 --tag 26May18 \\
      --null-res-pt 0 --null-res-rho 0 \\
      --alt-res-pt  0 --alt-res-rho  1

  # Test residual rho order: (0,1) vs (0,2)
  python run_ftest_pipeline.py --year 2024 --tag 26May18 \\
      --null-res-pt 0 --null-res-rho 1 \\
      --alt-res-pt  0 --alt-res-rho  2

  # Test residual pt order: (0,0) vs (1,0)
  python run_ftest_pipeline.py --year 2024 --tag 26May18 \\
      --null-res-pt 0 --null-res-rho 0 \\
      --alt-res-pt  1 --alt-res-rho  0

  # Test MC template rho order: (0,1) vs (0,2)
  python run_ftest_pipeline.py --year 2024 --tag 26May18 \\
      --null-mc-pt 0 --null-mc-rho 1 \\
      --alt-mc-pt  0 --alt-mc-rho  2

IMPORTANT: Requires CMSSW environment with Combine installed.
  source /cvmfs/cms.cern.ch/cmsset_default.sh
  cd /path/to/CMSSW_X_Y_Z/src && cmsenv
  cd /path/to/fitting/
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: str, check: bool = True) -> int:
    print(f"\n[PIPELINE] {cmd}")
    ret = os.system(cmd)
    if check and ret != 0:
        raise RuntimeError(f"Command failed (exit {ret}): {cmd}")
    return ret


def poly_tag(mc_pt: int, mc_rho: int, res_pt: int, res_rho: int) -> str:
    return f"mc{mc_pt}{mc_rho}_res{res_pt}{res_rho}"


def n_params(pt_order: int, rho_order: int) -> int:
    """Number of free parameters in a 2D Bernstein polynomial."""
    return (pt_order + 1) * (rho_order + 1)


# ---------------------------------------------------------------------------
# Step 1 & 2: build datacards
# ---------------------------------------------------------------------------

def build_datacards(year: str, tag: str, analysis: str,
                    mc_pt: int, mc_rho: int, res_pt: int, res_rho: int,
                    indir: str, outdir: str) -> Path:
    """Run make_datacards.py and return the model directory."""
    label = poly_tag(mc_pt, mc_rho, res_pt, res_rho)
    card_outdir = Path(outdir) / f"ftest_{label}"

    run(
        f"python make_datacards.py"
        f" --year {year}"
        f" --tag {tag}"
        f" --analysis {analysis}"
        f" --indir {indir}"
        f" --outdir {card_outdir}"
        f" --mc-pt-order {mc_pt}"
        f" --mc-rho-order {mc_rho}"
        f" --res-pt-order {res_pt}"
        f" --res-rho-order {res_rho}"
    )
    return card_outdir / tag / year / "datacards" / f"{analysis}Model_{year}"


# ---------------------------------------------------------------------------
# Step 3: build workspace
# ---------------------------------------------------------------------------

def build_workspace(model_dir: Path) -> Path:
    """Run the generated build script (combineCards + text2workspace)."""
    build_script = model_dir / "build.sh"
    if not build_script.exists():
        raise FileNotFoundError(f"Build script not found: {build_script}")

    # Run inside the model directory so relative paths work
    run(f"cd {model_dir} && bash build.sh")
    workspace = model_dir / "workspace.root"
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not created: {workspace}")
    return workspace


# ---------------------------------------------------------------------------
# Step 4: run F-test
# ---------------------------------------------------------------------------

def run_ftest(workspace_null: Path, workspace_alt: Path,
              ntoys: int, seed: int, ftest_tag: str,
              ftest_dir: Path) -> None:
    """Run run_ftest.py from the ftest output directory."""
    ftest_dir.mkdir(parents=True, exist_ok=True)
    # run_ftest.py writes output files to the CWD
    run(
        f"cd {ftest_dir} && python {Path(__file__).parent / 'run_ftest.py'}"
        f" --null {workspace_null.resolve()}"
        f" --alt  {workspace_alt.resolve()}"
        f" --ntoys {ntoys}"
        f" --seed {seed}"
        f" --tag {ftest_tag}"
    )


# ---------------------------------------------------------------------------
# Step 5: plot
# ---------------------------------------------------------------------------

def plot_ftest(ftest_dir: Path, ftest_tag: str,
               p_null: int, p_alt: int, nbins: int) -> float | None:
    """Run plot_ftest.py and return the p-value (parsed from stdout)."""
    import subprocess
    cmd = (
        f"cd {ftest_dir} && python {Path(__file__).parent / 'plot_ftest.py'}"
        f" --tag {ftest_tag}"
        f" --p1 {p_null}"
        f" --p2 {p_alt}"
        f" --nbins {nbins}"
    )
    print(f"\n[PIPELINE] {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=False,
                            text=True, stdout=subprocess.PIPE)
    print(result.stdout)
    # Parse p-value from output line: "P-Value: 0.1234"
    for line in result.stdout.splitlines():
        if line.startswith("P-Value:"):
            try:
                return float(line.split(":")[1].strip())
            except ValueError:
                pass
    return None


# ---------------------------------------------------------------------------
# Single comparison helper
# ---------------------------------------------------------------------------

def run_single_comparison(args, null_orders: tuple, alt_orders: tuple,
                           indir: str) -> float | None:
    """
    Run one null-vs-alt comparison. Returns p-value or None on failure.
    null_orders / alt_orders: (mc_pt, mc_rho, res_pt, res_rho)
    """
    null_mc_pt, null_mc_rho, null_res_pt, null_res_rho = null_orders
    alt_mc_pt,  alt_mc_rho,  alt_res_pt,  alt_res_rho  = alt_orders

    null_label = poly_tag(null_mc_pt, null_mc_rho, null_res_pt, null_res_rho)
    alt_label  = poly_tag(alt_mc_pt,  alt_mc_rho,  alt_res_pt,  alt_res_rho)
    ftest_tag  = f"{args.year}_{null_label}_vs_{alt_label}"

    p_null = n_params(null_mc_pt, null_mc_rho) + n_params(null_res_pt, null_res_rho)
    p_alt  = n_params(alt_mc_pt,  alt_mc_rho)  + n_params(alt_res_pt,  alt_res_rho)

    print(f"\n{'='*60}")
    print(f"Comparing: {null_label} ({p_null} params) vs {alt_label} ({p_alt} params)")
    print(f"{'='*60}")

    ftest_dir = Path(args.ftest_outdir) / ftest_tag

    if not args.skip_datacards:
        null_model_dir = build_datacards(
            args.year, args.tag, args.analysis,
            null_mc_pt, null_mc_rho, null_res_pt, null_res_rho,
            indir, args.outdir)
        try:
            alt_model_dir = build_datacards(
                args.year, args.tag, args.analysis,
                alt_mc_pt, alt_mc_rho, alt_res_pt, alt_res_rho,
                indir, args.outdir)
        except RuntimeError as e:
            print(f"[WARN] Alt model ({alt_label}) failed to build: {e}")
            print(f"[WARN] Model too complex — stopping scan, keeping null ({null_label})")
            return None
    else:
        null_model_dir = (Path(args.outdir) / f"ftest_{null_label}" /
                          args.tag / args.year / "datacards" /
                          f"{args.analysis}Model_{args.year}")
        alt_model_dir  = (Path(args.outdir) / f"ftest_{alt_label}"  /
                          args.tag / args.year / "datacards" /
                          f"{args.analysis}Model_{args.year}")

    if not args.skip_workspace:
        null_ws = build_workspace(null_model_dir)
        alt_ws  = build_workspace(alt_model_dir)
    else:
        null_ws = null_model_dir / "workspace.root"
        alt_ws  = alt_model_dir  / "workspace.root"

    run_ftest(null_ws, alt_ws, args.ntoys, args.seed, ftest_tag, ftest_dir)
    nbins = args.nbins or 21
    return plot_ftest(ftest_dir, ftest_tag, p_null, p_alt, nbins)


# ---------------------------------------------------------------------------
# Auto-scan: sequentially test increasing orders until p > threshold
# ---------------------------------------------------------------------------

def auto_scan(args, indir: str, p_threshold: float = 0.05) -> None:
    """
    Automatically scan polynomial orders, starting from the base model,
    incrementally increasing rho order then pt order for the residual.

    For each step:
      - p-value < threshold → alt model justified, move up, continue
      - p-value ≥ threshold → null model sufficient, STOP → recommended order

    Also tests MC template rho order independently.
    """
    results = []

    def record(null_ord, alt_ord, pval):
        results.append((null_ord, alt_ord, pval))
        status = "✓ prefer alt (p < 0.05)" if pval < p_threshold else "✗ keep null (p ≥ 0.05)"
        print(f"\n  p-value = {pval:.4f}  →  {status}")

    # -----------------------------------------------------------------------
    # Scan 1: residual rho order (pt fixed at base)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SCAN: residual rho order (pt=0 fixed)")
    print("="*60)
    mc_pt, mc_rho = args.base_mc_pt, args.base_mc_rho
    res_pt = 0
    current_rho = 0

    for next_rho in range(1, args.max_res_rho + 1):
        pval = run_single_comparison(
            args,
            null_orders=(mc_pt, mc_rho, res_pt, current_rho),
            alt_orders =(mc_pt, mc_rho, res_pt, next_rho),
            indir=indir,
        )
        if pval is None:
            print(f"[WARN] Stopping residual rho scan at {current_rho}→{next_rho} (fit failed or p-value unavailable)")
            break
        record((mc_pt, mc_rho, res_pt, current_rho),
               (mc_pt, mc_rho, res_pt, next_rho), pval)
        if pval >= p_threshold:
            print(f"\n  → STOP: residual rho order {current_rho} is sufficient")
            break
        current_rho = next_rho

    recommended_res_rho = current_rho

    # -----------------------------------------------------------------------
    # Scan 2: residual pt order (rho fixed at recommended)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"SCAN: residual pt order (rho={recommended_res_rho} fixed)")
    print("="*60)
    current_pt = 0

    for next_pt in range(1, args.max_res_pt + 1):
        pval = run_single_comparison(
            args,
            null_orders=(mc_pt, mc_rho, current_pt, recommended_res_rho),
            alt_orders =(mc_pt, mc_rho, next_pt,    recommended_res_rho),
            indir=indir,
        )
        if pval is None:
            break
        record((mc_pt, mc_rho, current_pt, recommended_res_rho),
               (mc_pt, mc_rho, next_pt,    recommended_res_rho), pval)
        if pval >= p_threshold:
            print(f"\n  → STOP: residual pt order {current_pt} is sufficient")
            break
        current_pt = next_pt

    recommended_res_pt = current_pt

    # -----------------------------------------------------------------------
    # Scan 3: MC template rho order (always start from 0)
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("SCAN: MC template rho order")
    print("="*60)
    current_mc_rho = 0

    for next_mc_rho in range(1, args.max_mc_rho + 1):
        pval = run_single_comparison(
            args,
            null_orders=(mc_pt, current_mc_rho, recommended_res_pt, recommended_res_rho),
            alt_orders =(mc_pt, next_mc_rho,    recommended_res_pt, recommended_res_rho),
            indir=indir,
        )
        if pval is None:
            break
        record((mc_pt, current_mc_rho, recommended_res_pt, recommended_res_rho),
               (mc_pt, next_mc_rho,    recommended_res_pt, recommended_res_rho), pval)
        if pval >= p_threshold:
            print(f"\n  → STOP: MC rho order {current_mc_rho} is sufficient")
            break
        current_mc_rho = next_mc_rho

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("AUTO-SCAN SUMMARY")
    print("="*60)
    print(f"\n  {'Comparison':<45} {'p-value':>8}  {'Decision'}")
    print(f"  {'-'*70}")
    for null_ord, alt_ord, pval in results:
        null_s = poly_tag(*null_ord)
        alt_s  = poly_tag(*alt_ord)
        decision = "prefer alt" if pval < p_threshold else "keep null ← STOP"
        print(f"  {null_s} vs {alt_s:<25} {pval:>8.4f}  {decision}")

    print(f"\n  ✓ RECOMMENDED:")
    print(f"    MC template : pt={mc_pt}, rho={current_mc_rho}")
    print(f"    Residual    : pt={recommended_res_pt}, rho={recommended_res_rho}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full F-test pipeline for QCD transfer function polynomial order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Common args
    parser.add_argument("--year",     required=True,
                        choices=["2022", "2022EE", "2023", "2023BPix", "2024"])
    parser.add_argument("--tag",      required=True)
    parser.add_argument("--analysis", default="zgcr")
    parser.add_argument("--indir",    default=None)
    parser.add_argument("--outdir",   default="results/ftest")
    parser.add_argument("--ftest-outdir", default="ftest_results")
    parser.add_argument("--ntoys",    type=int, default=500)
    parser.add_argument("--seed",     type=int, default=123456)
    parser.add_argument("--nbins",    type=int, default=None)
    parser.add_argument("--skip-datacards", action="store_true")
    parser.add_argument("--skip-workspace", action="store_true")

    # --- Mode 1: auto-scan ---
    scan = parser.add_argument_group("Auto-scan mode (--auto-scan)")
    scan.add_argument("--auto-scan", action="store_true",
                      help="Automatically scan polynomial orders and find the "
                           "recommended configuration")
    scan.add_argument("--base-mc-pt",  type=int, default=0)
    scan.add_argument("--base-mc-rho", type=int, default=1,
                      help="Base MC template rho order used for residual scans (default: 1)")
    scan.add_argument("--max-res-rho", type=int, default=3,
                      help="Max residual rho order to test (default: 3)")
    scan.add_argument("--max-res-pt",  type=int, default=2,
                      help="Max residual pt order to test (default: 2)")
    scan.add_argument("--max-mc-rho",  type=int, default=3,
                      help="Max MC template rho order to test (default: 3)")
    scan.add_argument("--p-threshold", type=float, default=0.05,
                      help="p-value threshold to stop (default: 0.05)")

    # --- Mode 2: single comparison ---
    single = parser.add_argument_group("Single comparison mode")
    single.add_argument("--null-mc-pt",  type=int, default=0)
    single.add_argument("--null-mc-rho", type=int, default=1)
    single.add_argument("--null-res-pt", type=int, default=0)
    single.add_argument("--null-res-rho",type=int, default=0)
    single.add_argument("--alt-mc-pt",   type=int, default=0)
    single.add_argument("--alt-mc-rho",  type=int, default=1)
    single.add_argument("--alt-res-pt",  type=int, default=0)
    single.add_argument("--alt-res-rho", type=int, default=1)

    args = parser.parse_args()
    indir = args.indir or f"results/{args.tag}"

    if args.auto_scan:
        auto_scan(args, indir, p_threshold=args.p_threshold)
    else:
        # Single comparison
        pval = run_single_comparison(
            args,
            null_orders=(args.null_mc_pt, args.null_mc_rho,
                         args.null_res_pt, args.null_res_rho),
            alt_orders =(args.alt_mc_pt,  args.alt_mc_rho,
                         args.alt_res_pt,  args.alt_res_rho),
            indir=indir,
        )
        if pval is not None:
            decision = "keep null (simpler model sufficient)" \
                       if pval >= 0.05 else "prefer alt (more complex model justified)"
            print(f"\n  Final p-value: {pval:.4f}  →  {decision}")


if __name__ == "__main__":
    main()
