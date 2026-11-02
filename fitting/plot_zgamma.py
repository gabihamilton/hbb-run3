"""
Z-Gamma Control Region Plotter
------------------------------
A unified plotting tool for Z+Gamma control regions (H->bb/cc analysis).
This script reads histograms from a ROOT file and produces stacked plots
comparing Data vs. MC backgrounds.

It supports two visualization modes:
1. Simple (Default): Groups backgrounds into broad categories (e.g., "Other").
2. Detailed: Splits backgrounds into specific processes (W+jets, Z+jets, QCD, etc.).

Usage:
    python plot_zgamma.py <filename> --year <year> [options]

Arguments:
    filename       Path to the input .root file.
    --year         Data-taking year (2022, 2022EE, 2023, 2023BPix, 2022-2023).
    --mode         Region to plot: 'bb', 'cc', or 'fail'.
    --detailed     (Flag) If present, shows the detailed process breakdown.

Examples:
    # 1. Standard plot for 2022EE (Pass BB)
    python plot_zgamma.py comparisons/fitting_2022EE_control.root --year 2022EE

    # 2. Fail region for 2023
    python plot_zgamma.py comparisons/fitting_2023_control.root --year 2023 --mode fail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

# --- CONFIGURATION ---
hep.style.use("CMS")

# Luminosity in pb^-1
LUMI_DICT = {
    "2022": 7980.5,
    "2022EE": 26671.6,
    "2023": 18084.4,
    "2023BPix": 9692.1,
    "2022-2023": 62428.6,
}

# --- DEFINITIONS ---
# 1. Simple Mode (High-level summary)
SIMPLE_COLORS = {
    r"$\gamma$+jets": "#1f4e5f",  # Dark Teal (includes QCD)
    r"Other": "#5e6c7a",  # Slate Grey (W/Z+jets, Diboson, SingleT)
    r"$t\bar{t}$": "#7f964f",  # Olive Green
    r"$t\bar{t}+\gamma$": "#8e44ad",  # Violet
    r"W+$\gamma$": "#d16a4c",  # Salmon
    r"Z+$\gamma$": "#6a3d6a",  # Purple
    "Data": "black",
}

SIMPLE_STACK = [
    r"$\gamma$+jets",
    r"Other",
    r"$t\bar{t}$",
    r"$t\bar{t}+\gamma$",
    r"W+$\gamma$",
    r"Z+$\gamma$",
]

# 2. Detailed Mode (Granular breakdown)
DETAILED_COLORS = {
    r"$\gamma$+jets": "#1f4e5f",  # Dark Teal
    r"QCD": "#f1c40f",  # Yellow/Gold
    r"W+jets": "#3498db",  # Blue
    r"Z+jets": "#9b59b6",  # Light Purple
    r"$t\bar{t}$": "#7f964f",  # Olive Green
    r"$t\bar{t}+\gamma$": "#8e44ad",  # Violet
    r"Single Top": "#95a5a6",  # Grey
    r"Diboson": "#bdc3c7",  # Light Grey
    r"W+$\gamma$": "#d16a4c",  # Salmon
    r"Z+$\gamma$": "#6a3d6a",  # Dark Purple
    "Data": "black",
}

DETAILED_STACK = [
    r"Diboson",
    r"Single Top",
    r"QCD",
    r"W+jets",
    r"Z+jets",
    r"$t\bar{t}$",
    r"$t\bar{t}+\gamma$",
    r"$\gamma$+jets",
    r"W+$\gamma$",
    r"Z+$\gamma$",
]

# 3. Datacard Mode (Cristina's grouping)
DATACARD_COLORS = {
    r"zgammabb": "#6a3d6a",  # Dark Purple (Signal)
    r"wgammacs": "#d16a4c",  # Salmon (Resonant W)
    r"qcd": "#1f4e5f",  # DarkTeal (QCD)
    r"other": "#5e6c7a",  # Slate Grey (Other Backgrounds)
    "Data": "black",
}

DATACARD_STACK = [
    r"qcd",
    r"other",
    r"wgammacs",
    r"zgammabb",
]


def get_process_name(name, grouping="simple"):
    """
    Determines process label.
    grouping: 'simple', 'detailed', or 'datacard'
    """
    # --- DATA ---
    if any(x in name for x in ["data_obs", "Data", "Run2022", "Run2023"]):
        return "Data"

    # --- DATACARD MODE ---
    if grouping == "datacard":
        if "GJets" in name or "QCD" in name:
            return "qcd"
        if ("Zgamma" in name or "Zjets" in name or "ZJets" in name) and "bb_" in name:
            return "zgammabb"
        if ("Wgamma" in name or "Wjets" in name or "WJets" in name) and "bb_" in name:
            return "wgammacs"
        return "other"

    # --- SIMPLE / DETAILED MODES ---
    detailed = grouping == "detailed"

    # 1. V + Gamma
    if "Zgamma" in name:
        return r"Z+$\gamma$"
    if "WGamma" in name or "Wgamma" in name:
        return r"W+$\gamma$"

    # 2. Top + Gamma
    if "TTGamma" in name:
        return r"$t\bar{t}+\gamma$"

    # 3. Top Pair
    if "TTbar" in name or "ttbar" in name or "TTTo" in name:
        return r"$t\bar{t}$"

    # 4. Single Top
    if "singlet" in name or "SingleTop" in name:
        return r"Single Top" if detailed else r"Other"

    # 5. QCD & Gamma+Jets (THIS WAS MISSING)
    # In Simple mode: Both group into "gamma+jets" (Teal)
    # In Detailed mode: They stay separate
    if "QCD" in name:
        return r"QCD" if detailed else r"$\gamma$+jets"
    if "GJets" in name or "GJet" in name or "GJ_" in name:
        return r"$\gamma$+jets"

    # 6. V + Jets
    if "ZJets" in name or "Zjets" in name or "DYJets" in name:
        return r"Z+jets" if detailed else r"Other"
    if "WJets" in name or "Wjets" in name:
        return r"W+jets" if detailed else r"Other"

    # 7. Diboson
    if "VV" in name or "Diboson" in name:
        return r"Diboson" if detailed else r"Other"

    # Default fallback
    return r"Other"


def main(args):
    # --- CONFIG SETUP ---
    # 1. Luminosity
    if args.year not in LUMI_DICT:
        raise ValueError(
            f"Year '{args.year}' not found in configuration. Available: {list(LUMI_DICT.keys())}"
        )

    # Convert pb^-1 to fb^-1
    lumi_fb = LUMI_DICT[args.year] / 1000.0

    # 2. Region Label
    if args.mode == "bb":
        category = "pass_bb"
        region_label = r"TXbb $> 0.95$"
    elif args.mode == "cc":
        category = "pass_cc"
        region_label = r"TXcc $> 0.95$"
    elif args.mode == "fail":
        category = "fail"
        region_label = "Fail Region"
    else:
        raise ValueError("Invalid mode")

    # 3. Colors & Stacks
    if args.grouping == "detailed":
        color_map = DETAILED_COLORS
        stack_order = DETAILED_STACK
        suffix = "detailed"
    elif args.grouping == "datacard":
        color_map = DATACARD_COLORS
        stack_order = DATACARD_STACK
        suffix = "datacard"
    else:
        color_map = SIMPLE_COLORS
        stack_order = SIMPLE_STACK
        suffix = "simple"

    # --- FILE LOADING ---
    print(f"Opening {args.filename}...")
    if not Path(args.filename).exists():
        print(f"Error: File {args.filename} not found.")
        sys.exit(1)

    with uproot.open(args.filename) as f:
        keys = [k for k in f if category in k]
        print(f"Found {len(keys)} histograms for category '{category}' in year {args.year}")

        hists = {}
        data_hist = None

        # --- PROCESS GROUPING ---
        for key in keys:
            h = f[key].to_hist()
            name = key.split(";")[0]  # Clean cycle numbers

            proc = get_process_name(name, grouping=args.grouping)

            # Accumulate
            if proc == "Data":
                if data_hist is None:
                    data_hist = h
                else:
                    data_hist += h
            else:
                if proc not in hists:
                    hists[proc] = h
                else:
                    hists[proc] += h

    # --- SORTING FOR PLOT ---
    final_stack_names = [p for p in stack_order if p in hists and hists[p].sum() > 0]
    stack_hists = [hists[p] for p in final_stack_names]
    stack_colors = [color_map.get(p, "grey") for p in final_stack_names]

    # Print Yields
    print(f"\n--- Yields ({suffix.upper()}) ---")
    if data_hist:
        print(f"{'Data':<20}: {data_hist.sum():.2f}")
    for p, h in hists.items():
        print(f"{p:<20}: {h.sum():.2f}")
    print("-" * 30)

    # --- PLOTTING ---
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    if stack_hists:
        hep.histplot(
            stack_hists,
            label=final_stack_names,
            stack=True,
            histtype="fill",
            color=stack_colors,
            ax=ax,
        )

    if data_hist:
        hep.histplot(
            data_hist,
            label="Data",
            histtype="errorbar",
            color="black",
            ax=ax,
            markersize=10,
            yerr=True,
        )

        # Ratio Calculation
        total_mc_vals = sum([h.values() for h in stack_hists])
        r_num = data_hist.values()

        ratio = np.divide(r_num, total_mc_vals, out=np.ones_like(r_num), where=total_mc_vals != 0)
        centers = data_hist.axes[0].centers
        rax.errorbar(centers, ratio, yerr=0, fmt="ko", markersize=4)

    # --- STYLING ---
    ax.set_ylabel("Events / GeV")
    ax.set_xlim(0, 200)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=None, loc="upper right", ncol=2, fontsize=12)

    ax.text(
        0.95,
        0.92,
        f"{region_label}, $200 < p_T < 1200$ GeV",
        transform=ax.transAxes,
        ha="right",
        fontsize=16,
    )

    rax.set_ylabel("Data / Bkg")
    rax.set_ylim(0, 2.5)
    rax.axhline(1, color="gray", linestyle="--")
    rax.set_xlabel(r"Jet 0 $m_{sd}$ [GeV]", fontsize=18)

    # UPDATED LABEL HERE
    hep.cms.label("Private Work", data=True, lumi=f"{lumi_fb:.1f}", year=args.year, ax=ax, loc=0)

    out_name = f"comparisons/plot_zgamma_{args.year}_{args.mode}_{suffix}.png"
    plt.savefig(out_name)

    # --- FINAL OUTPUT MESSAGE ---
    full_path = Path(out_name).resolve()
    print(f"\n{'='*50}")
    print("SUCCESS: Plot saved to:")
    print(f"  {full_path}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ZGamma control regions from ROOT file.")
    parser.add_argument("filename", type=str, help="Path to the ROOT file containing histograms.")

    # REQUIRED YEAR ARGUMENT
    parser.add_argument(
        "--year",
        type=str,
        required=True,
        choices=list(LUMI_DICT.keys()),
        help="Year for luminosity and labeling (e.g., 2022, 2022EE).",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="bb",
        choices=["bb", "cc", "fail"],
        help="Tagger category (bb, cc, or fail).",
    )

    # Replace parser.add_argument("--detailed"...) with:
    parser.add_argument(
        "--grouping",
        type=str,
        default="simple",
        choices=["simple", "detailed", "datacard"],
        help="Grouping style: 'simple', 'detailed', or 'datacard' (cristina style).",
    )
    args = parser.parse_args()
    main(args)
