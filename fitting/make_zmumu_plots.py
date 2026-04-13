#!/usr/bin/env python3
"""
Z(mumu) control region stack plots, split by photon presence.

Produces:
  - No-photon category:  lead muon pT, sublead muon pT, pt(mumu), MET
  - Gamma category (>=1 tight photon, pT>120):
                         lead muon pT, sublead muon pT, pt(mumu), MET,
                         photon pT, dphi(photon, lead muon)

Usage (from fitting/):
  python make_zmumu_plots.py --year 2024 --tag Test_v15 --outdir plots/zmumu/
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from hbb import utils
from hbb.common_vars import LUMI

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# Process definitions for zmumu CR
# ---------------------------------------------------------------------------
PROCESSES = {
    "Zll":     {"color": "#3B78E7", "label": "Z+jets (DY)"},
    "Wjets":   {"color": "#28B463", "label": "W+jets"},
    "ttbar":   {"color": "#E74C3C", "label": "t#bar{t}"},
    "singlet": {"color": "#F39C12", "label": "Single t"},
    "VV":      {"color": "#9B59B6", "label": "VV"},
    "Wgamma":  {"color": "#82E0AA", "label": "W#gamma"},
    "Zgamma":  {"color": "#85C1E9", "label": "Z#gamma"},
}

# Stack order: smallest contribution on top
STACK_ORDER = ["Zgamma", "Wgamma", "VV", "singlet", "Wjets", "ttbar", "Zll"]

# ---------------------------------------------------------------------------
# Columns to load
# ---------------------------------------------------------------------------
COLS = [
    "weight",
    "finalWeight",
    "GenFlavor",
    # zmumu-specific
    "Zmm_MuonLead_pt",
    "Zmm_MuonLead_phi",
    "Zmm_MuonSublead_pt",
    "Zmm_MuonPair_mll",
    "Zmm_MuonPair_pt",
    "Zmm_ntightPhotons",
    # photon
    "Photon0_pt",
    "Photon0_phi",
    # MET
    "MET",
]

# ---------------------------------------------------------------------------
# Variable definitions: (column_or_derived, bins, xlabel)
# ---------------------------------------------------------------------------
VARS_BOTH = [
    ("Zmm_MuonLead_pt",   np.linspace(0, 500, 26), r"Lead muon $p_T$ [GeV]"),
    ("Zmm_MuonSublead_pt",np.linspace(0, 400, 26), r"Sublead muon $p_T$ [GeV]"),
    ("Zmm_MuonPair_pt",   np.linspace(0, 600, 31), r"$p_T(\mu\mu)$ [GeV]"),
    ("MET",               np.linspace(0, 300, 31),  r"MET [GeV]"),
]

VARS_GAMMA = [
    ("Photon0_pt",        np.linspace(100, 600, 26), r"Photon $p_T$ [GeV]"),
    ("dphi_photon_muon",  np.linspace(0, np.pi, 32), r"$\Delta\phi(\gamma, \mu_{lead})$"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dphi(phi1: pd.Series, phi2: pd.Series) -> pd.Series:
    """Compute |delta phi| wrapped to [0, pi]."""
    raw = np.abs(phi1.values - phi2.values)
    return pd.Series(np.where(raw > np.pi, 2 * np.pi - raw, raw), index=phi1.index)


def get_values(df: pd.DataFrame, var: str) -> pd.Series:
    if var == "dphi_photon_muon":
        return dphi(df["Photon0_phi"], df["Zmm_MuonLead_phi"])
    return df[var]


def make_stack_plot(
    all_events: dict[str, pd.DataFrame],
    selection_mask: dict[str, pd.Series],
    var: str,
    bins: np.ndarray,
    xlabel: str,
    year: str,
    category_label: str,
    outpath: Path,
):
    """Make a single stacked histogram + data/MC ratio plot."""
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    mc_hists = []
    mc_labels = []
    mc_colors = []
    mc_total = np.zeros(len(bins) - 1)

    for proc in STACK_ORDER:
        if proc not in all_events:
            continue
        df = all_events[proc]
        mask = selection_mask.get(proc)
        if mask is not None:
            df = df[mask]
        if df.empty:
            continue

        vals = get_values(df, var).fillna(-999)
        weights = df["finalWeight"].astype(float)

        h, _ = np.histogram(vals, bins=bins, weights=weights)
        mc_hists.append(h)
        mc_labels.append(PROCESSES[proc]["label"])
        mc_colors.append(PROCESSES[proc]["color"])
        mc_total += h

    if mc_hists:
        hep.histplot(
            mc_hists,
            bins=bins,
            stack=True,
            histtype="fill",
            label=mc_labels,
            color=mc_colors,
            ax=ax,
        )

    # MC total uncertainty band
    mc_err = np.sqrt(mc_total)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    ax.bar(
        bin_centers,
        2 * mc_err,
        bottom=mc_total - mc_err,
        width=np.diff(bins),
        color="gray",
        alpha=0.3,
        label="MC stat. unc.",
    )

    # Data (if available)
    data_total = None
    if "Muondata" in all_events:
        df_data = all_events["Muondata"]
        mask = selection_mask.get("Muondata")
        if mask is not None:
            df_data = df_data[mask]
        if not df_data.empty:
            vals_data = get_values(df_data, var).fillna(-999)
            data_total, _ = np.histogram(vals_data, bins=bins)
            data_err = np.sqrt(data_total)
            hep.histplot(
                data_total,
                bins=bins,
                histtype="errorbar",
                color="black",
                label="Data",
                ax=ax,
                yerr=data_err,
            )

    ax.set_ylabel("Events")
    ax.legend(fontsize=11, ncol=2)
    lumi_val = LUMI.get(year, 0) / 1000.0
    hep.cms.label(ax=ax, data=(data_total is not None), lumi=lumi_val, year=year)
    ax.set_title(f"Z(μμ) CR — {category_label}", fontsize=13, pad=40)

    # Ratio panel
    if data_total is not None and mc_total.sum() > 0:
        ratio = np.where(mc_total > 0, data_total / mc_total, np.nan)
        ratio_err = np.where(mc_total > 0, np.sqrt(data_total) / mc_total, np.nan)
        rax.errorbar(
            bin_centers, ratio, yerr=ratio_err, fmt="o", color="black", markersize=4
        )
        rax.axhline(1, color="gray", linestyle="--", linewidth=1)
        rax.set_ylim(0.5, 1.5)
        rax.set_ylabel("Data/MC")
    else:
        rax.set_visible(False)

    rax.set_xlabel(xlabel)

    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    year = args.year
    tag = args.tag
    outdir = Path(args.outdir)

    if args.personal_path:
        data_dir = Path(f"/eos/uscms/store/group/lpchbbrun3/gmachado/{tag}/{year}")
    else:
        data_dir = Path(f"/eos/uscms/store/group/lpchbbrun3/skims/{tag}/{year}")

    region = "control-zmumu"

    # Try loading each process; skip gracefully if parquets are absent
    all_events: dict[str, pd.DataFrame] = {}

    processes_to_load = list(PROCESSES.keys()) + ["Muondata"]
    with open(Path(__file__).parent / "pmap_run3.json") as f:
        import json
        pmap = json.load(f)

    for proc in processes_to_load:
        if proc not in pmap:
            continue
        datasets = pmap[proc]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = utils.load_samples(
                data_dir=data_dir,
                samples={proc: datasets},
                columns=COLS,
                region=region,
            )
        if loaded and proc in loaded and not loaded[proc].empty:
            all_events[proc] = loaded[proc]
            n = len(loaded[proc])
            print(f"  Loaded {proc}: {n} events")
        else:
            print(f"  [skip] {proc}: no parquets found")

    if not all_events:
        print("ERROR: No events loaded. Check tag/year/path.")
        return

    # Apply mll Z-peak window (if column present)
    mll_cut = {}
    for proc, df in all_events.items():
        if "Zmm_MuonPair_mll" in df.columns:
            mll_cut[proc] = (df["Zmm_MuonPair_mll"] > 76) & (df["Zmm_MuonPair_mll"] < 106)
        else:
            mll_cut[proc] = pd.Series(True, index=df.index)

    # -----------------------------------------------------------------------
    # Category splits
    # -----------------------------------------------------------------------
    PHOTON_PT_CUT = 120.0  # same as used in zgamma CR

    no_photon_mask = {}
    gamma_mask = {}

    for proc, df in all_events.items():
        base = mll_cut[proc]
        if "Zmm_ntightPhotons" in df.columns:
            no_photon_mask[proc] = base & (df["Zmm_ntightPhotons"] == 0)
            has_photon = (df["Zmm_ntightPhotons"] >= 1)
            if "Photon0_pt" in df.columns:
                has_photon = has_photon & (df["Photon0_pt"] > PHOTON_PT_CUT)
            gamma_mask[proc] = base & has_photon
        else:
            no_photon_mask[proc] = base
            gamma_mask[proc] = pd.Series(False, index=df.index)

    # -----------------------------------------------------------------------
    # No-photon category plots
    # -----------------------------------------------------------------------
    print("\n--- No-photon category ---")
    for var, bins, xlabel in VARS_BOTH:
        outpath = outdir / year / f"zmumu_nophoton_{var}.png"
        make_stack_plot(
            all_events, no_photon_mask, var, bins, xlabel,
            year, "No photon", outpath,
        )

    # -----------------------------------------------------------------------
    # Gamma category plots
    # -----------------------------------------------------------------------
    print("\n--- Gamma category ---")
    for var, bins, xlabel in VARS_BOTH + VARS_GAMMA:
        outpath = outdir / year / f"zmumu_gamma_{var}.png"
        make_stack_plot(
            all_events, gamma_mask, var, bins, xlabel,
            year, f"$\\geq$1 tight photon ($p_T>${PHOTON_PT_CUT:.0f} GeV)", outpath,
        )

    print(f"\nDone. Plots in: {outdir / year}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z(mumu) CR stack plots")
    parser.add_argument("--year",    required=True, help="e.g. 2024")
    parser.add_argument("--tag",     required=True, help="e.g. Test_v15")
    parser.add_argument("--outdir",  default="plots/zmumu/", help="Output directory")
    parser.add_argument(
        "--personal-path", action="store_true",
        help="Use personal EOS path (.../gmachado/...) instead of group shared path"
    )
    args = parser.parse_args()
    main(args)
