#!/usr/bin/env python3
"""
Z(mumu) control region stack plots, split by photon presence.
DY is split into pT(ll) bins; n(AK8 jets) is also plotted.

Produces:
  - No-photon category:  lead μ pT, sublead μ pT, pt(μμ), MET, n(AK8 jets)
  - Gamma category (>=1 tight photon, pT>120 GeV):
                         lead μ pT, sublead μ pT, pt(μμ), MET, n(AK8 jets),
                         photon pT, Δφ(γ, lead μ)

Usage (from fitting/):
  # 2024 (personal EOS path):
  python make_zmumu_plots.py --year 2024 --tag Test_v15 --outdir plots/zmumu/ --personal-path

  # Older years (shared EOS path):
  for year in 2022 2022EE 2023 2023BPix; do
      python make_zmumu_plots.py --year $year --tag Test_v15 --outdir plots/zmumu/
  done
"""

from __future__ import annotations

import argparse
import json
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
# DY (Zll) sub-groups split by pT(ll) — loaded with inline dataset lists
# ---------------------------------------------------------------------------
# NOTE: We use ONLY the PTLL-binned samples here, NOT the inclusive 0J/1J/2J samples.
# The inclusive jet-multiplicity samples (0J/1J/2J) cover ALL pT(ll) including the same
# ranges as the PTLL-binned samples. Stacking both would double-count DY at high pT(ll)
# (the boosted regime our selection lives in), causing ~2x MC over-prediction.
DY_GROUPS = {
    "Zll_PTLL_100to200": {
        "datasets": [
            "DYto2L-2Jets_MLL-50_PTLL-100to200_1J",
            "DYto2L-2Jets_MLL-50_PTLL-100to200_2J",
        ],
        "color": "#2471A3",
        "label": r"DY $p_T^{ll}$ 100–200",
    },
    "Zll_PTLL_200to400": {
        "datasets": [
            "DYto2L-2Jets_MLL-50_PTLL-200to400_1J",
            "DYto2L-2Jets_MLL-50_PTLL-200to400_2J",
        ],
        "color": "#2E86C1",
        "label": r"DY $p_T^{ll}$ 200–400",
    },
    "Zll_PTLL_400to600": {
        "datasets": [
            "DYto2L-2Jets_MLL-50_PTLL-400to600_1J",
            "DYto2L-2Jets_MLL-50_PTLL-400to600_2J",
        ],
        "color": "#5DADE2",
        "label": r"DY $p_T^{ll}$ 400–600",
    },
    "Zll_PTLL_600": {
        "datasets": [
            "DYto2L-2Jets_MLL-50_PTLL-600_1J",
            "DYto2L-2Jets_MLL-50_PTLL-600_2J",
        ],
        "color": "#AED6F1",
        "label": r"DY $p_T^{ll}$ >600",
    },
}

# ---------------------------------------------------------------------------
# Other MC processes — loaded via pmap_run3.json
# ---------------------------------------------------------------------------
OTHER_PROCESSES = {
    "Wjets":   {"color": "#28B463", "label": "W+jets"},
    "ttbar":   {"color": "#E74C3C", "label": r"$t\bar{t}$"},
    "singlet": {"color": "#F39C12", "label": "Single t"},
    "VV":      {"color": "#9B59B6", "label": "VV"},
    "Wgamma":  {"color": "#82E0AA", "label": r"W$\gamma$"},
    "Zgamma":  {"color": "#85C1E9", "label": r"Z$\gamma$"},
}

# Stack order: smallest contribution on top; lowest pT bin at bottom
STACK_ORDER = [
    "Zgamma", "Wgamma", "VV", "singlet", "Wjets", "ttbar",
    "Zll_PTLL_600", "Zll_PTLL_400to600", "Zll_PTLL_200to400", "Zll_PTLL_100to200",
]

# Combined style lookup
PROC_STYLE: dict[str, dict] = {
    **{k: {"color": v["color"], "label": v["label"]} for k, v in DY_GROUPS.items()},
    **OTHER_PROCESSES,
}

# ---------------------------------------------------------------------------
# Columns to load from parquet
# ---------------------------------------------------------------------------
COLS = [
    "weight",
    "GenFlavor",
    # zmumu-specific
    "Zmm_MuonLead_pt",
    "Zmm_MuonLead_phi",
    "Zmm_MuonSublead_pt",
    "Zmm_MuonPair_mll",
    "Zmm_MuonPair_pt",
    "Zmm_ntightPhotons",
    "Zmm_nak8",
    # photon
    "Photon0_pt",
    "Photon0_phi",
    # MET
    "MET",
]

# ---------------------------------------------------------------------------
# Variable definitions: (column_or_derived, bins, xlabel)
# ---------------------------------------------------------------------------
NAK8_BINS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

VARS_BOTH = [
    ("Zmm_MuonLead_pt",    np.linspace(0, 500, 26),  r"Lead muon $p_T$ [GeV]"),
    ("Zmm_MuonSublead_pt", np.linspace(0, 400, 26),  r"Sublead muon $p_T$ [GeV]"),
    ("Zmm_MuonPair_pt",    np.linspace(0, 600, 31),  r"$p_T(\mu\mu)$ [GeV]"),
    ("MET",                np.linspace(0, 300, 31),   r"MET [GeV]"),
    ("Zmm_nak8",           NAK8_BINS,                 r"Number of AK8 jets"),
]

VARS_GAMMA = [
    ("Photon0_pt",        np.linspace(100, 600, 26), r"Photon $p_T$ [GeV]"),
    ("dphi_photon_muon",  np.linspace(0, np.pi, 32), r"$\Delta\phi(\gamma, \mu_\mathrm{lead})$"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def dphi(phi1: pd.Series, phi2: pd.Series) -> pd.Series:
    """Compute |Δφ| wrapped to [0, π]."""
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
        mc_labels.append(PROC_STYLE[proc]["label"])
        mc_colors.append(PROC_STYLE[proc]["color"])
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

    # MC stat uncertainty band
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

    # Data
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
    ax.legend(fontsize=10, ncol=2)
    lumi_val = round(LUMI.get(year, 0) / 1000.0, 2)
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
        rax.set_ylim(0, 2)
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
    print(f"\n=== Z(μμ) CR plots: {year}  [{data_dir}] ===\n")

    all_events: dict[str, pd.DataFrame] = {}

    # --- Load DY pT-bin sub-groups (inline dataset lists) ---
    for proc, info in DY_GROUPS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = utils.load_samples(
                data_dir=data_dir,
                samples={proc: info["datasets"]},
                columns=COLS,
                region=region,
            )
        if loaded and proc in loaded and not loaded[proc].empty:
            all_events[proc] = loaded[proc]
            print(f"  Loaded {proc}: {len(loaded[proc]):,} events")
        else:
            print(f"  [skip] {proc}: no parquets found")

    # --- Load other MC + data via pmap ---
    with open(Path(__file__).parent / "pmap_run3.json") as f:
        pmap = json.load(f)

    for proc in list(OTHER_PROCESSES.keys()) + ["Muondata"]:
        if proc not in pmap:
            print(f"  [skip] {proc}: not in pmap")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = utils.load_samples(
                data_dir=data_dir,
                samples={proc: pmap[proc]},
                columns=COLS,
                region=region,
            )
        if loaded and proc in loaded and not loaded[proc].empty:
            all_events[proc] = loaded[proc]
            print(f"  Loaded {proc}: {len(loaded[proc]):,} events")
        else:
            print(f"  [skip] {proc}: no parquets found")

    if not all_events:
        print("ERROR: No events loaded. Check --tag / --year / --personal-path.")
        return

    # --- mll Z-peak window ---
    mll_cut = {}
    for proc, df in all_events.items():
        if "Zmm_MuonPair_mll" in df.columns:
            mll_cut[proc] = (df["Zmm_MuonPair_mll"] > 76) & (df["Zmm_MuonPair_mll"] < 106)
        else:
            mll_cut[proc] = pd.Series(True, index=df.index)

    # --- Photon-split category masks ---
    PHOTON_PT_CUT = 120.0
    no_photon_mask: dict[str, pd.Series] = {}
    gamma_mask:     dict[str, pd.Series] = {}

    for proc, df in all_events.items():
        base = mll_cut[proc]
        if "Zmm_ntightPhotons" in df.columns:
            no_photon_mask[proc] = base & (df["Zmm_ntightPhotons"] == 0)
            has_photon = df["Zmm_ntightPhotons"] >= 1
            if "Photon0_pt" in df.columns:
                has_photon = has_photon & (df["Photon0_pt"] > PHOTON_PT_CUT)
            gamma_mask[proc] = base & has_photon
        else:
            no_photon_mask[proc] = base
            gamma_mask[proc] = pd.Series(False, index=df.index)

    # --- No-photon category ---
    print(f"\n--- No-photon category ({year}) ---")
    for var, bins, xlabel in VARS_BOTH:
        make_stack_plot(
            all_events, no_photon_mask, var, bins, xlabel,
            year, "No photon",
            outdir / year / f"zmumu_nophoton_{var}.png",
        )

    # --- Gamma category ---
    print(f"\n--- Gamma category ({year}) ---")
    for var, bins, xlabel in VARS_BOTH + VARS_GAMMA:
        make_stack_plot(
            all_events, gamma_mask, var, bins, xlabel,
            year, rf"$\geq$1 tight photon ($p_T>${PHOTON_PT_CUT:.0f} GeV)",
            outdir / year / f"zmumu_gamma_{var}.png",
        )

    print(f"\nDone. Plots in: {outdir / year}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z(mumu) CR stack plots")
    parser.add_argument("--year",    required=True,
                        help="Year: 2022, 2022EE, 2023, 2023BPix, 2024")
    parser.add_argument("--tag",     required=True, help="Skim tag, e.g. Test_v15")
    parser.add_argument("--outdir",  default="plots/zmumu/", help="Output directory")
    parser.add_argument(
        "--personal-path", action="store_true",
        help="Use personal EOS path (.../gmachado/...) instead of shared path",
    )
    args = parser.parse_args()
    main(args)
