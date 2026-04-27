#!/usr/bin/env python3
"""
Z(μμ) control region stack plots, split by photon presence.
DY split into pT(ll) bins (PTLL-binned samples only — no inclusive 0J/1J/2J
to avoid double-counting in the boosted regime).

MC statistical uncertainty uses sum(w²) via hist.Hist.variances(),
consistent with python/plotting.py. Plot style matches python/plotting.py.

Produces:
  - No-photon category:  lead μ pT, sublead μ pT, pt(μμ), MET, n(AK8 jets)
  - Gamma category (>=1 tight photon, pT>120 GeV):
                         lead μ pT, sublead μ pT, pt(μμ), MET, n(AK8 jets),
                         photon pT, Δφ(γ, lead μ)

Usage (from fitting/):
  # 2024 (personal EOS path):
  python make_zmumu_plots.py --year 2024 --tag Test_v15 --outdir plots/zmumu/ --personal-path

  # Older years (personal EOS path, different tag):
  for year in 2022 2022EE 2023 2023BPix; do
      python make_zmumu_plots.py --year $year --tag Test_v15_v14_private --outdir plots/zmumu/ --personal-path
  done
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from hist.intervals import ratio_uncertainty

from hbb import utils
from hbb.common_vars import LUMI

hep.style.use("CMS")

# ---------------------------------------------------------------------------
# DY (Zll) sub-groups split by pT(ll)
#
# NOTE: ONLY PTLL-binned samples are used — NOT the inclusive 0J/1J/2J samples.
# The inclusive jet-multiplicity samples already cover all pT(ll); stacking them
# together with the PTLL-binned samples double-counts DY at high pT(ll) (the
# boosted regime where our selection lives), causing ~2x MC over-prediction.
# ---------------------------------------------------------------------------
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
    "Zmm_MuonLead_pt",
    "Zmm_MuonLead_phi",
    "Zmm_MuonSublead_pt",
    "Zmm_MuonPair_mll",
    "Zmm_MuonPair_pt",
    "Zmm_ntightPhotons",
    "Zmm_nak8",
    "Photon0_pt",
    "Photon0_phi",
    "MET",
]

# ---------------------------------------------------------------------------
# Variable definitions: (column_or_derived, bins, xlabel)
# ---------------------------------------------------------------------------
NAK8_BINS = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5])

VARS_BOTH = [
    ("Zmm_MuonPair_mll",   np.linspace(80, 100, 21), r"$m(\mu\mu)$ [GeV]"),
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
) -> None:
    """
    Stack histogram + Data/MC ratio plot.

    MC statistical uncertainty uses hist.Hist.variances() = sum(w²) per bin,
    consistent with python/plotting.py. Style matches plotting.py.
    """
    plt.rcParams.update({"font.size": 24})
    fig, (ax, rax) = plt.subplots(
        2, 1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    plt.subplots_adjust(hspace=0)

    # ------------------------------------------------------------------
    # Build hist.Hist objects for each MC process.
    # hist.Hist automatically accumulates sum(w²) in .variances() when
    # events are filled with weights — this is the correct MC stat unc.
    # ------------------------------------------------------------------
    h_mc: dict[str, hist.Hist] = {}
    for proc in STACK_ORDER:
        if proc not in all_events:
            continue
        df = all_events[proc]
        mask = selection_mask.get(proc)
        if mask is not None:
            df = df[mask]
        if df.empty:
            continue

        vals = get_values(df, var).fillna(-999).values
        weights = df["finalWeight"].astype(float).values

        h = hist.Hist(hist.axis.Variable(bins, label=xlabel), storage=hist.storage.Weight())
        h.fill(vals, weight=weights)
        h_mc[proc] = h

    if not h_mc:
        plt.close(fig)
        return

    mc_ordered = [p for p in STACK_ORDER if p in h_mc]
    tot_mc = sum(h_mc[p] for p in mc_ordered)
    bin_edges = tot_mc.axes[0].edges

    # Stack fill
    hep.histplot(
        [h_mc[p] for p in mc_ordered],
        stack=True,
        histtype="fill",
        label=[PROC_STYLE[p]["label"] for p in mc_ordered],
        color=[PROC_STYLE[p]["color"] for p in mc_ordered],
        ax=ax,
    )

    # MC stat uncertainty band — sqrt(sum(w²)) from hist.variances()
    mc_vals = tot_mc.values()
    mc_vars = tot_mc.variances()
    mc_err  = np.sqrt(np.where(mc_vars > 0, mc_vars, 0))
    ax.stairs(
        np.maximum(mc_vals + mc_err, 0),
        bin_edges,
        baseline=np.maximum(mc_vals - mc_err, 0),
        fill=True,
        color="gray",
        alpha=0.4,
        label="MC stat. unc.",
        zorder=3,
    )

    # Data — filled with weight=1, so variances() == values() == counts (Poisson)
    h_data: hist.Hist | None = None
    if "Muondata" in all_events:
        df_data = all_events["Muondata"]
        mask = selection_mask.get("Muondata")
        if mask is not None:
            df_data = df_data[mask]
        if not df_data.empty:
            vals_data = get_values(df_data, var).fillna(-999).values
            h_data = hist.Hist(hist.axis.Variable(bins, label=xlabel), storage=hist.storage.Weight())
            h_data.fill(vals_data)
            hep.histplot(
                h_data,
                histtype="errorbar",
                color="k",
                label="Data",
                xerr=True,
                ax=ax,
                zorder=4,
            )

    # Axes and labels
    ax.set_ylabel("Events / bin")
    ax.set_xlabel(None)
    ax.xaxis.grid(True, which="major")
    ax.yaxis.grid(True, which="major")

    # Legend — style from plotting.py
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        ncol=2,
        fontsize="x-small",
        labelspacing=0.2,
        columnspacing=0.8,
        handlelength=1.0,
        handleheight=0.8,
        loc="upper right",
        markerscale=0.7,
    )
    hep.yscale_legend(ax, soft_fail=True)

    lumi_val = round(LUMI.get(year, 0) / 1000.0, 2)
    hep.cms.label(ax=ax, data=(h_data is not None), lumi=lumi_val, year=year, com=13.6)
    ax.text(
        0.05, 0.95, f"Z(μμ) CR — {category_label}",
        transform=ax.transAxes, fontsize=18,
        verticalalignment="top",
    )

    # ------------------------------------------------------------------
    # Ratio panel: Data/MC with Poisson uncertainty on data
    # Uses ratio_uncertainty() from hist.intervals — same as plotting.py
    # ------------------------------------------------------------------
    if h_data is not None and mc_vals.sum() > 0:
        data_vals = h_data.values()
        ratio = np.where(mc_vals > 0, data_vals / mc_vals, np.nan)
        yerr  = ratio_uncertainty(data_vals, mc_vals, "poisson")

        hep.histplot(
            ratio,
            bin_edges,
            ax=rax,
            yerr=yerr,
            histtype="errorbar",
            color="k",
            xerr=True,
            zorder=4,
        )
        rax.axhline(1, color="gray", ls="--", linewidth=1)
        rax.set_ylim(0, 2.2)
        rax.set_ylabel(r"$\frac{\mathrm{Data}}{\mathrm{Bkg}}$", y=0.5)
    else:
        rax.set_visible(False)

    rax.set_xlabel(xlabel)
    rax.xaxis.grid(True, which="major")
    rax.yaxis.grid(True, which="major")

    # Auto-zoom x-axis: skip leading/trailing empty bins
    combined = mc_vals.copy()
    if h_data is not None:
        combined = combined + h_data.values()
    nonzero = np.where(combined > 0)[0]
    if len(nonzero) > 0:
        x_min = bins[max(0, nonzero[0] - 1)]
        x_max = bins[min(len(bins) - 1, nonzero[-1] + 2)]
        ax.set_xlim(x_min, x_max)
        rax.set_xlim(x_min, x_max)

    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    year   = args.year
    tag    = args.tag
    outdir = Path(args.outdir)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.personal_path:
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

    # --- Photon-split category masks ---
    # Note: mll and pT(mumu)>300 cuts are already applied at processor level.
    PHOTON_PT_CUT = 120.0
    no_photon_mask: dict[str, pd.Series] = {}
    gamma_mask:     dict[str, pd.Series] = {}

    for proc, df in all_events.items():
        if "Zmm_ntightPhotons" in df.columns:
            no_photon_mask[proc] = df["Zmm_ntightPhotons"] == 0
            has_photon = df["Zmm_ntightPhotons"] >= 1
            if "Photon0_pt" in df.columns:
                has_photon = has_photon & (df["Photon0_pt"] > PHOTON_PT_CUT)
            gamma_mask[proc] = has_photon
        else:
            no_photon_mask[proc] = pd.Series(True, index=df.index)
            gamma_mask[proc] = pd.Series(False, index=df.index)

    # --- No-photon category ---
    print(f"\n--- No-photon category ({year}) ---")
    for var, bins, xlabel in VARS_BOTH:
        outpath = outdir / year / f"zmumu_nophoton_{var}.png"
        outpath.parent.mkdir(parents=True, exist_ok=True)
        make_stack_plot(
            all_events, no_photon_mask, var, bins, xlabel,
            year, "No photon", outpath,
        )

    # --- Gamma category ---
    print(f"\n--- Gamma category ({year}) ---")
    for var, bins, xlabel in VARS_BOTH + VARS_GAMMA:
        outpath = outdir / year / f"zmumu_gamma_{var}.png"
        make_stack_plot(
            all_events, gamma_mask, var, bins, xlabel,
            year, rf"$\geq$1 tight $\gamma$ ($p_T>${PHOTON_PT_CUT:.0f} GeV)", outpath,
        )

    print(f"\nDone. Plots in: {outdir / year}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z(μμ) CR stack plots")
    parser.add_argument("--year",    required=True,
                        help="Year: 2022, 2022EE, 2023, 2023BPix, 2024")
    parser.add_argument("--tag",     required=True, help="Skim tag, e.g. Test_v15")
    parser.add_argument("--outdir",  default="plots/zmumu/", help="Output directory")
    parser.add_argument(
        "--personal-path", action="store_true",
        help="Use personal EOS path (.../gmachado/...) instead of shared path",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Full path to the directory containing the parquets for this year, "
             "e.g. /eos/uscms/store/group/lpchbbrun3/lara/MyTag/2024 "
             "Overrides --tag and --personal-path.",
    )
    args = parser.parse_args()
    main(args)
