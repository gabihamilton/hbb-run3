#!/usr/bin/env python3
"""
plot_disc_zgcr.py
-----------------
Plot the modified TXbbXcc discriminant in the zgamma CR pre-selection region
(inclusive, before pass/fail split) for Z(cc) and W(cs) processes.

Two discriminants are compared side by side:
  1. Current:  TXbbXcc       = (Xbb + Xcc) / (Xbb + Xcc + QCD)
  2. Modified: TXbbXcc_Wcs   = (Xbb + Xcc) / (Xbb + Xcc + QCD + Xcs)

Z(cc)  = Zgamma + Zjets  with generator-level charm flavor (GenFlavor == 2)
W(cs)  = Wgamma + Wjets  with generator-level charm flavor (GenFlavor == 2)

Usage
-----
    python plot_disc_zgcr.py \\
        --year 2022EE \\
        --tag 26Feb03 \\
        --outdir plots/disc

    # Override EOS path (e.g., custom skim location):
    python plot_disc_zgcr.py \\
        --year 2024 \\
        --tag 26Feb03 \\
        --data-dir /eos/uscms/store/group/lpchbbrun3/skims/26Feb03/2024 \\
        --outdir plots/disc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

# -- repo utilities ----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbb import utils

plt.style.use(hep.style.CMS)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGION = "control-zgamma"
GENFLAVOR_CHARM = 2          # GenFlavor == 2 → c / cs decay
NBINS = 40
DISC_RANGE = (0.0, 1.0)

# Processes and their display labels / colors
PROCESSES = {
    "Zgamma": {"gfilt": GENFLAVOR_CHARM, "label": r"$Z(\gamma) \to cc$", "color": "royalblue"},
    "Zjets":  {"gfilt": GENFLAVOR_CHARM, "label": r"$Z(jets) \to cc$",   "color": "cornflowerblue"},
    "Wgamma": {"gfilt": GENFLAVOR_CHARM, "label": r"$W(\gamma) \to cs$", "color": "tomato"},
    "Wjets":  {"gfilt": GENFLAVOR_CHARM, "label": r"$W(jets) \to cs$",   "color": "salmon"},
}

# How to visually merge the four processes into two groups
GROUPS = {
    "zcc": {
        "procs":  ["Zgamma", "Zjets"],
        "label":  r"$Z(cc)$  [Zgamma + Zjets]",
        "color":  "royalblue",
        "hatch":  None,
    },
    "wcs": {
        "procs":  ["Wgamma", "Wjets"],
        "label":  r"$W(cs)$  [Wgamma + Wjets]",
        "color":  "tomato",
        "hatch":  "///",
    },
}

# Columns needed from parquet
COLS_BASE = [
    "weight",
    "FatJet0_pt",
    "FatJet0_msd",
    "FatJet0_phi",
    "GenFlavor",
    # discriminant (pre-computed in the skim)
    "FatJet0_ParTPXbbXcc",
    # raw ParT probs for the modified discriminant
    "FatJet0_ParTPXbb",
    "FatJet0_ParTPXcc",
    "FatJet0_ParTPQCD",
    "FatJet0_ParTPXcs",
]

COLS_PHOTON = [
    "Photon0_pt",
    "Photon0_phi",
    "MET",
    "Photon200",
    "Photon110EB_TightID_TightIso",
]

# PyArrow pre-filters (loose — tightened in Python after loading)
PQ_FILTERS = [
    ("FatJet0_msd", ">=", 15.0),
    ("FatJet0_msd", "<=", 210.0),
    ("FatJet0_pt",  ">=", 240.0),
    ("Photon0_pt",  ">=", 100.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_preselection(df: pd.DataFrame) -> pd.Series:
    """Zgamma CR pre-selection — mirrors make_hists.py logic exactly."""
    dphi_raw = np.abs(df["Photon0_phi"] - df["FatJet0_phi"])
    dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

    # MET may be a plain float column or a struct with a .pt attribute
    met_col = df["MET"] if "MET" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
    met_pt = met_col.pt if hasattr(met_col, "pt") else met_col

    trigger = df["Photon200"].astype(bool) | df["Photon110EB_TightID_TightIso"].astype(bool)

    sel = (
        (df["FatJet0_msd"] > 20)  & (df["FatJet0_msd"] < 201) &
        (df["FatJet0_pt"]  > 250) &
        (df["Photon0_pt"]  > 120) &
        (dphi              > 2.2) &
        (met_pt            < 50)  &
        trigger
    )
    return sel


def compute_disc_modified(df: pd.DataFrame) -> pd.Series:
    """TXbbXcc with Prob(Wcs) added to the denominator."""
    num = df["FatJet0_ParTPXbb"] + df["FatJet0_ParTPXcc"]
    den = num + df["FatJet0_ParTPQCD"] + df["FatJet0_ParTPXcs"]
    return num / den.replace(0, np.nan)   # avoid division by zero


def fill_group_hists(
    events_dict: dict[str, pd.DataFrame],
    nbins: int = NBINS,
    disc_range: tuple = DISC_RANGE,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Build weighted histograms for both discriminants, per group (zcc / wcs).

    Returns
    -------
    {group_name: {"current": values, "modified": values, "edges": edges}}
    """
    edges = np.linspace(disc_range[0], disc_range[1], nbins + 1)
    result = {g: {"current": np.zeros(nbins), "modified": np.zeros(nbins)} for g in GROUPS}

    for grp_name, grp_cfg in GROUPS.items():
        for proc in grp_cfg["procs"]:
            if proc not in events_dict:
                print(f"  [WARN] process {proc!r} not found — skipping")
                continue
            df = events_dict[proc]

            sel = apply_preselection(df)
            # keep only charm-flavor events
            sel = sel & (df["GenFlavor"] == GENFLAVOR_CHARM)
            df = df[sel]

            if len(df) == 0:
                print(f"  [WARN] {proc}: 0 events after pre-selection + flavor cut")
                continue

            w = df["finalWeight"].values

            disc_cur = df["FatJet0_ParTPXbbXcc"].values
            disc_mod = compute_disc_modified(df).values

            h_cur, _ = np.histogram(disc_cur, bins=edges, weights=w)
            h_mod, _ = np.histogram(disc_mod, bins=edges, weights=w)

            result[grp_name]["current"]  += h_cur
            result[grp_name]["modified"] += h_mod
            print(f"  {proc} ({grp_name}): {len(df)} events, sumw = {w.sum():.3g}")

    # Store edges once
    for g in GROUPS:
        result[g]["edges"] = edges

    return result


def norm_hist(vals: np.ndarray) -> np.ndarray:
    """Shape-normalise to unit area."""
    total = vals.sum()
    return vals / total if total > 0 else vals


def separation(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    <S²> separation between two normalised histograms:
      S² = 0.5 * sum( (p_i - q_i)^2 / (p_i + q_i) )
    Returns a value in [0, 1]; 0 = identical, 1 = fully separated.
    """
    p = norm_hist(h1)
    q = norm_hist(h2)
    denom = p + q
    with np.errstate(invalid="ignore"):
        s = 0.5 * np.sum(np.where(denom > 0, (p - q) ** 2 / denom, 0.0))
    return s


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_comparison_plot(
    hists: dict,
    year: str,
    outdir: Path,
) -> None:
    """
    Two-panel figure:
      left  — current TXbbXcc = (Xbb+Xcc)/(Xbb+Xcc+QCD)
      right — modified TXbbXcc_Wcs = (Xbb+Xcc)/(Xbb+Xcc+QCD+Xcs)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    fig.subplots_adjust(wspace=0.32)

    disc_keys   = ["current",  "modified"]
    disc_titles = [
        r"$T_{Xbb+Xcc}$   =   $\frac{P_{bb}+P_{cc}}{P_{bb}+P_{cc}+P_{QCD}}$",
        r"$T_{Xbb+Xcc}^{\,Wcs}$   =   $\frac{P_{bb}+P_{cc}}{P_{bb}+P_{cc}+P_{QCD}+P_{cs}}$",
    ]

    for ax, dkey, dtitle in zip(axes, disc_keys, disc_titles):
        edges = hists["zcc"]["edges"]
        centres = 0.5 * (edges[:-1] + edges[1:])
        width = edges[1] - edges[0]

        for grp_name, grp_cfg in GROUPS.items():
            h_raw = hists[grp_name][dkey]
            h_norm = norm_hist(h_raw)
            ax.bar(
                centres, h_norm,
                width=width,
                color=grp_cfg["color"],
                alpha=0.55,
                hatch=grp_cfg["hatch"],
                label=grp_cfg["label"],
                edgecolor="none",
            )
            ax.step(
                edges, np.append(h_norm, h_norm[-1]),
                where="post",
                color=grp_cfg["color"],
                linewidth=1.5,
            )

        sep = separation(hists["zcc"][dkey], hists["wcs"][dkey])
        ax.text(
            0.04, 0.97, f"Sep. = {sep:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=11, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )

        ax.set_title(dtitle, fontsize=12, pad=8)
        ax.set_xlabel("Discriminant value", fontsize=12)
        ax.set_ylabel("Normalised events / bin", fontsize=12)
        ax.set_xlim(*DISC_RANGE)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=10, loc="upper center")
        hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=12)

    fname = outdir / f"disc_zcc_vs_wcs_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {fname}")


def make_ratio_plot(
    hists: dict,
    year: str,
    outdir: Path,
) -> None:
    """
    Ratio Z(cc)/W(cs) for both discriminants on the same axis.
    A ratio > 1 in a region means Z(cc) is more abundant there.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    styles = {
        "current":  dict(color="navy",   ls="-",  lw=2,  label=r"Current $T_{Xbb+Xcc}$"),
        "modified": dict(color="darkred", ls="--", lw=2,  label=r"Modified $T_{Xbb+Xcc}^{Wcs}$"),
    }

    for dkey, sty in styles.items():
        edges = hists["zcc"]["edges"]
        centres = 0.5 * (edges[:-1] + edges[1:])

        h_z = norm_hist(hists["zcc"][dkey])
        h_w = norm_hist(hists["wcs"][dkey])

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(h_w > 0, h_z / h_w, np.nan)

        ax.step(
            edges, np.append(ratio, ratio[-1]),
            where="post",
            **sty,
        )

    ax.axhline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("Discriminant value", fontsize=12)
    ax.set_ylabel(r"$Z(cc)$ / $W(cs)$  (shape-normalised)", fontsize=12)
    ax.set_xlim(*DISC_RANGE)
    ax.set_ylim(0, None)
    ax.legend(fontsize=11)
    hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=12)

    fname = outdir / f"disc_ratio_zcc_over_wcs_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Discriminant comparison plot for Z(cc) and W(cs)")
    parser.add_argument("--year",     required=True,
                        choices=["2022", "2022EE", "2023", "2023BPix", "2024"])
    parser.add_argument("--tag",      required=True,
                        help="Skim tag, e.g. 26Feb03")
    parser.add_argument("--outdir",   default="plots/disc",
                        help="Output directory for plots")
    parser.add_argument("--data-dir", default=None,
                        help="Override full EOS path to skim directory for this year")
    parser.add_argument("--pmap",     default="pmap_run3.json",
                        help="Path to the process-to-dataset map JSON")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else \
               Path(f"/eos/uscms/store/group/lpchbbrun3/skims/{args.tag}/{args.year}")
    print(f"Loading from: {data_dir}")

    with Path(args.pmap).open() as f:
        pmap = json.load(f)

    cols = COLS_BASE + COLS_PHOTON

    # Load all four processes
    events_dict: dict[str, pd.DataFrame] = {}
    for proc in list(PROCESSES.keys()):
        if proc not in pmap:
            print(f"[WARN] {proc} not in pmap — skipping")
            continue
        print(f"\n>>> Loading {proc} ...")
        loaded = utils.load_samples(
            data_dir=data_dir,
            samples={proc: pmap[proc]},
            columns=cols,
            region=REGION,
            variation=None,
            filters=PQ_FILTERS,
        )
        if loaded:
            events_dict[proc] = loaded[proc]
        else:
            print(f"  [WARN] No events loaded for {proc}")

    if not events_dict:
        print("ERROR: no events loaded. Check your --data-dir / --tag / --year.")
        return

    print("\n>>> Building histograms ...")
    hists = fill_group_hists(events_dict)

    # Print separation numbers
    sep_cur = separation(hists["zcc"]["current"],  hists["wcs"]["current"])
    sep_mod = separation(hists["zcc"]["modified"], hists["wcs"]["modified"])
    print(f"\n  Separation  current    : {sep_cur:.4f}")
    print(f"  Separation  modified   : {sep_mod:.4f}")
    delta = sep_mod - sep_cur
    print(f"  Delta (mod - cur)      : {delta:+.4f}  "
          f"({'improvement' if delta > 0 else 'worse'})")

    print("\n>>> Making plots ...")
    make_comparison_plot(hists, args.year, outdir)
    make_ratio_plot(hists, args.year, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
