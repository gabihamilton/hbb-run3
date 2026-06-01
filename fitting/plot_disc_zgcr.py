#!/usr/bin/env python3
"""
plot_disc_zgcr.py
-----------------
Plot the modified TXbbXcc discriminant in the zgamma CR pre-selection region
(inclusive, before pass/fail split) for Z(cc), Z(bb), W(cs), and QCD processes.

Two discriminants are compared side by side:
  1. Current:  TXbbXcc       = (Xbb + Xcc) / (Xbb + Xcc + QCD)
  2. Modified: TXbbXcc_Wcs   = (Xbb + Xcc) / (Xbb + Xcc + QCD + Xcs)

Z(cc)  = Zgamma + Zjets  with generator-level charm flavor (GenFlavor == 2)
Z(bb)  = Zgamma + Zjets  with generator-level bb flavor    (GenFlavor == 3)
W(cs)  = Wgamma + Wjets  with generator-level charm flavor (GenFlavor == 2)
QCD    = GJets            (no GenFlavor cut)

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
        --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/Test_v15/2024 \\
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
GENFLAVOR_BB    = 3          # GenFlavor == 3 → bb decay
GENFLAVOR_CHARM = 2          # GenFlavor == 2 → c / cs decay
NBINS = 40
DISC_RANGE = (0.0, 1.0)

# Groups: each has a list of processes, an optional GenFlavor cut, and plot style.
# gfilt=None means no GenFlavor cut (used for QCD).
GROUPS = {
    "zcc": {
        "procs":  ["Zgamma", "Zjets"],
        "gfilt":  GENFLAVOR_CHARM,
        "label":  r"$Z(cc)$  [Zgamma + Zjets]",
        "color":  "royalblue",
        "hatch":  None,
    },
    "zbb": {
        "procs":  ["Zgamma", "Zjets"],
        "gfilt":  GENFLAVOR_BB,
        "label":  r"$Z(bb)$  [Zgamma + Zjets]",
        "color":  "navy",
        "hatch":  "...",
    },
    "wcs": {
        "procs":  ["Wgamma", "Wjets"],
        "gfilt":  GENFLAVOR_CHARM,
        "label":  r"$W(cs)$  [Wgamma + Wjets]",
        "color":  "tomato",
        "hatch":  "///",
    },
    "qcd": {
        "procs":  ["GJets"],
        "gfilt":  None,
        "label":  r"QCD  [GJets]",
        "color":  "forestgreen",
        "hatch":  "xxx",
    },
}

# Processes that carry a GenFlavor column in the parquet
PROCS_WITH_GENFLAVOR = {"Zgamma", "Zjets", "Wgamma", "Wjets"}

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

# PyArrow pre-filters — match actual preselection cuts as closely as possible
# to minimise memory footprint (especially for large high-pT bins like PTG-600)
PQ_FILTERS = [
    ("FatJet0_msd", ">=", 40.0),
    ("FatJet0_msd", "<=", 205.0),
    ("FatJet0_pt",  ">=", 250.0),
    ("Photon0_pt",  ">=", 120.0),
]

# Extra PyArrow filters per group.
# NOTE: GenFlavor integer-column filters are intentionally NOT applied here —
# PyArrow predicate pushdown on integer columns can silently return empty
# results due to type-matching quirks (GenFlavor==3 is particularly affected).
# GenFlavor selection is done in-memory in fill_group_hists instead.
PQ_FILTERS_EXTRA: dict[str, list | None] = {
    "zcc": None,
    "zbb": None,
    "wcs": None,
    "qcd": None,
}


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
    Build weighted histograms for both discriminants, per group.

    Each group's gfilt determines the GenFlavor cut:
      - int  → keep only events with GenFlavor == gfilt
      - None → no GenFlavor cut (used for QCD)

    Returns
    -------
    {group_name: {"current": values, "modified": values, "edges": edges,
                  "sumw_current": float, "sumw_modified": float}}
    where sumw_* is the total weighted yield (= expected events for the full
    discriminant range after preselection).
    """
    edges = np.linspace(disc_range[0], disc_range[1], nbins + 1)
    result = {
        g: {"current": np.zeros(nbins), "modified": np.zeros(nbins),
            "edges": edges, "sumw_current": 0.0, "sumw_modified": 0.0}
        for g in GROUPS
    }

    for grp_name, grp_cfg in GROUPS.items():
        gfilt = grp_cfg.get("gfilt")
        for proc in grp_cfg["procs"]:
            key = f"{grp_name}:{proc}"
            if key not in events_dict:
                print(f"  [WARN] {key!r} not found — skipping")
                continue
            df = events_dict[key]

            sel = apply_preselection(df)

            # GenFlavor filtering done fully in Python — pyarrow pushdown on
            # integer columns can silently return empty results.
            if gfilt is not None and "GenFlavor" in df.columns:
                sel = sel & (df["GenFlavor"] == gfilt)

            df = df[sel]

            if len(df) == 0:
                print(f"  [WARN] {proc} ({grp_name}): 0 events after selection")
                continue

            w = df["finalWeight"].values
            disc_cur = df["FatJet0_ParTPXbbXcc"].values
            disc_mod = compute_disc_modified(df).values

            h_cur, _ = np.histogram(disc_cur, bins=edges, weights=w)
            h_mod, _ = np.histogram(disc_mod, bins=edges, weights=w)

            result[grp_name]["current"]        += h_cur
            result[grp_name]["modified"]        += h_mod
            result[grp_name]["sumw_current"]   += w.sum()
            result[grp_name]["sumw_modified"]  += w.sum()
            print(f"  {proc} ({grp_name}): {len(df)} events, sumw = {w.sum():.3g}")

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
    Two-panel figure showing all four groups:
      left  — current TXbbXcc = (Xbb+Xcc)/(Xbb+Xcc+QCD)
      right — modified TXbbXcc_Wcs = (Xbb+Xcc)/(Xbb+Xcc+QCD+Xcs)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
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
                alpha=0.45,
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

        sep_zcc_wcs = separation(hists["zcc"][dkey], hists["wcs"][dkey])
        sep_zbb_qcd = separation(hists["zbb"][dkey], hists["qcd"][dkey])
        ax.text(
            0.04, 0.97,
            f"Sep. Z(cc)/W(cs) = {sep_zcc_wcs:.3f}\nSep. Z(bb)/QCD  = {sep_zbb_qcd:.3f}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, color="black",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7),
        )

        ax.set_title(dtitle, fontsize=12, pad=8)
        ax.set_xlabel("Discriminant value", fontsize=12)
        ax.set_ylabel("Normalised events / bin", fontsize=12)
        ax.set_xlim(*DISC_RANGE)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=9, loc="upper center")
        hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=12)

    fname = outdir / f"disc_all_groups_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {fname}")


def make_ratio_plot(
    hists: dict,
    year: str,
    outdir: Path,
) -> None:
    """
    Two-panel ratio plot:
      left  — Z(cc) / W(cs) for current and modified discriminants
      right — Z(bb) / QCD   for current and modified discriminants
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.32)

    ratio_pairs = [
        ("zcc", "wcs", r"$Z(cc)$ / $W(cs)$  (shape-normalised)"),
        ("zbb", "qcd", r"$Z(bb)$ / QCD  (shape-normalised)"),
    ]
    styles = {
        "current":  dict(color="navy",    ls="-",  lw=2, label=r"Current $T_{Xbb+Xcc}$"),
        "modified": dict(color="darkred", ls="--", lw=2, label=r"Modified $T_{Xbb+Xcc}^{Wcs}$"),
    }

    for ax, (num_grp, den_grp, ylabel) in zip(axes, ratio_pairs):
        edges = hists[num_grp]["edges"]
        for dkey, sty in styles.items():
            h_num = norm_hist(hists[num_grp][dkey])
            h_den = norm_hist(hists[den_grp][dkey])
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(h_den > 0, h_num / h_den, np.nan)
            ax.step(edges, np.append(ratio, ratio[-1]), where="post", **sty)

        ax.axhline(1.0, color="gray", ls=":", lw=1)
        ax.set_xlabel("Discriminant value", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlim(*DISC_RANGE)
        ax.set_ylim(0, None)
        ax.legend(fontsize=11)
        hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=12)

    fname = outdir / f"disc_ratios_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


def make_yield_plot(
    hists: dict,
    year: str,
    outdir: Path,
    working_point: float = 0.82,
) -> None:
    """
    Two-panel figure showing absolute expected event yields (sumW per bin),
    not shape-normalised.  Shows the actual event-count scales so you can
    see, e.g., how many Z(cc) events survive vs W(cs) background.

    Left panel  — current discriminant
    Right panel — modified discriminant

    Each group's legend entry includes its total yield and the yield above
    the WP=0.82 threshold (so you can read off pass-region counts directly).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    disc_keys   = ["current",  "modified"]
    disc_titles = [
        r"$T_{Xbb+Xcc}$  (current)",
        r"$T_{Xbb+Xcc}^{\,Wcs}$  (modified)",
    ]

    for ax, dkey, dtitle in zip(axes, disc_keys, disc_titles):
        edges   = hists["zcc"]["edges"]
        centres = 0.5 * (edges[:-1] + edges[1:])
        width   = edges[1] - edges[0]

        for grp_name, grp_cfg in GROUPS.items():
            h = hists[grp_name][dkey]
            total_yield = h.sum()
            # yield above working point
            wp_mask     = centres >= working_point
            pass_yield  = h[wp_mask].sum()
            label = (
                f"{grp_cfg['label'].split('[')[0].strip()}  "
                f"[total={total_yield:.1f},  WP>{working_point}={pass_yield:.1f}]"
            )
            ax.bar(
                centres, h,
                width=width,
                color=grp_cfg["color"],
                alpha=0.40,
                hatch=grp_cfg["hatch"],
                label=label,
                edgecolor="none",
            )
            ax.step(
                edges, np.append(h, h[-1]),
                where="post",
                color=grp_cfg["color"],
                linewidth=1.5,
            )

        # Working-point line
        ax.axvline(working_point, color="black", ls="--", lw=1.5,
                   label=f"WP = {working_point}")

        ax.set_title(dtitle, fontsize=13, pad=8)
        ax.set_xlabel("Discriminant value", fontsize=12)
        ax.set_ylabel("Expected events / bin", fontsize=12)
        ax.set_xlim(*DISC_RANGE)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8, loc="upper center")
        hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=12)

    fname = outdir / f"disc_yields_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discriminant comparison plot for Z(cc), Z(bb), W(cs), QCD"
    )
    parser.add_argument("--year",     required=True,
                        choices=["2022", "2022EE", "2023", "2023BPix", "2024"])
    parser.add_argument("--tag",      default=None,
                        help="Skim tag, e.g. 26May18 (used only if --data-dir not given)")
    parser.add_argument("--outdir",   default="plots/disc",
                        help="Output directory for plots")
    parser.add_argument("--data-dir", default=None,
                        help="Full path to skim directory for this year")
    parser.add_argument("--qcd-data-dir", default=None,
                        help="Fallback path for GJets (QCD) when primary skims don't "
                             "include it (e.g. /eos/.../Test_v15/<year>)")
    parser.add_argument("--max-events", type=int, default=500_000,
                        help="Max events per process (random subsample, weights "
                             "preserved). Set to 0 to disable. Default: 500000")
    parser.add_argument("--pmap",     default="pmap_run3.json",
                        help="Path to the process-to-dataset map JSON")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_dir = Path(args.data_dir)
    elif args.tag:
        data_dir = Path(f"/eos/uscms/store/group/lpchbbrun3/skims/{args.tag}/{args.year}")
    else:
        raise ValueError("Provide either --data-dir or --tag")

    qcd_dir  = Path(args.qcd_data_dir) if args.qcd_data_dir else None
    max_ev   = args.max_events if args.max_events > 0 else None
    print(f"Loading from: {data_dir}")
    if qcd_dir:
        print(f"QCD (GJets) from: {qcd_dir}")
    if max_ev:
        print(f"Max events per process: {max_ev:,}")

    with Path(args.pmap).open() as f:
        pmap = json.load(f)

    cols_with_gf    = COLS_BASE + COLS_PHOTON
    cols_without_gf = [c for c in COLS_BASE if c != "GenFlavor"] + COLS_PHOTON

    events_dict: dict[str, pd.DataFrame] = {}
    for grp_name, grp_cfg in GROUPS.items():
        extra = PQ_FILTERS_EXTRA.get(grp_name)
        filters = PQ_FILTERS + extra if extra else PQ_FILTERS
        has_gf = grp_cfg["gfilt"] is not None
        cols = cols_with_gf if has_gf else cols_without_gf

        src_dir = (qcd_dir if (grp_name == "qcd" and qcd_dir is not None)
                   else data_dir)

        for proc in grp_cfg["procs"]:
            key = f"{grp_name}:{proc}"
            if proc not in pmap:
                print(f"[WARN] {proc} not in pmap — skipping")
                continue
            gf_label = f"GenFlavor=={grp_cfg['gfilt']}" if grp_cfg["gfilt"] is not None else "no GF"
            print(f"\n>>> Loading {proc} for [{grp_name}] ({gf_label}) ...")
            loaded = utils.load_samples(
                data_dir=src_dir,
                samples={proc: pmap[proc]},
                columns=cols,
                region=REGION,
                variation=None,
                filters=filters,
            )
            if loaded:
                df = loaded[proc]
                if max_ev is not None and len(df) > max_ev:
                    print(f"    Subsampling {len(df):,} → {max_ev:,} rows")
                    df = df.sample(n=max_ev, random_state=42)
                events_dict[key] = df
            else:
                print(f"  [WARN] No events loaded for {proc} [{grp_name}]")

    if not events_dict:
        print("ERROR: no events loaded. Check your --data-dir / --tag / --year.")
        return

    print("\n>>> Building histograms ...")
    hists = fill_group_hists(events_dict)

    # Print separation numbers for both pairs
    for num, den in [("zcc", "wcs"), ("zbb", "qcd")]:
        sep_cur = separation(hists[num]["current"],  hists[den]["current"])
        sep_mod = separation(hists[num]["modified"], hists[den]["modified"])
        delta   = sep_mod - sep_cur
        print(f"\n  {num.upper()} vs {den.upper()}:")
        print(f"    Separation  current  : {sep_cur:.4f}")
        print(f"    Separation  modified : {sep_mod:.4f}")
        print(f"    Delta                : {delta:+.4f}  "
              f"({'improvement' if delta > 0 else 'worse'})")

    print("\n>>> Making plots ...")
    make_comparison_plot(hists, args.year, outdir)
    make_ratio_plot(hists, args.year, outdir)
    make_yield_plot(hists, args.year, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
