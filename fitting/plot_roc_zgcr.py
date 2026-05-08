#!/usr/bin/env python3
"""
plot_roc_zgcr.py
----------------
ROC curves for the current and modified TXbbXcc discriminants in the zgamma CR.

Compares:
  Current:  TXbbXcc     = (Xbb+Xcc) / (Xbb+Xcc+QCD)
  Modified: TXbbXcc_Wcs = (Xbb+Xcc) / (Xbb+Xcc+QCD+Xcs)

Three signal vs background scenarios:
  1. Z(cc) vs QCD
  2. Z(cc) vs QCD+W(cs)
  3. Z(bb) vs QCD

For each scenario the ROC curve is plotted for both discriminants,
and the working point WP=0.82 on the current discriminant is marked
with a vertical dotted line — the corresponding point on the modified
discriminant is also marked so we can read off the equivalent threshold.

Usage
-----
    python plot_roc_zgcr.py \\
        --year 2024 \\
        --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/Test_v15/2024 \\
        --outdir plots/roc

    # All years at once:
    for year in 2022 2022EE 2023 2023BPix 2024; do
        python plot_roc_zgcr.py --year $year \\
            --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/Test_v15_v14_private/$year \\
            --outdir plots/roc
    done
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

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hbb import utils

plt.style.use(hep.style.CMS)

# ---------------------------------------------------------------------------
# Constants — shared with plot_disc_zgcr.py
# ---------------------------------------------------------------------------

REGION          = "control-zgamma"
GENFLAVOR_BB    = 1
GENFLAVOR_CHARM = 2
WORKING_POINT   = 0.82     # current discriminant WP

PQ_FILTERS = [
    ("FatJet0_msd", ">=", 40.0),
    ("FatJet0_msd", "<=", 205.0),
    ("FatJet0_pt",  ">=", 250.0),
    ("Photon0_pt",  ">=", 120.0),
]

COLS_BASE = [
    "weight",
    "FatJet0_pt", "FatJet0_msd", "FatJet0_phi",
    "FatJet0_ParTPXbbXcc",
    "FatJet0_ParTPXbb", "FatJet0_ParTPXcc",
    "FatJet0_ParTPQCD", "FatJet0_ParTPXcs",
    "GenFlavor",
]
COLS_PHOTON = [
    "Photon0_pt", "Photon0_phi", "MET",
    "Photon200", "Photon110EB_TightID_TightIso",
]

PROCS_WITH_GENFLAVOR = {"Zgamma", "Zjets", "Wgamma", "Wjets"}

# Groups to load — same structure as plot_disc_zgcr.py
GROUPS = {
    "zcc": {"procs": ["Zgamma", "Zjets"], "gfilt": GENFLAVOR_CHARM},
    "zbb": {"procs": ["Zgamma", "Zjets"], "gfilt": GENFLAVOR_BB},
    "wcs": {"procs": ["Wgamma", "Wjets"], "gfilt": GENFLAVOR_CHARM},
    "qcd": {"procs": ["GJets"],           "gfilt": None},
}

PQ_FILTERS_EXTRA = {
    "zcc": [("GenFlavor", "==", GENFLAVOR_CHARM)],
    "zbb": [("GenFlavor", "==", GENFLAVOR_BB)],
    "wcs": [("GenFlavor", "==", GENFLAVOR_CHARM)],
    "qcd": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def apply_preselection(df: pd.DataFrame) -> pd.Series:
    dphi_raw = np.abs(df["Photon0_phi"] - df["FatJet0_phi"])
    dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)
    met_col = df["MET"] if "MET" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
    met_pt = met_col.pt if hasattr(met_col, "pt") else met_col
    trigger = df["Photon200"].astype(bool) | df["Photon110EB_TightID_TightIso"].astype(bool)
    return (
        (df["FatJet0_msd"] > 40) & (df["FatJet0_msd"] < 201) &
        (df["FatJet0_pt"]  > 250) &
        (df["Photon0_pt"]  > 120) &
        (dphi > 2.2) & (met_pt < 50) & trigger
    )


def compute_disc_modified(df: pd.DataFrame) -> np.ndarray:
    num = df["FatJet0_ParTPXbb"] + df["FatJet0_ParTPXcc"]
    den = (num + df["FatJet0_ParTPQCD"] + df["FatJet0_ParTPXcs"]).replace(0, np.nan)
    return (num / den).fillna(0.0).values


def roc_curve(scores_sig: np.ndarray, weights_sig: np.ndarray,
              scores_bkg: np.ndarray, weights_bkg: np.ndarray,
              n_points: int = 500) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weighted ROC curve.
    Returns (fpr, tpr, thresholds) where fpr = background efficiency,
    tpr = signal efficiency.
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    tpr = np.array([np.sum(weights_sig[scores_sig > t]) / np.sum(weights_sig)
                    for t in thresholds])
    fpr = np.array([np.sum(weights_bkg[scores_bkg > t]) / np.sum(weights_bkg)
                    for t in thresholds])
    return fpr, tpr, thresholds


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Trapezoidal AUC."""
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def find_threshold_at_fpr(scores: np.ndarray, weights: np.ndarray,
                           target_fpr: float, n_points: int = 1000) -> float:
    """Find the score threshold that gives a specific background efficiency."""
    thresholds = np.linspace(0.0, 1.0, n_points)
    fprs = np.array([np.sum(weights[scores > t]) / np.sum(weights) for t in thresholds])
    # interpolate
    idx = np.searchsorted(-fprs, -target_fpr)
    idx = np.clip(idx, 1, len(thresholds) - 1)
    return float(thresholds[idx])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all_groups(data_dir: Path, pmap: dict, year: str) -> dict[str, pd.DataFrame]:
    """Returns events_dict keyed by 'group:proc'."""
    cols_with_gf    = COLS_BASE + COLS_PHOTON
    cols_without_gf = [c for c in COLS_BASE if c != "GenFlavor"] + COLS_PHOTON

    events_dict: dict[str, pd.DataFrame] = {}
    for grp_name, grp_cfg in GROUPS.items():
        extra = PQ_FILTERS_EXTRA.get(grp_name)
        filters = PQ_FILTERS + extra if extra else PQ_FILTERS
        has_gf = grp_cfg["gfilt"] is not None
        cols = cols_with_gf if has_gf else cols_without_gf

        for proc in grp_cfg["procs"]:
            key = f"{grp_name}:{proc}"
            if proc not in pmap:
                print(f"[WARN] {proc} not in pmap")
                continue
            print(f"\n>>> Loading {proc} for [{grp_name}] ...")
            loaded = utils.load_samples(
                data_dir=data_dir,
                samples={proc: pmap[proc]},
                columns=cols,
                region=REGION,
                variation=None,
                filters=filters,
            )
            if loaded:
                events_dict[key] = loaded[proc]
    return events_dict


def collect_group_arrays(events_dict: dict, grp_name: str,
                         gfilt: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (disc_current, disc_modified, weights) arrays for a group,
    concatenating over all its processes.
    """
    grp_procs = GROUPS[grp_name]["procs"]
    cur_list, mod_list, w_list = [], [], []

    for proc in grp_procs:
        key = f"{grp_name}:{proc}"
        if key not in events_dict:
            continue
        df = events_dict[key]
        sel = apply_preselection(df)
        if gfilt is not None and "GenFlavor" in df.columns:
            sel = sel & (df["GenFlavor"] == gfilt)
        df = df[sel]
        if len(df) == 0:
            continue
        cur_list.append(df["FatJet0_ParTPXbbXcc"].values)
        mod_list.append(compute_disc_modified(df))
        w_list.append(df["finalWeight"].values)

    if not cur_list:
        return np.array([]), np.array([]), np.array([])

    return (np.concatenate(cur_list),
            np.concatenate(mod_list),
            np.concatenate(w_list))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_roc_plot(events_dict: dict, year: str, outdir: Path) -> None:
    """
    Three-panel ROC plot:
      1. Z(cc) vs QCD
      2. Z(cc) vs QCD + W(cs)
      3. Z(bb) vs QCD
    Each panel shows both current and modified discriminants.
    The WP=0.82 operating point is marked.
    """
    # Collect arrays per group
    arrays = {}
    for grp_name, grp_cfg in GROUPS.items():
        cur, mod, w = collect_group_arrays(events_dict, grp_name, grp_cfg["gfilt"])
        arrays[grp_name] = {"cur": cur, "mod": mod, "w": w}

    # Background efficiency at WP=0.82 on current discriminant (using QCD as reference)
    qcd_cur = arrays["qcd"]["cur"]
    qcd_w   = arrays["qcd"]["w"]
    if len(qcd_w) > 0:
        wp_fpr = float(np.sum(qcd_w[qcd_cur > WORKING_POINT]) / np.sum(qcd_w))
    else:
        wp_fpr = None

    scenarios = [
        ("zcc", "qcd",  None,     r"$Z(cc)$ vs QCD"),
        ("zcc", "wcs",  "qcd",    r"$Z(cc)$ vs QCD + $W(cs)$"),
        ("zbb", "qcd",  None,     r"$Z(bb)$ vs QCD"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.subplots_adjust(wspace=0.3)

    styles = {
        "cur": dict(color="navy",    ls="-",  lw=2, label=f"Current  (AUC={{:.3f}})"),
        "mod": dict(color="darkred", ls="--", lw=2, label=f"Modified (AUC={{:.3f}})"),
    }

    for ax, (sig_grp, bkg_grp, extra_bkg, title) in zip(axes, scenarios):
        sig  = arrays[sig_grp]

        # Build background arrays (optionally merge two background groups)
        if extra_bkg:
            bkg_cur = np.concatenate([arrays[bkg_grp]["cur"], arrays[extra_bkg]["cur"]])
            bkg_mod = np.concatenate([arrays[bkg_grp]["mod"], arrays[extra_bkg]["mod"]])
            bkg_w   = np.concatenate([arrays[bkg_grp]["w"],   arrays[extra_bkg]["w"]])
        else:
            bkg_cur = arrays[bkg_grp]["cur"]
            bkg_mod = arrays[bkg_grp]["mod"]
            bkg_w   = arrays[bkg_grp]["w"]

        if len(sig["w"]) == 0 or len(bkg_w) == 0:
            ax.set_title(f"{title}\n(no events)")
            continue

        for disc_key, sig_scores, bkg_scores in [
            ("cur", sig["cur"], bkg_cur),
            ("mod", sig["mod"], bkg_mod),
        ]:
            fpr, tpr, thresholds = roc_curve(sig_scores, sig["w"], bkg_scores, bkg_w)
            area = auc(fpr, tpr)
            sty = dict(styles[disc_key])
            sty["label"] = sty["label"].format(area)
            ax.plot(fpr, tpr, **sty)

            # Mark operating point
            if wp_fpr is not None:
                # For current: WP=0.82 directly
                # For modified: find threshold giving same background efficiency
                if disc_key == "cur":
                    wp_th = WORKING_POINT
                else:
                    wp_th = find_threshold_at_fpr(bkg_scores, bkg_w, wp_fpr)
                wp_sig_eff = float(np.sum(sig["w"][sig_scores > wp_th]) / np.sum(sig["w"]))
                wp_bkg_eff = float(np.sum(bkg_w[bkg_scores > wp_th]) / np.sum(bkg_w))
                ax.scatter([wp_bkg_eff], [wp_sig_eff],
                           color=sty["color"], s=80, zorder=5,
                           label=f"WP={'curr' if disc_key=='cur' else 'equiv'}: "
                                 f"th={wp_th:.3f}, sig={wp_sig_eff:.2f}, bkg={wp_bkg_eff:.2f}")

        ax.plot([0, 1], [0, 1], color="gray", ls=":", lw=1)
        ax.set_xlabel("Background efficiency (FPR)", fontsize=11)
        ax.set_ylabel("Signal efficiency (TPR)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc="lower right")
        hep.cms.label("Preliminary", data=False, ax=ax, year=year, fontsize=10)

    fname = outdir / f"roc_{year}.png"
    fig.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {fname}")


def print_equivalent_wps(events_dict: dict, year: str) -> None:
    """
    Print the threshold on the modified discriminant that gives the same
    background efficiency as WP=0.82 on the current discriminant,
    for each background scenario.
    """
    print(f"\n{'='*60}")
    print(f"  Equivalent working points for {year}")
    print(f"  Current WP = {WORKING_POINT:.2f} on TXbbXcc")
    print(f"{'='*60}")

    qcd_cur = np.concatenate([arrays["cur"] for grp, arrays in
                              {g: collect_group_arrays(events_dict, g, GROUPS[g]["gfilt"])
                               for g in ["qcd"]}.items()
                              for arrays in [{"cur": arrays[0], "w": arrays[2]}]
                              if len(arrays[2]) > 0])

    # Recollect cleanly
    arrays = {}
    for grp_name, grp_cfg in GROUPS.items():
        cur, mod, w = collect_group_arrays(events_dict, grp_name, grp_cfg["gfilt"])
        arrays[grp_name] = {"cur": cur, "mod": mod, "w": w}

    qcd_cur = arrays["qcd"]["cur"]
    qcd_w   = arrays["qcd"]["w"]
    if len(qcd_w) == 0:
        print("  No QCD events — cannot compute equivalent WP")
        return

    wp_fpr = float(np.sum(qcd_w[qcd_cur > WORKING_POINT]) / np.sum(qcd_w))
    print(f"  Background eff. at WP=0.82 (QCD): {wp_fpr:.4f}")

    for bkg_grp, extra in [("qcd", None), ("qcd", "wcs")]:
        if extra:
            bkg_mod = np.concatenate([arrays[bkg_grp]["mod"], arrays[extra]["mod"]])
            bkg_w   = np.concatenate([arrays[bkg_grp]["w"],   arrays[extra]["w"]])
            label = "QCD + W(cs)"
        else:
            bkg_mod = arrays[bkg_grp]["mod"]
            bkg_w   = arrays[bkg_grp]["w"]
            label = "QCD"

        if len(bkg_w) == 0:
            continue
        equiv_th = find_threshold_at_fpr(bkg_mod, bkg_w, wp_fpr)
        print(f"  Equiv. WP on modified disc ({label}): {equiv_th:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="ROC curves for current vs modified TXbbXcc")
    parser.add_argument("--year",     required=True,
                        choices=["2022", "2022EE", "2023", "2023BPix", "2024"])
    parser.add_argument("--data-dir", required=True,
                        help="Full path to skim directory for this year")
    parser.add_argument("--outdir",   default="plots/roc")
    parser.add_argument("--pmap",     default="pmap_run3.json")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.pmap) as f:
        pmap = json.load(f)

    print(f"Loading from: {args.data_dir}")
    events_dict = load_all_groups(Path(args.data_dir), pmap, args.year)

    if not events_dict:
        print("ERROR: no events loaded.")
        return

    print("\n>>> Making ROC plots ...")
    make_roc_plot(events_dict, args.year, outdir)
    print_equivalent_wps(events_dict, args.year)
    print("\nDone.")


if __name__ == "__main__":
    main()
