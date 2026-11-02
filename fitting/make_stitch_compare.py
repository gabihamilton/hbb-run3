#!/usr/bin/env python3
"""
control-zgamma (bb-side Zgamma) BEFORE/AFTER gen-pT stitching comparison — MC only.

Three curves per variable, to make the stitching story unambiguous:
  - STACKED (before)     : sum of ZGto2QG-1Jets (fully inclusive) + all 4 Bin-PTG-N,
                           stacked as the buggy pipeline did = multi-counted.        [red fill]
  - STITCHED (after)     : AN-23-206 App.H — drop ZGto2QG-1Jets, keep each Bin-PTG-N
                           cut to its EXCLUSIVE gen-photon-pT window.                 [black line]
  - INCLUSIVE-ALONE (ref): the single fully-inclusive ZGto2QG-1Jets, correct norm.   [blue dashed]
The stitched sum should ~overlay the inclusive-alone reference (validates the method),
while the stacked version sits ~2x higher (the bug). Ratio panel = curve / inclusive-alone.

Windows + column name are imported-equivalent to template_utils (_PTG_EDGES, GenGammaPt).

Usage (from fitting/, hbb env):
  python make_stitch_compare.py --year 2024 --tag 24Aug07_v15 \
      --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024 \
      --outdir plots/stitch_compare/ [--bb-only]
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from hbb import utils
from hbb.common_vars import LUMI

hep.style.use("CMS")

# Same windows as template_utils._PTG_EDGES (AN-23-206 App.H)
PTG_EDGES = {"100": (100.0, 200.0), "200": (200.0, 400.0),
             "400": (400.0, 600.0), "600": (600.0, np.inf)}
INCLUSIVE = "ZGto2QG-1Jets"
BINNED = {"100": "ZGto2QG-1Jets_Bin-PTG-100", "200": "ZGto2QG-1Jets_Bin-PTG-200",
          "400": "ZGto2QG-1Jets_Bin-PTG-400", "600": "ZGto2QG-1Jets_Bin-PTG-600"}
ALL_SAMPLES = [INCLUSIVE] + list(BINNED.values())

COLS = ["weight", "GenFlavor", "Photon0_pt", "FatJet0_pt", "FatJet0_msd",
        "FatJet0_ParTmassX2p", "GenGammaPt", "FatJet0_eta", "FatJet0_phi",
        "Photon0_eta", "Photon0_phi", "MET", "Photon200", "Photon110EB_TightID_TightIso"]

# control-zgamma region pre-selection (matches template_utils.py fit templates)
PT_MIN = 250.0  # setup pt_min_scale

VARS = [
    ("Photon0_pt", np.linspace(100, 700, 31), r"Photon $p_T$ [GeV]"),
    ("FatJet0_msd", np.linspace(40, 200, 33), r"FatJet $m_{SD}$ [GeV]"),
    ("FatJet0_ParTmassX2p", np.linspace(40, 220, 37), r"Regressed mass [GeV]"),
    ("FatJet0_pt", np.linspace(250, 850, 31), r"FatJet $p_T$ [GeV]"),
    ("dR_photon_fatjet", np.linspace(0, 5, 26), r"$\Delta R(\gamma,\,\mathrm{AK8})$"),
]


def _add_dr(df):
    deta = df["Photon0_eta"].astype(float) - df["FatJet0_eta"].astype(float)
    dphi = np.abs(df["Photon0_phi"].astype(float) - df["FatJet0_phi"].astype(float))
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    df["dR_photon_fatjet"] = np.sqrt(deta**2 + dphi**2)
    df["_dphi_gj"] = dphi
    return df


def _region_mask(df):
    """control-zgamma pre-selection, same as template_utils (topo + trigger + pt>250).
    NOT applying the mSD observable window, so shapes are visible."""
    met = df["MET"].astype(float).values
    dphi = df["_dphi_gj"].values
    trig = (df["Photon200"].astype(bool).values
            | df["Photon110EB_TightID_TightIso"].astype(bool).values)
    return ((dphi > 2.2) & (met < 50.0) & (df["Photon0_pt"].astype(float).values > 120.0)
            & trig & (df["FatJet0_pt"].astype(float).values > PT_MIN))


def _hfill(vals, w, bins, xlabel):
    h = hist.Hist(hist.axis.Variable(bins, label=xlabel), storage=hist.storage.Weight())
    h.fill(np.asarray(vals, dtype=float), weight=np.asarray(w, dtype=float))
    return h


def _empty(bins, xlabel):
    return hist.Hist(hist.axis.Variable(bins, label=xlabel), storage=hist.storage.Weight())


def make_plot(accum, var, bins, xlabel, year, outpath, sub):
    """accum: dict role -> filled hist.Hist for this (var, flavor)."""
    plt.rcParams.update({"font.size": 20})
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True)

    h_stack = accum.get("stack")
    h_stitch = accum.get("stitch")
    h_incl = accum.get("incl")

    if h_stack is None:
        plt.close(fig)
        return

    def integ(h):
        return h.values().sum() if h is not None else 0.0

    hep.histplot(h_stack, ax=ax, histtype="fill", color="#E74C3C", alpha=0.55,
                 label=f"Stacked (before, bug)  [{integ(h_stack):.0f}]")
    if h_incl is not None:
        hep.histplot(h_incl, ax=ax, histtype="step", color="#2171B5", linestyle="--",
                     linewidth=2, label=f"Inclusive-alone (ref)  [{integ(h_incl):.0f}]")
    if h_stitch is not None:
        hep.histplot(h_stitch, ax=ax, histtype="step", color="black", linewidth=2.5,
                     label=f"Stitched (after, AN)  [{integ(h_stitch):.0f}]")

    ax.set_ylabel("Events / bin")
    ax.set_xlabel("")  # avoid duplicate label over the ratio panel (sharex)
    ax.set_yscale("log")
    ax.legend(fontsize="small", loc="upper right", framealpha=0.9)
    ax.text(0.04, 0.93, f"control-z$\\gamma$ presel\nZGto2QG · {sub}",
            transform=ax.transAxes, fontsize=13, va="top")
    lumi_val = round(LUMI.get(year, 0) / 1000.0, 2)
    hep.cms.label(ax=ax, data=False, lumi=lumi_val, year=year, com=13.6)

    # ratio panel: curve / inclusive-alone
    ref = h_incl.values() if h_incl is not None else None
    centers = 0.5 * (bins[1:] + bins[:-1])
    if ref is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            r_stack = np.where(ref > 0, h_stack.values() / ref, np.nan)
            rax.step(centers, r_stack, where="mid", color="#E74C3C", label="stacked/incl")
            if h_stitch is not None:
                r_st = np.where(ref > 0, h_stitch.values() / ref, np.nan)
                rax.step(centers, r_st, where="mid", color="black", label="stitched/incl")
        rax.axhline(1.0, ls=":", color="#2171B5")
    rax.set_ylabel("/ incl", fontsize=13)
    rax.set_xlabel(xlabel)
    rax.set_ylim(0, 3.0)
    rax.legend(fontsize="xx-small", loc="upper right", ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}")


def main(args):
    data_dir = Path(args.data_dir) if args.data_dir else Path(
        f"/eos/uscms/store/group/lpchbbrun3/gmachado/{args.tag}/{args.year}")
    outdir = Path(args.outdir)
    region = "control-zgamma"
    print(f"\n=== stitch before/after: {args.year} [{data_dir}] ===\n")

    # Accumulators: flavor -> var -> role -> hist.Hist. Filled one dataset at a time
    # so only a single DataFrame is ever resident (the inclusive sample is ~10M rows).
    # GenFlavor coding (template_utils flavor_cuts): bb=3, cc(=c)=2.
    FLAVORS = {"_allflav": ("all flavor", None), "_bb": ("bb (signal)", 3),
               "_cc": ("cc (signal)", 2)}
    acc = {tag: {v[0]: {r: _empty(v[1], v[2]) for r in ("stack", "stitch", "incl")}
                 for v in VARS} for tag in FLAVORS}
    loaded_any = False

    for name in ALL_SAMPLES:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loaded = utils.load_samples(
                data_dir=data_dir, samples={name: [name]}, columns=COLS, region=region)
        if not (loaded and name in loaded and not loaded[name].empty):
            print(f"  [skip] {name}: no parquets")
            continue
        df = _add_dr(loaded[name])
        del loaded
        loaded_any = True
        print(f"  Loaded {name}: {len(df):,} events")
        ptg = None
        for k, v in BINNED.items():
            if v == name:
                ptg = k
        gwin = df["GenGammaPt"].astype(float).values if ptg else None
        w_all = df["finalWeight"].astype(float).values
        regmask = _region_mask(df)
        genflav = df["GenFlavor"].values
        for tag, (_lbl, fval) in FLAVORS.items():
            m = regmask if fval is None else (regmask & (genflav == fval))
            for var, bins, _xl in VARS:
                vals = df[var].values
                # stack: every dataset, no gen cut
                acc[tag][var]["stack"].fill(vals[m].astype(float), weight=w_all[m])
                if ptg is not None:  # stitch: binned only, gen window
                    lo, hi = PTG_EDGES[ptg]
                    ms = m & (gwin >= lo) & (gwin < hi)
                    acc[tag][var]["stitch"].fill(vals[ms].astype(float), weight=w_all[ms])
                if name == INCLUSIVE:  # reference
                    acc[tag][var]["incl"].fill(vals[m].astype(float), weight=w_all[m])
        del df

    if not loaded_any:
        print("ERROR: nothing loaded.")
        return

    for tag, (lbl, _fval) in FLAVORS.items():
        for var, bins, xlabel in VARS:
            make_plot(acc[tag][var], var, bins, xlabel, args.year,
                      outdir / args.year / f"stitch_compare_{var}{tag}.png", lbl)
    print(f"\nDone. Plots in: {outdir / args.year}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--year", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--outdir", default="plots/stitch_compare/")
    p.add_argument("--bb-only", action="store_true",
                   help="Restrict to GenFlavor==bb (the zgammabb signal template)")
    main(p.parse_args())
