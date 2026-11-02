#!/usr/bin/env python3
"""
Stitching validation for the photon-pT-binned 2024 V+γ samples (AN-23-206 App.H), LHE version.

The Bin-PTG-{100,200,400,600} samples are INCLUSIVE-above (photon pT > threshold, no max), so each
one is cut to its exclusive LHE-photon-pT window before summing:
    [100,200)  [200,400)  [400,600)  [600,inf)   on  LHEGammaPt
Per family and variable, the plot shows
  - STITCHED: the 4 windowed bins stacked, one colour per PTG bin         [filled stack]
  - UNSTITCHED: the 4 bins summed with no window (the old multi-counting)  [red step]
  - INCLUSIVE: the fully-inclusive sample, where one exists               [black dashed]
Ratio panel = stitched / inclusive and unstitched / inclusive (or unstitched / stitched when
there is no inclusive sample, as for DYGto2LG).

The binned samples come from the LHE-photon reskim (they carry LHEGammaPt); the inclusive
references come from the base skim tag. Run from fitting/ in the hbb env:
  python make_stitch_lhe.py --year 2024 \
      --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024 \
      --lhe-data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/26Sep17_lhephoton_v15/2024 \
      --outdir plots/stitch_lhe/
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

PTG_EDGES = {"100": (100.0, 200.0), "200": (200.0, 400.0),
             "400": (400.0, 600.0), "600": (600.0, np.inf)}
PTG_COLORS = {"100": "#3f90da", "200": "#ffa90e", "400": "#bd1f01", "600": "#94a4a2"}
PTG_LABELS = {"100": r"PTG-100, $p_T^{\gamma,LHE}\in[100,200)$",
              "200": r"PTG-200, $p_T^{\gamma,LHE}\in[200,400)$",
              "400": r"PTG-400, $p_T^{\gamma,LHE}\in[400,600)$",
              "600": r"PTG-600, $p_T^{\gamma,LHE}\geq 600$"}

ZG_COLS = ["weight", "GenFlavor", "Photon0_pt", "Photon0_phi", "FatJet0_pt", "FatJet0_phi",
           "FatJet0_ParTmassX2p", "MET", "Photon200", "Photon110EB_TightID_TightIso"]
MM_COLS = ["weight", "GenFlavor", "Photon0_pt", "Zmm_MuonPair_mll", "Zmm_MuonPair_pt"]

FAMILIES = {
    "ZGto2QG": {
        "label": r"Z$\gamma$, Z$\to$qq (ZGto2QG)",
        "region": "control-zgamma",
        "inclusive": "ZGto2QG-1Jets",
        "binned": "ZGto2QG-1Jets_Bin-PTG-{}",
        "cols": ZG_COLS,
        "vars": [
            ("LHEGammaPt", np.linspace(0, 1000, 51), r"LHE photon $p_T$ [GeV]"),
            ("Photon0_pt", np.linspace(100, 1000, 46), r"Reco photon $p_T$ [GeV]"),
            ("FatJet0_ParTmassX2p", np.linspace(60, 220, 21), r"Jet regressed mass [GeV]"),
            ("FatJet0_pt", np.linspace(250, 1000, 31), r"Jet $p_T$ [GeV]"),
        ],
    },
    "WGto2QG": {
        "label": r"W$\gamma$, W$\to$qq (WGto2QG)",
        "region": "control-zgamma",
        "inclusive": "WGto2QG-1Jets",
        "binned": "WGto2QG-1Jets_Bin-PTG-{}",
        "cols": ZG_COLS,
        "vars": [
            ("LHEGammaPt", np.linspace(0, 1000, 51), r"LHE photon $p_T$ [GeV]"),
            ("Photon0_pt", np.linspace(100, 1000, 46), r"Reco photon $p_T$ [GeV]"),
            ("FatJet0_ParTmassX2p", np.linspace(60, 220, 21), r"Jet regressed mass [GeV]"),
            ("FatJet0_pt", np.linspace(250, 1000, 31), r"Jet $p_T$ [GeV]"),
        ],
    },
    "DYGto2LG": {
        "label": r"Z$\gamma$, Z$\to\ell\ell$ (DYGto2LG)",
        "region": "control-zmumu",
        "inclusive": None,  # no fully-inclusive DYGto2LG sample
        "binned": "DYGto2LG-1Jets_Bin-MLL-50-PTG-{}",
        "cols": MM_COLS,
        "vars": [
            ("LHEGammaPt", np.linspace(0, 1000, 51), r"LHE photon $p_T$ [GeV]"),
            ("Photon0_pt", np.linspace(0, 1000, 51), r"Reco photon $p_T$ [GeV]"),
            ("Zmm_MuonPair_mll", np.linspace(80, 100, 11), r"$m_{\mu\mu}$ [GeV]"),
            ("Zmm_MuonPair_pt", np.linspace(0, 1000, 41), r"$p_T^{\mu\mu}$ [GeV]"),
        ],
    },
}

PT_MIN = 250.0  # setup pt_min_scale


def _zgcr_presel(df):
    """control-zgamma fit pre-selection (template_utils.py): topo + trigger + jet pT."""
    dphi = np.abs(df["Photon0_phi"].astype(float).values - df["FatJet0_phi"].astype(float).values)
    dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
    met = df["MET"].astype(float).values
    trig = (df["Photon200"].astype(bool).values
            | df["Photon110EB_TightID_TightIso"].astype(bool).values)
    return ((dphi > 2.2) & (met < 50.0) & (df["Photon0_pt"].astype(float).values > 120.0)
            & trig & (df["FatJet0_pt"].astype(float).values > PT_MIN))


def _new(bins, xlabel):
    return hist.Hist(hist.axis.Variable(bins, label=xlabel), storage=hist.storage.Weight())


def _load(data_dir, dataset, cols, region):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loaded = utils.load_samples(data_dir=Path(data_dir), samples={dataset: [dataset]},
                                    columns=cols, region=region)
    if not (loaded and dataset in loaded and not loaded[dataset].empty):
        return None
    return loaded[dataset]


def _ratio(num, den):
    n, d = num.values(), den.values()
    nv = num.variances()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(d > 0, n / d, np.nan)
        e = np.where(d > 0, np.sqrt(nv) / d, np.nan)
    return r, e


def make_plot(h, var, bins, xlabel, fam, sel_label, year, outpath):
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10),
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.07},
                                  sharex=True)
    ptgs = list(PTG_EDGES)
    stitched = sum((h["stitch"][p] for p in ptgs[1:]), h["stitch"][ptgs[0]].copy())
    tot = stitched.values().sum()

    hep.histplot([h["stitch"][p] for p in ptgs], ax=ax, stack=True, histtype="fill",
                 color=[PTG_COLORS[p] for p in ptgs],
                 label=[f"{PTG_LABELS[p]}  [{h['stitch'][p].values().sum():.1f}]" for p in ptgs])
    hep.histplot(h["unstitched"], ax=ax, histtype="step", color="#d62728", linewidth=2,
                 label=f"Unstitched sum (no window)  [{h['unstitched'].values().sum():.1f}]")
    has_incl = h["incl"] is not None and h["incl"].values().sum() > 0
    if has_incl:
        hep.histplot(h["incl"], ax=ax, histtype="errorbar", color="black", marker="o",
                     markersize=5, label=f"Inclusive sample  [{h['incl'].values().sum():.1f}]")

    ax.set_yscale("log")
    ymax = max(h["unstitched"].values().max(), stitched.values().max())
    ax.set_ylim(max(1e-3, 1e-4 * ymax), 60 * ymax)
    ax.set_ylabel("Events / bin")
    ax.set_xlabel("")
    ax.legend(fontsize=13, loc="upper right", framealpha=0.9)
    ax.text(0.03, 0.97, f"{fam['label']}\n{sel_label}\nstitched total = {tot:.1f}",
            transform=ax.transAxes, fontsize=14, va="top")
    hep.cms.label(ax=ax, data=False, label="Simulation Private Work",
                  lumi=round(LUMI.get(year, 0) / 1000.0), year=year, com=13.6)

    if has_incl:
        r, e = _ratio(stitched, h["incl"])
        rax.errorbar(stitched.axes[0].centers, r, yerr=e, fmt="o", color="black", ms=4,
                     label="stitched / inclusive")
        ru, _ = _ratio(h["unstitched"], h["incl"])
        rax.step(stitched.axes[0].centers, ru, where="mid", color="#d62728",
                 label="unstitched / inclusive")
        rax.set_ylabel("/ inclusive", fontsize=16)
    else:
        ru, _ = _ratio(h["unstitched"], stitched)
        rax.step(stitched.axes[0].centers, ru, where="mid", color="#d62728",
                 label="unstitched / stitched")
        rax.set_ylabel("/ stitched", fontsize=16)
    rax.axhline(1.0, ls="--", color="gray")
    rax.set_ylim(0, 3)
    rax.set_xlabel(xlabel)
    rax.legend(fontsize=12, loc="upper right", ncol=2)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outpath.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}.png")


def run_family(name, fam, args):
    print(f"\n=== {name} ({fam['region']}) ===")
    sels = {"skim": f"{fam['region']} skim"}
    if fam["region"] == "control-zgamma":
        sels["presel"] = r"control-z$\gamma$ fit preselection"
    vars_ = fam["vars"]
    H = {s: {v: {"stitch": {p: _new(b, xl) for p in PTG_EDGES}, "unstitched": _new(b, xl),
                 "incl": _new(b, xl) if fam["inclusive"] else None}
             for v, b, xl in vars_}
         for s in sels}

    for ptg, (lo, hi) in PTG_EDGES.items():
        ds = fam["binned"].format(ptg)
        df = _load(args.lhe_data_dir, ds, fam["cols"] + ["LHEGammaPt"], fam["region"])
        if df is None:
            print(f"  [skip] {ds}: no parquets")
            continue
        w = df["finalWeight"].astype(float).values
        lhe = df["LHEGammaPt"].astype(float).values
        win = (lhe >= lo) & (lhe < hi)
        masks = {"skim": np.ones(len(df), bool)}
        if "presel" in sels:
            masks["presel"] = _zgcr_presel(df)
        print(f"  {ds}: {len(df):,} events, {win.sum():,} in LHE window [{lo:.0f},{hi:.0f})")
        for s, m in masks.items():
            for v, _b, _xl in vars_:
                x = lhe if v == "LHEGammaPt" else df[v].astype(float).values
                H[s][v]["unstitched"].fill(x[m], weight=w[m])
                H[s][v]["stitch"][ptg].fill(x[m & win], weight=w[m & win])
        del df

    if fam["inclusive"]:
        inc_vars = [v for v, _b, _xl in vars_ if v != "LHEGammaPt"]  # base skim has no LHEGammaPt
        df = _load(args.data_dir, fam["inclusive"], fam["cols"], fam["region"])
        if df is None:
            print(f"  [skip] {fam['inclusive']}: no parquets")
        else:
            print(f"  {fam['inclusive']}: {len(df):,} events (inclusive reference)")
            w = df["finalWeight"].astype(float).values
            masks = {"skim": np.ones(len(df), bool)}
            if "presel" in sels:
                masks["presel"] = _zgcr_presel(df)
            for s, m in masks.items():
                for v in inc_vars:
                    H[s][v]["incl"].fill(df[v].astype(float).values[m], weight=w[m])
            del df
        for s in sels:  # no inclusive LHE-photon-pT distribution to compare to
            H[s]["LHEGammaPt"]["incl"] = None

    for s, sel_label in sels.items():
        for v, b, xl in vars_:
            make_plot(H[s][v], v, b, xl, fam, sel_label, args.year,
                      Path(args.outdir) / args.year / name / f"stitch_{name}_{v}_{s}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--year", default="2024")
    p.add_argument("--data-dir", required=True, help="base skim dir (inclusive references)")
    p.add_argument("--lhe-data-dir", required=True, help="LHE-photon reskim dir (PTG bins)")
    p.add_argument("--outdir", default="plots/stitch_lhe/")
    p.add_argument("--families", default=",".join(FAMILIES), help="comma-separated subset")
    p.add_argument("--mass-col", default="FatJet0_ParTmassX2p",
                   help="jet regressed-mass column (FatJet0_ParTmassX2p or FatJet0_ParTmassGeneric)")
    args = p.parse_args()
    for fam in FAMILIES.values():
        fam["cols"] = [args.mass_col if c == "FatJet0_ParTmassX2p" else c for c in fam["cols"]]
        fam["vars"] = [(args.mass_col if v == "FatJet0_ParTmassX2p" else v, b, xl)
                       for v, b, xl in fam["vars"]]
    for name in args.families.split(","):
        run_family(name, FAMILIES[name], args)


if __name__ == "__main__":
    main()
