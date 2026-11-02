#!/usr/bin/env python3
"""
Plot the event-by-event JMS/JMR template variations (nominal vs JMS up/down, JMR up/down)
from a collect_hists ROOT file, with a variation/nominal ratio panel.

JMR down is drawn as the datacard uses it: mirrored around nominal (2*nom - up) when the
requested JMR - unc < 1 cannot be reached by smearing (the filled JMRDown equals nominal).

  python make_jmsr_variation_plots.py \
      --infile results_zgcr_regmassgen_jmsrevt/26Oct02_regmassgen_jmsrevt/fitting_2024_zgcr_regmassgen_jmsrevt.root \
      --outdir plots/26Oct02_regmassgen_jmsrevt/jmsr_variations
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

plt.style.use(hep.style.CMS)

# (template stem, label)
PANELS = [
    ("zgcr_pass_bb_pt1_Zgammabb", r"Z$\gamma\to$bb, pass-bb"),
    ("zgcr_pass_cc_pt1_Zgammac", r"Z$\gamma\to$cc, pass-cc"),
    ("zgcr_pass_cc_pt1_Wgammac", r"W$\gamma\to$cs, pass-cc"),
    ("zgcr_fail_pt1_Zgammabb", r"Z$\gamma\to$bb, fail"),
]
SYST = "unconstrained"  # JMSunconstrainedUp etc.


def moments(h, c):
    mu = (h * c).sum() / h.sum()
    return mu, np.sqrt((h * (c - mu) ** 2).sum() / h.sum())


def plot_one(f, stem, label, kind, args):
    nom_h = f[f"{stem}_nominal"]
    nom = nom_h.values()
    edges = nom_h.axis().edges()
    c = 0.5 * (edges[1:] + edges[:-1])
    up = f[f"{stem}_{kind}{SYST}Up"].values()
    dn = f[f"{stem}_{kind}{SYST}Down"].values()
    if kind == "JMR" and np.allclose(dn, nom):
        dn = np.clip(2 * nom - up, 0, None)
        dn_label = r"JMR $-1\sigma$ (mirrored)"
    else:
        dn_label = rf"{kind} $-1\sigma$"
    unc = {"JMS": "1%", "JMR": "10%"}[kind]

    fig, (ax, rax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    for vals, lab, col, ls in [
        (nom, "Nominal", "black", "-"),
        (up, rf"{kind} $+1\sigma$ ({unc})", "#B5541A", "-"),
        (dn, dn_label, "#2C5FA8", "--"),
    ]:
        mu, sd = moments(vals, c)
        hep.histplot(vals, edges, ax=ax, color=col, ls=ls, lw=2,
                     label=f"{lab}: mean {mu:.1f}, std {sd:.1f} GeV")
    ax.set_ylabel("Events / 8 GeV")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
    ax.legend(fontsize=17, loc="upper right", title=label, title_fontsize=19)
    hep.cms.label("Private Work", data=False, year="2024", lumi=109, ax=ax, fontsize=20)

    with np.errstate(divide="ignore", invalid="ignore"):
        for vals, col, ls in [(up, "#B5541A", "-"), (dn, "#2C5FA8", "--")]:
            hep.histplot(np.where(nom > 0, vals / nom, 1.0), edges, ax=rax, color=col, ls=ls, lw=2)
    rax.axhline(1, color="black", lw=1)
    rax.set_ylim(0.6, 1.4)
    rax.set_ylabel("Var. / nom.")
    rax.set_xlabel("Jet regressed mass (generic) [GeV]")
    rax.set_xlim(edges[0], edges[-1])

    out = Path(args.outdir) / f"{kind}_{stem.replace('zgcr_', '').replace('_pt1', '')}.png"
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main(args):
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    with uproot.open(args.infile) as f:
        for stem, label in PANELS:
            for kind in ("JMS", "JMR"):
                plot_one(f, stem, label, kind, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="collect_hists ROOT file with JMS/JMR templates")
    parser.add_argument("--outdir", required=True)
    main(parser.parse_args())
