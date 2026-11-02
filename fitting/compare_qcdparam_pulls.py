#!/usr/bin/env python3
"""
Compare qcdparam nuisance pulls across the three decorrelation fits
(bb-only, cc-only, simultaneous) — answers Cristina's qcdparam-correlation
question directly from the existing fitDiagnostics files, no Impacts needed.

Run on LPC in a cmsenv shell (needs ROOT). IMPORTANT: do NOT also activate the
hbb conda env in the same shell (ROOT cling flags collide), and use `python3`
(bare `python` under cmsenv hits system python -> encodings error):
    cd ~/nobackup/CMSSW_14_1_0_pre4/src && cmsenv
    cd ~/nobackup/hbb-run3/fitting/results/Test_v15/2024/datacards/decorrelation_2024
    python3 ~/nobackup/hbb-run3/fitting/compare_qcdparam_pulls.py

Reads fit_s RooFitResult from each file; a "pull" here is the postfit value
(nuisances are prefit N(0,1), so the value itself is the pull in sigma).
"""
import ROOT
import re

# (label, filename) — edit paths/names if they differ on disk
FITS = [
    ("bb-only",      "fitDiagnostics_passbb_only_2024.root"),
    ("cc-only",      "fitDiagnostics_passcc_only_2024_rcc.root"),
    ("simultaneous", "fitDiagnostics_simultaneous_2024.root"),
]

PARAM_RE = re.compile(r"qcdparam_ptbin0zgcr2024_\d+")

ROOT.gErrorIgnoreLevel = ROOT.kWarning


def load_pulls(fname):
    try:
        f = ROOT.TFile.Open(fname)
    except OSError:
        f = None
    if not f or f.IsZombie():
        print(f"  !! could not open {fname} (skipping)")
        return {}
    fit = f.Get("fit_s")
    if not fit:
        print(f"  !! no fit_s in {fname}")
        return {}
    out = {}
    pars = fit.floatParsFinal()
    for i in range(pars.getSize()):
        p = pars.at(i)
        if PARAM_RE.fullmatch(p.GetName()):
            out[p.GetName()] = (p.getVal(), p.getError())
    f.Close()
    return out


results = {}
for label, fname in FITS:
    print(f"Loading {label}: {fname}")
    results[label] = load_pulls(fname)

# union of param names, sorted by index
names = set()
for d in results.values():
    names |= set(d.keys())
names = sorted(names, key=lambda n: int(n.rsplit("_", 1)[1]))

labels = [lbl for lbl, _ in FITS]
header = f"{'qcdparam':<32}" + "".join(f"{l:>22}" for l in labels)
print("\n" + header)
print("-" * len(header))
for n in names:
    row = f"{n:<32}"
    for l in labels:
        if n in results[l]:
            v, e = results[l][n]
            row += f"{v:>+10.2f} ± {e:<8.2f}"
        else:
            row += f"{'--':>22}"
    print(row)

# flag opposite-sign pulls between bb-only and cc-only (the decorrelation signal)
print("\nOpposite-sign bb-only vs cc-only (decorrelation flags):")
flagged = False
for n in names:
    if n in results["bb-only"] and n in results["cc-only"]:
        vb = results["bb-only"][n][0]
        vc = results["cc-only"][n][0]
        if vb * vc < 0 and abs(vb) > 0.3 and abs(vc) > 0.3:
            print(f"  {n:<32} bb={vb:+.2f}  cc={vc:+.2f}")
            flagged = True
if not flagged:
    print("  (none above |0.3σ| threshold)")
