#!/usr/bin/env python3
"""
Quick validation plots for Z(mumu) control region.
Plots m(mumu) and dimuon pair pT from DY MC.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import mplhep as hep
from hbb import utils

hep.style.use("CMS")

DATA_DIR = Path("/eos/uscms/store/user/lpchbbrun3/gmachado/26Mar26_v14_private/2023")
REGION = "control-zmumu"
OUTDIR = Path("plots_zmumu")
OUTDIR.mkdir(exist_ok=True)

DY_SUBSAMPLES = [
    "DYto2L-2Jets_MLL-50_PTLL-100to200_1J",
    "DYto2L-2Jets_MLL-50_PTLL-200to400_1J",
    "DYto2L-2Jets_MLL-50_PTLL-400to600_1J",
    "DYto2L-2Jets_MLL-50_PTLL-600_1J",
    "DYto2L-2Jets_MLL-50_PTLL-100to200_2J",
    "DYto2L-2Jets_MLL-50_PTLL-200to400_2J",
    "DYto2L-2Jets_MLL-50_PTLL-400to600_2J",
    "DYto2L-2Jets_MLL-50_PTLL-600_2J",
]

COLS = ["Zmm_MuonPair_mll", "Zmm_MuonPair_pt", "Zmm_MuonLead_pt",
        "Zmm_MuonSublead_pt", "weight"]

events = utils.load_samples(
    data_dir=DATA_DIR,
    samples={"DY": DY_SUBSAMPLES},
    columns=COLS,
    region=REGION,
)

if not events or "DY" not in events or len(events["DY"]) == 0:
    print("No events found! Check the path.")
    exit()

df = events["DY"]
print(f"Total events passing selection: {len(df)}")
w = df["weight"]

# Plot 1: m(mumu)
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(df["Zmm_MuonPair_mll"], bins=50, range=(50, 150), weights=w, histtype="step", linewidth=2)
ax.set_xlabel(r"$m(\mu\mu)$ [GeV]")
ax.set_ylabel("Events")
#ax.set_title("Z(mumu) Control Region - 2023")
hep.cms.label("Private Work", data=False, ax=ax, com=13.6)

fig.savefig(OUTDIR / "zmumu_mll_2023.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved mll plot")

# Plot 2: dimuon pair pT
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(df["Zmm_MuonPair_pt"], bins=50, range=(200, 800), weights=w, histtype="step", linewidth=2)
ax.set_xlabel(r"$p_T(\mu\mu)$ [GeV]")
ax.set_ylabel("Events")
#ax.set_title("Z(mumu) Control Region - 2023")
hep.cms.label("Private Work", data=False, ax=ax, com=13.6)
fig.savefig(OUTDIR / "zmumu_pairpt_2023.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved pair pT plot")

# Plot 3: lead muon pT
fig, ax = plt.subplots(figsize=(10, 8))
ax.hist(df["Zmm_MuonLead_pt"], bins=50, range=(50, 600), weights=w, histtype="step", linewidth=2)
ax.set_xlabel(r"Lead muon $p_T$ [GeV]")
ax.set_ylabel("Events")
hep.cms.label("Private Work", data=False, ax=ax, com=13.6)
fig.savefig(OUTDIR / "zmumu_leadmuon_pt_2023.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved lead muon pT plot")

print(f"\nAll plots saved to {OUTDIR}/")
