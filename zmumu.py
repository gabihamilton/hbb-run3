import awkward as ak
import vector
import hist
import mplhep as hep
import matplotlib.pyplot as plt
import glob

# 1. Grab the parquet files directly from your EOS group space
zmumu_files = glob.glob("/eos/uscms/store/group/lpchbbrun3/gmachado/25Mar26_v14_private/2023/*/parquet/nominal/control-zmumu/*.parquet")
zgamma_files = glob.glob("/eos/uscms/store/group/lpchbbrun3/gmachado/25Mar26_v14_private/2023/*/parquet/nominal/control-zgamma/*.parquet")

zmumu_data = ak.from_parquet(zmumu_files)
zgamma_data = ak.from_parquet(zgamma_files)

print("OK")
# 2. Reconstruct the Z mass 
# (Note: Update 'muon1_pt', etc. to whatever the actual column names are in your framework's skims!)
def get_mass(events):
    mu1 = vector.zip({"pt": events.muon1_pt, "eta": events.muon1_eta, "phi": events.muon1_phi, "mass": events.muon1_mass})
    mu2 = vector.zip({"pt": events.muon2_pt, "eta": events.muon2_eta, "phi": events.muon2_phi, "mass": events.muon2_mass})
    return (mu1 + mu2).mass

m_zmumu = get_mass(zmumu_data)
m_zgamma = get_mass(zgamma_data)

# 3. Create the plot
fig, ax = plt.subplots(figsize=(10, 8))
hep.style.use(hep.style.CMS)

# We use density=True to normalize the shapes to 1 since we don't have the cross-sections yet
h_zmumu = hist.Hist(hist.axis.Regular(40, 70, 110, name="mass", label=r"$m(\mu\mu)$ [GeV]"))
h_zmumu.fill(m_zmumu)
h_zmumu.plot(ax=ax, density=True, label=r"$Z(\mu\mu)$ + jets", color="blue", histtype="step", linewidth=2)

h_zgamma = hist.Hist(hist.axis.Regular(40, 70, 110, name="mass", label=r"$m(\mu\mu)$ [GeV]"))
h_zgamma.fill(m_zgamma)
h_zgamma.plot(ax=ax, density=True, label=r"$Z(\mu\mu) + \gamma$", color="red", histtype="step", linewidth=2)

ax.set_ylabel("Normalized Events")
ax.legend()
hep.cms.label(data=False, year="2023", ax=ax, lumi=None) # Lumi hidden since it's shape-only

plt.savefig("cristina_validation_zmumu.pdf", bbox_inches="tight")
print("Plot saved to cristina_validation_zmumu.pdf!")
