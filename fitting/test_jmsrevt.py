# Unit test: event-by-event JMS/JMR in fill_binned_histogram on a synthetic Z->bb sample
import json, shelve, tempfile, os, sys, types
import numpy as np, pandas as pd
sys.path.insert(0, "fitting")
import template_utils as tu
from scalesmear import MorphHistW2

setup = json.load(open("fitting/setup_zgcr_gen_jmsrevt.json"))
N = 200000
rng = np.random.default_rng(7)
m = rng.normal(93, 10, N)
df = pd.DataFrame({
    "FatJet0_ParTmassGeneric": m, "FatJet0_pt": np.full(N, 400.0),
    "FatJet0_ParTPXccVsQCD": 0.1, "FatJet0_ParTPXbbVsQCD": 0.99, "FatJet0_ParTPXbbXcc": 0.99,
    "FatJet0_ParTPXbb": 0.9, "FatJet0_ParTPXcc": 0.05, "FatJet0_ParTPQCD": 0.02, "FatJet0_ParTPXcs": 0.01,
    "Photon0_phi": np.zeros(N), "FatJet0_phi": np.full(N, 3.0), "MET": np.zeros(N),
    "Photon200": True, "Photon110EB_TightID_TightIso": True, "Photon0_pt": np.full(N, 300.0),
    "GenFlavor": np.full(N, 3), "finalWeight": np.ones(N), "sum_genWeight": np.ones(N),
})
args = types.SimpleNamespace(debug=False, save_templates=True, save_plotting_pkl=False)
systs = ["nominal", "JMSunconstrainedUp", "JMSunconstrainedDown", "JMRunconstrainedUp", "JMRunconstrainedDown"]
out = os.path.join(tempfile.mkdtemp(), "t")
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    tu.fill_binned_histogram({"Zgamma": df.copy()}, "zgcr", setup, args, systs=systs, template_outfile_path=out, dataset="ZGto2QG-unittest")
db = shelve.open(out)
H = {s: db[f"zgcr_pass_bb_pt1_Zgammabb_{s}"].values() for s in systs}
edges = np.linspace(60, 220, 21); c = 0.5*(edges[1:]+edges[:-1])
raw, _ = np.histogram(m, edges)
def stats(h): mu=(h*c).sum()/h.sum(); return h.sum(), mu, np.sqrt((h*(c-mu)**2).sum()/h.sum())
print("nominal == raw (JMS=JMR=1):", np.array_equal(H["nominal"], raw))
for s in systs: print(f"{s:22s} N={stats(H[s])[0]:8.0f} mean={stats(H[s])[1]:6.2f} std={stats(H[s])[2]:5.2f}")
mo = MorphHistW2((raw.astype(float), edges, raw.astype(float)))
mean = (raw*c).sum()/raw.sum()
for s, (sh, sc) in {"JMSunconstrainedUp": (0.01*mean, 1.0), "JMSunconstrainedDown": (-0.01*mean, 1.0), "JMRunconstrainedUp": (0.0, 1.1)}.items():
    ref = mo.get(shift=sh, scale=sc)[0]
    d = (H[s]-H["nominal"]); dr = (ref-raw)
    print(f"{s:22s} evt-vs-morph delta corr={np.corrcoef(d, dr)[0,1]:.3f}  sum|evt delta|/sum|morph delta|={np.abs(d).sum()/np.abs(dr).sum():.2f}")
print("JMR down == nominal (no narrowing):", np.array_equal(H["JMRunconstrainedDown"], H["nominal"]))
