#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import pickle
import json
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils

# --- 1. GLOBAL PROCESS MAP ---
PROCESS_MAP = {
    "data": "data_obs", "ttbar": "ttbar", "Wjets": "Wjets", "Zjets": "Zjets",
    "Zgamma": "Zgamma", "Wgamma": "Wgamma", "GJets": "GJets", "QCD": "QCD",
    "singlet": "singlet", "VV": "VV", "TTGamma": "TTGamma",
    "EWKW": "EWKW", "EWKZ": "EWKZ", 
    "ggF": "ggF", "VBF": "VBF", "WH": "WH", "ZH": "ZH", "ttH": "ttH"
}

# --- 2. DEFINE CATEGORIES (Preserving Old Script Categories) ---
categories = [
    "inclusive", "pass", "pass_bb", "pass_cc", "fail", # Standard/ZGamma
    "ggf_pass_bb", "ggf_pass_cc", "ggf_fail",          # Signal - ggF
    "vbf_pass_bb", "vbf_pass_cc", "vbf_fail",          # Signal - VBF
    "vh_pass_bb",  "vh_pass_cc",  "vh_fail"            # Signal - VH
]

# PT Bins set to match your Old Script for exact comparison
ptbins = np.array([250, 1200]) 

axis_to_histaxis = {
    "category": hist.axis.StrCategory(categories, name="category", label="Category", growth=False),
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 20, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
    "met": hist.axis.Regular(50, 0, 300, name="met", label="MET [GeV]"),
    "photon_pt": hist.axis.Regular(50, 0, 1200, name="photon_pt", label=r"Photon $p_{T}$ [GeV]"),
    "delta_phi": hist.axis.Regular(32, 0, 3.2, name="delta_phi", label=r"$\Delta\phi(\gamma, \text{jet})$"),
}

axis_to_column = {
    "pt1": "FatJet0_pt", "pt2": "FatJet1_pt", "msd1": "FatJet0_msd",
    "category": "category", "genflavor": "GenFlavor", "met": "MET",
    "photon_pt": "Photon0_pt", "delta_phi": "delta_phi_photon_jet", 
}

# --- 3. FILLING LOGIC ---
def fill_ptbinned_histogram(h, events, axis, region):
    for _process_name, data in events.items():
        # dPhi Calculation
        dphi = np.zeros(len(data))
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        met_pt = data["MET"].to_numpy() if "MET" in data.columns else np.zeros(len(data))
        mjj = data["VBFPair_mjj"].to_numpy() if "VBFPair_mjj" in data.columns else np.zeros(len(data))

        if axis == "delta_phi": var_vals = dphi
        elif axis == "met": var_vals = met_pt
        else: var_vals = data[axis_to_column[axis]].to_numpy()

        weight_val = data["weight"].astype(float).to_numpy()
        pt = data["FatJet0_pt"].to_numpy()
        msd = data["FatJet0_msd"].to_numpy()
        genflavordata = data["GenFlavor"].astype(int).to_numpy() if "GenFlavor" in data.columns else np.zeros(len(data), dtype=int)

        Txbbxcc = data["FatJet0_ParTPXbbXcc"].to_numpy()
        Txbb = data["FatJet0_ParTPXbbVsQCD"].to_numpy()
        Txcc = data["FatJet0_ParTPXccVsQCD"].to_numpy()

        # Selections
        working_point = 0.95 
        pass_cut = (Txbbxcc > working_point)
        fail_cut = (Txbbxcc <= working_point)
        bb_cut = (Txbb > Txcc)
        cc_cut = (Txcc > Txbb)
        
        selection_dict = {}

        if region == "control-zgamma":
            trigger_mask = True
            if "Photon200" in data.columns: 
                trigger_mask = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
            
            pre = (data["Photon0_pt"] > 120) & (msd > 20) & (msd < 200) & (pt > 250) & trigger_mask & (dphi > 2.2) & (met_pt < 50)
            
            selection_dict = {
                "inclusive": pre,
                "pass":      pre & pass_cut,
                "pass_bb":   pre & pass_cut & bb_cut,
                "pass_cc":   pre & pass_cut & cc_cut,
                "fail":      pre & fail_cut,
            }
        else:
            # Signal logic (keeps VBF/ggF split)
            pre = (pt > 450) & (msd > 40) 
            is_vbf = (mjj > 1000) & pre
            is_ggf = (~is_vbf) & pre 

            selection_dict = {
                "inclusive": pre,
                "ggf_pass_bb": is_ggf & pass_cut & bb_cut,
                "ggf_pass_cc": is_ggf & pass_cut & cc_cut,
                "ggf_fail":    is_ggf & fail_cut,
                "vbf_pass_bb": is_vbf & pass_cut & bb_cut,
                "vbf_pass_cc": is_vbf & pass_cut & cc_cut,
                "vbf_fail":    is_vbf & fail_cut,
            }

        for category, selection in selection_dict.items():
            if np.sum(selection) == 0: continue
            h.fill(var_vals[selection], pt[selection], category=category, genflavor=genflavordata[selection], weight=weight_val[selection])
    return h

# --- 4. EXPORT TO ROOT ---
def export_to_root(histograms, output_root_path, region, samples_qq):
    print(f"\n--- Exporting to ROOT: {output_root_path} ---")

    region_map = {"control-zgamma": "zgcr", "control-tt": "mucr"}
    reg_prefix = region_map.get(region, "")

    with uproot.recreate(output_root_path) as fout:
        for process, h in histograms.items():
            proc_name = PROCESS_MAP.get(process, process)
            should_split_flavor = process in samples_qq
            pt_axis = h.axes["pt1"]

            for i_pt in range(len(pt_axis.edges) - 1):
                for category in h.axes["category"]:
                    # REMOVED the 'if inclusive in category: continue' line so they export!
                    if h[:, i_pt, category, sum].sum() == 0: continue

                    bin_suffix = f"pt{i_pt+1}"
                    if "vbf" in category: bin_suffix = f"mjj{i_pt+1}"

                    # Construct name: zgcr_inclusive_pt1_...
                    name_start = f"{reg_prefix}_{category}" if reg_prefix else category
                    base_name = f"{name_start}_{bin_suffix}"

                    if should_split_flavor:
                        h_bb = h[:, i_pt, category, 3]
                        fout[f"{base_name}_{proc_name}bb_nominal"] = h_bb
                        h_light = h[:, i_pt, category, sum] + (-1 * h_bb)
                        fout[f"{base_name}_{proc_name}_nominal"] = h_light
                    else:
                        h_1d = h[:, i_pt, category, sum]
                        fout[f"{base_name}_{proc_name}_nominal"] = h_1d

    print(f"Saved ROOT file to {output_root_path}")


def main(args):
    year = args.year
    region = args.region

    # Load JSON
    with open("pmap_run3.json") as f: pmap = json.load(f)
    
    # Flavor Splitting
    split_map = {
        "control-zgamma": ["Wjets", "Zjets", "Zgamma", "TTGamma", "Wgamma"],
        "signal-vbf":     ["Wjets", "Zjets", "ttbar"],
        "signal-all":     ["Wjets", "Zjets", "ttbar"],
    }
    samples_qq = split_map.get(region, ["Wjets", "Zjets", "ttbar"])

    # [FIXED] Updated Path
    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/skims/26Jan16"
    path_to_dir = f"{MAIN_DIR}/{year}"

    # [FIXED] Updated Columns (Removed finalWeight)
    cols = [
        "weight", 
        "FatJet0_pt", 
        "FatJet0_msd", 
        "FatJet0_ParTPXbbVsQCD", 
        "FatJet0_ParTPXccVsQCD", 
        "FatJet0_ParTPXbbXcc",
        "GenFlavor"
    ]
    if region == "control-zgamma":
        cols.extend(["Photon0_pt", "Photon0_phi", "FatJet0_phi", "MET", "Photon200", "Photon110EB_TightID_TightIso"])
    elif "signal" in region:
        cols.extend(["VBFPair_mjj"])

    # Data Key
    data_key = "EGammadata" if region == "control-zgamma" else "Jetdata"
    
    samples = {}
    for key, val in pmap.items():
        if "data" in key:
            if key == data_key: samples["data"] = val
        else:
            samples[key] = val

    # Processing
    histograms = {}
    var_name = args.variable
    
    for process, datasets in samples.items():
        print(f"Processing {process}...")
        h = hist.Hist(
            axis_to_histaxis[var_name],
            axis_to_histaxis["pt1"],
            axis_to_histaxis["category"],
            axis_to_histaxis["genflavor"],
        )
        
        load_cols = list(set(cols)) if process != "data" else [c for c in cols if "GenFlavor" not in c]
        
        for dataset in datasets:
            # Handle directory name (underscore vs dash)
            region_dir = region.replace("-", "_") if "signal" in region else region
            
            events = utils.load_samples(Path(path_to_dir), {process: [dataset]}, columns=load_cols, region=region_dir)
            if events:
                h = fill_ptbinned_histogram(h, events, var_name, region)
                del events
                gc.collect()
        
        if h.sum() > 0: histograms[process] = h

    # Export
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    if args.save_root:
        if "signal" in region:
            root_file = outdir / f"fitting_{year}_signal.root"
        else:
            root_file = outdir / f"fitting_{year}_{region}.root"
            
        export_to_root(histograms, root_file, region, samples_qq)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--variable", default="msd1")
    parser.add_argument("--outdir", default="histograms")
    parser.add_argument("--save-root", action="store_true")
    args = parser.parse_args()
    main(args)