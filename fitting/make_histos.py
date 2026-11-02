#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import pickle
import json  # <--- NEW IMPORT
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils

# --- 1. GLOBAL PROCESS MAP (Matches pmap_run3.json keys) ---
PROCESS_MAP = {
    # Data
    "data":       "data_obs",  
    
    # Backgrounds (Keys match pmap_run3.json)
    "ttbar":      "ttbar",
    "Wjets":      "Wjets",
    "Zjets":      "Zjets",
    "Zgamma":     "Zgamma",
    "Wgamma":     "Wgamma",
    "GJets":      "GJets",
    "QCD":        "QCD",
    "singlet":    "singlet",
    "VV":         "VV",
    "TTGamma":    "TTGamma",
    "EWKW":       "EWKW",
    "EWKZ":       "EWKZ",
    
    # Signals
    "ggF":        "ggF",
    "VBF":        "VBF",
    "WH":         "WH",
    "ZH":         "ZH",
    "ttH":        "ttH"
}

# Define the possible ptbins
# NOTE: Set to [250, 1200] for comparison. Set to [250, 500, 1200] for production.
ptbins = np.array([250, 1200]) 

# Define the histogram axes
categories = ["inclusive", "pass", "pass_bb", "pass_cc", "fail"]

axis_to_histaxis = {
    "category": hist.axis.StrCategory(categories, name="category", label="Category", growth=False),
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 20, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "genflavor": hist.axis.IntCategory([0, 1, 2, 3], name="genflavor", label="Gen Flavor"),
    "met": hist.axis.Regular(50, 0, 300, name="met", label="MET [GeV]"),
    "photon_pt": hist.axis.Regular(50, 0, 1200, name="photon_pt", label=r"Photon $p_{T}$ [GeV]"),
    "delta_phi": hist.axis.Regular(
        32, 0, 3.2, name="delta_phi", label=r"$\Delta\phi(\gamma, \text{jet})$"
    ),
}

# add more as needed
axis_to_column = {
    "pt1": "FatJet0_pt",
    "pt2": "FatJet1_pt",
    "msd1": "FatJet0_msdmatched",  # Matched mass to sync with Script B
    "category": "category",
    "genflavor": "GenFlavor",
    "met": "MET",
    "photon_pt": "Photon0_pt",
    "delta_phi": "delta_phi_photon_jet", 
}


def fill_ptbinned_histogram(h, events, axis, region):
    """
    Fills a histogram with events from a single dataset.
    """
    for _process_name, data in events.items():

        # --- Calculate dPhi ---
        dphi = np.nan
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        # --- Extract MET ---
        met_pt = None
        if "MET" in data.columns:
            met_col = data["MET"]
            if met_col.dtype == "object":
                try:
                    met_pt = met_col.apply(lambda d: d["pt"]).to_numpy()
                except Exception:
                    met_pt = met_col
            else:
                met_pt = met_col.to_numpy()

        if met_pt is None:
            print("WARNING: MET column missing. Setting MET=0.")
            met_pt = np.zeros(len(data))

        # --- Extract Plotting Variable ---
        if axis_to_column[axis] == "delta_phi_photon_jet":
            var_series = dphi
        elif axis == "met":
            var_series = met_pt
        else:
            var_series = data[axis_to_column[axis]]

        weight_val = data["finalWeight"].astype(float)

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(np.int8)
            if not isRealData
            else np.zeros_like(var_series, dtype=np.int8)
        )

        # Trigger Logic
        trigger_mask = True
        if region == "control-zgamma":
            if "Photon200" in data.columns and "Photon110EB_TightID_TightIso" in data.columns:
                trigger_mask = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
            else:
                # Safe fallback if trigger columns missing in MC
                pass 

        # Event selection variables
        Txcc = data["FatJet0_ParTPXccVsQCD"]
        Txbb = data["FatJet0_ParTPXbbVsQCD"]
        Txbbxcc = data["FatJet0_ParTPXbbXcc"] 
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        photon_pt = data["Photon0_pt"] if "Photon0_pt" in data.columns else None

        # --- Clean Memory ---
        del data
        gc.collect()

        # --- Selections ---
        basic_cuts = (
            (photon_pt > 120) & (msd > 20) & (msd < 200) & (pt > 250) & (pt < 1200) & (trigger_mask)
        )

        topo_cuts = (dphi > 2.2) & (met_pt < 50)
        pre_selection = basic_cuts & topo_cuts
        working_point = 0.95

        selection_dict = {
            "inclusive": pre_selection,
            "pass_bb": pre_selection & (Txbbxcc > working_point) & (Txbb > Txcc),
            "pass_cc": pre_selection & (Txbbxcc > working_point) & (Txcc > Txbb),
            "fail": pre_selection & (Txbbxcc <= working_point),
            "pass": pre_selection & (Txbbxcc > working_point),
        }

        for category, selection in selection_dict.items():
            h.fill(
                var_series[selection],
                pt[selection],
                category=category,
                genflavor=genflavordata[selection],
                weight=weight_val[selection],
            )
    return h


def export_to_root(histograms, output_root_path, region, samples_qq):
    print(f"\n--- Exporting to ROOT: {output_root_path} ---")

    region_map = {
        "control-zgamma": "zgcr",
        "control-tt": "mucr",
        "signal-all": "sr",
    }
    reg_name = region_map.get(region, region)

    with uproot.recreate(output_root_path) as fout:
        for process, h in histograms.items():

            # Use Global PROCESS_MAP
            proc_name = PROCESS_MAP.get(process, process)

            should_split_flavor = process in samples_qq

            pt_axis = h.axes["pt1"]
            cat_axis = h.axes["category"]

            for i_pt in range(len(pt_axis.edges) - 1):
                pt_bin_name = f"pt{i_pt+1}"

                for category in cat_axis:
                    base_name = f"{reg_name}_{category}_{pt_bin_name}"

                    if should_split_flavor:
                        # 1. BB Component
                        h_bb = h[:, i_pt, category, 3]
                        name_bb = f"{base_name}_{proc_name}bb_nominal"
                        fout[name_bb] = h_bb

                        # 2. Light Component
                        h_all_flav = h[:, i_pt, category, sum]
                        h_light = h_all_flav + (-1 * h_bb)
                        name_light = f"{base_name}_{proc_name}_nominal"
                        fout[name_light] = h_light

                    else:
                        h_1d = h[:, i_pt, category, sum]
                        name = f"{base_name}_{proc_name}_nominal"
                        fout[name] = h_1d

    print(f"Saved ROOT file to {output_root_path}")


def main(args):
    year = args.year
    region = args.region

    # --- 1. LOAD JSON SOURCE ---
    with open("pmap_run3.json") as f:
        pmap = json.load(f)

    # --- 2. DEFINE FLAVOR SPLITTING (Using JSON Keys) ---
    split_map = {
        "control-zgamma": ["Wjets", "Zjets", "Zgamma", "TTGamma", "Wgamma"],
        "control-tt":     ["ttbar", "Wjets", "Zjets", "singlet"],
        "signal-vh":      ["Wjets", "Zjets", "ttbar"],
        "signal-vbf":     ["Wjets", "Zjets", "ttbar"],
        "signal-ggf":     ["Wjets", "Zjets", "ttbar"],
        "signal-all":     ["Wjets", "Zjets", "ttbar"],
    }
    samples_qq = split_map.get(region, ["Wjets", "Zjets", "ttbar"])
    print(f"Splitting flavor for: {samples_qq}")

    # --- 3. SETUP PATHS ---
    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3"
    dir_name = "skims/26Jan16"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"
    filters = None
    variable_to_plot = args.variable

    # --- 4. DEFINE COLUMNS ---
    base_columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_ParTPXbbVsQCD",
        "FatJet0_ParTPXccVsQCD",
        "FatJet0_ParTPXbbXcc", 
    ]

    if region == "control-zgamma":
        base_columns.extend([
            "Photon0_pt", "Photon200", "Photon110EB_TightID_TightIso",
            "Photon0_phi", "FatJet0_phi", "MET"
        ])

    var_cols = axis_to_column[variable_to_plot]
    if isinstance(var_cols, str):
        if var_cols == "delta_phi_photon_jet":
            if "Photon0_phi" not in base_columns:
                base_columns.extend(["Photon0_phi", "FatJet0_phi"])
        else:
            base_columns.append(var_cols)
    elif isinstance(var_cols, list):
        base_columns.extend(var_cols)

    load_columns_mc = list(set(base_columns + ["GenFlavor"]))
    load_columns_data = list(set(base_columns))
    data_dir = Path(path_to_dir) / year

    # --- 5. SELECT DATA KEY BASED ON REGION ---
    if region == "control-zgamma":
        data_key = "EGammadata"
    elif region == "control-tt":
        data_key = "Muondata" # Check pmap if it's Muondata or ParkingSingleMuondata
    else:
        data_key = "Jetdata"

    # --- 6. BUILD SAMPLES DICT FROM JSON ---
    samples = {}
    for key, file_list in pmap.items():
        if "data" in key:
            if key == data_key:
                # Important: Map this to "data" so script logic works
                samples["data"] = file_list
        else:
            samples[key] = file_list

    # --- 7. PROCESSING LOOP ---
    hists_to_make = [variable_to_plot]

    for hist_name in hists_to_make:
        print(f"\n--- Processing variable: {hist_name} ---")
        histograms = {}

        for process, datasets in samples.items():
            load_columns = load_columns_data if process == "data" else load_columns_mc
            print(f"Processing {process} for year {year}...")

            h = hist.Hist(
                axis_to_histaxis[hist_name],
                axis_to_histaxis["pt1"],
                axis_to_histaxis["category"],
                axis_to_histaxis["genflavor"],
            )

            for dataset in datasets:
                events = utils.load_samples(
                    data_dir,
                    {process: [dataset]}, # Note: process here is e.g. "Wjets"
                    columns=load_columns,
                    region=region,
                    filters=filters,
                )

                if not events:
                    continue

                h = fill_ptbinned_histogram(h, events, hist_name, region)
                del events
                gc.collect()

            if h.sum() == 0:
                print(f"WARNING: No events for {process}. Skipping.")
                continue
            histograms[process] = h

        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        pkl_file = output_dir / f"histograms_{variable_to_plot}_{year}_{region}.pkl"
        with pkl_file.open("wb") as f:
            pickle.dump(histograms, f)
        print(f"Pickle saved to {pkl_file}")

        if args.save_root and variable_to_plot == "msd1":
            root_file = output_dir / f"fitting_{year}_{region}.root"
            export_to_root(histograms, root_file, region, samples_qq)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--region", required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--outdir", default="histograms")
    parser.add_argument("--save-root", action="store_true")
    args = parser.parse_args()

    main(args)