#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import pickle
from pathlib import Path

import hist
import numpy as np
from common import common_mc, data_by_year, data_by_year_muon, data_by_year_zgamma

from hbb import utils

# Define the possible ptbins
ptbins = np.array([250, 500, 1200])

# Define the histogram axes
axis_to_histaxis = {
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 20, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
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
    "msd1": "FatJet0_msd",
    # "mass1": "FatJet0_pnetMass",
    "category": "category",
    "genflavor": "GenFlavor",
    "met": "MET",
    "photon_pt": "Photon0_pt",
    "delta_phi": "delta_phi_photon_jet",  # This will be calculated on the fly
}


# --- FUNCTION MODIFIED ---
def fill_ptbinned_histogram(h, events, axis, region):
    """
    Fills a histogram with events from a single dataset.
    """
    for _process_name, data in events.items():

        # --- 1. CALCULATE CUT VARIABLES (Changed) ---
        # We calculate these NOW because we need them for the selection,
        # regardless of what variable we are actually plotting.

        # --- Calculate dPhi ---
        dphi = np.nan
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            # Wrap values > pi
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        # --- Extract MET ---
        # Initialize met_pt with a default (safe) value or extraction
        met_pt = None
        if "MET" in data.columns:
            met_col = data["MET"]
            # Check if it's an object/struct (often the case in older coffea versions or nanomet)
            if met_col.dtype == "object":
                try:
                    met_pt = met_col.apply(lambda d: d["pt"]).to_numpy()
                except Exception:
                    # Fallback if apply fails, assuming it might be a direct array
                    met_pt = met_col
            else:
                # If it's already a float/int array
                met_pt = met_col.to_numpy()

        # If MET is missing (shouldn't happen if loaded correctly), set to 0 so we don't crash,
        # but print warning
        if met_pt is None:
            print("WARNING: MET column missing or unreadable. Setting MET=0 for selection.")
            met_pt = np.zeros(len(data))

        # --- 2. EXTRACT PLOTTING VARIABLE ---
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
                print("WARNING: Trigger columns not found for zgamma region.")

        # Event selection columns
        Txcc = data["FatJet0_ParTPXccVsQCD"]
        Txbb = data["FatJet0_ParTPXbbVsQCD"]
        Txbbxcc = data["FatJet0_ParTPXbbXcc"]  # for lara's category
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        photon_pt = data["Photon0_pt"] if "Photon0_pt" in data.columns else None

        # --- 3. DELETE DATAFRAME ---
        del data
        gc.collect()

        # --- 4. DEFINE SELECTION (Updated with Cuts) ---

        # Standard kinematic cuts
        basic_cuts = (
            (photon_pt > 120) & (msd > 20) & (msd < 200) & (pt > 250) & (pt < 1200) & (trigger_mask)
        )

        # NEW CUTS: MET and DeltaPhi
        # Note: We use 'dphi' and 'met_pt' calculated earlier
        topo_cuts = (dphi > 2.2) & (  # Back-to-back cut (Removes QCD)
            met_pt < 50
        )  # MET cut (Removes ttbar)

        pre_selection = basic_cuts & topo_cuts

        working_point = 0.95

        selection_dict = {
            "inclusive": pre_selection,
            "pass_bb": pre_selection & (Txbbxcc > working_point) & (Txbb > Txcc),
            "pass_cc": pre_selection & (Txbbxcc > working_point) & (Txcc > Txbb),
            "fail": pre_selection & (Txbbxcc <= working_point),
            "pass": pre_selection & (Txbbxcc > working_point),  # (Union of pass_bb and pass_cc)
        }

        # Fill histograms
        for category, selection in selection_dict.items():
            h.fill(
                var_series[selection],
                pt[selection],
                category=category,
                genflavor=genflavordata[selection],
                weight=weight_val[selection],
            )
    return h


def main(args):
    year = args.year
    region = args.region

    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/"
    # dir_name = "gmachado/25Oct27_v12"
    dir_name = "gmachado/25Nov19_stable_v14_private"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    filters = None
    variable_to_plot = args.variable

    # 1. Define base columns ALWAYS needed for selections
    base_columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        # "FatJet0_pnetTXbb",
        # "FatJet0_pnetTXcc",
        # "FatJet0_ParTPXbb",
        # "FatJet0_ParTPXcc",
        "FatJet0_ParTPXbbVsQCD",
        "FatJet0_ParTPXccVsQCD",
        "FatJet0_ParTPXbbXcc",  # for lara's cat
    ]

    # 2. Add columns needed for the region
    if region == "control-zgamma":
        base_columns.extend(
            [
                "Photon0_pt",
                "Photon200",
                "Photon110EB_TightID_TightIso",
                # --- NEW: Added columns needed for cuts ---
                "Photon0_phi",
                "FatJet0_phi",
                "MET",
            ]
        )

    # 3. Add columns needed for the specific variable
    var_cols = axis_to_column[variable_to_plot]
    if isinstance(var_cols, str):
        if var_cols == "delta_phi_photon_jet":
            # Ensure we don't add duplicates if they are already in base
            if "Photon0_phi" not in base_columns:
                base_columns.extend(["Photon0_phi", "FatJet0_phi"])
        else:
            base_columns.append(var_cols)
    elif isinstance(var_cols, list):
        base_columns.extend(var_cols)

    # 4. Create the final lists
    load_columns_mc = list(set(base_columns + ["GenFlavor"]))
    load_columns_data = list(set(base_columns))

    data_dir = Path(path_to_dir) / year

    if region == "control-zgamma":
        data_samples = data_by_year_zgamma.get(year, {})
    elif region == "control-tt":
        data_samples = data_by_year_muon.get(year, {})
    else:
        data_samples = data_by_year.get(year, {})

    samples = {
        **common_mc,
        "data": data_samples,
    }

    variable_to_plot = args.variable
    hists_to_make = [variable_to_plot]

    print(f"Will create histogram files for: {', '.join(hists_to_make)}")

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
                    {process: [dataset]},
                    columns=load_columns,
                    region=region,
                    filters=filters,
                )

                if not events:
                    print(f"No events found for dataset {dataset} in year {year}. Skipping.")
                    continue

                h = fill_ptbinned_histogram(h, events, hist_name, region)

                del events
                gc.collect()

            if h.sum() == 0:
                print(
                    f"WARNING: No events were found for the entire '{process}' process group. Skipping."
                )
                continue
            histograms[process] = h

        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"histograms_{hist_name}_{year}_{region}.pkl"

        with output_file.open("wb") as f:
            pickle.dump(histograms, f)

        print(f"Histograms for {hist_name} saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
        "--region",
        help="region",
        type=str,
        required=True,
        choices=[
            "signal-all",
            "signal-ggf",
            "signal-vh",
            "signal-vbf",
            "control-tt",
            "control-zgamma",
        ],
    )
    parser.add_argument(
        "--variable",
        help="The variable to plot.",
        type=str,
        required=True,
        choices=["msd1", "met", "photon_pt", "delta_phi", "mass1"],
    )
    parser.add_argument(
        "--outdir", help="Output directory to save histograms.", type=str, default="histograms"
    )
    args = parser.parse_args()

    main(args)
