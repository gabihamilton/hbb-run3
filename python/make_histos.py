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
# ptbins = np.array([300, 450, 500, 550, 600, 675, 800, 1200])
ptbins = np.array([200, 500, 1200])
# ptbins = np.array([0, 500, 1200])

# ptbins_zgamma = np.array([200, 300, 450, 500, 550, 600, 675, 800, 1200])

# Define the histogram axes
axis_to_histaxis = {
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 0, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
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
    "mass1": "FatJet0_pnetMass",
    "category": "category",
    "genflavor": "GenFlavor",
    "met": "MET",
    "photon_pt": "Photon0_pt",
    "delta_phi": "delta_phi_photon_jet",  # This will be calculated on the fly
}


# --- FUNCTION MODIFIED ---
# It now takes an existing histogram `h` as an argument to fill
def fill_ptbinned_histogram(h, events, axis, region):
    """
    Fills a histogram with events from a single dataset.
    """
    for _process_name, data in events.items():

        # --- 1. EXTRACT ALL COLUMNS WE NEED ---
        # Handle dphi calculation FIRST, only if we need it
        if axis == "delta_phi":
            if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
                dphi = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
                # Wrap values > pi
                dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
                delta_phi_photon_jet = dphi
            else:
                print("WARNING: Missing dphi columns, filling with NaN.")
                delta_phi_photon_jet = np.nan
        else:
            delta_phi_photon_jet = np.nan  # Define as NaN for other axes

        # Now, get the main variable series
        if axis_to_column[axis] == "delta_phi_photon_jet":
            var_series = delta_phi_photon_jet
        else:
            var_series = data[axis_to_column[axis]]  # Get MET, msd, etc.

        weight_val = data["finalWeight"].astype(float)

        isRealData = "GenFlavor" not in data.columns
        genflavordata = (
            data["GenFlavor"].astype(np.int8)
            if not isRealData
            else np.zeros_like(var_series, dtype=np.int8)
        )

        # 1. Implement trigger OR for the control-zgamma region
        trigger_mask = True  # Default to pass for all other regions
        if region == "control-zgamma":
            # --- FIX ---
            # Use the columns we dynamically loaded
            if "Photon200" in data.columns and "Photon110EB_TightID_TightIso" in data.columns:
                trigger_mask = (
                    data["Photon200"] | data["Photon110EB_TightID_TightIso"]
                )  # | data["Photon30EB_TightID_TightIso"]
            else:
                print(
                    "WARNING: Trigger columns not found for zgamma region. No trigger selection applied."
                )

        # Event selection columns
        Txcc = data["FatJet0_pnetTXcc"]
        Txbb = data["FatJet0_pnetTXbb"]
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        photon_pt = data["Photon0_pt"] if "Photon0_pt" in data.columns else None

        # --- 2. DELETE THE ORIGINAL DATAFRAME ---
        # We have extracted all columns, so we can free this memory
        del data
        gc.collect()

        # --- 3. NOW, SAFELY TRANSFORM THE 'var' SERIES ---
        if axis == "met" and var_series.dtype == "object":
            print("Detected 'object' type for MET column. Attempting to extract 'pt' from dicts...")
            try:
                # This should now have enough memory to succeed
                var_series = var_series.apply(lambda d: d["pt"])
                print("Successfully extracted MET 'pt'.")
            except Exception as e:
                print(
                    f"ERROR: Failed to extract 'pt' from MET column. First element is: {var_series.iloc[0]}"
                )
                print(f"Error was: {e}")
                raise

        # --- 4. CONTINUE WITH THE REST OF THE LOGIC ---
        # print("pt min:", np.min(pt), " pt max:", np.max(pt))
        pre_selection = (
            (photon_pt > 120) & (msd > 20) & (msd < 200) & (pt > 250) & (pt < 1200) & (trigger_mask)
        )

        selection_dict = {
            "inclusive": pre_selection,
            "bb_pass": pre_selection & (Txbb > 0.95),
            "bb_fail": pre_selection & (Txbb < 0.95),
            "cc_pass": pre_selection & (Txcc > 0.95),
            "cc_fail": pre_selection & (Txcc < 0.95),
            "bbcc_fail": pre_selection & (Txbb < 0.95) & (Txcc < 0.95),
            "bbfail_ccpass": pre_selection & (Txbb < 0.95) & (Txcc > 0.95),
            "bbcc_pass": pre_selection & (Txbb > 0.95) & (Txcc > 0.95),
        }

        # Fill histograms
        for category, selection in selection_dict.items():
            h.fill(
                var_series[selection],  # Use the (potentially transformed) var_series
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
    dir_name = "gmachado/25Nov11_v14_private"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    filters = None
    variable_to_plot = args.variable

    # 1. Define base columns ALWAYS needed for selections
    base_columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
        "FatJet0_pnetTXcc",
    ]

    # 2. Add columns needed for the region
    if region == "control-zgamma":
        # You only check for Photon200 in your trigger logic
        base_columns.extend(
            ["Photon0_pt", "Photon200", "Photon110EB_TightID_TightIso"]
        )  # , "Photon30EB_TightID_TightIso"])

    # 3. Add columns needed for the specific variable
    # (This uses your existing axis_to_column map)
    var_cols = axis_to_column[variable_to_plot]
    if isinstance(var_cols, str):
        # Handle 'delta_phi_photon_jet' which is not a real column
        if var_cols == "delta_phi_photon_jet":
            base_columns.extend(["Photon0_phi", "FatJet0_phi"])
        else:
            base_columns.append(var_cols)
    elif isinstance(var_cols, list):
        base_columns.extend(var_cols)

    # 4. Create the final lists
    load_columns_mc = list(set(base_columns + ["GenFlavor"]))
    load_columns_data = list(set(base_columns))  # 'set' avoids duplicates

    data_dir = Path(path_to_dir) / year

    # ---- Select correct data samples and mc ----

    if region == "control-zgamma":
        data_samples = data_by_year_zgamma.get(year, {})
    elif region == "control-tt":
        data_samples = data_by_year_muon.get(year, {})
    else:
        # Default to JetMET for signal regions
        data_samples = data_by_year.get(year, {})

    samples = {
        **common_mc,
        "data": data_samples,
    }

    # Define which histograms to create based on the region
    # For zgamma, we add the new ones. For other regions, we just make the msd1 plot.
    # Define which histograms to create based on the region
    variable_to_plot = args.variable
    hists_to_make = [variable_to_plot]

    print(f"Will create histogram files for: {', '.join(hists_to_make)}")

    # --- *** MAIN STRUCTURE MODIFIED *** ---
    # Loop over each histogram we want to create a file for
    for hist_name in hists_to_make:
        print(f"\n--- Processing variable: {hist_name} ---")

        # Create a new, empty dictionary for *this variable's* histograms
        histograms = {}

        # --- This is the original loop, now INSIDE the hist_name loop ---
        for process, datasets in samples.items():
            load_columns = load_columns_data if process == "data" else load_columns_mc
            print(f"Processing {process} for year {year}...")

            # Create a new histogram for each process
            h = hist.Hist(
                axis_to_histaxis[hist_name],
                axis_to_histaxis["pt1"],
                axis_to_histaxis["category"],
                axis_to_histaxis["genflavor"],
            )

            # Loop through each dataset within the process
            for dataset in datasets:
                # Load only one dataset at a time to save memory
                # search_path = Path(data_dir / dataset / "parquet" / region)
                # print(f"\n[DEBUG] Script is searching for files in: {search_path}\n")

                events = utils.load_samples(
                    data_dir,
                    {process: [dataset]},  # Pass a list with a single dataset
                    columns=load_columns,
                    region=region,
                    filters=filters,
                )

                if not events:
                    print(f"No events found for dataset {dataset} in year {year}. Skipping.")
                    continue

                # Fill the histogram with the events from this single dataset
                h = fill_ptbinned_histogram(h, events, hist_name, region)

                # --- ADD THESE TWO LINES ---
                del events  # Explicitly delete the large dataframe
                gc.collect()
                # --- END OF FIX ---
            # --- ADDED CHECK ---
            # Only add the histogram to our dictionary if it has entries
            if h.sum() == 0:
                print(
                    f"WARNING: No events were found for the entire '{process}' process group. Skipping."
                )
                continue
            # Add the fully filled histogram for the process to the dictionary
            histograms[process] = h

        # --- *** FILE SAVING MODIFIED *** ---
        # Save a separate file FOR EACH variable
        output_dir = Path(args.outdir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # The output filename now includes the variable name
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
