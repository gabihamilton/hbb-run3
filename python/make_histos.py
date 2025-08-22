#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import warnings
from pathlib import Path

import hist
import numpy as np
from common import common_mc, data_by_year

from hbb import utils

# Define the possible ptbins
ptbins = np.array([300, 450, 500, 550, 600, 675, 800, 1200])

# Define the histogram axes
axis_to_histaxis = {
    # "pt1": hist.axis.Regular(30, 300, 900, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    # "pt2": hist.axis.Regular(30, 300, 900, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "pt1": hist.axis.Variable(ptbins, name="pt1", label=r"Jet 0 $p_{T}$ [GeV]"),
    "pt2": hist.axis.Variable(ptbins, name="pt2", label=r"Jet 1 $p_{T}$ [GeV]"),
    "msd1": hist.axis.Regular(23, 40, 201, name="msd1", label="Jet 0 $m_{sd}$ [GeV]"),
    "mass1": hist.axis.Regular(30, 0, 200, name="mass1", label="Jet 0 PNet mass [GeV]"),
    "category": hist.axis.StrCategory([], name="category", label="Category", growth=True),
}

# add more as needed
axis_to_column = {
    "pt1": "FatJet0_pt",
    "pt2": "FatJet1_pt",
    "msd1": "FatJet0_msd",
    "mass1": "FatJet0_pnetMass",
    "category": "category",
}


def fill_ptbinned_histogram(events, axis):
    """
    Fills histogram after event selection of any variable.
    The histogram has a pt-binned axis for FatJet0.

    :param events: Dictionary of events loaded from parquet files.
    :param axis: String to fill the histogram for. Needs to be one of the keys in axis_to_histaxis.
    :return: histogram filled with the selected events.
    """

    if axis == "pt1":
        # Ensure the axis is valid
        warnings.warn(
            f"Cannot use pt1 axis for histogram filling since that is used already. Axis: {axis}",
            stacklevel=2,
            category=UserWarning,
        )
        exit(1)

    h = hist.Hist(axis_to_histaxis[axis], axis_to_histaxis["pt1"], axis_to_histaxis["category"])

    for _process_name, data in events.items():
        weight_val = data["finalWeight"].astype(float)
        var = data[axis_to_column[axis]]

        ### Event selection

        # Leading FatJet
        Txbb = data["FatJet0_pnetTXbb"]
        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]

        # Pre-selection criteria
        pre_selection = (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)

        # Define the selection dictionary
        selection_dict = {
            "pass": pre_selection & (Txbb > 0.95),
            "fail": pre_selection & (Txbb < 0.95),
        }

        # Fill histograms
        for category, selection in selection_dict.items():
            h.fill(
                var[selection],
                pt[selection],
                category=category,
                weight=weight_val[selection],
            )

    return h


def main(args):
    year = args.year

    # Set the main directory where parquet files are stored
    # TODO: make the dir_name an argument
    MAIN_DIR = "/eos/uscms/store/user/lpchbbrun3/"
    dir_name = "cmantill/25Jun25_v12"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    # Define the columns to load for each sample
    load_columns = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        # "FatJet0_pnetMass",
        "FatJet0_pnetTXbb",
    ]
    # Example filters
    # filters = [
    #    ("FatJet0_pt", ">", 300),  # Filter for FatJet0
    #    ("FatJet0_msd", ">", 40),  # Filter for FatJet0
    # ]
    filters = None

    # Initialize histogram dictionary
    # each key will correspond to one process
    histograms = {}

    data_dir = Path(path_to_dir) / year

    # list of all datasets in the directory
    # full_dataset_list = [
    #    p.name for p in data_dir.iterdir() if p.is_dir()
    # ]
    # print("Full samples list:", full_dataset_list)

    # lists of samples
    samples = {
        **common_mc,
        "data": data_by_year[year],
    }

    # Loop through each process individually to avoid loading everything at once
    for process, datasets in samples.items():
        # Load only one sample at a time
        events = utils.load_samples(
            data_dir,
            {process: datasets},  # Dictionary with one process
            columns=load_columns,
            region=args.region,
            filters=filters,
        )

        # Fill histograms with the loaded events dictionary
        h = fill_ptbinned_histogram(events, "msd1")
        if process not in histograms:
            histograms[process] = h
        else:
            histograms[process] += h  # Combine histograms if process already exists

    # Define the output file
    output_file = Path(f"histograms_{year}.pkl")

    # Save histograms to a pickle
    with output_file.open("wb") as f:
        pickle.dump(histograms, f)

    print(f"Histograms saved to {output_file}")


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
        choices=["signal-all", "signal-ggf"],  # add more as needed
    )
    args = parser.parse_args()

    main(args)
