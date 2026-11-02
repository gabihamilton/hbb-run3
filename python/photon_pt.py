#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import yaml

# Assuming common.py and hbb/utils.py are accessible
from common import common_mc, data_by_year
from hbb import utils
from hbb.common_vars import LUMI

# Set plotting style
plt.style.use(hep.style.CMS)


def main(args):
    # --- MODIFIED ---
    # Create a string for all years for filenames and labels
    year_str = "all-years" if len(args.year) > 3 else "-".join(args.year)
    total_lumi = sum(LUMI[y] for y in args.year)

    region = "control-zgamma"  # Hardcode the region
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running for {year_str}, region={region}")
    # --- END MODIFIED ---

    # --- Setup Paths ---
    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/"
    # --- MODIFIED: Pointing to the 25Oct29_v12 path from your logs ---
    dir_name = "gmachado/25Oct29_v12"
    path_to_dir = f"{MAIN_DIR}/{dir_name}/"

    # --- MODIFIED: Columns defined outside loop ---
    load_columns_mc = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
        "Photon0_pt",  # <-- CHANGED
        "GenFlavor",
    ]
    load_columns_data = [
        "weight",
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
        "Photon0_pt",  # <-- CHANGED
    ]
    filters = None
    # --- END MODIFIED ---

    # --- Define Histogram ---
    # We will fill all processes into this single histogram
    # --- MODIFIED: Defined outside the loop ---
    h = hist.Hist(
        hist.axis.Regular(50, 0, 500, name="photon_pt", label=r"Photon $p_{T}$ [GeV]"),
        hist.axis.StrCategory([], name="process", label="Process", growth=True),
        hist.axis.StrCategory([], name="category", label="Category", growth=True),
    )
    # --- END MODIFIED ---

    # --- MODIFIED: Loop over all years ---
    all_samples = {}  # To get a combined list of MC for plotting

    # --- ADDED: Define processes to plot individually ---
    # All other MC will be grouped into 'other'
    known_processes = {"data", "gjets", "wgamma", "zgamma", "tt"}
    # --- END ADDED ---

    for year in args.year:
        print(f"\nProcessing year: {year}")
        data_dir = Path(path_to_dir) / year
        samples = {
            **common_mc,
            "data": data_by_year[year],
        }
        all_samples.update(samples)  # Add any new processes
        # --- END MODIFIED ---

        # --- Loop over all processes and fill histogram ---
        for process, datasets in samples.items():
            load_columns = load_columns_data if process == "data" else load_columns_mc
            print(f"  Processing {process}...")

            # --- MODIFIED: Determine process name for plotting ---
            # Group unknown processes into 'other'
            if process not in known_processes:
                plot_process_name = "other"
            else:
                plot_process_name = process
            # --- END MODIFIED ---

            # --- MODIFIED: Loop over each dataset individually to save memory ---
            for dataset in datasets:
                print(f"    Loading dataset: {dataset}...")
                print(f"    Will search in path: {Path(data_dir / dataset / 'parquet' / region)}")

                events_dict = utils.load_samples(
                    data_dir,
                    {process: [dataset]},  # <-- Load only one dataset
                    columns=load_columns,
                    region=region,
                    filters=filters,
                )

                # Check if the process key exists OR if the dataframe is empty
                if process not in events_dict or events_dict[process].empty:
                    print(f"    No events found for dataset {dataset}. Skipping.")
                    continue

                data = events_dict[process]  # Get the dataframe (now for just one dataset)

                # --- Apply Selections ---

                # Check that all required columns were loaded
                required_cols = ["Photon0_pt", "FatJet0_pt", "FatJet0_msd", "FatJet0_pnetTXbb"]
                # --- BUG FIX 2: Check against required_cols, not data.columns ---
                if not all(col in data.columns for col in required_cols):
                    print(f"    [WARN] Missing one or more columns in {process} ({dataset}). Skipping.")
                    continue

                weight_val = data["finalWeight"].astype(float)
                photon_pt = data["Photon0_pt"]

                # Define the *exact same* pre-selection as the other script
                Txbb = data["FatJet0_pnetTXbb"]
                msd = data["FatJet0_msd"]
                pt = data["FatJet0_pt"]

                # Base jet selection
                pre_selection = (msd > 40) & (msd < 200) & (pt > 300) & (pt < 1200)

                # Add a basic photon selection (e.g., pT > 20 GeV)
                photon_selection = (photon_pt > 20)

                selection_dict = {
                    "pass": pre_selection & photon_selection & (Txbb > 0.95),
                    "fail": pre_selection & photon_selection & (Txbb < 0.95),
                }

                # Fill histogram for both categories
                for category, selection in selection_dict.items():
                    if np.sum(selection) == 0:
                        print(f"    No events in {category} for {dataset}")
                        continue

                    h.fill(
                        photon_pt[selection],
                        process=plot_process_name,  # <-- MODIFIED: Use grouped name
                        category=category,
                        weight=weight_val[selection],
                    )
            # --- END: Inner dataset loop ---

    print("\nAll processes and years filled. Now plotting...")

    # --- ADDED: Load style file ---
    style_path = Path("style_hbb.yaml")
    if style_path.exists():
        with style_path.open() as f:
            style = yaml.safe_load(f)
        print("Loaded style from style_hbb.yaml")
    else:
        style = {}
        print("Warning: style_hbb.yaml not found. Using default styles.")
    # --- END ADDED ---

    # --- Plotting ---

    # Plot 'pass' and 'fail' categories separately
    for category in h.axes["category"]:
        print(f"  Plotting {category} category...")

        h_cat = h[{"category": category}]

        if h_cat.sum() == 0:
            print(f"  No data in {category} category, skipping plot.")
            continue

        # --- MODIFIED: Define processes to plot ---
        bkg_order = ["wgamma", "zgamma", "other", "tt"]
        onto = "gjets"

        h_data = h_cat[{"process": "data"}]
        h_onto = h_cat[{"process": onto}]
        h_stack_components = h_cat[{"process": bkg_order}]
        # --- END MODIFIED ---

        fig, ax = plt.subplots(figsize=(10, 8))

        # --- MODIFIED: Plot 'onto' then stack, using style ---
        # Plot the 'onto' background first
        if h_onto.sum() > 0:
            # --- BUG FIX: Unpack style dictionary ---
            h_onto.plot(ax=ax, histtype="fill", **style.get(onto, {}))

        # --- BUG FIX: Manually plot stack to avoid mplhep error ---
        if h_stack_components.sum() > 0:
            # Create a list of (name, histogram) tuples, filtering out empty ones
            hists_with_names = []
            for p in bkg_order:
                h_proc = h_stack_components[{"process": p}]
                if h_proc.sum() > 0:
                    hists_with_names.append((p, h_proc))
            
            if hists_with_names:
                # Unzip the list of tuples
                processes_to_plot, stack_hists_to_plot = zip(*hists_with_names)
                
                # Manually get labels and colors from the style dict
                labels = [style.get(p, {}).get("label", p) for p in processes_to_plot]
                colors = [style.get(p, {}).get("color", None) for p in processes_to_plot]

                hep.histplot(
                    list(stack_hists_to_plot), # Pass the list of histograms
                    ax=ax,
                    stack=True,
                    histtype="fill",
                    label=labels,
                    color=colors
                )
        # --- END BUG FIX ---

        # Plot data
        if h_data.sum() > 0:
            # --- BUG FIX: Pass label/color via defaults in style.get ---
            h_data.plot(ax=ax, histtype="errorbar", **style.get("data", {"label": "Data", "color": "k"}))
        # --- END MODIFIED ---

        ax.legend(title=f"{category.capitalize()} Region", prop={"size": 10})  # Smaller legend
        ax.set_ylabel("Events")
        ax.set_yscale("log")  # Log scale is usually helpful
        ax.set_ylim(0.1, ax.get_ylim()[1] * 10)  # Adjust ylim for log scale

        # --- MODIFIED ---
        hep.cms.label(
            "Work in Progress",
            data=True,
            year=year_str,
            ax=ax,
            lumi=total_lumi / 1000.0,  # Convert to fb^-1
            lumi_format="{:0.1f}",
        )

        plot_filename = output_dir / f"photon_pt_{year_str}_{region}_{category}.png"
        # --- END MODIFIED ---

        fig.savefig(plot_filename)
        print(f"  Saved plot to {plot_filename}")
        plt.close(fig)

    # --- ADDED: Plotting for all categories summed ---
    print("  Plotting all categories (pass+fail) summed...")

    # Sum over the category axis
    h_all_cats = h[{"category": sum}]

    if h_all_cats.sum() == 0:
        print("  No data in summed histogram, skipping plot.")
    else:
        # --- MODIFIED: Define processes to plot ---
        bkg_order = ["wgamma", "zgamma", "other", "tt"]
        onto = "gjets"

        h_data_all = h_all_cats[{"process": "data"}]
        h_onto_all = h_all_cats[{"process": onto}]
        h_stack_all_components = h_all_cats[{"process": bkg_order}]
        # --- END MODIFIED ---

        fig_all, ax_all = plt.subplots(figsize=(10, 8))

        # --- MODIFIED: Plot 'onto' then stack, using style ---
        if h_onto_all.sum() > 0:
            # --- BUG FIX: Unpack style dictionary ---
            h_onto_all.plot(ax=ax_all, histtype="fill", **style.get(onto, {}))

        # --- BUG FIX: Manually plot stack to avoid mplhep error ---
        if h_stack_all_components.sum() > 0:
            # Create a list of (name, histogram) tuples, filtering out empty ones
            hists_with_names_all = []
            for p in bkg_order:
                h_proc = h_stack_all_components[{"process": p}]
                if h_proc.sum() > 0:
                    hists_with_names_all.append((p, h_proc))

            if hists_with_names_all:
                # Unzip the list of tuples
                processes_to_plot_all, stack_hists_to_plot_all = zip(*hists_with_names_all)
                
                # Manually get labels and colors from the style dict
                labels_all = [style.get(p, {}).get("label", p) for p in processes_to_plot_all]
                colors_all = [style.get(p, {}).get("color", None) for p in processes_to_plot_all]
                
                hep.histplot(
                    list(stack_hists_to_plot_all), # Pass the list of histograms
                    ax=ax_all,
                    stack=True,
                    histtype="fill",
                    label=labels_all,
                    color=colors_all
                )
        # --- END BUG FIX ---

        # Plot data
        if h_data_all.sum() > 0:
            # --- BUG FIX: Pass label/color via defaults in style.get ---
            h_data_all.plot(ax=ax_all, histtype="errorbar", **style.get("data", {"label": "Data", "color": "k"}))
        # --- END MODIFIED ---

        ax_all.legend(title="All Categories (Pass+Fail)", prop={"size": 10}) # Smaller legend
        ax_all.set_ylabel("Events")
        ax_all.set_yscale("log")  # Log scale is usually helpful
        ax_all.set_ylim(0.1, ax_all.get_ylim()[1] * 10)  # Adjust ylim for log scale

        hep.cms.label(
            "Work in Progress",
            data=True,
            year=year_str,
            ax=ax_all,
            lumi=total_lumi / 1000.0,  # Convert to fb^-1
            lumi_format="{:0.1f}",
        )

        plot_filename_all = output_dir / f"photon_pt_{year_str}_{region}_all-categories.png"

        fig_all.savefig(plot_filename_all)
        print(f"  Saved plot to {plot_filename_all}")
        plt.close(fig_all)
    # --- END ADDED ---


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Photon pT for Z-gamma control region.")
    # --- MODIFIED ---
    parser.add_argument(
        "--year",
        help="List of years to run over",
        type=str,
        required=True,
        nargs="+",  # Accept multiple years
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    # --- END MODIFIED ---
    parser.add_argument(
        "--outdir", help="Output directory to save plots.", type=str, default="plots"
    )
    args = parser.parse_args()

    main(args)


