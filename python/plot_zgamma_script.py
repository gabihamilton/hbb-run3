#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from plotting import (
    ratio_plot,  # Assumes ratio_plot is in a local file named plotting.py
)

from hbb.common_vars import LUMI  # Assumes LUMI is in a local file

hep.style.use("CMS")


def plot_zgamma_control_region(hists, category, year, outdir):
    """
    This function is specifically designed to plot the Z+gamma control region.
    It slices the histograms by pt bins and creates a stacked plot for each.
    """
    pt_axis_name = "pt1"

    # Load style configuration from a YAML file
    try:
        with Path("style_hbb.yaml").open() as stream:
            style = yaml.safe_load(stream)
    except FileNotFoundError:
        print("Warning: style_hbb.yaml not found. Using empty style dictionary.")
        style = {}

    # Get the first histogram to inspect its axes
    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes[pt_axis_name]
    pt_edges = pt_axis.edges

    print("Axis order:", first_hist.axes.name)

    # Loop over each momentum (pt) bin to make a separate plot
    for i in range(len(pt_edges) - 1):
        histograms_to_plot = {}

        # Define the pt bin edges for this iteration
        pt_low = pt_edges[i]
        pt_high = pt_edges[i + 1]
        i_start = pt_axis.index(pt_low)

        print(f"Processing pt bin: {pt_low} - {pt_high}")

        # Project the 3D histogram to a 1D mass histogram for the current pt bin
        for process, h in hists.items():
            # Select the pt bin and the category ('pass' or 'fail')
            h_proj = h[:, i_start, category]
            histograms_to_plot[process] = h_proj

        # --- Z+Gamma Control Region Specific Configuration ---
        # Define which processes are backgrounds and which is the main component to stack on.
        # In this control region, 'zgamma' is the main process.
        zgamma_bkgs = ["wjets", "other", "top"]

        # Create the stacked plot with a ratio panel
        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=[],  # No signal in a control region
            bkgs=zgamma_bkgs,
            onto="zgamma",  # Stack other backgrounds on top of zgamma
            style=style,
        )

        # Add CMS labels to the plot
        luminosity = (
            LUMI[year] / 1000.0
            if "-" not in year
            else sum(LUMI[y] / 1000.0 for y in year.split("-"))
        )
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.0f}",
            com=13.6,
            year=year,
        )

        # Save the plot to a file
        output_filename = f"{outdir}/{year}_control-zgamma_{category}_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_filename, dpi=300, bbox_inches="tight")
        print(f"Saved plot: {output_filename}")
        plt.close(fig)


def main(args):
    # Load the histograms from the pickle file(s)
    histograms = {}
    region = "control-zgamma"  # Hardcode the region

    for year in args.year:
        print(f"Loading histograms for year: {year}, region: {region}")
        pkl_path = Path(args.indir) / f"histograms_{year}_{region}.pkl"

        if not pkl_path.exists():
            print(f"Error: Input file not found at {pkl_path}")
            continue

        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)

        # Combine histograms if multiple years are provided
        if not histograms:
            histograms = histograms_tmp
        else:
            for process, h in histograms_tmp.items():
                if process in histograms:
                    histograms[process] += h
                else:
                    histograms[process] = h

    if not histograms:
        print("No histograms were loaded. Exiting.")
        return

    print("Processes in loaded histograms:", histograms.keys())

    # Create a single year string for labeling if multiple years are combined
    year_str = args.year[0] if len(args.year) == 1 else "-".join(args.year)

    # Create the output directory if it doesn't exist
    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # We typically look at the 'pass' category for analysis
    category = "pass"

    print(
        f"Plotting histograms for category: {category}, year(s): {year_str}, output directory: {args.outdir}\n"
    )
    plot_zgamma_control_region(histograms, category, year_str, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make plots for the Z+gamma control region.")
    parser.add_argument(
        "--year",
        help="List of years to process. Histograms will be summed.",
        type=str,
        required=True,
        nargs="+",
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
        "--indir",
        help="Input directory containing the histogram pickle files.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="Output directory for saving plots.",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
