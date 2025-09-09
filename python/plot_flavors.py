#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import hist
import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")

# Maps the integer genflavor code to a string name
flavor_map = {
    3: "b-jet",
    2: "c-jet",
    1: "light-jet",
}


def plot_ptbin_stack_flavors(hists, category, year, outdir, region, style):
    """Plots a stacked histogram for each pt bin, splitting W/Z jets by flavor."""

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low = pt_axis.edges[i]
        pt_high = pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}

        # --- NEW LOGIC TO SPLIT BY FLAVOR ---
        for process, h in hists.items():
            if process in ["wjets", "zjets"]:
                # For W and Z jets, loop over the flavors and create a new histogram for each

                # First, select the pt bin and category, leaving a 2D hist (msd vs genflavor)
                h_2d = h[:, i_start, category, :]

                for flavor_code, flavor_name in flavor_map.items():
                    # Create a new key for this specific flavor, e.g., "zjets_b-jet"
                    new_key = f"{process}_{flavor_name}"
                    # Select the flavor by its integer code and project to 1D msd axis
                    histograms_to_plot[new_key] = h_2d[:, hist.loc(flavor_code)]
            else:
                # For all other processes (QCD, Top, Data, etc.), just sum over the flavors
                histograms_to_plot[process] = h[:, i_start, category, :].project("msd1")

        # Define the new stacking order with flavor components
        bkg_order = [
            "hbb",
            "other",
            "top",
            "wjets_light-jet",
            "wjets_c-jet",
            "zjets_light-jet",
            "zjets_c-jet",
            "zjets_b-jet",
        ]

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=[],  # No separate signals for this plot
            bkgs=bkg_order,
            onto="qcd",
            style=style,
        )

        # Style and save the plot
        luminosity = (
            LUMI[year] / 1000.0
            if "-" not in year
            else sum(LUMI[y] / 1000.0 for y in year.split("-"))
        )
        # This places "Private Work" on the LEFT, and lumi/energy on the RIGHT
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year,
            loc=0,
        )

        # --- NEW: Set the legend title instead of the plot title ---
        legend = ax.get_legend()
        if legend:  # Check if a legend exists before modifying it
            legend.set_title(
                f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV",
                prop={"size": 14},
            )

        output_name = f"{outdir}/{year}_{region}_{category}_byflavor_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"Saved plot to {output_name}")
        plt.close(fig)


def main(args):
    histograms = {}
    # year_str = "-".join(args.year)

    year_str = "all-years" if len(args.year) > 3 else "-".join(args.year)

    for year in args.year:
        print(f"Loading histograms for year: {year}, region: {args.region}")
        pkl_path = Path(args.indir) / f"histograms_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            print(f"Error: File not found at {pkl_path}. Skipping.")
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)
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

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    style_path = Path("style_hbb.yaml")
    with style_path.open() as f:
        style = yaml.safe_load(f)

    for category in ["pass", "fail"]:
        print(f"Plotting histograms for category: {category}, year: {year_str}...")
        plot_ptbin_stack_flavors(histograms, category, year_str, args.outdir, args.region, style)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot stacked histograms with flavor breakdown.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
        "--indir", help="Input directory containing histograms", type=str, required=True
    )
    parser.add_argument("--outdir", help="Output directory for plots", type=str, required=True)
    parser.add_argument("--region", help="Analysis region to plot", type=str, required=True)
    args = parser.parse_args()
    main(args)
