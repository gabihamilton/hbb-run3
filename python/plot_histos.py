#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import yaml
from plotting import ratio_plot

from hbb.common_vars import LUMI

hep.style.use("CMS")


def plot_ptbin_stack(hists, category, year, outdir, save_individual, region):
    pt_axis_name = "pt1"

    # Load style configuration
    style_path = Path("style_hbb.yaml")
    with style_path.open() as stream:
        style = yaml.safe_load(stream)

    mass_lo = 115
    mass_hi = 135

    first_hist = next(iter(hists.values()))
    pt_axis = first_hist.axes[pt_axis_name]
    pt_edges = pt_axis.edges

    print("Axis order:", first_hist.axes.name)

    for i in range(len(pt_edges) - 1):
        histograms_to_plot = {}
        pt_low = pt_edges[i]
        pt_high = pt_edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        for process, h in hists.items():
            # --- CRITICAL CHANGE 1: Sum over the new 'genflavor' axis ---
            # First, select the pt and category, leaving a 2D hist (msd vs flavor)
            h_proj_2d = h[:, i_start, category, :]
            # Then, project it down to the 1D mass axis, summing over flavors
            h_proj = h_proj_2d.project("msd1")

            if process == "data":
                # Blind the mass window
                edges = h_proj.axes[0].edges
                mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                data_val = h_proj.values()
                data_val[mask] = 0
                h_proj.values()[:] = data_val

            histograms_to_plot[process] = h_proj

        bkg_order = ["zjets", "wjets", "other", "top"]
        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=["hbb"],
            bkgs=bkg_order,
            onto="qcd",
            style=style,
        )

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
        # Use the region in the output filename for clarity
        fig.savefig(
            f"{outdir}/{year}_{region}_{category}_ptbin{pt_low}_{pt_high}.png",
            dpi=300,
            bbox_inches="tight",
        )

        if save_individual:
            # Save individual histograms for debugging
            for process, histo in histograms_to_plot.items():
                fig_indiv, ax_indiv = plt.subplots(figsize=(8, 6))
                hep.histplot(histo, ax=ax_indiv, histtype="step", color="black")
                ax_indiv.set_title(f"{process} - {category} - ptbin {pt_low}_{pt_high}")
                ax_indiv.set_ylabel("Events")
                ax_indiv.grid(True)
                plt.savefig(
                    f"hist_{process}_{category}_pt{pt_low}_{pt_high}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close(fig_indiv)


def main(args):
    histograms = {}
    for year in args.year:
        # --- CRITICAL CHANGE 2: Use the region to find the correct input file ---
        print(f"Loading histograms for year: {year}, region: {args.region}")
        pkl_path = Path(args.indir) / f"histograms_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            print(f"Error: File not found at {pkl_path}. Skipping.")
            continue

        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)

        print("Histograms loaded successfully!")
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

    category = "pass"  # Assuming we are plotting the 'pass' category
    print(
        f"Plotting histograms for category: {category}, year: {year}, output directory: {args.outdir} \n"
    )
    plot_ptbin_stack(
        histograms,
        category,
        year,
        args.outdir,
        save_individual=args.save_individual,
        region=args.region,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",  # Accepts one or more arguments, if more arguments are given, then histograms are summed
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument(
        "--indir",
        help="Input directory containing histograms",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="Output directory for saving histograms",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save_individual",
        help="Save individual histograms for each process",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--region",
        help="Analysis region to plot",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    main(args)
