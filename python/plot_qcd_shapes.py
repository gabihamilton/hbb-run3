#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")

def plot_qcd_pass_fail(hists, year_str, outdir, region, norm_type):
    """
    For each pt bin, plots the normalized 'pass' and 'fail' distributions for the QCD sample.
    """
    
    if "qcd" not in hists:
        print("No 'qcd' histogram found in the input file. Exiting.")
        return
        
    h_qcd = hists["qcd"]
    pt_axis = h_qcd.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i+1]
        i_start = pt_axis.index(pt_low)
        
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        h_pass = h_qcd[:, i_start, "pass", :].project("msd1")
        h_fail = h_qcd[:, i_start, "fail", :].project("msd1")

        if h_pass.sum() == 0 or h_fail.sum() == 0:
            print("  Skipping pt bin due to zero events in pass or fail.")
            continue

        # --- NEW NORMALIZATION LOGIC TO MATCH REFERENCE ---
        bin_width = h_pass.axes[0].widths[0]

        # --- NEW: Choose normalization based on the --norm-type argument ---
        if norm_type == "shape":
            # Probability density (normalized to unit area, then divided by bin width)
            h_pass_toplot = (h_pass / h_pass.sum()) / bin_width
            h_fail_toplot = (h_fail / h_fail.sum()) / bin_width
            ylabel = f"Normalized Events / {bin_width:g} GeV"
        
        elif norm_type == "density":
            # Physical density (fail region scaled to pass region, then divided by bin width)
            pass_yield = h_pass.sum()
            fail_yield = h_fail.sum()
            h_fail_scaled = h_fail * (pass_yield / fail_yield)
            h_pass_toplot = h_pass / bin_width
            h_fail_toplot = h_fail_scaled / bin_width
            ylabel = f"Events / {bin_width:g} GeV"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the scaled 'fail' region as blue points
        hep.histplot(h_fail_toplot, ax=ax, label="QCD MC fail", color="blue", histtype="errorbar", yerr=True if norm_type=="density" else False)
        
        # Plot the 'pass' region as black points
        hep.histplot(h_pass_toplot, ax=ax, label="QCD MC pass", color="black", histtype="errorbar", yerr=True if norm_type=="density" else False)
        
        ax.legend()
        ax.set_xlabel("Jet $m_{sd}$ [GeV]")
        # Update the y-axis label
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        hep.cms.label("Private Work", data=False, ax=ax, com=13.6, year=year_str, loc=0)
        legend = ax.get_legend()
        if legend:  # Check if a legend exists before modifying it
            legend.set_title(
                f"{pt_low:g} < $p_T$ < {pt_high:g} GeV",
                prop={"size": 14},
            )
        #ax.set_title(f"{pt_low:g} < $p_T$ < {pt_high:g} GeV", loc='right')

        output_name = f"{outdir}/{year_str}_{region}_qcd_{norm_type}_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_name}")
        plt.close(fig)

def main(args):
    histograms = {}
    if len(args.year) > 3:
        year_str = "all-years"
    else:
        year_str = "-".join(args.year)

    for year in args.year:
        pkl_path = Path(args.indir) / f"histograms_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            print(f"Error: File not found at {pkl_path}. Skipping.")
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)
            # Sum histograms across years
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
    
    plot_qcd_pass_fail(histograms, year_str, args.outdir, args.region, args.norm_type)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot QCD pass/fail shape comparisons.")
    parser.add_argument("--year", help="List of years", type=str, required=True, nargs="+", choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--indir", help="Input directory for .pkl files", type=str, default="histograms")
    parser.add_argument("--outdir", help="Output directory for plots", type=str, default="plots/qcd_shapes")
    parser.add_argument("--region", help="Analysis region", type=str, required=True)
    parser.add_argument(
        "--norm-type",
        help="Type of normalization to apply",
        type=str,
        default="shape",
        choices=["shape", "density"],
    )
    args = parser.parse_args()
    main(args)
