#!/usr/bin/env python3
"""
Unified and flexible plotting script for the Hbb analysis.

This script serves as a central manager for creating various types of plots
from the histogram `.pkl` files. Its behavior is controlled by three main
command-line arguments: `--stack-by`, `--category-type`, and `--pt-scope`.

This allows for the generation of a wide range of plots, from detailed,
binned distributions to high-level, fully inclusive summaries.
"""
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

# --- Globals ---
flavor_map = {3: "b-jet", 2: "c-jet", 1: "light-jet"}
mass_lo = 115
mass_hi = 135

# --- Main Plotting Function ---
def plot_manager(hists, style, year_str, year_list, region, outdir, stack_by, category_type, pt_scope):
    """
    Master plotting function that handles all plot variations based on arguments.
    """
    
    # 1. Prepare histograms based on Pass/Fail category choice
    if category_type == "inclusive":
        # Sum over the 'category' axis to combine pass and fail
        hists_to_process = {p: h[:, :, hist.sum, :] for p, h in hists.items() if h.sum() > 0}
        categories_to_loop = ["inclusive"]
    else: # 'separate'
        hists_to_process = hists
        categories_to_loop = ["pass", "fail"]

    first_hist = next((h for h in hists_to_process.values() if h.sum() > 0), None)
    if not first_hist:
        print("All histograms are empty for the selection. No plots to make.")
        return
    pt_axis = first_hist.axes["pt1"]

    # 2. Determine the set of pT bins to loop over
    if pt_scope == "binned":
        pt_bins_to_loop = list(range(len(pt_axis.edges) - 1))
    else: # 'inclusive'
        pt_bins_to_loop = [None] # Use None as a placeholder for a single, inclusive loop

    # 3. Main plotting loop
    for category in categories_to_loop:
        for i_pt in pt_bins_to_loop:
            
            # --- Prepare histograms for this specific plot ---
            histograms_to_plot = {}
            # Project by pT bin if binned, otherwise sum over pT axis
            if i_pt is not None:
                pt_low, pt_high = pt_axis.edges[i_pt], pt_axis.edges[i_pt+1]
                for process, h in hists_to_process.items():
                    # The slice must be 4D if category is not inclusive
                    if category == "inclusive":
                         # h is already 3D (msd, pt, flavor), just slice by pt
                        histograms_to_plot[process] = h[:, pt_axis.index(pt_low), :].project("msd1")
                    else:
                        if h.sum() == 0 or category not in h.axes["category"]:
                            continue
                        # h is 4D, slice by pt and category
                        histograms_to_plot[process] = h[:, pt_axis.index(pt_low), category, :].project("msd1")
            else: # pt-inclusive
                for process, h in hists_to_process.items():
                    if category != "inclusive" and (h.sum() == 0 or category not in h.axes["category"]):
                        continue
                    histograms_to_plot[process] = h[:, :, :].project("msd1")
            
            # --- Logic for stacking by flavor ---
            if stack_by == "flavor":
                flavored_hists = {}
                for process, h in histograms_to_plot.items():
                    if process in ["wjets", "zjets"]:
                        # For flavor plots, we need the un-projected hist
                        h_unproj = hists_to_process[process]
                        h_2d = h_unproj[:, i_pt if i_pt is not None else hist.sum, :]
                        for code, name in flavor_map.items():
                            flavored_hists[f"{process}_{name}"] = h_2d[:, hist.loc(code)]
                    else:
                        flavored_hists[process] = h
                histograms_to_plot = flavored_hists

            # --- Data Blinding (only for separate pass/fail) ---
            if "data" in histograms_to_plot and category == "pass":
                h_proj = histograms_to_plot["data"]
                edges, data_val = h_proj.axes[0].edges, h_proj.values()
                mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                data_val[mask] = 0
                h_proj.values()[:] = data_val

            # --- Define region-specific backgrounds and signals ---
            # (This logic is now applied to all plot types)
            if "control-zgamma" in region:
                signals, bkg_order, onto = [], ["ttgamma", "wgamma", "gjets", "qcd", "other", "top"], "zgamma"
            elif "control-tt" in region:
                signals, bkg_order, onto = [], ["wjets", "zjets", "qcd", "other", "hbb"], "top"
            else:
                signals, bkg_order, onto = ["hbb"], ["zjets", "wjets", "other", "top"], "qcd"

            if stack_by == 'flavor':
                if "control-tt" in region:
                    bkg_order = ["wjets_light-jet", "wjets_c-jet", "zjets_light-jet", "zjets_c-jet", "zjets_b-jet", "qcd", "other", "hbb"]
                elif "control-zgamma" not in region: # Default flavor order for signal region
                    bkg_order = ["hbb", "other", "top", "wjets_light-jet", "wjets_c-jet", "zjets_light-jet", "zjets_c-jet", "zjets_b-jet"]

            if onto not in histograms_to_plot and not any(k.startswith(onto) for k in histograms_to_plot):
                print(f"    WARNING: Main background '{onto}' not found. Skipping plot.")
                continue

            # --- Final Plotting Call ---
            legend_title = f"{category.capitalize()} Region"
            if i_pt is not None:
                legend_title += f", {pt_low:g} < $p_T$ < {pt_high:g} GeV"
            else:
                legend_title += ", all $p_T$"

            fig, (ax, rax) = ratio_plot(histograms_to_plot, sigs=signals, bkgs=bkg_order, onto=onto, style=style, sort_by_yield=True, legend_title=legend_title)
            
            luminosity = sum(LUMI[y] / 1000.0 for y in year_list)
            hep.cms.label("Private Work", data=True, ax=ax, lumi=luminosity, lumi_format="{:0.1f}", com=13.6, year=year_str, loc=0)

            # Construct a clear filename
            filename = f"{outdir}/{year_str}_{region}_{category}_{stack_by}"
            if i_pt is not None:
                filename += f"_ptbin{pt_low}_{pt_high}.png"
            else:
                filename += "_ptinclusive.png"
            
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)

def plot_qcd_shapes(hists, year_str, outdir, region, norm_type):
    """For each pt bin, plots the normalized 'pass' and 'fail' distributions for the QCD sample."""
    if "qcd" not in hists or hists["qcd"].sum() == 0:
        print("No 'qcd' histogram with entries found in the input file. Exiting.")
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

        fig, ax = plt.subplots(figsize=(10, 8))
        
        if norm_type == "shape":
            hep.histplot(h_fail, ax=ax, label="QCD MC fail", color="blue", histtype="errorbar", yerr=True, density=True)
            hep.histplot(h_pass, ax=ax, label="QCD MC pass", color="black", histtype="errorbar", yerr=True, density=True)
            ylabel = "Probability Density"
        
        elif norm_type == "density":
            bin_width = h_pass.axes[0].widths[0]
            pass_yield = h_pass.sum()
            fail_yield = h_fail.sum()
            h_fail_scaled = h_fail * (pass_yield / fail_yield)
            h_pass_toplot = h_pass / bin_width
            h_fail_toplot = h_fail_scaled / bin_width
            ylabel = f"Events / {bin_width:g} GeV"
            hep.histplot(h_fail_toplot, ax=ax, label="QCD MC fail", color="blue", histtype="errorbar", yerr=True)
            hep.histplot(h_pass_toplot, ax=ax, label="QCD MC pass", color="black", histtype="errorbar", yerr=True)
        
        ax.set_xlabel("Jet $m_{sd}$ [GeV]")
        ax.set_ylabel(ylabel)
        ax.grid(True)
        
        hep.cms.label("Private Work", data=False, ax=ax, com=13.6, year=year_str)
        
        ax.legend(
            title=f"{pt_low:g} < $p_T$ < {pt_high:g} GeV",
            prop={'size': 14},
            title_fontsize=16,
            loc='upper right'
        )

        output_name = f"{outdir}/{year_str}_{region}_qcd_{norm_type}_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_name}")
        plt.close(fig)
        
# --- Main Function ---
def main(args):
    histograms = {}
    if len(args.year) > 3:
        year_str = "all-years"
    else:
        year_str = "-".join(args.year)

    for year in args.year:
        pkl_path = Path(args.indir) / f"histograms_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            print(f"Error: File not found: {pkl_path}. Skipping.")
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)
            for process, h in histograms_tmp.items():
                if process in histograms:
                    histograms[process] += h
                else:
                    histograms[process] = h

    if not histograms:
        print("No histograms loaded. Exiting.")
        return

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Path("style_hbb.yaml").open() as f:
        style = yaml.safe_load(f)

    # Call the master plotting function with all the options
    if args.plot_type == "qcd_shape":
        plot_qcd_shapes(histograms, year_str, args.outdir, args.region, args.norm_type)
    else:
        plot_manager(histograms, style, year_str, args.year, args.region, args.outdir, 
                     args.stack_by, args.category_type, args.pt_scope)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified plotting script for Hbb analysis.")
    parser.add_argument("--year", help="List of years", type=str, required=True, nargs="+", choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--indir", help="Input directory for .pkl files", type=str, required=True)
    parser.add_argument("--outdir", help="Output directory for plots", type=str, required=True)
    parser.add_argument("--region", help="Analysis region", type=str, required=True)
    
    # --- Main Control Arguments ---
    parser.add_argument("--plot-type", help="Type of plot to produce", type=str, default="process", choices=["process", "flavor", "qcd_shape"])
    parser.add_argument("--stack-by", help="Stack by 'process' or 'flavor'", type=str, default="process", choices=["process", "flavor"])
    parser.add_argument("--category-type", help="Plot pass/fail categories 'separate'ly or summed ('inclusive')", type=str, default="separate", choices=["separate", "inclusive"])
    parser.add_argument("--pt-scope", help="Plot each pT bin ('binned') or the sum of all bins ('inclusive')", type=str, default="binned", choices=["binned", "inclusive"])
    
    # Argument for QCD shape plot
    parser.add_argument("--norm-type", help="Norm for QCD shape plot ('shape' or 'density')", type=str, default="shape", choices=["shape", "density"])
    
    args = parser.parse_args()
    main(args)


