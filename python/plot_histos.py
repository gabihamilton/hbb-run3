#!/usr/bin/env python3
"""
... (script docstring unchanged) ...
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

# --- Globals for Plotting Logic ---
flavor_map = {3: "b-jet", 2: "c-jet", 1: "light-jet"}

mass_lo = 115
mass_hi = 135

categories = [
    "bb_pass",
    "bb_fail",
    "cc_pass",
    "cc_fail",
    "bbcc_fail",
    "bbfail_ccpass",
    "bbcc_pass",
]

# --- NEW: Dictionary to map category names to descriptive labels ---
category_labels = {
    "bb_pass": "TXbb > 0.95",
    "bb_fail": "TXbb < 0.95",
    "cc_pass": "TXcc > 0.95",
    "cc_fail": "TXcc < 0.95",
    "bbcc_fail": "TXbb < 0.95 & TXcc < 0.95",
    "bbfail_ccpass": "TXbb < 0.95 & TXcc > 0.95",
    "bbcc_pass": "TXbb > 0.95 & TXcc > 0.95",
}


# --- Function 1: Plotting Stacked by Process ---
# --- MODIFIED ---
# Added `variable` as an argument
def plot_by_process(
    hists, category, year_str, year_list, outdir, region, style, variable, ptinclusive=False
):
    """Plots a stacked histogram for each pt bin, with grouping and sorting handled by ratio_plot."""

    first_hist = next((h for h in hists.values() if h.sum() > 0), None)
    if not first_hist:
        print(f"All histograms are empty for {category} category. Skipping plot.")
        return
    pt_axis = first_hist.axes["pt1"]

    if ptinclusive:
        pt_bins = [(pt_axis.edges[0], pt_axis.edges[-1])]
        print("--- Preparing pT-inclusive plot ---")
    else:
        pt_bins = [(pt_axis.edges[i], pt_axis.edges[i + 1]) for i in range(len(pt_axis.edges) - 1)]
        print("--- Preparing plots for each pT bin ---")

    for pt_low, pt_high in pt_bins:
        # for i in range(len(pt_axis.edges) - 1):
        # pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i+1]
        # i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            if h.sum() == 0 or category not in h.axes["category"]:
                continue

            # --- MODIFIED ---
            # Changed hard-coded "msd1" to the dynamic `variable` argument
            h_proj = h[:, hist.loc(pt_low) : hist.loc(pt_high), category, :].project(variable)

            # --- MODIFIED ---
            # Added a check so we only blind data for the msd1 plot
            if process == "data" and region != "control-zgamma" and variable == "msd1":
                edges = h_proj.axes[0].edges
                mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                data_val = h_proj.values()
                data_val[mask] = 0
                h_proj.values()[:] = data_val
            histograms_to_plot[process] = h_proj

        # Conditional title and filename
        # --- MODIFIED ---
        # Added `variable` to the output filename
        if ptinclusive:
            legend_title = f"{category.capitalize()} Region, $p_T$-inclusive"
            output_name = (
                f"{outdir}/{year_str}_{region}_{category}_{variable}_process_ptinclusive.png"
            )
            print(f"  Processing pT range: {pt_low} - {pt_high} (inclusive)")
        else:
            legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"
            output_name = f"{outdir}/{year_str}_{region}_{category}_{variable}_process_ptbin{pt_low}_{pt_high}.png"
            print(f"  Processing pT bin: {pt_low} - {pt_high}")

        # Region-specific plotting logic
        if "control-zgamma" in region:
            signals = []
            bkg_order = ["wgamma", "zgamma", "other", "tt"]
            onto = "gjets"
            # --- UPDATED: Print a summary yield table ---
            print("\n--- Yield Table for this Bin ---")
            all_mc_hists = {p: h for p, h in histograms_to_plot.items() if p != "data"}
            total_mc_yield = sum(h.sum() for h in all_mc_hists.values())

            # Sort processes by yield for the table
            sorted_yields = sorted(
                all_mc_hists.items(), key=lambda item: item[1].sum(), reverse=True
            )

            print(f"{'Process':<20} | {'Yield':>12} | {'Percentage':>12}")
            print("-" * 49)
            for process_name, h in sorted_yields:
                yield_val = h.sum()
                percentage = (yield_val / total_mc_yield) * 100 if total_mc_yield > 0 else 0
                print(f"{process_name:<20} | {yield_val:>12.2f} | {percentage:>11.2f}%")
            print("-" * 49)
            print(f"{'Total MC':<20} | {total_mc_yield:>12.2f} | {'100.00%':>12}")
            if "data" in histograms_to_plot:
                print(f"{'Data':<20} | {histograms_to_plot['data'].sum():>12.0f} |")
            print("--------------------------------\n")
            # ---
        elif "control-tt" in region:
            signals = []
            bkg_order = ["wjets", "zjets", "qcd", "other", "hbb"]
            onto = "top"
        else:
            # Default for signal regions
            signals = ["hbb"]
            bkg_order = ["zjets", "wjets", "other", "top"]
            onto = "qcd"

        # legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"
        # --- UPDATED LEGEND TITLE ---
        label = category_labels.get(category, category.replace("_", " ").title())
        legend_title = f"{label} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=signals,
            bkgs=bkg_order,
            onto=onto,
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )
        # --- MODIFIED ---
        # We assume ratio_plot is smart enough to use the axis label from the histogram
        # If not, we would need to pass `xlabel=first_hist.axes[0].label` to ratio_plot
        # For now, no change to the call is needed.

        luminosity = sum(LUMI[y] / 1000.0 for y in year_list)
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        # --- MODIFIED ---
        # Added `variable` to the output filename
        output_name = f"{outdir}/{year_str}_{region}_{category}_{variable}_process_ptbin{pt_low}_{pt_high}.png"
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 2: Plotting Stacked by Flavor ---
# --- MODIFIED ---
# Added `variable` as an argument
def plot_by_flavor(hists, category, year_str, year_list, outdir, region, style, variable):
    """Plots a stacked histogram for each pt bin, splitting W/Z jets by flavor."""
    first_hist = next((h for h in hists.values() if h.sum() > 0), None)
    if not first_hist:
        print(f"All histograms are empty for {category} category. Skipping plot.")
        return
    pt_axis = first_hist.axes["pt1"]

    mass_lo = 115
    mass_hi = 135

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing pt bin: {pt_low} - {pt_high}")

        histograms_to_plot = {}
        for process, h in hists.items():
            if h.sum() == 0 or category not in h.axes["category"]:
                continue

            if process in ["wjets", "zjets"]:
                # This h_2d is now (variable, genflavor)
                h_2d = h[:, i_start, category, :]
                for flavor_code, flavor_name in flavor_map.items():
                    new_key = f"{process}_{flavor_name}"
                    # This projects onto the `variable` axis
                    histograms_to_plot[new_key] = h_2d[:, hist.loc(flavor_code)]
            else:
                # --- MODIFIED ---
                # Changed hard-coded "msd1" to the dynamic `variable` argument
                h_proj = h[:, i_start, category, :].project(variable)

                # --- MODIFIED ---
                # Added a check so we only blind data for the msd1 plot
                if process == "data" and variable == "msd1":
                    edges = h_proj.axes[0].edges
                    mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
                    data_val = h_proj.values()
                    data_val[mask] = 0
                    h_proj.values()[:] = data_val
                histograms_to_plot[process] = h_proj

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

        legend_title = f"{category.capitalize()} Region, {pt_low:g} < $p_T$ < {pt_high:g} GeV"

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=[],
            bkgs=bkg_order,
            onto="qcd",
            style=style,
            sort_by_yield=True,
            legend_title=legend_title,
        )
        # As before, we assume ratio_plot will pick up the correct axis label

        luminosity = sum(LUMI[y] / 1000.0 for y in year_list)
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        # --- MODIFIED ---
        # Added `variable` to the output filename
        output_name = (
            f"{outdir}/{year_str}_{region}_{category}_{variable}_flavor_ptbin{pt_low}_{pt_high}.png"
        )
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- Function 3: QCD Pass/Fail Shape Comparison ---
# --- MODIFIED ---
# Added `variable` as an argument
def plot_qcd_shapes(hists, year_str, outdir, region, norm_type, variable):
    """For each pt bin, plots the normalized 'pass' and 'fail' distributions for the QCD sample."""
    if "qcd" not in hists or hists["qcd"].sum() == 0:
        print("No 'qcd' histogram with entries found in the input file. Exiting.")
        return
    h_qcd = hists["qcd"]
    pt_axis = h_qcd.axes["pt1"]

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"Processing pt bin: {pt_low} - {pt_high}")

        # --- MODIFIED ---
        # Changed hard-coded "msd1" to the dynamic `variable` argument
        h_pass = h_qcd[:, i_start, "pass", :].project(variable)
        h_fail = h_qcd[:, i_start, "fail", :].project(variable)

        if h_pass.sum() == 0 or h_fail.sum() == 0:
            print("  Skipping pt bin due to zero events in pass or fail.")
            continue

        fig, ax = plt.subplots(figsize=(10, 8))

        if norm_type == "shape":
            hep.histplot(
                h_fail,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            hep.histplot(
                h_pass,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
                density=True,
            )
            ylabel = "Probability Density"

        elif norm_type == "density":
            bin_width = h_pass.axes[0].widths[0]
            pass_yield = h_pass.sum()
            fail_yield = h_fail.sum()
            h_fail_scaled = h_fail * (pass_yield / fail_yield)
            h_pass_toplot = h_pass / bin_width
            h_fail_toplot = h_fail_scaled / bin_width
            # --- MODIFIED ---
            # Make the Y-axis label dynamic based on bin width and units (from axis label)
            xlabel_text = h_pass.axes[0].label
            units = (
                xlabel_text[xlabel_text.find("[") + 1 : xlabel_text.find("]")]
                if "[" in xlabel_text
                else ""
            )
            ylabel = f"Events / {bin_width:g} {units}"

            hep.histplot(
                h_fail_toplot,
                ax=ax,
                label="QCD MC fail",
                color="blue",
                histtype="errorbar",
                yerr=True,
            )
            hep.histplot(
                h_pass_toplot,
                ax=ax,
                label="QCD MC pass",
                color="black",
                histtype="errorbar",
                yerr=True,
            )

        # --- MODIFIED ---
        # Set the X-axis label dynamically from the histogram's axis
        ax.set_xlabel(h_pass.axes[0].label)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        hep.cms.label("Private Work", data=False, ax=ax, com=13.6, year=year_str)

        ax.legend(
            title=f"{pt_low:g} < $p_T$ < {pt_high:g} GeV",
            prop={"size": 14},
            title_fontsize=16,
            loc="upper right",
        )

        # --- MODIFIED ---
        # Added `variable` to the output filename
        output_name = (
            f"{outdir}/{year_str}_{region}_{variable}_qcd_{norm_type}_ptbin{pt_low}_{pt_high}.png"
        )
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        print(f"  Saved plot to {output_name}")
        plt.close(fig)


# --- NEW Function 4: Inclusive Plots (Pass + Fail) ---
# --- MODIFIED ---
# Added `variable` as an argument
def plot_inclusive(
    hists, year_str, year_list, outdir, region, style, inclusive_scope, stack_by, variable
):

    # Now that we have more categories beyond pass and fail, we need to sum the right categories
    pass_cat = "bb_pass"
    fail_cat = "bb_fail"
    print(f"  Creating inclusive plot from '{pass_cat}' + '{fail_cat}' categories.")

    """Plots inclusive histograms summed over pass and fail categories."""
    hists_incl = {}
    for process, h in hists.items():
        if h.sum() > 0 and pass_cat in h.axes["category"] and fail_cat in h.axes["category"]:
            # Project to 4D (variable, pt, cat, flavor), select pass/fail, then sum over cat
            hists_incl[process] = h[..., [hist.loc(pass_cat), hist.loc(fail_cat)], :][
                ..., hist.sum, :
            ]

    if not hists_incl:
        print(
            "No histograms with the required pass/fail categories were found. No inclusive plots to make."
        )
        return

    # hists_incl = {p: h[:, :, hist.sum, :] for p, h in hists.items() if h.sum() > 0}
    # if not hists_incl:
    #    print("All histograms are empty. No inclusive plots to make.")
    #    return

    first_hist = next(iter(hists_incl.values()))
    pt_axis = first_hist.axes["pt1"]

    # This list will hold the configurations for each plot we need to make
    plot_configs = []

    # Logic for pt-binned plots
    if inclusive_scope in ["pt-binned", "all"]:
        for i in range(len(pt_axis.edges) - 1):
            pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
            # Slicing for this specific pt bin
            # --- MODIFIED ---
            # Changed hard-coded "msd1" to the dynamic `variable` argument
            h_slice = {
                p: h[:, pt_axis.index(pt_low), :].project(variable) for p, h in hists_incl.items()
            }
            plot_configs.append(
                {
                    "hists": h_slice,
                    "title": f"Inclusive (Pass+Fail), {pt_low:g} < $p_T$ < {pt_high:g} GeV",
                    # --- MODIFIED ---
                    # Added `variable` to the output filename
                    "filename": f"{outdir}/{year_str}_{region}_inclusive_{variable}_{stack_by}_ptbin{pt_low}_{pt_high}.png",
                    "pt_slice": i,  # Store index for flavor splitting
                }
            )

    # Logic for fully inclusive plot
    if inclusive_scope in ["pt-inclusive", "all"]:
        # --- MODIFIED ---
        # Changed hard-coded "msd1" to the dynamic `variable` argument
        h_slice = {p: h[:, hist.sum, :].project(variable) for p, h in hists_incl.items()}
        plot_configs.append(
            {
                "hists": h_slice,
                "title": "Inclusive (Pass+Fail, all $p_T$)",
                # --- MODIFIED ---
                # Added `variable` to the output filename
                "filename": f"{outdir}/{year_str}_{region}_inclusive_{variable}_{stack_by}_allpt.png",
                "pt_slice": hist.sum,  # Use hist.sum for inclusive
            }
        )

    # Main loop to generate the plots
    for config in plot_configs:
        print(f"  Processing: {config['title']}")

        histograms_to_plot = config["hists"]

        # --- ADDED: Data Blinding ---
        # --- MODIFIED ---
        # Added a check so we only blind data for the msd1 plot
        if "data" in histograms_to_plot and variable == "msd1":
            h_proj = histograms_to_plot["data"]
            edges, data_val = h_proj.axes[0].edges, h_proj.values()
            mask = (edges[:-1] >= mass_lo) & (edges[:-1] < mass_hi)
            data_val[mask] = 0
            h_proj.values()[:] = data_val

        # Handle flavor stacking
        if stack_by == "flavor":
            flavored_hists = {}
            for process, h in histograms_to_plot.items():
                if process in ["wjets", "zjets"]:
                    # This h_2d is (variable, genflavor)
                    h_2d = hists_incl[process][:, config["pt_slice"], :]
                    for code, name in flavor_map.items():
                        # This projects onto the `variable` axis
                        flavored_hists[f"{process}_{name}"] = h_2d[:, hist.loc(code)]
                else:
                    flavored_hists[process] = h
            histograms_to_plot = flavored_hists

        # Region-aware logic for stacking order
        if "control-zgamma" in region:
            signals, bkg_order, onto = (
                [],
                ["ttgamma", "wgamma", "zgamma", "qcd", "other", "top", "zjets", "wjets", "hbb"],
                "gjets",
            )
            if "zgamma" in histograms_to_plot:
                zgamma_yield = histograms_to_plot["zgamma"].sum()
                print(f"    Z+Gamma MC Yield in this bin: {zgamma_yield:.2f}")

            # --- UPDATED: Debug printout for ALL histograms in this bin ---
            # print("\n--- Histogram Yields for this Bin ---")
            # for process_name in sorted(histograms_to_plot.keys()):
            #    print(f"\n--- {process_name} ---")
            #    print(histograms_to_plot[process_name])
            # print("-------------------------------------\n")
            # ---
            # ---
        elif "control-tt" in region:
            signals, bkg_order, onto = [], ["wjets", "zjets", "qcd", "other", "hbb"], "top"
        else:
            signals, bkg_order, onto = ["hbb"], ["zjets", "wjets", "other", "top"], "qcd"

        if stack_by == "flavor":
            # Update bkg_order for flavor plots
            if "control-tt" in region:
                bkg_order = [
                    "wjets_light-jet",
                    "wjets_c-jet",
                    "zjets_light-jet",
                    "zjets_c-jet",
                    "zjets_b-jet",
                    "qcd",
                    "other",
                    "hbb",
                ]
            elif "control-zgamma" not in region:
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

        if onto not in histograms_to_plot and not any(
            k.startswith(onto) for k in histograms_to_plot
        ):
            print(f"    WARNING: Main background '{onto}' not found. Skipping plot.")
            continue

        fig, (ax, rax) = ratio_plot(
            histograms_to_plot,
            sigs=signals,
            bkgs=bkg_order,
            onto=onto,
            style=style,
            sort_by_yield=True,
            legend_title=config["title"],
        )
        # As before, we assume ratio_plot will pick up the correct axis label

        luminosity = sum(LUMI[y] / 1000.0 for y in year_list)
        hep.cms.label(
            "Private Work",
            data=True,
            ax=ax,
            lumi=luminosity,
            lumi_format="{:0.1f}",
            com=13.6,
            year=year_str,
            loc=0,
        )

        fig.savefig(config["filename"], dpi=300, bbox_inches="tight")
        plt.close(fig)


# --- MODIFIED ---
# Added `variable` as an argument
def plot_reference(hists, category, year_str, year_list, outdir, region, style, variable):
    """
    Creates individual, non-stacked plots for each sample in each pt bin.
    These are useful for debugging and as a reference.
    """
    first_hist = next((h for h in hists.values() if h.sum() > 0), None)
    if not first_hist:
        print(f"All histograms are empty for {category} category. Skipping plot.")
        return
    pt_axis = first_hist.axes["pt1"]

    # Create a dedicated subdirectory for these reference plots
    ref_outdir = Path(outdir) / "ref_plots"
    ref_outdir.mkdir(parents=True, exist_ok=True)

    for i in range(len(pt_axis.edges) - 1):
        pt_low, pt_high = pt_axis.edges[i], pt_axis.edges[i + 1]
        i_start = pt_axis.index(pt_low)
        print(f"  Processing reference plots for pt bin: {pt_low} - {pt_high}")

        # Loop over every single process in the input file
        for process, h in hists.items():
            if h.sum() == 0 or category not in h.axes["category"]:
                continue

            # Project to 1D for this specific process, category, and pt bin
            # --- MODIFIED ---
            # Changed hard-coded "msd1" to the dynamic `variable` argument
            h_proj = h[:, i_start, category, :].project(variable)

            if h_proj.sum() == 0:
                continue

            # Create the plot
            fig, ax = plt.subplots(figsize=(10, 8))

            # Get style for this specific process
            process_style = style.get(process, {})
            label = process_style.get("label", process)
            color = process_style.get("color", "black")

            hep.histplot(h_proj, ax=ax, label=label, color=color, histtype="step", lw=2)

            ax.legend()
            # --- MODIFIED ---
            # Set the X-axis label dynamically from the histogram's axis
            ax.set_xlabel(h_proj.axes[0].label)
            ax.set_ylabel("Events")
            ax.grid(True)
            ax.set_yscale("log")  # Use a log scale to see small contributions

            luminosity = sum(LUMI[y] / 1000.0 for y in year_list)
            hep.cms.label(
                "Private Work",
                data=True,
                ax=ax,
                lumi=luminosity,
                lumi_format="{:0.1f}",
                com=13.6,
                year=year_str,
            )
            ax.set_title(
                f"{label} - {category.replace('_',' ').title()} \n {pt_low:g} < $p_T$ < {pt_high:g} GeV",
                loc="right",
            )

            # --- MODIFIED ---
            # Added `variable` to the output filename
            output_name = f"{ref_outdir}/{year_str}_{region}_{category}_{variable}_{process}_ptbin{pt_low}_{pt_high}.png"
            fig.savefig(output_name, dpi=300, bbox_inches="tight")
            plt.close(fig)


# --- Main Function: The Control Center ---
def main(args):
    histograms = {}
    year_str = "all-years" if len(args.year) > 3 else "-".join(args.year)
    # if len(args.year) > 3:
    #    year_str = "all-years"
    # else:
    #    year_str = "-".join(args.year)

    for year in args.year:
        # --- MODIFIED ---
        # The pkl_path now uses `args.variable` to find the correct file
        pkl_path = Path(args.indir) / f"histograms_{args.variable}_{year}_{args.region}.pkl"
        if not pkl_path.exists():
            # --- MODIFIED ---
            # Updated error message to be more informative
            print(
                f"Error: File not found at {pkl_path}."
                f" Did you run make_histos for --variable {args.variable}? Skipping."
            )
            continue
        with pkl_path.open("rb") as f:
            histograms_tmp = pickle.load(f)

            # --- MODIFIED ---
            # This printout is now specific to the variable being plotted
            print(f"\nLoading {args.variable} histograms for year {year}...")

            # Print the total yield for this year
            qcd_yield = histograms_tmp.get("qcd", hist.Hist()).sum()
            data_yield = histograms_tmp.get("data", hist.Hist()).sum()
            print(f"  Year {year}:")
            print(f"    Data Yield ({args.variable}): {data_yield:.2f}")
            print(f"    QCD MC Yield ({args.variable}): {qcd_yield:.2f}")

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

    # Call the correct plotting function based on --plot-type
    # --- MODIFIED ---
    # Passed `args.variable` to every plotting function
    if args.plot_type == "process":
        for category in categories:
            print(
                f"Plotting {args.variable} histograms by process for category: {category}, year: {year_str}..."
            )
            plot_by_process(
                histograms,
                category,
                year_str,
                args.year,
                args.outdir,
                args.region,
                style,
                args.variable,  # <-- ADDED
                ptinclusive=(args.inclusive_scope == "pt-inclusive"),
            )
    elif args.plot_type == "flavor":
        for category in categories:
            print(
                f"Plotting {args.variable} histograms by flavor for category: {category}, year: {year_str}..."
            )
            plot_by_flavor(
                histograms,
                category,
                year_str,
                args.year,
                args.outdir,
                args.region,
                style,
                args.variable,  # <-- ADDED
            )
    elif args.plot_type == "qcd_shape":
        print(f"Plotting {args.variable} QCD pass/fail shapes for year: {year_str}...")
        plot_qcd_shapes(
            histograms,
            year_str,
            args.outdir,
            args.region,
            args.norm_type,
            args.variable,  # <-- ADDED
        )
    elif args.plot_type == "inclusive":
        print(f"Plotting {args.variable} inclusive (pass+fail) histograms for year: {year_str}...")
        plot_inclusive(
            histograms,
            year_str,
            args.year,
            args.outdir,
            args.region,
            style,
            args.inclusive_scope,
            args.stack_by,
            args.variable,  # <-- ADDED
        )
    elif args.plot_type == "reference":
        for category in categories:
            print(
                f"Plotting {args.variable} reference histograms for category: {category}, year: {year_str}..."
            )
            plot_reference(
                histograms,
                category,
                year_str,
                args.year,
                args.outdir,
                args.region,
                style,
                args.variable,  # <-- ADDED
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified plotting script for Hbb analysis.")
    parser.add_argument(
        "--year",
        help="List of years",
        type=str,
        required=True,
        nargs="+",
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument("--indir", help="Input directory for .pkl files", type=str, required=True)
    parser.add_argument("--outdir", help="Output directory for plots", type=str, required=True)
    parser.add_argument("--region", help="Analysis region", type=str, required=True)
    # --- MODIFIED ---
    # Added the new --variable argument
    parser.add_argument(
        "--variable",
        help="Variable to plot",
        type=str,
        default="msd1",
        choices=["msd1", "met", "photon_pt", "delta_phi"],
    )
    parser.add_argument(
        "--stack-by",
        help="For inclusive plots, stack by 'process' or 'flavor'",
        type=str,
        default="process",
        choices=["process", "flavor"],
    )

    parser.add_argument(
        "--plot-type",
        help="Type of plot to produce",
        type=str,
        default="process",
        choices=["process", "flavor", "qcd_shape", "inclusive", "reference"],
    )
    parser.add_argument(
        "--norm-type",
        help="Normalization for QCD shape plot ('shape' or 'density')",
        type=str,
        default="shape",
        choices=["shape", "density"],
    )
    parser.add_argument(
        "--inclusive-scope",
        help="Scope for inclusive plots ('pt-binned' or 'pt-inclusive')",
        type=str,
        default="pt-binned",
        choices=["pt-binned", "pt-inclusive"],
    )
    args = parser.parse_args()
    main(args)
