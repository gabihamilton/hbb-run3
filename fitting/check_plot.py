from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

# --- CONFIGURATION ---
# Make sure this points to the STABLE tag you decided on
FILENAME = "parT_results/25Nov19_stable_v14_private/2022/testsignalregion.root"
CATEGORY = "pass_bb"
LUMI = 34.7

hep.style.use("CMS")

# 1. VISUAL CONFIGURATION
# Matches image_1dc440.jpg style
color_map = {
    r"$\gamma$+jets": "#1f4e5f",  # Dark Teal (Main background)
    r"Other": "#5e6c7a",  # Slate Grey
    r"$t\bar{t}$": "#7f964f",  # Olive Green
    r"W+$\gamma$": "#d16a4c",  # Salmon/Red
    r"Z+$\gamma$": "#6a3d6a",  # Purple
    "Data": "black",
}

# Stack order (Bottom -> Top)
stack_order = [
    r"$\gamma$+jets",
    r"Other",
    r"$t\bar{t}$",
    r"W+$\gamma$",
    r"Z+$\gamma$",
]


def main():
    print(f"Opening {FILENAME}...")
    try:
        f = uproot.open(FILENAME)
    except FileNotFoundError:
        print(f"Error: Could not find {FILENAME}. Check your path/tag.")
        return

    hists = {}
    data_hist = None

    # Pythonic iteration (fixes SIM118)
    keys = [k for k in f if CATEGORY in k]
    print(f"Found {len(keys)} histograms for category '{CATEGORY}'")

    if not keys:
        print("Error: No keys found. Check if the category name is correct.")
        return

    for key in keys:
        h = f[key].to_hist()
        name = key.split(";")[0]  # remove ;1 suffix

        # 2. ROBUST PROCESS GROUPING
        # Logic to map file names to the labels in our legend
        if "Run2022" in name or "Data" in name or "EGamma" in name:
            proc = "Data"
        elif "Zgamma" in name:
            proc = r"Z+$\gamma$"
        elif "TTGamma" in name or "TTbar" in name or "TTTo" in name:
            proc = r"$t\bar{t}$"
        elif "WGamma" in name:
            proc = r"W+$\gamma$"
        # This catches Gamma+Jets (often named GJ_PTG or GJet)
        elif "GJet" in name or "GJ_" in name or "QCD" in name:
            proc = r"$\gamma$+jets"
        elif "ZJets" in name or "DYJets" in name:
            proc = r"Other"  # Merging Z+jets into Other based on plot size
        elif "WJets" in name:
            proc = r"Other"  # Merging W+jets into Other
        else:
            proc = r"Other"

        # Accumulate
        if proc == "Data":
            if data_hist is None:
                data_hist = h
            else:
                data_hist += h
        else:
            if proc not in hists:
                hists[proc] = h
            else:
                hists[proc] += h

    # 3. BUILD STACK
    # Filter out empty processes and enforce order
    final_stack_names = [p for p in stack_order if p in hists]
    stack_hists = [hists[p] for p in final_stack_names]
    stack_colors = [color_map.get(p, "grey") for p in final_stack_names]

    # 4. PLOTTING
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Main Stack
    if stack_hists:
        hep.histplot(
            stack_hists,
            label=final_stack_names,
            stack=True,
            histtype="fill",
            color=stack_colors,
            ax=ax,
        )

    # Data
    if data_hist:
        hep.histplot(
            data_hist,
            label="Data",
            histtype="errorbar",
            color="black",
            ax=ax,
            markersize=10,
            yerr=True,
        )

        # Ratio Calculation
        total_mc = sum(stack_hists)
        r_num = data_hist.values()
        r_den = total_mc.values()
        # Safe division
        ratio = np.divide(r_num, r_den, out=np.ones_like(r_num), where=r_den != 0)

        # Ratio Plot
        centers = data_hist.axes[0].centers
        rax.errorbar(centers, ratio, yerr=0, fmt="ko", markersize=4)

    # 5. STYLING
    ax.set_ylabel("Events / GeV")
    ax.set_xlim(0, 200)  # Match reference X-axis

    # Legend: Reverse handles so top of stack = top of legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title=None, loc="upper right", ncol=2, fontsize=14)

    # Text Overlay (The "TXbb > 0.95" text)
    ax.text(
        0.95,
        0.92,
        r"TXbb $> 0.95$ Region, $200 < p_T < 1200$ GeV",
        transform=ax.transAxes,
        ha="right",
        fontsize=16,
    )

    # Ratio Styling
    rax.set_ylabel("Data / Bkg")
    rax.set_ylim(0, 2.5)
    rax.axhline(1, color="gray", linestyle="--")
    rax.set_xlabel(r"Jet 0 $m_{sd}$ [GeV]", fontsize=18)

    # CMS Label
    hep.cms.label("Private Work", data=True, lumi=LUMI, year="2022", ax=ax, loc=0)

    out_name = "check_zgamma_plot_styledOLD.png"
    plt.savefig(out_name)
    print(f"\nPlot saved to {out_name}")


if __name__ == "__main__":
    main()
