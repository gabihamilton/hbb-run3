from __future__ import annotations

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot

# --- CONFIGURATION ---
FILENAME = "parT_results/25Nov11_v14_private/2022/testsignalregion.root"
CATEGORY = "pass_bb"
LUMI = 34.7

hep.style.use("CMS")

# 1. Define Colors to match reference
# (Approximate hex codes based on the image)
color_map = {
    r"$\gamma$+jets": "#2c5f78",  # Dark blue
    "Other": "#5e6c7a",  # Gray-blue
    r"$t\bar{t}$": "#7f964f",  # Olive green
    r"W+$\gamma$": "#d16a4c",  # Orange-red
    r"Z+$\gamma$": "#6a3d6a",  # Purple
    "Data": "black",
}

# 2. Define Stack Order (Bottom to Top)
# This controls both the stack and the legend order (reversed in legend)
stack_order_pref = [
    r"$\gamma$+jets",
    "Other",
    r"$t\bar{t}$",
    r"W+$\gamma$",
    r"Z+$\gamma$",
]


def main():
    print(f"Opening {FILENAME}...")
    f = uproot.open(FILENAME)

    hists = {}
    data_hist = None

    keys = [k for k in f if CATEGORY in k]
    print(f"Found {len(keys)} histograms for category '{CATEGORY}'")

    if not keys:
        print("Error: No keys found for this category.")
        return

    for key in keys:
        h = f[key].to_hist()
        name = key.split(";")[0]

        # 3. Refined Process Grouping
        if "Run2022" in name or "Data" in name:
            proc = "Data"
        elif "Zgamma" in name:
            proc = r"Z+$\gamma$"
        elif "TTGamma" in name or "TTbar" in name or "TTTo" in name:
            proc = r"$t\bar{t}$"
        elif "WGamma" in name:
            proc = r"W+$\gamma$"
        # Important: Your sample list might name this GJets or similar
        elif "GJet" in name or "QCD_Pt" in name:
            proc = r"$\gamma$+jets"
        else:
            proc = "Other"

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

    # 4. Finalize Stack Order
    # Only include processes that actually have histograms
    final_stack_order = [p for p in stack_order_pref if p in hists]
    # Add any unexpected processes to the top
    for p in hists:
        if p not in final_stack_order:
            final_stack_order.append(p)

    stack_hists = [hists[p] for p in final_stack_order]
    stack_colors = [color_map.get(p, "grey") for p in final_stack_order]

    # --- Plotting ---
    fig, (ax, rax) = plt.subplots(
        2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    if stack_hists:
        hep.histplot(
            stack_hists,
            label=final_stack_order,
            stack=True,
            histtype="fill",
            color=stack_colors,
            ax=ax,
        )

    if data_hist:
        hep.histplot(data_hist, label="Data", histtype="errorbar", color=color_map["Data"], ax=ax)

        # Ratio
        total_mc = sum(stack_hists) if stack_hists else None
        if total_mc:
            r_num = data_hist.values()
            r_den = total_mc.values()
            ratio = np.divide(r_num, r_den, out=np.ones_like(r_num), where=r_den != 0)
            centers = data_hist.axes[0].centers
            rax.errorbar(centers, ratio, yerr=0, fmt="ko", markersize=4)

    # --- Styling ---
    ax.set_ylabel("Events / GeV")
    # Reverse legend order to match stack (top of stack = top of legend)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="upper right", ncol=2)

    ax.set_yscale("linear")
    # Set x-axis range to match reference
    ax.set_xlim(0, 200)

    rax.set_ylabel("Data / Pred")
    rax.set_ylim(0, 2.5)
    rax.axhline(1, color="gray", linestyle="--")
    rax.set_xlabel(r"Jet 0 $m_{sd}$ [GeV]")

    hep.cms.label("Private Work", data=True, lumi=LUMI, year="2022", ax=ax)

    out_name = "check_zgamma_plot_styled.png"
    plt.savefig(out_name)
    print(f"\nPlot saved to {out_name}")


if __name__ == "__main__":
    main()
