#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import gc # Garbage collection for memory management

import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep # Recommended for CMS style plots
import numpy as np
from common import common_mc
from hbb import utils
from sklearn.metrics import auc, roc_curve

# Use CMS style
hep.style.use("CMS")

# --- Main Plotting Function ---
def plot_roc_curves(roc_data, year, region, outdir):
    """
    Plots ROC curves from the calculated data.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Iterate through taggers and backgrounds
    for label, data in roc_data.items():
        fpr = data["fpr"]
        tpr = data["tpr"]
        roc_auc = data["auc"]
        
        # Style choice: solid for PNet, dashed for ParT
        linestyle = "--" if "ParT" in label else "-"
        
        ax.plot(
            tpr,  # Signal efficiency on x-axis
            fpr,  # Background mistag rate on y-axis
            lw=2,
            linestyle=linestyle,
            label=f"{label} (AUC = {roc_auc:.3f})",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Signal Efficiency (H $\\to$ bb)", fontsize=20)
    ax.set_ylabel("Background Mistag Rate", fontsize=20)
    
    # Add diagonal line for random guessing
    ax.plot([0, 1], [0, 1], linestyle=':', color='gray', label='Random')

    ax.legend(loc="lower right", fontsize=16)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Set axis limits
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1e-3, 1.0) # Adjusted range for better visibility

    # Add CMS Label
    hep.cms.label("Private Work", data=False, year=year, ax=ax, loc=0)

    # Save the plot
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_filename = output_dir / f"roc_comparison_{year}_{region}.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    print(f"✅ Plot saved to {plot_filename}")


def main(args):
    year = args.year
    region = args.region

    # --- Data Loading Configuration ---
    # Update this to your correct input path
    MAIN_DIR = "/eos/uscms/store/group/lpchbbrun3/"
    dir_name = "gmachado/25Nov19_stable_production" # Updated to your stable tag
    path_to_dir = Path(MAIN_DIR) / dir_name / year

    # 1. Update Load Columns: Added ParTPXbb
    load_columns = [
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb", # ParticleNet
        "FatJet0_ParTPXbb", # Particle Transformer
        "GenFlavor",
        "weight"            # Needed if you want weighted ROCs
    ]

    # Use QCD or GJets depending on what you want to test against
    # For Hbb vs QCD, load QCD. For Hbb vs GJets background, load GJets.
    # Let's assume you want to check against the main background in your region.
    if "zgamma" in region:
        samples_to_process = {"GJets": common_mc["gjets"]}
        sig_samples = {"Hbb": common_mc["ggf-hbb"]} # Load signal explicitly if needed
    else:
        # Default fallback
        samples_to_process = {"QCD": common_mc["qcd"]}
        sig_samples = {"Hbb": common_mc["ggf-hbb"]}

    # --- 2. Load Signal and Background Separately ---
    # It is safer/clearer to load Signal and Background explicitely to ensure
    # we have true signal events (H->bb) vs true background events.
    
    print("Loading Signal (Hbb)...")
    sig_dict = utils.load_samples(path_to_dir, sig_samples, load_columns, region)
    if not sig_dict: return
    # Concatenate dictionary values into one array
    # utils.load_samples returns {process: df}, so we grab the first value
    sig_events = next(iter(sig_dict.values())) 
    
    print(f"Loading Background ({list(samples_to_process.keys())[0]})...")
    bkg_dict = utils.load_samples(path_to_dir, samples_to_process, load_columns, region)
    if not bkg_dict: return
    bkg_events = next(iter(bkg_dict.values()))

    # --- Event Pre-selection ---
    # Apply cuts to both arrays
    sig_mask = (sig_events.FatJet0_msd > 40) & (sig_events.FatJet0_pt > 200)
    bkg_mask = (bkg_events.FatJet0_msd > 40) & (bkg_events.FatJet0_pt > 200)
    
    sig_events = sig_events[sig_mask]
    bkg_events = bkg_events[bkg_mask]

    print(f"Signal events: {len(sig_events)}, Background events: {len(bkg_events)}")

    # --- 3. Define Taggers to Compare ---
    taggers = {
        "ParticleNet": "FatJet0_pnetTXbb",
        "ParT": "FatJet0_ParTPXbb"
    }

    roc_data = {}

    # --- 4. ROC Calculation ---
    for tagger_name, tagger_col in taggers.items():
        print(f"Calculating ROC for {tagger_name}...")
        
        # Check if columns exist (safety check)
        if tagger_col not in sig_events.columns or tagger_col not in bkg_events.columns:
            print(f"Warning: {tagger_col} not found. Skipping.")
            continue

        # Prepare arrays for sklearn
        # Signal = 1, Background = 0
        y_true = np.concatenate([np.ones(len(sig_events)), np.zeros(len(bkg_events))])
        y_scores = np.concatenate([sig_events[tagger_col], bkg_events[tagger_col]])
        weights = np.concatenate([sig_events["weight"], bkg_events["weight"]])

        # Handle NaNs if present
        valid_mask = ~np.isnan(y_scores)
        y_true = y_true[valid_mask]
        y_scores = y_scores[valid_mask]
        weights = weights[valid_mask]

        # Calculate ROC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, sample_weight=weights)
        roc_auc = auc(fpr, tpr) # Standard AUC calculation

        roc_data[tagger_name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc}
        
    # --- Plotting ---
    plot_roc_curves(roc_data, year, region, args.outdir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make ROC curves for taggers.")
    parser.add_argument("--year", type=str, required=True, choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--region", type=str, default="control-zgamma")
    parser.add_argument("--outdir", type=str, default="roc_plots")
    args = parser.parse_args()

    main(args)