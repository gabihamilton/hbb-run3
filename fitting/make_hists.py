#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils

# --- REGION DIRECTORY MAPPING ---
# Maps the keys in setup.json to the actual directory names on EOS
REGION_MAP = {
    "zgcr": "control-zgamma",
    "mucr": "control-tt",
    "vh": "signal-vh",
    "vbf": "signal-vbf",
    "ggf": "signal-ggf",
}


def fill_ptbinned_histogram(h, events, region_key, setup, weight_syst="nominal"):
    for process_name, data in events.items():
        is_data = "data" in process_name.lower()

        # --- 1. WEIGHTING LOGIC ---
        if not is_data and weight_syst != "nominal" and weight_syst in data.columns:
            # Systematic weights (like btagSF) usually need normalization by sum_genWeight
            weight_val = data[weight_syst].astype(float) / data["sum_genWeight"].astype(float)
        else:
            # load_samples already calculated finalWeight (weight / sum_genWeight)
            weight_val = data["finalWeight"].astype(float)

        # --- 2. VARIABLE EXTRACTION ---
        var_col = setup["observable"]["branch_name"]
        pt = data["FatJet0_pt"]
        msd = data["FatJet0_msd"]

        # Robust MET extraction from the parquet record
        if "MET" in data.columns:
            met_pt = data["MET"].pt if hasattr(data["MET"], "pt") else data["MET"]
        else:
            met_pt = np.zeros(len(data))

        dphi = np.nan
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        var_series = dphi if var_col == "delta_phi_photon_jet" else data[var_col]

        is_mc = "GenFlavor" in data.columns
        genflavordata = (
            data["GenFlavor"].astype(np.int8) if is_mc else np.zeros(len(data), dtype=np.int8)
        )

        # --- 3. SELECTION LOGIC ---
        working_point = setup.get("working_point", 0.95)
        obs_min, obs_max = setup["observable"]["min"], setup["observable"]["max"]
        basic_cuts = (msd > obs_min) & (msd < obs_max)

        actual_reg_name = REGION_MAP.get(region_key, region_key)

        if "zgamma" in actual_reg_name:
            # Specific Z-Gamma logic from Gabi's script
            trigger = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
            topo_cuts = (dphi > 2.2) & (met_pt < 50) & (data["Photon0_pt"] > 120)
            pre_selection = basic_cuts & topo_cuts & trigger & (pt > 250)
        else:
            # Lara's Signal Region logic
            pre_selection = basic_cuts & (pt > 450)

        Txcc = data["FatJet0_ParTPXccVsQCD"]
        Txbb = data["FatJet0_ParTPXbbVsQCD"]
        Txbbxcc = data["FatJet0_ParTPXbbXcc"]

        selection_dict = {
            "pass_bb": pre_selection & (Txbbxcc > working_point) & (Txbb > Txcc),
            "pass_cc": pre_selection & (Txbbxcc > working_point) & (Txcc > Txbb),
            "fail": pre_selection & (Txbbxcc <= working_point),
            "pass": pre_selection & (Txbbxcc > working_point),
            "inclusive": pre_selection,
        }

        # --- 4. FILLING ---
        for category, selection in selection_dict.items():
            if category in h.axes["category"]:
                h.fill(
                    var_series[selection],
                    pt[selection],
                    category=category,
                    genflavor=genflavordata[selection],
                    weight=weight_val[selection],
                )
    return h


def export_to_root(histograms, output_root_path, region_key, samples_qq, syst, data_key):
    suffix = "nominal" if syst == "nominal" else syst

    # Ensure directory exists
    output_root_path.parent.mkdir(parents=True, exist_ok=True)

    # Use uproot.update for writing
    # If the file doesn't exist, uproot.update will create it.
    with uproot.update(output_root_path) as fout:
        for process, h in histograms.items():
            # Standard Combine naming for data
            proc_name = "data_obs" if process == data_key else process

            # Check if process should be split (e.g., Wjets, Zjets)
            # and ensure we aren't trying to split the actual data stream
            should_split = any(s in process for s in samples_qq) and "data" not in process.lower()

            for i_pt in range(len(h.axes["pt1"].edges) - 1):
                pt_bin = f"pt{i_pt+1}"
                for category in h.axes["category"]:
                    # Template: {region}_{category}_{ptBin}_{process}_{suffix}
                    base = f"{region_key}_{category}_{pt_bin}_{proc_name}"

                    if should_split:
                        # Flavor mapping: 3=bb, 2=c, 1=light(uds), 0=light(g)
                        fout[f"{base}bb_{suffix}"] = h[:, i_pt, category, 3]
                        fout[f"{base}c_{suffix}"] = h[:, i_pt, category, 2]
                        fout[f"{base}light_{suffix}"] = (
                            h[:, i_pt, category, 1] + h[:, i_pt, category, 0]
                        )
                    else:
                        # For Data or non-split MC, sum over all flavor axes
                        fout[f"{base}_{suffix}"] = h[:, i_pt, category, sum]


def main(args):
    with Path(args.setup).open() as f:
        setup = json.load(f)
    with Path("pmap_run3.json").open() as f:
        pmap = json.load(f)

    for region_key, reg_cfg in setup["categories"].items():
        print("\n" + "=" * 50)
        print(f"STARTING REGION: {region_key}")
        print("=" * 50)

        pt_bins = np.array(reg_cfg["bins"])
        obs = setup["observable"]
        region_to_load = REGION_MAP.get(region_key, region_key)

        # Determine Data Stream (e.g., EGammadata for zgamma)
        if "zgamma" in region_to_load:
            data_map_key = "EGammadata"
        elif "tt" in region_to_load:
            data_map_key = "Muondata"
        else:
            data_map_key = "Jetdata"

        folder_systs = ["JES", "JER", "UES", "MuonPTScale", "MuonPTRes"]
        systs_to_run = ["nominal"]
        if setup.get("do_systematics"):
            for s in setup.get("active_systematics", []):
                systs_to_run.extend([f"{s}Up", f"{s}Down"])

        obs_name = setup["observable"]["name"]  # e.g., "msd"

        output_root = Path(args.outdir) / f"fitting_{args.year}_{region_key}_{obs_name}.root"

        # Ensure outdir exists
        output_root.parent.mkdir(parents=True, exist_ok=True)

        # Delete existing file to start fresh for this region
        # (This avoids the errors by ensuring we never 'update' a corrupted or old file)
        if output_root.exists():
            print(f"Cleaning up existing file: {output_root}")
            output_root.unlink()

        # Initialize a fresh ROOT file
        uproot.recreate(output_root).close()

        # Define Hist Axes
        axis_var = hist.axis.Regular(obs["nbins"], obs["min"], obs["max"], name=obs["name"])
        axis_pt = hist.axis.Variable(pt_bins, name="pt1")
        axis_cat = hist.axis.StrCategory(
            ["pass_bb", "pass_cc", "fail", "pass", "inclusive"], name="category"
        )
        axis_flav = hist.axis.IntCategory([0, 1, 2, 3], name="genflavor")

        # Define columns to load
        cols = [
            "weight",
            "FatJet0_pt",
            "FatJet0_msd",
            "FatJet0_ParTPXbbVsQCD",
            "FatJet0_ParTPXccVsQCD",
            "FatJet0_ParTPXbbXcc",
            "GenFlavor",
        ]
        if data_map_key == "EGammadata":
            cols += [
                "Photon0_pt",
                "Photon0_phi",
                "FatJet0_phi",
                "MET",
                "Photon200",
                "Photon110EB_TightID_TightIso",
            ]

        for syst in systs_to_run:
            print(f"\n>>> Running Systematic Pass: {syst}")
            is_folder = any(fs in syst for fs in folder_systs)
            variation = syst if is_folder else "nominal"

            histograms = {}
            for process, datasets in pmap.items():
                if "data" in process.lower() and process != data_map_key:
                    continue
                if process == data_map_key and syst != "nominal":
                    continue

                h = hist.Hist(axis_var, axis_pt, axis_cat, axis_flav)
                for dataset in datasets:
                    events = utils.load_samples(
                        data_dir=Path(
                            f"/eos/uscms/store/group/lpchbbrun3/skims/{args.tag}/{args.year}"
                        ),
                        samples={process: [dataset]},
                        columns=cols,
                        region=region_to_load,
                        variation=variation,
                    )
                    if events:
                        h = fill_ptbinned_histogram(
                            h, events, region_key, setup, syst if not is_folder else "nominal"
                        )
                    # Memory management within dataset loop
                    del events
                    gc.collect()

                if h.sum() > 0:
                    histograms[process] = h

            if args.save_root:
                export_to_root(
                    histograms,
                    output_root,
                    region_key,
                    setup.get("samples_qq", []),
                    syst,
                    data_map_key,
                )
                print(
                    f"  [SUCCESS] Updated {output_root} with {len(histograms)} processes for systematic: {syst}"
                )

            # --- NEW: Save Pickles for plotting pipeline ---
            pickle_path = (
                Path(args.outdir) / f"hists_{args.year}_{region_key}_{obs_name}_{syst}.pkl"
            )
            with pickle_path.open("wb") as f:
                pickle.dump(histograms, f)
            print(f"  [SUCCESS] Pickles saved to: {pickle_path}")

            # Memory management after each systematic pass
            del histograms
            gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Histogram Maker for Signal and CR")
    parser.add_argument("--year", required=True, choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--tag", required=True, help="Tag for the skims directory (e.g., 26Feb03)")
    parser.add_argument("--setup", required=True, help="Path to setup.json file")
    parser.add_argument("--outdir", default="results", help="Directory to save ROOT files")
    parser.add_argument("--save-root", action="store_true", help="Actually write the ROOT file")

    args = parser.parse_args()

    # Ensure outdir exists before starting
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    main(args)
