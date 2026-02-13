#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils

# --- REGION DIRECTORY MAPPING ---
# Maps the keys in your setup.json to the actual directory names on EOS
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

        if not is_data and weight_syst != "nominal" and weight_syst in data.columns:
            weight_val = data[weight_syst].astype(float) / data["sum_genWeight"].astype(float)
        else:
            weight_val = data["finalWeight"].astype(float)

        var_col = setup["observable"]["branch_name"]
        pt = data["FatJet0_pt"]
        msd = data["FatJet0_msd"]

        dphi = np.nan
        if "Photon0_phi" in data.columns and "FatJet0_phi" in data.columns:
            dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
            dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        var_series = dphi if var_col == "delta_phi_photon_jet" else data[var_col]
        is_mc = "GenFlavor" in data.columns
        genflavordata = (
            data["GenFlavor"].astype(np.int8) if is_mc else np.zeros(len(data), dtype=np.int8)
        )

        working_point = setup.get("working_point", 0.95)
        obs_min, obs_max = setup["observable"]["min"], setup["observable"]["max"]
        basic_cuts = (msd > obs_min) & (msd < obs_max)

        # Check the actual directory name for Zgamma logic
        actual_reg_name = REGION_MAP.get(region_key, region_key)
        if "zgamma" in actual_reg_name:
            met_pt = data["MET"].to_numpy() if "MET" in data.columns else np.zeros(len(data))
            trigger = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
            topo_cuts = (dphi > 2.2) & (met_pt < 50) & (data["Photon0_pt"] > 120)
            pre_selection = basic_cuts & topo_cuts & trigger
        else:
            pre_selection = basic_cuts & (pt > 450)

        Txcc, Txbb, Txbbxcc = (
            data["FatJet0_ParTPXccVsQCD"],
            data["FatJet0_ParTPXbbVsQCD"],
            data["FatJet0_ParTPXbbXcc"],
        )

        selection_dict = {
            "pass_bb": pre_selection & (Txbbxcc > working_point) & (Txbb > Txcc),
            "pass_cc": pre_selection & (Txbbxcc > working_point) & (Txcc > Txbb),
            "fail": pre_selection & (Txbbxcc <= working_point),
            "pass": pre_selection & (Txbbxcc > working_point),
        }

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
    with (
        uproot.update(output_root_path)
        if output_root_path.exists()
        else uproot.recreate(output_root_path)
    ) as fout:
        for process, h in histograms.items():
            proc_name = "data_obs" if process == data_key else process
            should_split = any(s in process for s in samples_qq) and process != data_key

            for i_pt in range(len(h.axes["pt1"].edges) - 1):
                pt_bin = f"pt{i_pt+1}"
                for category in h.axes["category"]:
                    base = f"{region_key}_{category}_{pt_bin}"
                    if should_split:
                        fout[f"{base}_{proc_name}bb_{suffix}"] = h[:, i_pt, category, 3]
                        fout[f"{base}_{proc_name}c_{suffix}"] = h[:, i_pt, category, 2]
                        fout[f"{base}_{proc_name}light_{suffix}"] = (
                            h[:, i_pt, category, 1] + h[:, i_pt, category, 0]
                        )
                    else:
                        fout[f"{base}_{proc_name}_{suffix}"] = h[:, i_pt, category, sum]


def main(args):
    with Path(args.setup).open() as f:
        setup = json.load(f)
    with Path("pmap_run3.json").open() as f:
        pmap = json.load(f)

    region_key = next(iter(setup["categories"]))
    reg_cfg = setup["categories"][region_key]
    pt_bins = np.array(reg_cfg["bins"])
    obs = setup["observable"]

    # Use the map to find the EOS directory name
    region_to_load = REGION_MAP.get(region_key, region_key)

    if "zgamma" in region_to_load:
        data_map_key = "EGammadata"
    elif "tt" in region_to_load:
        data_map_key = "Muondata"
    else:
        data_map_key = "Jetdata"

    print(f"--- Running Region Key: {region_key} ---")
    print(f"--- Loading from EOS Directory: {region_to_load} ---")
    print(f"--- Targeting Data Stream: {data_map_key} ---")

    folder_systs = ["JES", "JER", "UES", "MuonPTScale", "MuonPTRes"]
    systs_to_run = ["nominal"]
    if setup.get("do_systematics"):
        for s in setup.get("active_systematics", []):
            systs_to_run.extend([f"{s}Up", f"{s}Down"])

    output_root = Path(args.outdir) / f"fitting_{args.year}_{region_key}.root"
    if output_root.exists():
        output_root.unlink()

    axis_var = hist.axis.Regular(obs["nbins"], obs["min"], obs["max"], name=obs["name"])
    axis_pt = hist.axis.Variable(pt_bins, name="pt1")
    axis_cat = hist.axis.StrCategory(["pass_bb", "pass_cc", "fail", "pass"], name="category")
    axis_flav = hist.axis.IntCategory([0, 1, 2, 3], name="genflavor")

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

            print(f"  Processing Process: {process}")
            h = hist.Hist(axis_var, axis_pt, axis_cat, axis_flav)
            for dataset in datasets:
                events = utils.load_samples(
                    Path(f"/eos/uscms/store/group/lpchbbrun3/skims/{args.tag}/{args.year}"),
                    {process: [dataset]},
                    columns=cols,
                    region=region_to_load,
                    variation=variation,
                )
                if events:
                    h = fill_ptbinned_histogram(
                        h, events, region_key, setup, syst if not is_folder else "nominal"
                    )
                gc.collect()

            if h.sum() > 0:
                histograms[process] = h

        if args.save_root:
            export_to_root(
                histograms, output_root, region_key, setup.get("samples_qq", []), syst, data_map_key
            )

    print("\n" + "=" * 50)
    print(f"DONE: Histograms saved to {output_root}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, choices=["2022", "2022EE", "2023", "2023BPix"])
    parser.add_argument("--tag", required=True)
    parser.add_argument("--setup", required=True)
    parser.add_argument("--outdir", default="results")
    parser.add_argument("--save-root", action="store_true")
    args = parser.parse_args()
    main(args)
