#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import hist
import numpy as np
import uproot

from hbb import utils


def fill_hists(outdict, events, region, reg_cfg, obs_cfg, qq_true):
    h = hist.Hist(
        hist.axis.Regular(
            obs_cfg["nbins"],
            obs_cfg["min"],
            obs_cfg["max"],
            name=obs_cfg["name"],
            label=obs_cfg["name"],
        )
    )

    bins_list = reg_cfg["bins"]
    bin_pname = reg_cfg["bin_pname"]
    str_bin_br = reg_cfg["branch_name"]

    for _process_name, data in events.items():

        weight_val = data["finalWeight"].astype(float)
        s = "nominal"

        bin_br = data[str_bin_br]
        obs_br = data[obs_cfg["branch_name"]]

        # Extracting variables
        Txcc = data["FatJet0_ParTPXccVsQCD"]
        Txbb = data["FatJet0_ParTPXbbVsQCD"]
        Txbbxcc = data["FatJet0_ParTPXbbXcc"]

        msd = data["FatJet0_msd"]
        pt = data["FatJet0_pt"]
        photon_pt = data["Photon0_pt"]

        # --- FIX: Calculate dPhi manually (handling wrapping) ---
        dphi_raw = np.abs(data["Photon0_phi"] - data["FatJet0_phi"])
        dphi = np.where(dphi_raw > np.pi, 2 * np.pi - dphi_raw, dphi_raw)

        # --- FIX: Extract MET (Robust) ---
        # MET is loaded as a dictionary (e.g. {'pt':.., 'phi':..}), so we extract 'pt'
        if data["MET"].dtype == "object":
            met_pt = data["MET"].apply(lambda x: x["pt"])
        else:
            met_pt = data["MET"]

        # --- 1. TRIGGER LOGIC ---
        trigger_mask = True
        is_data = "GenFlavor" not in data.columns
        has_trigger_cols = ("Photon200" in data.columns) and (
            "Photon110EB_TightID_TightIso" in data.columns
        )

        if is_data and has_trigger_cols:
            trigger_mask = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
        elif is_data and not has_trigger_cols:
            print(f"WARNING: Trigger missing for {_process_name}!")

        # --- 2. DEFINE SELECTION ---
        # Note: Your reference script cuts pt > 250, even though plot title said 200.
        # This matches the reference script logic.
        basic_cuts = (
            (photon_pt > 120) & (msd > 20) & (msd < 200) & (pt > 250) & (pt < 1200) & trigger_mask
        )

        # --- FIX: Uncomment and apply topological cuts ---
        topo_cuts = (dphi > 2.2) & (met_pt < 50)

        # Combined Pre-Selection
        pre_selection = basic_cuts & topo_cuts

        # --- 3. CATEGORIZATION ---
        genf = data["GenFlavor"] if "GenFlavor" in data.columns else 0

        # --- FIX: Change WP to 0.95 to match reference ---
        WP = 0.95

        selection_dict = {
            # Pass BB: Combined > 0.82 AND bb score is higher than cc
            "pass_bb": pre_selection & (Txbbxcc > WP) & (Txbb > Txcc),
            # Pass CC: Combined > 0.82 AND cc score is higher than bb
            "pass_cc": pre_selection & (Txbbxcc > WP) & (Txcc > Txbb),
            # Fail: Combined <= 0.82
            "fail": pre_selection & (Txbbxcc <= WP),
        }

        cut_bb = genf == 3
        cut_qq = genf != 3

        for i in range(len(bins_list) - 1):
            bin_cut = (bin_br > bins_list[i]) & (bin_br < bins_list[i + 1])

            for category, selection in selection_dict.items():
                if qq_true:
                    # Logic for splitting backgrounds into light/bb
                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}_{s}"
                    h.view()[:] = 0
                    h.fill(
                        obs_br[selection & bin_cut & cut_qq],
                        weight=weight_val[selection & bin_cut & cut_qq],
                    )
                    if name not in outdict:
                        outdict[name] = h.copy()
                    else:
                        outdict[name] += h.copy()

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}bb_{s}"
                    h.view()[:] = 0
                    h.fill(
                        obs_br[selection & bin_cut & cut_bb],
                        weight=weight_val[selection & bin_cut & cut_bb],
                    )
                    if name not in outdict:
                        outdict[name] = h.copy()
                    else:
                        outdict[name] += h.copy()
                else:
                    # Logic for Data/Signal
                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}_{s}"
                    h.view()[:] = 0
                    h.fill(obs_br[selection & bin_cut], weight=weight_val[selection & bin_cut])
                    if name not in outdict:
                        outdict[name] = h.copy()
                    else:
                        outdict[name] += h.copy()
    return outdict


def main(args):
    year = args.year
    tag = args.tag

    # --- FIX 2: Hardcoded path to YOUR directory ---
    path_to_dir = f"/eos/uscms/store/group/lpchbbrun3/gmachado/{tag}"

    # --- FIX 3: Hardcoded Z-Gamma Sample List ---
    # We include TTGamma and Zgamma here so they get split into bb/light
    samples_qq = ["Wjets", "Zjets", "Zgamma", "TTGamma"]

    columns = [
        "weight",
        "FatJet0_pt",  # Kinematic cut
        "FatJet0_msd",  # Observable/Kinematic cut
        "FatJet0_ParTPXbbVsQCD",  # ParticleNet b-tagger
        "FatJet0_ParTPXccVsQCD",  # ParticleNet c-tagger
        "FatJet0_ParTPXbbXcc",  # ParticleNet bb+cc combined tagger
        "Photon0_pt",  # New Kinematic cut
        "FatJet0_phi",
        "Photon0_phi",
        "MET",
        "GenFlavor",  # MC Truth matching
        "Photon200",  # Trigger OR logic
        "Photon110EB_TightID_TightIso",  # Trigger OR logic
    ]

    data_dirs = [Path(path_to_dir) / year]

    out_path = f"parT_results/{tag}/{year}"
    output_file = f"{out_path}/testsignalregion.root"

    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    if Path(output_file).is_file():
        Path(output_file).unlink()
    fout = uproot.create(output_file)

    # --- FIX 4: Hardcoded to load setup_zgamma.json ---
    os.popen(f"cp setup_zgamma.json {out_path}")
    with Path("setup_zgamma.json").open() as f:
        setup = json.load(f)
        cats = setup["categories"]
        obs_cfg = setup["observable"]

    with Path("pmap_run3.json").open() as f:
        pmap = json.load(f)

    # --- FIX 5: Hardcoded Filters for Low pT ---
    # This ensures we don't cut out the Z-Gamma events
    filters = [
        ("FatJet0_pt", ">", 250),
        ("FatJet0_pt", "<", 2000),
    ]

    if obs_cfg["branch_name"] not in columns:
        columns.append(obs_cfg["branch_name"])

    out_hists = {}
    for process, datasets in pmap.items():
        for dataset in datasets:
            for reg, cfg in cats.items():
                for data_dir in data_dirs:

                    events = utils.load_samples(
                        data_dir,
                        {process: [dataset]},
                        columns=columns,
                        region=cfg["name"],
                        filters=filters,
                    )

                    if not events:
                        continue

                    fill_hists(out_hists, events, reg, cfg, obs_cfg, (process in samples_qq))

    for name, h in out_hists.items():
        fout[name] = h

    print(f"Histograms saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make histograms for a given year.")
    parser.add_argument(
        "--year",
        help="year",
        type=str,
        required=True,
        choices=["2022", "2022EE", "2023", "2023BPix"],
    )
    parser.add_argument("--tag", help="tag", type=str, required=True)
    args = parser.parse_args()

    main(args)
