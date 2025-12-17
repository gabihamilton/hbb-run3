#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import hist
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

        # TODO add in systematics functionality
        weight_val = data["finalWeight"].astype(float)
        s = "nominal"

        bin_br = data[str_bin_br]
        obs_br = data[obs_cfg["branch_name"]]

        Txbb = data["FatJet0_pnetTXbb"]
        Txcc = data["FatJet0_pnetTXcc"]
        Txbbxcc = data["FatJet0_pnetXbbXcc"]

        # --- NEW TRIGGER LOGIC START ---
        # 1. Define the trigger mask
        # We assume MC passes (trigger_mask = True)
        trigger_mask = True

        # 2. Check if we are running on Data (no GenFlavor) AND if columns exist
        is_data = "GenFlavor" not in data.columns
        has_trigger_cols = ("Photon200" in data.columns) and (
            "Photon110EB_TightID_TightIso" in data.columns
        )

        if is_data and has_trigger_cols:
            # Implement the OR logic from your script
            trigger_mask = data["Photon200"] | data["Photon110EB_TightID_TightIso"]
        elif is_data and not has_trigger_cols:
            print(
                f"WARNING: Trigger columns missing for data process {_process_name}! No trigger applied."
            )
        # -------------------------------

        # --- Updated Pre-Selection ---
        pre_selection = (
            (obs_br > obs_cfg["min"])
            & (obs_br < obs_cfg["max"])
            & trigger_mask  # <--- The mask is applied here
        )

        # --- FIX 1: Safety check to prevent crashing on Data/QCD ---
        genf = data["GenFlavor"] if "GenFlavor" in data.columns else 0

        pre_selection = (obs_br > obs_cfg["min"]) & (obs_br < obs_cfg["max"])
        WP = 0.85

        selection_dict = {
            "pass_bb": pre_selection & (Txbbxcc > WP) & (Txbb > Txcc),
            "pass_cc": pre_selection & (Txbbxcc > WP) & (Txcc > Txbb),
            "fail": pre_selection & (Txbbxcc <= WP),
            "pass": pre_selection & (Txbbxcc > WP),
        }

        cut_bb = genf == 3
        # cut_qq = (genf > 0) & (genf < 3)
        cut_qq = genf != 3

        for i in range(len(bins_list) - 1):
            bin_cut = (bin_br > bins_list[i]) & (bin_br < bins_list[i + 1]) & pre_selection

            for category, selection in selection_dict.items():
                if qq_true:
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

                    name = f"{region}_{category}_{bin_pname}{i+1}_{_process_name}_{s}"
                    h.view()[:] = 0
                    h.fill(
                        obs_br[selection & bin_cut],
                        weight=weight_val[selection & bin_cut],
                    )
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
        "FatJet0_pt",
        "FatJet0_msd",
        "FatJet0_pnetTXbb",
        "FatJet0_pnetTXcc",
        "FatJet0_pnetXbbXcc",
        "VBFPair_mjj",
        "GenFlavor",
        "Photon200",  # <--- Add triggers
        "Photon110EB_TightID_TightIso",
    ]

    data_dirs = [Path(path_to_dir) / year]

    out_path = f"results/{tag}/{year}"
    output_file = f"{out_path}/signalregion.root"

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
        ("FatJet0_pt", ">", 200),
        ("FatJet0_pt", "<", 2000),
        ("VBFPair_mjj", ">", -2),
        ("VBFPair_mjj", "<", 13000),
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
