"""
Common functions for processors.
"""

from __future__ import annotations

# In src/hbb/utils.py
import pickle
import warnings
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import pyarrow as pa
from coffea.analysis_tools import PackedSelection

P4 = {
    "eta": "Eta",
    "phi": "Phi",
    "mass": "Mass",
    "pt": "Pt",
}

def add_selection(
    name: str,
    sel: np.ndarray,
    selection: PackedSelection,
    cutflow: dict,
    isData: bool,
    genWeights: ak.Array = None,
):
    """adds selection to PackedSelection object and the cutflow dictionary"""
    if isinstance(sel, ak.Array):
        sel = sel.to_numpy()

    selection.add(name, sel.astype(bool))
    cutflow[name] = (
        np.sum(selection.all(*selection.names))
        if isData
        # add up genWeights for MC
        else np.sum(genWeights[selection.all(*selection.names)])
    )


def check_selector(sample: str, selector: str | list[str]):
    if not isinstance(selector, (list, tuple)):
        selector = [selector]

    for s in selector:
        if s.endswith("?"):
            if s[:-1] == sample:
                return True
        elif s.startswith("*"):
            if s[1:] in sample:
                return True
        else:
            if sample.startswith(s):
                return True

    return False


def get_sum_genweights(data_dir: Path, dataset: str) -> float:
    """
    Get the sum of genweights for a given dataset.
    :param data_dir: The directory where the datasets are stored.
    :param dataset: The name of the dataset to get the genweights for.
    :return: The sum of genweights for the dataset.
    """
    total_sumw = 0

    try:
        # Load the genweights from the pickle file
        for pickle_file in list(Path(data_dir / dataset / "pickles").glob("*.pkl")):
            with Path(pickle_file).open("rb") as file:
                out_dict = pickle.load(file)
            # The sum of weights is stored in the "sumw" key
            # You can access it like this:
            for key in out_dict:
                sumw = next(iter(out_dict[key]["nominal"]["sumw"].values()))
            total_sumw += sumw
    except:
        warnings.warn(
            f"Error loading genweights for dataset: {dataset}. Skipping.",
            category=UserWarning,
            stacklevel=2,
        )
        total_sumw = 1

    # print(f"Total sum of weights for all pickles for {dataset}: {total_sumw}")
    return total_sumw


def load_samples(
    data_dir: Path,
    samples: dict,
    columns: list[str],
    region: str,
    extra_columns: dict = None,
    filters: list = None,
    variation: str = None,
    chunked: bool = False,
):
    """
    Load samples from a specified directory.
    If chunked=True, yields a dictionary for each parquet file to save memory.
    Otherwise, returns a single dictionary containing concatenated DataFrames.
    """
    if not chunked:
        # --- OLD BEHAVIOR (Kept so other scripts don't break) ---
        events_dict = {}
        for process, datasets in samples.items():
            events_list = []
            for dataset in datasets:
                columns_to_load = columns[:]
                if extra_columns and dataset in extra_columns:
                    columns_to_load += extra_columns[dataset]

                search_path = Path(data_dir / dataset / "parquet" / "nominal" / region)
                if variation:
                    search_path = Path(data_dir / dataset /  "parquet" / variation / region)

                try:
                    if search_path.exists():
                        file_list = [f for f in search_path.iterdir() if f.name.endswith(".parquet")]
                    else:
                        file_list = []

                    if not file_list:
                        continue

                    events = pd.read_parquet(file_list, filters=filters, columns=columns_to_load)
                except Exception as e:
                    warnings.warn(f"Error loading {dataset}: {e}")
                    continue

                if "data" not in process:
                    sum_genweights = get_sum_genweights(data_dir, dataset)
                    sum_gw_f32 = np.float32(sum_genweights)
                    
                    events["weight"] = events["weight"].astype(np.float32)
                    events["weight_nonorm"] = events["weight"]
                    events["finalWeight"] = np.divide(events["weight"].to_numpy(), sum_gw_f32, dtype=np.float32)
                    events["sum_genWeight"] = np.full(len(events), sum_gw_f32, dtype=np.float32)
                else:
                    events["weight_nonorm"] = events["weight"]
                    events["finalWeight"] = events["weight"]

                events_list.append(events)

            if events_list:
                events_dict[process] = events_list[0] if len(events_list) == 1 else pd.concat(events_list, ignore_index=True)
        return events_dict

    else:
        # --- NEW CHUNKED BEHAVIOR (For massive datasets) ---
        for process, datasets in samples.items():
            for dataset in datasets:
                columns_to_load = columns[:]
                if extra_columns and dataset in extra_columns:
                    columns_to_load += extra_columns[dataset]

                search_path = Path(data_dir / dataset / "parquet" / "nominal" / region)
                if variation:
                    search_path = Path(data_dir / dataset /  "parquet" / variation / region)
                
                if search_path.exists():
                    file_list = [f for f in search_path.iterdir() if f.name.endswith(".parquet")]
                else:
                    file_list = []

                if not file_list:
                    continue
                    
                sum_genweights = get_sum_genweights(data_dir, dataset)
                sum_gw_f32 = np.float32(sum_genweights)

                # Process and yield ONE file at a time
                for file_path in file_list:
                    try:
                        events = pd.read_parquet([file_path], filters=filters, columns=columns_to_load)
                    except Exception as e:
                        warnings.warn(f"Error loading {file_path.name}: {e}")
                        continue
                        
                    if "data" not in process:
                        events["weight"] = events["weight"].astype(np.float32)
                        events["weight_nonorm"] = events["weight"]
                        events["finalWeight"] = np.divide(events["weight"].to_numpy(), sum_gw_f32, dtype=np.float32)
                        events["sum_genWeight"] = np.full(len(events), sum_gw_f32, dtype=np.float32)
                    else:
                        events["weight_nonorm"] = events["weight"]
                        events["finalWeight"] = events["weight"]
                        
                    yield {process: events}