from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import uproot


def get_chi2_values(filename):
    if not Path(filename).exists():
        print(f"Error: Could not find file {filename}")
        return None
    try:
        with uproot.open(filename) as f:
            return f["limit"]["limit"].array(library="np")
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


def main(tag):
    suffix = f"_{tag}" if tag else ""
    print(f"--- Reading ROOT files for tag: '{tag}' ---")

    # 1. Load Observed
    file_obs_null = f"higgsCombine_Observed_Null{suffix}.GoodnessOfFit.mH120.root"
    file_obs_alt = f"higgsCombine_Observed_Alt{suffix}.GoodnessOfFit.mH120.root"

    chi2_obs_null = get_chi2_values(file_obs_null)
    chi2_obs_alt = get_chi2_values(file_obs_alt)

    if chi2_obs_null is None or chi2_obs_alt is None:
        return

    val_obs_null = chi2_obs_null[0]
    val_obs_alt = chi2_obs_alt[0]
    q_obs = val_obs_null - val_obs_alt

    print(f"Observed Improvement (q): {q_obs:.4f}")

    # 2. Load Toys
    seed = 123456
    file_toys_null = f"higgsCombine_Toys_Null{suffix}.GoodnessOfFit.mH120.{seed}.root"
    file_toys_alt = f"higgsCombine_Toys_Alt{suffix}.GoodnessOfFit.mH120.{seed}.root"

    vals_toys_null = get_chi2_values(file_toys_null)
    vals_toys_alt = get_chi2_values(file_toys_alt)

    if vals_toys_null is None or vals_toys_alt is None:
        return

    n_toys = min(len(vals_toys_null), len(vals_toys_alt))
    vals_toys_null = vals_toys_null[:n_toys]
    vals_toys_alt = vals_toys_alt[:n_toys]
    q_toys = vals_toys_null - vals_toys_alt

    # 3. Stats
    n_above = np.sum(q_toys > q_obs)
    p_value = n_above / n_toys

    print(f"P-Value: {p_value:.4f}")
    if p_value < 0.05:
        print(">> SIGNIFICANT (Order 1 needed)")
    else:
        print(">> NOT SIGNIFICANT (Stick to Order 0)")

    # 4. Plot
    plt.figure(figsize=(8, 6))
    plt.hist(q_toys, bins=20, alpha=0.6, label="Toys (Null Hypothesis)")
    plt.axvline(
        q_obs, color="red", linestyle="dashed", linewidth=2, label=f"Observed (p={p_value:.3f})"
    )
    plt.xlabel(r"$\Delta \chi^2$")
    plt.title(f"F-Test: Order 0 vs 1 ({tag})")
    plt.legend()

    outname = f"ftest_result{suffix}.png"
    plt.savefig(outname)
    print(f"Plot saved as {outname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="", help="Tag used in filenames (e.g. 2022)")
    args = parser.parse_args()
    main(args.tag)
