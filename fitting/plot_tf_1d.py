# python3 plot_tf_1d.py --year 2022 --tag 25Nov11_v14_private

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
import json
import argparse
from pathlib import Path
import rhalphalib as rl

plt.style.use(hep.style.CMS)

def plot_tf_slice(args):
    year = args.year
    tag = args.tag
    cat = "zgcr"
    region = "bb" # or 'cc'
    
    # 1. Load Config to get ranges
    with open("setup_zgamma.json") as f:
        setup = json.load(f)
        msd_cfg = setup["observable"]
        
    # Define Physics Range
    pt_min, pt_max = 200, 1200
    rho_min, rho_max = -6.0, -2.1
    
    # 2. Load Initial Values (Polynomial Parameters)
    initF = Path(f"results/{tag}/{year}/initial_vals/initial_vals_{cat}_{region}.json")
    if not initF.exists():
        print(f"Error: {initF} not found. Run the card maker first.")
        return

    with open(initF) as f:
        initial_vals = np.array(json.load(f)['initial_vals'])
        
    print(f"Loaded parameters shape: {initial_vals.shape}")

    # 3. Create the Polynomial Object
    # We reconstruct the same basis function used in the fit
    tf_poly = rl.BasisPoly("tf_debug",
                           (initial_vals.shape[0]-1, initial_vals.shape[1]-1),
                           ['pt', 'rho'], 
                           basis='Bernstein',
                           init_params=initial_vals,
                           limits=(0, 1), 
                           coefficient_transform=None)

    # 4. Generate Points for Plotting
    # We choose 3 fixed pT slices to show how it evolves
    pt_slices = [300, 600, 900] 
    msd_range = np.linspace(msd_cfg["min"], msd_cfg["max"], 100)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for pt_val in pt_slices:
        # Calculate Scaled Coordinates
        # pT Scaling: (pt - 200) / (1200 - 200)
        pt_scaled = (pt_val - pt_min) / (pt_max - pt_min)
        
        # Rho Scaling: 2*log(msd/pt) mapped to [0,1]
        rho_val = 2 * np.log(msd_range / pt_val)
        rho_scaled = (rho_val - rho_min) / (rho_max - rho_min)
        
        # Valid Mask (TF is only defined inside the rho cuts)
        valid_mask = (rho_scaled >= 0) & (rho_scaled <= 1)
        
        # Evaluate Polynomial
        # Input must be same shape, so we broadcast pt_scaled
        pt_input = np.full_like(rho_scaled, pt_scaled)
        tf_values = tf_poly(pt_input, rho_scaled, nominal=True)
        
        # Plot
        ax.plot(msd_range[valid_mask], tf_values[valid_mask], 
                linewidth=2, label=f"$p_T = {pt_val}$ GeV")

    ax.set_xlabel(r"$m_{SD}$ [GeV]")
    ax.set_ylabel("Transfer Factor (Pass / Fail)")
    ax.set_title(f"QCD Transfer Factor ({year}, Region {region})", fontsize=20)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    outname = f"results/{tag}/{year}/plots/TF_1D_slice_{region}.png"
    plt.savefig(outname)
    print(f"Saved plot to {outname}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    args = parser.parse_args()
    
    plot_tf_slice(args)
