"""
ROOT Yield Inspector
--------------------
A general-purpose utility to inspect histograms inside ROOT files without opening a GUI.
It calculates the integral (yield) of histograms and automatically estimates a Data/MC 
ratio by looking for keywords like 'Data' or 'EGamma' in the histogram names.

Usage:
    python check_yields.py <filename> [options]

Arguments:
    filename       Path to the .root file you want to check.
    --filter, -f   (Optional) Only process histograms containing this substring.

Examples:
    # 1. Check all histograms in the file
    python check_yields.py results/output.root

    # 2. Check only histograms related to the Control Region (zgcr)
    python check_yields.py results/output.root --filter zgcr

    # 3. Check specific variables (e.g., pT bin 1)
    python check_yields.py results/output.root -f pt1
"""

import uproot
import numpy as np
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Search for histograms and check yields in any ROOT file.")
    parser.add_argument("filename", help="Path to the ROOT file")
    parser.add_argument("--filter", "-f", type=str, default="", help="Only show histograms containing this string (e.g., 'pt1', 'zgcr')")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: '{args.filename}' not found.")
        sys.exit(1)

    print(f"Opening {args.filename}...")
    
    try:
        with uproot.open(args.filename) as f:
            all_keys = f.keys()
            # clean up keys (uproot often adds ;1 ;2 cycles)
            clean_keys = sorted(list(set([k.split(';')[0] for k in all_keys])))
            
            # Filter keys based on user input
            matches = [k for k in clean_keys if args.filter in k]
            
            if not matches:
                print(f"No histograms found matching filter: '{args.filter}'")
                print(f"Available keys (first 5): {clean_keys[:5]}...")
                return

            print(f"\nFound {len(matches)} matching histograms. Checking yields...\n")
            print(f"{'Histogram Name':<50} | {'Events':>12}")
            print("-" * 65)

            total_mc = 0
            data_yield = 0
            found_data = False

            for name in matches:
                # Only process objects that look like histograms (have .values())
                obj = f[name]
                if not hasattr(obj, "values"):
                    continue

                yield_val = np.sum(obj.values())
                print(f"{name:<50} | {yield_val:12.2f}")

                # Heuristic: Try to guess if this is Data or MC
                # Most analyses use 'Data', 'data', or 'EGamma' in the name
                is_data = any(x in name.lower() for x in ['data', 'egamma', 'singlemuon', 'doublemuon'])
                
                if is_data:
                    data_yield += yield_val
                    found_data = True
                else:
                    total_mc += yield_val

            print("-" * 65)
            
            # Summary stats
            if found_data:
                print(f"Total Data (estimated): {data_yield:.2f}")
                print(f"Total MC (estimated)  : {total_mc:.2f}")
                if total_mc > 0:
                    print(f"Data/MC Ratio         : {data_yield/total_mc:.2f}")
                else:
                    print("MC yield is 0.")
            else:
                print(f"Total Sum (No 'Data' found in names): {total_mc:.2f}")

    except Exception as e:
        print(f"Failed to read file: {e}")

if __name__ == "__main__":
    main()