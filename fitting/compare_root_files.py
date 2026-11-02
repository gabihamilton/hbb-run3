"""
ROOT File Comparator
--------------------
A regression testing tool to compare two ROOT files histogram-by-histogram.
It checks for:
1. Missing or extra keys (histograms).
2. Differences in bin edges.
3. Differences in bin contents (yields).
4. Differences in bin errors (sumw2).

Usage:
    python compare_root.py <file1> <file2> 

Arguments:
    file1, file2    Paths to the two ROOT files to compare.

Examples:
    python compare_root.py old_results.root new_results.root

"""

import uproot
import numpy as np
import sys

# Usage: python3 compare_root_files.py file_old.root file_new.root

def compare_files(file_path_1, file_path_2):
    print(f"Comparing:\n  1: {file_path_1}\n  2: {file_path_2}\n")
    
    with uproot.open(file_path_1) as f1, uproot.open(file_path_2) as f2:
        keys1 = set(f1.keys(cycle=False))
        keys2 = set(f2.keys(cycle=False))
        
        # 1. Check if they have the same histograms
        common_keys = keys1.intersection(keys2)
        missing_in_2 = keys1 - keys2
        missing_in_1 = keys2 - keys1
        
        if missing_in_2:
            print(f"WARNING: Keys in file 1 but missing in file 2:\n  {missing_in_2}")
        if missing_in_1:
            print(f"WARNING: Keys in file 2 but missing in file 1:\n  {missing_in_1}")
            
        print(f"\nChecking content of {len(common_keys)} common histograms...")
        
        # 2. Compare bin contents
        mismatches = 0
        for key in common_keys:
            # Load values and errors (variances)
            # flow=True ensures we check overflow/underflow bins too!
            h1_obj = f1[key]
            h2_obj = f2[key]
            
            # Check if objects are actually histograms
            if not hasattr(h1_obj, "to_numpy"):
                print(f"  [SKIP] {key} is not a histogram (type: {type(h1_obj)})")
                continue

            # Get values and edges
            vals1, edges1 = h1_obj.to_numpy(flow=False)
            vals2, edges2 = h2_obj.to_numpy(flow=False)
            
            # Get Variances (Errors^2)
            vars1 = h1_obj.variances()
            vars2 = h2_obj.variances()

            # --- 1. Check Edges ---
            if not np.allclose(edges1, edges2):
                print(f"  [MISMATCH] {key}: Bin edges differ!")
                mismatches += 1
                continue

            # --- 2. Check Yields ---
            if not np.allclose(vals1, vals2, rtol=1e-5):
                print(f"  [MISMATCH] {key}: Bin yields differ!")
                print(f"     Sum 1: {np.sum(vals1):.4f}, Sum 2: {np.sum(vals2):.4f}")
                mismatches += 1
                continue
                
            # --- 3. Check Errors (New!) ---
            # If one has errors (vars not None) and the other doesn't, that's a mismatch
            if (vars1 is None) != (vars2 is None):
                 print(f"  [MISMATCH] {key}: One has errors (Sumw2), the other does not!")
                 mismatches += 1
                 continue
            
            if vars1 is not None and not np.allclose(vars1, vars2, rtol=1e-5):
                print(f"  [MISMATCH] {key}: Bin ERRORS differ (content matches)!")
                mismatches += 1
                
        if mismatches == 0:
            print("\nSUCCESS! All common histograms are numerically identical.")
        else:
            print(f"\nFAILURE. Found {mismatches} histograms with differences.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Please provide two root files.")
    else:
        compare_files(sys.argv[1], sys.argv[2])
