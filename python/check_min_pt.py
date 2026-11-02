#!/usr/bin/env python3
import argparse
import pandas as pd

def main(args):
    file_path = args.file
    column = args.column
    print(f"--- Checking file: {file_path} ---")
    
    try:
        # Load only the column we care about
        events = pd.read_parquet(file_path, columns=[column])
        
        if events.empty:
            print(f"No events found in file.")
            return
        
        min_val = events[column].min()
        max_val = events[column].max()
        
        print(f"Successfully read column '{column}'")
        print(f"  Min value: {min_val}")
        print(f"  Max value: {max_val}")

    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check min/max of a parquet column.")
    parser.add_argument("file", help="Path to the .parquet file")
    parser.add_argument(
        "--column", 
        help="Column to check (use GenBoson_pt for gen-level photon pT)", 
        type=str, 
        default="GenBoson_pt"
    )
    args = parser.parse_args()
    main(args)
