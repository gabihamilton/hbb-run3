#!/usr/bin/env python3
import argparse
import pyarrow.parquet as pq

def main(args):
    file_path = args.file
    print(f"--- Checking file: {file_path} ---")

    try:
        # Open the parquet file
        parquet_file = pq.ParquetFile(file_path)
        
        # 1. Check if 'MET' is in the schema at all
        if 'MET' not in parquet_file.schema.names:
            print("ERROR: 'MET' column not found in file schema.")
            return

        # 2. Print the schema for just the 'MET' column
        met_field = parquet_file.schema.field('MET')
        print(f"\nSchema for 'MET': {met_field}")
        
        # 3. Try to read *only* the 'MET' column
        print("\nAttempting to read the 'MET' column...")
        met_table = parquet_file.read(columns=['MET'])
        
        print("\nSuccessfully read 'MET' column.")
        
        # 4. Print the first 5 values
        print("First 5 'MET' values:")
        print(met_table['MET'][:5])

    except Exception as e:
        print(f"\n--- SCRIPT FAILED ---")
        print(f"An error occurred: {e}")
        print("This confirms the 'MET' column in this file is corrupted or unreadable by pyarrow/pandas.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the MET column in a parquet file.")
    parser.add_argument("file", help="Path to the .parquet file to check")
    args = parser.parse_args()
    main(args)
