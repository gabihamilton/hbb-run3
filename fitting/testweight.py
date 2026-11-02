import uproot
import pandas as pd
import numpy as np
from pathlib import Path

# Update this path to one of your existing MC parquet files
sample_file = "/eos/uscms/store/group/lpchbbrun3/skims/26Jan16/2022/ZZ/parquet/nominal/signal-vh/part0.parquet"

def check_weights(file_path):
    print(f"Checking file: {file_path}")
    # Load a small chunk of data
    df = pd.read_parquet(file_path, columns=["weight", "pileupUp"])
    
    nominal_mean = df["weight"].mean()
    syst_mean = df["pileupUp"].mean()
    ratio = syst_mean / nominal_mean

    print(f"--- Weight Diagnostic ---")
    print(f"Nominal Weight Mean: {nominal_mean:.2e}")
    print(f"Systematic Weight Mean: {syst_mean:.2e}")
    
    if 0.5 < ratio < 1.5:
        print("\nRESULT: [Full Weight detected]")
        print("Your script should use: weight = data[syst]")
    else:
        print("\nRESULT: [Scale Factor detected]")
        print("Your script should use: weight = data['finalWeight'] * data[syst]")
        print(f"Ratio is {ratio:.2f} (Expected ~1.0 if it was a full weight)")

if __name__ == "__main__":
    if Path(sample_file).exists():
        check_weights(sample_file)
    else:
        print("File not found. Please update 'sample_file' path.")