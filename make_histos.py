import sys
import os
import pickle
from pathlib import Path
import numpy as np
import hist
from hbb import utils
import argparse
import json

# Argument parser for command-line input
parser = argparse.ArgumentParser(description="Process histograms for a given year.")
parser.add_argument("year", type=str, help="Year for data processing")
args = parser.parse_args()

# Use the provided year
year = args.year  # 2023


MAIN_DIR = "/Users/gbibim/Here/genZ/data"
#dir_name = "children" #"new"  # data for older samples new for the files with lhe variables
dir_name = "PNetchildren" 
path_to_dir = f"{MAIN_DIR}/{dir_name}/" 


# Define samples file
args.samples_file = f"samples.json"
# Load samples from JSON file
with open(args.samples_file, "r") as f:
    samples = json.load(f)

dirs = {path_to_dir: samples}

load_columns = [
    ("weight", 1),
    ("ak8FatJetPt", 1),
    ("ak8FatJetmsoftdrop", 1),
    ('ak8FatJetParTPQCD1HF', 1),
    ('ak8FatJetParTPQCD2HF', 1),
    ('ak8FatJetParTPQCD0HF', 1),
    ('ak8FatJetParTPXbb', 1),
    ("AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35", 1), # for 2022 and a small fraction of 2023
    ("AK8PFJet230_SoftDropMass40_PNetBB0p06", 1), #new for 2023
]

load_columns_V = load_columns + [
    ("GenVPt", 1),
    ("GenVis_bb", 1),
    ("GenVis_cc", 1),
    ("GenVis_cs", 1),
]
    

# Define pt bins
ptbins = np.array([450,  500, 550, 600, 675, 800, 1200]) # 500, 550, 600, 675, 800,



# Define histogram axes
msd_axis = hist.axis.Regular(24, 40, 200, name="msd", label="mSD [GeV]")  

# Initialize histogram dictionary
histograms = {}

for category in samples.keys():
    histograms[category] = {
        "pass": {bin_edge: hist.Hist(msd_axis) for bin_edge in ptbins[:-1]},
        "fail": {bin_edge: hist.Hist(msd_axis) for bin_edge in ptbins[:-1]},
    }

def get_ptbin(pt):
    """Returns an array of bin lower edges corresponding to pt values."""
    pt = np.asarray(pt)  # Ensure pt is a NumPy array

    # Get the bin for each element in pt
    bins = np.digitize(pt, ptbins) - 1  # Find which bin each pt belongs to

    # Map bins to lower edge of corresponding ptbin, handling out-of-range cases
    bins[bins < 0] = 0  # Assign lowest bin if pt is too small
    bins[bins >= len(ptbins) - 1] = len(ptbins) - 2  # Assign highest valid bin

    return ptbins[bins]  # Return the corresponding bin lower edge

# Function to fill histograms
def fill_mass(events, zto, sample):
    for key, data in events.items():
        weight = data["finalWeight"]
        msd = data["ak8FatJetmsoftdrop"][0]
        pt = data["ak8FatJetPt"][0]
        Pxbb = data["ak8FatJetParTPXbb"][0]
        PQCD = (
            data["ak8FatJetParTPQCD1HF"][0]
            + data["ak8FatJetParTPQCD2HF"][0]
            + data["ak8FatJetParTPQCD0HF"][0]
        )

        # Compute discriminator
        Txbb = Pxbb / (Pxbb + PQCD)

        HLTs = ( data["AK8PFJet250_SoftDropMass40_PFAK8ParticleNetBB0p35"][0] |
            data["AK8PFJet230_SoftDropMass40_PNetBB0p06"][0] 
        )

        selection = (msd > 60) & (HLTs) & (Txbb>0.90) & (pt > 450) & (pt < 1200) & (msd < 200)
        fail = (Txbb<0.95) & (msd > 60) & (HLTs) & (pt > 450) & (pt < 1200) & (msd < 200)

        # Get ptbin for each pt value
        ptbins_selected = get_ptbin(pt[selection])
        ptbins_fail = get_ptbin(pt[fail])

        # Fill histograms per ptbin
        for pbin, msd_val, weight_val in zip(ptbins_selected, msd[selection], weight[selection]):
            histograms[category]["pass"][pbin].fill(msd_val, weight=weight_val)

        for pbin, msd_val, weight_val in zip(ptbins_fail, msd[fail], weight[fail]):
            histograms[category]["fail"][pbin].fill(msd_val, weight=weight_val)

        # Clear intermediate arrays
        del weight, msd, Pxbb, PQCD, selection, HLTs, fail, ptbins_selected, ptbins_fail


# Process samples
for category, sample_list in samples.items():
    for input_dir, dirs_samples in dirs.items():
        # Loop through each sample individually to avoid loading everything at once
        for sample in sample_list:
            try:
                # Load only one sample at a time
                events = utils.load_samples(
                    input_dir,
                    category,
                    [sample],  # List containing a single sample
                    year,
                    columns=utils.format_columns(
                        load_columns_V if category in {"Zto2Q", "Wto2Q"} else load_columns
                    ),
                )

                # Fill histograms with the loaded sample
                fill_mass(events, category, sample)  # See function definition below
                #fill_discriminator(events, zto, sample)  # See function definition

            except KeyError as e:
                print(f"Warning: Missing key {e} in sample {sample}. Skipping.")

            # Ensure the sample is deleted from memory after use
            del events


# Define the output file
output_file = f"histograms_{year}.pkl"

# Save histograms to a pickle 
with open(output_file, "wb") as f:
    pickle.dump(histograms, f)

print(f"Histograms saved to {output_file}")
