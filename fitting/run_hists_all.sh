#!/bin/bash

TAG="25Nov11_v14_private"

# Loop over the 4 eras
for YEAR in 2022 2022EE 2023 2023BPix
do
    echo "Processing Histograms for $YEAR..."
    python3 make_zg_hists.py --year $YEAR --tag $TAG
done
