#!/bin/bash

TAG="25Nov11_v14_private"

for YEAR in 2022 2022EE 2023 2023BPix
do
    echo "Making Card for $YEAR..."
    python3 make_zg_cards.py --year $YEAR --tag $TAG
done
