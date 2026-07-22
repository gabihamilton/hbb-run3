#!/bin/bash
# run_combined_fit.sh
# Combined Z+gamma + Z->mumu fit with shared z_norm constraint.
#
# Usage:
#   bash run_combined_fit.sh --year 2024 --tag Test_v15
#
# Requires cmsenv to be active (for combineTool.py).

set -e

YEAR=2024
TAG=Test_v15

while [[ $# -gt 0 ]]; do
    case $1 in
        --year) YEAR=$2; shift 2 ;;
        --tag)  TAG=$2;  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

ZGCR_DIR="results/${TAG}/${YEAR}/datacards/zgcrModel_${YEAR}"
ZMMCR_DIR="results/${TAG}/${YEAR}/datacards/zmmcrModel_${YEAR}"
OUTDIR="results/${TAG}/${YEAR}/datacards/combined_${YEAR}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "=== Copying and patching Z+gamma datacards ==="
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrpasscc${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "../../../../${ZGCR_DIR}/$card" .
    # Add z_norm rateParam to link with Z->mumu CR
    echo "z_norm rateParam * zgammabb 1 [0,5]" >> $card
    echo "z_norm rateParam * zgammacc 1 [0,5]" >> $card
done

echo "=== Copying Z->mumu datacard ==="
cp "../../../../${ZMMCR_DIR}/ptbin0zmmcrinclusive${YEAR}.txt" .

# Copy ROOT files needed by the datacards (shapes)
cp "../../../../${ZGCR_DIR}/zgcrModel_${YEAR}.root" .
cp "../../../../${ZMMCR_DIR}/zmmcrModel_${YEAR}.root" .

echo "=== Combining datacards ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrpasscc${YEAR}=ptbin0zgcrpasscc${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    ptbin0zmmcrinclusive${YEAR}=ptbin0zmmcrinclusive${YEAR}.txt \
    > combined_zgcr_zmmcr_${YEAR}.txt

echo "=== Building workspace ==="
text2workspace.py combined_zgcr_zmmcr_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_cc[1,-100,100]" \
    -o workspace_combined_${YEAR}.root

echo "=== Running FitDiagnostics ==="
combine -M FitDiagnostics \
    -d workspace_combined_${YEAR}.root \
    --redefineSignalPOIs r_bb,r_cc \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _combined_${YEAR}

echo ""
echo "Done. Results in: $OUTDIR"
echo "  Workspace:     workspace_combined_${YEAR}.root"
echo "  FitDiagnostics: fitDiagnostics_combined_${YEAR}.root"
