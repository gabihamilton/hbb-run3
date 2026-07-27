#!/bin/bash
# run_decorrelation_test.sh
# Runs three separate fits to test pass_bb / pass_cc background decorrelation:
#   1. pass_bb + fail only  -> r_bb
#   2. pass_cc + fail only  -> r_cc
#   3. pass_bb + pass_cc + fail simultaneously -> r_bb, r_cc
# Compare nuisance pulls across the three fits to check for background correlation.
#
# Usage:
#   bash run_decorrelation_test.sh --year 2024 --tag Test_v15
#
# Requires cmsenv to be active.

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

FITTING_DIR="$(cd "$(dirname "$0")" && pwd)"
ZGCR_DIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/zgcrModel_${YEAR}"
OUTDIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/decorrelation_${YEAR}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

# Copy shape ROOT file
cp "${ZGCR_DIR}/zgcrModel_${YEAR}.root" .

# Copy datacards
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrpasscc${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "${ZGCR_DIR}/$card" .
done

echo "=== Fit 1: pass_bb + fail only (r_bb) ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    > card_passbb_fail_${YEAR}.txt

text2workspace.py card_passbb_fail_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_cc[1,1,1]" \
    -o workspace_passbb_${YEAR}.root

combine -M FitDiagnostics \
    -d workspace_passbb_${YEAR}.root \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _passbb_only_${YEAR}

echo ""
echo "=== Fit 2: pass_cc + fail only (r_cc) ==="
combineCards.py \
    ptbin0zgcrpasscc${YEAR}=ptbin0zgcrpasscc${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    > card_passcc_fail_${YEAR}.txt

text2workspace.py card_passcc_fail_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,1,1]" \
    --PO "map=.*/zgammacc:r_cc[1,-100,100]" \
    -o workspace_passcc_${YEAR}.root

combine -M FitDiagnostics \
    -d workspace_passcc_${YEAR}.root \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _passcc_only_${YEAR}

echo ""
echo "=== Fit 3: pass_bb + pass_cc + fail simultaneously (r_bb, r_cc) ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrpasscc${YEAR}=ptbin0zgcrpasscc${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    > card_passbb_passcc_fail_${YEAR}.txt

text2workspace.py card_passbb_passcc_fail_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_cc[1,-100,100]" \
    -o workspace_simultaneous_${YEAR}.root

combine -M FitDiagnostics \
    -d workspace_simultaneous_${YEAR}.root \
    --redefineSignalPOIs r_bb,r_cc \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _simultaneous_${YEAR}

echo ""
echo "Done. Results in: $OUTDIR"
echo "  Fit 1 (bb only):       fitDiagnosticsTest_passbb_only_${YEAR}.root"
echo "  Fit 2 (cc only):       fitDiagnosticsTest_passcc_only_${YEAR}.root"
echo "  Fit 3 (simultaneous):  fitDiagnosticsTest_simultaneous_${YEAR}.root"
echo ""
echo "Compare nuisance pulls (especially JMS, JMR, lumi) across the three fits."
