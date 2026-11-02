#!/bin/bash
# run_zgcr_vhp_fit.sh
# Standalone Z+gamma tag-and-probe efficiency fit reproducing Raffaele Gerosa's
# boosted Z->bb calibration STRUCTURE on our Z+gamma region.
#
# Structure (self-contained, no mumu region):
#   - z_norm rateParam on zgammabb in BOTH passbb and fail  -> overall Zgamma norm,
#     anchored by the fail (anti-tag) Z peak  =  mu_pre-tag reference.
#   - r_bb mapped to the PASSBB channel ONLY                -> tag efficiency SF
#     =  mu_post-tag / mu_pre-tag,  directly comparable to Raffaele's VHP ~0.92.
#
# Discriminant: p(X->bb) vs QCD > 0.99 (VHP), set in setup_zgcr_vhp.json.
# Requires cmsenv (combineTool.py). Run AFTER make_datacards --analysis zgcr.
#
# Usage:
#   bash run_zgcr_vhp_fit.sh --year 2024 --tag 24Aug07_v15

set -e

YEAR=2024
TAG=24Aug07_v15

while [[ $# -gt 0 ]]; do
    case $1 in
        --year) YEAR=$2; shift 2 ;;
        --tag)  TAG=$2;  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

FITTING_DIR="$(cd "$(dirname "$0")" && pwd)"
# make_datacards --analysis zgcr_vhp -> zgcr_vhpModel_${YEAR}/ ; channel prefix stays "zgcr"
# (from the categories key), so cards are ptbin0zgcrpassbb/fail and shapes zgcr_vhpModel_${YEAR}.root.
ZGCR_DIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/zgcr_vhpModel_${YEAR}"
OUTDIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/zgcr_vhp_fit_${YEAR}"

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "=== Copying + patching Z+gamma passbb/fail datacards (add z_norm on zgammabb) ==="
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "${ZGCR_DIR}/$card" .
    # Overall Zgamma normalization, shared across passbb + fail (pre-tag anchor from fail).
    echo "z_norm rateParam * zgammabb 1 [0,5]" >> $card
done

cp "${ZGCR_DIR}/zgcr_vhpModel_${YEAR}.root" .

echo "=== Combining passbb + fail ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    > zgcr_vhp_${YEAR}.txt

echo "=== Building workspace (r_bb = tag SF, PASSBB only) ==="
text2workspace.py zgcr_vhp_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*passbb.*/zgammabb:r_bb[1,-100,100]" \
    -o workspace_zgcr_vhp_${YEAR}.root

echo "=== FitDiagnostics (r_bb, with shapes) ==="
combine -M FitDiagnostics \
    -d workspace_zgcr_vhp_${YEAR}.root \
    --redefineSignalPOIs r_bb \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _zgcr_vhp_${YEAR}

echo "=== MultiDimFit singles (clean r_bb + z_norm errors) ==="
combine -M MultiDimFit \
    -d workspace_zgcr_vhp_${YEAR}.root \
    --algo singles \
    --redefineSignalPOIs r_bb,z_norm \
    --setParameters r=1 --freezeParameters r \
    --cminDefaultMinimizerStrategy 1 \
    -n _zgcr_vhp_singles_${YEAR}

echo ""
echo "Done. Results in: $OUTDIR"
echo "  r_bb (tag efficiency SF) = Raffaele-comparable; his VHP was 0.92 +0.08/-0.06"
echo "  fitDiagnostics_zgcr_vhp_${YEAR}.root ; higgsCombine_zgcr_vhp_singles_${YEAR}.MultiDimFit.mH120.root"
