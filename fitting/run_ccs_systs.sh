#!/bin/bash
# run_ccs_systs.sh
# Combined Z+gamma + Z->mumu, ccs-joint model (r_bb, r_ccs), shared z_norm.
# Uses the STANDARD zgcr card (which carries the experimental systematics).
# NO theory_unc. Output: combined_ccs_systs_<YEAR>/
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
DCROOT="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards"
ZGCR_DIR="${DCROOT}/zgcrModel_${YEAR}"
ZMMCR_DIR="${DCROOT}/zmmcrModel_${YEAR}"
OUTDIR="${DCROOT}/combined_ccs_systs_${YEAR}"

mkdir -p "$OUTDIR"; cd "$OUTDIR"

echo "=== Copy + patch Z+gamma cards (add z_norm rateParam) ==="
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrpasscc${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "${ZGCR_DIR}/$card" .
    echo "z_norm rateParam * zgammabb 1 [0,5]" >> $card
    echo "z_norm rateParam * zgammacc 1 [0,5]" >> $card
done
echo "=== Copy Z->mumu card (already has z_norm) ==="
cp "${ZMMCR_DIR}/ptbin0zmmcrinclusive${YEAR}.txt" .
cp "${ZGCR_DIR}/zgcrModel_${YEAR}.root" .
cp "${ZMMCR_DIR}/zmmcrModel_${YEAR}.root" .

echo "=== combineCards ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrpasscc${YEAR}=ptbin0zgcrpasscc${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    ptbin0zmmcrinclusive${YEAR}=ptbin0zmmcrinclusive${YEAR}.txt \
    > combined_ccs_systs_${YEAR}.txt

echo "=== text2workspace (cc+cs tied to r_ccs) ==="
text2workspace.py combined_ccs_systs_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_ccs[1,-100,100]" \
    --PO "map=.*/wgammacs:r_ccs[1,-100,100]" \
    -o workspace_ccs_systs_${YEAR}.root

echo "=== FitDiagnostics (r_bb, r_ccs) ==="
combine -M FitDiagnostics \
    -d workspace_ccs_systs_${YEAR}.root \
    --redefineSignalPOIs r_bb,r_ccs \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _ccs_systs_${YEAR}

echo "=== DONE OUTDIR=$OUTDIR ==="
