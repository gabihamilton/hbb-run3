#!/bin/bash
# run_combined_fit_thunc_ccs.sh
# "Everything" fit: Z+gamma + Z->mumu combined, WITH theory_unc, AND cc+cs tied to one POI r_ccs.
# Model per process:
#   zgammabb : z_norm * r_bb
#   zgammacc : z_norm * r_ccs      (Z normalization x cc+cs tagger SF)
#   wgammacs : r_ccs               (W->cs, tied to cc; no z_norm since it is not a Z process)
#   zmm      : z_norm
# Outputs to combined_theoryunc_ccs_<YEAR>/ (separate from the separate-r_cc thunc version).
#
# Requires cmsenv. Theory_unc cards must already be in THUNC_DIR (same as run_combined_fit_thunc.sh).
#
# Usage:
#   bash run_combined_fit_thunc_ccs.sh --year 2024 --tag Test_v15

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
THUNC_DIR="${DCROOT}/thunc_cards_${YEAR}"
OUTDIR="${DCROOT}/combined_theoryunc_ccs_${YEAR}"

if [[ ! -d "$THUNC_DIR" ]]; then
    echo "ERROR: $THUNC_DIR does not exist. scp the 4 theory_unc cards there first."; exit 1
fi

mkdir -p "$OUTDIR"
cd "$OUTDIR"

echo "=== Copying and patching Z+gamma theory_unc datacards (add z_norm rateParam) ==="
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrpasscc${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "${THUNC_DIR}/$card" .
    echo "z_norm rateParam * zgammabb 1 [0,5]" >> $card
    echo "z_norm rateParam * zgammacc 1 [0,5]" >> $card
done

echo "=== Copying Z->mumu theory_unc datacard (already has z_norm) ==="
cp "${THUNC_DIR}/ptbin0zmmcrinclusive${YEAR}.txt" .

echo "=== Copying ROOT shape files ==="
cp "${ZGCR_DIR}/zgcrModel_${YEAR}.root" .
cp "${ZMMCR_DIR}/zmmcrModel_${YEAR}.root" .

echo "=== Combining datacards ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrpasscc${YEAR}=ptbin0zgcrpasscc${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    ptbin0zmmcrinclusive${YEAR}=ptbin0zmmcrinclusive${YEAR}.txt \
    > combined_zgcr_zmmcr_thunc_ccs_${YEAR}.txt

echo "=== Building workspace (cc+cs tied to r_ccs) ==="
text2workspace.py combined_zgcr_zmmcr_thunc_ccs_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_ccs[1,-100,100]" \
    --PO "map=.*/wgammacs:r_ccs[1,-100,100]" \
    -o workspace_combined_thunc_ccs_${YEAR}.root

echo "=== Running FitDiagnostics (r_bb, r_ccs) ==="
combine -M FitDiagnostics \
    -d workspace_combined_thunc_ccs_${YEAR}.root \
    --redefineSignalPOIs r_bb,r_ccs \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _combined_thunc_ccs_${YEAR}

echo ""
echo "Done. Results in: $OUTDIR"
echo "  Workspace:      workspace_combined_thunc_ccs_${YEAR}.root"
echo "  FitDiagnostics: fitDiagnostics_combined_thunc_ccs_${YEAR}.root"
echo "Read r_bb, r_ccs, z_norm, theory_unc from fit_s (2-POI -> use single-POI for clean r_bb/r_ccs errors)."
