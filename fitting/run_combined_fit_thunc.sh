#!/bin/bash
# run_combined_fit_thunc.sh
# Combined Z+gamma + Z->mumu fit WITH the theory_unc placeholder (20% lognormal, shared).
# Identical to run_combined_fit.sh EXCEPT:
#   - reads theory_unc datacards from THUNC_DIR (so the originals stay untouched)
#   - writes everything to combined_theoryunc_<YEAR>/ (so combined_<YEAR>/ stays as the no-thunc version)
#
# Requires cmsenv. Put the 4 theory_unc cards in THUNC_DIR first (see comment below).
#
# Usage:
#   bash run_combined_fit_thunc.sh --year 2024 --tag Test_v15

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
ZGCR_DIR="${DCROOT}/zgcrModel_${YEAR}"        # source of the ROOT shape file only
ZMMCR_DIR="${DCROOT}/zmmcrModel_${YEAR}"      # source of the ROOT shape file only
THUNC_DIR="${DCROOT}/thunc_cards_${YEAR}"     # <-- theory_unc datacards live here (scp them in first)
OUTDIR="${DCROOT}/combined_theoryunc_${YEAR}"

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
    > combined_zgcr_zmmcr_thunc_${YEAR}.txt

echo "=== Building workspace ==="
text2workspace.py combined_zgcr_zmmcr_thunc_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_cc[1,-100,100]" \
    -o workspace_combined_thunc_${YEAR}.root

echo "=== Running FitDiagnostics ==="
combine -M FitDiagnostics \
    -d workspace_combined_thunc_${YEAR}.root \
    --redefineSignalPOIs r_bb,r_cc \
    --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties \
    --cminDefaultMinimizerStrategy 1 \
    -n _combined_thunc_${YEAR}

echo ""
echo "Done. Results in: $OUTDIR"
echo "  Workspace:      workspace_combined_thunc_${YEAR}.root"
echo "  FitDiagnostics: fitDiagnostics_combined_thunc_${YEAR}.root"
echo "Compare r_bb / r_cc (and their errors) vs combined_${YEAR}/ (no theory_unc)."
