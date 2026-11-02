#!/bin/bash
# run_zgcr_vhp_fit_B_mumu.sh
# METHOD B — mumu-referenced tag-and-probe (literal Raffaele structure = your slide, corrected).
#   z_norm on zgammabb (passbb+fail) AND on zmm (shared) -> mu_pre-tag anchored by the Z->mumu CR.
#   r_bb mapped to PASSBB only                            -> tag efficiency SF = r_bb.
# Reuses the SAME VHP Zgamma templates as Method A; additionally needs the zmmcr datacard.
#
# PREREQ: the zmmcr card must exist at
#   results/${TAG}/${YEAR}/datacards/zmmcrModel_${YEAR}/ptbin0zmmcrinclusive${YEAR}.txt
# (make_zmmcr_datacard.py; its card already carries `z_norm rateParam * zmm`).
#
# Requires cmsenv. Usage: bash run_zgcr_vhp_fit_B_mumu.sh --year 2024 --tag 24Aug07_v15
set -e
YEAR=2024
TAG=24Aug07_v15
MODEL=zgcr_vhp   # datacard model name (make_datacards --analysis); pt400 variant = zgcr_vhp_pt400
ZMMDIR=""        # override path to the zmmcrModel dir (default = plain-DY μμ card); μμ+γ variant sets this
SUFFIX=""        # output-dir suffix to keep variants separate (e.g. _gamma)
while [[ $# -gt 0 ]]; do
  case $1 in
    --year)   YEAR=$2;   shift 2 ;;
    --tag)    TAG=$2;    shift 2 ;;
    --model)  MODEL=$2;  shift 2 ;;
    --zmmdir) ZMMDIR=$2; shift 2 ;;
    --suffix) SUFFIX=$2; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

FITTING_DIR="$(cd "$(dirname "$0")" && pwd)"
ZGCR_DIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/${MODEL}Model_${YEAR}"
ZMMCR_DIR="${ZMMDIR:-${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/zmmcrModel_${YEAR}}"
OUTDIR="${FITTING_DIR}/results/${TAG}/${YEAR}/datacards/${MODEL}_fit_B${SUFFIX}_${YEAR}"

if [ ! -f "${ZMMCR_DIR}/ptbin0zmmcrinclusive${YEAR}.txt" ]; then
  echo "ERROR: zmmcr card not found at ${ZMMCR_DIR}. Produce it first (make_zmmcr_datacard.py)."
  exit 2
fi

mkdir -p "$OUTDIR"; cd "$OUTDIR"

echo "=== Zgamma passbb/fail (add z_norm on zgammabb) ==="
for card in ptbin0zgcrpassbb${YEAR}.txt ptbin0zgcrfail${YEAR}.txt; do
    cp "${ZGCR_DIR}/$card" .
    echo "z_norm rateParam * zgammabb 1 [0,5]" >> $card
done
cp "${ZGCR_DIR}/${MODEL}Model_${YEAR}.root" .

echo "=== Z->mumu card (already carries z_norm on zmm) ==="
cp "${ZMMCR_DIR}/ptbin0zmmcrinclusive${YEAR}.txt" .
cp "${ZMMCR_DIR}/zmmcrModel_${YEAR}.root" .

echo "=== Combine cards (Zgamma passbb+fail + mumu) ==="
combineCards.py \
    ptbin0zgcrpassbb${YEAR}=ptbin0zgcrpassbb${YEAR}.txt \
    ptbin0zgcrfail${YEAR}=ptbin0zgcrfail${YEAR}.txt \
    ptbin0zmmcrinclusive${YEAR}=ptbin0zmmcrinclusive${YEAR}.txt \
    > zgcr_vhp_B_${YEAR}.txt

echo "=== Workspace (r_bb = tag SF, PASSBB only; z_norm shared with mumu) ==="
text2workspace.py zgcr_vhp_B_${YEAR}.txt \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*passbb.*/zgammabb:r_bb[1,-100,100]" \
    -o workspace_zgcr_vhp_B_${YEAR}.root

echo "=== FitDiagnostics + MultiDimFit singles ==="
combine -M FitDiagnostics -d workspace_zgcr_vhp_B_${YEAR}.root \
    --redefineSignalPOIs r_bb --setParameters r=1 --freezeParameters r \
    --saveShapes --saveWithUncertainties --cminDefaultMinimizerStrategy 1 \
    -n _zgcr_vhp_B_${YEAR}
combine -M MultiDimFit -d workspace_zgcr_vhp_B_${YEAR}.root --algo singles \
    --redefineSignalPOIs r_bb,z_norm --setParameters r=1 --freezeParameters r \
    --cminDefaultMinimizerStrategy 1 -n _zgcr_vhp_B_singles_${YEAR}

echo ""
echo "Done (Method B). r_bb = mumu-referenced tag SF; compare to Method A + Raffaele 0.92."
