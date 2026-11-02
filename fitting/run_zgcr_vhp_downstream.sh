#!/bin/bash
# run_zgcr_vhp_downstream.sh  (hbb env, NOT cmsenv)
# After make_hists_dask.py has populated the EOS staging shelves for the VHP run:
#   collect_hists -> bridge collect's generic name to the setup root_filename -> make_datacards.
# Usage:  bash run_zgcr_vhp_downstream.sh --year 2024 --tag 24Aug07_v15
set -e
YEAR=2024
TAG=24Aug07_v15
while [[ $# -gt 0 ]]; do
  case $1 in
    --year) YEAR=$2; shift 2 ;;
    --tag)  TAG=$2;  shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done
cd "$(dirname "$0")"   # fitting/

echo "=== collect_hists (VHP) ==="
python collect_hists.py --year "$YEAR" --tag "$TAG" \
  --outdir results_zgcr_vhp --save-templates --save-plotting-pkl

INDIR="results_zgcr_vhp/${TAG}"
echo "=== collect wrote (generic name): ==="
ls -la "${INDIR}"/*.root

# collect always writes fitting_{year}_msd.root; make_datacards wants root_filename from
# setup_zgcr_vhp.json = fitting_{year}_zgcr_vhp.root
cp "${INDIR}/fitting_${YEAR}_msd.root" "${INDIR}/fitting_${YEAR}_zgcr_vhp.root"

echo "=== make_datacards (--analysis zgcr_vhp reads setup_zgcr_vhp.json) ==="
python make_datacards.py --year "$YEAR" --tag "$TAG" --analysis zgcr_vhp --indir "$INDIR"

echo ""
echo "Done. Datacards in results/${TAG}/${YEAR}/datacards/zgcr_vhpModel_${YEAR}/"
echo "Next (cmsenv, separate terminal):  bash run_zgcr_vhp_fit.sh --year ${YEAR} --tag ${TAG}"
