#!/bin/bash
# run_hist_container.sh <setup.json> <outdir>
# Non-interactive driver for make_hists_dask.py inside the coffea-dask apptainer
# container (mimics ./shell but execs a command instead of an interactive shell).
# Run from the repo root on LPC (foreground; keep the SSH session attached):
#   bash fitting/run_hist_container.sh setup_zgcr_regmass_check.json results_zgcr_regmass_moddisc
set -e

SETUP="${1:?usage: run_hist_container.sh <setup.json> <outdir>}"
OUTDIR="${2:?usage: run_hist_container.sh <setup.json> <outdir>}"

cd "$(dirname "$0")/.."   # repo root
export X509_USER_PROXY=$(voms-proxy-info -path)
export COFFEA_IMAGE=coffeateam/coffea-dask-almalinux9:2025.12.0-py3.10
export COFFEA_IMAGE_FULL=$(realpath /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE})
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/etc/condor/config.d/01_cmslpc_interactive,/usr/local/bin/cmslpc-local-conf.py:/usr/local/bin/cmslpc-local-conf.py.orig,.cmslpc-local-conf:/usr/local/bin/cmslpc-local-conf.py

mkdir -p /uscmst1b_scratch/lpc1/3DayLifetime/$USER

apptainer exec -B ${PWD}:/srv --pwd /srv "$COFFEA_IMAGE_FULL" /bin/bash -c "
source /srv/.bashrc && cd fitting && python -u make_hists_dask.py --year 2024 --setup ${SETUP} \
  --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024 \
  --save-templates --outdir ${OUTDIR}"
