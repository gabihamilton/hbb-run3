#!/bin/bash
# run_vhp_hist_container.sh
# Non-interactive driver for make_hists_dask.py inside the coffea-dask apptainer
# container (mimics ./shell but execs a command instead of an interactive shell).
# Run from the repo root on LPC:  nohup bash fitting/run_vhp_hist_container.sh > fitting/vhp_hist.log 2>&1 &
set -e

cd "$(dirname "$0")/.."   # repo root
export X509_USER_PROXY=$(voms-proxy-info -path)
export COFFEA_IMAGE=coffeateam/coffea-dask-almalinux9:2025.12.0-py3.10
export COFFEA_IMAGE_FULL=$(realpath /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE})
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/etc/condor/config.d/01_cmslpc_interactive,/usr/local/bin/cmslpc-local-conf.py:/usr/local/bin/cmslpc-local-conf.py.orig,.cmslpc-local-conf:/usr/local/bin/cmslpc-local-conf.py

mkdir -p /uscmst1b_scratch/lpc1/3DayLifetime/$USER

apptainer exec -B ${PWD}:/srv --pwd /srv "$COFFEA_IMAGE_FULL" /bin/bash -c '
source /srv/.bashrc && cd fitting && python make_hists_dask.py --year 2024 --setup setup_zgcr_vhp.json \
  --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024 \
  --save-templates --save-plotting-pkl --outdir results_zgcr_vhp'
