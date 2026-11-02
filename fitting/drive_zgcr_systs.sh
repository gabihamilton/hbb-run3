#!/bin/bash
# Non-interactive replica of ./shell + make_hists_dask.py for the zgcr systs run.
set +e
cd ~/nobackup/hbb-run3 || exit 9
export X509_USER_PROXY=$HOME/x509up_u24293
export COFFEA_IMAGE=coffeateam/coffea-dask-almalinux9:2025.12.0-py3.10
export COFFEA_IMAGE_FULL=$(realpath /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE})
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/etc/condor/config.d/01_cmslpc_interactive,/usr/local/bin/cmslpc-local-conf.py:/usr/local/bin/cmslpc-local-conf.py.orig,.cmslpc-local-conf:/usr/local/bin/cmslpc-local-conf.py

echo "=== HOST=$(hostname) START=$(date) ==="
echo "=== PROXY: $(voms-proxy-info -timeleft 2>/dev/null) sec left ==="
echo "=== IMAGE_FULL=$COFFEA_IMAGE_FULL ==="

apptainer exec -B ${PWD}:/srv --pwd /srv "$COFFEA_IMAGE_FULL" /bin/bash -c '
  source /srv/.bashrc >/dev/null 2>&1
  cd /srv/fitting || exit 8
  echo "=== python: $(which python) ==="
  python -c "import distributed, dask; print(\"distributed\", distributed.__version__, \"| dask\", dask.__version__)" || { echo "NO_DISTRIBUTED"; exit 7; }
  echo "=== active_systematics ==="; grep active_systematics setup_zgcr.json
  echo "=== launching make_hists_dask.py ==="
  python make_hists_dask.py --year 2024 --setup setup_zgcr.json \
    --data-dir /eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024 \
    --save-templates --save-plotting-pkl --outdir results_zgcr_systs
  echo "=== DASK_EXIT=$? ==="
'
echo "=== WRAPPER_DONE exit=$? END=$(date) ==="
