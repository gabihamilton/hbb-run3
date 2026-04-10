#!/bin/bash
set -e

source /cvmfs/cms.cern.ch/cmsset_default.sh

export MAMBA_ROOT_PREFIX=/uscms_data/d3/gmachado/micromamba
export PATH="${MAMBA_ROOT_PREFIX}/bin:${PATH}"
eval "$(micromamba shell hook --shell bash)"
micromamba activate hbb

cd /uscms_data/d3/gmachado/hbb-run3/fitting

python make_hists.py "$@"
