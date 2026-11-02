#!/bin/bash
# run_ftest_gen_pipeline.sh
# F-test of the rhalphabet TF orders on the GENERIC regressed mass (event-by-event JMS/JMR setup),
# zgcr-only cards (passbb + passcc + fail). Templates from run_regmassgen_jmsrevt_pipeline.sh.
# Only rho orders are scanned: there is a single pt bin (250-1200), so pt orders are degenerate.
# Current orders: MC pt0/rho1, residual pt0/rho0 (bb and cc). Each comparison varies ONE region's order.
#
# Launch detached on LPC, from the repo root:
#   nohup setsid bash fitting/run_ftest_gen_pipeline.sh > fitting/ftest_gen.log 2>&1 &
set -uo pipefail

REPO=$HOME/nobackup/hbb-run3
F=$REPO/fitting
TAG=26Oct02_regmassgen_jmsrevt
HOUT=results_zgcr_regmassgen_jmsrevt
BASE_SETUP=setup_zgcr_gen_jmsrevt.json
OUT=results_ftest
CARDS=$F/$OUT/$TAG/2024/datacards
WORK=$F/$OUT/$TAG/2024/ftest
NTOYS=500
NBINS=60    # 3 channels x 20 mass bins
SUMMARY=$F/RESULTS_ftest_gen.txt

step() { echo; echo "|$(date '+%F %T')| ===== $*"; }
die() { echo "|$(date '+%F %T')| FATAL: $*"; echo "FAILED at: $*" >> "$SUMMARY"; exit 1; }
hbb() {
    bash -c "export MAMBA_ROOT_PREFIX=\$HOME/nobackup/micromamba
             eval \"\$(\$HOME/nobackup/.local/bin/micromamba shell hook -s bash)\"
             micromamba activate hbb && cd $F && $1"
}
cms() {
    bash -c "source /cvmfs/cms.cern.ch/cmsset_default.sh
             cd \$HOME/nobackup/CMSSW_14_1_0_pre4/src && eval \$(scramv1 runtime -sh)
             cd $F && $1"
}

mkdir -p $F/$OUT
echo "F-test (generic regmass, event JMS/JMR), tag $TAG, $NTOYS toys, started $(date)" > "$SUMMARY"
[ -f $F/$HOUT/$TAG/fitting_2024_zgcr_regmassgen_jmsrevt.root ] || die "templates missing - run the jmsrevt pipeline first"

# variant name -> "mc_bb mc_cc res_bb res_cc" rho orders (pt orders stay 0)
declare -A V=(
    [base]="1 1 0 0"
    [resbb1]="1 1 1 0"
    [rescc1]="1 1 0 1"
    [mcbb2]="2 1 0 0"
    [mccc2]="1 2 0 0"
    [mcbb0]="0 1 0 0"
    [mccc0]="1 0 0 0"
)
# comparison -> "null alt"
declare -A C=(
    [res_bb_rho0_vs_rho1]="base resbb1"
    [res_cc_rho0_vs_rho1]="base rescc1"
    [mc_bb_rho1_vs_rho2]="base mcbb2"
    [mc_cc_rho1_vs_rho2]="base mccc2"
    [mc_bb_rho0_vs_rho1]="mcbb0 base"
    [mc_cc_rho0_vs_rho1]="mccc0 base"
)
npar() { set -- ${V[$1]}; echo $(( $1 + $2 + $3 + $4 + 4 )); }   # sum of (rho+1) over the 4 TFs

# ---------------------------------------------------------------- 1. datacards + workspaces per variant
for v in "${!V[@]}"; do
    step "1. datacards ftg_$v (${V[$v]})"
    read mbb mcc rbb rcc <<< "${V[$v]}"
    hbb "python - <<PY
import json
d = json.load(open('$BASE_SETUP'))
c = d['categories']['zgcr']
c['tfmc_order'] = {'bb': {'pt': 0, 'rho': $mbb}, 'cc': {'pt': 0, 'rho': $mcc}}
c['tfres_order'] = {'bb': {'pt': 0, 'rho': $rbb}, 'cc': {'pt': 0, 'rho': $rcc}}
json.dump(d, open('setup_ftg_$v.json', 'w'), indent=4)
PY
python make_datacards.py --year 2024 --tag $TAG --analysis ftg_$v --indir $HOUT/$TAG --outdir $OUT > $OUT/datacards_ftg_$v.log 2>&1" ||
        die "make_datacards ftg_$v"
    cms "cd $CARDS/ftg_${v}_mainModel_2024 && bash build.sh > build.log 2>&1" || die "build ftg_$v"
done
rm -f $F/setup_ftg_*.json

# ---------------------------------------------------------------- 2. F-tests (parallel)
step "2. F-tests ($NTOYS toys each, in parallel)"
for c in "${!C[@]}"; do
    read null alt <<< "${C[$c]}"
    mkdir -p $WORK/$c
    cms "cd $WORK/$c && python3 $F/run_ftest.py --null $CARDS/ftg_${null}_mainModel_2024/workspace.root \
             --alt $CARDS/ftg_${alt}_mainModel_2024/workspace.root --ntoys $NTOYS --tag $c \
             --extra '--setParameterRanges CMS_jms_2024=-10,10:CMS_jmr_2024=-10,10' > run_ftest.log 2>&1" &
done
wait

# ---------------------------------------------------------------- 3. F-stat plots + summary
step "3. plot + summary"
{ echo; printf "  %-24s %-8s %-8s %8s %8s  %s\n" comparison null alt p_null p_alt "F_obs / valid toys / p-value"; } >> "$SUMMARY"
for c in res_bb_rho0_vs_rho1 res_cc_rho0_vs_rho1 mc_bb_rho1_vs_rho2 mc_cc_rho1_vs_rho2 mc_bb_rho0_vs_rho1 mc_cc_rho0_vs_rho1; do
    read null alt <<< "${C[$c]}"
    p1=$(npar $null); p2=$(npar $alt)
    hbb "cd $WORK/$c && python $F/plot_ftest.py --tag $c --p1 $p1 --p2 $p2 --nbins $NBINS > plot.log 2>&1"
    res=$(grep -E "Observed F-Stat|Valid Toys|P-Value" $WORK/$c/plot.log | awk -F: '{printf "%s /", $2}')
    printf "  %-24s %-8s %-8s %8s %8s %s\n" $c $null $alt $p1 $p2 "$res" >> "$SUMMARY"
done
echo "  (p < 0.05 -> the alt (higher) order is needed)" >> "$SUMMARY"

step "DONE"
echo "finished $(date)" >> "$SUMMARY"
cat "$SUMMARY"
