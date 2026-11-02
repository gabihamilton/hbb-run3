#!/bin/bash
# run_regmassgen_jmsrevt_pipeline.sh
# Zγ-CR chain on the GENERIC regressed mass with EVENT-BY-EVENT JMS/JMR (J. Duarte's gist):
#   m' = m*JMS*(1 + z*sqrt(JMR^2-1)*sigma/m), z~N(0,1) per event, filled in template_utils
#   (setup "jmsr_mode": "event"; 1 sigma = JMS 1% (~0.9 GeV), JMR 10%; sigma = 10 GeV).
#   hist -> collect -> closure vs the Sep 25 generic nominal -> datacards (corr / decorr)
#   -> ccs-joint fits + robust profiled errors + stat-only -> postfit plots.
# New names only (tag 26Oct02_regmassgen_jmsrevt); the Sep 25 generic shelves are parked, not deleted.
#
# Launch detached on LPC, from the repo root:
#   nohup setsid bash fitting/run_regmassgen_jmsrevt_pipeline.sh > fitting/jmsrevt_pipeline.log 2>&1 &
set -uo pipefail

REPO=$HOME/nobackup/hbb-run3
F=$REPO/fitting
TAG=26Oct02_regmassgen_jmsrevt
OLD_TAG=26Sep25_regmassgen_lhe
X2P_TAG=26Sep25_regmass_lhe   # the mumu+gamma card (mll, independent of jet mass) is reused from here
BASE=/eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024
LHE=/eos/uscms/store/group/lpchbbrun3/gmachado/26Sep17_lhephoton_v15/2024
HOUT=results_zgcr_regmassgen_jmsrevt
OLD_ROOT=$F/results_zgcr_regmassgen_lhe/$OLD_TAG/fitting_2024_zgcr_regmassgen.root
NEW_ROOT=$F/$HOUT/$TAG/fitting_2024_zgcr_regmassgen_jmsrevt.root
EOSU=/store/group/lpchbbrun3/$USER
CARDS=$F/results/$TAG/2024/datacards
SUMMARY=$F/RESULTS_regmassgen_jmsrevt.txt
START=${START:-0}   # START=5 reruns only the fits/plots on existing cards
# shapeU nuisances get a [-1,1] range in the workspace by default (the rhalphalib lo/hi is not
# written to the card) -> widen JMS/JMR explicitly, per model
jr() { case $1 in *decorr*) echo "CMS_jms_2024_bb=-10,10:CMS_jms_2024_ccs=-10,10:CMS_jmr_2024_bb=-10,10:CMS_jmr_2024_ccs=-10,10" ;;
                  *) echo "CMS_jms_2024=-10,10:CMS_jmr_2024=-10,10" ;; esac; }
ranges() { echo "--setParameterRanges r_bb=-1,5:r_ccs=-1,5:z_norm=0,5:$(jr $1) --robustFit 1 --cminDefaultMinimizerStrategy 0"; }
export X509_USER_PROXY=$HOME/x509up_u$(id -u)

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
fmt() {
    grep -E "^ *$1 *:" "$2" | head -1 |
        awk -v n="$3" '{split($4, e, "/"); printf "$%s = %.2f^{%+.2f}_{%+.2f}$", n, $3, e[2], e[1]}'
}

if [ "$START" -lt 5 ]; then
echo "Generic-regmass event-by-event JMS/JMR pipeline, tag $TAG, started $(date)" > "$SUMMARY"
voms-proxy-info -file "$X509_USER_PROXY" -exists -valid 2:00 || die "grid proxy missing or < 2h left"

# ---------------------------------------------------------------- 0. park the Sep 25 generic shelves
# collect_hists SUMS every shelf in FITTING_TEMPLATES with the same name -> move (not delete) the old ones.
step "0. move Sep 25 generic shelves aside"
if xrdfs root://cmseos.fnal.gov stat $EOSU/FITTING_TEMPLATES >/dev/null 2>&1; then
    xrdfs root://cmseos.fnal.gov stat $EOSU/FITTING_TEMPLATES_regmassgen_26Sep25 >/dev/null 2>&1 &&
        die "FITTING_TEMPLATES_regmassgen_26Sep25 already exists; not overwriting"
    xrdfs root://cmseos.fnal.gov mv $EOSU/FITTING_TEMPLATES $EOSU/FITTING_TEMPLATES_regmassgen_26Sep25 ||
        die "could not move FITTING_TEMPLATES"
fi

# ---------------------------------------------------------------- 1. hist (container)
step "1. make_hists_dask (container)"
cd $REPO
export COFFEA_IMAGE=coffeateam/coffea-dask-almalinux9:2025.12.0-py3.10
export COFFEA_IMAGE_FULL=$(realpath /cvmfs/unpacked.cern.ch/registry.hub.docker.com/${COFFEA_IMAGE})
export APPTAINER_BINDPATH=/uscmst1b_scratch,/cvmfs,/cvmfs/grid.cern.ch/etc/grid-security:/etc/grid-security,/etc/condor/config.d/01_cmslpc_interactive,/usr/local/bin/cmslpc-local-conf.py:/usr/local/bin/cmslpc-local-conf.py.orig,.cmslpc-local-conf:/usr/local/bin/cmslpc-local-conf.py
mkdir -p /uscmst1b_scratch/lpc1/3DayLifetime/$USER
apptainer exec -B ${PWD}:/srv --pwd /srv "$COFFEA_IMAGE_FULL" /bin/bash -c "
    source /srv/.bashrc && cd fitting && python -u make_hists_dask.py --year 2024 \
      --setup setup_zgcr_gen_jmsrevt.json --data-dir $BASE --overlay-data-dir $LHE \
      --save-templates --outdir $HOUT" || die "make_hists_dask"
NSHELF=$(xrdfs root://cmseos.fnal.gov ls $EOSU/FITTING_TEMPLATES 2>/dev/null | grep -c '\.dat$')
echo "template shelves: $NSHELF" >> "$SUMMARY"
[ "$NSHELF" -ge 70 ] || die "only $NSHELF template shelves - hist step incomplete"

# ---------------------------------------------------------------- 2. collect + closure
step "2. collect_hists"
hbb "python collect_hists.py --year 2024 --tag $TAG --outdir $HOUT --save-templates > collect_jmsrevt.log 2>&1 &&
     cp $HOUT/$TAG/fitting_2024_msd.root $NEW_ROOT" || die "collect_hists"
# JMS=JMR=1 nominal must reproduce the Sep 25 generic nominal exactly; print the up/down effects.
hbb "python - >> $SUMMARY <<'PY'
import uproot, numpy as np
new, old = uproot.open('$NEW_ROOT'), uproot.open('$OLD_ROOT')
c = None
print('closure (nominal vs Sep 25 generic) and JMS/JMR effects:')
for k in ['pass_bb_pt1_Zgammabb', 'pass_cc_pt1_Zgammac', 'pass_cc_pt1_Wgammac', 'fail_pt1_Zgammabb', 'fail_pt1_Wgammac', 'pass_bb_pt1_EGammadata']:
    n = new[f'zgcr_{k}_nominal'].values(); o = old[f'zgcr_{k}_nominal'].values()
    e = new[f'zgcr_{k}_nominal'].axis().edges(); c = 0.5 * (e[1:] + e[:-1])
    mu = lambda h: (h * c).sum() / max(h.sum(), 1e-9)
    sd = lambda h: np.sqrt((h * (c - mu(h)) ** 2).sum() / max(h.sum(), 1e-9))
    line = f'  {k:24s} N {n.sum():9.1f} (old {o.sum():9.1f}, max|diff| {np.abs(n - o).max():.2e})'
    if 'data' not in k:
        for s in ['JMSunconstrainedUp', 'JMSunconstrainedDown', 'JMRunconstrainedUp']:
            v = new[f'zgcr_{k}_{s}'].values()
            line += f' | {s[:3]}{s[-2:] if s.endswith(\"Up\") else \"Dn\"} dmean {mu(v) - mu(n):+.2f} dstd {sd(v) - sd(n):+.2f}'
    print(line)
PY" || die "closure dump"

# ---------------------------------------------------------------- 3. datacards
step "3. make_datacards x2"
for a in zgcr_gen_jmsrevt zgcr_gen_jmsrevt_decorr; do
    hbb "python make_datacards.py --year 2024 --tag $TAG --analysis $a --indir $HOUT/$TAG > datacards_$a.log 2>&1" ||
        die "make_datacards $a"
done

# ---------------------------------------------------------------- 4. mumu+gamma card (jet-mass independent)
step "4. reuse mumu+gamma card"
mkdir -p $F/results_zmmcr_gamma_cards/$TAG/2024/datacards
cp -r $F/results_zmmcr_gamma_cards/$X2P_TAG/2024/datacards/zmmcrModel_2024 \
      $F/results_zmmcr_gamma_cards/$TAG/2024/datacards/ || die "copy mumu card"

fi  # START < 5

# ---------------------------------------------------------------- 5. fits
{ echo; echo "=== [$(date '+%F %T')] JMS/JMR ranges [-10,10]"; echo "=== ccs-joint fits (robust profiled errors). 1 sigma: JMS = 1%, JMR = 10% -> JMS = 1 + 0.01*theta, JMR = 1 + 0.10*theta"; } >> "$SUMMARY"
for m in zgcr_gen_jmsrevt_main zgcr_gen_jmsrevt_decorr_main; do
    step "5. fit $m"
    D=$CARDS/${m}_combined_ccs_mumugamma_2024
    cms "bash run_combined_ccs_fit.sh --year 2024 --tag $TAG --model $m --extra '--setParameterRanges $(jr $m)' > fitlog_$m.txt 2>&1" || die "fit $m"
    cms "cd $D && combine -M MultiDimFit -d workspace_ccs_2024.root --algo singles \
             --redefineSignalPOIs r_bb,r_ccs,z_norm $(ranges $m) -n _robust > robust.log 2>&1" || die "robust $m"
    cms "cd $D && python3 - > pulls.txt <<'PY'
import ROOT
f = ROOT.TFile.Open('fitDiagnostics_ccs_mumugamma_2024.root')
r = f.Get('fit_s')
print(f'  fit_s status {r.status()} covQual {r.covQual()}')
for p in r.floatParsFinal():
    if 'jm' in p.GetName():
        unc = 0.01 if 'jms' in p.GetName() else 0.10
        print(f'  {p.GetName():22s} {p.getVal():+.2f} +- {p.getError():.2f}  ->  {p.GetName()[4:7].upper()} = {1 + unc * p.getVal():.3f} +- {unc * p.getError():.3f}')
PY"
    { echo "--- $m"; grep -E "^ *(r_bb|r_ccs|z_norm) *:" $D/robust.log; cat $D/pulls.txt; } >> "$SUMMARY"
done

step "5b. stat-only breakdown (correlated model)"
D=$CARDS/zgcr_gen_jmsrevt_main_combined_ccs_mumugamma_2024
RANGES=$(ranges zgcr_gen_jmsrevt_main)
cms "cd $D && combine -M MultiDimFit -d workspace_ccs_2024.root --redefineSignalPOIs r_bb,r_ccs,z_norm $RANGES \
         --saveWorkspace -n _snap > /dev/null 2>&1 &&
     combine -M MultiDimFit -d higgsCombine_snap.MultiDimFit.mH120.root --snapshotName MultiDimFit --algo singles \
         --redefineSignalPOIs r_bb,r_ccs,z_norm $RANGES \
         --freezeParameters CMS_jms_2024,CMS_jmr_2024,CMS_lumi_13TeV_2024 -n _stat > stat.log 2>&1" ||
    echo "stat-only breakdown failed (non-fatal)"
{ echo; echo "=== stat-only (JMS, JMR, lumi frozen)"; grep -E "^ *(r_bb|r_ccs|z_norm) *:" $D/stat.log; } >> "$SUMMARY"

# ---------------------------------------------------------------- 6. postfit plots
for m in zgcr_gen_jmsrevt_main zgcr_gen_jmsrevt_decorr_main; do
    step "6. postfit plots $m"
    D=$CARDS/${m}_combined_ccs_mumugamma_2024
    T="$(fmt r_bb $D/robust.log 'r_{bb}')\n$(fmt r_ccs $D/robust.log 'r_{ccs}')\n$(fmt z_norm $D/robust.log 'z_{norm}')"
    hbb "cd $D && combine_postfits -i fitDiagnostics_ccs_mumugamma_2024.root -o plots_zgcr --fit all --data --unblind \
            --sigs zgammabb,zgammacc --rmap zgammabb:r_bb,zgammacc:r_ccs --bkgs wgammacs,qcd,other --onto qcd \
            --cats 'passbb:ptbin0zgcrpassbb2024;passcc:ptbin0zgcrpasscc2024;fail:ptbin0zgcrfail2024' \
            --xlabel 'Jet regressed mass (generic) [GeV]' --year 2024 --lumi 109 --dpi 100 --noroot \
            --poi-text '$T' > pf.log 2>&1" || echo "postfit plots $m failed (non-fatal)"
done

step "DONE"
echo "finished $(date)" >> "$SUMMARY"
cat "$SUMMARY"
