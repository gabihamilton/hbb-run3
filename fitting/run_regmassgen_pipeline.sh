#!/bin/bash
# run_regmassgen_pipeline.sh
# Unattended Zγ-CR chain on the GENERIC GloParT regressed mass (FatJet0_ParTmassGeneric):
#   hist (coffea-dask container) -> collect -> datacards (nominal / JMSR / JMSR-decorr)
#   -> ccs-joint fits + robust profiled errors + stat-only -> postfit plots -> stitch plots.
# Everything is written under new names (tag 26Sep25_regmassgen_lhe, setups setup_zgcr_gen*),
# so the ParTmassX2p results (26Sep25_regmass_lhe) are left untouched.
#
# Launch detached on LPC (survives logout), from the repo root:
#   nohup setsid bash fitting/run_regmassgen_pipeline.sh > fitting/regmassgen_pipeline.log 2>&1 &
set -uo pipefail

REPO=$HOME/nobackup/hbb-run3
F=$REPO/fitting
TAG=26Sep25_regmassgen_lhe
X2P_TAG=26Sep25_regmass_lhe   # the mumu+gamma card (mll, independent of jet mass) is reused from here
BASE=/eos/uscms/store/group/lpchbbrun3/gmachado/24Aug07_v15/2024
LHE=/eos/uscms/store/group/lpchbbrun3/gmachado/26Sep17_lhephoton_v15/2024
HOUT=results_zgcr_regmassgen_lhe
EOSU=/store/group/lpchbbrun3/$USER
CARDS=$F/results/$TAG/2024/datacards
SUMMARY=$F/RESULTS_regmassgen.txt
RANGES="--setParameterRanges r_bb=-1,5:r_ccs=-1,5:z_norm=0,5 --robustFit 1 --cminDefaultMinimizerStrategy 0"
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
# "r_bb :  +0.823  -0.256/+0.350 (68%)" -> "$r_{bb} = 0.82^{+0.35}_{-0.26}$"
fmt() {
    grep -E "^ *$1 *:" "$2" | head -1 |
        awk -v n="$3" '{split($4, e, "/"); printf "$%s = %.2f^{%+.2f}_{%+.2f}$", n, $3, e[2], e[1]}'
}

echo "Generic-regmass pipeline, tag $TAG, started $(date)" > "$SUMMARY"
voms-proxy-info -file "$X509_USER_PROXY" -exists -valid 2:00 || die "grid proxy missing or < 2h left"

# ---------------------------------------------------------------- 0. park the X2p shelves
# collect_hists SUMS every shelf in FITTING_TEMPLATES with the same histogram name, so the X2p
# shelves must not sit next to the new ones. Moved (not deleted).
step "0. move X2p shelves aside"
if xrdfs root://cmseos.fnal.gov stat $EOSU/FITTING_TEMPLATES >/dev/null 2>&1; then
    xrdfs root://cmseos.fnal.gov stat $EOSU/FITTING_TEMPLATES_regmassX2p_26Sep25 >/dev/null 2>&1 &&
        die "FITTING_TEMPLATES_regmassX2p_26Sep25 already exists; not overwriting"
    xrdfs root://cmseos.fnal.gov mv $EOSU/FITTING_TEMPLATES $EOSU/FITTING_TEMPLATES_regmassX2p_26Sep25 ||
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
      --setup setup_zgcr_gen.json --data-dir $BASE --overlay-data-dir $LHE \
      --save-templates --outdir $HOUT" || die "make_hists_dask"
NSHELF=$(xrdfs root://cmseos.fnal.gov ls $EOSU/FITTING_TEMPLATES 2>/dev/null | grep -c '\.dat$')
echo "template shelves written: $NSHELF (X2p run had 75)"
echo "template shelves: $NSHELF (X2p run: 75)" >> "$SUMMARY"
[ "$NSHELF" -ge 70 ] || die "only $NSHELF template shelves - hist step incomplete"

# ---------------------------------------------------------------- 2. collect + sanity yields
step "2. collect_hists"
hbb "python collect_hists.py --year 2024 --tag $TAG --outdir $HOUT --save-templates > collect_regmassgen.log 2>&1 &&
     cp $HOUT/$TAG/fitting_2024_msd.root $HOUT/$TAG/fitting_2024_zgcr_regmassgen.root" || die "collect_hists"
hbb "python - >> $SUMMARY <<'PY'
import uproot
f = uproot.open('$HOUT/$TAG/fitting_2024_zgcr_regmassgen.root')
e = f['zgcr_pass_bb_pt1_Zgammabb_nominal'].axis().edges()
print(f'observable axis: {e[0]:.0f}-{e[-1]:.0f} GeV, {len(e)-1} bins')
for k in ['pass_bb_pt1_Zgammabb', 'pass_bb_pt1_EGammadata', 'pass_cc_pt1_Zgammac', 'pass_cc_pt1_Wgammac',
          'pass_cc_pt1_EGammadata', 'fail_pt1_Zgammabb', 'fail_pt1_EGammadata']:
    print(f'  {k:28s} {f[\"zgcr_\" + k + \"_nominal\"].values().sum():12.1f}')
PY" || die "yield dump"

# ---------------------------------------------------------------- 3. datacards
step "3. make_datacards x3"
for a in zgcr_gen zgcr_gen_jmsr zgcr_gen_jmsr_decorr; do
    hbb "python make_datacards.py --year 2024 --tag $TAG --analysis $a --indir $HOUT/$TAG > datacards_$a.log 2>&1" ||
        die "make_datacards $a"
done

# ---------------------------------------------------------------- 4. mumu+gamma card (jet-mass independent)
step "4. reuse mumu+gamma card"
mkdir -p $F/results_zmmcr_gamma_cards/$TAG/2024/datacards
cp -r $F/results_zmmcr_gamma_cards/$X2P_TAG/2024/datacards/zmmcrModel_2024 \
      $F/results_zmmcr_gamma_cards/$TAG/2024/datacards/ || die "copy mumu card"

# ---------------------------------------------------------------- 5. fits
{ echo; echo "=== ccs-joint fits (robust profiled errors: MultiDimFit singles, robustFit, tight ranges)"; } >> "$SUMMARY"
for m in zgcr_gen_main zgcr_gen_jmsr_main zgcr_gen_jmsr_decorr_main; do
    step "5. fit $m"
    D=$CARDS/${m}_combined_ccs_mumugamma_2024
    cms "bash run_combined_ccs_fit.sh --year 2024 --tag $TAG --model $m > fitlog_$m.txt 2>&1" || die "fit $m"
    cms "cd $D && combine -M MultiDimFit -d workspace_ccs_2024.root --algo singles \
             --redefineSignalPOIs r_bb,r_ccs,z_norm $RANGES -n _robust > robust.log 2>&1" || die "robust $m"
    cms "cd $D && python3 - > pulls.txt <<'PY'
import ROOT
f = ROOT.TFile.Open('fitDiagnostics_ccs_mumugamma_2024.root')
r = f.Get('fit_s')
print(f'  fit_s status {r.status()} covQual {r.covQual()}')
for p in r.floatParsFinal():
    if 'jm' in p.GetName():
        print(f'  {p.GetName():22s} {p.getVal():+.2f} +- {p.getError():.2f}')
PY"
    { echo "--- $m"; grep -E "^ *(r_bb|r_ccs|z_norm) *:" $D/robust.log; cat $D/pulls.txt; } >> "$SUMMARY"
done

step "5b. stat-only breakdown (JMSR model)"
D=$CARDS/zgcr_gen_jmsr_main_combined_ccs_mumugamma_2024
cms "cd $D && combine -M MultiDimFit -d workspace_ccs_2024.root --redefineSignalPOIs r_bb,r_ccs,z_norm $RANGES \
         --saveWorkspace -n _snap > /dev/null 2>&1 &&
     combine -M MultiDimFit -d higgsCombine_snap.MultiDimFit.mH120.root --snapshotName MultiDimFit --algo singles \
         --redefineSignalPOIs r_bb,r_ccs,z_norm $RANGES \
         --freezeParameters CMS_jms_2024,CMS_jmr_2024,CMS_lumi_13TeV_2024 -n _stat > stat.log 2>&1 &&
     combine -M MultiDimFit -d higgsCombine_snap.MultiDimFit.mH120.root --snapshotName MultiDimFit --algo singles \
         --redefineSignalPOIs r_bb,r_ccs $RANGES \
         --freezeParameters CMS_jms_2024,CMS_jmr_2024,CMS_lumi_13TeV_2024,z_norm -n _statz > statz.log 2>&1" ||
    echo "stat-only breakdown failed (non-fatal)"
{ echo; echo "=== JMSR stat-only (JMS, JMR, lumi frozen)"; grep -E "^ *(r_bb|r_ccs|z_norm) *:" $D/stat.log
  echo "=== JMSR stat-only + z_norm frozen"; grep -E "^ *(r_bb|r_ccs) *:" $D/statz.log; } >> "$SUMMARY"

# ---------------------------------------------------------------- 6. postfit plots
for m in zgcr_gen_main zgcr_gen_jmsr_main; do
    step "6. postfit plots $m"
    D=$CARDS/${m}_combined_ccs_mumugamma_2024
    T="$(fmt r_bb $D/robust.log 'r_{bb}')\n$(fmt r_ccs $D/robust.log 'r_{ccs}')\n$(fmt z_norm $D/robust.log 'z_{norm}')"
    Z="$(fmt z_norm $D/robust.log 'z_{norm}')"
    hbb "cd $D && combine_postfits -i fitDiagnostics_ccs_mumugamma_2024.root -o plots_zgcr --fit all --data --unblind \
            --sigs zgammabb,zgammacc --rmap zgammabb:r_bb,zgammacc:r_ccs --bkgs wgammacs,qcd,other --onto qcd \
            --cats 'passbb:ptbin0zgcrpassbb2024;passcc:ptbin0zgcrpasscc2024;fail:ptbin0zgcrfail2024' \
            --xlabel 'Jet regressed mass (generic) [GeV]' --year 2024 --lumi 109 --dpi 100 --noroot \
            --poi-text '$T' > pf.log 2>&1 &&
         combine_postfits -i fitDiagnostics_ccs_mumugamma_2024.root -o plots_zmm --fit all --data --unblind \
            --sigs zmm --rmap zmm:z_norm --bkgs ttbar,other --cats 'zmm:ptbin0zmmcrinclusive2024' \
            --xlabel '\$m_{\\mu\\mu}\$ [GeV]' --year 2024 --lumi 109 --dpi 100 --noroot \
            --poi-text '$Z' > pfz.log 2>&1" || echo "postfit plots $m failed (non-fatal)"
done

# ---------------------------------------------------------------- 7. stitching plots on the generic mass
step "7. stitch plots (generic mass)"
hbb "python make_stitch_lhe.py --year 2024 --data-dir $BASE --lhe-data-dir $LHE --families ZGto2QG,WGto2QG \
        --mass-col FatJet0_ParTmassGeneric --outdir plots/stitch_lhe_generic/ > stitch_generic.log 2>&1" ||
    echo "stitch plots failed (non-fatal)"

step "DONE"
echo "finished $(date)" >> "$SUMMARY"
cat "$SUMMARY"
