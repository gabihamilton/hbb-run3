from __future__ import annotations

import argparse
import json
import pickle  # You ARE using this for dumping the model
import warnings
from pathlib import Path

import numpy as np

# import pandas as pd  <-- Remove this if you aren't using dataframes
import rhalphalib as rl
import ROOT

from hbb.common_vars import LUMI

# Suppress annoying ROOT warnings
ROOT.gROOT.SetBatch(True)
warnings.filterwarnings("ignore")

rl.util.install_roofit_helpers()

eps = 0.001

### DIFF: Disabled systematics and Muon CR for the initial Z-Gamma validation fit.
### Original had these set to True.
do_systematics = False
do_muon_CR = False

lumi_err = {"2022": 1.01, "2023": 1.02}


def badtemp_ma(hvalues, mask=None):
    # Need minimum size & more than 1 non-zero bins
    tot = np.sum(hvalues[mask])
    count_nonzeros = np.sum(hvalues[mask] > 0)
    return bool(tot < eps or count_nonzeros < 2)


def get_template(year, tag, sName, region, ptbin, cat, obs, syst):
    """
    Read msd template from root file
    """
    ### DIFF: Updated naming convention to match 'make_zg_hists.py' output.
    ### Original script had complex logic for 'ggf_', 'vbf_', etc.
    ### We now use a standardized format: {cat}_{region}_pt{bin}_{process}_{syst}

    f = ROOT.TFile.Open(f"results/{tag}/{year}/signalregion.root")

    reg_clean = region.rstrip("_")
    name = f"{cat}_{reg_clean}_pt{ptbin}_{sName}_{syst}"

    h = f.Get(name)

    ### DIFF: Added safety check.
    ### Original script crashes if histogram is missing. This returns a dummy empty hist.
    if not h:
        return (
            np.zeros(len(obs.binning) - 1),
            obs.binning,
            obs.name,
            np.zeros(len(obs.binning) - 1),
        )

    sumw = []
    sumw2 = []

    for i in range(1, h.GetNbinsX() + 1):
        if h.GetBinContent(i) < 0:
            sumw += [0]
            sumw2 += [0]
        else:
            sumw += [h.GetBinContent(i)]
            sumw2 += [h.GetBinError(i) * h.GetBinError(i)]

    return (np.array(sumw), obs.binning, obs.name, np.array(sumw2))


def zgamma_rhalphabet(args):
    """
    Create the data cards for Z-Gamma!
    """

    year = args.year
    tag = args.tag

    print("Running Z-Gamma Card Maker for " + year)

    datacard_dir = Path(f"results/{tag}/{year}/datacards/")
    initvals_dir = Path(f"results/{tag}/{year}/initial_vals/")

    if not datacard_dir.exists():
        datacard_dir.mkdir(parents=True, exist_ok=True)

    if not initvals_dir.exists():
        initvals_dir.mkdir(parents=True, exist_ok=True)

    ### DIFF: Loading 'setup_zgamma.json' instead of 'setup.json'
    ### This ensures we use the low-pT binning (200 GeV) instead of Higgs binning.
    with Path("setup_zgamma.json").open() as f:
        setup = json.load(f)
        cats_cfg = setup["categories"]

    total_model_bins = []

    sys_lumi_uncor = rl.NuisanceParameter(f"CMS_lumi_13TeV_{year[:4]}", "lnN")

    validbins = {}

    msd_cfg = setup["observable"]
    msdbins = np.linspace(msd_cfg["min"], msd_cfg["max"], msd_cfg["nbins"] + 1)
    msd = rl.Observable(msd_cfg["name"], msdbins)

    ### DIFF: Only running the Z-Gamma Control Region category ('zgcr')
    ### Original ran 'ggf', 'vh', 'vbf'.
    cats = ["zgcr"]

    # --- QCD ESTIMATION (TF) ---
    tf_params = {}
    for cat in cats:

        ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        # Derive 2D array
        ptpts, msdpts = np.meshgrid(
            ptbins[:-1] + 0.3 * np.diff(ptbins),
            msdbins[:-1] + 0.5 * np.diff(msdbins),
            indexing="ij",
        )
        rhopts = 2 * np.log(msdpts / ptpts)

        ### DIFF: Changed pT scaling geometry.
        ### Original scaled from 450-1200 (Higgs). We scale 200-1200 (Z-Boson).
        ptscaled = (ptpts - 200.0) / (1200.0 - 200.0)
        rhoscaled = (rhopts - (-6.0)) / ((-2.1) - (-6.0))

        validbins[cat] = (rhoscaled >= 0.0) & (rhoscaled <= 1.0)
        rhoscaled[~validbins[cat]] = 1

        tf_params[cat] = {}
        fitfailed_qcd = {}

        # We model pass_bb and pass_cc vs fail
        for reg in ["bb", "cc"]:
            fitfailed_qcd[reg] = 0

            # Simple retry loop
            while fitfailed_qcd[reg] < 2:

                qcdmodel = rl.Model(f"qcdmodel_{cat}_{reg}")
                qcdpass, qcdfail = 0.0, 0.0

                for ptbin in range(npt):
                    failCh = rl.Channel(f"ptbin{ptbin}{cat}fail{year}{reg}")
                    passCh = rl.Channel(f"ptbin{ptbin}{cat}pass{year}{reg}")
                    qcdmodel.addChannel(failCh)
                    qcdmodel.addChannel(passCh)

                    # Using QCD MC for the Transfer Factor
                    failTempl = get_template(
                        year,
                        tag,
                        "QCD",
                        "fail_",
                        ptbin + 1,
                        cat,
                        obs=msd,
                        syst="nominal",
                    )
                    passTempl = get_template(
                        year,
                        tag,
                        "QCD",
                        f"pass_{reg}_",
                        ptbin + 1,
                        cat,
                        obs=msd,
                        syst="nominal",
                    )

                    failCh.setObservation(failTempl, read_sumw2=True)
                    passCh.setObservation(passTempl, read_sumw2=True)

                    qcdfail += failCh.getObservation()[0].sum()
                    qcdpass += passCh.getObservation()[0].sum()

                qcdeff = qcdpass / qcdfail
                print(f"Inclusive P/F ({reg}) from Monte Carlo = " + str(qcdeff))

                # INITIAL VALUES
                initF = (
                    Path(f"results/{tag}/{year}/initial_vals") / f"initial_vals_{cat}_{reg}.json"
                )

                ### DIFF: Added fallback for missing initial values.
                ### Original assumes file exists. We default to [1,1] so the fit can bootstrap itself.
                if initF.exists():
                    with initF.open() as f:
                        initial_vals = np.array(json.load(f)["initial_vals"])
                else:
                    print(f"No initial vals found for {reg}, using default Order 1 poly.")
                    initial_vals = np.array([[1.0, 1.0], [1.0, 1.0]])

                print(
                    "TFpf order "
                    + str(initial_vals.shape[0] - 1)
                    + " in pT, "
                    + str(initial_vals.shape[1] - 1)
                    + " in rho"
                )

                tf_MCtempl = rl.BasisPoly(
                    "tf_MCtempl_" + cat + reg + year,
                    (initial_vals.shape[0] - 1, initial_vals.shape[1] - 1),
                    ["pt", "rho"],
                    basis="Bernstein",
                    init_params=initial_vals,
                    limits=(0, 10),
                    coefficient_transform=None,
                )

                tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)

                for ptbin in range(npt):
                    failCh = qcdmodel[f"ptbin{ptbin}{cat}fail{year}{reg}"]
                    passCh = qcdmodel[f"ptbin{ptbin}{cat}pass{year}{reg}"]
                    failObs = failCh.getObservation()[0]

                    qcdparams = np.array(
                        [
                            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}{cat}{year}{reg}_{i}", 0)
                            for i in range(msd.nbins)
                        ]
                    )
                    sigmascale = 10.0
                    scaledparams = (
                        failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
                    )

                    fail_qcd = rl.ParametericSample(
                        f"ptbin{ptbin}{cat}fail{year}{reg}_qcd",
                        rl.Sample.BACKGROUND,
                        msd,
                        scaledparams,
                    )
                    failCh.addSample(fail_qcd)
                    pass_qcd = rl.TransferFactorSample(
                        f"ptbin{ptbin}{cat}pass{year}{reg}_qcd",
                        rl.Sample.BACKGROUND,
                        tf_MCtempl_params[ptbin, :],
                        fail_qcd,
                    )
                    passCh.addSample(pass_qcd)

                    failCh.mask = validbins[cat][ptbin]
                    passCh.mask = validbins[cat][ptbin]

                # Run the Fit
                qcdfit_ws = ROOT.RooWorkspace("w")
                simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
                qcdfit = simpdf.fitTo(
                    obs,
                    ROOT.RooFit.Extended(True),
                    ROOT.RooFit.SumW2Error(True),
                    ROOT.RooFit.Strategy(2),
                    ROOT.RooFit.Save(),
                    ROOT.RooFit.Minimizer("Minuit2", "migrad"),
                    ROOT.RooFit.PrintLevel(-1),
                )
                qcdfit_ws.add(qcdfit)

                # Check status
                if qcdfit.status() != 0:
                    fitfailed_qcd[reg] += 1
                    print(f"Fit failed for {reg}, retrying...")
                else:
                    # Save results if successful
                    allparams = dict(zip(qcdfit.nameArray(), qcdfit.valueArray()))
                    pvalues = []
                    for _, p in enumerate(tf_MCtempl.parameters.reshape(-1)):
                        p.value = allparams[p.name]
                        pvalues += [p.value]
                    new_values = np.array(pvalues).reshape(tf_MCtempl.parameters.shape)
                    with initF.open("w") as outfile:
                        json.dump({"initial_vals": new_values.tolist()}, outfile)
                    break

            print("Fitted QCD for category " + cat + " region " + reg)

            param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
            decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(
                tf_MCtempl.name + "_deco", qcdfit, param_names
            )
            tf_MCtempl.parameters = decoVector.correlated_params.reshape(
                tf_MCtempl.parameters.shape
            )

            # Data Residual (Blinded for now / identity)
            tf_dataResidual = rl.BasisPoly(
                "tf_dataResidual_" + year + cat + reg,
                (0, 0),
                ["pt", "rho"],
                basis="Bernstein",
                init_params=np.array([[1]]),
                limits=(0, 20),
                coefficient_transform=None,
            )

            tf_params[cat][reg] = (
                qcdeff * tf_MCtempl(ptscaled, rhoscaled) * tf_dataResidual(ptscaled, rhoscaled)
            )

    # --- BUILD ACTUAL MODEL ---
    model = rl.Model("zgammaModel_" + year)

    ### DIFF: Changed Samples and Signal Definition.
    ### Original used ggF, VBF, WH, ZH.
    ### We use Zgammabb as Signal, and Zgamma/GJets/TTGamma as Backgrounds.
    samps = ["Zgammabb", "Zgamma", "Wjets", "Zjets", "GJets", "TTGamma", "QCD"]
    sigs = ["Zgammabb"]

    for cat in cats:
        ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        for ptbin in range(npt):
            for region in ["pass_bb_", "pass_cc_", "fail_"]:

                ch_name = f"ptbin{ptbin}{cat}{region.replace('_', '')}{year}"
                total_model_bins.append(ch_name)

                ch = rl.Channel(ch_name)
                model.addChannel(ch)

                # Add MC Samples
                for sName in samps:

                    # Skip QCD in the main loop (handled via data-driven later)
                    if sName == "QCD":
                        continue

                    templ = get_template(
                        year,
                        tag,
                        sName,
                        region,
                        ptbin + 1,
                        cat,
                        obs=msd,
                        syst="nominal",
                    )
                    nominal = templ[0]

                    # Skip empty samples
                    if badtemp_ma(nominal):
                        # print("Sample {} is too small, skipping".format(ch.name + '_' + sName))
                        continue

                    stype = rl.Sample.SIGNAL if sName in sigs else rl.Sample.BACKGROUND

                    sample = rl.TemplateSample(ch.name + "_" + sName, stype, templ)

                    # Simple Lumi Uncertainty
                    sample.setParamEffect(
                        sys_lumi_uncor,
                        lumi_err[year[:4]] ** (LUMI[year[:4]] / LUMI["2022-2023"]),
                    )

                    if do_systematics:
                        sample.autoMCStats(lnN=True)

                    ch.addSample(sample)

                ### DIFF: Using 'Jetdata' for observation.
                ### Z-Gamma triggers might eventually require EGamma data, but keeping Jetdata structure for now.
                data_obs = get_template(
                    year, tag, "Jetdata", region, ptbin + 1, cat, obs=msd, syst="nominal"
                )
                ch.setObservation(data_obs[0:3])

    # --- ADD DATA-DRIVEN QCD ---
    for cat in cats:
        ptbins = np.array(cats_cfg[cat]["bins"])
        npt = len(ptbins) - 1

        for ptbin in range(npt):

            failCh = model[f"ptbin{ptbin}{cat}fail{year}"]
            passChbb = model[f"ptbin{ptbin}{cat}passbb{year}"]
            passChcc = model[f"ptbin{ptbin}{cat}passcc{year}"]

            qcdparams = np.array(
                [
                    rl.IndependentParameter(f"qcdparam_ptbin{ptbin}{cat}{year}_{i}", 0)
                    for i in range(msd.nbins)
                ]
            )
            initial_qcd = failCh.getObservation().astype(float)

            # Subtract other backgrounds from Data in Fail Region
            for sample in failCh:
                initial_qcd -= sample.getExpectation(nominal=True)

            if np.any(initial_qcd < 0.0):
                initial_qcd[np.where(initial_qcd < 0)] = 0
                print("Warning: negative QCD estimate in some bins")

            sigmascale = 10
            scaledparams = (
                initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcdparams
            )

            fail_qcd = rl.ParametericSample(
                name=f"ptbin{ptbin}{cat}fail{year}_qcd",
                sampletype=rl.Sample.BACKGROUND,
                observable=msd,
                params=scaledparams,
            )
            failCh.addSample(fail_qcd)

            pass_qcdbb = rl.TransferFactorSample(
                name=f"ptbin{ptbin}{cat}passbb{year}_qcd",
                sampletype=rl.Sample.BACKGROUND,
                transferfactor=tf_params[cat]["bb"][ptbin, :],
                dependentsample=fail_qcd,
                observable=msd,
            )
            passChbb.addSample(pass_qcdbb)

            pass_qcdcc = rl.TransferFactorSample(
                name=f"ptbin{ptbin}{cat}passcc{year}_qcd",
                sampletype=rl.Sample.BACKGROUND,
                transferfactor=tf_params[cat]["cc"][ptbin, :],
                dependentsample=fail_qcd,
                observable=msd,
            )
            passChcc.addSample(pass_qcdcc)

            mask = validbins[cat][ptbin]
            failCh.mask = mask
            passChcc.mask = mask
            passChbb.mask = mask

    # --- SAVE OUTPUT ---
    with (datacard_dir / f"zgammaModel_{year}.pkl").open("wb") as fout:
        pickle.dump(model, fout)

    modeldir = datacard_dir / f"zgammaModel_{year}"
    model.renderCombine(modeldir)

    out_cards = ""
    for card in total_model_bins:
        out_cards += f"{card}={card}.txt "

    ### DIFF: Physics Model Configuration
    ### We map the 'Zgammabb' sample to the signal strength 'r'.
    t2w_cfg = "-P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel --PO verbose --PO 'map=.*/Zgammabb:r[1,-20,20]'"

    build_sh = modeldir / "build.sh"
    with build_sh.open("w") as f:
        f.write(f"combineCards.py {out_cards} > model_combined.txt\n")
        f.write(f"text2workspace.py {t2w_cfg} model_combined.txt -o workspace.root")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--year", help="year", type=str, required=True)
    parser.add_argument("--tag", help="tag", type=str, required=True)

    args = parser.parse_args()

    zgamma_rhalphabet(args)
