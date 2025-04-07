import pickle
import numpy as np
import rhalphalib as rl
import scipy.stats
import ROOT
import os
import sys
from pathlib import Path
import ROOT


# Initialize rhalphalib
rl.util.install_roofit_helpers()
rl.ParametericSample.PreferRooParametricHist = False

# Define output directory
output_dir = Path("rhalphabet_datacards")
output_dir.mkdir(exist_ok=True)

# Function to load and structure templates like get_templates
def get_templates(years):
    templates_summed = {}

    for year in years:
        # Load histograms
        histogram_file = f"histograms_{year}.pkl"
        with open(histogram_file, "rb") as f:
            histograms = pickle.load(f)

        for process, hist_dict in histograms.items():
            # Ensure process is initialized in templates_summed
            if process not in templates_summed:
                templates_summed[process] = {"pass": {}, "fail": {}}  # Initialize regions

            for region in ["pass", "fail"]:
                if region not in hist_dict:
                    print(f"Warning: '{region}' missing in {process} for {year}. Skipping.")
                    continue  # Skip if region is missing

                for ptbin, hist in hist_dict[region].items():
                    #if ptbin not in templates_summed[process][region]:
                    #    templates_summed[process][region][ptbin] = hist.copy()  # Initialize
                    #else:
                    #    templates_summed[process][region][ptbin] += hist  # Sum across years

                    if ptbin not in templates_summed[process][region]:
                        h = hist.copy()
                    else:
                        h = templates_summed[process][region][ptbin] + hist

                    # Patch negative bins to zero
                    np_array = np.asarray(h)
                    np_array[np_array < 0] = 0
                    templates_summed[process][region][ptbin] = h

    return templates_summed

def get_hist(process, region, ptbin, histograms):
    """
    Retrieve the histogram for a given process, region, and ptbin.

    Args:
        process (str): Name of the process (e.g., "QCD", "Wto2Q", etc.).
        region (str): Either "pass" or "fail".
        ptbin (int): The pt bin key to retrieve from the histogram.
        histograms (dict): Dictionary containing all histograms.

    Returns:
        tuple: (sumw, binning, obs_name, sumw2) where:
            - sumw: The histogram bin values.
            - binning: The bin edges.
            - obs_name: The observable name.
            - sumw2: The bin errors squared (if available).
    """
    if process not in histograms:
        raise KeyError(f"Process '{process}' not found in histograms.")

    if region not in histograms[process]:
        raise KeyError(f"Region '{region}' not found in histograms for process '{process}'.")

    if ptbin not in histograms[process][region]:
        raise KeyError(f"Ptbin '{ptbin}' not found in histograms for process '{process}', region '{region}'.")

    hist = histograms[process][region][ptbin]


    binning = hist.axes[0].edges  # Get bin edges
    obs_name = hist.axes[0].name  # Observable name (e.g., "msd")
    sumw2 = np.zeros_like(hist)

    return (hist.values(), hist.axes[0].edges, hist.axes[0].name, hist.variances())



def model_rhalphabet():

    ######## SETTING IT UP ########
    years = ["2023"]
    # Extract structured templates
    histograms = get_templates(years)

    jec = rl.NuisanceParameter("CMS_jec", "lnN")
    massScale = rl.NuisanceParameter("CMS_msdScale", "shape")
    lumi = rl.NuisanceParameter("CMS_lumi", "lnN")
    tqqeffSF = rl.IndependentParameter("tqqeffSF", 1.0, 0, 10)
    tqqnormSF = rl.IndependentParameter("tqqnormSF", 1.0, 0, 10)

    # Define the pt bins
    ptbins = np.array([450, 500, 550, 600, 675, 800, 1200]) #  500, 550, 600, 675, 800,
    npt = len(ptbins) - 1

    # Extract binning information
    #msdbins = np.linspace(40, 201, 25)

    msd_axis = histograms["QCD"]["pass"][ptbins[0]].axes[0]  # First pt bin for binning info
    msdbins = msd_axis.edges
    #print(msdbins)

    msd = rl.Observable("msd", msdbins)

    # here we derive these all at once with 2D array
    ptpts, msdpts = np.meshgrid(ptbins[:-1] + 0.3 * np.diff(ptbins), msdbins[:-1] + 0.5 * np.diff(msdbins), indexing="ij")
    rhopts = 2 * np.log(msdpts / ptpts)
    ptscaled = (ptpts - 450.0) / (1200.0 - 450.0)
    rhoscaled = (rhopts - (-6)) / ((-2.1) - (-6))
    validbins = (rhoscaled >= 0) & (rhoscaled <= 1)
    rhoscaled[~validbins] = 1  # we will mask these out later

    ######## BUILDING THE MODEL ########
    # Build the QCD MC pass+fail model and fit to polynomial
    qcdmodel = rl.Model("qcdmodel")
    qcdpass, qcdfail = 0.0, 0.0

    for ptbin in ptbins[:-1]:
        failCh = rl.Channel(f"ptbin{ptbin}fail")
        passCh = rl.Channel(f"ptbin{ptbin}pass")
        qcdmodel.addChannel(failCh)
        qcdmodel.addChannel(passCh)

        # FOR PICKLES
        failTempl = get_hist("QCD", "fail", ptbin, histograms)
        passTempl = get_hist("QCD", "pass", ptbin, histograms)
        
        failCh.setObservation(failTempl, read_sumw2=True)
        passCh.setObservation(passTempl, read_sumw2=True)

        qcdfail += sum([val for val in failCh.getObservation()[0]])
        qcdpass += sum([val for val in passCh.getObservation()[0]])

    # Compute QCD efficiency
    qcdeff = qcdpass / qcdfail
    print(f"QCD pass: {qcdpass}, fail: {qcdfail}, eff: {qcdeff}")
    tf_MCtempl = rl.BernsteinPoly("tf_MCtempl", (1, 1), ["pt", "rho"], limits=(0, 10))    #OLD

    #tf_MCtempl = rl.BasisPoly("tf_MCtempl", (1,1), ["pt", "rho"], basis='Bernstein', limits=(0, 10))  # (2,2) original

    tf_MCtempl_params = qcdeff * tf_MCtempl(ptscaled, rhoscaled)
    
    # Apply transfer function to QCD
    for ptbin in ptbins[:-1]:
        failCh = qcdmodel[f"ptbin{ptbin}fail"]
        passCh = qcdmodel[f"ptbin{ptbin}pass"]
        failObs = np.maximum(failCh.getObservation()[0], 1e-3)  # Avoid division by zero
        #failObs = failCh.getObservation()[0]
        #failObs = failObs[0]  # Extract sumw (bin counts)
        qcdparams = np.array([
            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}_msdbin{i}", 0)
            for i in range(msd.nbins)
        ])
        sigmascale = 10.0 # original 10 (works for fit strategy 0 and 2)   # 3.0 for fitstrategy 1 
        #scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** np.array(qcdparams, dtype=object)
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams

        fail_qcd = rl.ParametericSample(f"{failCh.name}_qcd", rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        # Use the index instead of the bin lower edge
        ptbin_index = np.where(ptbins[:-1] == ptbin)[0][0]  # Find the index
        pass_qcd = rl.TransferFactorSample(f"{passCh.name}_qcd", rl.Sample.BACKGROUND, tf_MCtempl_params[ptbin_index, :], fail_qcd)
        passCh.addSample(pass_qcd)

    # Fit QCD Model with RooFit
    qcdfit_ws = ROOT.RooWorkspace("qcdfit_ws")
    simpdf, obs = qcdmodel.renderRoofit(qcdfit_ws)
    qcdfit = simpdf.fitTo(
        obs,
        ROOT.RooFit.Extended(True),
        ROOT.RooFit.SumW2Error(True),
        ROOT.RooFit.Strategy(0),  # 0: fast fit, 1: more accurate fit, 2: very accurate fit
        ROOT.RooFit.Save(),
        ROOT.RooFit.Minimizer("Minuit2", "migrad"),
        ROOT.RooFit.PrintLevel(1),
    )
    

    qcdfit_ws.add(qcdfit)
    if "pytest" not in sys.modules:
        qcdfit_ws.writeToFile(str(output_dir / "qcdfit.root"))
    if qcdfit.status() != 0:
        raise RuntimeError("Could not fit QCD")
    
    #print("QCD Fit Result:")
    #qcdfit.Print("v")
 

    # Decorrelate nuisance parameters 
    param_names = [p.name for p in tf_MCtempl.parameters.reshape(-1)]
    decoVector = rl.DecorrelatedNuisanceVector.fromRooFitResult(tf_MCtempl.name + "_deco", qcdfit, param_names)
    #print("Decorrelated parameters:", decoVector.correlated_params)
    #tf_MCtempl.parameters = decoVector.correlated_params.reshape(tf_MCtempl.parameters.shape)
    tf_MCtempl_params_final = tf_MCtempl(ptscaled, rhoscaled)
    #tf_dataResidual = rl.BernsteinPoly("tf_dataResidual", (2, 2), ["pt", "rho"], limits=(0, 10))
    #tf_dataResidual_params = tf_dataResidual(ptscaled, rhoscaled)
    tf_params = qcdeff * tf_MCtempl_params_final #* tf_dataResidual_params

 
    
    ######## BUILDING THE ACTUAL FIT MODEL ########
    # Build the signal model   
    model = rl.Model("testModel")
    sigs = ['WH', 'ZH', 'VBF', 'ttH', 'ggH']
    for ptbin in ptbins[:-1]:
        for region in ["pass", "fail"]:
            ch = rl.Channel(f"ptbin{ptbin}{region}")
            model.addChannel(ch)

            for process, templ in histograms.items():
                if process == "QCD" or process == "data":
                    continue

                templ = templ[region][ptbin]
                #print(templ)
                stype = rl.Sample.SIGNAL if process in sigs else rl.Sample.BACKGROUND
                sample = rl.TemplateSample(f"{ch.name}_{process}", stype, templ)
                
                templ_array = np.array(templ)
                print(f"DEBUG: {ch.name}_{process} has min value: {templ_array.min()}")
                if templ_array.min() < 0:
                    print(templ)
                # mock systematics
                jecup_ratio = np.random.normal(loc=1, scale=0.05, size=msd.nbins)
                msdUp = np.linspace(0.9, 1.1, msd.nbins)
                msdDn = np.linspace(1.2, 0.8, msd.nbins)

                # for jec we set lnN prior, shape will automatically be converted to norm systematic
                sample.setParamEffect(jec, jecup_ratio)
                sample.setParamEffect(massScale, msdUp, msdDn)
                sample.setParamEffect(lumi, 1.027)

                ch.addSample(sample)

            # Set observed data
            data_obs = get_hist("data", region, ptbin, histograms)
            ch.setObservation(data_obs, read_sumw2=True)

    # Define QCD model
    for ptbin in ptbins[:-1]:
        failCh = model[f"ptbin{ptbin}fail"]
        passCh = model[f"ptbin{ptbin}pass"]
        failObs = np.maximum(failCh.getObservation()[0], 1e-3)  # Avoid division by zero
        #failObs = failCh.getObservation()[0]
        initial_qcd = failObs.astype(float)
        for sample in failCh:
            initial_qcd -= sample.getExpectation(nominal=True)
        if np.sum(initial_qcd) == 0:
            raise ValueError("Initial QCD prediction is zero. This is likely a problem.")
        qcdparams = np.array([
            rl.IndependentParameter(f"qcdparam_ptbin{ptbin}_msdbin{i}", 0)
            for i in range(msd.nbins)
        ])
        sigmascale = 10.0
        scaledparams = failObs * (1 + sigmascale / np.maximum(1.0, np.sqrt(failObs))) ** qcdparams
        fail_qcd = rl.ParametericSample(f"ptbin{ptbin}fail_qcd", rl.Sample.BACKGROUND, msd, scaledparams)
        failCh.addSample(fail_qcd)
        # Use the index instead of the bin lower edge
        ptbin_index = np.where(ptbins[:-1] == ptbin)[0][0]  # Find the index
        pass_qcd = rl.TransferFactorSample(f"ptbin{ptbin}pass_qcd", rl.Sample.BACKGROUND, tf_params[ptbin_index,:], fail_qcd)
        passCh.addSample(pass_qcd)


    # Save workspace and model
    ws_path = output_dir / "last_testModel"
    pkl_path = output_dir / "last_testModel.pkl"

    with open(pkl_path, "wb") as fout:
        pickle.dump(model, fout)

    model.renderCombine(ws_path)
    print(f"Workspace saved: {ws_path}")
    print(f"Model pickle saved: {pkl_path}")



if __name__ == "__main__":
    model_rhalphabet()



