### Going from skims to datacards

## 1. Environment Setup
Set up fitting environment (only needs to be done once)
```
micromamba activate hbb
```

```
micromamba install root cms-combine -c conda-forge
```

## 2. Producing Histograms
The `make_hists.py` script reads skims from EOS and produces two outputs used for downstream analysis.
```
python make_hists.py \
    --year 2022 \
    --tag 26Feb03 \
    --setup setup_zgcr.json \
    --variable msd1 \
    --save-root
```
Outputs:

- ROOT File: results/fitting_{year}_{region}.root (Used by Datacard Maker)

- Pickle File: results/histograms_msd1_{year}_{region}.pkl (Used by Plotter)

Note: Ensure your setup.json has the correct branch_name (e.g., FatJet0_msd) to avoid KeyErrors.

## 3. Validation Plotting
To verify the Data/MC agreement before running the fit, use the unified plotter. This script expects the .pkl files generated in Step 2.
```
python plot_control.py \
    --year 2022 \
    --indir results \
    --outdir plots \
    --region control-zgamma \
    --variable msd1
```

## 4. Datacard Creation
The make_datacards.py script builds the statistical model using rhalphalib. It performs the QCD Transfer Factor (TF) fit and prepares the workspace for Combine.
```
python make_datacards.py \
    --year 2022 \
    --tag 26Feb03 \
    --indir results \
    --analysis zgcr
```



Lara's pipeline:
Definition of categories and fit observable are in setup.json

Create signal region root pass / fail histograms from skims
```
python3 make_hists.py --year 2022EE --tag 25July21
```

Create your datacards and fit your QCD MC transfer factors
```
python3 make_datacards.py --year 2022EE --tag 25July21
```
