from __future__ import annotations

year_map = {
    "2022": ["2022"],
    "2022EE": ["2022EE"],
    "2023": ["2023"],
    "2023BPix": ["2023BPix"],
    "2022-2023": ["2022", "2022EE", "2023", "2023BPix"],
}

common_mc = {
    "ggf-hbb": {"GluGluHto2B_PT-200_M-125"},
    "vh-hbb": {
        "WplusH_Hto2B_Wto2Q_M-125",
        "WminusH_Hto2B_Wto2Q_M-125",
        "ggZH_Hto2B_Zto2Q_M-125",
        "ZH_Hto2B_Zto2Q_M-125",
    },
    "vbf-hbb": {
        "VBFHto2B_M-125_dipoleRecoilOn",
    },
    "qcd": {
        "QCD_HT-200to400",
        "QCD_HT-400to600",
        "QCD_HT-600to800",
        "QCD_HT-800to1000",
        "QCD_HT-1000to1200",
        "QCD_HT-1200to1500",
        "QCD_HT-1500to2000",
        "QCD_HT-2000",  # nao tem em 2023
    },
    "tt": {"TTto2L2Nu", "TTto4Q", "TTtoLNu2Q"},
    # "tt_had": {"TTto4Q"},
    # "tt_dilep": {"TTto2L2Nu"},
    # "tt_semilep": {"TTtoLNu2Q"},
    "singletop": {
        "TBbarQ_t-channel_4FS",
        "TbarBQ_t-channel_4FS",
        "TWminusto2L2Nu",
        "TWminusto4Q",
        "TWminustoLNu2Q",
        "TbarWplusto2L2Nu",
        "TbarWplusto4Q",
        "TbarWplustoLNu2Q",
    },
    "diboson": {
        "WW",
        "WZ",
        "ZZ",
    },
    "wjets": {
        "Wto2Q-3Jets_HT-200to400",
        "Wto2Q-3Jets_HT-400to600",
        "Wto2Q-3Jets_HT-600to800",
        "Wto2Q-3Jets_HT-800",
    },
    "zjets": {
        "Zto2Q-4Jets_HT-200to400",
        "Zto2Q-4Jets_HT-400to600",
        "Zto2Q-4Jets_HT-600to800",
        "Zto2Q-4Jets_HT-800",
    },
    "ewkv": {
        "VBFWtoLNu",
        "VBFWto2Q",
        "VBFZto2Q",
        "VBFZto2L",
        "VBFZto2Nu",
    },
    "zgamma": {
        "ZGto2NuG-1Jets_PTG-100to200",
        "ZGto2NuG-1Jets_PTG-200to400",
        "ZGto2NuG-1Jets_PTG-400to600",
        "ZGto2NuG-1Jets_PTG-600",
        "ZGto2QG-1Jets_PTG-100to200",
        "ZGto2QG-1Jets_PTG-200",
    },
    "wgamma": {
        "WGtoLNuG-1Jets_PTG-100to200",  # nao tem em 2023
        "WGtoLNuG-1Jets_PTG-200to400",
        "WGtoLNuG-1Jets_PTG-400to600",
        "WGtoLNuG-1Jets_PTG-600",
        "WGto2QG-1Jets_PTG-100to200",
        "WGto2QG-1Jets_PTG-200",
    },
    "ttgamma": {
        "TTG-1Jets_PTG-100to200",
        "TTG-1Jets_PTG-200",
        "TTG-1Jets_PTG-10to100",
    },
    "gjets": {
        "GJ_PTG-100to200",
        "GJ_PTG-200to400",
        "GJ_PTG-400to600",
        "GJ_PTG-600",
    },
}

data_by_year = {
    "2022": {
        "JetMET_Run2022C_single",
        "JetMET_Run2022C",
        "JetMET_Run2022D",
    },
    "2022EE": {
        "JetMET_Run2022E",
        "JetMET_Run2022F",
        "JetMET_Run2022G",
    },
    "2023": {
        "JetMET_Run2023Cv1",
        "JetMET_Run2023Cv2",
        "JetMET_Run2023Cv3",
        "JetMET_Run2023Cv4",
    },
    "2023BPix": {
        "JetMET_Run2023D",
    },
}

# --- ADDED for control-tt region ---
data_by_year_muon = {
    "2022": {"Muon_Run2022C", "Muon_Run2022D"},
    "2022EE": {"Muon_Run2022E", "Muon_Run2022F", "Muon_Run2022G"},
    "2023": {"Muon_Run2023Cv1", "Muon_Run2023Cv2", "Muon_Run2023Cv3", "Muon_Run2023Cv4"},
    "2023BPix": {"Muon_Run2023D"},
}

# --- ADDED for control-zgamma region ---
data_by_year_zgamma = {
    "2022": {"EGamma_Run2022C", "EGamma_Run2022D"},
    "2022EE": {"EGamma_Run2022E", "EGamma_Run2022F", "EGamma_Run2022G"},
    "2023": {"EGamma_Run2023Cv1", "EGamma_Run2023Cv2", "EGamma_Run2023Cv3", "EGamma_Run2023Cv4"},
    "2023BPix": {"EGamma_Run2023D"},
}
