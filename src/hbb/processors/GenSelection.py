"""
Gen selection functions for skimmer.

Author(s): Raghav Kansal, Cristina Mantilla Suarez
"""

from __future__ import annotations

import awkward as ak
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray

from ..utils import P4

d_PDGID = 1
u_PDGID = 2
s_PDGID = 3
c_PDGID = 4
b_PDGID = 5
g_PDGID = 21
TOP_PDGID = 6

ELE_PDGID = 11
vELE_PDGID = 12
MU_PDGID = 13
vMU_PDGID = 14
TAU_PDGID = 15
vTAU_PDGID = 16

G_PDGID = 22
Z_PDGID = 23
W_PDGID = 24
HIGGS_PDGID = 25

b_PDGIDS = [511, 521, 523]

GEN_FLAGS = ["fromHardProcess", "isLastCopy"]


def getBosons(genparticles):
    absid = abs(genparticles.pdgId)
    return genparticles[
        # no gluons
        (absid >= 22)
        & (absid <= 25)
        & genparticles.hasFlags(["fromHardProcess", "isLastCopy"])
    ]


def bosonFlavor(bosons):
    childid = abs(bosons.children.pdgId)
    genflavor = (
        ak.any(childid == 5, axis=-1) * 3
        + ak.any(childid == 4, axis=-1) * 2
        + ak.all(childid < 4, axis=-1) * 1
    )
    return ak.fill_none(genflavor, 0)


def gen_selection_Hbb(
    events: NanoEventsArray,
    fatjets: FatJetArray,
):
    """Gets H, bb, 4-vectors + Higgs children information"""

    # finding the two gen higgs
    higgs = events.GenPart[
        (abs(events.GenPart.pdgId) == HIGGS_PDGID) * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    higgs_children = higgs.children

    GenHiggsVars = {f"GenH{key}": higgs[var] for (var, key) in P4.items()}
    GenHiggsVars["GenHChildren"] = abs(higgs_children.pdgId[:, :, 0])

    is_bb = abs(higgs_children.pdgId) == b_PDGID
    bs = ak.flatten(higgs_children[is_bb], axis=2)
    GenbVars = {f"Genb{key}": bs[var] for (var, key) in P4.items()}

    # match fatjets to bb
    b_h1 = ak.firsts(higgs_children[is_bb][:, 0:1])  # first b quark
    b_h2 = ak.firsts(higgs_children[is_bb][:, 1:2])  # second b quark
    matched_to_higgs = (
        fatjets.metric_table(higgs) < 0.8
    )  # metric_table returns the deltaR between the fatjet and the higgs

    # boolean array that checks if the fatjet is matched to the Higgs
    fatjets["HiggsMatch"] = ak.any(matched_to_higgs, axis=2)
    # index of the fatjet that is matched to the Higgs
    fatjets["HiggsMatchIndex"] = ak.mask(
        ak.argmin(fatjets.metric_table(higgs), axis=2), fatjets["HiggsMatch"] == 1
    )
    # number of b quarks matched to the first b
    fatjets["NumBMatchedH1"] = ak.sum(fatjets.metric_table(b_h1) < 0.8, axis=2)
    # number of b quarks matched to the second b
    fatjets["NumBMatchedH2"] = ak.sum(fatjets.metric_table(b_h2) < 0.8, axis=2)

    return {**GenHiggsVars, **GenbVars}


def gen_selection_V(
    events: NanoEventsArray,
    fatjets: FatJetArray,
    skim_vars: dict,
):
    """Get W/Z and children information"""
    vs = events.GenPart[
        ((abs(events.GenPart.pdgId) == W_PDGID) | (abs(events.GenPart.pdgId) == Z_PDGID))
        * events.GenPart.hasFlags(GEN_FLAGS)
    ]
    vs_flat = ak.firsts(vs)
    vs_children = vs.children
    vs_pdgId = abs(vs_children.pdgId)

    GenVVars = {f"GenV{key}": vs_flat[var] for (var, key) in skim_vars.items()}
    GenVVars["GenVChildren"] = vs_pdgId

    vs_flat["is_bb"] = (vs_pdgId[:, :, 0] == b_PDGID) & (vs_pdgId[:, :, 1] == b_PDGID)
    vs_flat["is_cc"] = (vs_pdgId[:, :, 0] == c_PDGID) & (vs_pdgId[:, :, 1] == c_PDGID)
    vs_flat["is_cs"] = ((vs_pdgId[:, :, 0] == c_PDGID) & (vs_pdgId[:, :, 1] == s_PDGID)) | (
        (vs_pdgId[:, :, 0] == s_PDGID) & (vs_pdgId[:, :, 1] == c_PDGID)
    )

    GenVVars["GenVis_bb"] = vs_flat["is_bb"]
    GenVVars["GenVis_cc"] = vs_flat["is_cc"]
    GenVVars["GenVis_cs"] = vs_flat["is_cs"]

    # quarks of the first W/Z: W/Z -> qq
    q_v1 = vs_children[:, 0]
    delta_r = fatjets.metric_table(q_v1)
    matched_mask = delta_r < 0.8
    # number of quarks matched to the first W/Z
    fatjets["NumQMatchedV1"] = ak.sum(matched_mask, axis=2)

    # matched_to_v = (
    #    fatjets.metric_table(vs) < 0.8
    # )  # metric_table returns the deltaR between the fatjet and the W/Z
    # boolean array that checks if the fatjet is matched to the W/Z
    # is_fatjet_matched = ak.any(
    #    matched_to_v, axis=2
    # )  # checks if any of the fatjets is matched to the W/Z

    return {**GenVVars}
