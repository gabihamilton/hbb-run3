# Z→bb tagger scale-factor — method note & Raffaele reproduction

_Branch: `zbb-tagsf-raffaele-repro`. Purpose: measure a **tagger efficiency SF** in the Z+γ
region using a proper tag-and-probe, cross-check it against Raffaele Gerosa's boosted Z→bb
calibration (~0.9), and decide with the group which fit structure to adopt._

---

## 1. The physics distinction (the whole point)

A tagger scale factor we can **apply in the SR** is an **efficiency ratio**:

    SF = ε_data / ε_MC ,   ε = N_pass / N_total = N_pass / (N_pass + N_fail)

It is a property of the tagger acting on real bb jets, so it **transfers** to H→bb / Z→bb in the SR.
This is what Raffaele measures:  SF = μ_post-tag / μ_pre-tag.

A **normalization / signal strength** answers a different question ("is the overall Z→bb rate
right?") and is **production-specific** — it does NOT transfer to the SR.

**Both are called "scale factors," but only the efficiency one is the tagger calibration.**

### Illustration (perfect tagger, but Zγ cross-section 2.5× too high in MC)
| | total | pass (60%) | fail (40%) |
|---|---|---|---|
| MC   | 1000 | 600 | 400 |
| data |  400 | 240 | 160 |

- A single POI on **pass+fail** → 400/1000 = **0.4**. Looks like a terrible SF; the tagger is perfect. The 0.4 is 100% the normalization error.
- An **efficiency** measurement → pass fraction 240/400 = 60% = MC → **SF = 1.0**. Correct.

A normalization problem masquerades as a bad "SF" when you scale pass+fail together, and cancels
the moment you measure an efficiency.

---

## 2. Why last week's r_bb ≈ 0.4 is NOT a comparable baseline

Last week's combined fit mapped `map=.*/zgammabb:r_bb` — the `.*/` matches **every** channel, so
`r_bb` scaled the Z→bb template in **passbb, passcc, AND fail** simultaneously. That freezes the
pass/fail ratio (the efficiency) at its MC value, so `r_bb` measured a **total normalization**,
not a tagger efficiency. It is a **different quantity** from the SF we now want — so **0.4 is not
the number to compare against**; it's shown only to explain the change of method.

(Subtlety: the intended slide — SF = (z_norm·r_bb)/z_norm = r_bb — is a valid tag-and-probe, but it
requires `r_bb` on the **pass peak only**. The datacard put `r_bb` on fail too, so the slide's
`z_norm` cancellation never actually happened.)

---

## 3. The comparison baseline = Raffaele (external), not last week

Raffaele Gerosa, boosted X→bb calibration (`Rafaelesupdate.pdf`), tag = p(X→bb) vs QCD:

| WP | cut | efficiency SF |
|---|---|---|
| VHP | > 0.99 | **0.92** (+0.08/−0.06) |
| HP | [0.975, 0.99] | 0.88 (+0.09/−0.08) |
| MP | [0.95, 0.975] | 0.91 (+0.12/−0.10) |

JMS 0.975–0.982 (data ~1–2% below MC), JMR ~unity. Selection: Z→μμ CR (pre-tag) + Z→bb dijet
(probe), leading Xbb jet pT **[400,500]**, GloParT regressed mass. **The number to reproduce: ~0.9.**

---

## 4. Three fit structures (same target quantity, different pre-tag reference)

| | μ_pre-tag reference | uses Z→μμ? | r_bb on |
|---|---|---|---|
| **Raffaele** | Z→μμ CR (same DY production as his Z→bb) | yes | pass |
| **A (self-contained)** | Zγ **fail** region (same Zγ sample) | **no** | passbb only |
| **B (slide, corrected = literal Raffaele)** | Z→μμ CR, shared `z_norm` | yes | passbb only |

Wiring in all cases: `z_norm` on zgammabb in **pass+fail** (sets the total); `r_bb` on **passbb only**
(moves pass relative to fail = efficiency). Then **SF = r_bb**.

- **A** advantage: no cross-process assumption (normalization & efficiency from the same Zγ sample).
  Caveat (Raffaele's own): the fail region is low-S/B, so `z_norm` from fail can be delicate → watch
  its uncertainty.
- **B** advantage: μμ is a clean high-S/B Z peak → robust `z_norm`. Caveat: μμ is **DY**, the signal is
  **Zγ** — different production, so a Zγ cross-section mismodeling leaks into the SF.
- **A vs B is itself diagnostic:** agree → Zγ normalization is fine, it's a genuine tagger story;
  disagree → the gap is the Zγ cross-section.

_Footnote: A measures a pass/fail **odds** ratio, = the **efficiency** ratio only when the pass
fraction is small — true for VHP>0.99, so fine. B (μμ-referenced) is the efficiency ratio exactly._

---

## 4b. OPEN ITEM — the μμ reference is plain DY, NOT μμ+γ (do next for method B)

**Finding (Aug 24, code inspection — nothing changed):** the `control-zmumu` region is plain
Drell-Yan Z→μμ, with **no photon requirement**:
- `categorizer.py:782` region cuts = muontrigger, lumimask, metfilter, twoloosemuon, twomuon_zmm,
  muonkin_leadzmm, muonpairkin_zmm — **no `atleastonephoton`** (the Zγ region at :774 HAS it).
- `template_utils.py` zmmcr selection = mll window & dimuon-pt only — no photon.
- `zmm` process = `Zjets` group = `DYto2L/2E/2Mu` (`pmap_run3.json`), **not** the `DYGto2LG`
  (DY+photon) samples — even though `DYGto2LG-1Jets_Bin-MLL-50-PTG-*` DO exist in the 24Aug07_v15 skims.

**Why it matters:** μμ+γ (Zγ→μμ) is the **same production** as the Zγ→bb signal → it would be a
confound-free pre-tag reference (like Raffaele's all-DY setup). The current plain-DY μμ region is a
**different** production → it IS the "Zγ-vs-DY confound" that method B carries. Making the μμ region
require a photon (Zγ→μμ) would remove that confound and make method B airtight.

**Next step (when we build the μμ card for method B):** add a photon requirement to `control-zmumu`
(e.g. `atleastonephoton`) and include the `DYGto2LG` samples, so the reference becomes Zγ→μμ. Requires a
processor/region change + re-histogramming the μμ CR. NOT done here (this study runs method A first).

## 4a0. HEADLINE (Aug 26) — our disc + mSD + μμ+γ reference → r_bb ≈ 1 (tagger fine)

ccs-joint combined fit, OUR modified 4-body disc, mSD, WP 0.82, with the **μμ+γ (DYGto2LG) reference**
= THE STANDARD now (default in `run_combined_ccs_fit.sh`; single-POI profiled errors):
| reference | r_bb | r_ccs | z_norm |
|---|---|---|---|
| plain-DY μμ | 0.43 | 0.34 | 0.995 |
| **μμ+γ (Zγ→μμ) — STANDARD** | **1.065 −0.24/+0.30** | **0.683 −0.16/+0.18** | **0.412 −0.07/+0.09** |

**r_bb consistent with unity → the tagger is well-modelled in our standard config.** This *recovers the old
Test_v15 r_bb≈1.05, r_ccs≈0.68* — but now z_norm≈0.41 is **physically justified** (a real ~2× Zγ MC
over-normalization, measured on BOTH decay channels via μμ+γ), NOT the old-skim artifact that gave z_norm=0.55.
Mechanism: `z_norm·r_bb = pass data/MC = 0.44` (fixed); μμ+γ sets z_norm=0.41 → r_bb=0.44/0.41≈1.
Plain-DY μμ gives z_norm≈1 (can't see the Zγ-specific over-norm) → leaves it all in r_bb (0.43).

**JMS/JMR floated (Aug 28, datacard-level morph, no re-hist):** r_bb=1.071/r_ccs=0.669/z_norm=0.438 —
essentially unchanged. **CMS_jms_2024 = −0.46σ ± 0.18** (fit shifts MC mass DOWN ~3 GeV → confirms the
"MC peak too high" shoulder), **CMS_jmr_2024 = +0.07σ ± 0.42** (~0, resolution fine). ⇒ mass IS mis-modelled
(JMS significant) but the **tag SF is robust to it** (r_bb anchored by the μμ+γ normalization). Model `zgcr_jmsr`.

**JMS/JMR bb-vs-ccs DECORRELATION test (Aug 28, model zgcr_jmsr_decorr, jmsr_decorrelate:true, wgammacs added to jmsr_processes; make_datacards.py patched for per-flavor CMS_jm{s,r}_{year}_{bb,ccs}):**
| | shared | decoupled |
|---|---|---|
| r_bb | 1.071 | 1.129 |
| r_ccs | 0.669 | **0.794** |
| JMS bb / ccs | −0.46 (one) | **−0.34 / −0.96** |
| JMR bb / ccs | +0.07 (one) | **+0.44 / −0.29** |
**→ DECORRELATE:** JMR bb/ccs opposite signs (+0.44 / −0.29 — shared washes to 0); JMS same sign but very
different size (shared under-corrects ccs). r_ccs moves 0.67→0.79 (JMS/JMR matters for the ccs SF). Caveat:
2-POI fit_s pull errors unreliable (tiny) — get single-POI ± before quoting. (make_datacards.py.bak = original.)

**⇒ The low r_bb=0.27 (VHP regmass) was NOT a tagger problem** — it's the regmass+tight-VHP observable/WP
(pass data/MC=0.13 vs 0.44 for mSD/0.82). Tagger SF ≈ 1 on mSD; regmass/VHP is where it drops.
Caveat: μμ+γ has ~121 events → z_norm ~15%, r_bb ±0.30. Script: `run_combined_ccs_fit.sh --zmmdir <μμ+γ card>`.

## 4c. Apples-to-apples decomposition (our disc, no z_norm) — resolves "why r_bb changed so much"

Old combined gave r_bb≈1.07 (z_norm=0.555); new gives 0.13–0.27. All standalone-Zγ, no-z_norm r_bb below
(the number data actually fixes = pass data/MC). The change decomposes into 4 independent pieces, no bug:

| disc / WP | observable | r_bb (no z_norm) | pass-only vs pass+fail |
|---|---|---|---|
| modified 4-body, 0.82 | mSD | **0.44** | identical (0.445 / 0.446) |
| modified 4-body, 0.82 | regmass | **0.25** | identical (0.250 / 0.252) |
| raw p(Xbb)vsQCD, 0.99 (VHP) | regmass | **0.13** | identical (0.128 / 0.129) |

1. **z_norm↔r_bb partition:** product z_norm·r_bb is invariant (=pass data/MC). Smaller z_norm → bigger r_bb.
   (old z_norm=0.55·1.07=0.59; new μμγ 0.48·0.27=0.13.)
2. **Observable mSD→regmass: 0.44→0.25** (~43% drop), disc/WP fixed. Big lever — regressed-mass Z→bb peak
   has different data/MC than mSD (JMS/mass-modelling; mSD pass data=4593 vs regmass=3246 — the mSD
   soft-drop low-mass tail is in the 40–201 window but excluded by regmass 60–220).
3. **Disc/WP modified-0.82 → raw-VsQCD-0.99 (VHP): 0.25→0.13** (tighter WP, worst-modelled tail).
4. **pass-only vs pass+fail: MOOT** in the Zγ region — fail is QCD-swamped so the fail Z→bb (0.4%) can't
   constrain; the map only matters once z_norm is independently pinned (by μμ).

## 5. Provenance (scripts on this branch)

| file | role |
|---|---|
| `setup_zgcr_vhp.json` | VHP: `disc_mode:bbvsqcd`, WP 0.99, regressed mass 60–220, bb-only, r_bb passbb-only |
| `template_utils.py` | `disc_mode:"bbvsqcd"` gate (p(Xbb)vsQCD pass/fail); default path unchanged |
| `run_vhp_hist_container.sh` | make_hists_dask driver (coffea-dask container) |
| `run_zgcr_vhp_downstream.sh` | collect_hists → bridge root name → make_datacards (`--analysis zgcr_vhp`) |
| `run_zgcr_vhp_fit.sh` | **Method A** fit (Zγ passbb+fail; z_norm on both; r_bb passbb-only) |
| `run_zgcr_vhp_fit_B_mumu.sh` | **Method B** fit (+ zmmcr card; z_norm shared with μμ) |

Run: 2024, tag 24Aug07_v15, 109 fb⁻¹. pt cut: currently **>250** (baseline-controlled). Raffaele is
at pt **[400,500]** — a follow-up variant (`pt_min_scale:400`, `bins:[400,500]`) matches his phase space.

---

## 6. Results (fill as fits complete)

| config | pt | r_bb (=SF) | z_norm | fit quality |
|---|---|---|---|---|
| Raffaele VHP (external) | [400,500] | 0.92 +0.08/−0.06 | — | — |
| **A** Zγ fail-referenced | >250 | railed (−100) | **2.1 ± 2.7** | **FAILED — z_norm unconstrained** |
| **B** μμ-referenced | >250 | **0.129 −0.042/+0.045** | 0.997 ± 0.020 | clean, no rail |
| **B**, pt[400,500] (Raffaele window) | [400,500] | fit fails | — | **stats-starved: 16 pass events** |
| **B+μμγ** — μμ+γ reference (Zγ→μμ, DYGto2LG) | >250 | **0.271 −0.096/+0.117** | **0.476 ± 0.09** | clean; z_norm absorbs Zγ norm |

**μμ+γ reference (Aug 25) — the confound removed:** switched the μμ CR to Z(μμ)+γ using the dedicated
**DYGto2LG** sample (μμ analog of the bb region's ZGto2QG). μμ+γ yields: data=121, MC(Zγ→μμ)=244 →
**data/MC≈0.49**, i.e. the Zγ MC over-predicts ~2× on the μμ side TOO — consistent with the bb side.
Result: **z_norm=0.476** (absorbs the Zγ over-normalization), **r_bb=0.271** (the clean tag efficiency).
Sanity: r_bb·z_norm = 0.27×0.48 = 0.13 = plain-DY (0.13×1.0) — the fit just re-partitioned norm↔efficiency.
**But r_bb≈0.27 is still ≪ Raffaele's 0.92** → beyond the normalization, a **residual tag-efficiency gap**
remains. Candidates: JMS/JMR (mass modelling), QCD subtraction in the near-signal-pure pass, disc/pt-window
differences vs Raffaele, low pass stats. (Implementation: `make_zmumu_plots.py --gamma`, DY→DYGto2LG.)

**pt[400,500] (Raffaele's window):** pass data=16 / MC zgammabb=44 → **overshoot persists (~2.75×)**, so it
is NOT a low-pt artifact. But the fit **fails to converge** (FitDiagnostics + robustFit both fail) — only
16 data events in pass. The photon requirement starves the statistics at high pt, so we **cannot reproduce
Raffaele's phase space** with this selection. (Raffaele's Z→bb dijet has far more events there.)

### BOTTOM LINE (for the group)
Reproducing the tag-and-probe **did not recover ~0.9**, and the postfit shows **why**: it's not the fit
machinery — the **Zγ→bb MC over-predicts data by ~2×** (pt-independent), and a **plain-DY μμ** reference
can't separate that normalization from the tag efficiency. Method A (self-normalize off fail) fails (fail
S/B too low). pt[400,500] is stats-limited. **Two fixes, both worth doing:** (1) **μμ+γ reference** (§4b) —
lets z_norm absorb the Zγ normalization so r_bb is the clean efficiency; (2) fix the **Zγ→bb MC normalization**
(V+γ corrections). The clean tagger SF is blocked on one of these, not on the fitting method.

**Result (Aug 24, VHP p(Xbb)vsQCD>0.99, regressed mass, pt>250, 2024):**
- **Method A failed** as Raffaele warned: yields are pass data=94 / MC zgammabb=214; fail data=339492 with
  zgammabb=1370 = **0.4% of fail**. The fail Z→bb peak is too buried to pin `z_norm` → `z_norm=2.1±2.7`
  (unconstrained) → `r_bb` rails. This *validates* using μμ (his reason), and is a finding in itself.
- **Method B clean:** `z_norm=0.997±0.020` (μμ-pinned), `r_bb = 0.129 ± 0.04`. **Well below Raffaele's 0.92.**
- **Caveats before over-reading the 0.13:** (i) μμ here is plain DY, not Zγ→μμ → carries the Zγ-vs-DY
  normalization confound (§4b) — if the Zγ cross-section is overestimated in MC, it suppresses r_bb;
  (ii) pt>250 vs Raffaele's [400,500]; (iii) MC zgammabb includes Zjets→bb, not just Zγ→bb.
  Next diagnostics: postfit plot of pass (is there a real Z peak in data?), pt[400,500] variant, μμ+γ upgrade.

**Postfit plot (VHP passbb, regressed mass — `plots_vhp_B/`):** data DOES show a real Z→bb peak
(~21 events at 85–95 GeV) → the region + fit work, not noise. But **MC zgammabb overshoots data**:
prefit peak ~84 vs data ~21 (~4× at peak; 214 vs 94 = ~2.3× in yield). r_bb=0.13 is the fit scaling that
overshoot down. ⇒ the low r_bb is dominated by a **Zγ→bb MC over-prediction** (= the meeting's "Z norms
~0.4"), NOT mainly tagger inefficiency. **pt[400,500] won't fix a 2× normalization; the μμ+γ upgrade (§4b)
would** — a Zγ→μμ reference lets z_norm absorb the Zγ normalization, isolating the true efficiency in r_bb.

_Not comparable (different method, shown for context): last week r_bb≈0.4 = total normalization._
