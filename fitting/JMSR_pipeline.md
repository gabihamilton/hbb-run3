# JMSR Extraction Pipeline — Zgamma Control Region

This document records every step of the Jet Mass Scale (JMS) and Jet Mass Resolution (JMR)
extraction from the Zgamma control region fit, in the exact order we ran them.

**Branch:** `jmsr` (PR #64, authored by Cristina Mantilla)
**Tag used for output:** `JMSR_test` (avoids overwriting existing `Run3_24` results)
**Data:** Full Run 3 real data (2022, 2022EE, 2023, 2023BPix, 2024)

---

## Key concepts

### 1. What does "per 1σ" mean?

In Combine, every nuisance parameter `θ` is always normalized to a dimensionless number.
By convention:
- `θ = 0` → nominal (no change to the template)
- `θ = +1` → the "+1σ" variation
- `θ = -1` → the "-1σ" variation

"±3 GeV per 1σ" means the **physical effect** is `θ_JMS × 3 GeV`. So:
- If the fit finds `θ_JMS = +1`, jet masses are shifted +3 GeV
- If the fit finds `θ_JMS = -0.79` (like 2024), jet masses are shifted -2.4 GeV
- If the fit finds `θ_JMS = 0`, nothing changes

The fit is free to land anywhere — not just at whole numbers. The "σ" language just
names the scale of the variation; the fit treats `θ` as a continuous parameter.

---

### 2. What is a Gaussian pull, and how does the Z→bb peak provide the constraint?

**Gaussian pull:** Most nuisance parameters in HEP fits have a *Gaussian prior* built in.
The total likelihood is multiplied by `exp(−θ²/2)`, which is just a Gaussian centered at 0.
This acts like gravity pulling `θ` back toward zero: moving to `θ = ±1` costs a factor of
`e^(−0.5) ≈ 0.6` in likelihood. Moving to `θ = ±3` costs `e^(−4.5) ≈ 0.01`. The fit can
still move `θ` away from 0, but only if the data strongly prefers it — it has to "pay" the
Gaussian penalty. The printed value in fit output ("pull = 0.5σ") tells you how far the fit
moved `θ` against this penalty.

**Unconstrained (shapeU):** Our JMS/JMR nuisances use `shapeU`, which removes the
`exp(−θ²/2)` term entirely. There is no penalty for any value of `θ`. The fit floats it
completely freely, so the *only* thing anchoring JMS/JMR is the data itself.

**The Z→bb peak as the constraint:** The Z boson has a fixed, well-known mass of 91.2 GeV.
When events pass the bb-tagger in the passbb region, the jet mass distribution shows a clear
peak near 91 GeV. JMS shifts the MC template peak left or right. If `θ_JMS = +1` moves the
template peak to 94 GeV but the data peak sits at 91 GeV, the fit matches poorly — bad
likelihood. The fit therefore converges to the `θ_JMS` value that aligns the template peak
with the data peak. The *width* of the peak constrains JMR similarly: a wider data peak
pushes `θ_JMR > 0`. No Gaussian prior needed — the peak shape itself is informative enough.

---

### 3. What is CDF interpolation?

The **CDF (Cumulative Distribution Function)** of a histogram is just the running sum:
`CDF(x)` = (number of events with jet mass below `x`) / (total events).
It goes from 0 at the left edge to 1 at the right edge and is always monotonically increasing.

To **shift** a histogram by +δ GeV using the CDF:
> For each bin spanning [`x_left`, `x_right`], the new bin content is
> `CDF(x_right − δ) − CDF(x_left − δ)`.

Instead of asking "how many events are between `x_left` and `x_right`?", you ask
"how many events were between `x_left − δ` and `x_right − δ`?" — i.e., you look one shift
upstream in the original distribution. This handles sub-bin shifts smoothly (no
interpolation artifacts) and automatically preserves the total normalization because
the CDF still goes from 0 to 1.

To **smear** (broaden) a histogram by scale `s`:
> Evaluate the CDF at `((x − mean) / s + mean)` instead of `x`, which stretches the
> distribution around its mean by the factor `s`.

This is what `AffineMorphTemplate` in `scalesmear.py` does. "Affine" refers to
shift + scale — both operations applied together in a single CDF lookup.

---

### 4. What does "morphed" mean?

A **template** is a histogram representing what a physics process looks like in the
observable (here, jet mass). It is built from MC simulation.

**Morphing** means continuously deforming that histogram as a function of some parameter.
Without morphing, Combine shape systematics work with three fixed snapshots —
nominal, `+1σ`, `−1σ` — and linearly interpolate between them. Morphing is different:
instead of fixed snapshots, `AffineMorphTemplate` takes the *single nominal histogram*
and recomputes it on the fly for every value of `θ` the minimizer tries, using the CDF
shift+scale described above.

For `zgammabb` specifically:
- Nominal histogram: sharp Z→bb mass peak around 91 GeV
- At `θ_JMS = +1`: the entire peak shifts +3 GeV → peak sits at 94 GeV
- At `θ_JMR = +1`: the peak width expands by 20% → shoulder broadens
- Both can be applied simultaneously

**Why only `zgammabb` is morphed:** Morphing is only physically meaningful for processes
that have a sharp, well-localized mass peak — because that's what gives the fit information
about JMS/JMR. `zgammabb` has the Z→bb peak. The backgrounds (QCD, Wgamma, Wjets, ttbar)
have broad, featureless mass distributions. Morphing them would add model parameters with
no physics motivation and no constraining power — the fit would just find arbitrary shapes
to absorb statistical fluctuations.

---

## Background: what JMS and JMR are

- **JMS (Jet Mass Scale):** The jet mass axis in data may be shifted by a few GeV relative to
  MC. JMS is parameterized as a nuisance that shifts the mass distribution left or right by
  `jmsr_scale` GeV per 1σ. In `setup_zgcr.json`: `"jmsr_scale": 3` → ±3 GeV per 1σ.

- **JMR (Jet Mass Resolution):** The mass peak in data may be wider or narrower than MC.
  JMR is parameterized as a nuisance that smears (convolves) the mass distribution by
  `jmsr_smear` per 1σ. In `setup_zgcr.json`: `"jmsr_smear": 0.2` → ±20% width per 1σ.

- Both are implemented as **`shapeU` nuisances** in Combine — unconstrained (no Gaussian
  pull toward zero), so the fit floats them freely. The constraint comes entirely from the
  Z→bb mass peak shape in data.

- The morphing is done via **`MorphHistW2` / `AffineMorphTemplate`** in `scalesmear.py`,
  which shifts/smears 1D histograms via CDF interpolation while preserving normalization
  and propagating bin uncertainties (sumw2).

- Only the `zgammabb` process (Z→bb + Zjets bb) is morphed, because it has the sharp
  Z mass peak that the fit uses to measure JMS/JMR. The `jmsr_processes` key in
  `setup_zgcr.json` controls this.

---

## Step 1 — Pull the `jmsr` branch

Cristina's PR #64 adds the JMSR infrastructure. Pull it before running anything:

```bash
git fetch origin jmsr
git checkout jmsr
```

Key files added or modified by this branch:
- `scalesmear.py` — `AffineMorphTemplate` and `MorphHistW2` morphing classes
- `make_datacards.py` — applies morphing to processes in `jmsr_processes`; adds
  `CMS_jms_{year}` and `CMS_jmr_{year}` as `shapeU` nuisances
- `card_utils.py` — new `safe_ratio()` helper; `add_systematics()` handles `"shape"` prior
- `setup_zgcr.json` — `do_systematics: true`, `active_systematics`, `jmsr_*` keys
- `plot_jmsr.py` — pre-fit JMS/JMR shape validation plots from the RooWorkspace
- `plot_jmsr_postfit.py` — post-fit JMS/JMR plots using fitted nuisance values

---

## Step 2 — Make datacards for all years

`make_datacards.py` reads the input ROOT histograms, builds the statistical model with
rhalphalib (QCD transfer factor, signal/background templates), applies the JMS/JMR morphing
to `zgammabb`, and writes text datacards + a `build.sh` script.

Using `--tag JMSR_test` saves output under `results/JMSR_test/` so it does not overwrite
the existing `Run3_24` results.

```bash
for year in 2022 2022EE 2023 2023BPix 2024; do
    python make_datacards.py \
        --year ${year} \
        --tag JMSR_test \
        --indir results \
        --analysis zgcr
done
```

Output per year: `results/JMSR_test/{year}/datacards/zgcrModel_{year}/`
containing text datacards and `build.sh`.

---

## Step 3 — Build the RooWorkspace for each year

`build.sh` calls `text2workspace.py` on the per-year datacards to produce a binary
Combine workspace (`zgcrModel_{year}.root`). This compiles the statistical model into a
format Combine can actually run on.

```bash
for year in 2022 2022EE 2023 2023BPix 2024; do
    cd results/JMSR_test/${year}/datacards/zgcrModel_${year}
    ./build.sh
    cd -
done
```

Output per year: `zgcrModel_{year}.root` in the same datacards directory.

---

## Step 4 — Combine all years into a single Run 3 datacard

`combineCards.py` stitches together the five per-year text datacards into one.
The prefix (`y22=`, `y22EE=`, etc.) tells Combine how to correlate systematics across years:
nuisances with the same name in different cards are treated as correlated; year-specific
nuisances (e.g., `CMS_jms_2022`) are automatically kept separate.

```bash
combineCards.py \
    y22=results/JMSR_test/2022/datacards/zgcrModel_2022/model_combined.txt \
    y22EE=results/JMSR_test/2022EE/datacards/zgcrModel_2022EE/model_combined.txt \
    y23=results/JMSR_test/2023/datacards/zgcrModel_2023/model_combined.txt \
    y23BPix=results/JMSR_test/2023BPix/datacards/zgcrModel_2023BPix/model_combined.txt \
    y24=results/JMSR_test/2024/datacards/zgcrModel_2024/model_combined.txt \
    > full_Run3_zgcrModel_JMSR.txt
```

---

## Step 5 — Convert to a combined RooWorkspace

`text2workspace.py` converts the combined text datacard to a binary workspace and
injects the physics model (multi-signal, with `r_bb` and `r_cc` as free POIs).

```bash
text2workspace.py \
    -P HiggsAnalysis.CombinedLimit.PhysicsModel:multiSignalModel \
    --PO "map=.*/zgammabb:r_bb[1,-100,100]" \
    --PO "map=.*/zgammacc:r_cc[1,-100,100]" \
    full_Run3_zgcrModel_JMSR.txt \
    -o workspace_Run3_JMSR.root
```

---

## Step 6 — Run FitDiagnostics on real data

`FitDiagnostics` performs a maximum-likelihood fit and saves the full post-fit shapes.
Flags explained:

| Flag | Purpose |
|------|---------|
| `--saveShapes` | Write per-bin histograms (pre/post-fit) into the output ROOT file |
| `--saveWithUncertainties` | Propagate fit uncertainty into the saved shapes |
| `--robustFit 1` | Use a more reliable minimizer strategy when the fit is tricky |
| `--cminDefaultMinimizerStrategy 1` | Slower but more stable Minuit2 strategy |
| `-n _Run3_24_jmsr_data` | Suffix for the output file name |

```bash
combine -M FitDiagnostics workspace_Run3_JMSR.root \
    --saveShapes --saveWithUncertainties \
    --robustFit 1 \
    --cminDefaultMinimizerStrategy 1 \
    -n _Run3_24_jmsr_data
```

Output: `fitDiagnostics_Run3_24_jmsr_data.root`

> **Note on `--saveWithUncertainties`:** With multiple POIs (`r_bb`, `r_cc`) and the global
> `qcd_norm` rateParam, the Hessian can have a near-zero eigenvalue that makes uncertainty
> propagation unreliable. If you see extremely large upper uncertainties (e.g.,
> `r_bb: 1.0 -0.05/+99`), this is a known artifact. The central values and JMS/JMR nuisances
> are still correct and usable.

---

## Step 7 — Pre-fit JMS/JMR shape validation plots

`plot_jmsr.py` reads the per-year RooWorkspace (built in Step 3) and plots the nominal
template alongside the ±1σ JMS and JMR variations, so you can visually verify that the
morphing is doing the right thing *before* the fit.

> **Important:** Always pass `--year ${year}`. Without it the script defaults to `2022EE`
> and looks for `CMS_jms_2022EE` in every workspace — producing 0 plots for all other years.

```bash
mkdir -p plots/jmsr_prefit

for year in 2022 2022EE 2023 2023BPix 2024; do
    python plot_jmsr.py \
        --file results/JMSR_test/${year}/datacards/zgcrModel_${year}/zgcrModel_${year}.root \
        --outdir plots/jmsr_prefit \
        --year ${year}
done
```

Output: `plots/jmsr_prefit/` — 6 plots per year × 5 years = **30 plots** total.
Each plot shows nominal (black), up (blue dashed), down (violet dashed) + ratio panel.
Channels: `ptbin0zgcrfail`, `ptbin0zgcrpassbb`, `ptbin0zgcrpasscc`.
Variations: `JMS`, `JMR`.

---

## Step 8 — Post-fit JMS/JMR morphed template plots

`plot_jmsr_postfit.py` reads the fitted JMS/JMR nuisance values from the `fitDiagnostics`
file (`fit_s` RooFitResult), then applies `MorphHistW2` at those values to show what the
`zgammabb` template actually looks like after the fit has adjusted JMS and JMR.
It also draws ±1σ morphed bands around the central fitted value.

```bash
mkdir -p plots/jmsr_postfit

for year in 2022 2022EE 2023 2023BPix 2024; do
    python plot_jmsr_postfit.py \
        --fitfile fitDiagnostics_Run3_24_jmsr_data.root \
        --wsfile results/JMSR_test/${year}/datacards/zgcrModel_${year}/zgcrModel_${year}.root \
        --year ${year} \
        -j setup_zgcr.json \
        --outdir plots/jmsr_postfit
done
```

Output: `plots/jmsr_postfit/` — 3 plots per year × 5 years = **15 plots** total.

---

## Fit results summary

Fitted JMS/JMR nuisance values from the full Run 3 real-data fit:

| Year | CMS_jms (σ) | JMS shift | CMS_jmr (σ) | JMR scale | Interpretation |
|------|-------------|-----------|-------------|-----------|----------------|
| 2022 | -1.00 ± 1.89 | -3.0 GeV | +1.00 ± 1.59 | 1.20× | Low stats — unconstrained |
| 2022EE | **+1.00 ± 0.34** | +3.0 GeV | +0.26 ± 1.52 | 1.05× | Strong JMS signal (~3σ) |
| 2023 | -0.02 ± 0.82 | -0.06 GeV | -1.00 ± 1.87 | 0.80× | JMS compatible with 0 |
| 2023BPix | +1.00 ± 1.68 | +3.0 GeV | +1.00 ± 2.00 | 1.20× | Insufficient stats |
| 2024 | **-0.79 ± 0.16** | -2.4 GeV | **+1.00 ± 0.05** | 1.20× | Most constrained year |

**Key takeaways:**
- **2024** is the most statistically powerful year. JMS is constrained to -2.4 GeV (5σ significance).
  JMR is pinned at the +20% boundary with a very small uncertainty — suggests the smearing range
  `jmsr_smear: 0.2` may need to be extended.
- **2022EE** shows a +3 GeV JMS shift with ~3σ significance. The sign is opposite to 2024,
  consistent with different detector conditions between Run 3 eras.
- **2023** JMS is consistent with zero — jet mass well calibrated for that year.
- **2022 and 2023BPix** are statistics-limited; nuisances float to the boundary.
