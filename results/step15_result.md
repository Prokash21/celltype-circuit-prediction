# Step 15: Does the signal type (dF/F vs deconvolved) explain the small cell-type label gain in Bugeon?
Date: 2026-10-18

## What was run
- `scripts/step15_signal.py <ds> <phase>` for these datasets:
  - `dff`: the step 8 cache, reproducing steps 11 / 13
  - `A`: deconvolved Allen at 10 Hz
  - `B`: deconvolved Allen at 4.3 Hz
  - `A_bfree`: sensitivity check, ridge only

  The phases were `all extract`, `all deconv`, `<ds> gram / features / cluster / ridge / mlp / collect`, `all classify`, `all deconv_bfree` and `all analyse` (writes `results/step15/`). Then `scripts/step15_plots.py` remakes every plot from the CSV files in `results/step15/` only
- **Environment:** Python 3.11 (`.venv`, rebuilt), PyTorch 2.14.1 on CPU, scikit-learn 1.7.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, allensdk 2.16.2 (with setuptools < 70 for `pkg_resources`), and **oasis-deconv 0.3.2** (pip). These are the same library versions as steps 11–13, plus OASIS
- **Data:** nothing was on disk, so everything was rebuilt:
  - **Allen:** `ophys_experiments.json` and the 46 NWB files came from the Allen S3 mirror via step 10's `extract`, unchanged. While each NWB was on disk, the native-rate (~30 Hz) dF/F and the step 4 stimulus features were also kept (`data/step15/raw30/`)
  - **Bugeon:** downloaded from https://ndownloader.figshare.com/files/38046633. The MD5 is 4556ed80ce98ca13f909129e25f60f3a, which matches. figshare returns HTTP 500 to curl's default user agent and a normal redirect with a browser user agent, so the download was not blocked. The dataset's variable table `DatasetTypesV2.xlsx` (figshare file 38046648, MD5 0d367c2f…) was also fetched
  - Nothing under `data/` is committed
- **Same settings as before:** 46 mice (VISp, three_session_A, Cux2 13 / Sst 12 / Vip 12 / Pvalb 9), 5 leave-mice-out folds, chronological 80/20 split, val = last 10% of the train part, stimulus inputs, test windows, ridge alpha grid, MLP architecture / optimiser / batches / early stopping / seeding formula, k-means settings, and the classifier with its inner CV are all those of steps 8–13
- **Wall time on 4 CPU cores:**

  | Phase | Time |
  |---|---|
  | extract (46 NWB) | 16 min |
  | OASIS (3,296 neurons) | ~3.5 min |
  | ridge statistics (dff / A / B) | 6 / 6 / 0.5 min |
  | ridge, all variants (dff / A / B) | 1 min each or less |
  | MLP dff (9 fold-0 reruns) / A (70 fits) / B (70 fits) | 22 / 65 / 55 min |
  | classify (3 datasets) | 3 min |
  | free-baseline sensitivity (deconv + ridge) | 28 min |

### Task 1: deconvolution
- **OASIS** (Friedrich, Zhou & Paninski 2017; package `oasis-deconv` 0.3.2), AR(1), L1 penalty, package defaults (`penalty=1`, `b_nonneg=True`)
- **Per neuron, fitted on the train part only** (first 80% of the 30 Hz frames):
  - `oasis.functions.deconvolve(y_train, penalty=1)` estimates the decay g (autocovariance, fudge factor 0.98) and the noise sn (PSD of 0.25–0.5 × Nyquist)
  - It then optimises the baseline b ≥ 0 and the penalty λ under the noise constraint |c − y|² = sn² T
  - The **whole** trace is deconvolved with these fixed g, b and λ (`oasis_methods.oasisAR1(y − b, g, lam=λ)`), so no parameter sees the test part
- **Check:** on the train part, the fixed-parameter rerun equals the constrained fit except in a short boundary region at the end of the train part:
  - all 3,296 neurons: correlation ≥ 0.9993 and relative L1 difference ≤ 0.21%
  - at most 8 frames differ per neuron, and all of them lie in the last 204 frames (≤ 7 s) of the train part (`data/step15/oasis_check.csv`, not committed)
  - this comes from OASIS's end-of-trace pool correction
- **Version A** (isolates signal type):
  - z-score the deconvolved trace at 30 Hz with the train-part mean / std, then take 3-frame means inside the train and the test part. This is exactly step 5's `load_session`
  - The same windows as before: past 30 → next 10 bins (3 s → 1 s), test stride 10
  - Its stimulus bins, `n_fit` and array shapes were asserted equal to the dF/F cache
- **Version B** (Bugeon-like):
  - 7-frame means (30.1 Hz / 7 ≈ 4.3 Hz), then z-score with the train-part statistics of the bins
  - Past 13 → next 4 frames (3.0 s → 0.93 s), test stride 4
  - Stimulus features are binned the same way
- **No neuron had zero deconvolved variance** in the train part

### Task 2: models, on dff, A and B (same code)
- **Ridge** (step 12 / 13 closed form from summed per-neuron statistics, alpha on val, refit on train + val):
  - b: no label, line-balanced
  - d: one model per Cre line
  - e: shuffled label, from step 8's shuffled session groups (5 seeds, line-balanced)
  - k1: one model with equal row weights, as a reference
  - km16: activity summary, one model per k-means cluster with equal row weights (step 13)
- **MLP** (step 9 / 13 FiLM network, line-balanced batches, 3 seeds; m3 has 5):
  - m1: none
  - m2: Cre FiLM
  - m3: shuffled FiLM (step 9 groups)
  - cont: continuous-summary FiLM. The standardised S vector → linear → 16 dims, zero-initialised gamma / beta, using m2's seeds (step 13)
- **Activity summary:**
  - The S features are recomputed from each dataset's own train part: step 11's `s_features`, 37 features
  - For B: step 12's 4.3 Hz autocorrelation lags and power bands. The train-part z std, which is 1 by construction, is dropped, leaving 34 features
  - Per fold, on training mice only: z-score, PCA keeping 95% of the variance, k-means with k = 16 (n_init 10, seed = fold)
  - k = 16 was step 13's val choice in every fold. **It was not re-chosen for the deconvolved data** (see Problems)
- **Scores:**
  - Unit = mouse (one session each)
  - Score = 1 − ΣSSE / ΣSSE_copylast over the mouse's neurons, with copy-last on the same signal
  - Gain = variant − no label
- **Statistics:**
  - Two-sided Wilcoxon over the 46 mice, paired
  - "Fraction surviving" = mean over mice of the deconvolved gain ÷ mean of the dF/F gain, with a 95% percentile bootstrap over mice (2,000 resamples, paired)
  - "Share of the Allen–Bugeon gap" = (G_dF/F − G_deconv) / (G_dF/F − G_Bugeon), with Allen mice and Bugeon sessions resampled independently
- **Bugeon rows** come from the step 13 (none, true, km16 = chosen k, cont) and step 12 (e, m3) CSVs. Unit = session (17)

### Reproduction check (`results/step15/reproduction_check.csv`)
| Model | Reference | Max abs. difference in mouse score | Max rel. difference in neuron SSE | Mean score here / before |
|---|---|---|---|---|
| ridge b (none) | step 11 | 1.0e-7 | 4.9e-7 | 0.2997 / 0.2997 |
| ridge d (true line) | step 11 | 1.4e-7 | 4.7e-7 | 0.3249 / 0.3249 |
| ridge e (shuffled) | step 11 | 6.2e-8 | 4.9e-7 | 0.2938 / 0.2938 |
| ridge k1 | step 13 | 8.7e-8 | 4.8e-7 | 0.2784 / 0.2784 |
| ridge km16 (summary) | step 13 | 1.8e-7 | 4.8e-7 | 0.3470 / 0.3470 |
| copy-last SSE | step 11 | 1.2e-7 | 4.9e-7 | |
| MLP m1, fold 0 refit (3 seeds) | step 11 | 6.1e-8 | 4.9e-7 | 0.418 / 0.418 (11 mice) |
| MLP m2, fold 0 refit | step 11 | 4.3e-8 | 4.6e-7 | 0.438 / 0.438 |
| MLP cont, fold 0 refit | step 13 | 5.1e-8 | 4.5e-7 | 0.458 / 0.458 |

- **The Allen dF/F numbers reproduce steps 11 and 13 to float32 rounding:**
  - none / true / summary are 0.2997 / 0.3249 / 0.3470 for ridge
  - for the MLP, the gains are +0.0239 (label) and +0.0436 (summary)
  - the S classifier gives 0.868, which equals step 11's full-precision task-4 value
- MLP m1 / m2 / m3 / cont for folds 1–4 on dF/F are read from `results/step11` and `results/step13`. Only fold 0 was refit
- Small note: on dF/F, PCA keeps 18 components here (19–20 for A, B). Step 13's text says "13–14 for Allen". Its k = 16 labels and SSE are reproduced exactly with the same code, so that sentence in step 13 looks like a reporting slip

## Data
- **Allen Visual Coding 2P:**
  - VISp, three_session_A, 46 mice, 3,296 neurons (Cux2 2,905, Sst 122, Vip 153, Pvalb 116), 30.1–31.0 Hz, ~64 min per session
  - Reporters: Cux2 = Ai93, GCaMP6f; Sst / Vip = Ai148, GCaMP6f; Pvalb = Ai162, **GCaMP6s** (step 11)
- **Bugeon et al. 2022:** 17 sessions, 4 mice, 8,199 labelled neuron-sessions, ~4.3 Hz, as in steps 12–13

## Results

### Task 1: what deconvolution does to the Allen traces (`results/step15/oasis_params.csv`, `sparsity.csv`, `example_traces.csv`; plots (a) and `step15_sparsity.png`)
| Line (indicator) | Neurons | OASIS decay τ, median [IQR] | λ > 0 | Zeros: 30 Hz frames | Zeros: A, 10 Hz bins | Zeros: B, 4.3 Hz bins |
|---|---|---|---|---|---|---|
| Cux2 (GCaMP6f) | 2,905 | 0.41 s [0.30, 0.57] | 27% | 0.944 | 0.856 | 0.723 |
| Sst (GCaMP6f) | 122 | 0.56 s [0.43, 0.69] | 20% | 0.947 | 0.869 | 0.754 |
| Vip (GCaMP6f) | 153 | 0.76 s [0.57, 0.99] | 5% | 0.942 | 0.866 | 0.759 |
| Pvalb (GCaMP6s) | 116 | **0.83 s** [0.52, 1.15] | 16% | 0.948 | 0.870 | 0.767 |
| all Allen | 3,296 | 0.43 s | 25% | 0.944 | 0.858 | **0.728** |
| **Bugeon** `frame.neuralActivity` (4.3 Hz) | 8,199 | – | – | – | – | **0.403** (per session 0.27–0.79) |

Fractions are of exact zeros in the train part, as the mean over neurons. Test-part values agree within 0.02.

- **Sparsity, Allen B vs Bugeon:**
  - Allen B (73% zero bins) is sparser than Bugeon on average (40%)
  - The ~79% figure in the step brief holds only for the first four SB025 sessions (0.79 each)
  - The other 13 sessions are 27–40% zeros. Some Bugeon neurons are almost never exactly zero (per-neuron minimum 0.2%)
- **The decay constants follow the indicators:**
  - Pvalb (GCaMP6s) has the longest τ
  - Vip is long too. Its dF/F autocorrelation at 1 s (0.39) is also as high as Pvalb's (0.36; `features_by_line.csv`)
- **The noise constraint is not binding for 75% of neurons:**
  - With b ≥ 0, the unpenalised non-negative fit already leaves RSS above the noise level (median RSS / sn² = 1.08), so λ = 0 for 75% of neurons
  - For these neurons the result is plain non-negative AR(1) deconvolution, the same as suite2p's OASIS, which also has no sparsity penalty
  - A free baseline makes λ > 0 in 99.6% of neurons and RSS / sn² = 1.00, with fewer zeros (B bins 0.59). This was run as a sensitivity check (below)
- **Example traces (plot (a)):** each fluorescence transient becomes a burst of spikes at its onset, and the slow decay disappears. For the GCaMP6s Pvalb neuron, a slow plateau becomes a dense train of spikes

### Task 2: does the genetic label still help on deconvolved Allen? (`results/step15/gains.csv`, `survival.csv`, `label_vs_summary.csv`; plots (b) and `step15_mice.png`)
Mean gain over no label across units; units up = mice (Allen, 46) or sessions (Bugeon, 17) where the variant beats no label; p = Wilcoxon of variant vs none:

| | Allen dF/F (10 Hz) | Allen deconvolved A (10 Hz) | Allen deconvolved B (4.3 Hz) | Bugeon (deconvolved, 4.3 Hz) |
|---|---|---|---|---|
| score, none: ridge b / MLP m1 | 0.300 / 0.404 | 0.436 / 0.444 | 0.404 / 0.420 | 0.496 / 0.509 |
| **label gain, ridge (d − b)** | **+0.0252** (40/46, p 1e-7) | +0.0070 (39/46, p 3e-7) | +0.0112 (40/46, p 8e-9) | +0.0014 (9/17, p 0.26) |
| **label gain, MLP (m2 − m1)** | **+0.0239** (43/46, p 2e-7) | +0.0029 (39/46, p 1e-4) | +0.0077 (41/46, p 4e-7) | +0.0018 (14/17, p 2e-4) |
| **summary gain, ridge (km16 − b)** | **+0.0473** (43/46) | +0.0082 (40/46, p 2e-7) | +0.0107 (35/46, p 6e-6) | +0.0133 (17/17) |
| **summary gain, MLP (cont − m1)** | **+0.0436** (44/46) | +0.0118 (44/46, p 3e-12) | +0.0200 (44/46, p 8e-11) | +0.0174 (17/17) |
| shuffled label, ridge (e − b) | −0.0059 (7/46) | −0.0012 (7/46) | −0.0020 (8/46) | −0.0001 (3/17) |
| shuffled label, MLP (m3 − m1) | +0.0022 (20/46, p 0.75) | −0.0006 (18/46, p 0.08) | −0.0034 (13/46, p 2e-5) | +0.0000 (10/17, p 0.75) |
| summary − label, ridge | +0.0221 (39/46, p 4e-7) | +0.0011 (33/46, p 0.018) | −0.0005 (22/46, p 0.77) | +0.0119 (17/17) |
| summary − label, MLP | +0.0197 (42/46, p 4e-8) | **+0.0090** (43/46, p 2e-12) | **+0.0123** (42/46, p 1e-9) | +0.0157 (17/17) |

**Fraction of the dF/F gain that survives deconvolution** (95% bootstrap CI over mice; Wilcoxon of the deconvolved vs dF/F per-mouse gain, 46 mice; "higher" = mice where the deconvolved gain is larger):

| Gain | Family | Version | Surviving | Mice higher | Wilcoxon p | Share of the Allen dF/F − Bugeon gap removed |
|---|---|---|---|---|---|---|
| label | ridge | A | **0.28** [0.22, 0.35] | 8/46 | 1.1e-6 | 0.76 [0.67, 0.88] |
| label | ridge | B | **0.45** [0.37, 0.56] | 9/46 | 3.8e-5 | **0.59** [0.47, 0.70] |
| label | MLP | A | **0.12** [0.03, 0.21] | 4/46 | 8.3e-8 | 0.95 [0.87, 1.09] |
| label | MLP | B | **0.32** [0.21, 0.51] | 10/46 | 2.3e-5 | **0.73** [0.56, 0.84] |
| summary | ridge | A | 0.17 [0.13, 0.21] | 2/46 | 1.8e-11 | 1.15 [1.03, 1.34] |
| summary | ridge | B | 0.23 [0.16, 0.30] | 5/46 | 1.4e-10 | 1.08 [0.94, 1.29] |
| summary | MLP | A | 0.27 [0.22, 0.34] | 3/46 | 1.3e-9 | 1.21 [1.05, 1.58] |
| summary | MLP | B | 0.46 [0.40, 0.56] | 3/46 | 1.0e-9 | 0.90 [0.77, 1.13] |

**Label gain relative to the summary gain of the same signal** (how much of what per-neuron conditioning can give at all the genetic label gives; bootstrap over units):

| | Allen dF/F | Allen A | Allen B | Allen A, free baseline | Bugeon |
|---|---|---|---|---|---|
| ridge | 0.53 [0.41, 0.65] | 0.86 [0.66, 1.15] | 1.05 [0.82, 1.48] | 1.09 [0.88, 1.38] | 0.11 [−0.07, 0.24] |
| MLP | 0.55 [0.37, 0.70] | 0.24 [0.06, 0.39] | 0.39 [0.25, 0.51] | – | 0.10 [0.08, 0.12] |

- **In absolute score units, deconvolution removes most of the Allen label gain:**
  - Only 12–45% of it survives
  - With the Bugeon-like version B, the Allen–Bugeon difference in label gain shrinks by 59% (ridge) to 73% (MLP)
  - Every surviving fraction is clearly below 1 (p ≤ 4e-5); 36–42 of 46 mice lose gain
  - The deconvolved label still helps significantly (39–41/46 mice), unlike Bugeon's ridge label
- **But deconvolution removes the activity-summary gain just as much or more:**
  - Only 17–46% survives, and the deconvolved Allen summary gain falls to or below Bugeon's (gap share 0.90–1.21)
  - So the loss is not specific to the genetic label: per-neuron conditioning of any kind helps less on deconvolved Allen traces
- **Normalised by the summary gain, the label in deconvolved Allen is still 2–10× stronger than in Bugeon:**
  - MLP: 0.24–0.39 vs 0.10
  - ridge: 0.86–1.05 vs 0.11
  - So the part of the Allen–Bugeon difference that is specific to the label survives deconvolution
- **Shuffled-label controls stay ≤ 0** on deconvolved data (ridge e and MLP m3), so the deconvolved label gain is not an effect of grouping alone

### Task 2: per Cre line (`results/step15/line_gains.csv`; plots (c) and `step15_lines.png`)
Label gain, mean over the line's mice. Entries are deconvolved gain (fraction surviving, mice where it beats dF/F, Wilcoxon vs dF/F):

| Line | n | Ridge dF/F | Ridge A | Ridge B | MLP dF/F | MLP A | MLP B |
|---|---|---|---|---|---|---|---|
| Cux2 (6f) | 13 | +0.014 | +0.002 (0.15; 1/13, p 0.017) | +0.003 (0.22; p 0.013) | +0.013 | +0.002 (0.18; 0/13, p 2e-4) | +0.006 (0.44; 0/13, p 2e-4) |
| Sst (6f) | 12 | +0.017 | +0.005 (0.29; p 0.08) | +0.009 (0.56; p 0.20) | +0.000 | −0.003 (p 0.42) | +0.001 (p 0.97) |
| Vip (6f) | 12 | +0.023 | +0.010 (0.43; p 0.034) | +0.016 (0.69; p 0.30) | +0.023 | +0.007 (0.29; 0/12, p 5e-4) | +0.011 (0.48; p 0.064) |
| **Pvalb (6s)** | 9 | **+0.056** | +0.013 (**0.23**; 1/9, p 0.008) | +0.019 (0.34; 1/9, p 0.008) | **+0.072** | +0.006 (**0.08**; 0/9, p 0.004) | +0.015 (0.20; 0/9, p 0.004) |

- **Pvalb, the GCaMP6s line, had the largest label gain on dF/F and loses the most:** with the MLP only 8% (A) and 20% (B) survive, and all 9 mice drop
- So most of Pvalb's dF/F label gain was indicator kinetics: a slow-decay model that the label lets the network switch on
- Its activity summary also collapses (MLP +0.093 → +0.013 A / +0.032 B)
- The GCaMP6f lines keep more: Vip 29–69%, Cux2 15–44%
- Summary − label on deconvolved data, per line:
  - MLP: positive in every line (e.g. Pvalb A +0.007, 8/9 mice; B +0.017, 7/9)
  - ridge: near zero or negative (Sst B −0.007, 2/12 mice)

### Task 3: cell-type typing after deconvolution (C1) (`results/step15/classifier_summary.csv`, `confusions.csv`, `feature_importance.csv`, `feature_separability.csv`, `features_by_line.csv`, `neuron_predictions.csv`; plots (d) and `step15_typing.png`)
Step 11 S classifier (single-neuron features; logistic regression or gradient boosting chosen by inner CV over training mice, with logistic regression chosen in all folds of all datasets), 46 held-out mice:

| Signal | Features | Balanced accuracy (neurons) | Per fold | Sessions (vote) | Recall Cux2 / Sst / Vip / Pvalb |
|---|---|---|---|---|---|
| dF/F (full precision) | 37 | 0.868 | 0.878 ± 0.009 | 45/46 | 0.90 / 0.85 / 0.86 / 0.86 |
| **deconvolved A** | 37 | **0.916** | 0.924 ± 0.008 | **46/46** | 0.91 / 0.91 / 0.90 / **0.95** |
| deconvolved B | 34 | 0.894 | 0.899 ± 0.015 | 46/46 | 0.90 / 0.89 / 0.88 / 0.91 |

Confusion after deconvolution (A, held-out neurons; rows = true; columns = predicted Cux2 / Sst / Vip / Pvalb):
- Cux2: 2,642 / 160 / 90 / 13
- Sst: 6 / 111 / 3 / 2
- Vip: 5 / 0 / 137 / 11
- Pvalb: 0 / 1 / 5 / **110**

On dF/F, Pvalb was 5 / 2 / 9 / 100.

- **Pvalb is not harder to type once GCaMP6s kinetics are removed; it is easier.** Recall goes from 0.86 to 0.95 (A) and 0.91 (B), and all 9 Pvalb sessions are voted right
- **Which features change most** (Kruskal–Wallis ε² over the 4 lines, and Pvalb-vs-rest AUC, dF/F → A):
  - **Gain separability:**
    - 10 Hz z std: ε² 0.02 → 0.17, Pvalb AUC 0.45 → 0.02
    - power 1–2 Hz: 0.06 → 0.18, AUC 0.31 → 0.05
    - time above z = 2: 0.14 → 0.26, AUC 0.73 → 0.97
    - IEI 10–30 s: 0.13 → 0.23
    - autocorrelation at 0.1 s: AUC 0.51 → 0.20
    - power 2–5 Hz: AUC 0.57 → 0.89
    - event rate: AUC 0.88 → 0.99
  - **Lose separability:**
    - IEI 3–10 s: ε² 0.04 → 0.00
    - power 0.5–1 Hz: 0.15 → 0.12
    - IEI 1–3 s: 0.08 → 0.06
    - IEI 0–0.3 s: Pvalb AUC 0.96 → 0.89
    - DSI / OSI
  - **Most important features (permutation, logistic regression):**
    - dF/F: power 0.5–1 Hz, kurtosis, skew, short IEIs
    - A: power 0.5–1 Hz, 10 Hz z std, event rate, kurtosis, power 1–2 Hz
- **Two things make deconvolved Pvalb separable:**
  - **Higher event rate, a plausibly biological difference:** median 11.7 events / min vs Cux2 3.2 and Vip 8.2. The slow indicator hides this in dF/F (4.2 vs 1.8 and 3.9)
  - **A trace of the indicator that survives deconvolution:** with Pvalb's long fitted τ, the deconvolved signal has more fast, frame-to-frame fluctuation. Its 10 Hz z std is 0.62 vs 0.68–0.76 in other lines, its autocorrelation at 0.1 s is 0.21 vs 0.36–0.44, and it has more 2–5 Hz power. Plot (a) shows the dense spike trains under the GCaMP6s plateau
- So deconvolution removes the smooth decay shape but does not make the lines, or the indicator, indistinguishable

### Task 4: Bugeon with fluorescence (`results/step15/bugeon_files.csv`)
- **Not possible: the Bugeon release holds no fluorescence or dF/F**
- Every file type in the zip (1,037 files, 38 kinds) is listed with the dataset's own description from `DatasetTypesV2.xlsx`
- The only per-frame neural signal is `frame.neuralActivity.npy` (69 block files, 1.2 GB). The dataset describes it as "Deconvolved calcium trace for each neuron/frame"
  - Its values are non-negative up to −4e-5 rounding
  - 24–79% of values per block are exact zeros
- The other per-frame files are `frame.times`, `frame.states` (cortical state) and `plane.offset_times`
- The rest is per-neuron (depth, plane, t-type, gene counts, receptive-field maps, positions), stimulus, behaviour or metadata
- There is no F, Fneu, spks-before-deconvolution or ΔF/F file, so none / true type / summary could not be rerun on Bugeon fluorescence

### Sensitivity: OASIS with a free baseline (ridge only; `A_bfree` rows in the CSVs)
- `b_nonneg=False`: the baseline is fitted freely (median −0.014 dF/F), λ > 0 in 99.6% of neurons, and the noise constraint is met
- Fewer zeros: 0.77 of 10 Hz bins
- Ridge results agree with A:
  - label gain +0.0060 (40/46; surviving 0.24 [0.19, 0.32])
  - summary gain +0.0056 (surviving 0.12)
  - summary − label −0.0005 (26/46, p 0.79)
- So the conclusions do not depend on whether the OASIS sparsity penalty is active

**Key question: how much of the difference in genetic-label gain between Allen and Bugeon is due to signal type rather than to removing the one-line-per-recording and indicator confounds? Does the activity summary still beat the genetic label on deconvolved data?**
- **In absolute score units, mostly signal type:**
  - Deconvolving Allen with Bugeon-like binning (B) removes 59% (ridge) to 73% (MLP) of the Allen − Bugeon label-gain difference
  - 10 Hz deconvolution (A) removes 76–95%
  - For the GCaMP6s line Pvalb, 80–92% of the MLP label gain disappears
- **But it is not label-specific:**
  - Deconvolution removes 54–83% of the activity-summary gain too, pushing it to or below Bugeon's level
  - What drops is how much any per-neuron conditioning helps with this signal and copy-last baseline
- **Measured relative to the summary gain, the label in deconvolved Allen still carries 0.24–0.39 (MLP) or 0.86–1.05 (ridge) of it, against 0.10–0.11 in Bugeon**
  - This label-specific difference does not come from the signal type
  - It is consistent with the one-line-per-recording / indicator confounds: the lines remain perfectly separable by recording, and Pvalb by its residual kinetics
- **Does the summary still beat the label on deconvolved data?**
  - MLP: yes, clearly. A: +0.0090, 43/46 mice, p 2e-12. B: +0.0123, 42/46, p 1e-9
  - ridge with k-means-16: no longer. A: +0.0011, 33/46, p 0.018. B: −0.0005, 22/46, p 0.77. Free baseline: −0.0005, p 0.79
  - So the continuous summary keeps its advantage; the per-cluster ridge only ties the label

## Plots
- `plots/step15_signal.png`:
  - (a) example traces, first 60 s of the test part, dF/F vs OASIS-deconvolved, one neuron per Cre line. The neuron is the one with the largest train-part dF/F 99th percentile among the line's mice
  - (b) label gain and summary gain (ridge and MLP) for Allen dF/F / deconvolved A / deconvolved B / Bugeon, mean ± sem over units
  - (c) per-line label gain per mouse, dF/F vs A vs B, ridge and MLP, with Wilcoxon p vs dF/F
  - (d) S-classifier confusion matrices after deconvolution (A and B)
- `plots/step15_mice.png`: per-mouse gain on deconvolved (A, B) vs dF/F for label and summary, ridge and MLP, with the fraction surviving and Wilcoxon p
- `plots/step15_typing.png`:
  - confusion matrices for dF/F / A / B
  - per-line recall
  - per-feature Kruskal–Wallis ε² and Pvalb-vs-rest AUC, dF/F vs A
- `plots/step15_sparsity.png`:
  - fraction of zeros (Allen at 30 Hz / A / B per line vs Bugeon per type, with the Bugeon per-session range)
  - OASIS decay constants per line
  - sparsity vs decay per neuron
- `plots/step15_lines.png`: per-line summary gain and shuffled-label control, dF/F vs A vs B

Numbers behind the plots (`results/step15/`, 4.3 MB):

| File | Contents |
|---|---|
| `neuron_scores.csv` | per neuron and dataset (dff, A, B, A_bfree): test SSE and score of every variant (r_none, r_true, r_e, r_k1, r_km16, m_none, m_true, m_m3, m_cont) and copy-last |
| `mouse_scores.csv`, `bugeon_session_scores.csv` | unit scores of every variant |
| `gains.csv` | main table: gain per dataset, family and gain type, with sem, relative gain, units up, Wilcoxon |
| `survival.csv` | fraction of the dF/F gain surviving (bootstrap CI), Wilcoxon deconvolved vs dF/F, share of the Allen − Bugeon gap |
| `label_vs_summary.csv` | label gain ÷ summary gain per dataset |
| `line_gains.csv` | per Cre line: every gain, survival and Wilcoxon |
| `classifier_summary.csv`, `confusions.csv`, `feature_importance.csv`, `feature_separability.csv`, `features_by_line.csv`, `neuron_predictions.csv` | task 3 |
| `oasis_params.csv`, `sparsity.csv`, `example_traces.csv` | task 1 (per-neuron g, τ, b, λ, sn, RSS / sn², zero fractions; Bugeon per type and session) |
| `bugeon_files.csv` | task 4 file inventory |
| `reproduction_check.csv`, `ridge_alphas.csv`, `mlp_fits.csv` | checks, alphas, MLP epochs / val loss / time |

## Problems / errors
- **Bugeon has no fluorescence** (task 4), so the direct test (the Bugeon label gain on dF/F) is not possible with this release
- **The scores are relative to a different copy-last baseline on each signal.** Copy-last is much worse on spiky deconvolved targets: score with no label is 0.30 / 0.40 on dF/F against 0.44 / 0.44 on A. So absolute gains are not on a common scale across signals or datasets. The label ÷ summary ratio is the comparison least affected by this
- **Deconvolution is not causal.** OASIS places a spike from the fluorescence rise that follows it, so the last input bins can carry a little information about the frames just after the window (mostly a frame or two at 30 Hz, more where pools merge). The Bugeon data have the same property. Version B at 4.3 Hz smooths this over 7 frames
- **The OASIS noise constraint is inactive for 75% of neurons with the default b ≥ 0** (λ = 0). The free-baseline sensitivity run (ridge) gives the same conclusions
- **Allen B is not exactly Bugeon-like:**
  - **Sparsity:** 73% zero bins vs Bugeon's 40% (27–79% by session)
  - **Plane timing:** Bugeon's 4.3 Hz is a 7-plane volume rate with per-plane offsets, while B is a 7-frame mean of a single plane
  - **Pipeline:** Bugeon's deconvolution method and preprocessing are not stated in the release and were not checked
  - **Labels:** the lines remain one per recording
- **k = 16 was not re-chosen for the deconvolved data.** It is step 13's dF/F val choice. A different k might give a larger ridge summary gain there. The MLP continuous summary needs no k
- **The B summary differs from Bugeon's:** it uses step 11's Allen S features with 4.3 Hz lags and bands (34 features), not step 12's Bugeon S features (39, other stimuli)
- **Units differ:** Bugeon has 17 sessions of 4 mice; Allen has 46 mice. The gap bootstrap resamples both, but the Bugeon session-level CIs ignore the mouse structure (steps 12–13)
- **dF/F S classifier:** on full-precision features it gets 45/46 session votes (one Pvalb session wrong) against 46/46 in step 11 on the rounded CSV features. The balanced accuracy, 0.868, equals step 11's full-precision value
- **MLP epoch cap:** of the dF/F fold-0 refits, 1 m1 and 2 m2 reached the 40-epoch cap. These are the same fits as steps 9–11. None of the 140 A / B fits did
- **Environment:**
  - figshare rejects curl's default user agent (HTTP 500), so a browser user agent was used
  - `pip install torch` from the PyTorch index was blocked; the PyPI wheel was used, run on CPU
  - setuptools ≥ 70 lacks `pkg_resources`, which allensdk needs, so it was pinned below 70
- **Code bug found and fixed while running:** the B feature names first included each session's exact frame rate, which broke stacking. The names now use the nominal 4.3 Hz, as step 12. No reported number was affected

### Code and ideas from other work
- **This repo:**
  - Steps 3 / 4 / 5: windows, stimulus features, `load_session`, `downsample`
  - Step 8: sessions, folds, shuffled session groups
  - Step 9: MLP, `Sampler`, `gather`, m3 groups
  - Step 10: extract, `session_vote`
  - Step 11: `s_features`, `fit_featset`, `bacc`, `confusion`
  - Step 12: `neuron_stats`, `assemble`, `fit_weighted`, `balanced_weights`, `predict`, `wil`, Bugeon S lags / bands
  - Step 13: `allen_gram`, `test_data`, `fit_group`, `fit_clusters`, `CondAllen` continuous FiLM, `AllenSampler`, `forecast`
- **OASIS:** Friedrich, Zhou & Paninski 2017, PLoS Comput Biol 13: e1005423 (Friedrich & Paninski 2016, NIPS). The `oasis-deconv` package (github.com/j-friedrich/OASIS) was used as a library: `deconvolve`, `oasisAR1`, `GetSn`. Its Cython source was read to confirm how λ, b and the AR(1) kernel carry over from the constrained fit to `oasisAR1`
- **suite2p** (Pachitariu et al. 2017) is mentioned only for comparison: its OASIS step also runs without a sparsity penalty. It was not run
- **Allen Brain Observatory** (de Vries et al. 2020): the dataset, and the reporter per line
- **Bugeon et al. 2022:** the dataset and its `DatasetTypesV2.xlsx` variable table
- No code was copied from other repositories

## What I noticed
- Deconvolving Allen shrinks the genetic-label gain to roughly the Bugeon range in absolute terms, most strongly for the GCaMP6s line. But it shrinks the label-free summary gain just as much, so "signal type" mainly changes how much any per-neuron conditioning can help. It does not specifically erase type information: the lines are even easier to classify from deconvolved activity (0.92 vs 0.87).
- Relative to what conditioning can give, the label in Allen is still several times stronger than in Bugeon. That residue fits the recording / indicator confound, and part of the indicator survives deconvolution as excess fast fluctuation in Pvalb.
