# Step 11: Does the activity classifier recognise the neuron, or the recording?
Date: 2026-10-17

## What was run
- `scripts/step11_confound.py` with phases `extract`, `ridge`, `mlp`, `classify` (task 1), `sessions` (task 2), `forecast` and `assign` (task 3) and `windows` (task 4). Then `scripts/step11_plots.py`, which remakes every plot from the CSV files in `results/step11/` only
- Environment: Python 3.10.20 (`.venv`), PyTorch 2.14.1 on CPU, scikit-learn 1.7.2, allensdk 2.16.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, statsmodels 0.14.4 (same versions as step 10)
- Data: `data/cache` did not exist. It was rebuilt with step 10's `extract` phase (Allen S3 mirror, unchanged), which also writes the per-neuron raw dF/F statistics
- Same 46 sessions, 5 folds, seeds, chronological 80/20 split, z-scoring, 10 Hz bins, past 30 → next 10 bins, stimulus inputs and test windows as steps 8–10
- Reused code:
  - Ridge b, d and e: step 10's `ridge` (step 8 functions, seeds and draws). Step 8's per-neuron model a is never used by b, d or e, so it is skipped. `process_grams` builds the same Gram sums, in the same order, and the same val windows as `step5_shared.process_session`
  - MLP m1, m2 and m3: step 10's `mlp` (step 9 `run_job`, same seeds), including the m1 seed-0 embedding (feature set b)
  - Classifiers: step 10's `make_clf` (quantile → logistic regression with C ∈ {0.01, 0.1, 1}; histogram gradient boosting), balanced class weights, and a 4-fold inner CV over the training mice (StratifiedGroupKFold, groups = mice, seed = fold). Per feature set, the inner CV picks C and the classifier in each outer fold
  - Forecasting helpers: `mlp_forecast`, `session_vote`, `compare` (Wilcoxon) and `recovered` (bootstrap) from step 10
- **Reproduction of step 10** (all checks passed):
  - Ridge alphas are identical to step 8: b 0.1; d 0.1 for Cux2, Vip and Pvalb; Sst 3.2e5 or 1e6
  - Per-mouse scores of b, d-true, e, m1, m2-true and m3 match `results/step10/mice_scores.csv` to ≤ 1e-6 in all 46 mice (`results/step11/reproduction_check.csv`). The mean scores are b 0.2997, d 0.3249, e 0.2938, m1 0.4039, m2 0.4279 and m3 0.4061
  - Per-neuron m1, m2-true and m3 SSE recomputed from the saved weights equal step 9's `run_job` evaluation to < 0.1%. The same 6 MLP fits hit the 40-epoch cap as in steps 9 and 10
  - m1 embedding: feature set b gives 0.900 (logreg) and 0.892 (gboost), as in step 10. A+b selected gives 0.925, as in step 10 (1 neuron of 3,296 differs)
  - Labels from step 10's selected classifier, scored through the new CSV, recover 0.95 (ridge) and 1.00 (MLP) of the gain, as in step 10
  - Rounding caveat: `neuron_features.csv` stores 5 significant digits. On these rounded features, A gives 0.907 / 0.887 (step 10: 0.902 / 0.881) and S gives 0.877. With full-precision features recomputed from the cache, A gives exactly step 10's 0.902 / 0.881 and S gives 0.868 (task 4, full train part). So rounding alone moves balanced accuracy by up to ~0.01. Differences smaller than this should not be read as real
- Wall time on 4 CPU cores:

  | Phase | Time |
  |---|---|
  | extract | ~22 min |
  | ridge | 77 min, almost all of it in Gram sums over non-contiguous window views (kept to stay bit-identical to step 10) |
  | mlp | 81 min (55 fits) |
  | classify | 9 min |
  | sessions | 3 min |
  | forecast | ~3 min |
  | windows | 7 min |

### Tasks
- **Task 1: feature groups.** Taken from `results/step10/neuron_features.csv` (train part only). Number of neurons and imaging depth were added from the session metadata
  - **S**, single-neuron and scale-free (37 features):
    - z std (10 Hz), skew, kurtosis
    - Event rate and time above z = 2
    - IEI histogram (6 bins)
    - Autocorrelation at 6 lags
    - Power in 5 bands
    - Mean z per stimulus (3)
    - Grating on − blank
    - 8 directions, tuning peak, DSI and OSI
  - **R**, recording-level (7 features): dF/F mean, std, median and p99; correlation with the population mean; number of neurons in the session; imaging depth
  - **A**: all 42 hand features (step 10 set a)
  - **b**: m1 embedding. It sees only the neuron's own z-scored past and the stimulus, so it counts as single-neuron
  - **S+b**
  - **A+b**: step 10's best set, kept as a reproduction check
- **Task 2: session-only classifier.** One row per session, with 8 recording summaries from the train part:
  - Number of neurons and depth
  - Mean and median pairwise correlation (10 Hz z-traces)
  - Mean and median correlation with the population mean
  - Median dF/F std and median dF/F mean
  - Same classifiers and inner CV. Tested on the 5 outer folds and with leave-one-mouse-out (46 splits). Also run on subsets: n neurons + depth, correlations only, dF/F scale only. The Cre reporter of each session was read from `ophys_experiments.json`
- **Task 3: forecasting with labels from each classifier.**
  - For every neuron, the test SSE was computed under each of the 4 line labels: ridge d with that line's model, and MLP m2 with that line's FiLM embedding (mean over 3 seeds). Also saved: no label (b, m1), the trained-shuffled controls (e, m3) and copy-last. All in `results/step11/neuron_scores_all_labels.csv`
  - Mouse score = 1 − ΣSSE / ΣSSE_copylast over the mouse's neurons, with each neuron's assigned label. So any future label assignment can be scored from this CSV alone (`assign` phase, function `mouse_scores`)
  - Label sets scored:
    - true
    - Per-neuron and session-vote labels of the S, S+b, R and A classifiers (selected per fold by inner CV)
    - Step 10's selected classifier (a+b) per neuron
    - Shuffled: the true labels permuted across all 3,296 neurons. 20 permutations (seed 0); mouse scores averaged over them
  - Fraction of the gain recovered = mean over mice of (pred − none) / mean over mice of (true − none), with a 95% bootstrap CI over mice (2,000 resamples), as in step 10
  - Paired two-sided Wilcoxon over mice: pred vs none and pred vs true
- **Task 4: window length.**
  - S features were recomputed from windows of 3 s, 30 s and 3 min of the train part: 10 non-overlapping windows per neuron, at random positions, shared by all neurons of a session. Also from the full train part (~51 min)
  - Traces stay z-scored with the full-train-part statistics, exactly as the forecasting input is
  - Features that a short window cannot define are left missing, then imputed (logreg: training median + missing indicators; gboost: native NaN handling). Examples: a stimulus that does not occur in the window, the 5 s autocorrelation in a 3 s window, the 0–0.1 Hz band
  - Each window is one sample (held-out mice as before). Also reported: a neuron-level score that averages the log-probabilities of a neuron's 10 windows

### Which ideas came from which paper
None of these papers was re-read or run in this step. The numbers quoted for NuCLR and NeuPRINT are from the task description.
- **LOLCAT** (Schneider et al. 2023, Cell Reports; code read in step 10)
  - It classifies cell type from many 3 s snippets of activity and pools them with attention. This motivated task 4: the 3 s window and the question of how much recording is needed. The "10 windows pooled per neuron" score is a crude stand-in for its snippet pooling
  - The S features (IEI histogram, per-stimulus features, class balancing) are the step 10 LOLCAT-style features
- **NeuPRINT** (Mi et al. 2023, NeurIPS)
  - It learns a time-invariant representation per neuron and gives population-level statistics to the model as separate context. That split between the neuron's own dynamics and the population / recording around it is the S vs R split here. Correlation with the population mean was moved to R for this reason
  - Its ~0.6 accuracy on mixed-type data is one of the two literature references for "typing without recording cues"
- **NuCLR**: neuron-level contrastive representation; 0.39 macro F1 for 5 inhibitory subclasses on Bugeon et al. 2022. That is mixed-type data, where several types share one field of view, so the recording cannot reveal the type. This is the main reason to suspect step 10's 92.5%. NuCLR's own method was not used
- **de Vries et al. 2020** (Nat Neurosci; the Allen Brain Observatory Visual Coding 2P paper): the dataset design. Each experiment images one Cre line, with one reporter, at one targeted depth
  - The metadata check confirms the reporters: Cux2 = Ai93 (GCaMP6f, Camk2a-tTA), Sst and Vip = Ai148 (GCaMP6f), Pvalb = Ai162 (**GCaMP6s**)
  - The neuron-count and depth differences between lines follow from this design
  - DSI / OSI here are step 10's vector-sum versions, not that paper's definitions

## Data
- Experiment ids: the same 46 sessions as steps 8–10, one per mouse, VISp, three_session_A
- Cre line / area / depth: Cux2-CreERT2 13 mice (175 / 275 µm), Sst-IRES-Cre 12 (265–375 µm), Vip-IRES-Cre 12 (175 / 275 µm), Pvalb-IRES-Cre 9 (175–375 µm); all VISp
- Neurons: 3,296 (Cux2 2,905, Sst 122, Vip 153, Pvalb 116). Neurons per recording: Cux2 140–353, Sst 7–15, Vip 4–24, Pvalb 3–32
- Frames / frame rate / duration: 30.07–30.15 Hz, ~64 min per session → ~10 Hz bins. Classifier features use the train part (first 80%, ~51 min); forecasts are scored on the last 20%

## Results

### Task 1: classifier per feature group (46 held-out mice)
Neuron-level balanced accuracy (pooled; each neuron predicted once), per-fold mean ± std, and session-vote accuracy. Chance = 0.25. "sel" = the classifier chosen per fold by inner CV (S, R, b and S+b chose logreg in all 5 folds). Full table: `results/step11/classifier_summary.csv`.

| Features | n | Classifier | Inner CV | Held-out bal. acc (neurons) | Per fold | Sessions correct | Recall Cux2 / Sst / Vip / Pvalb |
|---|---|---|---|---|---|---|---|
| **S** (single-neuron) | 37 | **sel = logreg** | 0.847 | **0.877** | 0.886 ± 0.012 | **46/46** | 0.90 / 0.87 / 0.87 / 0.87 |
| S | | gboost | 0.801 | 0.861 | 0.873 ± 0.046 | 45/46 | 0.98 / 0.82 / 0.80 / 0.84 |
| **R** (recording) | 7 | **sel = logreg** | 0.733 | **0.729** | 0.742 ± 0.123 | **36/46** | 0.97 / 0.83 / 0.69 / 0.42 |
| R | | gboost | 0.660 | 0.690 | 0.709 ± 0.183 | 32/46 | 1.00 / 0.52 / 0.67 / 0.58 |
| **A** (all 42) | 42 | **sel** (logreg 4 folds, gboost 1) | 0.868 | **0.887** | 0.902 ± 0.042 | **46/46** | 0.93 / 0.89 / 0.86 / 0.87 |
| A | | logreg | 0.867 | 0.907 | 0.915 ± 0.021 | 46/46 | 0.93 / 0.89 / 0.90 / 0.91 |
| **b** (m1 embedding) | 256 | **sel = logreg** | 0.876 | **0.900** | 0.910 ± 0.021 | **46/46** | 0.95 / 0.88 / 0.88 / 0.89 |
| **S+b** | 293 | **sel = logreg** | 0.888 | **0.913** | 0.919 ± 0.006 | **46/46** | 0.96 / 0.89 / 0.88 / 0.92 |
| A+b (step 10) | 298 | sel (logreg 3, gboost 2) | 0.895 | 0.925 | 0.927 ± 0.028 | 46/46 | 0.99 / 0.89 / 0.91 / 0.92 |

Confusion matrices (held-out neurons, selected classifier; rows = true, columns = predicted; order Cux2, Sst, Vip, Pvalb):

| true \ pred | **S**: Cux2 | Sst | Vip | Pvalb | **R**: Cux2 | Sst | Vip | Pvalb | **S+b**: Cux2 | Sst | Vip | Pvalb |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Cux2 (2,905) | **2,610** | 203 | 86 | 6 | **2,824** | 26 | 32 | 23 | **2,795** | 70 | 30 | 10 |
| Sst (122) | 12 | **106** | 2 | 2 | 2 | **101** | 12 | 7 | 10 | **108** | 2 | 2 |
| Vip (153) | 3 | 0 | **133** | 17 | 1 | 19 | **106** | 27 | 9 | 0 | **135** | 9 |
| Pvalb (116) | 3 | 2 | 10 | **101** | 1 | 28 | 38 | **49** | 3 | 1 | 5 | **107** |

The A matrix (2,700 / 165 / 32 / 8; 11 / 108 / 3 / 0; 6 / 0 / 132 / 15; 2 / 0 / 13 / 101) and all session-level matrices are in `results/step11/confusions.csv` and `plots/step11_confusions.png`.

- **Removing the recording-level features costs little.** S alone reaches 0.877, against 0.887–0.907 for A. S+b reaches 0.913, against 0.925 for A+b. Every recording still gets the right session vote from S (46/46)
- S loses mostly on Cux2: 7% of Cux2 neurons are called Sst and 3% Vip, because without the neuron count and population correlation the rare sparse-looking Cux2 neurons look like interneurons. One Cux2 mouse (645413759, 226 neurons) has only 53% of neurons right, but its vote is still right
- Interneuron recall changes little between A and S (Sst 0.89 → 0.87, Vip 0.86–0.90 → 0.87, Pvalb 0.87–0.91 → 0.87)
- Per mouse, fraction of neurons labelled right with S: median Cux2 0.95, Sst 0.88, Vip 0.91, Pvalb 0.88; minimum 0.53 (Cux2), 0.73, 0.69, 0.67; 9/46 mice fully correct
- **R alone is much weaker than S (0.729) and fails on the interneuron subclasses.** Pvalb recall is 0.42, and 10 of 46 session votes are wrong (1 Sst, 5 Vip, 4 Pvalb). R separates Cux2 from interneurons almost perfectly (Cux2 recall 0.97) through the neuron count
- Most useful features (held-out permutation importance, drop in balanced accuracy; `results/step11/feature_importance.csv`):
  - S, logreg: power 0.5–1 Hz 0.083, kurtosis 0.065, skew 0.061, short IEIs (< 0.3 s) 0.042, autocorrelation at 1 s 0.036
  - S, gboost: short IEIs 0.085, power 0.5–1 Hz 0.074, kurtosis 0.066, grating on − blank 0.037, mean z during natural movies 0.025
  - R: the number of neurons dominates (0.38 logreg, 0.44 gboost), then dF/F p99 / mean (logreg) and depth (gboost)

### Task 2: how identifiable is the recording itself?
Session-level summaries per line (median [min, max] over sessions; `results/step11/session_features.csv`):

| Summary | Cux2 (13) | Sst (12) | Vip (12) | Pvalb (9) |
|---|---|---|---|---|
| neurons | 215 [140, 353] | 10 [7, 15] | 12.5 [4, 24] | 10 [3, 32] |
| depth (µm) | 175 [175, 275] | 275 [265, 375] | 175 [175, 275] | 200 [175, 375] |
| mean pairwise corr | 0.029 [0.022, 0.114] | 0.23 [0.09, 0.31] | 0.28 [0.19, 0.40] | 0.33 [0.19, 0.44] |
| mean corr with pop mean | 0.15 [0.14, 0.33] | 0.41 [0.20, 0.47] | 0.50 [0.36, 0.60] | 0.52 [0.36, 0.62] |
| median dF/F std | 0.087 [0.061, 0.111] | 0.13 [0.09, 0.22] | 0.21 [0.10, 0.29] | 0.11 [0.06, 0.20] |
| median dF/F mean | 0.0065 [0.0031, 0.0078] | 0.017 [0.008, 0.037] | 0.054 [0.013, 0.081] | 0.024 [0.007, 0.069] |
| reporter | Ai93 (GCaMP6f) | Ai148 (GCaMP6f) | Ai148 (GCaMP6f) | **Ai162 (GCaMP6s)** |

Session-only classifier, sessions with the right Cre line (of 46; classifier selected by inner CV; full table with both classifiers: `results/step11/session_classifier.csv`):

| Session features | 5 outer folds | Leave one mouse out | Per line, 5 folds: Cux2 / Sst / Vip / Pvalb |
|---|---|---|---|
| **all 8 summaries** | **36/46** (bal. acc 0.76) | **34/46** (0.71) | 13/13, 10/12, 8/12, 5/9 |
| n neurons + depth | 24/46 | 29/46 | 11/13, 5/12, 6/12, 2/9 |
| correlations only | 27/46 | 27/46 | 12/13, 7/12, 4/12, 4/9 |
| dF/F scale only | 35/46 | 35/46 | 13/13, 8/12, 10/12, 4/9 |

- **The recording alone reveals Cux2 vs interneuron perfectly.** A single threshold on the neuron count (Cux2 ≥ 140, every interneuron recording ≤ 32) separates them 46/46. Population correlation also nearly separates them (Cux2 mean pairwise r ≤ 0.11 vs ≥ 0.09 for Sst and ≥ 0.19 for Vip / Pvalb). Even the logistic regression on n neurons + depth gets only 11/13 Cux2 sessions in the 5-fold test; it is underfitting with ~37 training rows
- **Among the 3 interneuron lines, the recording summaries are only partly informative:**
  - 23/33 interneuron sessions right with all 8 summaries (5 folds); chance for 3 lines would be ~11
  - Pvalb is the hardest (5/9)
  - The dF/F scale summaries are the most useful recording cue (35/46)
- Gradient boosting cannot fit 37–45 rows (min 20 samples per leaf) and predicts a constant in the 5-fold test (13/46). The inner CV correctly chose logreg in every split
- So the recording alone reveals the line only for the excitatory / inhibitory split (36/46 overall, vs 46/46 from neuron-level S features). Within interneurons, the neuron-level S classifier (46/46 sessions, Sst / Vip / Pvalb recall 0.87 each) does clearly better than the recording summaries

### Task 3: forecasting with labels from each feature group
Fraction of the true-label gain recovered = mean(pred − none) / mean(true − none) over the 46 mice (none = ridge b / MLP m1; true = ridge d / MLP m2 with the true line). The true-label gain is +0.0252 score (ridge) and +0.0239 (MLP). Full table, per line: `results/step11/gain_recovered.csv`.

| Labels | Ridge: recovered [95% CI] | MLP: recovered [95% CI] |
|---|---|---|
| **S per-neuron** | **1.02** [0.98, 1.10] | **0.99** [0.94, 1.10] |
| S session vote | 1.00 (= true in 46/46) | 1.00 |
| S+b per-neuron | 0.97 [0.83, 1.10] | 0.90 [0.78, 1.01] |
| S+b session vote | 1.00 | 1.00 |
| A per-neuron | 1.05 [0.98, 1.16] | 0.96 [0.89, 1.02] |
| A session vote | 1.00 | 1.00 |
| step 10 (a+b) per-neuron | 0.95 [0.73, 1.10] | 1.00 [0.96, 1.03] |
| R per-neuron | −0.84 [−2.14, 0.40] | −2.05 [−3.98, 0.44] |
| R session vote | −1.02 [−2.43, 0.34] | −2.33 [−4.22, 0.34] |
| shuffled (true labels permuted, 20×) | −1.30 [−1.75, −0.89] | −3.45 [−4.70, −2.72] |

Per line, S per-neuron labels:

| Line | Ridge [95% CI] | MLP [95% CI] |
|---|---|---|
| Cux2 | 0.96 [0.88, 1.06] | 1.01 [0.95, 1.05] |
| Sst | 1.17 [1.00, 2.36] | (ratio meaningless*) |
| Vip | 1.12 [1.03, 1.35] | 0.99 [0.93, 1.04] |
| Pvalb | 0.93 [0.86, 0.99] | 0.94 [0.86, 0.99] |

\* For Sst with the MLP the true-label gain is ~0 (as in steps 9 and 10), so the ratio is meaningless.

Paired Wilcoxon over the 46 mice (score; "A > B" = mice where A is higher; full table with per-line tests: `results/step11/stats.csv`):

| Comparison | Ridge: A > B / ties, p | MLP: A > B / ties, p |
|---|---|---|
| true vs none (reference) | 40/46, 1.3e-07 | 43/46, 1.9e-07 |
| S per-neuron vs none | 40/46, **1.6e-08** | 43/46, **3.3e-08** |
| S per-neuron vs true | 19 > / 9 ties / 18 <, **0.41** | 19 > / 9 ties / 18 <, **0.56** |
| S+b per-neuron vs none | 40/46, 3.3e-08 | 43/46, 2.3e-07 |
| S+b per-neuron vs true | 22 > / 12 ties, 0.14 | 17 > / 12 ties, 0.65 |
| A per-neuron vs none | 41/46, 1.0e-09 | 43/46, 1.6e-07 |
| A per-neuron vs true | 19 > / 13 ties, 0.26 | 18 > / 13 ties, 0.77 |
| step 10 (a+b) per-neuron vs true | 17 > / 14 ties, 0.31 | 21 > / 14 ties, 0.14 |
| R per-neuron vs none | 33/46, 0.12 | 35/46, 0.018 (mean −0.049) |
| R per-neuron vs true | 3 > / 21 ties, 2.4e-04 | 4 > / 21 ties, 4.5e-04 |
| R session vs true | 1 > / 36 ties, 0.0069 | 0 > / 36 ties, 0.0051 |
| shuffled vs none | 14/46, 2.1e-05 (worse) | 13/46, 1.6e-07 (worse) |
| shuffled vs true | 3/46, 2.5e-10 | 1/46, 1.1e-11 |

- **S labels recover the whole true-label gain:** 102% (ridge) and 99% (MLP). They beat no label as clearly as the true label does and are not distinguishable from the true label (p 0.41 and 0.56)
- Per line, S-label recovery is ≥ 0.93 everywhere. Pvalb loses ~6–7% (CI excludes 1): 15 Pvalb neurons are mislabelled, 10 of them as Vip
- In Sst and Vip with ridge, S labels again recover more than 100% (1.17, 1.12): mislabelled neurons forecast better with the line they resemble, as in step 10
- Wrong labels are costly. R labels give the wrong line to 10 sessions, and the shuffled control gives wrong labels to most interneurons. Both turn the gain negative: a wrong per-line model is worse than no label
  - This "shuffled" is a test-time permutation of the labels with the true-label models. It is a much harsher control than step 10's trained-shuffled models e / m3 (−0.006 and +0.002 score vs none), whose per-neuron scores are also in the CSV
- So the true-label gain needs labels that are *right per recording or per neuron*. R alone is not enough; S is

### Task 4: how long must a neuron be watched?
S classifier trained and tested per window length (`results/step11/window_accuracy.csv`). Single-window balanced accuracy, per-fold mean ± std; "10 windows pooled" = mean log-probability over a neuron's 10 windows:

| Window | Selected clf | Missing features | One window: bal. acc (pooled) | Per fold | Recall Cux2 / Sst / Vip / Pvalb | 10 windows pooled |
|---|---|---|---|---|---|---|
| 3 s (one forecasting window) | gboost (5/5) | 37% | **0.465** | 0.476 ± 0.022 | 0.64 / 0.34 / 0.48 / 0.41 | 0.615 |
| 30 s | gboost (5/5) | 29% | **0.645** | 0.662 ± 0.027 | 0.79 / 0.52 / 0.64 / 0.64 | 0.798 |
| 3 min | gboost (5/5) | 23% | **0.778** | 0.789 ± 0.019 | 0.89 / 0.67 / 0.74 / 0.81 | 0.921 |
| full train part (~51 min) | logreg (5/5) | 0% | **0.868** | 0.878 ± 0.009 | 0.90 / 0.85 / 0.86 / 0.86 | — |

- A single 3 s forecasting window carries some cell-type information (0.465, chance 0.25) but far too little to type the neuron: Sst recall is 0.34
- Accuracy rises steadily with window length, and a few minutes are needed to get near the full-recording value
- Pooling 10 separate 3 min windows (30 min of data) gives 0.921, better than one set of features from the full ~51 min (0.868). Training on many short windows works as data augmentation. This is in line with LOLCAT's design of pooling many short snippets
- Logreg is clearly worse than gboost for short windows (0.413, 0.535, 0.716), because many features are missing there and the median imputation is crude
- The full-train-part row uses full-precision features, so it is 0.868 rather than task 1's 0.877 from the rounded CSV (see the rounding caveat above)

**Key question: without recording-level cues, how well can single neurons be typed, and how much of the forecasting gain do those labels still recover?**
- **Typing:**
  - With only the neuron's own scale-free activity (S, 37 features from ~51 min of activity), held-out neurons are typed with 0.877 balanced accuracy (0.868 on full-precision features), down only 1–3 points from all features (0.887–0.907). With the m1 embedding added (S+b), 0.913
  - Recording-level features alone give 0.729 and only 36/46 sessions. They mainly separate Cux2 from interneurons (by neuron count); a session-only classifier gets 36/46
  - So the step 10 classifier does mostly recognise the neuron's activity pattern, not the recording
  - It needs minutes of activity: a single 3 s window gives only 0.465
- **Forecasting:** S per-neuron labels recover 102% (ridge, CI 98–110%) and 99% (MLP, CI 94–110%) of the true-label gain. They are indistinguishable from the true label (p 0.41, 0.56)
- **Caveat:** "S" removes the obvious recording cues but not all of them (see Problems). The 0.877 is still well above NuCLR / NeuPRINT on mixed-type data, so it should be read as an upper bound for single-neuron typing in this single-line dataset

## Plots
- `plots/step11_confound.png`: four panels
  - (a) held-out balanced accuracy for S, R, A, b, S+b and A+b (bars = pooled neurons, dots = folds, diamonds = session-vote accuracy)
  - (b) confusion matrix of the S classifier
  - (c) fraction of the true-label gain recovered by each label set, ridge and MLP, with 95% bootstrap CIs (axis cut at −1, lower values written in red)
  - (d) S classifier balanced accuracy vs window length (one window, and 10 windows pooled), with the R and A full-length levels
- `plots/step11_confusions.png`: neuron-level and session-vote confusion matrices for all 6 feature sets
- `plots/step11_sessions.png`: recording-level summaries per line (one dot per session), the reporter of each line, and the session-only classifier result
- `plots/step11_features.png`: held-out permutation importance of the S, R and A features (logreg and gradient boosting)
- `plots/step11_mice.png`: per held-out mouse, the fraction of neurons the S classifier labels right vs neuron count, and the forecasting gain with S labels vs true labels (ridge, MLP)

Numbers behind the plots (`results/step11/`, 2.2 MB):

| File | Contents |
|---|---|
| `classifier_summary.csv` | per feature set × classifier: inner-CV and held-out balanced accuracy, per-fold values, sessions correct, per-line recall |
| `classifier_folds.csv` | C, inner-CV accuracy and selection per fold |
| `confusions.csv` | neuron- and session-level confusion matrices of every feature set × classifier |
| `feature_importance.csv` | held-out permutation importance for S, R and A |
| `neuron_predictions.csv` | per neuron: predictions of every classifier, selected-classifier probabilities, session votes |
| `session_features.csv` | the 8 session summaries and the reporter line per session |
| `session_classifier.csv`, `session_predictions.csv` | task 2 results |
| `neuron_scores_all_labels.csv` | per neuron: test SSE and score with copy-last, b, ridge d with each line, e, m1, MLP m2 with each line, m3 (scores any label assignment from CSV) |
| `reproduction_check.csv` | step 11 vs step 10 per-mouse scores |
| `assignment_mice_scores.csv` | per mouse and family: score under each label set |
| `gain_recovered.csv` | fraction of the gain recovered, all lines and per line |
| `stats.csv` | all Wilcoxon tests |
| `window_accuracy.csv` | task 4 results |

## Problems / errors
- **S is not free of recording-level effects.** In single-line recordings, a neuron-level feature can only be separated from the recording in name, not in the data. Every neuron of a recording shares the label, so any cue shared by a recording also helps S:
  - **Indicator:** Pvalb mice express GCaMP6s (Ai162); all other lines express GCaMP6f. Slow kinetics raise the 1 s autocorrelation and the low-frequency power, two of the top S features. So part of "Pvalb-ness" may be "GCaMP6s-ness". Cux2 (Ai93, tTA via Camk2a) and Sst / Vip (Ai148) also differ in reporter construct and expression level
  - **Signal-to-noise:** z-scoring removes the dF/F scale, but noise level and expression still shape the 10 Hz z std, skew, kurtosis and event rate
  - **Shared brain state** within a recording (running, arousal) shapes everyone's dynamics. Mean z per stimulus and the power bands carry part of it
  - A clean test needs several types in the same field of view: optotagged or multi-reporter data, or a spatial-transcriptomics-matched set such as Bugeon et al. 2022, where NuCLR / NeuPRINT get 0.39–0.6
- **Session-vote results are not independent evidence:** S, A, b and S+b all give 46/46 votes, so their session-vote labels equal the true labels and recover exactly 100% by construction
- **R per-neuron predictions vary within a recording**, because dF/F and population correlation vary per neuron. That is why R neuron-level recall differs from its session vote. R contains no neuron-specific *activity pattern*, but it is not constant per recording
- **Session-only classifier:** with 37–45 training rows, gradient boosting cannot split (min 20 samples per leaf). The logreg chosen by inner CV underfits even the trivial neuron-count split (11/13 Cux2 with n neurons + depth, 5 folds). The 36/46 is therefore a lower bound on what a more flexible session classifier could do. Cux2 vs interneuron is separable at 46/46 by one threshold
- **Rounding of `neuron_features.csv`** (5 significant digits) changes held-out balanced accuracy by up to ~0.01 (A: 0.907 vs 0.902 on exact features; S: 0.877 vs 0.868). Task 1 uses the CSV as instructed; task 4's full-length row uses exact features
- **Window features:** short windows leave many features undefined (37% missing at 3 s), handled by imputation / native NaN support. The 10 windows per neuron share their positions across the neurons of a session. Windows are z-scored with full-train-part statistics, as the forecasting input is, so a "3 s" sample still uses the neuron's long-run mean and std
- **Shuffled control:** here it is a test-time permutation of the labels (true-label models), not step 10's trained-shuffled models e / m3, so its numbers are not comparable to step 10's e / m3 rows. e and m3 per-neuron SSEs are in `neuron_scores_all_labels.csv`
- **Bootstrap CIs:** the CI random draws differ from step 10's (one generator per model family here), so step 10's labels give the same point estimates (0.95, 1.00) but slightly different CIs
- **Ridge phase:** Gram sums over strided window views take ~4 s per neuron (0.06 s when contiguous). Kept unchanged so that b, d and e stay identical to step 10
- **The `forecast` phase** stopped once on a bug in the reproduction check, after the per-neuron CSV had been saved. The bug was a pandas column named `copy` clashing with `DataFrame.copy`. It was fixed, and the check and later phases were rerun from the saved CSV

## What I noticed
- Taking away every recording-level feature costs only ~1–3 points of neuron-level accuracy. S labels still recover the whole forecasting gain, while a classifier that sees only the recording mislabels 10 of 46 sessions and makes forecasts worse. In this dataset the classifier mostly reads the neuron's own activity pattern. That pattern can still include line-wide reporter and indicator differences (GCaMP6s in Pvalb), which only a mixed-type dataset could separate from cell type.
- Typing needs minutes of activity: a single 3 s window is right less than half the time, but ten 3-min windows pooled beat one long feature set. So a forecasting model cannot infer the type from its own input window. An explicit label, or a label inferred from a longer history, is what carries the gain.
