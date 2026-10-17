# Step 10: Can a cell type inferred from activity replace the true Cre label?
Date: 2026-10-16

## What was run
- `scripts/step10_inferred.py` with phases `extract`, `ridge`, `mlp`, `classify` and `forecast`, then `scripts/step10_plots.py`, which remakes every plot from the CSV files in `results/step10/` only
- Environment: Python 3.10.20 (`.venv`), PyTorch 2.14.1 on CPU, scikit-learn 1.7.2, allensdk 2.16.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, statsmodels 0.14.4
- Data: `data/cache` did not exist, so it was rebuilt with step 8's `extract` (Allen S3 mirror). The step 10 `extract` phase calls `step8_medium.extract` unchanged. It wraps `step5_shared.load_session` so that, while each NWB is still on disk, it also saves `data/cache/<id>_raw.npz`. That file holds per-neuron raw dF/F mean, std, median and 99th percentile over the train part (first 80% of frames). The step 8 cache keeps only z-scored traces, which have lost the dF/F scale
- Same 46 sessions, 5 folds (round-robin per line), chronological 80/20 split, z-score from the train part, 10 Hz bins, past 30 → next 10 bins, stimulus features and test windows (last 20%, stride 10) as steps 8 and 9
- Reused code:
  - Ridge b, d and e: `process_cached`, `fit_grouped`, `fit_grouped_balanced` and `shuffled_groups` from step 8, `fit_balanced` from step 7, `predict` from step 5. Same seeds and draws as step 8 `fit`
  - MLPs m1, m2 and m3: step 9's `load`, `make_jobs` (m4 dropped) and `run_job`. Same seeds, sampling, line balance, early stopping and network (510 → 256 → 256 → 10, FiLM dim 16). To keep the trained weights, `step9_mlp.MLP` is replaced by a subclass that registers itself, so `run_job` itself is unchanged
  - These reproduce steps 8 and 9 exactly:
    - Ridge, fold-pooled score: b 0.338 ± 0.031, d 0.351 ± 0.042; d beats b in 40/46 mice; same alphas as step 8
    - MLP: m1 0.426, m2 0.440, m3 0.426; m2 beats m1 in 43/46 mice; the same 6 fits hit the 40-epoch cap as in step 9
- **Task 1: cell-type classifier**, fitted per fold on the training mice only (35–39 mice, 2,449–2,732 neurons), tested on the held-out mice
  - (a) 42 hand features per neuron, from the train part (first 80%) of each recording only:
    - Raw dF/F mean, std, median and p99
    - Std of the 10 Hz z-trace, skew and kurtosis
    - Event rate and fraction of time above z = 2 (event = upward crossing of z = 2)
    - Inter-event-interval histogram with 6 log bins, 0.3–30 s
    - Autocorrelation at 0.1, 0.2, 0.5, 1, 2 and 5 s
    - Log fraction of Welch power in 5 bands (0–0.1, 0.1–0.5, 0.5–1, 1–2 and 2–5 Hz)
    - Mean z during gratings, natural movies and spontaneous activity, and grating on minus blank
    - Direction tuning: mean z for each of 8 directions minus blank, tuning peak, DSI and OSI (vector sums)
    - Correlation with the mean of the other neurons in the recording (leave-one-out)
  - (b) m1 embedding: the mean activity of the 2nd hidden layer (256 units) of the fold's m1 (seed 0) over all train-part windows of the neuron (stride 10)
  - (a+b): both
  - Classifiers:
    - Multinomial logistic regression: quantile → normal transform, then LogisticRegression with C ∈ {0.01, 0.1, 1}
    - Histogram gradient boosting: 200 iterations, lr 0.05, 15 leaves, L2 1.0
    - Both use `class_weight="balanced"`, i.e. each Cre line gets the same total weight
  - Model selection: a 4-fold inner CV over the training mice of each outer fold (StratifiedGroupKFold, groups = mice) picks C and the "best classifier" (out of 3 feature sets × 2 classifiers). The criterion is neuron-level balanced accuracy. The held-out mice are never used for selection
  - Session label = majority vote of the neuron labels in a recording; a tie would go to the highest mean probability, but none occurred
- **Task 2: forecasting with the inferred label.** Training always uses the true label; only the label given at test time changes
  - Ridge:
    - b: no label
    - d-true: true line's model
    - d-pred-neuron: each neuron gets the per-line model of its own predicted line
    - d-pred-session: the model of the session vote
    - e: shuffled, 5 seeds
  - MLP:
    - m1: no label, 3 seeds
    - m2-true, m2-pred-neuron, m2-pred-session: the same trained m2 networks (3 seeds) with different test labels
    - m2-soft: the FiLM embedding is the class probabilities × the embedding table, so γ and β are the probability-weighted mix of the line γ and β
    - m3: shuffled, 5 seeds
  - Labels come from the classifier selected per fold by inner CV
  - Check: m1 and m2-true, recomputed from the saved weights, match the `run_job` test SSE to < 0.1%
- **Task 3: statistics.** The score per held-out mouse is 1 − SSE / SSE_copylast (mean over seeds). A paired two-sided Wilcoxon signed-rank test over mice (scipy; zero differences dropped) is run for all lines and per line. Fold-pooled scores (mean ± std over 5 folds, as in steps 8 and 9) are also given
- **Task 4: fraction of the gain recovered** = mean over mice of (pred − none) / mean over mice of (true − none), in score units, with a 95% bootstrap CI over mice (2,000 resamples)
- Wall time on 4 CPU cores: extraction ~50 min (one curl reset, then resumed); ridge 88 min (step 8's per-neuron model a, 1 BLAS thread per worker); MLP 90 min (55 fits); classifier 6 min; forecasting 3 min

### What came from LOLCAT and other sources
- **LOLCAT** (Schneider et al. 2023, Cell Reports). Code: `git clone https://github.com/nerdslab/lolcat` worked; the files were read but not run. The paper itself was not read
  - Used:
    - The question: predict the Cre line of a single neuron from its activity alone, then check it on held-out animals
    - The same 4-class setup as its Allen ophys dataset (`lolcat/ophys_dataset.py`: Pvalb, Vip, Sst, and excitatory; our excitatory class is Cux2 only)
    - Separate features per stimulus condition (drifting gratings, natural movies, spontaneous), which LOLCAT also uses
    - Inter-event-interval (IEI) distributions as features (`_compute_isi_distribtuion`). Ours is a 6-bin log histogram of intervals between z > 2 crossings of the 10 Hz dF/F; LOLCAT uses Allen's L0 events at 30 Hz, binned per 3 s snippet
    - Class-balanced training (its `DynamicBalancedSampler`); here done with balanced class weights
  - Not used: the LOLCAT network (MLP encoder per 3 s snippet + multi-head global attention pooling over the set of snippets, PyTorch Geometric). Its data splits (`calcium_splits.npy`) and "good cells" lists were not available in the repo
  - So the classifier is a "simple version" of the LOLCAT idea: hand features + logistic regression / gradient boosting, not the attention model
- **Step 9 / POCO:** the m1 embedding feature (b) and the soft FiLM (m2-soft) build on the step 9 FiLM network (POCO-style, zero-initialised). Mixing the embeddings by class probability is our own choice
- **POYO+ / NuCLR** (as in step 9, not read here): the idea of decoding cell type from a forecasting network's internal activity motivated feature set (b)
- Standard tools: scikit-learn (LogisticRegression, HistGradientBoostingClassifier, QuantileTransformer, StratifiedGroupKFold, permutation_importance) and scipy (Welch PSD, skew, kurtosis, Kruskal–Wallis, Wilcoxon)

## Data
- Experiment ids: the same 46 sessions as steps 8 and 9, one per mouse, VISp, three_session_A
- Cre line / area / depth: Cux2-CreERT2 13 mice (175 / 275 µm), Sst-IRES-Cre 12 (265–375 µm), Vip-IRES-Cre 12 (175 / 275 µm), Pvalb-IRES-Cre 9 (175–375 µm); all VISp
- Neurons: 3,296 in total (Cux2 2,905, Sst 122, Vip 153, Pvalb 116)
- Frames / frame rate / duration: 30.07–30.15 Hz, ~64 min per session → ~10 Hz bins. Classifier features use only the first 80% of each recording; forecasts are scored on the last 20%

## Results

### Task 1: classifier (held-out mice)
Neuron-level balanced accuracy (pooled over the 46 held-out mice, each neuron predicted once) and session-level accuracy (majority vote over each recording). Chance = 0.25:

| Features | Classifier | Inner-CV bal. acc (mean of 5 folds) | Held-out bal. acc, neurons (pooled) | Held-out bal. acc, per fold | Session label correct |
|---|---|---|---|---|---|
| a (hand) | logreg | 0.864 | 0.902 | 0.909 ± 0.028 | 46/46 |
| a (hand) | gboost | 0.823 | 0.881 | 0.894 ± 0.044 | 46/46 |
| b (m1 embedding) | logreg | 0.876 | 0.900 | 0.910 ± 0.021 | 46/46 |
| b (m1 embedding) | gboost | 0.846 | 0.892 | 0.889 ± 0.017 | 46/46 |
| a+b | logreg | **0.890** | **0.925** | **0.933 ± 0.009** | 46/46 |
| a+b | gboost | 0.877 | 0.921 | 0.920 ± 0.031 | 46/46 |
| **selected per fold** (inner CV) | a+b logreg (folds 0, 1, 3), a+b gboost (folds 2, 4) | — | **0.925** | — | **46/46** |

Confusion matrix of the selected classifier (held-out neurons; rows = true line, columns = predicted):

| true \ pred | Cux2 | Sst | Vip | Pvalb | recall |
|---|---|---|---|---|---|
| Cux2 (2,905) | **2,862** | 29 | 9 | 5 | 0.985 |
| Sst (122) | 13 | **108** | 1 | 0 | 0.885 |
| Vip (153) | 7 | 0 | **139** | 7 | 0.908 |
| Pvalb (116) | 2 | 0 | 7 | **107** | 0.922 |

- Session level: all 46 recordings get the right label from every feature set and classifier (the 4×4 session confusion matrix is diagonal: 13, 12, 12, 9). The lowest share of correct neurons in a recording is 0.73 (an Sst and a Vip mouse), so no vote was close
- **The session label is much easier than the single-neuron label.** Each Allen recording contains only one Cre line, so the vote pools 3–353 neurons that all share the answer, and it can use anything common to a recording: reporter / expression level, imaging depth, neuron density, and how correlated the population is. It is not evidence that every single neuron is typed correctly
- Per mouse, the fraction of neurons labelled correctly:

  | Line | median | min |
  |---|---|---|
  | Cux2 | 0.99 | 0.92 |
  | Sst | 0.88 | 0.73 |
  | Vip | 0.93 | 0.73 |
  | Pvalb | 0.94 | 0.78 |

  In 14/46 mice every neuron is labelled correctly
- The most useful hand features (held-out permutation importance; the full table is in `results/step10/features_summary.csv`):
  - logreg: power in 0.5–1 Hz, skew, kurtosis, dF/F std, short IEIs (< 0.3 s), dF/F median, autocorrelation at 1 s, correlation with the population mean
  - gradient boosting: correlation with the population mean (by far the largest: 0.15 drop in balanced accuracy), then short IEIs and kurtosis
  - Per line:
    - Cux2: sparse, very skewed traces (median skew 9, kurtosis 144) with low population correlation (median 0.17)
    - Interneurons: less skewed and more correlated with their population (median Sst 0.43, Vip 0.52, Pvalb 0.56)
    - Vip and Pvalb: slow autocorrelation (median 0.39 and 0.36 at 1 s) and many short IEIs
    - Vip: the largest dF/F std and p99
- Feature set (b) alone, the m1 embedding, is as good as the 42 hand features (0.900 vs 0.902), and together they are best. So the shared MLP does hold cell-type information in its hidden activity. Step 9 still showed that m1 cannot use it as well as an explicit label

### Task 2: forecasting with the inferred label
Held-out score, fold-pooled as in steps 8 and 9 (mean ± std over 5 folds); in brackets the mean of per-mouse scores:

| Model | Cux2 (13) | Sst (12) | Vip (12) | Pvalb (9) | All lines (46) |
|---|---|---|---|---|---|
| ridge b (no label) | 0.339 ± 0.035 | 0.384 ± 0.023 | 0.274 ± 0.041 | 0.192 ± 0.102 | 0.338 ± 0.031 (0.300) |
| ridge d-true | 0.351 ± 0.045 | 0.403 ± 0.024 | 0.294 ± 0.024 | 0.241 ± 0.074 | 0.351 ± 0.042 (0.325) |
| ridge d-pred-neuron | 0.352 ± 0.045 | **0.407 ± 0.023** | **0.296 ± 0.026** | 0.227 ± 0.062 | 0.351 ± 0.041 (0.324) |
| ridge d-pred-session | 0.351 ± 0.045 | 0.403 ± 0.024 | 0.294 ± 0.024 | 0.241 ± 0.074 | 0.351 ± 0.042 (0.325) |
| ridge e (shuffled) | 0.333 ± 0.033 | 0.378 ± 0.025 | 0.269 ± 0.043 | 0.186 ± 0.105 | 0.332 ± 0.029 (0.294) |
| MLP m1 (no label) | 0.424 ± 0.030 | 0.528 ± 0.016 | 0.376 ± 0.043 | 0.293 ± 0.121 | 0.426 ± 0.029 (0.404) |
| MLP m2-true | 0.437 ± 0.030 | 0.531 ± 0.016 | 0.396 ± 0.033 | **0.353 ± 0.064** | 0.440 ± 0.027 (0.428) |
| MLP m2-pred-neuron | 0.437 ± 0.029 | 0.531 ± 0.016 | 0.396 ± 0.034 | 0.352 ± 0.062 | 0.440 ± 0.027 (0.428) |
| MLP m2-pred-session | 0.437 ± 0.030 | 0.531 ± 0.016 | 0.396 ± 0.033 | **0.353 ± 0.064** | 0.440 ± 0.027 (0.428) |
| MLP m2-soft | **0.438 ± 0.029** | **0.533 ± 0.015** | **0.398 ± 0.034** | 0.352 ± 0.066 | **0.441 ± 0.027 (0.429)** |
| MLP m3 (shuffled) | 0.423 ± 0.032 | 0.527 ± 0.015 | 0.378 ± 0.039 | 0.303 ± 0.100 | 0.426 ± 0.030 (0.406) |

- pred-session is identical to true in every mouse, because all 46 session votes are correct
- In the 72 neurons of the 4 interneuron lines that the classifier mislabels, the predicted label often gives a *better* forecast than the true label. Median per-neuron score difference, pred − true:

  | True line | Wrong neurons | MLP | Ridge |
  |---|---|---|---|
  | Sst | 14 | +0.005 | +0.048 |
  | Vip | 14 | +0.006 | +0.008 |
  | Cux2 | 43 | −0.001 | −0.001 |
  | Pvalb | 9 | −0.006 | −0.029 |

  These neurons behave like the line they are assigned to
- The largest loss is a single neuron: Pvalb mouse 710504563, neuron 0, very active and labelled Cux2. It costs that mouse −0.113 score with ridge d-pred-neuron (from 17 neurons) and −0.015 with m2-pred-neuron. It explains the low Pvalb ridge d-pred-neuron row

### Task 3: statistics (per-mouse score, paired Wilcoxon over mice)
"A > B" = number of mice where A has the higher score; ties are mice with identical scores. Full table: `results/step10/stats.csv`.

| Comparison | All lines (46): A > B, p | Cux2 (13) | Sst (12) | Vip (12) | Pvalb (9) |
|---|---|---|---|---|---|
| **Ridge** | | | | | |
| d-true vs b (reference) | 40/46, 1.3e-07 | 12/13, 0.017 | 9/12, 0.052 | 10/12, 0.0068 | 9/9, 0.0039 |
| d-pred-neuron vs b (pred vs none) | 39/46, 4.7e-07 | 12/13, 0.017 | 9/12, 0.021 | 10/12, 0.0034 | 8/9, 0.055 |
| d-pred-session vs b | 40/46, 1.3e-07 | 12/13, 0.017 | 9/12, 0.052 | 10/12, 0.0068 | 9/9, 0.0039 |
| d-pred-neuron vs d-true (pred vs true) | 17 > / 15 < / 14 ties, 0.31 | 5 / 6 / 2, 0.93 | 7 / 2 / 3, 0.021 | 4 / 3 / 5, 0.40 | 1 / 4 / 4, 0.080 |
| d-pred-session vs d-true | identical in 46/46 | identical | identical | identical | identical |
| d-pred-neuron vs e (pred vs shuffled) | 39/46, 2.5e-08 | 12/13, 0.0024 | 9/12, 0.016 | 10/12, 0.0024 | 8/9, 0.039 |
| d-pred-session vs e | 39/46, 1.1e-08 | 12/13, 0.0024 | 8/12, 0.042 | 10/12, 0.0024 | 9/9, 0.0039 |
| **MLP** | | | | | |
| m2-true vs m1 (reference) | 43/46, 1.9e-07 | 13/13, 0.00024 | 9/12, 0.57 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-pred-neuron vs m1 (pred vs none) | 42/46, 1.9e-07 | 13/13, 0.00024 | 8/12, 0.62 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-pred-session vs m1 | 43/46, 1.9e-07 | 13/13, 0.00024 | 9/12, 0.57 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-soft vs m1 | 43/46, 2.5e-08 | 13/13, 0.00024 | 9/12, 0.47 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-pred-neuron vs m2-true (pred vs true) | 21 > / 11 < / 14 ties, 0.14 | 9 / 2 / 2, 0.050 | 5 / 4 / 3, 0.31 | 5 / 2 / 5, 0.40 | 2 / 3 / 4, 0.35 |
| m2-pred-session vs m2-true | identical in 46/46 | identical | identical | identical | identical |
| m2-soft vs m2-true | 36/46, 0.00065 | 12/13, 0.0024 | 9/12, 0.034 | 10/12, 0.0068 | 5/9, 0.50 |
| m2-pred-neuron vs m3 (pred vs shuffled) | 43/46, 3.2e-07 | 13/13, 0.00024 | 9/12, 0.57 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-pred-session vs m3 | 43/46, 4.0e-07 | 13/13, 0.00024 | 9/12, 0.57 | 12/12, 0.00049 | 9/9, 0.0039 |
| m2-soft vs m3 | 43/46, 1.6e-07 | 13/13, 0.00024 | 9/12, 0.57 | 12/12, 0.00049 | 9/9, 0.0039 |

- The inferred labels beat no label and the shuffled label as clearly as the true label does, and they are never significantly worse than the true label (pred-neuron vs true: p 0.31 ridge, 0.14 MLP)
- m2-soft is slightly but significantly *better* than m2-true: +0.0009 mean score, 36/46 mice, p 0.00065
- These tests use per-mouse **score**; steps 8 and 9 tested MSE. That is why Sst ridge d-true vs b is p 0.052 here; Sst MLP m2 vs m1 stays non-significant as in step 9

### Task 4: fraction of the true-label gain recovered
Mean over mice of (pred − none) / mean over mice of (true − none), in score units; 95% bootstrap CI over mice. Full table, including the fold-pooled version: `results/step10/gain_recovered.csv`.

| Line | Ridge pred-neuron | Ridge pred-session | MLP pred-neuron | MLP pred-session | MLP soft |
|---|---|---|---|---|---|
| **All lines (46)** | **0.95** [0.73, 1.11] | **1.00** | **1.00** [0.96, 1.03] | **1.00** | **1.04** [0.99, 1.16] |
| Cux2 (13) | 1.02 [0.99, 1.07] | 1.00 | 1.03 [1.00, 1.06] | 1.00 | 1.08 [1.04, 1.12] |
| Sst (12) | 1.26 [1.04, 2.83] | 1.00 | (2.1)* | 1.00 | (7.5)* |
| Vip (12) | 1.05 [0.97, 1.28] | 1.00 | 1.00 [0.95, 1.06] | 1.00 | 1.07 [1.02, 1.14] |
| Pvalb (9) | 0.75 [0.22, 1.00] | 1.00 | 0.97 [0.89, 1.01] | 1.00 | 0.97 [0.91, 1.02] |

\* For Sst with the MLP the true-label gain is almost zero (+0.0004 score, as in step 9), so the ratio is meaningless (CI −0.5 to 2.6 and −6 to 9). The inferred-label gains are +0.0009 (neuron) and +0.003 (soft).

- In absolute score (all lines): true-label gain +0.025 (ridge) and +0.024 (MLP). Inferred per-neuron label: +0.024 and +0.024. Soft: +0.025

**Key question: what fraction of the true-label gain does the inferred label recover?**
- **All of it, within noise.** The session-majority label recovers 100% for both ridge and MLP: it is correct for 46/46 mice, so its forecasts are identical to the true-label ones. The per-neuron label recovers 95% for ridge (CI 73–111%) and 100% for the MLP (CI 96–103%). Soft FiLM labels recover 104% (CI 99–116%) and are slightly better than the true label (36/46 mice, p 0.00065)
- Where per-neuron labels lose something, it is from a few very active Pvalb neurons labelled Cux2 (ridge Pvalb 75%). Where they gain, it is from Sst / Vip neurons that act like another line (ridge Sst 126%)

## Plots
- `plots/step10_inferred.png`: (a) held-out confusion matrix of the selected classifier (counts and row fractions), (b) fold-pooled score of all ridge and MLP models (bars mean ± std over folds, lines = folds), (c) per-mouse MLP gain with the inferred vs the true label (o = pred-neuron, x = pred-session), (d) the same for ridge
- `plots/step10_classifiers.png`: held-out balanced accuracy of every feature set × classifier (neurons; dots = folds, dashes = inner CV) and session-vote accuracy
- `plots/step10_confusions.png`: neuron-level and session-level confusion matrices for all 6 feature set × classifier combinations
- `plots/step10_features.png`: permutation importance of the hand features (logreg and gradient boosting) and per-line box plots of the top 8 features
- `plots/step10_mice.png`: fraction of neurons labelled correctly per mouse vs number of neurons; per-mouse cost of per-neuron labels (m2-pred-neuron − m2-true); per-neuron score difference for correctly vs wrongly labelled neurons, per line
- `plots/step10_examples.png`: 8 held-out neurons (per line, the correctly labelled neuron with the largest true-label gain and the mislabelled neuron with the largest loss), 60 s of test trace with the m1, m2-true, m2-pred and m2-soft forecasts

Numbers behind the plots (`results/step10/`, 3.1 MB):

| File | Contents |
|---|---|
| `mice_scores.csv` | per mouse: SSE and score of every model, classifier accuracy |
| `fold_scores.csv` | fold-pooled scores |
| `neuron_predictions.csv` | per neuron: predictions and probabilities of all 6 classifiers, session votes, selected classifier |
| `neuron_features.csv` | the 42 hand features per neuron |
| `features_summary.csv` | per-line mean / std, Kruskal–Wallis H, permutation importance, logreg coefficients |
| `classifier_folds.csv` | inner-CV and held-out accuracy per fold |
| `neuron_scores.csv` | per-neuron scores of the label variants |
| `stats.csv` | all Wilcoxon tests |
| `gain_recovered.csv` | fraction of the gain recovered |
| `example_traces.csv` | the example traces |

## Problems / errors
- **Session-level vote is the easy case:** each Allen recording holds one Cre line, so a majority vote over 3–340 same-line neurons is much easier than labelling one neuron. It can also use recording-level cues that are not cell type:
  - Transgenic reporter / GCaMP expression (affects dF/F scale; dF/F std and median are among the top features)
  - Neuron density and how synchronous the imaged population is (correlation with the population mean is the top gradient-boosting feature, and interneuron recordings have few, highly correlated neurons)
  - Imaging depth
  - The 100% session accuracy, and hence the 100% recovery of pred-session, should not be read as "cell type can be read out from any neuron". A test with several lines in the same field of view (e.g. optotagged or multi-reporter data) would be needed for that
- **Neuron-level labels are not independent of the recording either:** neurons of the same recording share these recording-level cues, and the classifier is trained on whole mice. The 0.925 balanced accuracy is per neuron, but it is not the accuracy for a single neuron with no other neurons around it
- **Inner CV for feature (b):** the m1 that gives the embedding was trained on all training mice of the outer fold, including the inner-validation mice. So the inner-CV scores for (b) and (a+b) could be optimistic. In practice the inner-CV accuracies (0.85–0.89) are below the held-out ones (0.89–0.93) for every feature set, so selection was not visibly biased toward (b). For the held-out mice, the embedding comes from an m1 that never saw them
- **Class imbalance:** Cux2 has 2,905 of 3,296 neurons. Balanced class weights make each line count equally; a plain accuracy would be dominated by Cux2 (0.985 recall)
- **Events** are threshold crossings of the 10 Hz z-scored dF/F (z > 2), not Allen's L0 events, so the IEI features are coarse (0.1 s resolution)
- **Download:** one NWB download was reset (curl exit 56) and the extraction was resumed; no data was lost (curl `-C -`, cached sessions skipped). allensdk 2.16.2 needed `setuptools < 70` (for `pkg_resources`) and `statsmodels < 0.14.5`
- **Ridge phase cost:** step 8's `process_cached` also fits the per-neuron model a, which step 10 does not need. With BLAS threads oversubscribed (4 workers × 4 threads), it was very slow. It was restarted with 1 thread per worker and then took 88 min
- 6 of 55 MLP fits stopped at the 40-epoch cap (1 m1, 4 m2, 1 m3), the same fits as in step 9

## What I noticed
- In this dataset the Cre label is not needed at test time: a classifier on train-part activity (hand features + m1 embedding) gets every recording's line right and 92.5% of single neurons. Its labels recover the whole true-label gain, for ridge and for the MLP. The recording-level vote is the easy, partly confounded case, but even per-neuron labels lose nothing on average.
- Mislabelled interneurons often forecast *better* with the label they "look like" than with their Cre label, and soft (probability-weighted) FiLM labels beat the true one-hot label in 36/46 mice. This suggests the useful conditioning variable is the neuron's activity type, which mostly but not always agrees with its Cre line.
