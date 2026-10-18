# Step 13: Label-free dynamics groups for forecasting (Bugeon and Allen)
Date: 2026-10-18

## What was run
- `scripts/step13_dynamics.py <dataset> <phase>` for `bugeon` and `allen`, phases `prep`, `features`, `cluster`, `ridge`, `mlp`, `collect`. Then `scripts/step13_dynamics.py both analyse`, which writes `results/step13/`. Then `scripts/step13_plots.py`, which remakes every plot from the CSV files in `results/step13/` only
- **Environment:** Python 3.11 (`.venv`, rebuilt for this step), PyTorch 2.14.1 on CPU, scikit-learn 1.7.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, allensdk 2.16.2. These are the same versions as steps 11 and 12. setuptools < 70 was needed because allensdk imports `pkg_resources`
- **Data:** neither dataset was on disk, so both were rebuilt:
  - **Bugeon:** downloaded from https://ndownloader.figshare.com/files/38046633. The MD5 is 4556ed80ce98ca13f909129e25f60f3a, which matches. It was unpacked to `data/bugeon/raw/` and step 12's `extract` was rerun. Its `neurons.csv`, `sessions.csv` and `blocks.csv` are byte-identical to `results/step12/`
  - **Allen:** `api.brain-map.org` is blocked here. `ophys_experiments.json` and the 46 NWB files came from the Allen S3 mirror, using step 11's `extract`, unchanged
  - Nothing under `data/` is committed
- **Same settings as before:** splits, folds, windows, seeds, inputs and model settings are those of step 11 (Allen) and step 12 (Bugeon):
  - Allen: 46 mice, 5 leave-mice-out folds, 10 Hz, past 30 → next 10 bins
  - Bugeon: 17 sessions of 4 mice, leave one mouse out, 4.3 Hz, past 13 → next 4 frames, 6 types
- **Wall time on 4 CPU cores:**

  | Phase | Bugeon | Allen |
  |---|---|---|
  | data (download + extract) | ~3 min | 14 min |
  | per-neuron ridge statistics | (from step 12 extract) | 7 min |
  | features / cluster | 20 s / 25 s | 20 s / 33 s |
  | ridge (all k, all variants) | 9 min | 19 min |
  | MLP | 81 min (66 fits) | 78 min (81 fits) |

### Reproduction check (`results/step13/reproduction_check.csv`)
| Dataset | Model | Compared with | Max abs. difference in unit score | Mean score here / before |
|---|---|---|---|---|
| Bugeon | ridge b (none), recomputed | step 12 | 1.2e-8 | 0.4960 / 0.4960 |
| Bugeon | ridge d (true type), recomputed | step 12 | 9.4e-9 | 0.4974 / 0.4974 |
| Bugeon | MLP m1, fold 0 refit (3 seeds) | step 12 | 4.5e-9 | identical epochs and val losses |
| Bugeon | MLP m2, fold 0 refit (3 seeds) | step 12 | 7.0e-9 | |
| Allen | ridge b, recomputed | step 11 | 1.0e-7 | 0.2997 / 0.2997 |
| Allen | ridge d (true type), recomputed | step 11 | 1.4e-7 | 0.3249 / 0.3249 |
| Allen | MLP m1, fold 0 refit | step 11 | 6.1e-8 | |
| Allen | MLP m2, fold 0 refit | step 11 | 4.3e-8 | |

- So the "none" and "true label" numbers below reproduce steps 11 and 12. The Allen ridge here uses per-neuron sufficient statistics (step 12's `neuron_stats`) instead of step 8's per-session sums. This changes only the order of the sums, and the same alphas are picked
- MLP m1 and m2 for folds 1–4 (Bugeon) and 1–4 (Allen) are read from `results/step11|12/neuron_scores_all_labels.csv`. Only fold 0 was refit to confirm that the training code still gives the same models in this environment
- The S features also match: Allen ≤ 5e-5 relative difference to `results/step10/neuron_features.csv`; Bugeon per-type means ≤ 4e-4 relative difference to `results/step12/features_by_type.csv` (both CSVs are rounded)

### Task 1: dynamics summary and clusters (inside each fold, training mice only)
- **Summary:** the S feature vector of each neuron, computed from the train part of its own recording (the first 80%; for Bugeon, the train part of every block). The same S function is used for both datasets:
  - Allen: step 11 `s_features`, 37 features
  - Bugeon: step 12 `s_features`, 39 features
  - Features: skew, kurtosis, event rate, time above z = 2, IEI histogram, autocorrelations, power bands, mean z per stimulus, stimulus on − off, direction responses, tuning peak, DSI and OSI
- **Standardise:** z-score each feature with the mean / std of the training neurons. A feature with zero variance gets std 1 (Bugeon's "IEI 0–0.3 s" is always 0 at 4.3 Hz)
- **PCA:** keep 95% of the variance. That is 14 PCs for Bugeon and 13–14 for Allen, in every fold
- **Clusters:** k-means (n_init 10) and a Gaussian mixture (full covariance, reg 1e-3, n_init 3) for k ∈ {2, 3, 4, 6, 8, 12, 16}
  - Also k = 24 and 32 as an exploratory extension. These were never eligible for selection; they were added because k = 16 was picked everywhere
  - Held-out neurons go to the nearest centroid (k-means) or to the most likely component (GMM)
  - No cluster is empty in any fold. The smallest training cluster at k = 16 has 63 neurons (Bugeon) and 14 (Allen)
- **No labels and no test data** enter the summary or the clustering
- **Choice of k:** made on the val windows (last 10% of the train part) of the training mice:
  - The per-cluster ridge models are fit on the fit part, with alpha picked on val
  - They give each training session a val score, 1 − SSE / SSE_copylast, computed from the sufficient statistics. The check against step 12's `val_mse` matched to < 1e-6
  - k maximises the mean over training sessions
  - **k = 16 was chosen in every fold of both datasets** (k-means; GMM 16, and 12 in Allen fold 0)

### Task 2: forecasting models
- **Ridge** (step 12 closed form from summed per-neuron statistics, alpha on val, refit on train + val):
  - **b (none):** class-balanced over the true types, as in steps 11 and 12
  - **d (true type):** one model per type, with equal row weights
  - **Per-cluster models for every k**, fit "like d": one model per cluster, trained on that cluster's training neurons with equal row weights. This is fully label-free (`r_km<k>`, `r_gmm<k>`)
  - **Sensitivity:** the same with class-balanced rows inside each cluster (`r_kmbal<k>`)
  - **Reference "k = 1":** one model on all training neurons with equal row weights. It isolates the effect of dropping the class balance used by b
  - **Shuffled-cluster control:** cluster ids are permuted over the training neurons (same sizes) and, separately, over the held-out neurons. One model per fake cluster, 5 seeds, every k (`r_kmshuf<k>`)
  - **"Chosen k" (`r_sel`)** = k-means at the val-chosen k of the fold
- **MLP** (step 9 / 12 FiLM network; same optimiser, epochs, early stopping and seeding formula, 3 seeds):
  - Batches are class-balanced over the true types, as for m1 and m2, so only the FiLM input changes between variants
  - All FiLM variants use m2's seeds, so they start from the same weights and see the same batches
  - The variants:
    - m1: no label (reused)
    - m2: true type (reused)
    - `m_cl`: FiLM from the cluster id at the chosen k (16)
    - `m_clnt`: cluster id at k = number of true types (6 Bugeon, 4 Allen)
    - `m_cont`: FiLM from the continuous standardised S vector through a linear layer (→ 16 dims, N(0, 0.02) init, zero-initialised gamma / beta), with no clustering
    - `m_contT`: continuous summary and the true-type embedding, concatenated (32 dims)
    - `m_clsh`: shuffled cluster ids at the chosen k (same permutation as ridge, seeds 0–2)
- **Comparison labels:** step 12's S-classifier labels (Bugeon) and step 11's S labels (Allen), read from their saved `neuron_predictions.csv` and scored with d / m2 (`r_S`, `m_S`)
- **Unit score** = 1 − ΣSSE / ΣSSE_copylast over the unit's neurons. Units are Bugeon sessions (17; mice for the sign check) and Allen mice (46; one session each)

## Data
- **Bugeon et al. 2022:** 4 mice, 17 sessions, 8,199 labelled neuron-sessions (EC 6,684, Lamp5 543, Pvalb 436, Vip 370, Sst 113, Sncg 53). Deconvolved activity at 4.27–4.29 Hz. The train part is 91.5 min per session (mean)
- **Allen Visual Coding 2P:** VISp, three_session_A, 46 mice (Cux2 13, Sst 12, Vip 12, Pvalb 9), 3,296 neurons. dF/F at 10 Hz. The train part is 51.2 min per session
- Details are as in `results/step12_result.md` and `results/step11_result.md`

## Results

### Task 2: does a label-free dynamics summary help as much as the genetic type?
Mean unit score, and the gain over no label (`results/step13/gains.csv`; Wilcoxon over units in `stats.csv`; counts = units where the variant beats no label):

| Variant | Bugeon ridge | Bugeon MLP | Allen ridge | Allen MLP |
|---|---|---|---|---|
| none (b / m1) | 0.4960 | 0.5085 | 0.2997 | 0.4039 |
| true type (d / m2) | +0.0014 (9/17) | +0.0018 (14/17) | +0.0252 (40/46) | +0.0239 (43/46) |
| S-classifier labels (steps 11 / 12) | +0.0062 (17/17) | +0.0027 (17/17) | +0.0258 (40/46) | +0.0237 (43/46) |
| **clusters, chosen k = 16** | **+0.0133** (17/17) | **+0.0066** (17/17) | **+0.0473** (43/46) | **+0.0302** (43/46) |
| clusters, k = n types (6 / 4) | +0.0096 (16/17) | +0.0048 (17/17) | +0.0264 (41/46) | +0.0258 (40/46) |
| clusters, GMM, chosen k | +0.0140 (17/17) | | +0.0433 (43/46) | |
| **continuous summary** | – | **+0.0174** (17/17) | – | **+0.0436** (44/46) |
| continuous + true type | – | +0.0175 (17/17) | – | +0.0469 (44/46) |
| shuffled clusters, chosen k | +0.0012 (8/17) | −0.0003 (3/17) | −0.0221 (17/46) | +0.0029 (26/46) |
| k = 1, equal row weights (ridge reference) | +0.0013 (8/17) | | −0.0213 (17/46) | |

Key paired comparisons (two-sided Wilcoxon; Bugeon over 17 sessions, smallest possible p = 1.5e-5; Allen over 46 mice):

| Comparison | Bugeon: up, mean diff, p | Bugeon mice (signs) | Allen: up, mean diff, p |
|---|---|---|---|
| MLP continuous vs none | 17/17, +0.0174, 1.5e-5 | 4/4 | 44/46, +0.0436, 3.7e-10 |
| **MLP continuous vs true type** | **17/17, +0.0157, 1.5e-5** | 4/4 | **42/46, +0.0197, 4.4e-8** |
| **MLP continuous + true vs continuous** | **11/17, +0.0001, 0.31** | 2/4 | **25/46, +0.0033, 0.28** |
| MLP continuous + true vs true | 17/17, +0.0157, 1.5e-5 | 4/4 | 44/46, +0.0229, 5.6e-9 |
| MLP cluster (k = 16) vs none | 17/17, +0.0066, 1.5e-5 | 4/4 | 43/46, +0.0302, 9.4e-9 |
| MLP cluster (k = 16) vs true type | 17/17, +0.0048, 1.5e-5 | 4/4 | 32/46, +0.0063, 0.0021 |
| MLP cluster (k = n types) vs true type | 15/17, +0.0030, 7.6e-5 | 4/4 | 25/46, +0.0018, 0.39 |
| MLP cluster vs shuffled cluster | 17/17, +0.0069, 1.5e-5 | 4/4 | 43/46, +0.0273, 3.6e-9 |
| MLP continuous vs cluster | 17/17, +0.0109, 1.5e-5 | 4/4 | 41/46, +0.0134, 2.4e-5 |
| MLP continuous vs S-classifier labels | 17/17, +0.0148, 1.5e-5 | 4/4 | 43/46, +0.0199, 3.3e-8 |
| MLP shuffled cluster vs none | 3/17, −0.0003, 0.0067 | 0/4 | 26/46, +0.0029, 0.29 |
| ridge cluster (k = 16) vs none | 17/17, +0.0133, 1.5e-5 | 4/4 | 43/46, +0.0473, 1.6e-12 |
| **ridge cluster (k = 16) vs true type** | **17/17, +0.0119, 1.5e-5** | 4/4 | **39/46, +0.0221, 4.4e-7** |
| ridge cluster vs shuffled cluster | 17/17, +0.0121, 1.5e-5 | 4/4 | 45/46, +0.0695, 8.5e-14 |
| ridge cluster (k = n types) vs true type | 17/17, +0.0082, 1.5e-5 | 4/4 | 24/46, +0.0011, 0.79 |
| ridge true vs none | 9/17, +0.0014, 0.26 | 2/4 | 40/46, +0.0252, 1.3e-7 |

Per-mouse gains in Bugeon (`stats.csv`, `per_mouse`; SB025 / SB026 / SB028 / SB030):
- MLP continuous vs none: +0.024 / +0.014 / +0.017 / +0.009
- MLP continuous + true vs continuous: +0.0005 / −0.0003 / −0.0003 / +0.0002
- ridge clusters vs true: +0.013 / +0.009 / +0.013 / +0.012

**4 mice give little power.** A Wilcoxon test over mice cannot go below p = 0.125. The session-level p-values treat sessions of a mouse as independent, which they are not. "4 of 4 mice up" is the honest animal-level statement.

- **Without any label, the dynamics summary beats the genetic type in both datasets:**
  - Bugeon (types mixed in one recording, one indicator): 10× the true-type gain with the MLP (+0.0174 vs +0.0018), and 10× with ridge (+0.0133 vs +0.0014)
  - Allen (one Cre line per recording): 1.8× the true-type gain with the MLP (+0.044 vs +0.024), and 1.9× with ridge (+0.047 vs +0.025)
- **The continuous summary beats the cluster id** (MLP, both datasets: +0.011 / +0.013), and more clusters beat fewer (panel a). The useful information is graded, not a few discrete groups
- **At an equal number of groups (k = number of types), unsupervised clusters match or beat the true type:**
  - Bugeon: better in every way (ridge +0.0096 vs +0.0014; MLP +0.0048 vs +0.0018)
  - Allen: indistinguishable from the true Cre line (ridge 24/46, p 0.79; MLP 25/46, p 0.39)
- **The genetic type adds nothing on top of the continuous summary:** continuous + true vs continuous is +0.0001 (Bugeon, p 0.31, 2/4 mice) and +0.0033 (Allen, p 0.28, 25/46 mice)
  - The only hint of extra value is Allen Pvalb: +0.0035, 5/9 mice, p 0.20; this is the GCaMP6s line
  - In Bugeon, it is Vip: +0.0007, 15/17 sessions, p 0.004; the only per-type test with p < 0.05
- **Grouping alone does not help:** the shuffled-cluster controls give nothing
  - MLP: −0.0003 (Bugeon), +0.0029 (Allen, n.s.)
  - Ridge: the shuffled control equals the "k = 1, equal weights" reference in both datasets (Bugeon +0.0012 vs +0.0013; Allen −0.022 vs −0.021). A random split just reproduces one unweighted model
  - So the gain is in which neurons are grouped together
- **Ridge weighting caveat:** with ridge, the Bugeon true-type gain (+0.0014) equals the effect of merely dropping b's class balance (k = 1: +0.0013). In Allen, dropping the balance costs −0.021 because Cux2 dominates. So the label-free per-cluster ridge models (equal weights) beat a balanced baseline despite carrying that −0.021 handicap in Allen. Class-balanced rows inside clusters (`kmbal`) give +0.0125 (Bugeon) and +0.0477 (Allen) at k = 16, about the same

### Task 1: gain vs k (`results/step13/k_sweep.csv`, `val_k.csv`; plots (a) and `step13_valk.png`)
| k | Bugeon ridge, k-means | Bugeon GMM | Bugeon shuffled | Allen ridge, k-means | Allen GMM | Allen shuffled |
|---|---|---|---|---|---|---|
| 2 | +0.0029 | +0.0035 | +0.0013 | +0.0164 | +0.0068 | −0.0215 |
| 3 | +0.0070 | +0.0074 | +0.0013 | +0.0257 | +0.0160 | −0.0215 |
| 4 | +0.0077 | +0.0082 | +0.0013 | +0.0264 | +0.0227 | −0.0217 |
| 6 | +0.0096 | +0.0109 | +0.0013 | +0.0395 | +0.0268 | −0.0216 |
| 8 | +0.0122 | +0.0116 | +0.0013 | +0.0413 | +0.0347 | −0.0218 |
| 12 | +0.0126 | +0.0132 | +0.0012 | +0.0458 | +0.0433 | −0.0217 |
| **16 (chosen)** | **+0.0133** | +0.0140 | +0.0012 | **+0.0473** | +0.0411 | −0.0221 |
| 24 (exploratory) | +0.0140 | +0.0144 | | +0.0487 | +0.0464 | |
| 32 (exploratory) | +0.0147 | +0.0148 | | +0.0500 | +0.0412 | |
| true type (d) | +0.0014 | | | +0.0252 | | |

- **Clusters beat the true type at small k:** from k = 3 in Bugeon and from k = 6 in Allen (k = 3–4 ties with it)
- **Val and test agree:** the val score (training mice) rises with k in every fold, as the test gain does, so k = 16 is the grid maximum everywhere. The exploratory k = 24 / 32 still improve slightly (+0.001 to +0.003), so the gain has not saturated within the grid
- **GMM vs k-means:** GMM is similar to k-means in Bugeon and somewhat worse in Allen

### Task 3: is it batch? (`results/step13/ami.csv`, `contingency.csv`, `cluster_profiles.csv`; plot (c) and `step13_batch.png`)
Adjusted mutual information (AMI) of the k-means clusters, mean over folds:

| | k | AMI with type | AMI with session | AMI with mouse | Reference |
|---|---|---|---|---|---|
| Bugeon, training neurons (3 mice) | 6 | 0.060 | 0.094 | 0.066 | AMI(type, mouse) = 0.021 |
| | 16 | 0.060 | 0.122 | 0.084 | |
| Bugeon, held-out mouse | 16 | 0.060 | 0.061 | (1 mouse) | |
| Allen, training neurons (~37 mice) | 4 | 0.164 | 0.111 (= mouse) | 0.111 | AMI(Cre line, mouse) = 0.293 |
| | 16 | 0.172 | 0.180 | 0.180 | |
| Allen, held-out mice (~9) | 16 | 0.187 | 0.213 | 0.213 | AMI(Cre line, mouse) = 0.514 |

Share of a cluster's neurons that come from its largest mouse (training neurons, k = 16), against what independent clusters would give:
- Bugeon: 0.60 vs 0.48
- Allen: 0.23 vs 0.13

**Allen: the clusters do not simply recover mice or Cre lines.**
- AMI with mouse (0.18) and with line (0.17) are both well below the 0.29 that clusters equal to the Cre lines would give
- At k = 16, fold 0 (rows = clusters):
  - Cux2 neurons spread over 12 clusters
  - Sst neurons concentrate in one cluster (cluster 9: 62 Sst, 104 Cux2)
  - **Pvalb and Vip share two clusters** (cluster 2: 47 Pvalb + 65 Vip; cluster 13: 42 Pvalb + 55 Vip)
  - So the summary groups the two GCaMP6s-heavy, sparse-firing lines together rather than separating them
- Each mouse's neurons spread over several clusters (`step13_batch.png`, lower middle)

**Bugeon: clusters mix types and sessions.**
- AMI with type is only 0.06; type purity (0.81) equals the EC share
- Every cluster holds neurons from all 4 mice (plot (c), right)
- The clusters carry somewhat more session than type information (AMI 0.12 vs 0.06 in the training mice, 0.06 vs 0.06 in the held-out mouse). Part of the summary is recording or brain-state related; e.g. kurtosis and event rate depend on the session's arousal
- But the gain transfers to held-out mice, where cluster ids come from centroids fitted on other animals, and it is 4/4 mice positive. So whatever the clusters encode is not mouse-specific
- Cluster examples at k = 16, fold 0 (`cluster_profiles.csv`):
  - cluster 1: 743 neurons, 738 EC; very sparse (kurtosis +2.7 SD), low event rate
  - cluster 11: Pvalb / Sst rich (83 Pvalb, 35 Sst, 230 EC); dense firing
  - cluster 14: Lamp5 / Pvalb rich

**Gain within each true type** (`results/step13/type_gain.csv`, plot `step13_types.png`; pooled over the type's neurons; units up and Wilcoxon over the units that contain the type):

| Type | n | MLP true | MLP cluster (16) | MLP continuous | MLP cont + true − cont | Ridge true | Ridge cluster (16) |
|---|---|---|---|---|---|---|---|
| Bugeon EC | 6,684 | +0.0015 (16/17) | +0.0069 (17/17) | +0.0174 (17/17) | +0.0000 (11/17, p 0.19) | +0.0007 (8/17) | +0.0140 (17/17) |
| Bugeon Lamp5 | 543 | +0.0015 (15/17) | +0.0039 (17/17) | +0.0140 (17/17) | −0.0000 (10/17) | +0.0014 (16/17) | +0.0075 (17/17) |
| Bugeon Pvalb | 436 | +0.0032 (12/16) | +0.0022 (12/16) | +0.0142 (16/16) | +0.0001 (8/16) | +0.0047 (11/16) | +0.0091 (16/16) |
| Bugeon Sncg | 53 | +0.0024 (9/14) | +0.0046 (12/14) | +0.0170 (13/14) | −0.0007 (6/14) | +0.0044 (11/14) | +0.0097 (10/14) |
| Bugeon Sst | 113 | +0.0006 (10/15) | +0.0004 (7/15) | +0.0094 (14/15, p 0.004) | −0.0011 (4/15) | +0.0001 (6/15) | −0.0012 (8/15) |
| Bugeon Vip | 370 | +0.0007 (13/17) | +0.0054 (15/17) | +0.0172 (16/17) | +0.0007 (15/17, p 0.004) | −0.0023 (2/17) | +0.0108 (16/17) |
| Allen Cux2 | 2,905 | +0.0132 (13/13) | +0.0170 (13/13) | +0.0291 (13/13) | −0.0001 (7/13) | +0.0129 (12/13) | +0.0395 (13/13) |
| Allen Sst | 122 | +0.0004 (9/12) | +0.0111 (9/12) | +0.0278 (10/12, p 0.04) | −0.0009 (6/12) | +0.0250 (9/12) | +0.0319 (10/12) |
| Allen Vip | 153 | +0.0186 (12/12) | +0.0178 (12/12) | +0.0375 (12/12) | +0.0001 (7/12) | +0.0147 (10/12) | +0.0362 (11/12) |
| Allen Pvalb | 116 | +0.0357 (9/9) | +0.0489 (9/9) | +0.0595 (9/9) | +0.0035 (5/9, p 0.20) | +0.0330 (9/9) | +0.0689 (9/9) |

- **Yes, a dynamics summary helps inside one type.** In Allen, every mouse holds one Cre line, so the true label is constant within a mouse. There the continuous summary still adds +0.016 over the true label for Cux2 (13/13 mice) and +0.024 for Pvalb (7/9)
- In Bugeon, the continuous summary helps every type (+0.009 to +0.017); the true label helps at most +0.003 (Pvalb)
- The only type where clusters at k = 16 do not help is Sst in Bugeon (n = 113): ridge −0.0012, MLP +0.0004. The continuous summary still gives +0.0094 there

### Task 4: how much history does the summary need? (`results/step13/history.csv`, plot (d))
The summary of held-out neurons is recomputed from the first 3 min, the first 30 min, or the full train part. The trained PCA / k-means / models are unchanged. Gain over no label, mean over units (units up):

| History | Bugeon ridge cluster | Bugeon MLP cluster | Bugeon MLP continuous | Allen ridge cluster | Allen MLP cluster | Allen MLP continuous |
|---|---|---|---|---|---|---|
| 3 min | +0.0042 (12/17) | +0.0024 (15/17) | **−0.090** (1/17) | −0.038 (21/46) | −0.081 (16/46) | **−0.83** (0/46) |
| 30 min | +0.0102 (17/17) | +0.0050 (16/17) | +0.0102 (16/17) | +0.0403 (43/46) | +0.0270 (43/46) | +0.0340 (44/46) |
| full (91.5 / 51.2 min) | +0.0133 (17/17) | +0.0066 (17/17) | +0.0174 (17/17) | +0.0473 (43/46) | +0.0302 (43/46) | +0.0436 (44/46) |
| true type, for reference | +0.0014 | +0.0018 | +0.0018 | +0.0252 | +0.0239 | +0.0239 |
| held-out neurons in the same cluster as with the full history | 3 min: 43%, 30 min: 69% | | | 3 min: 26%, 30 min: 71% | | |

- **30 minutes are enough to beat the genetic label** in both datasets, with every variant. The gain there is 60–90% of the full-history gain
- **3 minutes are not enough, and the continuous input then becomes harmful.** With 3 min, Bugeon's direction features are missing for most neurons (22% of all feature values are NaN and set to the training mean), and Allen's first 3 min contain only drifting gratings (no movie or spontaneous features). The remaining statistics are noisy and out of the training distribution
  - The FiLM layer, trained only on full-history summaries, extrapolates badly: −0.09 (Bugeon) and −0.83 (Allen)
  - Cluster ids bound the damage: Bugeon stays positive (+0.002 to +0.004); Allen drops below no label
  - Training with summaries from random short windows (as step 11's window test, LOLCAT style) would be the fix. It was not tried here
- The long-run summary needs on the order of 10–30 min of activity. This is consistent with step 11, where typing accuracy grew with window length

**Key question: without any labels, does a summary of a neuron's long-run dynamics improve forecasting as much as or more than its genetic cell type, and does the genetic type add anything on top?**
- **Yes, more:**
  - **Bugeon** (mixed types, one indicator): the S summary of the train part, given to the MLP as a continuous FiLM input, raises the score by +0.0174, against +0.0018 for the transcriptomic type (17/17 sessions and 4/4 mice better than the true type)
  - **Allen** (one Cre line and reporter per recording): +0.044 vs +0.024 (42/46 mice better than the true type, p 4e-8)
  - **Label-free k-means clusters with per-cluster ridge models:** +0.013 vs +0.0014 (Bugeon) and +0.047 vs +0.025 (Allen)
  - **Shuffled-cluster controls give nothing**, so the gain comes from which neurons are grouped together, not from splitting the data
- **The genetic type adds nothing measurable on top of the continuous summary** (+0.0001, p 0.31, 2/4 mice in Bugeon; +0.003, p 0.28, 25/46 mice in Allen)
- **Caveats:**
  - The summary needs ~30 min of the neuron's own activity from the same recording
  - With 3 min it fails, badly for the continuous input
  - In Bugeon part of the summary is session-related
  - 4 mice limit any animal-level claim

## Plots
- `plots/step13_dynamics.png`:
  - (a) ridge score gain over no label vs k: k-means, GMM, class-balanced k-means and shuffled clusters, with mean ± sem over units. Lines show the true-type, S-classifier and k = 1 references; stars mark the MLP cluster-id variants. Bugeon and Allen
  - (b) gain of true type / clusters (k = 16 and k = n types) / continuous / continuous + true / shuffled clusters, for MLP and ridge, per dataset. One dot per unit, with counts and Wilcoxon p
  - (c) Bugeon, fold 0, k = 16: share of each type's and of each mouse's neurons in each cluster
  - (d) gain vs history length of the summary (3 min, 30 min, full train part); the 3 min continuous values are off scale and printed
- `plots/step13_types.png`: gain within each true type, MLP and ridge, Bugeon and Allen
- `plots/step13_batch.png`:
  - AMI of clusters with type / session / mouse vs k
  - largest-mouse share per cluster
  - type purity
  - Allen fold 0 cluster × Cre line and cluster × mouse
- `plots/step13_valk.png`: val score gain over b vs k per fold (the k selection)

Numbers behind the plots (`results/step13/`, 12.2 MB, almost all `neuron_scores.csv`):

| File | Contents |
|---|---|
| `neuron_scores.csv` | per neuron (both datasets): test SSE and score of every variant:<br>• ridge: none, true, S, k1, every `km` / `gmm` / `kmbal` / `kmshuf` k, sel, nt, selshuf, gmmsel, 3 / 30 min histories<br>• MLP: none, true, S, cl, clnt, cont, contT, clsh, 3 / 30 min histories, fold-0 reruns<br>• also the chosen k and the S label |
| `unit_scores.csv` | per unit (Bugeon sessions and mice; Allen mice): score of every variant |
| `gains.csv` | mean score and gain over no label per variant (plot b) |
| `stats.csv` | all Wilcoxon tests; Bugeon also per-mouse signs and per-mouse differences |
| `k_sweep.csv`, `val_k.csv`, `chosen_k.csv` | gain vs k (test); val score vs k per fold; chosen k |
| `type_gain.csv` | within-type gains and tests |
| `history.csv` | task 4: gain per history length, cluster agreement with the full history |
| `ami.csv`, `contingency.csv`, `cluster_profiles.csv` | task 3: AMI with type / session / mouse, purity, cluster × type / mouse tables (chosen k and k = n types), mean standardised S feature per cluster (fold 0) |
| `mlp_fits.csv` | MLP epochs, best val loss, seconds per fit |
| `reproduction_check.csv` | reproduction of steps 11 / 12 |

## Problems / errors
- **k was chosen at the top of the grid (16) in every fold**, and the val score kept rising at k = 24 / 32 (exploratory). So the "chosen k" is limited by the grid. The continuous summary, which needs no k, does better anyway
- **"Label-free" applies to the summary, the clustering and the conditioning input, not to everything:**
  - The MLP's class-balanced batches use the true types for every variant, including m1. This keeps the comparison with steps 11 and 12 exact
  - The ridge "none" baseline b is class-balanced
  - The per-cluster ridge models use equal row weights and so are fully label-free; the k = 1 reference shows what dropping the balance alone does
- **The summary is not from another session.** It comes from the first 80% of the same recording whose last 20% is forecast. The z-scoring of the forecaster inputs already uses that part, but a summary from a separate session was not tested. The val windows used to choose k lie inside the period the summary is computed from; the test windows never do
- **Short-history summaries were never seen in training.** Missing features (no stimulus of a kind, a direction never shown) were set to the training mean
  - For Bugeon, "first N minutes" takes the first part of every block in proportion to its length (at least 128 frames per block), because the blocks are separate recordings
  - For Allen it is the first N minutes, which are drifting gratings only
- **Statistics:**
  - Bugeon has 4 mice: a Wilcoxon over mice cannot go below p = 0.125, so only signs are reported per mouse. Session-level tests treat sessions as independent
  - Allen within-type tests have 9–13 mice per line
- **MLP reruns:** only fold 0 of m1 / m2 was refit (6 fits per dataset). The other folds of m1 / m2 come from the step 11 / 12 CSV files
- **Epoch cap:** 13 of 147 fits reached the 40-epoch cap (Allen: 2 m2, 1 m1, 1 cl, 1 clnt, 2 clsh; Bugeon: 3 clnt, 2 clsh, 1 cont), as in steps 9–12
- **Environment issues:** a fresh `.venv` needed `setuptools<70` for allensdk's `pkg_resources`. `api.brain-map.org` was blocked, so Allen data again came from the S3 mirror. No reported number was affected

### Code and ideas from other work
- **This repo:**
  - Step 3: alpha grid, windows
  - Step 4: stimulus windows
  - Step 5: `Gram` ridge solver
  - Step 7: balanced weights
  - Step 8: sessions and folds
  - Step 9: `MLP` with zero-initialised FiLM, `Sampler`, `gather`, training settings and seeding
  - Step 10: hand-feature constants
  - Step 11: Allen `s_features`, extract
  - Step 12: Bugeon extract, S features, `neuron_stats`, `assemble`, `val_mse`, `fit_weighted`, `balanced_weights`, `fake_labels` (applied to clusters), `Sampler`, `wil`
- **NeuPRINT** (Mi et al. 2023): learns a time-invariant representation of each neuron from its activity and conditions a forecaster on it. The continuous-summary FiLM input is the hand-feature version of that idea
- **NuCLR** (Arora et al. 2025): neuron representations learned from activity alone, from which cell type is read out. This motivated asking whether such a label-free representation already carries the forecasting-relevant "type"; its Bugeon loading conventions entered via step 12
- **POCO:** zero-initialised FiLM conditioning (via step 9), so every conditioned model starts as m1
- **LOLCAT** (Schneider et al. 2023): inter-event-interval features and class-balanced sampling (via step 10), and the question of how much activity a summary needs (task 4, as step 11's window test)
- **Single-cell RNA-seq practice:** z-score, PCA, then clustering (as in Seurat / Scanpy) for the cluster pipeline
- No code was copied from these repositories, and none was re-read for this step

## What I noticed
- A neuron's own long-run activity statistics are a better "type" for forecasting than its genetic label. This holds even where the label looked strong (Allen, where it is confounded with recording and indicator), and the label then adds nothing on top. The useful information is graded: continuous > 16 clusters > 6 clusters ≥ true types.
- The catch is the history: the summary needs tens of minutes of the same neuron. A model that should use short histories must be trained on short-history summaries, or it fails worse than using no summary at all.
