# Step 14: Atlas of dynamics types + fast recogniser, vs per-neuron models, across history lengths (Bugeon and Allen)
Date: 2026-10-18

## What was run
- `scripts/step14_atlas.py <dataset> <phase>` for `bugeon` and `allen`, phases `step13`, `features`, `windows`, `recognise`, `ridge`, `mlp`, `finetune`, `forecast` (Allen also `extract`). Then `scripts/step14_atlas.py both analyse`, which writes `results/step14/`. Then `scripts/step14_plots.py`, which remakes every plot from the CSV files in `results/step14/` only
- **Environment:** a fresh `.venv` with Python 3.11, PyTorch 2.14.1 (CPU), scikit-learn 1.7.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, allensdk 2.16.2, statsmodels 0.13.5 (allensdk 2.16 fails to import with newer statsmodels), setuptools 69.5 (< 70, for allensdk's `pkg_resources`). These are the versions of steps 11–13
- **Data** (nothing under `data/` is committed):
  - **Bugeon:** downloaded from https://ndownloader.figshare.com/files/38046633. The MD5 is 4556ed80ce98ca13f909129e25f60f3a, which matches. It was unpacked to `data/bugeon/raw/`, and step 12's `extract` was rerun through step 13's `prep`. Its `neurons.csv`, `sessions.csv` and `blocks.csv` are byte-identical to `results/step12/`
  - **Allen:** `ophys_experiments.json` and the 46 NWB files came from the Allen S3 mirror (https://allen-brain-observatory.s3.amazonaws.com/visual-coding-2p/) using step 11's `extract`, unchanged. One hook was added (`step14_atlas.py allen extract`): it also saves the running speed of every 10 Hz bin of the train and test parts while the NWB is on disk, because task 3 needs it and step 11's cache does not keep it
  - No download was blocked
- **Same settings as step 13:** splits, folds, windows, seeds, inputs and model settings
  - Allen: 46 mice, 5 leave-mice-out folds, 10 Hz, past 30 → next 10 bins
  - Bugeon: 17 sessions of 4 mice, leave one mouse out, 4.3 Hz, past 13 → next 4 frames
  - The forecast test set (last 20%) is unchanged
- **Wall time** on 4 CPU cores, with several phases running at once (so times are inflated):

  | Phase | Bugeon | Allen |
  |---|---|---|
  | data (download + extract), step 13 prep / features / cluster | ~5 min | ~25 min |
  | step 13 ridge (only for its `chosen_k.csv`; Allen's was taken from `results/step13/chosen_k.csv`) | 8 min | – |
  | features (atlas, task 3 summaries, states) | 0.5 min | 1.3 min |
  | windows (recogniser inputs) | 13 min | 6 min |
  | recognise | ~3 h | 1.6 h |
  | ridge (incl. per-neuron ridge at 4 histories) | 20 min | 42 min |
  | MLP training (60 / 75 fits) | ~4.5 h | 3.5 h |
  | per-neuron fine-tune | 2.9 h | 1.2 h |
  | MLP forecast of all variants | 25 min | 4 min |

### Reproduction of step 13 (`results/step14/reproduction_check.csv`)
Every number below was recomputed here; the MLP networks were retrained with step 13's `run_job`. Step 13's model of each kind is the mean over 3 seeds.

| Dataset | Here | Step 13 | Max abs. difference in unit score | Mean unit score here / step 13 |
|---|---|---|---|---|
| Bugeon | ridge none | r_none | 1.2e-8 | 0.4960 / 0.4960 |
| Bugeon | ridge true type | r_true | 9.5e-9 | 0.4974 / 0.4974 |
| Bugeon | ridge atlas cluster, full history | r_km16 | 5.7e-9 | 0.5093 / 0.5093 |
| Bugeon | MLP none (m1, retrained) | m_none (step 12) | 1.7e-7 | 0.5085 / 0.5085 |
| Bugeon | MLP cluster FiLM (retrained) | m_cl | 1.5e-7 | 0.5151 / 0.5151 |
| Bugeon | MLP continuous summary (retrained) | m_cont | 2.1e-7 | 0.5260 / 0.5260 |
| Bugeon | naive 3 / 30 min summary (ridge cl., MLP cl., MLP cont.) | r_sel_h3/h30, m_cl_h3/h30, m_cont_h3/h30 | ≤ 2.7e-7 | e.g. MLP cont 3 min 0.4187 / 0.4187 |
| Allen | ridge none | r_none | 1.0e-7 | 0.2997 / 0.2997 |
| Allen | ridge true type | r_true | 1.4e-7 | 0.3249 / 0.3249 |
| Allen | ridge atlas cluster | r_km16 | 1.8e-7 | 0.3470 / 0.3470 |
| Allen | MLP none (m1, retrained) | m_none (step 11) | 1.5e-7 | 0.4039 / 0.4039 |
| Allen | MLP cluster FiLM | m_cl | 1.1e-7 | 0.4342 / 0.4342 |
| Allen | MLP continuous summary | m_cont | 1.2e-7 | 0.4475 / 0.4475 |
| Allen | naive 3 / 30 min summary | as above | ≤ 2.1e-6 | MLP cont 3 min −0.4279 / −0.4279 |

- **The check passes.** none / true / r_km16 / m_cont (and m_cl, the 3 and 30 min histories) reproduce step 13 to < 3e-6 in every unit
- Further checks:
  - **Atlas and features:**
    - The atlas k-means labels are identical to step 13's in every fold (asserted)
    - The step 13 S features at 3 / 30 min / full are identical (asserted)
    - Retrained MLPs had the same epochs and best val losses as step 13's logs (e.g. Bugeon fold 0 cl seed 0: 13 epochs, 0.7402)
  - **New code vs reference computations:**
    - The new forecast code (first layer computed once, FiLM applied per variant) matches `run_job`'s own forecast to < 1e-5 relative SSE (`data/step14/*/forecast_check.csv`)
    - The new ridge code matches step 13's per-neuron SSE to 4e-9 (a session checked during development)
- The MLP true-type model (m2) is not retrained: its SSE is read from steps 11 / 12, as in step 13

### Methods
- **History H** of a held-out neuron: the first 3, 10 or 30 min, or all of the train part (Bugeon 91.5 min, Allen 51.2 min on average), always from its start. For Bugeon, a piece of every block in proportion to its length (≥ 128 frames), as step 13
- **Task 1 (K1), per-neuron models fitted on H only:**
  - **Per-neuron ridge** (step 8 model "a": own past + stimulus → next window):
    - Fit windows: targets in the first 90% of H. Val windows: targets in the last 10% of H (for Bugeon, per block piece)
    - Alpha from step 3's grid, picked on val, then refit on all of H
    - Val windows may reach back into the first 90% for their past. This was needed because with 3 min of Bugeon a 10% val part (13–20 frames per block) is shorter than one window. For the full history the fit windows are exactly step 12's
  - **Per-neuron MLP fine-tune (POCO-style unit embedding):**
    - The trained continuous-summary network (`cont`, 3 seeds) is frozen. Each held-out neuron gets its own 16-dim FiLM embedding, initialised at the population mean (the layer's bias, i.e. what an average summary gives)
    - Fitted with Adam (lr 0.05; picked from 0.005 / 0.02 / 0.05 on one session of fold 0) on random batches of 32 of its fit windows, at most 500 steps
    - Early stopping on its val windows (up to 256), checked every 10 steps. Each neuron keeps its best embedding
    - Scored per seed, then the SSE is averaged over the 3 seeds like every MLP variant
  - Compared with shared models: none, true type, the step 13 summary computed from the same H (naive: continuous FiLM and k = 16 clusters), and the full-history summary (upper bound)
- **Task 2 (E2), atlas + recogniser:**
  - **Atlas:** on training mice only, step 13's z-score → PCA (95%) → k-means k = 16 of the full-history S summaries, with:
    - the per-cluster ridge models (equal row weights, = step 13 r_km16)
    - the cluster-FiLM MLP (= step 13 m_cl)
    - the continuous-summary MLP (= m_cont)
  - **Recogniser inputs** (`windows`): the S features (same function as the summary) of short windows:
    - Lengths: 30 s, 3, 10 and 30 min; 20 random windows per neuron per length (same positions for all neurons of a session)
    - Allen: contiguous windows
    - Bugeon: 30 s = 128 contiguous frames inside one block (the Welch segment, 29.8 s). 3 / 10 / 30 min = a piece of every block in proportion to its length, as H
    - Training rows: training-mice neurons, 10 windows per neuron per length (Bugeon, 246k rows) or 20 (Allen, 211k rows)
    - log10 window length is an extra input
  - **Teacher / student:**
    - Target = the neuron's full-history atlas cluster
    - Candidates: logistic regression (C = 1; quantile-normal transform; missing features = training median + missing indicators) and gradient boosting (sklearn HistGradientBoosting, 200 iterations max, early stopping, NaN native)
    - Picked by inner CV over the training mice (GroupKFold by mouse, 3 / 4 folds) on log-loss. **Gradient boosting was picked in every fold of both datasets** (inner-CV log-loss Bugeon 1.33 vs 1.40, Allen 1.21 vs 1.41)
  - **Genetic type from the same windows** (comparison): the same classifier family, balanced class weights, tile length picked on balanced accuracy
  - **Short → long regressor:** ridge (alpha 1–10 000 on inner CV; 1 or 10 picked) from the window features to the full-history z-scored S vector
  - **Evidence accumulation:**
    - For history H, non-overlapping tiles of one length are taken from the start of the train part
    - The recogniser log-probabilities are averaged over the tiles inside H and renormalised (softmax of the mean log-probability)
    - The regressor's predictions are averaged over the tiles
    - The tile length is picked per H on the inner-CV (training mice) log-loss / MSE of the accumulated prediction. Picks: Bugeon clusters 3 min for H = 3 and 10 min, 30 min for 30 min and full; Allen 30 s for 3 min, 3 min for 10 min, 30 min for 30 min and full (`recogniser.csv`, `windows_picked`)
    - With the full history the recogniser sees 30-min tiles, not the full summary
  - **Forecasting with the recognised type:**
    - (a) **Hard:** ridge cluster model / MLP cluster embedding of the most likely cluster
    - (b) **Soft:** ridge = per-cluster predictions weighted by the probabilities; MLP = FiLM fed the probability-weighted cluster embedding
    - (c) **E2 continuous:** the cont network fed the regressor's predicted full-history summary
- **Task 3 (K4):**
  - (a) Every raw S feature centred within its session (session mean over its neurons subtracted), then z-score / PCA / k-means k = 16 on training neurons
  - (b) Every neuron given its session's mean S vector; k = min(16, number of distinct training sessions): Bugeon 11–14 (so roughly one model per training session), Allen 16
  - Each is scored with per-cluster ridge models (equal weights) and with a newly trained continuous-summary MLP (3 seeds, m2's seeds)
  - **State of every test window:** Bugeon `frame.states` (0 = running) majority over the window's 17 frames; Allen mean running speed over the window's 40 bins > 1 cm/s
  - Bugeon test windows: 11,698 running / 13,085 stationary. Allen: 11,461 / 23,909, plus 13 windows without speed. SSE is kept per state
- **Task 4 (K3, Bugeon):**
  - 741 neuron-sessions (316 inhibitory cells, UniqueID in ≥ 2 sessions of a mouse; UniqueID is defined for inhibitory cells only)
  - The summary of the cell's other session nearest in date replaces its own (continuous MLP, cluster MLP, cluster ridge)
  - Scored on those cells only
- **Units:** score = 1 − ΣSSE / ΣSSE_copylast. Bugeon: 17 sessions (Wilcoxon) and 4 mice (signs only; 4 mice cannot give p < 0.125). Allen: 46 mice (Wilcoxon)

## Data
- **Bugeon et al. 2022:** 4 mice, 17 sessions, 8,199 labelled neuron-sessions (EC 6,684, Lamp5 543, Pvalb 436, Vip 370, Sst 113, Sncg 53), deconvolved activity at 4.27–4.29 Hz, 4 blocks per session (5 in one). Train part 91.5 min per session
- **Allen Visual Coding 2P:** VISp, three_session_A, 46 mice (Cux2 13, Sst 12, Vip 12, Pvalb 9), 3,296 neurons, dF/F at 10 Hz. Train part 51.2 min. The first 3 / 10 min are drifting gratings only (session A starts with gratings)

## Results

### Main result: gain over no label vs history length H (`gains_by_history.csv`, `plots/step14_atlas.png`)
Mean over units of (score − score of the no-label model of the same family); in brackets: units better than no label. "full" = 91.5 min (Bugeon) / 51.2 min (Allen).

**Bugeon, ridge** (17 sessions)

| Method | 3 min | 10 min | 30 min | full |
|---|---|---|---|---|
| true type | +0.0014 (9/17) | +0.0014 | +0.0014 | +0.0014 |
| naive short summary (step 13 clusters) | +0.0042 (12/17) | +0.0069 (14/17) | +0.0102 (17/17) | +0.0133 (17/17) |
| per-neuron ridge | −0.0469 (3/17) | −0.0040 (7/17) | +0.0171 (16/17) | **+0.0306** (17/17) |
| E2 hard | +0.0090 (16/17) | +0.0122 (17/17) | +0.0125 (17/17) | +0.0134 (17/17) |
| **E2 soft** | **+0.0128** (17/17) | **+0.0152** (17/17) | +0.0149 (17/17) | +0.0159 (17/17) |
| full-history summary (upper bound) | +0.0133 | +0.0133 | +0.0133 | +0.0133 (17/17) |

**Bugeon, MLP**

| Method | 3 min | 10 min | 30 min | full |
|---|---|---|---|---|
| true type | +0.0018 (14/17) | +0.0018 | +0.0018 | +0.0018 |
| naive short summary (continuous) | −0.0898 (1/17) | −0.0126 (5/17) | +0.0102 (16/17) | +0.0174 (17/17) |
| naive short cluster | +0.0024 (15/17) | +0.0032 (15/17) | +0.0050 (16/17) | +0.0066 (17/17) |
| per-neuron MLP fine-tune | −0.0038 (4/17) | +0.0077 (13/17) | **+0.0149** (16/17) | **+0.0210** (17/17) |
| E2 hard | +0.0044 (17/17) | +0.0063 (17/17) | +0.0062 (17/17) | +0.0069 (17/17) |
| **E2 soft** | **+0.0065** (17/17) | **+0.0077** (17/17) | +0.0075 (17/17) | +0.0081 (17/17) |
| E2 continuous (predicted summary) | +0.0043 (13/17) | +0.0058 (11/17) | +0.0130 (17/17) | +0.0160 (17/17) |
| full-history summary (upper bound) | +0.0174 | +0.0174 | +0.0174 | +0.0174 (17/17) |
| full-history cluster | +0.0066 | +0.0066 | +0.0066 | +0.0066 (17/17) |

**Allen, ridge** (46 mice)

| Method | 3 min | 10 min | 30 min | full |
|---|---|---|---|---|
| true type | +0.0252 (40/46) | +0.0252 | +0.0252 | +0.0252 |
| naive short summary | −0.0380 (21/46) | −0.0188 (28/46) | +0.0403 (43/46) | +0.0473 (43/46) |
| per-neuron ridge | −0.1222 (5/46) | −0.0407 (20/46) | +0.0445 (41/46) | +0.0527 (40/46) |
| E2 hard | +0.0418 (42/46) | +0.0449 (43/46) | +0.0466 (42/46) | +0.0466 (42/46) |
| **E2 soft** | **+0.0499** (45/46) | **+0.0530** (46/46) | **+0.0517** (45/46) | +0.0517 (45/46) |
| full-history summary (upper bound) | +0.0473 | +0.0473 | +0.0473 | +0.0473 (43/46) |

**Allen, MLP**

| Method | 3 min | 10 min | 30 min | full |
|---|---|---|---|---|
| true type | +0.0239 (43/46) | +0.0239 | +0.0239 | +0.0239 |
| naive short summary (continuous) | −0.8319 (0/46) | −0.2487 (8/46) | +0.0340 (44/46) | +0.0436 (44/46) |
| naive short cluster | −0.0813 (16/46) | −0.1048 (21/46) | +0.0270 (43/46) | +0.0302 (43/46) |
| per-neuron MLP fine-tune | +0.0284 (42/46) | **+0.0460** (45/46) | **+0.0524** (45/46) | **+0.0513** (45/46) |
| E2 hard | +0.0286 (42/46) | +0.0274 (41/46) | +0.0298 (42/46) | +0.0298 (42/46) |
| **E2 soft** | **+0.0339** (45/46) | +0.0346 (45/46) | +0.0335 (45/46) | +0.0335 (45/46) |
| E2 continuous (predicted summary) | −0.0636 (21/46) | −0.0341 (30/46) | +0.0303 (43/46) | +0.0303 (43/46) |
| full-history summary (upper bound) | +0.0436 | +0.0436 | +0.0436 | +0.0436 (44/46) |
| full-history cluster | +0.0302 | +0.0302 | +0.0302 | +0.0302 (43/46) |

### Key comparisons (`stats.csv`; two-sided Wilcoxon; Bugeon over 17 sessions, smallest possible p = 1.5e-5, plus mouse signs; Allen over 46 mice)
Format: units where A beats B, mean difference A − B, p. **Bold** = the better side at that H with p < 0.01.

**E2 soft vs per-neuron model**

| H | Bugeon ridge | Bugeon mice | Bugeon MLP | Bugeon mice | Allen ridge | Allen MLP |
|---|---|---|---|---|---|---|
| 3 min | **17/17, +0.060, 1.5e-5** | 4/4 | **14/17, +0.010, 0.002** | 3/4 | **45/46, +0.172, 1e-13** | 30/46, +0.005, 0.008 |
| 10 min | **14/17, +0.019, 5e-4** | 4/4 | 10/17, +0.0000, 0.71 | 3/4 | **39/46, +0.094, 3e-7** | **8/46, −0.011, 3e-6** (fine-tune better) |
| 30 min | 9/17, −0.002, 0.68 | 2/4 | **1/17, −0.007, 1.5e-4** (fine-tune better) | 1/4 | 27/46, +0.007, 0.35 | **2/46, −0.019, 4e-12** (fine-tune better) |
| full | **0/17, −0.015, 1.5e-5** (per-neuron better) | 0/4 | **0/17, −0.013, 1.5e-5** (fine-tune better) | 0/4 | 20/46, −0.001, 0.74 | **4/46, −0.018, 2e-9** (fine-tune better) |

- E2 continuous vs fine-tune at 3 min: Bugeon 14/17, +0.008, p 4e-4 (4/4 mice). Allen 14/46, −0.092, p 1e-5 (E2 continuous fails in Allen, see below)

**E2 soft vs naive short summary** (ridge: naive clusters; MLP: naive clusters, and E2 continuous vs naive continuous)

| H | Bugeon ridge | Bugeon MLP (soft vs naive cluster) | Bugeon MLP (E2 cont. vs naive cont.) | Allen ridge | Allen MLP (soft vs naive cluster) | Allen MLP (E2 cont. vs naive cont.) |
|---|---|---|---|---|---|---|
| 3 min | 17/17, +0.0086, 1.5e-5 (4/4) | 17/17, +0.0041 (4/4) | 17/17, +0.094 (4/4) | 42/46, +0.088, 4e-10 | 41/46, +0.115, 6e-11 | 45/46, +0.77, 1e-13 |
| 10 min | 17/17, +0.0083 (4/4) | 17/17, +0.0045 (4/4) | 17/17, +0.018 (4/4) | 43/46, +0.072, 7e-13 | 44/46, +0.139, 1e-13 | 44/46, +0.22, 1e-11 |
| 30 min | 17/17, +0.0047 (4/4) | 17/17, +0.0026 (4/4) | 13/17, +0.003, 0.006 (4/4) | 36/46, +0.011, 2e-5 | 37/46, +0.006, 5e-5 | 24/46, −0.004, 0.54 |

**E2 vs true type**
- **E2 soft beats the genetic label at every H in both datasets, with both model families:**
  - Bugeon ridge 17/17 (+0.011 to +0.015, 4/4 mice); Bugeon MLP 17/17 (+0.005 to +0.006, 4/4)
  - Allen ridge 39–43/46 (+0.025 to +0.028, p ≤ 3e-8); Allen MLP 36–40/46 (+0.010, p ≤ 2e-5)
- **E2 hard also beats it,** except Allen MLP at 3 and 10 min (29/46, p 0.11; 31/46, p 0.08)

**Per-neuron model vs the full-history shared model (summary from the whole train part)**

| H | Bugeon ridge | Bugeon MLP | Allen ridge | Allen MLP |
|---|---|---|---|---|
| 3 min | 1/17, −0.060 | 0/17, −0.021 | 1/46, −0.170 | 6/46, −0.015, 2e-6 |
| 10 min | 3/17, −0.017, 0.002 | 1/17, −0.010 | 11/46, −0.088 | 20/46, +0.002, 0.24 |
| 30 min | 11/17, +0.004, 0.16 (3/4 mice) | 3/17, −0.003, 0.02 | 21/46, −0.003, 0.81 | **33/46, +0.009, 1e-4** |
| full | **17/17, +0.017, 1.5e-5** (4/4) | **16/17, +0.004, 3e-5** (4/4) | 27/46, +0.005, 0.12 | 28/46, +0.008, 0.01 |

**Soft vs hard:** soft beats hard at every H, in both datasets and both families (Bugeon ridge 17/17 every H; Allen ridge 39–44/46). The soft mixture of cluster models even beats hard assignment by the full-history summary from 10 min on:
- Bugeon ridge: E2 soft vs full-history clusters at 10 min 16/17, +0.0019, p 3e-4
- Allen ridge: 36/46, +0.006, p 3e-4

### What the recogniser achieves (`recogniser.csv`, `recogniser_folds.csv`, `recogniser_selection.csv`; plots (c) of `step14_atlas.png`)
Held-out mice. One window of each length; "accumulated" = tiles inside H, with the tile length each fold picked on inner CV.

| | Bugeon | Allen |
|---|---|---|
| **atlas cluster (16), one window: accuracy / balanced acc.** | | |
| 30 s | 0.37 / 0.27 | 0.48 / 0.35 |
| 3 min | 0.60 / 0.51 | 0.57 / 0.46 |
| 10 min | 0.68 / 0.61 | 0.65 / 0.58 |
| 30 min | 0.75 / 0.69 | 0.77 / 0.72 |
| chance: largest cluster / 1 / 16 | 0.14 / 0.06 | 0.16 / 0.06 |
| **atlas cluster, accumulated over H: accuracy** | | |
| H = 3 min (naive 3-min summary → nearest centroid, step 13) | **0.52** (0.43) | **0.55** (0.26) |
| H = 10 min (naive) | **0.63** (0.53) | **0.66** (0.35) |
| H = 30 min (naive) | **0.73** (0.69) | **0.78** (0.71) |
| H = full | 0.80 | 0.78 |
| **genetic type, one window: balanced acc.** (chance 1/6 Bugeon, 1/4 Allen) | | |
| 30 s → 30 min | 0.28 → 0.39 | 0.61 → 0.88 |
| accumulated, H = 3 min / full | 0.32 / 0.39 | 0.69 / 0.89 |
| **short → long regressor: R² of the full-history summary** | | |
| one window 30 s / 3 / 10 / 30 min | 0.11 / 0.55 / 0.70 / 0.78 | 0.14 / 0.37 / 0.52 / 0.72 |
| accumulated H = 3 min / 10 min / 30 min / full | 0.29 / 0.55 / 0.78 / 0.83 | 0.35 / 0.57 / 0.74 / 0.74 |

- **The recogniser recovers the long-history cluster far better than the naive short summary:**
  - With 3 min: 52% vs 43% (Bugeon) and 55% vs 26% (Allen)
  - With 10 min: 63% vs 53% and 66% vs 35%
  - The naive summary is a long-history statistic computed on a short window. Its values are out of the range seen in training, and stimulus features are missing (Allen's first 10 min are gratings only), so the nearest centroid is often wrong. The recogniser was trained on short windows, so it knows what a short window of each cluster looks like
- **Evidence accumulation helps less than a longer window:**
  - Allen at H = 3 min: 6 × 30 s tiles gave 0.55, vs 0.57 for one 3-min window of the train part
  - The inner CV picked one long tile wherever the history allowed it (30 min for H ≥ 30 min)
  - Summing short windows does not replace long ones, probably because the windows of one neuron are not independent
- **The atlas cluster is recognised as well as or better than the genetic type** (balanced accuracy 0.51 vs 0.35 at 3 min in Bugeon)
  - In Allen the genetic type is easier (0.73 at 3 min), because there the line is confounded with the recording (step 11)

### Task 3 (K4): neuron identity or session / brain state? (`k4_session.csv`, `k4_state.csv`, `plots/step14_k4.png`)
Full-history summaries. Gain over no label (units up), and its share of the full-summary gain:

| Summary | Bugeon ridge | Bugeon MLP | Allen ridge | Allen MLP |
|---|---|---|---|---|
| full summary | +0.0133 (17/17) | +0.0174 (17/17) | +0.0473 (43/46) | +0.0436 (44/46) |
| (a) centred within session (no session information) | +0.0099 (17/17, 4/4 mice) **75%** | +0.0074 (17/17, 4/4) **42%** | −0.0020 (34/46, p 0.07) −4% | +0.0042 (38/46, p 0.002) 10% |
| (b) session mean only (no neuron information) | +0.0041 (12/17, p 0.16; 3/4 mice) 31% | +0.0019 (10/17, p 0.07; 2/4) 11% | +0.0246 (38/46, p 3e-6) 52% | +0.0238 (41/46, p 9e-8) 54% |
| (a) vs (b) | 14/17, +0.0058, p 0.03 (4/4) | 14/17, +0.0055, p 0.003 (4/4) | 27/46, −0.027, p 0.96 | 28/46, −0.020, p 0.71 |

- **Bugeon (types mixed within each recording): the gain is mostly the neuron's own identity within the session, not session state:**
  - Removing all between-session differences keeps 75% (ridge) and 42% (MLP) of the gain, in 17/17 sessions and 4/4 mice
  - The session mean alone recovers only 31% / 11%, not significantly (p 0.16 / 0.07)
- **Allen (one Cre line per mouse): the gain is mostly between recordings:**
  - The session mean alone recovers about half (52–54%)
  - The within-session part alone gives nearly nothing (MLP 10%; ridge −4%). The ridge per-cluster models use equal row weights, which costs −0.021 in Allen by itself (step 13's k = 1 reference), so the MLP figure is the cleaner one
  - In Allen "session" = mouse = Cre line, reporter, depth and imaging conditions, so a session-level summary carries the line as well as brain state; it cannot be split further here
  - Neither (a) nor (b) alone reaches the full summary: the network uses the combination
- **Running vs stationary test windows (the summary's gain over no label, per state):**

  | | Bugeon ridge | Bugeon MLP | Allen ridge | Allen MLP |
  |---|---|---|---|---|
  | running windows | +0.0174 (17/17) | +0.0202 (17/17) | +0.0475 (39/45) | +0.0348 (41/45) |
  | stationary windows | +0.0060 (12/17) | +0.0116 (17/17) | +0.0375 (37/43) | +0.0433 (41/43) |
  | running − stationary, paired over sessions | 16/17, +0.011, p 3e-5 | 16/17, +0.009, p 0.002 | 19/45, +0.010, p 0.73 | 23/45, −0.009, p 0.92 |
  | (a) centred: running / stationary | +0.0143 / +0.0012 | +0.0119 / −0.0019 | −0.032 / +0.017 | +0.0048 / −0.0040 |

  - **In Bugeon the summary helps mostly while the mouse runs** (16/17 sessions), and its session-free part helps only then
  - The true type gives +0.0031 running and −0.0034 stationary (ridge)
  - This fits a summary that describes how each neuron responds when the network is driven (arousal / locomotion modulation differs between neurons), rather than a session-wide offset
  - Allen shows no difference between states

### Task 4 (K3, Bugeon): a summary from another session of the same cell (`k3_cross_session.csv`, `plots/step14_k3.png`)
741 neuron-sessions (316 inhibitory cells). Gain over no label, mean over 17 sessions (sessions up):

| Summary | ridge | MLP (continuous) | MLP (cluster) |
|---|---|---|---|
| true type | +0.0022 (11/17) | +0.0020 (14/17) | |
| same-session summary | +0.0098 (16/17) | +0.0145 (17/17) | +0.0036 (15/17) |
| **other-session summary (same cell)** | **−0.0001 (8/17, p 0.93)** | **−0.0109 (3/17, p 0.03)** | −0.0027 (4/17) |
| other vs same session | 0/17, −0.0099, p 1.5e-5 (0/4 mice) | 0/17, −0.025, p 1.5e-5 (0/4) | 0/17, −0.0063 |

- **The summary does not transfer between days.** From another session of the same cell it gives nothing (ridge) or hurts (MLP, −0.011, worse than the genetic label in 14/17 sessions)
- The UniqueID matching is sound:
  - The other-session summary is closer to the cell's own than a random same-type cell's in 70% of cells (median distance 5.0 vs 6.0 z-units)
  - Stimulus-tuning features are stable across sessions (r = 0.6–0.7: grating on − off, tuning peak, direction responses)
  - Kurtosis (r = 0.09), mean activity per block (0.15–0.19), skew (0.30), OSI / DSI (0.26–0.27) and the autocorrelations (0.3–0.4) are not
- **Together with task 3:** in Bugeon the useful part of the summary is specific to the neuron, but it is the neuron's state in this session, not a fixed identity carried across days. Part of the instability may also come from the recording (ROI extraction and deconvolution per session)

### Key question: with only minutes of a new neuron's activity, does mapping it onto an atlas learned from long recordings forecast better than fitting a model for that neuron alone, and is the gain the neuron's own identity rather than session state?
- **With 3 min: yes, the atlas wins, in both datasets and both model families:**
  - E2 soft vs the per-neuron model:
    - Bugeon ridge +0.060 (17/17, 4/4 mice); Bugeon MLP +0.010 (14/17, p 0.002, 3/4 mice)
    - Allen ridge +0.172 (45/46); Allen MLP +0.005 (30/46, p 0.008)
  - At 3 min E2 soft recovers about the whole full-history ridge gain: Bugeon +0.0128 vs +0.0133; Allen +0.0499 vs +0.0473
  - It recovers only 37% (Bugeon) and 78% (Allen) of the continuous-summary MLP gain
  - It also beats the naive short summary at every H < full, in every unit of Bugeon and 36–45/46 Allen mice; the naive summary is harmful at 3–10 min in Allen
- **At 10 min: about a tie.** The per-neuron ridge is still worse (Bugeon −0.004, Allen −0.041), but the per-neuron MLP fine-tune catches up:
  - Bugeon: equal to E2 soft (10/17, p 0.71)
  - Allen: better (+0.011, 38/46)
- **From 30 min on, the neuron's own model is better:**
  - MLP fine-tune vs E2 soft: Bugeon −0.007 (1/17); Allen −0.019 (2/46)
  - With the full train part, the per-neuron model beats every shared model, including the step 13 full-summary one: Bugeon ridge +0.017 (17/17, 4/4), Bugeon MLP +0.004 (16/17, 4/4), Allen MLP +0.008 (28/46, p 0.01). This answers objection K1: with 30+ min, a per-neuron model (or a per-neuron embedding) is at least as good, and with the full train part better
- **Identity vs state:**
  - **Bugeon:** mostly the neuron's own within-session identity (75% / 42% of the gain survive removing every between-session difference; the session mean alone gives 31% / 11%, not significant), concentrated in running periods. But that identity does not carry over to another day (task 4)
  - **Allen:** mostly between recordings (session mean alone 52–54%; within-session part ≈ 0–10%), and recording = mouse = Cre line there
- **Caveat:** 4 Bugeon mice. "4/4 mice" is the honest animal-level statement; session-level p-values treat sessions of a mouse as independent

## Plots
- `plots/step14_atlas.png` (main figure):
  - rows Bugeon / Allen
  - (left) ridge family and (middle) MLP family: gain over no label vs history H for true type, naive short summary (and cluster), per-neuron ridge / MLP fine-tune, E2 hard, E2 soft, E2 continuous and the full-history summary (upper bound); mean ± sem over units; off-scale values printed
  - (right) recogniser accuracy vs window length (one window; atlas cluster accuracy and balanced accuracy, genetic-type balanced accuracy), accumulated over H (stars), and the naive short summary's cluster agreement (grey crosses)
- `plots/step14_k4.png`: task 3. Gain of the full / session-centred / session-mean summaries (ridge and MLP; bars = mean, dots = units, % of the full gain, units up), and the gain in running vs stationary test windows
- `plots/step14_k3.png`: task 4. True type, same-session summary and other-session summary of the same cell, per session (lines join the same session)
- `plots/step14_pairs.png`: per-unit differences of the key comparisons at each H (E2 soft − per-neuron, E2 soft − naive, E2 − true type, per-neuron − full-history summary), with units up and Wilcoxon p

Numbers behind the plots (`results/step14/`, 5.2 MB, almost all `neuron_scores.csv`):

| File | Contents |
|---|---|
| `neuron_scores.csv` | per neuron (both datasets): SSE of copy-last and score of every variant at every H:<br>• ridge: r_none, r_true, r_full, r_naive_h*, r_pn_h* (per-neuron), r_hard_h*, r_soft_h*, r_ctr, r_sess, r_cross<br>• MLP: m_none, m_true, m_clfull, m_contfull, m_clnaive_h*, m_contnaive_h*, m_ft_h* (fine-tune), m_hard_h*, m_soft_h*, m_e2cont_h*, m_ctr, m_sess, m_clcross, m_contcross<br>• h3 / h10 / h30 / full = H |
| `unit_scores.csv` | per unit (Bugeon sessions and mice, Allen mice; also per state of the test windows, and the task 4 cells): score of every variant |
| `gains_by_history.csv` | main figure: gain over no label per dataset / family / method / H |
| `stats.csv` | all Wilcoxon tests (key comparisons at each H, each method vs none, task 3); Bugeon with per-mouse differences |
| `recogniser.csv`, `recogniser_folds.csv`, `recogniser_selection.csv` | recogniser and regressor accuracy per window length and per H (pooled / per fold), the naive summary's cluster agreement, and the inner-CV model and tile-length choices |
| `k4_session.csv`, `k4_state.csv` | task 3: session-centred / session-mean summaries; gains by running / stationary test windows |
| `k3_cross_session.csv` | task 4: tests on the 741 cells; summary distances; per-feature stability between sessions |
| `finetune_info.csv`, `per_neuron_ridge_info.csv` | per H: fit / val windows, fine-tune steps, val loss before / after, embedding shift; per-neuron ridge alphas |
| `mlp_fits.csv`, `history_minutes.csv`, `reproduction_check.csv` | MLP epochs and val losses; minutes per H; reproduction of step 13 |

## Problems / errors
- **The per-neuron fine-tune is probably slightly under-trained in Bugeon:**
  - Every Bugeon fine-tune ran to the 500-step cap (Allen: 390–440 steps on average); each neuron keeps its best-val embedding
  - The step size was halved midway through (batch 64 → 32) for run time; only the batch-32 run is reported
  - With longer training the per-neuron MLP could gain a little more at 10–30 min. Its 3-min result is limited by data (val loss fell 13% with 3 min, but test performance did not follow)
  - The 3-min Bugeon val set is small (65 windows per neuron)
- **Recogniser model search was reduced for run time.** A first run also tried logistic C = 0.1 and a full search for the genetic-type target. It was too slow on the shared CPU and was stopped. Allen fold 0 of that run had already finished, and there the type target picked logistic regression; that fold was discarded and all folds rerun with the reduced search. Nothing from the first run is reported
- **Out-of-memory kills:** with several phases running in parallel, the kernel killed three Bugeon MLP workers and the Bugeon recogniser once
  - MLP fits are checkpointed: 4 fits were lost and rerun with the same seeds
  - The recogniser restarted from its checkpoint (fold 0 kept)
  - Reproduction checks and fit logs show no effect
- **E2 continuous fails in Allen at 3–10 min** (−0.064, −0.034), although it beats the naive summary by +0.77 at 3 min
  - The regressor predicts the full summary with R² 0.35 at 3 min. Its predictions are shrunk to the mean in an unusual direction, and the FiLM layer of the Allen cont network is very sensitive to that (it was trained only on full summaries)
  - The soft cluster mixture is the safe way to use the atlas with little data
- **"E2 at full history" uses 30-min tiles,** not the full summary. That is why it differs from the full-history cluster (ridge E2 soft even beats it)
- **Short Allen histories contain only drifting gratings,** so movie and spontaneous features are missing there. The recogniser was trained with the same pattern (random windows of the train part contain one stimulus type), but the held-out tiles at H ≤ 10 min are always gratings
- **Bugeon 30 s windows** are one contiguous 128-frame piece of one block. Longer windows and H take a piece of every block (≥ 128 frames each), as step 13, so a "3 min" Bugeon window spans all blocks while 30 s spans one
- **Task 3 ridge numbers carry a weighting handicap.** The per-cluster ridge models use equal row weights (label-free), which costs −0.021 in Allen by itself (step 13). The MLP versions are class-balanced like their baseline and are the cleaner comparison
- **Running state:**
  - Allen: running if mean speed over the window > 1 cm/s; 13 windows had no speed and were left out of the state split only
  - Bugeon: "running" = state 0 of `frame.states` for most frames of the window
- **Step 13 discrepancy:** step 13's result file says Allen kept "13–14 PCs". The same pipeline, with identical cluster labels and sizes, keeps 18 PCs (0.953 of the variance) in every fold. Step 13's text looks wrong on this detail; its results are unaffected
- **4 Bugeon mice:** a Wilcoxon test over mice cannot go below p = 0.125, so mouse signs are given. Session tests treat the sessions of a mouse as independent

### Code and ideas from other work
- **This repo:**
  - Step 3: windows, alpha grid
  - Step 4: stimulus windows
  - Step 5: `Gram` solver
  - Step 8: model "a" (per-neuron ridge), sessions, folds
  - Step 9: MLP, FiLM, training settings
  - Step 10: classifier choices (quantile-normal logistic regression vs gradient boosting)
  - Step 11: Allen extract, `s_features`
  - Step 12: Bugeon extract, `s_features`, `neuron_stats`, `assemble`, `fit_weighted`, `wil`
  - Step 13: S summary, atlas clustering, `fit_clusters`, `run_job`, `cond_inputs`, `held_windows`, the history rule
  - All imported and reused, not copied
- **POCO** (pretrained forecaster with per-unit embeddings, fine-tuned for a new recording): the per-neuron MLP fine-tune freezes the trained network and learns only a 16-dim FiLM embedding per new neuron. POCO's zero-initialised FiLM also came in via step 9
- **NeuPRINT** (Mi et al. 2023, a time-invariant per-neuron representation fed to a forecaster): the continuous summary and E2 continuous (a predicted representation)
- **NuCLR** (Arora et al. 2025, neuron identity learned from activity alone): the premise that a label-free representation carries the relevant "type"; Bugeon loading conventions via step 12
- **LOLCAT** (Schneider et al. 2023): classify a neuron from many short windows and pool the evidence (the recogniser's training on random short windows and the accumulation over tiles); inter-event-interval features via step 10
- **Single-cell reference mapping:**
  - **Azimuth / Seurat v4** (Hao et al. 2021): build a reference atlas once from rich data, map a sparse query onto it and transfer labels with probabilities (E2 soft uses those probabilities as mixture weights)
  - **scArches** (Lotfollahi et al. 2022): freeze the reference model and train only a small query-specific part (the per-neuron embedding fine-tune is the analogue)
- No code was copied from these repositories, and none was read for this step

## What I noticed
- The usefulness of an atlas depends on how much of the neuron you have. With minutes, recognising its long-run "type" from a recogniser trained on short windows (soft probabilities) is clearly better than fitting it alone or computing its long-run statistics on too little data; with half an hour or more, a model of the neuron itself wins. A practical rule: soft atlas mapping below ~10 min, a per-neuron embedding above ~30 min.
- What the summary captures is specific to the neuron but not permanent: in Bugeon it survives removing the session average, matters most while the mouse runs, and does not transfer to the same cell on another day. "Dynamics type" here is closer to a neuron's current operating state than to a fixed cell identity.
