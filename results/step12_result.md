# Step 12: Test the cell-type label on mixed-type data (Bugeon et al. 2022)
Date: 2026-10-17

## What was run
- **Task 0, access:** a first run of this step stopped because figshare was blocked. After the network access was widened, `api.figshare.com` and `ndownloader.figshare.com` were reachable. Article 19448531 (version 2) has two files:

  | File | Size | URL | MD5 |
  |---|---|---|---|
  | `Bugeon et al 2022 v2.zip` | 946,468,709 B | https://ndownloader.figshare.com/files/38046633 | 4556ed80ce98ca13f909129e25f60f3a (checked, matches) |
  | `DatasetTypesV2.xlsx` (variable list) | 11,917 B | https://ndownloader.figshare.com/files/38046648 | 0d367c2fbc8f3e64e510247eea57dd7d |

  Both were downloaded to `data/bugeon/` and the zip unpacked to `data/bugeon/raw/` (1.5 GB, not committed). Everything in the zip was kept; only activity, labels, positions, stimulus, behaviour and timing files are read.
- `scripts/step12_bugeon.py` with phases `extract`, `ridge`, `mlp`, `forecast`, `classify` and `assign`, then `scripts/step12_plots.py`, which remakes every plot from the CSV files in `results/step12/` only
- Environment: Python 3.11 (`.venv`), PyTorch 2.14.1 on CPU, scikit-learn 1.7.2, numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, allensdk 2.16.2 (installed only because steps 3–9 import it at module level). Same library versions as step 11; step 11 used Python 3.10
- Wall time on 4 CPU cores:

  | Phase | Time |
  |---|---|
  | extract | 85 s |
  | ridge | 43 s |
  | mlp | 70 min (44 fits) |
  | classify | 9 min |
  | assign | 2 s |

### Preparation (task 1)
- **Labels:** `neuron.ttype.txt` gives `EC` (excitatory), `IN` (inhibitory, not matched to the transcriptomics) or a Tasic et al. 2018 t-type. The subclass is the first token of the t-type, as in NuCLR's and NeuPRINT's loaders. Classes used: EC, Lamp5, Pvalb, Sncg, Sst, Vip
  - **Merged:** Serpinf1 (only t-type Serpinf1-Aqp5-Vip, 15 neuron-sessions) merged into Vip; it is a Vip-expressing type. NuCLR and NeuPRINT drop it instead, which changes their 5-subclass counts by 15 of 1,515 cells
  - Sncg is the smallest class (53 neuron-sessions, 37 unique cells, 8–23 per mouse). It was kept separate, because it is one of the literature's 5 subclasses; all Sncg numbers are noisy
  - **Excluded:** the 1,954 unmatched `IN` neuron-sessions (about 115 per session) have no type, so they are left out of every model and score. No neuron had zero variance in its train part
- **No resampling:** the data are already one value per imaging volume at ~4.3 Hz (dt = 0.233 s), below the ~10 Hz of steps 3–11. Windows are past 13 frames (3.0 s) → next 4 frames (0.93 s)
- **Split:** the blocks of a session (Blank, Drifting Gratings, Natural Scenes 01 / 02) are separate recordings, each with its own clock starting at ~16 s. So the chronological 80/20 split is made inside every block:
  - The last 10% of each block's train part is val
  - z-score per neuron with the mean and std over the train parts of all blocks
  - Windows never cross a block edge. Test windows = last 20% of every block, stride 4 frames (non-overlapping targets), as before
- **Stimulus features**, 17 per frame (step 4 style; each value is the fraction of the frame covered):
  - Block type: gratings / natural scenes / blank
  - Grating on (contrast > 0)
  - 12 grating directions (one-hot while on)
  - Natural image on (non-blank image)

  Inputs per neuron: own past 13 frames + 17 × (13 past + 4 next) stimulus values = 302. Plane time offsets (≤ 0.2 s) are ignored
- **Score** = 1 − MSE / MSE_copylast, as before. Session score = 1 − ΣSSE / ΣSSE_copylast over the session's labelled neurons
- **Folds:** leave one mouse out (4 folds, 3 training mice each). All sessions of a mouse are always on the same side, so no neuron that appears in several sessions (`neuron.UniqueID`) is ever in train and test at once. This also holds for EC cells, which have no ID but can only reappear within the same mouse

### Models (task 2)
- **Ridge**, step 5 closed form, alpha grid of step 3, alpha picked on val, refit on train + val:
  - Per neuron, the extract phase saves the sufficient statistics (X'X split into own-past and stimulus blocks, X'Y, Y'Y, sums), so any weighted group of neurons is fit from sums. Checked against a direct fit: relative error ~1e-8 in X'X and X'Y, and identical val MSE
  - **b**: shared, no label, class-balanced: each type gets the same total row weight, as step 7's line balance
  - **d**: one model per type. All 6 models are applied to every held-out neuron
  - **e**: shuffled-type control. The training neurons' type labels are permuted (same group sizes as the true types), and one class-balanced model is fit per fake group. Held-out neurons get a permutation of their own labels. 5 seeds
- **MLP**: step 9's `MLP` class (2 × 256 hidden, FiLM from a 16-dim label embedding, zero-initialised gamma / beta), with input 302 and output 4
  - Same as step 9: Adam lr 1e-3, batch ~1,024 (170 rows per type), 400 batches per epoch, at most 40 epochs, early stopping on a class-balanced val MSE (20,000 val rows per type, patience 5), and the same seeding formula
  - Batches are class-balanced over the 6 true types (step 9 balanced over Cre lines)
  - **m1**: no label (3 seeds). **m2**: type FiLM (3 seeds); every held-out neuron is evaluated under all 6 labels. **m3**: shuffled FiLM with the same permutations as ridge e (5 seeds)
- **Shuffled at test time** (as step 11's harsh control): the true labels permuted within each session, scored with the true-label models (20 permutations)
- `results/step12/neuron_scores_all_labels.csv` has, for each of the 8,199 neuron-sessions:
  - Test SSE and score under copy-last, b, d with each of the 6 type labels, e, m1, m2 with each of the 6 labels (mean over 3 seeds) and m3
  - Mouse, session, t-type, unique ID, depth and plane

  Any future label set can be scored from this file alone (function `scores_for`)

### Classifier (task 3)
- **S features** (step 11 "S", train part of every block only, adapted to 4.3 Hz and this stimulus set), 40 features:
  - Skew, kurtosis
  - Event rate and time above z = 2
  - 6-bin IEI histogram
  - Autocorrelation at 1, 2, 4, 9 and 21 frames (0.23–4.9 s)
  - Power in 4 bands up to 2.2 Hz
  - Mean z per block type (3)
  - Grating on − off; natural image on − off
  - 12 direction responses, tuning peak, DSI, OSI

  Step 10 / 11's "z std (10 Hz)" is dropped: at the native rate it is 1 by construction. Temporal features use only frame pairs inside one block
- **S+state** (extra, not in step 11): S plus 3 behaviour-state features of the neuron: correlation with running speed, correlation with pupil area, and mean z running − stationary (`frame.states`). This was added because Bugeon et al.'s main finding is that state modulation follows transcriptomic type
- **Classifiers:** step 11's `fit_featset` (step 10's quantile → logistic regression with C ∈ {0.01, 0.1, 1}, and histogram gradient boosting; balanced class weights). The inner CV is over the 3 training mice (3 folds) and picks C and the classifier
  - Trained on the 3 training mice, predicts the held-out mouse
  - 6-class versions (EC + 5 subclasses) give the labels used for forecasting
  - 5-class inhibitory-only versions (trained and tested on inhibitory neurons) are the like-for-like comparison with NuCLR / NeuPRINT
- **Fraction of the true-label gain recovered** = mean(pred − none) / mean(true − none), with a 95% bootstrap CI (2,000 resamples). Computed over the 17 sessions (main) and over the 4 mice, using step 10's `recovered`

## Data
- **Experiment id:** Bugeon, Carandini & Harris, figshare 10.6084/m9.figshare.19448531.v2, for Bugeon et al. 2022, Nature 607:330. 4 mice (SB025: 6 sessions, SB026: 3, SB028: 5, SB030: 3), 17 sessions, 2019-10-04 to 2020-01-28
- **Area / depth:** mouse V1, 7 imaging planes per session. Neuron depth −19 to 397 µm (as given in `neuron.depth`). Median depth by type: Lamp5 67 µm, Sncg 125 µm, EC 159 µm, Vip 173 µm, Sst 186 µm, Pvalb 194 µm. Many types share every field of view, all with the same GCaMP indicator
- **Activity:** `frame.neuralActivity` is a **deconvolved** calcium trace (arbitrary units, ~79% zeros), not dF/F as in Allen
- **Frames / rate / duration:** 4.27–4.29 Hz. Per session 27,117–32,515 frames (105–127 min) in 4 blocks (SB030 2020-01-28: 5, with two Blank blocks):

  | Block | Mean length (min) |
  |---|---|
  | Blank (grey screen) | 21.6 |
  | Drifting Gratings | 32.3 |
  | Natural Scenes 01 | 29.7 |
  | Natural Scenes 02 | 29.7 |

  Windows per neuron and session: train ~21,000, val ~2,300, test ~1,450 (`results/step12/sessions.csv`, `blocks.csv`)
- **Stimuli:**
  - Drifting gratings: 740 trials per block; 12 directions (30° steps); sizes 5, 15 and 60°; 20 blank trials (contrast 0); ~0.5 s on, ~2.1 s gap
  - Natural scenes: ~1,050 presentations per block of 997 distinct images, ~0.5 s on, ~1.2 s off, ~53 blank
  - Blank: grey screen (spontaneous)
  - `neuron.rfMap` (sparse-noise receptive fields) is in the data but was not used
- **Behaviour:**
  - Running speed (64 Hz), pupil area and position (30 Hz)
  - Cortical state per frame (0 running, 1 stationary desynchronised, 2 stationary synchronised; state 2 occurs only in Blank blocks)
  - Fraction of frames running: Blank 0.33, gratings 0.53, natural scenes 0.68 (means over blocks; range 0.02–1.0)
- **Labels to cells:** one t-type line per ROI in `neuron.ttype.txt` (rows = columns of `frame.neuralActivity`). `neuron.prob_ttypes` and `neuron.gene_count` exist for matched inhibitory cells only
- **Neurons:** 10,153 ROIs, of which 8,199 labelled neuron-sessions are used

  | | EC | Lamp5 | Pvalb | Sncg | Sst | Vip (+Serpinf1) | IN (unmatched, excluded) |
  |---|---|---|---|---|---|---|---|
  | neuron-sessions | 6,684 | 543 | 436 | 53 | 113 | 370 (15) | 1,954 |
  | unique cells | (no ID) | 387 | 307 | 37 | 93 | 266 | (no ID) |
  | SB025 / SB026 / SB028 / SB030 | 1,464 / 945 / 2,865 / 1,410 | 224 / 125 / 86 / 108 | 152 / 58 / 140 / 86 | 23 / 10 / 12 / 8 | 41 / 13 / 33 / 26 | 193 / 44 / 86 / 47 | |

  - Per session: 89–735 labelled neurons; 36–666 EC (mean 393) and 46–165 identified inhibitory (mean 89). This matches the paper
  - Two shallow sessions (SB025 2019-10-04, SB026 2019-10-11) have almost only Lamp5 among the inhibitory cells
- **Neurons tracked across sessions:** 1,090 unique matched inhibitory cells (this matches the task's 1,090). 774 appear in 1 session, 226 in 2, 71 in 3 and 19 in 4, always within one mouse. EC and IN cells have no unique ID

## Results

### Task 2: does the true type label help when types are mixed?
Mean over the 17 sessions of the session score (`results/step12/scores.csv`):

| Model | Ridge | MLP |
|---|---|---|
| no label (b / m1) | 0.4960 | 0.5085 |
| **true type (d / m2)** | **0.4974** (+0.0014) | **0.5103** (+0.0018) |
| shuffled-type control, trained (e / m3) | 0.4958 (−0.0001) | 0.5086 (+0.0000) |
| true labels shuffled at test time (20×) | 0.4951 (−0.0009) | 0.5094 (+0.0008) |
| S-inferred type (task 3) | **0.5022** (+0.0062) | **0.5112** (+0.0027) |

For scale, in Allen (one Cre line per recording) the label gain was +0.025 for ridge and +0.024 for MLP on scores of 0.30 / 0.40 (steps 10–11 per-mouse means). Here it is 15–20× smaller on scores of 0.50.

Paired Wilcoxon over the **17 sessions** (two-sided; with n = 17 the smallest possible p is 1.5e-5; `results/step12/stats.csv`):

| Comparison | Ridge: sessions up, mean diff, p | MLP: sessions up, mean diff, p |
|---|---|---|
| true vs none | 9/17, +0.0014, **0.26** | 14/17, +0.0018, **2.1e-4** |
| true vs trained-shuffled control | 9/17, +0.0015, 0.21 | 14/17, +0.0017, 2.1e-4 |
| trained-shuffled control vs none | 3/17, −0.0001, 5.0e-4 | 10/17, +0.0000, 0.75 |
| test-time shuffled vs true | 0/17, −0.0023, 1.5e-5 | 0/17, −0.0009, 1.5e-5 |
| S-inferred vs none | **17/17**, +0.0062, 1.5e-5 | **17/17**, +0.0027, 1.5e-5 |
| S-inferred vs true | **17/17**, +0.0048, 1.5e-5 | **17/17**, +0.0009, 1.5e-5 |

Paired Wilcoxon over **neurons** (8,199 neuron-sessions; "unique cells" averages a cell's sessions first, 7,774 units):

| Comparison | Ridge | MLP |
|---|---|---|
| true vs none, neurons | 4,314 up, median +0.0005, mean −0.0003, p 5.4e-7 | 5,867 up, median +0.0016, p < 1e-300 |
| true vs none, unique cells | 4,049 / 7,774, p 5.3e-5 | 5,542 / 7,774, p 4.4e-300 |

Neurons of one session are not independent, so these p-values overstate the evidence.

Per mouse (score gain of the true label; signs only):
- Ridge: SB025 +0.0069, SB026 +0.0025, SB028 −0.0012, SB030 −0.0050 (2 of 4 up)
- MLP: SB025 +0.0027, SB026 +0.0016, SB028 +0.0020, SB030 −0.0004 (3 of 4 up)

**With 4 mice, a Wilcoxon test over mice cannot reach p < 0.125 even if all 4 agree. The study has little statistical power at the level of animals.** The session- and neuron-level tests treat sessions and neurons as independent, which they are not: sessions of a mouse share cells, and neurons of a session share state.

Per session (score; `results/step12/scores.csv`):

| Session | n | Ridge b | Ridge d | Ridge e | Ridge d(S) | MLP m1 | MLP m2 | MLP m3 | MLP m2(S) |
|---|---|---|---|---|---|---|---|---|---|
| SB025 10-04 | 89 | 0.4858 | 0.4937 | 0.4857 | 0.4989 | 0.4989 | 0.5025 | 0.4979 | 0.5025 |
| SB025 10-07 | 376 | 0.4941 | 0.5026 | 0.4942 | 0.5056 | 0.5038 | 0.5071 | 0.5032 | 0.5073 |
| SB025 10-08 | 414 | 0.4992 | 0.5069 | 0.4992 | 0.5097 | 0.5077 | 0.5108 | 0.5069 | 0.5108 |
| SB025 10-09 | 385 | 0.4906 | 0.4981 | 0.4907 | 0.5023 | 0.5008 | 0.5043 | 0.5002 | 0.5047 |
| SB025 10-23 | 429 | 0.4914 | 0.4978 | 0.4913 | 0.4996 | 0.5080 | 0.5098 | 0.5084 | 0.5109 |
| SB025 10-24 | 404 | 0.5048 | 0.5086 | 0.5047 | 0.5139 | 0.5223 | 0.5238 | 0.5227 | 0.5258 |
| SB026 10-11 | 188 | 0.4948 | 0.4922 | 0.4944 | 0.4970 | 0.5056 | 0.5071 | 0.5056 | 0.5091 |
| SB026 10-14 | 476 | 0.5093 | 0.5124 | 0.5092 | 0.5139 | 0.5220 | 0.5238 | 0.5219 | 0.5246 |
| SB026 10-16 | 531 | 0.5041 | 0.5072 | 0.5040 | 0.5086 | 0.5178 | 0.5193 | 0.5175 | 0.5199 |
| SB028 11-06 | 662 | 0.4875 | 0.4872 | 0.4873 | 0.4926 | 0.5017 | 0.5039 | 0.5024 | 0.5052 |
| SB028 11-07 | 624 | 0.4938 | 0.4925 | 0.4936 | 0.4970 | 0.5059 | 0.5077 | 0.5068 | 0.5084 |
| SB028 11-08 | 735 | 0.4888 | 0.4889 | 0.4886 | 0.4947 | 0.5013 | 0.5033 | 0.5018 | 0.5047 |
| SB028 11-12 | 469 | 0.4933 | 0.4906 | 0.4931 | 0.4992 | 0.5089 | 0.5109 | 0.5094 | 0.5120 |
| SB028 11-13 | 732 | 0.4786 | 0.4756 | 0.4784 | 0.4833 | 0.4935 | 0.4952 | 0.4938 | 0.4960 |
| SB030 01-08 | 531 | 0.5105 | 0.5057 | 0.5103 | 0.5112 | 0.5182 | 0.5181 | 0.5187 | 0.5188 |
| SB030 01-10 | 519 | 0.5111 | 0.5081 | 0.5109 | 0.5129 | 0.5206 | 0.5205 | 0.5206 | 0.5211 |
| SB030 01-28 | 635 | 0.4938 | 0.4872 | 0.4936 | 0.4966 | 0.5081 | 0.5072 | 0.5082 | 0.5089 |

Per type: score pooled over all sessions, gain of the true label over no label, and paired Wilcoxon over the sessions that contain the type (`results/step12/type_session_scores.csv`, `stats.csv`):

| Type | n | Ridge: d − b | Sessions up, p | MLP: m2 − m1 | Sessions up, p |
|---|---|---|---|---|---|
| EC | 6,684 | +0.0007 | 8/17, 0.49 | +0.0016 | 16/17, 7.6e-5 |
| Lamp5 | 543 | +0.0014 | 16/17, 2.1e-4 | +0.0015 | 15/17, 5.0e-4 |
| Pvalb | 436 | **+0.0047** | 11/16, 0.021 | **+0.0032** | 12/16, 0.0042 |
| Sncg | 53 | +0.0044 | 11/14, 0.0052 | +0.0024 | 9/14, 0.049 |
| Sst | 113 | +0.0001 | 6/15, 0.89 | +0.0006 | 10/15, 0.30 |
| Vip | 370 | **−0.0023** | 2/17, 8.4e-4 (worse) | +0.0008 | 13/17, 0.071 |

- **The true type label helps only a little.** Ridge: +0.0014, not significant over sessions (9/17, p 0.26), and negative in 2 of 4 mice. MLP: +0.0018, consistent over sessions (14/17, p 2e-4) and up in 3 of 4 mice
- The trained shuffled-type controls give nothing (e −0.0001, m3 +0.0000). So the small MLP gain comes from the true type, not from splitting the data into groups
- Per type, the label helps Pvalb most (+0.005 / +0.003), then Sncg and Lamp5
  - It does nothing for Sst and EC with ridge
  - It hurts Vip with ridge (15 of 17 sessions worse): the Vip-only ridge model forecasts Vip neurons worse than the shared model
- MLP fits:
  - Epochs used: median 28.5 (m1), 32 (m2), 34 (m3); 9 of 44 fits reached the 40-epoch cap (`results/step12/mlp_fits.csv`)
  - Ridge alphas (`results/step12/ridge_alphas.csv`): b 1e5–1e6; d_EC 3e5–1e6; interneuron models 3e2–3e4. Alphas are high because every neuron contributes ~21,000 windows

### Task 3: inferred labels
Held-out mice, selected classifier (logistic regression with C = 0.01 in 15 of 16 fold × set fits; `results/step12/classifier_summary.csv`, `confusions.csv`):

| Classifier | Classes | Balanced acc. (chance) | Per fold SB025 / SB026 / SB028 / SB030 | Macro F1 | Recall |
|---|---|---|---|---|---|
| **S** | 6 (EC + 5) | **0.407** (0.167) | 0.317 / 0.480 / 0.530 / 0.395 | 0.260 | EC 0.56, Lamp5 0.31, Pvalb 0.62, Sncg 0.40, Sst 0.40, Vip 0.15 |
| S → E vs I | 2 | **0.643** (0.5) | | | E 0.56, I 0.72 |
| S+state | 6 | 0.407 | 0.313 / 0.506 / 0.532 / 0.386 | 0.265 | Vip 0.21 |
| **S inh** | 5 inhibitory | **0.459** (0.2) | 0.395 / 0.475 / 0.544 / 0.470 | **0.385** | Lamp5 0.52, Pvalb 0.64, Sncg 0.45, Sst 0.54, Vip 0.15 |
| S+state inh | 5 inhibitory | 0.470 | 0.393 / 0.536 / 0.556 / 0.484 | 0.402 | Vip 0.21 |
| S inh, gboost (not selected) | 5 inhibitory | 0.395 | | 0.390 | |

Confusion matrix of the S classifier (6 classes; rows = true, columns = predicted):

| true \ pred | EC | Lamp5 | Pvalb | Sncg | Sst | Vip |
|---|---|---|---|---|---|---|
| EC (6,684) | **3,771** | 707 | 705 | 508 | 635 | 358 |
| Lamp5 (543) | 166 | **168** | 51 | 77 | 34 | 47 |
| Pvalb (436) | 87 | 27 | **271** | 3 | 36 | 12 |
| Sncg (53) | 12 | 9 | 2 | **21** | 0 | 9 |
| Sst (113) | 28 | 10 | 21 | 5 | **45** | 4 |
| Vip (370) | 127 | 67 | 39 | 43 | 37 | **57** |

- **Comparison with the literature (numbers from the task description):** NuCLR reports 0.39 macro F1 for the 5 subclasses on this dataset, and NeuPRINT ~0.6. The S classifier on inhibitory cells reaches macro F1 0.385 (0.402 with state features), at NuCLR's level and below NeuPRINT's
  - The protocols differ:
    - NuCLR's val split also holds out mice, but it drops Serpinf1 and uses a learned representation
    - NeuPRINT is evaluated on held-out neurons, not necessarily held-out mice, and its ~0.6 may be accuracy rather than macro F1
  - So the comparison is approximate
- **Against Allen (step 11):** this is far below step 11's 0.877 balanced accuracy for 4 Cre lines, where each recording held one line and line-wide indicator differences were available. This confirms step 11's caveat that most of the 0.877 came from recording- or indicator-level cues
- Most useful features (held-out permutation importance, logreg; `results/step12/feature_importance.csv`): kurtosis 0.044, autocorrelation at 0.47 s 0.027, event rate 0.027, power 0.5–1 Hz 0.027, skew 0.025, natural image on − off 0.023
  - EC neurons are much sparser (mean kurtosis 63 vs 8–21 for interneurons)
  - Pvalb is the most distinct type (recall 0.62–0.64), Vip the least (0.15)
- **State features add little:** +0.011 balanced accuracy for the 5 inhibitory classes, +0.000 for 6 classes. Correlation with running is in the top 10 for logreg only. Per type, mean z running − stationary is 0.07 for Pvalb vs 0.33 for Vip and Sncg, which is in line with the paper's axis, but it does not improve typing beyond S

Forecasting with the S labels (per-neuron, 6-class, selected classifier). Fraction of the true-label gain recovered = mean(pred − none) / mean(true − none), with a 95% bootstrap CI (`results/step12/gain_recovered.csv`):

| Labels | Ridge: over 17 sessions | Ridge: over 4 mice | MLP: over 17 sessions | MLP: over 4 mice |
|---|---|---|---|---|
| **S** | 4.45 [−34, 37]* | 6.9 [−36, 7.5]* | **1.50 [1.30, 1.86]** | 1.62 [1.31, 6.37] |
| S+state | 4.39 [−30, 37]* | 6.9* | 1.48 [1.28, 1.83] | 1.60 |
| shuffled at test time | −0.65* | −1.9* | 0.47 [0.13, 0.67] | 0.33 |
| trained-shuffled (e / m3) | −0.09 [−0.88, 0.84] | −0.14 | 0.02 [−0.11, 0.18] | 0.03 |

\* With ridge, the true-label gain (+0.0014) is not distinguishable from 0, so the ratio is meaningless. In absolute terms, S labels give **+0.0062 vs +0.0014** for true labels (17/17 sessions better than both no label and true labels).

- **Activity-inferred labels do better than the true labels**, even though they are wrong for most neurons (44% of EC cells are labelled as an interneuron type):
  - MLP: 150% of the true-label gain, 17/17 sessions above true labels (p 1.5e-5)
  - Ridge: more than 4 times the gain (+0.0062 vs +0.0014)
- **Why** (`results/step12/label_pairs.csv`): the classifier sorts neurons by dynamics (sparseness, burstiness, autocorrelation), and the per-type model trained on the most similar-looking population forecasts a neuron better than the model of its true type
  - Mean ridge gain by assigned label:
    - EC neurons labelled Sncg: +0.019
    - EC labelled Pvalb: +0.0065
    - EC labelled EC: +0.0049
    - All EC with the true label: +0.0007
  - Picking the best of the 6 labels for each neuron *on the test data* (an upper bound, not a method) gives session means of 0.505 (ridge) and 0.513 (MLP). With that choice only 35% of all neurons, and 37–39% of EC neurons, prefer the EC model, although 82% of neurons are EC
- Per type with MLP, S labels recover 0.88 (Lamp5), 0.99 (Pvalb), 1.48 (EC) and 1.53 (Sncg) of the true-label gain. Sst and Vip have near-zero true gains, so their ratios are meaningless
- **Shuffled labels within a session** (keeping each session's type mix) still give the MLP +0.0008 over no label (11/17, p 0.035) but stay below the true labels in 17/17 sessions. Most permuted labels of EC neurons are still EC (82% are EC), which explains the positive value

**Key question: when many cell types share one recording and one indicator, does knowing the type still improve forecasting, and can activity-inferred types recover that gain?**
- **Knowing the true type helps only marginally.** With the MLP, the true label adds +0.0018 score (14/17 sessions, p 2e-4; 3 of 4 mice), and the trained shuffled control adds nothing. With ridge the gain (+0.0014) is not significant (9/17 sessions) and goes negative in 2 of 4 mice. In both cases the gain is 15–20× smaller than in the Allen data (+0.024–0.025). There, the label was confounded with the recording and, for Pvalb, with GCaMP6s. Where the true type does help, it is mainly Pvalb (+0.003–0.005), then Sncg and Lamp5; Vip is hurt with ridge
- **Activity-inferred types do not just recover the gain; they exceed it** (MLP 150%, ridge +0.0062 vs +0.0014; 17/17 sessions), although the classifier types neurons poorly:
  - 0.41 balanced accuracy for 6 classes
  - E vs I 0.64
  - 0.39 macro F1 for the 5 inhibitory subclasses, as NuCLR
  - So what helps the forecast is a grouping by dynamics, not the transcriptomic type itself. In this dataset the "type" information useful for forecasting is mostly available from the neuron's own activity history

## Plots
- `plots/step12_bugeon.png`: four panels
  - (a) labelled neuron-sessions per type, stacked by mouse, with unique cells
  - (b) score per model (b, d, e, d with S labels; m1, m2, m3, m2 with S labels): bars = session mean, lines = sessions
  - (c) per-session gain over the no-label model of the true label, of the trained-shuffled control and of S-inferred labels, ridge and MLP, with sessions-up counts and Wilcoxon p
  - (d) confusion matrix of the S classifier (6 classes, held-out mice)
- `plots/step12_types.png`: per type, gain of the true label (pooled bar and one dot per session), of the shuffled control and of S labels, ridge and MLP, with neuron-level Wilcoxon
- `plots/step12_classifier.png`:
  - (a) balanced accuracy and macro F1 of S, S+state and the inhibitory-only classifiers, against NuCLR and NeuPRINT
  - (b–d) confusion matrices of S+state and of the two inhibitory-only classifiers
  - (e–f) top-15 permutation importances

Numbers behind the plots (`results/step12/`, 5.1 MB):

| File | Contents |
|---|---|
| `sessions.csv`, `blocks.csv` | per session and block: ROIs, neurons per type, IN / Serpinf1 counts, unique IDs, depth, frame rate, frames, minutes, windows, running fraction |
| `neurons.csv` | per labelled neuron-session: mouse, session, ROI index, t-type, type, unique ID, depth, plane, fold |
| `neuron_scores_all_labels.csv` | per neuron: test SSE and score under copy-last, b, d with each of 6 labels, e, m1, m2 with each of 6 labels, m3 |
| `scores.csv` | per session, mouse, type and overall, ridge and MLP: none, true, trained control, S, S+state, test-time shuffled |
| `type_session_scores.csv` | per session × type: none, true, S, control |
| `stats.csv` | all Wilcoxon tests (sessions, mice, neurons, unique cells; overall and per type) |
| `gain_recovered.csv` | fraction of the true-label gain recovered, with bootstrap CIs |
| `label_pairs.csv` | mean gain per (true, S-predicted) type pair; best-of-6 upper bound |
| `classifier_summary.csv`, `classifier_folds.csv`, `confusions.csv`, `feature_importance.csv`, `features_by_type.csv` | task 3 classifier |
| `neuron_predictions.csv` | per neuron: classifier probabilities and predictions of every set |
| `ridge_alphas.csv`, `mlp_fits.csv` | fitted alphas; MLP epochs and val loss per fit |

## Problems / errors
- **Frame rate:** the data are ~4.3 Hz, so the step's "~10 Hz" could not be reached without inventing samples. Windows are 13 → 4 frames (3.0 s → 0.93 s). Scores are not directly comparable with steps 3–11 (other rate, deconvolved activity instead of dF/F, other stimuli)
- **Blocks:** the four stimulus blocks of a session have separate clocks and their real order is not in the files. The 80/20 split was therefore done inside every block, so the test set has all stimulus types
- **IN cells:** 1,954 unmatched inhibitory neuron-sessions were dropped (no type). They are 19% of the ROIs, so the population is not complete
- **Serpinf1 merged into Vip** (15 neuron-sessions). NuCLR / NeuPRINT drop it
- **Sncg (53 neuron-sessions, 37 cells) and Sst (113, 93) are small.** Per-type numbers for them are noisy (e.g. the Sst gain ratio)
- **Statistics:**
  - 4 mice give little power: a Wilcoxon over mice cannot go below p = 0.125
  - Sessions of a mouse share cells and conditions, and neurons of a session share brain state, so the session-level and neuron-level p-values overstate the evidence
  - The "unique cells" rows average repeated recordings of matched inhibitory cells. EC cells have no ID, so their repeats (likely, given overlapping depths within a mouse) cannot be merged
- **Fraction recovered with ridge is undefined in practice:** the true-label gain is ~0 (CI of the ratio spans −34 to 37). The absolute gains are the meaningful numbers there
- **Classifier fold variation is large** (6-class balanced accuracy 0.32 for SB025 to 0.53 for SB028). SB025 holds most Lamp5 / Vip cells and two shallow sessions. The inner CV has only 3 training mice
- **Plane offsets** (up to 0.2 s within a frame) were ignored for the stimulus timing
- **Comparison numbers** for NuCLR and NeuPRINT are quoted from the task description, not re-run. Their splits and label sets differ (see task 3)
- **Two bugs fixed during the run:**
  - Autocorrelation feature names depended on each session's exact frame rate (4.27 vs 4.29 Hz), which gave missing columns; they are now named by frame lag
  - The pandas `copy` column clash from step 11 recurred in `scores_for` and was renamed
  - Both were fixed before any reported number was produced

### Code and ideas from other work
- **This repo:**
  - Step 3: alpha grid, 80/20 + 10% val fractions
  - Step 4: stimulus-feature style, "past + next" stimulus window
  - Step 5: `Gram` ridge solver
  - Step 7: line-balanced weighting (here class-balanced)
  - Step 9: `MLP` class with FiLM, optimiser, batch, epochs, early stopping, seeding formula
  - Step 10: `recovered` (fraction of the gain recovered), hand-feature definitions (`EVENT_Z`, `IEI_EDGES`), classifier pipeline
  - Step 11: `fit_featset` (inner-CV classifier selection), S feature set, test-time shuffle control
- **NuCLR** (Arora et al. 2025; github.com/nerdslab/nuclr; `preprocess/bugeon/prepare_data.py`, `src/evaluators/bugeon.py`, read only):
  - File layout and how blocks map to one session
  - Subclass = first token of the t-type; E / I = EC vs everything else
  - Exclusion of `IN` and Serpinf1 from the subclass task (here Serpinf1 was merged instead)
  - Leave-one-subject-out evaluation
- **NeuPRINT** (Mi et al. 2023; github.com/lumimim/NeuPRINT; `data_preprocess.py`, `label_generation.py`, read only): the same 5 subclasses, frame alignment of running speed and pupil (averaging behaviour samples into frames), and use of `frame.states`. Its population-context idea was not used
- **NetFormer** (github.com/NeuroAIHub/NetFormer; `NetFormer/data.py`): only confirms the same loader (`frame.neuralActivity`, `neuron.ttype.txt`, `neuron.UniqueID`); nothing used
- No code was copied from these repositories
- **Bugeon et al. 2022:** the dataset, and the motivation for the extra state features (state modulation along a transcriptomic axis)

## What I noticed
- Once cell type is separated from recording and indicator, the type label is worth very little for forecasting: +0.002 score, against +0.025 in the Allen data, where it was confounded with both. A classifier that types neurons poorly (0.41 balanced accuracy) still gives a larger gain than the true labels. Its labels group neurons with similar dynamics, and a model per dynamics group forecasts better than a model per transcriptomic type.
- So a useful "type" input for forecasting may simply be a summary of the neuron's own long-run activity statistics. The transcriptomic label adds only a little on top for Pvalb and Lamp5 / Sncg; with only 4 mice, even that is weak evidence at the animal level.
