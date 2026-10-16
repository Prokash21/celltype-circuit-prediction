# Step 8: Medium-scale test (all VISp session A mice, 4 Cre lines, plus depth)
Date: 2026-10-16

## What was run
- `scripts/step8_medium.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`). It reuses the preprocessing, stimulus features, model a and batched X'X fitting from `scripts/step5_shared.py`, and the line-balanced b, `row`/`summarize` and model definitions from `scripts/step7_robust.py`, extended to 4 lines / 4 groups. Inputs per neuron: own past 30 bins + 480 stimulus inputs → next 10 bins. Test windows = last 20% of each held-out session
- Two phases:
  - `extract` ran in the background (`nohup`, log `data/step8_extract.log`). It processes one session at a time: download the NWB from the Allen S3 mirror → z-scored 10 Hz dF/F (train/test parts), 10 Hz stimulus features, frame rate, depth and neuron count → `data/cache/<id>.npz` (float32) → delete the NWB. Cached sessions are skipped on restart
  - `fit` reads only the cache (log `data/step8_fit.log`). Model a and the Gram sums are computed per session in 4 worker processes
- Folds: per line, the mice sorted by experiment id are dealt round-robin into 5 groups (group = rank mod 5). Fold k holds out group k of every line and trains on all the other mice of all 4 lines (35–39 training sessions)
- Models (all with stimulus):
  - (b) shared, no label, line-balanced (w_line = N / (4 · N_line))
  - (d) one shared model per Cre line
  - (e) shuffled control for d: the training sessions are split into 4 random equal-size groups, each mixing lines, and each group is fit line-balanced. Held-out sessions get a random group. 5 seeds per fold
  - (a) per-neuron ridge + stim, as a reference
- Metrics: MSE, mean per-neuron r, and score = 1 − MSE / MSE_copylast
  - Per fold and line: pooled over that line's held-out mice in the fold. Summary = mean ± std over the 5 folds (population std)
  - Per held-out mouse: d vs b and d vs e counts, plus a paired two-sided Wilcoxon signed-rank test over held-out mice (scipy)
  - Depth: (d − b) MSE per held-out mouse vs imaging depth, using Spearman ρ and OLS `d−b ~ depth + C(cre)` (statsmodels)

## Data
- 46 sessions (one per mouse, lowest experiment id), VISp, three_session_A, 3,296 neurons:

| Cre line | Mice | Mice per fold (0–4) | Depths (µm, n mice) | Neurons (range per mouse) |
|---|---|---|---|---|
| Cux2-CreERT2 | 13 | 3, 3, 3, 2, 2 | 175 (7), 275 (6) | 2,905 (140–353) |
| Sst-IRES-Cre | 12 | 3, 3, 2, 2, 2 | 265 (1), 275 (6), 285 (1), 375 (4) | 122 (7–15) |
| Vip-IRES-Cre | 12 | 3, 3, 2, 2, 2 | 175 (7), 275 (5) | 153 (4–24) |
| Pvalb-IRES-Cre | 9 | 2, 2, 2, 2, 1 | 175 (2), 185, 195, 200, 225, 275 (1 each), 375 (2) | 116 (3–32) |

- Frames / frame rate / duration: 30.08–30.15 Hz, ~64 min per session → ~10 Hz bins (38,500–38,600 bins). Windows per session, train / val / test: 27,670–28,530 / 3,040–3,136 / 766–790
- Cache: 46 `.npz` files, 449 MB in total (0.4–50 MB each), in `data/cache/` (not committed). No NWB is kept
- Line weights per fold (Cux2, Sst, Vip, Pvalb): 0.28, 6.3–8.4, 4.9–5.8, 6.3–9.1

## Results
Summary over 5 folds (mean ± std; held-out mice pooled per fold):

| Cre line | Model | MSE | mean r | score |
|---|---|---|---|---|
| Cux2-CreERT2 | (a) ref | 0.405 ± 0.064 | 0.271 ± 0.043 | 0.364 ± 0.062 |
| Cux2-CreERT2 | (b) balanced | 0.420 ± 0.050 | 0.233 ± 0.051 | 0.339 ± 0.035 |
| Cux2-CreERT2 | (d) per line | **0.413 ± 0.055** | **0.243 ± 0.051** | **0.351 ± 0.045** |
| Cux2-CreERT2 | (e) shuffled | 0.424 ± 0.049 | 0.231 ± 0.052 | 0.333 ± 0.033 |
| Sst-IRES-Cre | (a) ref | 0.473 ± 0.126 | 0.483 ± 0.062 | 0.418 ± 0.055 |
| Sst-IRES-Cre | (b) balanced | 0.499 ± 0.114 | 0.409 ± 0.068 | 0.384 ± 0.023 |
| Sst-IRES-Cre | (d) per line | **0.481 ± 0.103** | **0.454 ± 0.078** | **0.403 ± 0.024** |
| Sst-IRES-Cre | (e) shuffled | 0.504 ± 0.117 | 0.404 ± 0.063 | 0.378 ± 0.025 |
| Vip-IRES-Cre | (a) ref | 0.233 ± 0.067 | 0.471 ± 0.064 | 0.321 ± 0.044 |
| Vip-IRES-Cre | (b) balanced | 0.249 ± 0.068 | 0.437 ± 0.066 | 0.274 ± 0.041 |
| Vip-IRES-Cre | (d) per line | **0.243 ± 0.068** | **0.459 ± 0.069** | **0.294 ± 0.024** |
| Vip-IRES-Cre | (e) shuffled | 0.250 ± 0.068 | 0.434 ± 0.064 | 0.269 ± 0.043 |
| Pvalb-IRES-Cre | (a) ref | 0.289 ± 0.079 | 0.560 ± 0.111 | 0.304 ± 0.054 |
| Pvalb-IRES-Cre | (b) balanced | 0.330 ± 0.082 | 0.527 ± 0.108 | 0.192 ± 0.102 |
| Pvalb-IRES-Cre | (d) per line | **0.313 ± 0.083** | **0.542 ± 0.110** | **0.241 ± 0.074** |
| Pvalb-IRES-Cre | (e) shuffled | 0.332 ± 0.082 | 0.524 ± 0.107 | 0.186 ± 0.105 |
| All lines | (b) balanced | 0.411 ± 0.040 | 0.257 ± 0.042 | 0.338 ± 0.031 |
| All lines | (d) per line | **0.403 ± 0.044** | **0.269 ± 0.040** | **0.351 ± 0.042** |
| All lines | (e) shuffled | 0.415 ± 0.039 | 0.254 ± 0.042 | 0.332 ± 0.029 |

Score per fold (b / d / e):

| Cre line | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 |
|---|---|---|---|---|---|
| Cux2 | 0.364 / 0.385 / 0.356 | 0.370 / 0.387 / 0.362 | 0.343 / 0.360 / 0.336 | 0.273 / 0.264 / 0.271 | 0.347 / 0.361 / 0.342 |
| Sst | 0.352 / 0.388 / 0.342 | 0.394 / 0.419 / 0.381 | 0.410 / 0.442 / 0.404 | 0.400 / 0.386 / 0.406 | 0.362 / 0.381 / 0.357 |
| Vip | 0.275 / 0.295 / 0.271 | 0.258 / 0.289 / 0.250 | 0.210 / 0.253 / 0.202 | 0.337 / 0.327 / 0.333 | 0.290 / 0.305 / 0.289 |
| Pvalb | 0.186 / 0.225 / 0.183 | 0.002 / 0.111 / −0.010 | 0.286 / 0.325 / 0.275 | 0.277 / 0.295 / 0.276 | 0.207 / 0.250 / 0.205 |

d vs b and d vs e (held-out MSE; e = mean over 5 seeds):

| Cre line | d beats b (folds) | d beats b (mice) | median d − b | Wilcoxon p, d vs b (mice) | d beats e (folds) | d beats e (mice) | median d − e | Wilcoxon p, d vs e (mice) |
|---|---|---|---|---|---|---|---|---|
| Cux2-CreERT2 | 4/5 | 12/13 | −0.0096 | 0.021 | 4/5 | 12/13 | −0.0144 | 0.0046 |
| Sst-IRES-Cre | 4/5 | 9/12 | −0.0192 | 0.027 | 4/5 | 8/12 | −0.0277 | 0.034 |
| Vip-IRES-Cre | 4/5 | 10/12 | −0.0073 | 0.0068 | 4/5 | 10/12 | −0.0080 | 0.0024 |
| Pvalb-IRES-Cre | 5/5 | 9/9 | −0.0200 | 0.0039 | 5/5 | 9/9 | −0.0218 | 0.0039 |
| All lines | 4/5 | 40/46 | −0.0096 | 2.5e-07 | 4/5 | 39/46 | −0.0132 | 2.5e-08 |

- The mice where d loses to b: Cux2 524691284 (+0.017), Sst 596779487 (+0.013), 601423209 (+0.013), 581150104 (+0.004), Vip 650079244 (+0.003), 657650110 (+0.006)
- Fold 3 is the only fold where pooled d loses to b, for Cux2, Sst and Vip. It holds out the same Sst (580163817) and Vip (650079244) mice as step 7 fold 3, where d also lost
- Alphas: b and the d models for Cux2, Vip and Pvalb pick 0.1 (grid minimum) in every fold. The Sst model of d picks 3.2e5 or 1e6, so it is heavily regularized

Depth (held-out mouse, d − b MSE):

| | n | median d − b | d beats b |
|---|---|---|---|
| < 250 µm | 20 | −0.0096 | 19/20 |
| ≥ 250 µm | 26 | −0.0106 | 21/26 |

- All mice: Spearman ρ(depth, d − b) = −0.20, p = 0.18. With the relative gain (d − b)/b, ρ = −0.06, p = 0.67
- OLS `d − b ~ depth + C(cre)`: depth slope −0.0050 per 100 µm, p = 0.19 (R² 0.19, mostly from line). For the relative gain the slope is −0.0029 per 100 µm, p = 0.75
- Within lines, Spearman ρ: Cux2 −0.08 (p 0.79), Sst −0.41 (p 0.19), Vip +0.27 (p 0.40), Pvalb −0.55 (p 0.12)
- The gain does not change significantly with depth. The largest gains are in the 375 µm Sst and Pvalb mice (−0.03 to −0.06), but these mice also have high MSE, and the trend disappears for the relative gain

## Plots
- plots/step8_folds.png: one panel per Cre line. Bars = pooled held-out MSE d − b per fold (blue = d better, red = b better); dots = individual held-out mice
- plots/step8_depth.png: d − b held-out MSE per mouse vs imaging depth (jittered ±4 µm), coloured by Cre line

## Problems / errors
- api.brain-map.org is still blocked. `ophys_experiments.json` and the 46 NWB files came from the Allen S3 mirror. Each NWB (~0.5 GB) was deleted after extraction, so peak disk use was one NWB. No download or extraction errors
- The extracted traces are stored as float32 to keep the cache small, so the numbers differ slightly from step 7 even for the same mouse and training set. For example, the step 7 mice also differ because b, d and e are now trained on all 4 lines and more mice
- Model a is slow: ~3.5 s per neuron with strided-view matmuls, so ~3 h on one core for 3,296 neurons. I stopped the first single-process fit and reran with sessions in 4 parallel processes (~1 h 10 min). Results do not depend on the number of workers
- `results/TEMPLATE.md` does not exist; I used `results/allen_metadata/TEMPLATE.md`
- Small lines: Pvalb mice have 3–32 neurons and fold 4 holds out 1 Pvalb mouse. Sst and Vip held-out mice have 4–24 neurons. Depth is confounded with line: all 375 µm mice are Sst or Pvalb, and Cux2 and Vip only have 175 and 275 µm
- The e seeds reuse the same seeds 0–4 in every fold. Folds 0 and 1 have the same number of training sessions per line, so their shuffled group compositions are identical, though the sessions in them differ

## What I noticed
- With 46 mice and 4 lines, the per-line model (d) beats both the balanced shared model and the shuffled control for every line. It wins for 40/46 held-out mice vs b (p = 2.5e-07), and each line is now significant on its own (p 0.004–0.03). Pvalb, the new line, is the most consistent case (9/9 mice, 5/5 folds), with the largest score gain (+0.05).
- The gain does not depend on imaging depth once Cre line is accounted for. The losses concentrate in a few mice (fold 3 especially) rather than at a particular depth.
