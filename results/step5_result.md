# Step 5: Shared model across sessions (no cell-type label)
Date: 2026-10-11

## What was run
- `scripts/step5_shared.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`); reuses split, z-score, 10 Hz binning, windows (`scripts/step3_linear.py`) and the 12 stimulus features (`scripts/step4_stimulus.py`)
- Inputs for every model and neuron: own past 30 bins + stimulus features for the past 30 and next 10 bins (30 + 40 × 12 = 510 inputs) → next 10 bins
- (a) ridge per neuron + stim, alpha picked **per neuron** by val MSE (last 10% of train), then refit on train + val
- (b) shared ridge + stim: **one** model for all neurons of all training sessions (neuron × window rows pooled, no neuron or Cre-line identity). X'X and X'Y are summed over (session, neuron) batches, reusing the session's stimulus S'S, so the stacked design matrix (18.7 M rows) is never built. One alpha by pooled val MSE, then refit on train + val
- Tests: within-session = last 20% of each session (a, and b trained on all 9 sessions); held-out mouse = b trained on the 6 sessions of mice 1–2 per line, tested on the last 20% of the 3rd mouse per line (same test windows as within-session, so all rows are comparable). The held-out session is z-scored with its own train-part mean/std; nothing else from it is used to fit b
- Score = 1 − MSE_model / MSE_copylast on the same windows. Per Cre line, neurons of the line's sessions are pooled: MSE over all neurons × windows × 10 steps, r = mean per-neuron Pearson r

## Data
- Selection: VISp, three_session_A; per line the lowest experiment id per mouse, then the 3 mice with the lowest ids. The 3rd (highest id) is the held-out mouse

| Cre line | Experiment id | Mouse | Depth (µm) | Neurons | Role |
|---|---|---|---|---|---|
| Cux2-CreERT2 | 501271265 | 222420 | 175 | 215 | train |
| Cux2-CreERT2 | 501704220 | 222426 | 175 | 142 | train |
| Cux2-CreERT2 | 501836392 | 222425 | 175 | 178 | held-out |
| Sst-IRES-Cre | 573720508 | 297620 | 275 | 10 | train |
| Sst-IRES-Cre | 576095926 | 296891 | 285 | 15 | train |
| Sst-IRES-Cre | 580043440 | 297504 | 275 | 10 | held-out |
| Vip-IRES-Cre | 583279803 | 300657 | 175 | 11 | train |
| Vip-IRES-Cre | 606353987 | 326489 | 175 | 13 | train |
| Vip-IRES-Cre | 612543999 | 336243 | 175 | 13 | held-out |

- Frames / frame rate / duration: ~115.5k frames per session, 30.08–30.15 Hz, ~64 min → ~10 Hz bins
- Windows per session train / val / test: ~27,700 / ~3,045 / 766–768
- Training rows for b: 18,679,324 (all 9 sessions), 12,490,622 (6 sessions); Cux2 neurons are 535 of 607 (88%) of the pool

## Results
Within-session test (last 20% of each session, 3 sessions per line):

| Cre line | Neurons | Model | MSE | mean r | score |
|---|---|---|---|---|---|
| Cux2-CreERT2 | 535 | copy-last | 0.574 | 0.191 | 0.000 |
| Cux2-CreERT2 | 535 | (a) ridge per neuron + stim | 0.374 | **0.249** | 0.348 |
| Cux2-CreERT2 | 535 | (b) shared + stim (all 9) | **0.371** | 0.240 | **0.354** |
| Sst-IRES-Cre | 35 | copy-last | 0.863 | 0.318 | 0.000 |
| Sst-IRES-Cre | 35 | (a) ridge per neuron + stim | **0.484** | **0.530** | **0.439** |
| Sst-IRES-Cre | 35 | (b) shared + stim (all 9) | 0.556 | 0.413 | 0.355 |
| Vip-IRES-Cre | 37 | copy-last | 0.354 | 0.555 | 0.000 |
| Vip-IRES-Cre | 37 | (a) ridge per neuron + stim | **0.254** | **0.592** | **0.282** |
| Vip-IRES-Cre | 37 | (b) shared + stim (all 9) | 0.273 | 0.575 | 0.231 |

Held-out mouse (test windows of the 3rd mouse per line; (a) and b (all 9) saw this mouse's train part, b (held-out) did not):

| Cre line | Neurons | Model | MSE | mean r | score |
|---|---|---|---|---|---|
| Cux2-CreERT2 | 178 | copy-last | 0.704 | 0.205 | 0.000 |
| Cux2-CreERT2 | 178 | (a) ridge per neuron + stim | 0.496 | **0.256** | 0.295 |
| Cux2-CreERT2 | 178 | (b) shared + stim (all 9) | **0.470** | 0.254 | **0.332** |
| Cux2-CreERT2 | 178 | (b) shared + stim (held-out) | 0.471 | 0.254 | 0.331 |
| Sst-IRES-Cre | 10 | copy-last | 0.631 | 0.254 | 0.000 |
| Sst-IRES-Cre | 10 | (a) ridge per neuron + stim | **0.367** | **0.408** | **0.419** |
| Sst-IRES-Cre | 10 | (b) shared + stim (all 9) | 0.372 | 0.336 | 0.411 |
| Sst-IRES-Cre | 10 | (b) shared + stim (held-out) | 0.371 | 0.339 | 0.412 |
| Vip-IRES-Cre | 13 | copy-last | 0.205 | 0.610 | 0.000 |
| Vip-IRES-Cre | 13 | (a) ridge per neuron + stim | **0.148** | **0.652** | **0.279** |
| Vip-IRES-Cre | 13 | (b) shared + stim (all 9) | 0.177 | 0.622 | 0.133 |
| Vip-IRES-Cre | 13 | (b) shared + stim (held-out) | 0.178 | 0.620 | 0.130 |

Per session, test score (a) / (b, all 9): Cux2 501271265 0.381 / 0.369, 501704220 0.389 / 0.368, 501836392 0.295 / 0.332; Sst 573720508 0.282 / 0.286, 576095926 0.572 / 0.383, 580043440 0.419 / 0.411; Vip 583279803 0.308 / 0.282, 606353987 0.245 / 0.214, 612543999 0.279 / 0.133

- Held-out mice, per neuron, b (held-out) beats (a) in test MSE for 63/178 Cux2 (median +0.010), 5/10 Sst (median +0.008), 1/13 Vip (median +0.023)
- Alpha: b picks 0.1 (bottom of the grid) in both fits; with 12–19 M rows this is effectively unregularised. (a) per-neuron alpha medians range 0.1 to 3.2e4 by session; 129/215 neurons of Cux2 501271265 sit at the grid minimum, up to 15 per Cux2 session at the maximum (1e7)
- b weights on own past for step 1: last bin 0.66, bin before 0.16, earlier bins ≈ 0 — mostly a damped copy-last plus a stimulus term

## Plots
- plots/step5_shared.png: 2 neurons per Cre line from the held-out mice (seeded random pick: Cux2 501836392 neurons 113, 150; Sst 580043440 neurons 2, 3; Vip 612543999 neurons 0, 12). True trace, copy-last, (a), and b (held-out); grey = grating on; 30 s from 5 s before the first grating epoch in test, as in step 4

## Problems / errors
- api.brain-map.org still blocked; `ophys_experiments.json` and the 9 NWB files were fetched from the Allen S3 mirror into `data/brain_observatory/` (not committed)
- Per-neuron alpha did not always help: for Sst 573720508, (a) here scores 0.282 vs 0.348 in step 4 with one shared alpha. Checked: the same code with the step 4 alpha (3.2e4 for all) gives the step 4 MSE (0.685) exactly, so the drop comes from noisy per-neuron alpha choice on the short val part (≈5 min), not a bug
- Only one held-out fold (the 3rd mouse per line), and Sst/Vip held-out results rest on 10 and 13 neurons

## What I noticed
- One shared model with no neuron or Cre-line label is about as good as per-neuron models for Cux2 (0.354 vs 0.348 within-session) and generalises to a new Cux2 mouse with no loss (held-out 0.331 vs 0.332 when that mouse was in training). Cux2 dominates the pool (88% of neurons), so the shared model is in effect a Cux2 model.
- For Sst and especially Vip the shared model is worse than per-neuron models (Vip held-out 0.130 vs 0.279; Sst r 0.34 vs 0.41), while training on vs holding out the mouse changes nothing. The gap is between cell types, not between mice: this is the room a cell-type label (or type-specific model) could fill.
- In the plot the shared model often rises toward the population mean within the 1 s (Cux2 neuron 113, Vip neuron 0) where the neuron stays low; (a) keeps the neuron's own level.
