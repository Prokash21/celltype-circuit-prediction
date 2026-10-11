# Step 4: Ridge per neuron with stimulus features
Date: 2026-10-12

## What was run
- `scripts/step4_stimulus.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`); reuses split, z-score, 10 Hz binning, windows and ridge from `scripts/step3_linear.py`
- 12 stimulus features per frame from the NWB stimulus tables, averaged into the same 10 Hz bins: stimulus type one-hot (drifting gratings, natural movie one + three, spontaneous), grating on (non-blank trial), grating direction one-hot (0–315° in 45° steps, 0 when no grating)
- Ridge per neuron, without vs with stimulus. With stimulus the inputs are the neuron's own past 30 bins + stimulus features for the past 30 bins AND the next 10 bins (30 + 40 × 12 = 510 inputs)
- Alpha as in step 3: one alpha per model per session, picked by val MSE on the last 10% of train, then refit on all train
- Extra evaluation "onset": test windows whose 1 s target starts at the bin holding a grating trial start (non-blank trials only)

## Data
- Experiment id: 573720508 (Sst) and 501271265 (Cux2), both three_session_A
- Cre line / area / depth: Sst-IRES-Cre / VISp / 275 µm; Cux2-CreERT2 / VISp / 175 µm
- Neurons: 10 (Sst); 215 (Cux2)
- Frames / frame rate / duration: 115,468 / 30.08 Hz (Sst); 115,755 / 30.15 Hz (Cux2); 64.0 min each → ~10 Hz bins
- Windows train / val / test / onset: 27,673 / 3,040 / 766 / 220 (Sst); 27,742 / 3,048 / 768 / 224 (Cux2)
- Stimulus share of frames (both sessions): gratings epoch 49%, natural movie 39%, spontaneous 8%, grating on 31%. The test part (last 20%) is 89% grating epoch.

## Results
Score = 1 − MSE_model / MSE_copylast on the same windows. MSE in z-units over all neurons × windows × 10 steps; r = mean per-neuron Pearson r.

All test windows:

| Session | Model | alpha | MSE | mean r | score |
|---|---|---|---|---|---|
| Sst 573720508 | copy-last | – | 1.050 | 0.379 | 0.000 |
| Sst 573720508 | mean | – | 1.288 | 0.061 | −0.227 |
| Sst 573720508 | ridge per neuron | 3.2e4 | 0.735 | 0.523 | 0.300 |
| Sst 573720508 | ridge per neuron + stim | 3.2e4 | **0.685** | **0.565** | **0.348** |
| Cux2 501271265 | copy-last | – | 0.470 | 0.189 | 0.000 |
| Cux2 501271265 | mean | – | 0.363 | 0.019 | 0.227 |
| Cux2 501271265 | ridge per neuron | 0.1 | **0.287** | 0.252 | **0.388** |
| Cux2 501271265 | ridge per neuron + stim | 0.1 | 0.290 | **0.256** | 0.383 |

Grating-onset windows (first 1 s after a grating starts):

| Session | Model | MSE | mean r | score |
|---|---|---|---|---|
| Sst 573720508 | copy-last | 0.967 | 0.018 | 0.000 |
| Sst 573720508 | mean | 1.098 | −0.012 | −0.135 |
| Sst 573720508 | ridge per neuron | 0.555 | 0.302 | 0.426 |
| Sst 573720508 | ridge per neuron + stim | **0.525** | **0.369** | **0.457** |
| Cux2 501271265 | copy-last | 0.417 | 0.072 | 0.000 |
| Cux2 501271265 | mean | 0.286 | −0.029 | 0.313 |
| Cux2 501271265 | ridge per neuron | 0.228 | 0.153 | 0.452 |
| Cux2 501271265 | ridge per neuron + stim | **0.224** | **0.175** | **0.464** |

- Per neuron (all test windows): stimulus lowers MSE for 10/10 Sst neurons (median −0.037, best −0.125, neuron 2) and for 42/215 Cux2 neurons (median +0.004, best −0.330, neuron 25)
- The no-stimulus rows reproduce step 3 exactly

## Plots
- plots/step4_stimulus.png (same neurons as step 3: Sst 7, 6; Cux2 66, 57; grey = grating on). The first 30 s of test are natural movie, so the plot shows 30 s starting 5 s before the first grating epoch in test (~68–100 s after test start).

## Problems / errors
- api.brain-map.org still blocked; used the S3 mirror files from step 2 (`data/brain_observatory/`, not committed)
- `get_stimulus_epoch_table()` fails for 501271265 ("more than 2 epochs cut"), so epochs were built from the per-trial stimulus tables: trials are joined into one epoch when the gap between them is ≤ 60 frames (2 s)
- Cux2 alpha is 0.1, the bottom of the grid, for both models: one alpha is shared by all 215 neurons, and the val minimum is almost unregularised

## What I noticed
- Stimulus helps clearly for Sst (score 0.300 → 0.348, all neurons improve) but not on average for Cux2 (0.388 → 0.383). In Cux2 a minority of neurons (42/215) improve a lot and most get slightly worse.
- At grating onsets stimulus helps a bit in both sessions (Sst r 0.30 → 0.37, Cux2 r 0.15 → 0.18). The plot shows the stim model rising earlier at some onsets (Cux2 neuron 66 at ~85 s), but most event starts are still missed.
- With alpha 0.1 the stim model can make large wrong swings at rare moments, e.g. Cux2 neuron 57 at ~76–78 s, the switch from natural movie to gratings (only a few such switches in train).
- Next idea: choose alpha per neuron, or penalise the stimulus inputs separately, so neurons that do not respond to gratings are not hurt.
