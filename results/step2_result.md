# Step 2: Simple forecasting baselines (copy-last, mean)
Date: 2026-10-12

## What was run
- `scripts/step2_baselines.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`)
- z-score each neuron's dF/F (mean/std from the train part only) → first 80% train / last 20% test → past 3 s (90 frames) predicts next 1 s (30 frames)
- Test windows step by 1 s (no overlap in targets), all inside the test part

## Data
- Experiment id: 573720508 (Sst) and 501271265 (Cux2), both three_session_A
- Cre line / area / depth: Sst-IRES-Cre / VISp / 275 µm; Cux2-CreERT2 / VISp / 175 µm
- Neurons: 10 (Sst); 215 (Cux2)
- Frames / frame rate / duration: 115,468 / 30.08 Hz / 64.0 min (Sst); 115,755 / 30.15 Hz / 64.0 min (Cux2)
- Cux2 pick: 16 Cux2-CreERT2 VISp three_session_A experiments; took the lowest id (same rule as step 1)
- Test: 23,094 frames, 766 windows (Sst); 23,151 frames, 768 windows (Cux2)

## Results
Test set. MSE is in z-units, over all neurons × windows × 30 steps. r = Pearson r per neuron (over all test windows × steps), then mean over neurons.

| Session | Baseline | MSE | mean r | MSE at step 1 | MSE at step 30 |
|---|---|---|---|---|---|
| Sst 573720508 | copy-last | 1.528 | 0.323 | 0.780 | 2.386 |
| Sst 573720508 | mean | 1.549 | 0.063 | 1.360 | 1.704 |
| Cux2 501271265 | copy-last | 0.857 | 0.161 | 0.579 | 1.027 |
| Cux2 501271265 | mean | 0.557 | 0.030 | 0.524 | 0.594 |

- Test variance of z-scored dF/F: 1.41 (Sst), 0.53 (Cux2). Predicting a constant 0 would give MSE close to these values.

## Plots
- plots/step2_baselines.png (first 30 s of test; Sst neurons 7, 6; Cux2 neurons 66, 57; seed 0)

## Problems / errors
- api.brain-map.org was blocked in this environment, so `BrainObservatoryCache` could not query it
- Fix: downloaded `ophys_experiments.json` and the two NWB files from the Allen public S3 mirror (`allen-brain-observatory/visual-coding-2p/`) into the same cache folder; selected with `BrainObservatoryApi.filter_ophys_experiments`, read with `BrainObservatoryNwbDataSet`

## What I noticed
- Copy-last is best right after the cut (step 1) but gets worse fast; by 1 s it is worse than mean. Mean is flat across the horizon.
- Mean wins on MSE for Cux2 but has almost no correlation (r ≈ 0.03–0.06): it mostly predicts "near baseline". Copy-last keeps more of the shape (higher r) but pays for noise in the last frame.
- Sst test variance (1.41) is higher than train (1.0), so the Sst neurons are more active or drift in the last 20%. MSE values are not directly comparable between sessions.
- Neither baseline can predict the start of a calcium event (e.g. Cux2 neuron 57 at ~13 s); the mean baseline only "sees" it after the event.
