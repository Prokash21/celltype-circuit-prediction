# Step 3: Linear forecasting models (ridge per neuron, ridge population)
Date: 2026-10-12

## What was run
- `scripts/step3_linear.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`)
- Same split and z-score as step 2 (first 80% train, last 20% test, mean/std from train) → average every 3 frames (~10 Hz) → past 3 s (30 bins) predicts next 1 s (10 bins)
- Models: copy-last, mean, ridge per neuron (own past only, 30 inputs), ridge population (all neurons' past, 30 × n_neurons inputs). All outputs are the 10 future bins.
- Ridge alpha: one alpha per model per session, picked by val MSE on the last 10% of train (grid 1e-1 … 1e7, 17 values); then refit on the full train part
- Train/val windows step by 1 bin; test windows step by 1 s (no overlap in targets), as in step 2

## Data
- Experiment id: 573720508 (Sst) and 501271265 (Cux2), both three_session_A
- Cre line / area / depth: Sst-IRES-Cre / VISp / 275 µm; Cux2-CreERT2 / VISp / 175 µm
- Neurons: 10 (Sst); 215 (Cux2)
- Frames / frame rate / duration: 115,468 / 30.08 Hz (Sst); 115,755 / 30.15 Hz (Cux2); 64.0 min each → 10.03 / 10.05 Hz bins
- Windows train / val / test: 27,673 / 3,040 / 766 (Sst); 27,742 / 3,048 / 768 (Cux2)

## Results
Test set. MSE in z-units over all neurons × windows × 10 steps. r = Pearson r per neuron (over all test windows × steps), then mean over neurons. Score = 1 − MSE_model / MSE_copylast (higher is better, 0 = copy-last).

| Session | Model | alpha | MSE | mean r | score | MSE step 1 | MSE step 10 |
|---|---|---|---|---|---|---|---|
| Sst 573720508 | copy-last | – | 1.050 | 0.379 | 0.000 | 0.325 | 1.869 |
| Sst 573720508 | mean | – | 1.288 | 0.061 | −0.227 | 1.133 | 1.432 |
| Sst 573720508 | ridge per neuron | 3.2e4 | 0.735 | 0.523 | 0.300 | 0.421 | 0.985 |
| Sst 573720508 | ridge population | 3.2e4 | **0.699** | **0.552** | **0.335** | 0.401 | 0.945 |
| Cux2 501271265 | copy-last | – | 0.470 | 0.189 | 0.000 | 0.237 | 0.613 |
| Cux2 501271265 | mean | – | 0.363 | 0.019 | 0.227 | 0.325 | 0.386 |
| Cux2 501271265 | ridge per neuron | 0.1 | **0.287** | **0.252** | **0.388** | 0.186 | 0.328 |
| Cux2 501271265 | ridge population | 1e5 | 0.301 | 0.224 | 0.360 | 0.238 | 0.327 |

- Downsampling alone lowers MSE a lot vs step 2 (copy-last Sst 1.528 → 1.050, Cux2 0.857 → 0.470), because averaging 3 frames removes frame noise.

## Plots
- plots/step3_linear.png (first 30 s of test; same neurons as step 2: Sst 7, 6; Cux2 66, 57)

## Problems / errors
- api.brain-map.org still blocked; used the S3 mirror files from step 2 (`data/brain_observatory/`, not committed)
- First run used alpha grid 1e-1 … 1e5 and the Cux2 population alpha landed on the top edge (1e5). Widened the grid to 1e7; the chosen alphas did not change, so 1e5 is a real minimum.
- Cux2 per-neuron alpha is at the bottom edge (0.1): with 30 inputs and ~28k windows, regularisation barely matters there.
- Lower alphas (<0.1) were not tested; with 30 inputs and ~28k windows the effect should be negligible.

## What I noticed
- Both ridge models beat both baselines on both sessions (score +0.30 to +0.39). Most of the gain is at the far end of the horizon (step 10), where copy-last is bad.
- At step 1, copy-last is still about as good as or better than ridge (Sst 0.325 vs 0.40–0.42), so the last value carries most of the very-short-term information.
- Population ridge helps a little for Sst (10 neurons, 0.699 vs 0.735) but not for Cux2 (215 neurons, 0.301 vs 0.287): with 6,450 inputs it needs strong regularisation, and it sometimes makes big wrong swings (Cux2 neuron 66 at ~11–13 s in the plot).
- No model predicts the onset of the large event in Cux2 neuron 57 (~13 s); ridge only follows it after it starts, like the baselines.
