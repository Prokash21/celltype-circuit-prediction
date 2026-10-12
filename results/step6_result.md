# Step 6: Does the Cre-line label help the shared model?
Date: 2026-10-12

## What was run
- `scripts/step6_celltype.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`); reuses the 9 sessions, preprocessing, stimulus features and batched X'X fitting from `scripts/step5_shared.py`. Inputs per neuron: own past 30 bins + 480 stimulus inputs → next 10 bins
- Models (all shared across neurons, all with stimulus):
  - (b) shared, no label (step 5 b)
  - (c) b + Cre one-hot (3 inputs, i.e. an offset per line). The one-hot is constant within a session, so its X'X terms come from each session's n, Σx, Σy; one alpha by pooled val MSE, then refit on train + val
  - (d) Cre interaction = one shared model per Cre line (one-hot × all inputs, intercept included). Each line's model is step 5 b fit on that line's sessions only, with its own alpha from that line's val windows
  - (e) shuffled control for d: the training sessions are split into 3 random equal-size groups (2 sessions each for held-out, 3 for within-session). Groups are redrawn until every group mixes Cre lines, so no fake group matches a true line. Within-session, each session is tested with its own fake group. Each held-out session gets a random group. d is then fit with these fake labels. 10 seeds (0–9), mean ± std reported
  - (a) step 5 per-neuron ridge + stim, shown as a reference row
- Main test is the held-out mouse: fit on mice 1–2 per line (6 sessions), test on the last 20% of the 3rd mouse per line, using the same test windows as step 5. Also within-session: fit on the train parts of all 9 sessions and test on the last 20% of each
- Score = 1 − MSE_model / MSE_copylast. Per line, neurons are pooled: MSE over neurons × windows × 10 steps, r = mean per-neuron Pearson r
- Decision: the label helps if d beats b AND d beats e (mean over seeds) in held-out-mouse test MSE

## Data
- Same 9 sessions as step 5 (VISp, three_session_A, 3 mice per line; the 3rd/highest id per line is held out):
  Cux2-CreERT2 501271265, 501704220, **501836392** (215 / 142 / **178** neurons); Sst-IRES-Cre 573720508, 576095926, **580043440** (10 / 15 / **10**); Vip-IRES-Cre 583279803, 606353987, **612543999** (11 / 13 / **13**)
- Cre line / area / depth: VISp, 175 µm (Cux2, Vip), 275–285 µm (Sst)
- Frames / frame rate / duration: ~115.5k frames per session, ~30 Hz, ~64 min → ~10 Hz bins; windows train / val / test ≈ 27,700 / 3,045 / 766–768 per session
- Check: b, and a, reproduce the step 5 numbers exactly

## Results
Held-out mouse (main test; (a) saw this mouse's train part, b–e did not):

| Cre line | Neurons | Model | MSE | mean r | score |
|---|---|---|---|---|---|
| Cux2-CreERT2 | 178 | copy-last | 0.704 | 0.205 | 0.000 |
| Cux2-CreERT2 | 178 | (a) ridge per neuron + stim (ref) | 0.496 | 0.256 | 0.295 |
| Cux2-CreERT2 | 178 | (b) shared, no label | 0.4706 | 0.254 | 0.331 |
| Cux2-CreERT2 | 178 | (c) shared + Cre one-hot | 0.4706 | 0.254 | 0.331 |
| Cux2-CreERT2 | 178 | (d) shared per Cre line | **0.4701** | 0.255 | **0.332** |
| Cux2-CreERT2 | 178 | (e) shuffled control (10 seeds) | 0.490 ± 0.015 | 0.239 ± 0.012 | 0.304 ± 0.022 |
| Sst-IRES-Cre | 10 | copy-last | 0.631 | 0.254 | 0.000 |
| Sst-IRES-Cre | 10 | (a) ridge per neuron + stim (ref) | 0.367 | 0.408 | 0.419 |
| Sst-IRES-Cre | 10 | (b) shared, no label | **0.3711** | 0.339 | **0.412** |
| Sst-IRES-Cre | 10 | (c) shared + Cre one-hot | 0.3711 | 0.339 | 0.412 |
| Sst-IRES-Cre | 10 | (d) shared per Cre line | 0.3782 | **0.388** | 0.401 |
| Sst-IRES-Cre | 10 | (e) shuffled control (10 seeds) | 0.375 ± 0.004 | 0.338 ± 0.020 | 0.407 ± 0.007 |
| Vip-IRES-Cre | 13 | copy-last | 0.205 | 0.610 | 0.000 |
| Vip-IRES-Cre | 13 | (a) ridge per neuron + stim (ref) | 0.148 | 0.652 | 0.279 |
| Vip-IRES-Cre | 13 | (b) shared, no label | 0.1780 | 0.620 | 0.130 |
| Vip-IRES-Cre | 13 | (c) shared + Cre one-hot | 0.1780 | 0.620 | 0.130 |
| Vip-IRES-Cre | 13 | (d) shared per Cre line | **0.1619** | **0.631** | **0.209** |
| Vip-IRES-Cre | 13 | (e) shuffled control (10 seeds) | 0.181 ± 0.004 | 0.608 ± 0.017 | 0.114 ± 0.022 |
| All lines | 201 | (b) shared, no label | 0.4467 | 0.282 | 0.331 |
| All lines | 201 | (d) shared per Cre line | **0.4456** | **0.286** | **0.333** |
| All lines | 201 | (e) shuffled control (10 seeds) | 0.464 ± 0.014 | 0.268 ± 0.011 | 0.305 ± 0.020 |

Within-session (all 9 sessions):

| Cre line | Neurons | Model | MSE | mean r | score |
|---|---|---|---|---|---|
| Cux2-CreERT2 | 535 | copy-last | 0.574 | 0.191 | 0.000 |
| Cux2-CreERT2 | 535 | (a) ridge per neuron + stim (ref) | 0.374 | **0.249** | 0.348 |
| Cux2-CreERT2 | 535 | (b) shared, no label | 0.3712 | 0.240 | 0.354 |
| Cux2-CreERT2 | 535 | (c) shared + Cre one-hot | 0.3712 | 0.240 | 0.354 |
| Cux2-CreERT2 | 535 | (d) shared per Cre line | **0.3704** | 0.241 | **0.355** |
| Cux2-CreERT2 | 535 | (e) shuffled control (10 seeds) | 0.3713 ± 0.0003 | 0.240 ± 0.001 | 0.354 ± 0.001 |
| Sst-IRES-Cre | 35 | copy-last | 0.863 | 0.318 | 0.000 |
| Sst-IRES-Cre | 35 | (a) ridge per neuron + stim (ref) | **0.484** | **0.530** | **0.439** |
| Sst-IRES-Cre | 35 | (b) shared, no label | 0.557 | 0.413 | 0.355 |
| Sst-IRES-Cre | 35 | (c) shared + Cre one-hot | 0.557 | 0.413 | 0.355 |
| Sst-IRES-Cre | 35 | (d) shared per Cre line | 0.498 | 0.503 | 0.424 |
| Sst-IRES-Cre | 35 | (e) shuffled control (10 seeds) | 0.543 ± 0.016 | 0.433 ± 0.016 | 0.372 ± 0.018 |
| Vip-IRES-Cre | 37 | copy-last | 0.354 | 0.555 | 0.000 |
| Vip-IRES-Cre | 37 | (a) ridge per neuron + stim (ref) | **0.254** | **0.592** | **0.282** |
| Vip-IRES-Cre | 37 | (b) shared, no label | 0.273 | 0.575 | 0.231 |
| Vip-IRES-Cre | 37 | (c) shared + Cre one-hot | 0.273 | 0.575 | 0.231 |
| Vip-IRES-Cre | 37 | (d) shared per Cre line | 0.260 | 0.587 | 0.267 |
| Vip-IRES-Cre | 37 | (e) shuffled control (10 seeds) | 0.271 ± 0.005 | 0.568 ± 0.010 | 0.235 ± 0.014 |
| All lines | 607 | (b) shared, no label | 0.3758 | 0.270 | 0.349 |
| All lines | 607 | (d) shared per Cre line | **0.3710** | 0.277 | **0.358** |
| All lines | 607 | (e) shuffled control (10 seeds) | 0.3750 ± 0.0008 | 0.271 ± 0.000 | 0.351 ± 0.001 |

**Decision (held-out mice, test MSE):**
- All lines pooled: d − b = −0.0011 and d − e = −0.018 (d beats e in 10/10 seeds) → **label helps: YES, but d beats b by only 0.2% in MSE (score 0.333 vs 0.331)**
- Per line: Cux2 YES (d − b −0.0005, a tiny margin; d beats e in 8/10 seeds); Sst **NO** (d − b +0.0071, d − e +0.0035, d beats e in 3/10 seeds); Vip YES (d − b −0.016, d − e −0.019, d beats e in 10/10 seeds)
- Within-session, d beats b and e for every line (Sst −0.059 vs b, Vip −0.013, Cux2 −0.0008)

Other numbers:
- (c) is identical to (b) to 4 decimals for every line. The one-hot weights are 0.001 for all 3 lines, because z-scoring per neuron already removes per-line offsets
- Alphas: b, c and the Cux2 / Vip models of d pick 0.1 (grid minimum). The Sst model of d picks 1e4 (9 sessions) and 3.2e4 (6 sessions), since it is fit on only 25–35 neurons
- Held-out, per neuron, d beats b in test MSE for 135/178 Cux2 (median −0.0017), 5/10 Sst (median +0.0030) and 9/13 Vip (median −0.0115) neurons
- Weights of the per-line models (held-out fit) on own past for step 1, last two bins: Cux2 0.16 / 0.66, Sst 0.21 / 0.50, Vip 0.11 / 0.87. So Vip's model sits closer to copy-last and Sst's is more damped and smoother
- e held-out per seed is bimodal for Cux2: MSE 0.469–0.473 when the held-out Cux2 session draws a fake group that contains a Cux2 training session (4 seeds), and 0.500–0.503 when it draws the Sst+Vip group (6 seeds)

## Plots
- plots/step6_celltype.png: the same held-out neurons as step 5 (Cux2 501836392 neurons 113, 150; Sst 580043440 neurons 2, 3; Vip 612543999 neurons 0, 12), showing the true trace, (b) and (d) for held-out mice. Grey = grating on; the 30 s window starts 5 s before the first grating epoch in test

## Problems / errors
- api.brain-map.org is still blocked; `ophys_experiments.json` and the 9 NWB files were fetched from the Allen S3 mirror into `data/brain_observatory/` (not committed)
- Only one held-out fold. The Sst and Vip held-out results rest on 10 and 13 neurons, and the Sst per-line model is trained on 25 neurons (2 sessions)
- With 6 training sessions and groups that must mix lines, 2 of the 3 fake groups always contain a Cux2 session. So for the held-out Cux2 mouse, e partly measures "random group" and partly "a model trained without Cux2". Its large held-out std (±0.015) comes from this, not from fit noise

## What I noticed
- The label helps only when it changes how the past and stimulus are used (d). An offset per line (c) does nothing. The gain is clear for Vip (held-out score 0.130 → 0.209, closing about half the gap to the per-neuron model) and for Sst within-session (0.355 → 0.424). For Cux2 it is negligible, because b is already in effect a Cux2 model.
- For a new Sst mouse, d does not help in MSE (0.401 vs 0.412) even though r rises (0.339 → 0.388). In the plot the Sst model makes larger stimulus-locked swings, which hit some transients (neuron 2) and overshoot elsewhere (neuron 3); with 25 training neurons this does not carry over to a new mouse.
