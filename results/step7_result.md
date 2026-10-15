# Step 7: Is the step 6 result robust? (leave-one-mouse-out, 5 mice per line)
Date: 2026-10-12

## What was run
- `scripts/step7_robust.py` (allensdk 2.16.2, Python 3.10, `.venv`, `setuptools<81`, `statsmodels<0.14.5`). It reuses the preprocessing, stimulus features, model a and batched X'X fitting from `scripts/step5_shared.py`, and the d model, shuffled grouping and pooled metrics from `scripts/step6_celltype.py`. Inputs per neuron: own past 30 bins + 480 stimulus inputs → next 10 bins
- 5 folds: fold k holds out the k-th mouse (by experiment id) of every line. Models are trained on the other 4 mice of every line (12 sessions) and tested on the last 20% of the 3 held-out sessions (same test windows as steps 5–6)
- Models:
  - (b) shared, no label, **line-balanced**: every training row is weighted by w_line = N / (3 · N_line) (N = training rows), so each Cre line has equal total weight in X'X, X'Y and in the val MSE used to pick alpha. Refit on train + val with the same weights
  - (d) one shared model per Cre line (as in step 6; each model sees one line, so no weighting)
  - (e) shuffled control for d: the 12 training sessions are split into 3 random groups of 4, and groups are redrawn until every group mixes lines. Each held-out session gets a random group. Each group's model is fit line-balanced as b. 5 seeds (0–4) per fold
  - (a) per-neuron ridge + stim, fit on the held-out session's own train part, as a reference only
- Per fold and line: held-out MSE, mean per-neuron r, score = 1 − MSE / MSE_copylast. Summary per line: mean ± std over the 5 folds (population std), how many folds d beats b and d beats e (e = mean over seeds), and a paired two-sided Wilcoxon signed-rank test over folds (scipy, exact)

## Data
- 15 sessions, VISp, three_session_A, lowest experiment id per mouse, then the 5 lowest per line (fold = rank):

| Fold | Cux2-CreERT2 (depth 175 µm) | Sst-IRES-Cre | Vip-IRES-Cre (175 µm) |
|---|---|---|---|
| 0 | 501271265 (215 neurons) | 573720508 (10, 275 µm) | 583279803 (11) |
| 1 | 501704220 (142) | 576095926 (15, 285 µm) | 606353987 (13) |
| 2 | 501836392 (178) | 580043440 (10, 275 µm) | 612543999 (13) |
| 3 | 502115959 (205) | 580163817 (7, **375 µm**) | 650079244 (24) |
| 4 | 502205092 (292) | 581150104 (8, 275 µm) | 652091264 (17) |

- Totals: Cux2 1032, Sst 50, Vip 78 neurons. The first 3 mice per line are the 9 sessions of steps 5–6, and fold 2 is the step 6 held-out split
- Frames / frame rate / duration: ~30 Hz, ~64 min per session → ~10 Hz bins; windows train / val / test ≈ 27,700 / 3,045 / 766–768 per session
- Line weights per fold (Cux2, Sst, Vip): 0.37–0.38, 6.7–9.4, 4.6–5.7

## Results
Per fold, held-out mouse test MSE (e = mean ± std over 5 seeds):

| Cre line | Fold | Held-out exp | (a) ref | (b) balanced | (d) per line | (e) shuffled | d − b | d − e |
|---|---|---|---|---|---|---|---|---|
| Cux2 | 0 | 501271265 | 0.2907 | 0.3072 | 0.2955 | 0.3151 ± 0.0058 | −0.0117 | −0.0196 |
| Cux2 | 1 | 501704220 | 0.3480 | 0.3723 | 0.3605 | 0.3805 ± 0.0051 | −0.0119 | −0.0201 |
| Cux2 | 2 | 501836392 | 0.4964 | 0.4817 | 0.4699 | 0.4884 ± 0.0053 | −0.0118 | −0.0184 |
| Cux2 | 3 | 502115959 | 0.3595 | 0.3753 | 0.3641 | 0.3776 ± 0.0053 | −0.0112 | −0.0135 |
| Cux2 | 4 | 502205092 | 0.3543 | 0.3835 | 0.3726 | 0.3852 ± 0.0041 | −0.0109 | −0.0126 |
| Sst | 0 | 573720508 | 0.7536 | 0.7068 | 0.6698 | 0.7243 ± 0.0286 | −0.0370 | −0.0545 |
| Sst | 1 | 576095926 | 0.3827 | 0.5180 | 0.4673 | 0.5371 ± 0.0294 | −0.0507 | −0.0698 |
| Sst | 2 | 580043440 | 0.3666 | 0.3683 | 0.3719 | 0.3755 ± 0.0099 | +0.0036 | −0.0036 |
| Sst | 3 | 580163817 | 0.3731 | 0.3871 | 0.3956 | 0.3935 ± 0.0127 | +0.0085 | +0.0022 |
| Sst | 4 | 581150104 | 0.4733 | 0.4474 | 0.4388 | 0.4603 ± 0.0191 | −0.0086 | −0.0215 |
| Vip | 0 | 583279803 | 0.4015 | 0.4172 | 0.4119 | 0.4179 ± 0.0026 | −0.0053 | −0.0061 |
| Vip | 1 | 606353987 | 0.2363 | 0.2437 | 0.2289 | 0.2439 ± 0.0023 | −0.0148 | −0.0150 |
| Vip | 2 | 612543999 | 0.1476 | 0.1723 | 0.1557 | 0.1726 ± 0.0022 | −0.0166 | −0.0169 |
| Vip | 3 | 650079244 | 0.2173 | 0.2427 | 0.2470 | 0.2444 ± 0.0046 | +0.0043 | +0.0026 |
| Vip | 4 | 652091264 | 0.1730 | 0.1977 | 0.1859 | 0.2006 ± 0.0033 | −0.0117 | −0.0146 |

Per fold, held-out score (r in brackets):

| Cre line | Fold | (b) balanced | (d) per line | (e) shuffled |
|---|---|---|---|---|
| Cux2 | 0 | 0.346 (0.225) | 0.371 (0.242) | 0.330 ± 0.012 (0.221) |
| Cux2 | 1 | 0.347 (0.208) | 0.368 (0.218) | 0.332 ± 0.009 (0.205) |
| Cux2 | 2 | 0.315 (0.241) | 0.332 (0.254) | 0.306 ± 0.008 (0.239) |
| Cux2 | 3 | 0.290 (0.234) | 0.311 (0.255) | 0.286 ± 0.010 (0.233) |
| Cux2 | 4 | 0.352 (0.256) | 0.370 (0.259) | 0.349 ± 0.007 (0.250) |
| Sst | 0 | 0.327 (0.511) | 0.362 (0.563) | 0.310 ± 0.027 (0.491) |
| Sst | 1 | 0.420 (0.462) | 0.477 (0.524) | 0.399 ± 0.033 (0.439) |
| Sst | 2 | 0.417 (0.365) | 0.411 (0.391) | 0.405 ± 0.016 (0.348) |
| Sst | 3 | 0.383 (0.301) | 0.369 (0.320) | 0.373 ± 0.020 (0.287) |
| Sst | 4 | 0.336 (0.562) | 0.349 (0.575) | 0.317 ± 0.028 (0.545) |
| Vip | 0 | 0.280 (0.551) | 0.290 (0.563) | 0.279 ± 0.004 (0.551) |
| Vip | 1 | 0.221 (0.523) | 0.269 (0.561) | 0.221 ± 0.007 (0.522) |
| Vip | 2 | 0.158 (0.607) | 0.240 (0.643) | 0.157 ± 0.011 (0.605) |
| Vip | 3 | 0.291 (0.344) | 0.278 (0.363) | 0.286 ± 0.014 (0.340) |
| Vip | 4 | 0.240 (0.416) | 0.285 (0.456) | 0.229 ± 0.013 (0.408) |

Summary over 5 folds (mean ± std):

| Cre line | Model | MSE | mean r | score |
|---|---|---|---|---|
| Cux2-CreERT2 | (a) ref | 0.370 ± 0.068 | 0.260 ± 0.021 | 0.357 ± 0.042 |
| Cux2-CreERT2 | (b) balanced | 0.384 ± 0.056 | 0.233 ± 0.016 | 0.330 ± 0.024 |
| Cux2-CreERT2 | (d) per line | **0.373 ± 0.056** | **0.246 ± 0.015** | **0.350 ± 0.025** |
| Cux2-CreERT2 | (e) shuffled | 0.389 ± 0.056 | 0.230 ± 0.015 | 0.321 ± 0.022 |
| Sst-IRES-Cre | (a) ref | 0.470 ± 0.147 | 0.495 ± 0.095 | 0.395 ± 0.104 |
| Sst-IRES-Cre | (b) balanced | 0.486 ± 0.122 | 0.440 ± 0.095 | 0.376 ± 0.039 |
| Sst-IRES-Cre | (d) per line | **0.469 ± 0.106** | **0.475 ± 0.101** | **0.394 ± 0.047** |
| Sst-IRES-Cre | (e) shuffled | 0.498 ± 0.127 | 0.422 ± 0.094 | 0.361 ± 0.040 |
| Vip-IRES-Cre | (a) ref | 0.235 ± 0.089 | 0.525 ± 0.093 | 0.306 ± 0.042 |
| Vip-IRES-Cre | (b) balanced | 0.255 ± 0.086 | 0.488 ± 0.095 | 0.238 ± 0.047 |
| Vip-IRES-Cre | (d) per line | **0.246 ± 0.089** | **0.517 ± 0.097** | **0.272 ± 0.018** |
| Vip-IRES-Cre | (e) shuffled | 0.256 ± 0.086 | 0.485 ± 0.097 | 0.234 ± 0.047 |
| All lines (pooled per fold) | (b) balanced | 0.380 ± 0.042 | 0.259 ± 0.010 | 0.330 ± 0.022 |
| All lines (pooled per fold) | (d) per line | **0.368 ± 0.042** | **0.275 ± 0.007** | **0.351 ± 0.025** |
| All lines (pooled per fold) | (e) shuffled | 0.385 ± 0.042 | 0.256 ± 0.010 | 0.321 ± 0.020 |

d vs b and d vs e (held-out MSE):

| Cre line | d beats b | Wilcoxon p (d vs b) | d beats e | Wilcoxon p (d vs e) |
|---|---|---|---|---|
| Cux2-CreERT2 | 5/5 | 0.0625 | 5/5 | 0.0625 |
| Sst-IRES-Cre | 3/5 | 0.31 | 4/5 | 0.13 |
| Vip-IRES-Cre | 4/5 | 0.13 | 4/5 | 0.13 |
| All lines (pooled per fold) | 5/5 | 0.0625 | 5/5 | 0.0625 |
| All 15 line-folds as pairs | 12/15 | 0.0012 | 13/15 | 0.0003 |

- With n = 5 pairs, the smallest possible two-sided exact Wilcoxon p is 0.0625 (all 5 folds with the same sign). So no single line can reach p < 0.05; Cux2 and the pooled all-lines result are at that floor
- Alphas: b, the Cux2 and Vip models of d all pick 0.1 (grid minimum) in every fold. The Sst model of d picks 3.2e4 or 1e5 (fit on 35–43 neurons)
- Fold 2 is the step 6 split. Its d values are close to step 6 (Cux2 0.4699 vs 0.4701, Sst 0.3719 vs 0.3782, Vip 0.1557 vs 0.1619; here d is trained on 4 mice per line instead of 2). Balanced b on the held-out Cux2 mouse is worse than step 6's unbalanced b (0.4817 vs 0.4706), so the Cux2 d − b gap grows from −0.0005 to −0.0118

## Plots
- plots/step7_robust.png: one panel per Cre line, bars = held-out MSE d − b for each fold (x labels: fold and held-out experiment id); blue = d better, red = b better

## Problems / errors
- api.brain-map.org is still blocked; `ophys_experiments.json` and the 15 NWB files were fetched from the Allen S3 mirror into `data/brain_observatory/` (not committed)
- My first quick listing matched Cux2 in any transgenic line and picked 501484643 (275 µm) as the 2nd Cux2 mouse. The script uses allensdk `filter_ophys_experiments(cre_lines=...)` as in steps 5–6, which selects 501836392 instead; the run uses that set
- 5 folds per line give at most p = 0.0625 for a paired Wilcoxon test. The per-line tests are low-powered by design; only the pooled 15 line-folds can go below 0.05, and those 15 pairs are not fully independent (the 3 lines of a fold share b and the training mice)
- Sst held-out sessions have 7–15 neurons, and fold 3 (580163817) is at 375 µm while the training Sst mice are at 265–285 µm. Fold 3 is where d loses for both Sst and Vip
- Balancing changes b compared with step 6: Cux2 now has ~0.37× weight per row and Sst ~7–9×. So the step 6 finding "d ≈ b for Cux2" does not carry over, because b is no longer in effect a Cux2 model

## What I noticed
- The step 6 result holds up and is stronger with 4 training mice per line: d beats b and e in all 5 folds for Cux2 and for all lines pooled, and in 12–13 of 15 line-folds (p ≤ 0.001 pooled). Per line, only Cux2 is fully consistent. Sst (3/5 vs b) and Vip (4/5) are mostly but not always better: both lines miss on fold 3, and Sst also misses on fold 2, which is the step 6 split.
- The step 6 Sst "NO" was a single bad fold rather than the rule. With 4 training mice, d gives the largest gains for Sst (folds 0–1: −0.04 to −0.05 MSE) but also the most variable ones. The Cux2 gain is tiny but very stable (−0.011 in every fold). Part of it reflects that balanced b gives up some Cux2 accuracy to fit Sst and Vip.
