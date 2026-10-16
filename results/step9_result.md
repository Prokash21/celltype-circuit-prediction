# Step 9: Does a neural network learn cell type by itself?
Date: 2026-10-16

## What was run
- `scripts/step9_mlp.py` (Python 3.10, `.venv`, PyTorch 2.14.1 on CPU, numpy / scipy / pandas as in step 8). Data come from the step 8 cache `data/cache/*.npz` (all 46 files present, no re-extraction needed). The script reuses the session list and folds from `scripts/step8_medium.py`, `windows` / `window_starts` from `scripts/step3_linear.py`, `stim_windows` from `scripts/step4_stimulus.py` and `neuron_r` from `scripts/step5_shared.py`
- Same 46 sessions, 5 folds (round-robin per line, fold k holds out group k of every line), inputs (own past 30 bins + 12 stimulus features × (past 30 + next 10 bins) = 510) → next 10 bins, and test windows (last 20% of each held-out session, stride 10 bins) as step 8
- All MLPs are 510 → 256 → 256 → 10 with ReLU. Models:
  - m1: shared MLP, no label
  - m2: MLP + Cre line via FiLM. A learned Cre embedding (dim 16) gives a scale γ and shift β for each hidden layer, after the ReLU: h · (1 + γ) + β. γ and β are linear in the embedding and zero-initialised, so m2 starts as m1
  - m3: as m2 with shuffled labels. The training sessions are split into 4 random equal-size groups that each mix lines (the same groupings as step 8 e, seeds 0–4), and each held-out mouse gets a random group. 5 seeds
  - m4: one MLP per Cre line, trained on that line's mice only
  - m1, m2 and m4 have 3 seeds each (init and sampling). Mouse and fold MSEs are averaged over seeds
- Training:
  - Rows (session, neuron, window start) are sampled on the fly from the train part (first 90% of the 80% train split, stride 1, as step 8), so the ~91M rows are never built in memory
  - Line-balanced as step 8 b: every batch has the same number of rows from each Cre line, uniform within a line
  - Adam, lr 1e-3, batch 1024, epoch = 400 batches (~410k rows), at most 40 epochs
  - Early stopping on val (the last 10% of train): a fixed sample of 20,000 val rows per line from the training mice, scored by line-balanced MSE. Patience 5; the best epoch is restored
- 115 model fits (23 per fold) ran in 4 worker processes with 1 thread each, in the background (`nohup`, log `data/step9_mlp.log`). That took ~1.6 h wall time and 6.9 CPU-h
- Reference rows: step 8 ridge b (line-balanced shared) and ridge d (per line), read per mouse from `data/cache/step8_mice.csv`. Pooled per fold the same way as the MLPs, they give the same numbers as step 8
- Metrics as step 8: MSE, mean per-neuron r, and score = 1 − MSE / MSE_copylast. Per fold they are pooled over the held-out mice of the line; the summary is mean ± std over the 5 folds (population std). Paired comparisons count held-out mice and use a paired two-sided Wilcoxon signed-rank test over mice (scipy)

### What was used from each reference
- **POCO** (Duan et al. 2025, arXiv 2506.14957; code read from github.com/yuvenduan/POCO, `zapbench_scripts/standalone_poco.py`):
  - Used: the idea of an MLP forecaster on each neuron's own past, conditioned by FiLM. POCO computes `in_proj(x) * alpha + beta` after a ReLU hidden layer, with alpha and beta linear in a per-unit embedding, and **zero-initialises** the alpha/beta layers. m2 and m3 follow this: FiLM after the ReLU of each hidden layer, linear in an embedding, zero-initialised
  - Changed:
    - We use (1 + γ) instead of α. POCO's MLP has no unconditioned path, so α = 0 at init gives a zero output; with (1 + γ), m2 starts exactly as m1
    - The conditioning embedding is the Cre line (or a shuffled group), not POCO's population encoder
    - POCO's POYO-based population encoder, tokenisation and multi-session unit embeddings were not used
- **DLinear** (Zeng et al. 2023; code read from github.com/cure-lab/LTSF-Linear, `models/DLinear.py`):
  - Used: the point that a linear map from past to future is a strong baseline. Ridge b and d from step 8 are kept as reference rows; they are the same linear past (+ stimulus) → future model family, with ridge instead of SGD
  - Used: the shared vs `individual` (per channel) linear option, which matches our shared (b / m1) vs per-line (d / m4) split
  - Not used: DLinear's trend/seasonal moving-average decomposition
- **POYO+** (Azabou et al., ICLR 2025) and **NuCLR**: neither paper nor its code was read here. arxiv.org was not reachable, and github.com/neuro-galaxy/poyo needs a login. The POYO model code in github.com/neuro-galaxy/torch_brain (`torch_brain/models/poyo.py`) and POCO's copy of it were read
  - Used: the claim that cell type can be decoded from activity, as the motivation for asking whether a shared MLP picks up cell type without a label (m1 vs m2)
  - Used: the learned-embedding design. POYO's unit and session embeddings are `nn.Embedding` with N(0, 0.02) init; our Cre embedding uses the same init
  - Seen but not used: torch_brain's Allen visual coding ophys pipeline also stores `cre_line` per session
  - Not done: a cell-type decoding probe on the MLP's hidden activity (the POYO+ / NuCLR analysis), because it was not part of this step

## Data
- Experiment ids: the same 46 sessions as step 8, one per mouse, VISp, three_session_A
- Cre line / area / depth: Cux2-CreERT2 13 mice (175 / 275 µm), Sst-IRES-Cre 12 (265–375 µm), Vip-IRES-Cre 12 (175 / 275 µm), Pvalb-IRES-Cre 9 (175–375 µm); all VISp
- Neurons: 3,296 in total (Cux2 2,905, Sst 122, Vip 153, Pvalb 116)
- Frames / frame rate / duration: 30.08–30.15 Hz, ~64 min per session → ~10 Hz bins (step 8 cache). Training rows available: ~91M (neuron × window). Each MLP saw at most 40 × 400 × 1024 ≈ 16M sampled rows

## Results
Summary over 5 folds (mean ± std, held-out mice pooled per fold):

| Cre line | Model | MSE | mean r | score |
|---|---|---|---|---|
| Cux2-CreERT2 | ridge b | 0.420 ± 0.050 | 0.233 ± 0.051 | 0.339 ± 0.035 |
| Cux2-CreERT2 | ridge d | 0.413 ± 0.055 | 0.243 ± 0.051 | 0.351 ± 0.045 |
| Cux2-CreERT2 | m1 MLP shared | 0.366 ± 0.044 | 0.336 ± 0.048 | 0.424 ± 0.030 |
| Cux2-CreERT2 | m2 MLP + Cre FiLM | **0.358 ± 0.043** | **0.361 ± 0.047** | **0.437 ± 0.030** |
| Cux2-CreERT2 | m3 MLP + shuffled FiLM | 0.367 ± 0.045 | 0.335 ± 0.048 | 0.423 ± 0.032 |
| Cux2-CreERT2 | m4 MLP per line | 0.363 ± 0.049 | 0.357 ± 0.045 | 0.430 ± 0.040 |
| Sst-IRES-Cre | ridge b | 0.499 ± 0.114 | 0.409 ± 0.068 | 0.384 ± 0.023 |
| Sst-IRES-Cre | ridge d | 0.481 ± 0.103 | 0.454 ± 0.078 | 0.403 ± 0.024 |
| Sst-IRES-Cre | m1 MLP shared | 0.379 ± 0.072 | 0.552 ± 0.071 | 0.528 ± 0.016 |
| Sst-IRES-Cre | m2 MLP + Cre FiLM | **0.376 ± 0.070** | **0.564 ± 0.068** | **0.531 ± 0.016** |
| Sst-IRES-Cre | m3 MLP + shuffled FiLM | 0.379 ± 0.073 | 0.550 ± 0.068 | 0.527 ± 0.015 |
| Sst-IRES-Cre | m4 MLP per line | 0.378 ± 0.070 | 0.559 ± 0.067 | 0.528 ± 0.016 |
| Vip-IRES-Cre | ridge b | 0.249 ± 0.068 | 0.437 ± 0.066 | 0.274 ± 0.041 |
| Vip-IRES-Cre | ridge d | 0.243 ± 0.068 | 0.459 ± 0.069 | 0.294 ± 0.024 |
| Vip-IRES-Cre | m1 MLP shared | 0.214 ± 0.059 | 0.524 ± 0.048 | 0.376 ± 0.043 |
| Vip-IRES-Cre | m2 MLP + Cre FiLM | **0.207 ± 0.058** | **0.543 ± 0.051** | **0.396 ± 0.033** |
| Vip-IRES-Cre | m3 MLP + shuffled FiLM | 0.213 ± 0.060 | 0.527 ± 0.050 | 0.378 ± 0.039 |
| Vip-IRES-Cre | m4 MLP per line | 0.209 ± 0.059 | 0.540 ± 0.054 | 0.394 ± 0.029 |
| Pvalb-IRES-Cre | ridge b | 0.330 ± 0.082 | 0.527 ± 0.108 | 0.192 ± 0.102 |
| Pvalb-IRES-Cre | ridge d | 0.313 ± 0.083 | 0.542 ± 0.110 | 0.241 ± 0.074 |
| Pvalb-IRES-Cre | m1 MLP shared | 0.285 ± 0.064 | 0.594 ± 0.085 | 0.293 ± 0.121 |
| Pvalb-IRES-Cre | m2 MLP + Cre FiLM | 0.267 ± 0.070 | 0.610 ± 0.097 | 0.353 ± 0.064 |
| Pvalb-IRES-Cre | m3 MLP + shuffled FiLM | 0.283 ± 0.068 | 0.598 ± 0.089 | 0.303 ± 0.100 |
| Pvalb-IRES-Cre | m4 MLP per line | **0.265 ± 0.070** | **0.612 ± 0.094** | **0.357 ± 0.062** |
| All lines | ridge b | 0.411 ± 0.040 | 0.257 ± 0.042 | 0.338 ± 0.031 |
| All lines | ridge d | 0.403 ± 0.044 | 0.269 ± 0.040 | 0.351 ± 0.042 |
| All lines | m1 MLP shared | 0.357 ± 0.036 | 0.359 ± 0.039 | 0.426 ± 0.029 |
| All lines | m2 MLP + Cre FiLM | **0.348 ± 0.035** | **0.383 ± 0.038** | **0.440 ± 0.027** |
| All lines | m3 MLP + shuffled FiLM | 0.357 ± 0.037 | 0.358 ± 0.039 | 0.426 ± 0.030 |
| All lines | m4 MLP per line | 0.352 ± 0.040 | 0.379 ± 0.036 | 0.433 ± 0.036 |

Paired comparisons over held-out mice (held-out MSE; "A beats B" = lower MSE; Wilcoxon p):

| Cre line | m2 beats m1 | p | m2 beats m3 | p | m4 beats m1 | p | m1 beats ridge b | p |
|---|---|---|---|---|---|---|---|---|
| Cux2-CreERT2 | 13/13 | 0.00024 | 13/13 | 0.00024 | 11/13 | 0.027 | 13/13 | 0.00024 |
| Sst-IRES-Cre | 9/12 | 0.47 | 9/12 | 0.47 | 8/12 | 0.97 | 12/12 | 0.00049 |
| Vip-IRES-Cre | 12/12 | 0.00049 | 12/12 | 0.00049 | 10/12 | 0.0034 | 12/12 | 0.00049 |
| Pvalb-IRES-Cre | 9/9 | 0.0039 | 9/9 | 0.0039 | 9/9 | 0.0039 | 8/9 | 0.0078 |
| All lines | 43/46 | 3.7e-07 | 43/46 | 8.1e-07 | 38/46 | 4.0e-05 | 45/46 | 5.7e-14 |

- m2 vs m4: m4 beats m2 in only 19/46 mice (median m4 − m2 = +0.0009, p 0.06). FiLM (one network, a label per line) is as good as or slightly better than separate networks; m2 is better for Cux2 (10/13 mice, p 0.04) and Sst (10/12, p 0.042)
- Sst: m2 is worse than m1 in 3 mice (576095926 +0.026, 596779487 +0.023, 580043440 +0.011), all at 275–285 µm. These mice drive the non-significant Sst result
- m3 (shuffled FiLM) is no better than m1 (all lines 0.426 vs 0.426 score), so the m2 gain comes from the true label, not from the extra parameters

Cell-type gain per held-out mouse, ridge (b − d) vs MLP (m1 − m2) and (m1 − m4) (MSE reduction; relative = gain / MSE of the label-free model; Wilcoxon = MLP gain vs ridge gain, paired over mice):

| Cre line | ridge b − d, median (rel.) | MLP m1 − m2, median (rel.) | MLP > ridge, mice | p | MLP m1 − m4, median (rel.) | MLP > ridge, mice | p |
|---|---|---|---|---|---|---|---|
| Cux2-CreERT2 | 0.0096 (2.6%) | 0.0083 (2.3%) | 3/13 | 0.13 | 0.0052 (1.5%) | 2/13 | 0.021 |
| Sst-IRES-Cre | 0.0192 (4.0%) | 0.0047 (1.5%) | 3/12 | 0.027 | 0.0042 (1.3%) | 2/12 | 0.012 |
| Vip-IRES-Cre | 0.0073 (3.4%) | 0.0061 (3.2%) | 6/12 | 0.52 | 0.0068 (3.4%) | 6/12 | 0.97 |
| Pvalb-IRES-Cre | 0.0200 (6.1%) | 0.0185 (5.4%) | 5/9 | 0.50 | 0.0191 (5.8%) | 4/9 | 0.43 |
| All lines | 0.0096 (3.2%) | 0.0084 (2.6%) | 17/46 | 0.16 | 0.0068 (2.1%) | 14/46 | 0.011 |

- In score units (all lines, mean over folds): ridge d − b = +0.013, MLP m2 − m1 = +0.014, MLP m4 − m1 = +0.007. MLP m1 − ridge b = +0.088

**Key question: does the cell-type gain shrink, stay or grow with an MLP instead of ridge?**
- **It stays for the FiLM model (m2), with a small, non-significant shrink:** median 0.0084 vs 0.0096 MSE (2.6% vs 3.2% relative), MLP gain larger than ridge in 17/46 mice, p = 0.16. In score units it is the same (+0.014 vs +0.013)
  - For Cux2, Vip and Pvalb the gain stays (p 0.13–0.52)
  - For Sst it shrinks clearly: 0.019 → 0.005, p = 0.027. With the MLP, the Sst label no longer helps significantly (9/12, p 0.47)
- **It shrinks for separate per-line networks (m4):** median 0.0068 vs 0.0096, 14/46, p = 0.011. This is mostly for Cux2 and Sst
- So the shared MLP does **not** learn cell type by itself well enough to make the label useless. It is much better than ridge for every line (45/46 mice), but adding the true Cre line still helps in 43/46 held-out mice and the shuffled label does not. The only line where the MLP seems to absorb most of the cell-type information is Sst

## Plots
- plots/step9_mlp.png: one panel per Cre line plus all lines. Bars = held-out score (mean ± std over 5 folds) of ridge b, ridge d, m1, m2, m3, m4; thin lines = individual folds

## Problems / errors
- **Network access:**
  - arxiv.org was not reachable, so the paper method sections could not be read, only the code
  - github.com web pages returned 403, but `git clone` worked for POCO, LTSF-Linear and torch_brain (cloned into the scratchpad, not the repo)
  - github.com/neuro-galaxy/poyo (POYO+) needs a login, and NuCLR code was not looked up. POYO+ and NuCLR are used only as stated above
  - The PyTorch CPU wheel index was blocked, so torch 2.14.1 was installed from PyPI (CUDA build, run on CPU)
- **Smaller training budget than ridge:**
  - Each MLP is trained on ≤16M sampled rows out of ~91M, on the train part only. Ridge in step 8 used all rows and refit on train + val
  - 6 of 115 fits stopped at the 40-epoch cap rather than by early stopping (1 m1, 4 m2, 1 m3), so some MLPs could improve slightly with more epochs. m2 is the model most often capped, so the m2 gain may be slightly underestimated
- **Seed noise:** the range of held-out MSE over the 3 seeds per mouse has a median of ~0.004–0.005 (m1, m2, m4), about half the cell-type gain. It reaches 0.02–0.04 for small Sst mice (e.g. 576095926), which adds to the Sst noise. Seed-averaged MSEs are used everywhere; per-seed MSEs are in `data/cache/step9_mice.csv` (not committed)
- **Small mice:** Pvalb fold 1 (670395999 with 3 neurons and 692345003 with 7) scores low for every model (pooled score: ridge b 0.00, m1 0.06, m2 0.24), so Pvalb's fold std is large. 670395999 is the only mouse where m1 is worse than ridge b (0.1652 vs 0.1627)
- No run errors. Wall time ~1.6 h (115 fits in 4 processes)

## What I noticed
- The MLP is a much better forecaster than ridge: score +0.09 overall and +0.14 for Sst, beating ridge b in 45/46 held-out mice. Even so, the Cre label adds about the same absolute gain as it did for ridge (~0.01 MSE, 43/46 mice), and the shuffled-label control adds nothing. So the shared MLP does not fully learn cell type from activity on its own, except mostly for Sst, where the gain shrinks to non-significant.
- One FiLM network with a Cre embedding (m2) matches or beats separate per-line networks (m4), especially for the small lines. Sharing the network while conditioning on cell type seems the better design.
