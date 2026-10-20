# Step 16 (Part 1 of 2): Neuropixels spikes, no calcium indicator. Genetic label vs activity summary, and synthetic calcium from the same spikes
Date: 2026-10-19 to 2026-10-20

## What was run
- `scripts/step16_extract.py`: resumable extraction of all 58 Allen Visual Coding Neuropixels sessions, one session at a time (download NWB → `data/step16/sessions/<id>.npz` → delete the NWB)
- `scripts/step16_neuropixels.py <phase>`: `labels` (task 1), `bin`, then per dataset `gram`, `features`, and per configuration `cluster`, `ridge`, `mlp` (tasks 2–3)
- `scripts/step16_analyse.py`: writes `results/step16/*.csv`. `scripts/step16_plots.py` makes every plot from those CSV files only
- **Environment:** Python 3.11 (`.venv`, rebuilt), numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, scikit-learn 1.7.2, PyTorch 2.14.1 (CPU), allensdk 2.16.2 (needs setuptools < 70 and statsmodels 0.13.5 to import), h5py 3.16. These are the library versions of steps 11–15. AllenSDK is imported only because the step 3–13 modules import it; the NWB files are read directly with h5py
- **Order of runs:** extraction → labels → binning → ridge statistics and S features per signal → cluster / ridge per configuration → MLPs (balanced, then uniform) → genotype check → `analyse` → plots. The phases ran from a resumable driver; a container restart during the calcium MLPs lost no results
- **Wall time on 4 CPU cores:**

  | Phase | Time |
  |---|---|
  | extraction (58 NWB, ~150 GB downloaded) | ~4 h |
  | ridge statistics, per signal | ~20–30 min |
  | ridge, per signal (4 configurations) | ~1.5 h |
  | MLP, per configuration (70 fits) | 50–150 min |
  | MLPs, total (24 configurations × 70 fits) | ~35 h |

## Data
- **Allen Brain Observatory Visual Coding Neuropixels** (Siegle et al. 2021), public S3 bucket `allen-brain-observatory/visual-coding-neuropixels/ecephys-cache`. Session NWB files only (1.7–3.3 GB each); no LFP files were downloaded. The cache tables `sessions.csv`, `units.csv`, `probes.csv` and `channels.csv` came from the same bucket
- **Sessions:** all 58 (`results/step16/manifest_summary.csv`)

  | Genotype | brain_observatory_1.1 | functional_connectivity |
  |---|---|---|
  | Pvalb-IRES-Cre x Ai32 | 5 | 3 |
  | Sst-IRES-Cre x Ai32 | 6 | 6 |
  | Vip-IRES-Cre x Ai32 | 5 | 3 |
  | wild type | 16 | 14 |

  One session per mouse (58 mice). Extraction order: Pvalb / Sst / Vip first, then wild type
- **Units:** peak channel in VISp, VISl, VISal, VISrl, VISam or VISpm; isi_violations < 0.5, amplitude_cutoff < 0.1, presence_ratio > 0.9; and quality = "good" (the noise clusters, which AllenSDK's units table also drops; 830 of the 17,608 units passing the three metrics). This leaves **16,816 units** in 58 sessions: VISp 3,864, VISl 2,154, VISal 3,146, VISrl 2,679, VISam 3,074, VISpm 1,899
- **Invalid times:** units on a probe whose invalid-time intervals cover > 20% of the analysed period would be dropped; no probe reached this (largest 11%), so no unit was dropped. Windows that touch an invalid interval of the unit's probe (or an interval tagged with no probe, e.g. "stimulus", "all_probes") are excluded, per probe
- **Kept per unit** (for Part 2 too): spike times (30 kHz ticks), waveform duration / half-width, firing rate, SNR, QC metrics, area, probe, channel, probe vertical / horizontal position, surface channel. **Per session:** every stimulus presentation, running speed, optogenetic stimulation epochs, invalid-time intervals, genotype and session type
- **Analysed period:** from the first stimulus onset to 1 s before the first light pulse, i.e. the whole session except the optotagging epoch (152–163 min). Four wild-type sessions (732592105, 737581020, 739448407, 743475441) have no optotagging module; for them it ends at the last stimulus
- **Two sessions have no VISp unit** (819186360, 819701982, both Vip); they are in the all-areas analyses only

## Results

### Task 1: labels (`results/step16/unit_labels.csv`, `label_counts.csv`, `optotag_summary.csv`, `optotag_psth.csv`, `optotag_psth_tagged_units.csv`; plot `step16_optotag.png`)
**Opto-tagging criterion** (AllenSDK optotagging example / Siegle et al. 2021, plus a significance test):
- Trials: the 10 ms square light pulses of the optotagging epoch (45–151 per session)
- Spikes counted per trial in an 8 ms baseline window [−10, −2) ms and an 8 ms evoked window [1, 9) ms from pulse onset. The evoked window excludes the light-onset artefact at 0 ms and requires a short latency (first spike within 9 ms)
- **Tagged = evoked rate / (baseline rate + 1 Hz) > 2** (the tutorial's criterion) **AND one-sided Wilcoxon signed-rank test, per-trial evoked vs baseline counts, p < 0.001**
- **Why p < 0.001:** the wild-type mice (no ChR2) give a direct false-positive rate. The tutorial ratio alone flags 13.2% of wild-type units (1,107 / 8,414). With p < 0.01 added, 0.40% are still flagged; at that rate up to ~10 of the 24 Vip "tags" could be false. With p < 0.001, 0.05% (4 units) are flagged. The expected false tags are then about 1 of 133 Pvalb, 2 of 72 Sst and 1 of 14 Vip. The p < 0.01 labels are kept in `unit_labels.csv` (`tagged_p01`)

| Scope | Line | Sessions | Units | Tagged (p < 0.001) | % | Tagged at p < 0.01 | Ratio > 2 only | Sessions with ≥ 1 tag | Tagged FS % | Untagged FS % | Median latency (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VISp | Pvalb | 8 | 425 | **23** | 5.4 | 29 | 57 | 4 | 100 | 17 | 4.5 |
| VISp | Sst | 12 | 823 | **20** | 2.4 | 24 | 129 | 8 | 60 | 17 | 5.2 |
| VISp | Vip | 7 | 561 | **2** | 0.4 | 2 | 55 | 2 | 0 | 16 | 6.1 |
| VISp | wild type | 29 | 2,055 | 1 (false) | 0.05 | 13 | 259 | 1 | – | 20 | – |
| all VIS | Pvalb | 8 | 2,200 | **133** | 6.0 | 159 | 353 | 6 | 92 | 12 | 4.3 |
| all VIS | Sst | 12 | 3,623 | **72** | 2.0 | 98 | 525 | 12 | 46 | 15 | 5.5 |
| all VIS | Vip | 8 | 2,579 | **14** | 0.5 | 24 | 249 | 6 | 21 | 15 | 3.3 |
| all VIS | wild type | 30 | 8,414 | 4 (false) | 0.05 | 30 | 1,107 | 2 | – | 17 | – |

Tagged units per area (all Cre sessions): VISp 23 / 20 / 2 (Pvalb / Sst / Vip), VISl 20 / 7 / 3, VISal 18 / 5 / 0, VISrl 22 / 10 / 5, VISam 33 / 21 / 4, VISpm 17 / 9 / 0 (`label_counts.csv` has every session × area × label).

- **Light responses (plot (top row)):** tagged units fire at 100–150 Hz during the pulse, from ~1–2 ms after onset, against 5–15 Hz at baseline. Untagged units in the same mice do not respond, apart from the small onset / offset artefact at 0 and 10 ms that the windows avoid. Pvalb tagged units show a fast, oscillating response; Sst a slower rise
- **The labels agree with waveform class:**
  - 100% of the VISp Pvalb tags (92% over all areas) are fast-spiking, against 12–17% of untagged units
  - Sst tags are mixed (46–60% FS)
  - Vip tags are regular-spiking (median waveform 0.55 ms over all areas)
- **The tag is sparse, especially for Vip:** only 2 Vip-tagged units in VISp (14 over all areas), in 2 (6) of the Vip sessions. The ratio-only criterion would give 55 (249)
- **Label sets for tasks 2–3:**
  - (i) tagged type: Pvalb / Sst / Vip / untagged. Cre sessions only (28; 27 with VISp units)
  - (ii) waveform class: FS (waveform_duration < 0.4 ms) vs RS. All 58 sessions (56 with VISp). 19% of VISp units and 16% of all VIS units are FS
  - (iii) shuffled: each label set permuted within each session (same counts per session), 5 seeds
- **No one-type-per-recording confound:** in every Cre session the tagged units are recorded together with ~95% untagged units, so the recording does not reveal the label of a unit

### Task 2: main comparison at 1 s
**Setup:**
- **Data:** spike counts in 100 ms bins, z-scored per unit with the train-part mean / sd
- **Windows:** past 30 → next 10 bins (3 s → 1 s). Train windows stride 1; test windows stride 10, in the last 20% (chronological). Val = last 10% of the train part
- **Stimulus inputs** (590 inputs per unit; *state exactly*):
  - 13 stimulus-type rows over the past 30 and next 10 bins. Each row is the fraction of the bin inside a block of that type: drifting_gratings, drifting_gratings_75_repeats, drifting_gratings_contrast, dot_motion, flashes, gabors, natural_movie_one, natural_movie_one_more_repeats, natural_movie_one_shuffled, natural_movie_three, natural_scenes, spontaneous, static_gratings. A block is the type's presentations with gaps ≤ 2 s merged
  - running speed (cm/s ÷ 10, mean over the bin) over the 40 bins before the forecast start. No future running is used
  - total: own past 30 + 13 × 40 + 40
- **Score:** 1 − SSE / SSE_copylast, summed over the units of a session. Unit of every test = session
- **Settings as steps 11–15:** 5 leave-sessions-out folds (fold = rank of the session id within its genotype, mod 5), ridge alpha grid 1e-1…1e7 chosen on val then refit on train + val, k-means k = 16 (n_init 10, seed = fold, PCA 95%) on the training sessions' units only
- **k on val:** the val choice (step 13 procedure, k ∈ {2, 3, 4, 6, 8, 12, 16}) was 16 in 18 of 20 configuration-folds and 12 in 2 (`val_k.csv`). "Summary" = the val-chosen k (`r_kmsel`); k = 16 is also reported (`r_km16`) and is nearly identical
- **S features** (31): step 11's single-unit statistics computed on the z-scored counts:
  - z std (1 by construction), skew, kurtosis, event rate and time above z = 2, six IEI bins, autocorrelation at 0.1–5 s, relative power in five bands
  - Neuropixels stimulus features: mean z in spontaneous / gabors / flashes / drifting gratings / natural movie one (the types common to both session types), grating on − off, flash on − off, tuning peak and OSI over the drifting-grating directions present
  - 0.2% of units lack a flash feature, which is set to the training mean, as in step 13
- **Equal-weight references (added; see below):** k1 (one model, equal row weights, steps 13 / 15) and e_eq (shuffled labels, one model per fake group with equal row weights)

#### Ridge (`results/step16/gains.csv`, `session_scores.csv`, `neuron_scores.csv`, `label_vs_summary.csv`, `val_k.csv`, `ridge_alphas.csv`; plot `step16_gains.png`)
Score with no label: 0.435 (VISp) and 0.429–0.431 (all VIS). Gains over no label are the mean over sessions; "up" = sessions where the variant beats the reference; p = Wilcoxon over sessions.

| Scope, label set | Sessions | Label (d − b) | Summary (km − b) | Shuffled (e − b) | Summary − label | k1 − b | **Label (d − k1), equal weights** | **Summary (km − k1)** | Shuffled (e_eq − k1) |
|---|---|---|---|---|---|---|---|---|---|
| VISp, opto-tag (i) | 27 | +0.0021 (24, 5e-7) | +0.0046 (26, 3e-8) | +0.0000 (16, 0.18) | +0.0025 (25, 6e-7) | **+0.0022** (25, 5e-7) | **−0.0001** (9, 0.20) | **+0.0024** (25, 3e-7) | −0.0001 (12, 0.02) |
| VISp, FS / RS (ii) | 56 | +0.0007 (44, 2e-4) | +0.0002, median +0.0024 (47, 6e-7) | −0.0000 (19) | −0.0005, median +0.0022 (47, 9e-7) | −0.0003 (30, 0.73) | **+0.0010** (48, 3e-7) | **+0.0005**, median +0.0024 (50, 2e-7) | −0.0000 (21, 0.08) |
| all VIS, opto-tag (i) | 28 | +0.0006 (23, 5e-6) | +0.0011, median +0.0027 (23, 0.016) | +0.0000 (17) | +0.0005, median +0.0021 (22, 0.03) | **+0.0007** (25, 1e-6) | **−0.0001** (10, 0.06) | **+0.0004**, median +0.0020 (22, 0.03) | −0.0000 (11) |
| all VIS, FS / RS (ii) | 58 | +0.0003 (43, 0.002) | +0.0007, median +0.0021 (49, 2e-4) | −0.0000 (21) | +0.0004, median +0.0019 (49, 0.001) | −0.0004 (27, 0.02) | **+0.0007** (51, 5e-9) | **+0.0011**, median +0.0021 (51, 3e-5) | −0.0000 (21) |

- **The opto-tag "label gain" over b is a reweighting effect, not label information:**
  - With label set (i), the tagged groups are ~3% of the units. The label-balanced no-label model b (step 12 design: each group gets equal total weight) is therefore trained mostly on the few tagged units, and it predicts the untagged 97% worse
  - An unlabelled model with equal row weights (k1) gains exactly as much as the label: +0.0022 vs +0.0021 in VISp
  - Against k1, the opto-tag label gives −0.0001 (9/27 sessions up, p 0.20) in VISp and −0.0001 (10/28, p 0.06) over all areas. The shuffled label shows the same picture (e_eq − k1 ≈ 0, d − e_eq −0.00002, p 0.66)
  - So **the genetic opto-tag label carries no forecasting information beyond an unlabelled shared model**
- **The label-free activity summary does help:** +0.0024 over k1 in VISp (25/27 sessions, p 3e-7), +0.0046 over b. Summary − label +0.0025 (25/27, p 6e-7), consistent over Pvalb (7/8), Sst (12/12) and Vip (6/7) sessions
- **Waveform class (FS / RS) carries a little information:**
  - +0.0010 over k1 in VISp (48/56, p 3e-7), +0.0007 over all areas (51/58, p 5e-9). Every line is positive (VISp: Pvalb 8/8, Sst 10/12, Vip 6/7, wild type 24/29)
  - The gain comes from the FS units: +0.0043 on FS units (40/56), against −0.0002 on RS units
- **Summary vs waveform label:**
  - More sessions favour the summary: VISp 47/56 (p 9e-7), all VIS 49/58 (p 0.001)
  - The VISp mean favours the label (−0.0005), because the per-cluster ridge fails badly for a few units in a few sessions. Two wild-type sessions (744228101, 771990200) lose 0.08 and 0.04: some held-out units fall into small clusters whose model fits them poorly
  - On RS units the summary wins in 52/56 sessions (+0.0025); on FS units the label wins (summary − label −0.0077, median −0.0011, 22/56 sessions favour the summary, p 0.31)
- **Tagged units only (VISp, 14 sessions with tags):**
  - The per-group label model is worse than k1 for the tagged units themselves (−0.0063, 4/14 sessions up, p 0.017). It is trained on only 23 + 20 + 2 tagged VISp units (and on the tagged units of training folds only)
  - The summary is +0.0011 (8/14, p 0.63) on them, and summary − label is +0.0074 (11/14, p 0.04)
- **Shuffled labels** (permuted within each session) are ≤ 0 in every configuration
- **Label ÷ summary ratio** (bootstrap over sessions, `label_vs_summary.csv`):
  - opto-tag VISp: **0.47 [0.37, 0.57]** with the balanced b, but **−0.04 [−0.09, 0.02]** with equal weights (all VIS: −0.18 [−1.04, 0.95])
  - FS / RS: unstable, because the mean ridge summary gain is near zero (outliers above): VISp 3.5 [−4.5, 5.0], equal weights 2.0 [−10, 6.4]; all VIS, equal weights 0.62 [0.15, 3.5]
- **Precision check (`ridge_precision_check.csv`):** the all-area configurations store the per-unit ridge statistics in float32 (memory), with float64 sums. Rerunning spikes VISp opto-tag this way changes session scores by ≤ 4e-7 and leaves every alpha and k unchanged

#### MLP, label-balanced batches (the step 9 / 12 / 13 design; `gains.csv`, `mlp_fits.csv`)
**Models:**
- m1: no label
- m2: label FiLM
- m3: shuffled-label FiLM, 5 seeds
- cont: continuous-summary FiLM, using the standardised S vector of the fold

**Settings:** step 9's network (590 → 256 → 256 → 10), Adam 1e-3, batch 1024 with equal rows per label group, 400 batches / epoch, ≤ 40 epochs, patience 5, 20,000 val rows per group, 3 seeds (5 for m3), seeding formula of steps 9 / 12 (cont uses m2's seeds). Epochs: median 11–26 per kind; 2 of 280 fits reached the 40-epoch cap. Score with no label (m1): 0.437 (VISp opto-tag).

| Scope, label set | Label (m2 − m1) | Summary (cont − m1) | Shuffled (m3 − m1) | Label − shuffled | Summary − label |
|---|---|---|---|---|---|
| VISp, opto-tag (i), 27 | +0.0013 (20, p 0.07) | **+0.0026** (22, p 7e-4) | +0.0001 (10, 0.90) | +0.0013 (17, 0.07) | +0.0012 (18, 0.07) |
| VISp, FS / RS (ii), 56 | **+0.0009** (41, 5e-5) | −0.0014, median +0.0011 (36, 0.15) | +0.0004 (30, 0.54) | +0.0005 (41, 4e-4) | −0.0023, median +0.0001 (32, 0.53) |
| all VIS, opto-tag (i), 28 | +0.0022 (27, 3e-6) | −0.0009, median +0.0007 (16, 0.81) | **+0.0011** (24, 2e-5) | +0.0012 (25, 6e-4) | −0.0032 (9, 0.008) |
| all VIS, FS / RS (ii), 58 | **+0.0010** (40, 4e-5) | −0.0007, median +0.0009 (36, 0.34) | −0.0001 (25, 0.43) | +0.0011 (50, 9e-8) | −0.0017, median +0.0004 (30, 0.89) |

Entries: mean gain (sessions up, Wilcoxon p).

- **With balanced batches the opto-tag label does not separate cleanly from its shuffled control:**
  - Over all areas, even the *shuffled* label FiLM beats m1 by +0.0011 (24/28 sessions, p 2e-5)
  - The no-label m1 is fragile in this design. Its batches and its early-stopping val loss weight each group equally, so 14 Vip, 72 Sst and 133 Pvalb units count as much as 8,000 untagged ones
  - The equal-weight ridge showed no label information at all. The uniform-sampling MLPs (m1u…contu, next section) test this directly
- **Waveform class gives a small, consistent gain:** +0.0009 / +0.0010 (41/56, 40/58 sessions); label − shuffled +0.0005 / +0.0011. FS units gain most (+0.0019, 47/56 in VISp)
- **The continuous summary is not robust on held-out sessions:**
  - Typically it helps: median +0.0007 to +0.0016; +0.0026 in 22/27 VISp opto-tag sessions; RS units +0.0004 to +0.0007
  - Its validation loss on the training sessions is the lowest of all four models (all-VIS FS/RS: 1.148 vs 1.166 for m1)
  - But it amplifies the errors of a few non-stationary held-out units, units whose own copy-last score is already very negative (e.g. one VISpm unit goes from −4.3 to −11.6). Those units dominate a few sessions' SSE sums: the worst session loses 0.04–0.10, and the means turn negative in 3 of 4 configurations
  - So summary − label is positive only for VISp opto-tag (+0.0012, 18/27, p 0.07). For FS / RS the medians are ≈ 0 (+0.0001 / +0.0004) and the means negative
- **Tagged units only:**
  - The label FiLM: −0.0060 (5/14 sessions) in VISp; −0.0020 (15/24) over all areas
  - The summary: +0.0040 (8/14) and −0.0001 (13/24)
  - Neither is significant, with only 14 / 24 sessions that have tagged units
- **Per line (VISp opto-tag):** label +0.0022 / +0.0008 / +0.0012 and summary +0.0032 / +0.0024 / +0.0022 for Pvalb / Sst / Vip sessions (no line significant for the label)
- **Label ÷ summary ratio (MLP, balanced):** VISp opto-tag 0.53 [0.02, 1.00]. In the other configurations the mean summary gain is ≤ 0, so the ratio is undefined (−0.6 to −2.4, CIs spanning ±5 to ±27; `label_vs_summary.csv`)

#### MLP, uniform row sampling (m1u, m2u, m3u, contu)
**Setup:** the same networks, optimiser, epochs, early stopping and seeds (offsets 3 / 4 / 5, contu uses m2u's). Batches are drawn uniformly over all units' windows, and the val loss is uniform over 20,000 × (number of groups) val rows. 3 of 280 fits reached the 40-epoch cap.

| Scope, label set | m1u − m1 (no label: uniform vs balanced) | Label (m2u − m1u) | Summary (contu − m1u) | Shuffled (m3u − m1u) | Label − shuffled | Summary − label |
|---|---|---|---|---|---|---|
| VISp, opto-tag (i), 27 | **+0.0040** (25, 7e-8) | +0.0002 (16, 0.29) | +0.0001, median +0.0010 (17, 0.61) | +0.0003 (17, 0.23) | −0.0001 (14, 0.64) | −0.0001, median +0.0006 (14, 0.77) |
| VISp, FS / RS (ii), 56 | −0.0005 (22, 0.006) | +0.0003 (38, 0.006) | −0.0018, median +0.0007 (31, 0.37) | +0.0004 (35, 0.004) | −0.0001 (33, 0.28) | −0.0021, median +0.0004 (32, 0.64) |
| all VIS, opto-tag (i), 28 | **+0.0030** (28, 7e-9) | +0.0001 (13, 0.83) | −0.0006, median +0.0002 (14, 0.71) | +0.0003 (19, 0.09) | −0.0002 (11, 0.08) | −0.0007 (14, 0.70) |
| all VIS, FS / RS (ii), 58 | −0.0002 (27, 0.66) | +0.0005 (41, 0.005) | −0.0019, median −0.0002 (28, 0.36) | +0.0002 (30, 0.23) | +0.0003 (35, 0.07) | −0.0025 (28, 0.09) |

- **Balanced batches were the largest effect in the opto-tag MLPs:** sampling uniformly raises the no-label MLP by +0.004 (VISp) and +0.003 (all VIS) in all but 2 of 55 sessions. That is more than any label or summary gain. Uniform m1u (0.441 in VISp) is the best spike model of this step
- **Opto-tag label:** with uniform sampling it adds nothing: +0.0002 (p 0.29) in VISp and +0.0001 (p 0.83) over all areas. Label − shuffled ≤ 0. For the tagged units themselves it is −0.0040 (3/14 sessions up). This matches the equal-weight ridge
- **Waveform label:** +0.0003 / +0.0005 (38/56, 41/58 sessions, p ≈ 0.005), concentrated in FS units (+0.0022, 45/56, p 2e-4). It is not clearly above the shuffled FiLM: label − shuffled −0.0001 (p 0.28) in VISp and +0.0003 (p 0.07) over all areas
- **Continuous summary:** not significant in any configuration. Medians +0.0007 to +0.0010 in VISp; means negative from the same non-stationary units as above. The summary is useful for ridge (+0.0024 over k1 in VISp opto-tag) but adds nothing reliable to a uniformly trained MLP, which already exceeds the best ridge
- **Label ÷ summary (MLP, uniform):** undefined (summary gain ≈ 0 or < 0): 3.1 [−5.3, 6.7], −0.2 [−4.3, 5.0], −0.2 [−3.8, 3.2], −0.3 [−1.4, −0.05]

#### Task 2 in one paragraph
On Neuropixels spikes, with types mixed in every recording:
- **The genetic opto-tag label carries no forecasting information** once the training weights are equal: ridge −0.0001 (9/27 sessions), MLP +0.0002 (p 0.29). The apparent gains with the step 12 label-balanced design (ridge +0.0021, MLP +0.0013) come from reweighting the ~3% tagged units, and a no-label model with equal weights obtains them too
- **The waveform class (FS / RS) carries a small, reliable amount** for ridge (+0.0010, 48/56) and a marginal amount for the MLP, almost all for FS units
- **The label-free activity summary beats the opto-tag label for ridge** (summary − label +0.0025, 25/27, p 6e-7), but not for an MLP trained without label balancing, where neither helps

Absolute gains are an order of magnitude smaller than for Allen dF/F, and similar to or smaller than Bugeon's. See the comparison table below.

#### Comparison with Allen and Bugeon (`results/step16/comparison.csv`)
Mean gain over no label; ratio = label gain ÷ summary gain with 95% bootstrap CI over units (mice / sessions). Allen and Bugeon rows are from step 15 (balanced design: b, m1).

| Dataset | Scope, label | Units | Label, ridge | Summary, ridge | Ratio, ridge | Label, MLP | Summary, MLP | Ratio, MLP |
|---|---|---|---|---|---|---|---|---|
| Allen dF/F (10 Hz) | VISp, Cre line | 46 mice | +0.0252 | +0.0473 | 0.53 [0.41, 0.65] | +0.0239 | +0.0436 | 0.55 [0.37, 0.70] |
| Allen deconvolved A (10 Hz) | VISp, Cre line | 46 mice | +0.0070 | +0.0082 | 0.86 [0.66, 1.15] | +0.0029 | +0.0118 | 0.24 [0.06, 0.39] |
| Allen deconvolved B (4.3 Hz) | VISp, Cre line | 46 mice | +0.0112 | +0.0107 | 1.05 [0.82, 1.48] | +0.0077 | +0.0200 | 0.39 [0.25, 0.51] |
| Bugeon (deconvolved, 4.3 Hz) | t-type | 17 sessions | +0.0014 | +0.0133 | 0.11 [−0.07, 0.24] | +0.0018 | +0.0174 | 0.10 [0.08, 0.12] |
| **Neuropixels spikes**, balanced | VISp, opto-tag | 27 sessions | +0.0021 | +0.0046 | 0.47 [0.37, 0.57] | +0.0013 | +0.0026 | 0.53 [0.02, 1.00] |
| **Neuropixels spikes**, equal weights / uniform | VISp, opto-tag | 27 sessions | −0.0001 | +0.0024 | **−0.04 [−0.09, 0.02]** | +0.0002 | +0.0001 | undefined (3.1 [−5.3, 6.7]) |
| Neuropixels spikes, balanced | VISp, FS / RS | 56 sessions | +0.0007 | +0.0002 | undefined (3.5 [−4.5, 5.0]) | +0.0009 | −0.0014 | undefined |
| Neuropixels spikes, equal weights / uniform | VISp, FS / RS | 56 sessions | +0.0010 | +0.0005 | 2.0 [−10, 6.4] | +0.0003 | −0.0018 | undefined |
| Neuropixels spikes, balanced | all VIS, opto-tag | 28 sessions | +0.0006 | +0.0011 | 0.56 [−4.5, 4.1] | +0.0022 | −0.0009 | undefined |
| Neuropixels spikes, equal weights / uniform | all VIS, opto-tag | 28 sessions | −0.0001 | +0.0004 | −0.18 [−1.0, 0.95] | +0.0001 | −0.0006 | undefined |
| Neuropixels spikes, balanced | all VIS, FS / RS | 58 sessions | +0.0003 | +0.0007 | 0.43 [−3.5, 3.6] | +0.0010 | −0.0007 | undefined |
| Neuropixels spikes, equal weights / uniform | all VIS, FS / RS | 58 sessions | +0.0007 | +0.0011 | 0.62 [0.15, 3.5] | +0.0005 | −0.0019 | undefined |

- **With the same balanced design as Allen / Bugeon,** the VISp opto-tag ratio (0.47 ridge, 0.53 MLP) looks Allen-like. But here that design produces a label gain that an unlabelled equal-weight model matches. With equal weights the ratio is −0.04 [−0.09, 0.02], at or below Bugeon's 0.10–0.11
- **The Allen result does not have this problem:** in Allen dF/F the equal-weight k1 scored *below* the balanced b (0.278 vs 0.300, step 15 reproduction table), even though Cux2 holds 88% of the neurons. So reweighting cannot explain the Allen label gain over b. The Allen and Bugeon MLP rows have no uniform-sampling counterpart (see Problems)
- "undefined": the mean summary gain is ≤ 0 or near 0, so the ratio and its CI are not interpretable

### Task 3: synthetic calcium from the same spikes (K5, C1)
**Simulation** (`calcium_arrays` in `step16_neuropixels.py`; deterministic, regenerated on demand; not stored):
- **Clean trace:** each unit's spike train at its native spike times, convolved with a double-exponential kernel exp(−t/τ_decay) − exp(−t/τ_rise), normalised to peak 1 per spike. It is evaluated exactly at 30 Hz frame times (3 frames per 100 ms bin, frame centres), as sums over the spikes, so no binning happens before the kernel. The model is linear: no indicator saturation or supralinearity
- **Kinetics:**

  | Indicator | τ_rise | τ_decay | Source |
  |---|---|---|---|
  | GCaMP6f | 0.050 s | **0.427 s** | decay: median OASIS decay of the step 15 GCaMP6f neurons (Cux2 / Sst / Vip, 3,180) in `results/step15/oasis_params.csv`; rise: approximate value from Chen et al. 2013 |
  | GCaMP6s | 0.150 s | **0.832 s** | decay: median OASIS decay of the step 15 GCaMP6s (Pvalb, Ai162) neurons, 116; rise: approximate, Chen et al. 2013 |
  | GCaMP6s, slow (sensitivity) | 0.150 s | 1.5 s | |

- **Noise matched to the Allen 2P SNR:**
  - Per Allen neuron, SNR = (train-part dF/F variance − sn²) / sn². sn is the OASIS noise estimate at 30 Hz from step 15; the dF/F std is step 10's, both from the train part. Medians: 6f lines 1.80 [IQR 0.66, 5.6], 6s line 1.41 [0.36, 4.7] (`allen_snr.csv`)
  - Each synthetic unit draws one SNR from the distribution of its indicator, at a quantile fixed by its unit id, so the same quantile is used in every version. White Gaussian noise of variance var(clean train trace) / SNR is added at 30 Hz, with the same noise realisation per unit in every version
- **Pipeline as Allen (step 5 `load_session`):** z-score at 30 Hz with the train-part mean / std, then 3-frame means → the same 10 Hz bins, windows, splits, stimulus inputs, folds and models as the spikes. The S summary is recomputed from each signal's own train part
- **Versions:**
  - `c6f`: all units GCaMP6f
  - `c6s`: all units GCaMP6s
  - `callen` (Allen-like): units of Pvalb mice get GCaMP6s, all others GCaMP6f, which mimics the Allen indicator confound (Pvalb = Ai162 / 6s)
  - `callen_slow`: as callen with the 1.5 s 6s decay
- **Models:** ridge (b, d, e, e_eq, k1, km) on VISp and all VIS, both label sets. MLP (balanced and uniform designs) on VISp, both label sets
- **Added check (ridge, VISp):** a recording-level label, the genotype of the mouse (Pvalb / Sst / Vip / wild-type mouse; one label per session, like Allen's Cre line). Its shuffled control permutes the genotypes across sessions (step 8 style). This tests directly whether the simulated indicator recreates a label gain when the label marks the recording

#### Ridge (`gains.csv`, `calcium_vs_spikes.csv`, `label_vs_summary.csv`; plot `step16_calcium.png`)
Score with no label (b): spikes 0.435, 6f 0.25–0.26, 6s 0.20–0.22, Allen-like 0.24, Allen-like slow 0.23–0.24 (VISp). Copy-last is a stronger baseline on the smooth calcium signals, so all scores are lower, as for Allen dF/F (0.30). Equal-weight design (gains over k1):

| Signal (VISp) | Opto-tag label | Summary | Ratio [95% CI] | FS / RS label | FS / RS summary | FS / RS ratio |
|---|---|---|---|---|---|---|
| spikes | −0.0001 (9/27) | +0.0024 (25/27) | −0.04 [−0.09, 0.02] | +0.0010 (48/56) | +0.0005 (50/56) | 2.0 [−10, 6.4] |
| GCaMP6f | −0.0003 (9/27) | +0.0139 (26/27) | −0.02 | +0.0051 (46/56) | +0.0046 (54/56) | 1.1 |
| GCaMP6s | −0.0002 (8/27) | +0.0350 (27/27) | −0.01 | +0.0061 (47/56) | +0.0334 (54/56) | 0.18 |
| Allen-like | +0.0002 (14/27) | +0.0280 (27/27) | +0.01 | +0.0052 (45/56) | +0.0136 (53/56) | 0.38 |
| Allen-like, 6s 1.5 s | +0.0008 | +0.0434 | +0.02 | +0.0050 | +0.0204 | 0.25 |

All VIS: the opto-tag label stays within −0.0002…+0.0008 and the summary is +0.011 to +0.034. FS / RS label +0.0024…+0.0031, summary +0.008…+0.033.

- **Change from spikes to calcium on the same units** (`calcium_vs_spikes.csv`, paired bootstrap over sessions):
  - The opto-tag label gain does not change (VISp: 6f −0.0002 [−0.0005, 0.0001]; Allen-like +0.0003 [−0.0003, 0.0009], p 0.47)
  - Its ratio to the summary stays ≈ 0 (Allen-like +0.007, CI of the change [−0.001, 0.095])
  - The FS / RS label gain grows ~5× (+0.004 to +0.005, CIs above 0, p ≤ 0.004)
- **The summary gain grows the most:**
  - 6f: 6×; 6s: 15×; Allen-like: 12×; slow: 18×
  - Over b it approaches Allen dF/F's size (+0.047 in step 15): +0.037 with GCaMP6s, +0.046 with the slow Allen-like version
  - In the Allen-like versions it is largest in the Pvalb mice (+0.061 / +0.11, vs +0.015 / +0.012 in Sst / Vip mice)
- **So a slow indicator mostly makes per-unit conditioning in general more useful.** The summary captures the per-unit decay (here set by the indicator), not the genetic label
- **Per line (Allen-like):** the opto-tag label is +0.0012 in Pvalb mice (+0.0031 with the 1.5 s decay) and ≈ 0 in Sst / Vip mice. Tagged and untagged units in a Pvalb mouse share 6s, but the "untagged" model is trained mostly on 6f units of other mice; the tagged-Pvalb model is a 6s model. This is the only place where the per-unit label picks up some indicator, and it is small because tagged units are few

#### Recording-level genotype label (added check; ridge, VISp, 56 sessions)
The label is the genotype of the mouse (Pvalb / Sst / Vip / wild type), the same for every unit of a recording, as Allen's Cre line. Equal-weight design (d − k1); shuffled = genotypes permuted across sessions:

| Signal | Label (d − k1) | Median | Sessions up | p | Label − shuffled (p) | Pvalb / Sst / Vip / wt mice | Ratio, equal weights [CI] |
|---|---|---|---|---|---|---|---|
| spikes | −0.0002 | −0.0002 | 17/56 | 0.02 | +0.0002 (0.09) | +0.0006 / −0.0010 / +0.0002 / −0.0001 | −0.35 [−0.97, 1.75] |
| GCaMP6f | −0.0005 | −0.0004 | 20/56 | 0.02 | +0.0003 (0.86) | +0.0010 / −0.0015 / −0.0002 / −0.0005 | −0.10 [−0.21, 0.11] |
| GCaMP6s | −0.0006 | −0.0004 | 24/56 | 0.02 | +0.0002 (0.70) | −0.0000 / −0.0007 / +0.0002 / −0.0010 | −0.02 [−0.04, −0.005] |
| **Allen-like** | **+0.0043** | +0.0000 | 29/56 | 0.53 | **+0.0055 (0.002)** | **+0.038** / −0.0010 / +0.0004 / −0.0018 | 0.31 [−0.59, 0.87] |
| **Allen-like, 6s 1.5 s** | **+0.0114** | −0.0000 | 28/56 | 0.54 | **+0.0134 (0.006)** | **+0.093** / −0.0010 / +0.0003 / −0.0032 | 0.56 [−0.36, 1.73] |

- **When the indicator is the same for every mouse (spikes, all 6f, all 6s), the genotype of the mouse carries nothing** (≤ 0, at or below the shuffled control)
- **When it differs by genotype (Allen-like), the genotype label gains in the Pvalb mice** (+0.038 per session; +0.093 with the 1.5 s decay), and nowhere else. The 8 Pvalb sessions are exactly the 8 largest gains, each ≥ +0.021, or ≥ +0.061 with the slow decay (plot `step16_calcium.png`). It is significant against its shuffled control (+0.0055, 37/56, p 0.002)
- In the balanced design (d − b) the Allen-like genotype gain is +0.0027 (38/56, p 0.003) and +0.0080 (41/56, p 0.001) for the slow version, against −0.0004 / −0.0010 / −0.0012 for spikes / 6f / 6s
- **So an indicator assigned per recording alone recreates an Allen-style, Pvalb-concentrated label gain**, with a label ÷ summary ratio of 0.3–0.6 (wide CIs), against Allen dF/F's 0.53 (step 15, where Pvalb had the largest per-line gain, +0.056 ridge). It does this with no genetic information in the activity
- The per-unit opto-tag label cannot pick this up, because tagged and untagged units in the same mouse share the indicator

#### MLP on synthetic calcium (VISp; `gains.csv`, `calcium_vs_spikes.csv`)
Mean gain (sessions up, Wilcoxon p):

| Signal | m1u − m1 (opto-tag) | Opto-tag label, uniform (m2u − m1u) | Opto-tag summary, uniform (contu − m1u) | FS / RS label, uniform | FS / RS summary, uniform | Opto-tag label, balanced (m2 − m1) | Opto-tag summary, balanced |
|---|---|---|---|---|---|---|---|
| spikes | +0.0040 (25/27) | +0.0002 (16/27, 0.29) | +0.0001 (17/27, 0.61) | +0.0003 (38/56, 0.006) | −0.0018, median +0.0007 (31/56) | +0.0013 (20/27) | +0.0026 (22/27) |
| GCaMP6f | +0.0143 (27/27) | −0.0001 (17/27, 0.7) | **+0.0134** (23/27, 2e-6) | +0.0005 (29/56, 0.9) | +0.0028 (50/56, 4e-8) | +0.0092 (23/27) | +0.0207 (27/27) |
| GCaMP6s | +0.0144 (26/27) | −0.0013 (11/27, 0.2) | **+0.0307** (27/27, 1e-8) | −0.0042 (15/56, 0.001) | +0.0200 (55/56, 1e-9) | +0.0080 (23/27) | +0.0351 (27/27) |
| Allen-like | +0.0113 (23/27) | −0.0001 (11/27, 0.4) | **+0.0249** (25/27, 7e-8) | +0.0011 (25/56, 0.7) | +0.0020 (49/56, 2e-8) | +0.0077 (21/27) | +0.0282 (27/27) |
| Allen-like, 1.5 s | +0.0113 (23/27) | +0.0016 (17/27, 0.2) | **+0.0353** (26/27, 3e-8) | −0.0047 (25/56, 0.3) | +0.0064 (50/56, 1e-8) | +0.0042 (18/27) | +0.0370 (27/27) |

- **The MLP agrees with the ridge:** with uniform sampling the opto-tag label adds nothing on any signal (−0.0013 to +0.0016, no p < 0.15), while the summary gains +0.013 to +0.035 (23–27/27 sessions)
- **The rebalancing artifact grows on calcium:** uniform beats balanced m1 by +0.011 to +0.014. The balanced label gains (+0.004 to +0.009, 18–23/27 sessions) are again of that size and vanish with uniform sampling
- **The FS / RS label FiLM does not help the uniform MLP on calcium** (−0.0047 to +0.0011), while the ridge FS / RS label grew to +0.005. For GCaMP6s the label FiLM and the shuffled FiLM are both below m1u (−0.0042 and −0.0101)
- **Label ÷ summary, uniform MLP:** −0.04 to +0.05 for the opto-tag on every calcium version (6f −0.006 [−0.13, 0.10]; Allen-like −0.003 [−0.16, 0.12]; slow 0.05 [−0.04, 0.11]); spikes undefined
- Fits: 1,680 MLP fits in this step; 12 (0.7%) reached the 40-epoch cap

**Key question:**
- **With spikes and no indicator, and types mixed within each recording, the genetic (opto-tag) label gain is zero, smaller even than Bugeon's**, once training weights are not label-balanced: ridge −0.0001 (9/27), MLP +0.0002 (p 0.29), ratio −0.04 [−0.09, 0.02] vs Bugeon 0.10–0.11. With the step 12 balanced design it looks like +0.002 with ratio ≈ 0.5, but an unlabelled equal-weight model gets the same gain
- **The waveform class (FS / RS) carries a small, reliable amount on spikes** (ridge +0.0010, 48/56). It is the better "label" on spikes for FS units
- **Does the activity summary beat the label?**
  - Yes for ridge (opto-tag: summary − label +0.0025, 25/27, p 6e-7; FS / RS: more sessions favour the summary, 47/56, though the mean favours the label because of a few cluster failures)
  - For the MLP on spikes neither helps: a uniformly trained MLP already exceeds every ridge
- **Does a simulated indicator alone recreate the Allen-style label gain?**
  - **For a per-unit genetic label, no:** on identical neurons, the opto-tag label stays ≈ 0 with GCaMP6f, GCaMP6s or the Allen-like mixture (ridge and MLP)
  - **For a per-recording label, yes, partly:** in the Allen-like versions the genotype of the mouse gains +0.038 to +0.093 in the Pvalb (6s) mice, with a ratio of 0.3–0.6, close to Allen's 0.53. With a uniform indicator it gains nothing
  - **What a slow indicator does recreate on every version is a large activity-summary gain:** +0.014 (6f) to +0.043 (slow Allen-like), up to Allen dF/F's +0.047, on neurons whose spike-level summary gain is +0.002
  - So the Allen label gain is consistent with the indicator confound (one indicator per line and recording) plus the generally larger value of per-unit conditioning on calcium. It does not need cell-type information

## Plots
- `plots/step16_neuropixels.png` (overview):
  - (a) units per label: tagged per line and FS, VISp and all VIS
  - (b) label vs summary gain for Neuropixels spikes next to Allen dF/F / deconvolved A / B and Bugeon, balanced and (hatched) equal-weight / uniform designs
  - (c) label ÷ summary ratio with bootstrap CI for every dataset and VISp configuration
  - (d) synthetic calcium: label gains (opto-tag, genotype, FS / RS) and summary gain per signal, on the same neurons
- `plots/step16_optotag.png` (task 1):
  - mean response to the 10 ms pulse, tagged vs untagged, per line
  - the tagging criterion with the wild-type false positives
  - waveform durations of tagged units
  - every tagged unit's response
- `plots/step16_gains.png` (task 2): per-session gains (label, summary, shuffled) for VISp / all VIS × opto-tag / FS-RS, for the four families (ridge balanced b, ridge equal weights k1, MLP balanced, MLP uniform), coloured by genotype, with mean ± sem and median
- `plots/step16_calcium.png` (task 3):
  - example tagged VISp units as z-scored spikes vs synthetic 6f / 6s
  - per-session label gains on each signal (opto-tag ridge and uniform MLP, genotype ridge, FS / RS ridge)
  - label ÷ summary ratio per signal

## Problems / errors
- **Label-balanced training creates an artificial label gain when one group dominates** (the most important caveat of this step):
  - The step 12 design (b and m1 balanced over the label groups) gives the 3% tagged units 75% of the weight, which handicaps the no-label baseline on the untagged majority
  - Equal-weight / uniform-sampling references were therefore added (ridge k1 / e_eq, MLP m1u / m2u / m3u / contu; 840 extra MLP fits). Both designs are reported. The conclusions about the opto-tag label rest on the equal-weight versions
  - In Allen dF/F, k1 scored below b, so this artifact does not explain the Allen label gain. But steps 12–15 (Bugeon, Allen) had no uniform-sampling MLPs, so their MLP rows in the comparison table are balanced only
- **Few tagged units:**
  - VISp has 23 Pvalb, 20 Sst and only 2 Vip tagged units (all VIS: 133 / 72 / 14)
  - The per-group label models of the tagged types are fitted on very few units (Vip: alpha 0.3–316 across folds)
  - The tagged-unit-only comparisons have 14 (VISp) / 24 (VIS) sessions and are not significant
  - With the p < 0.001 criterion, the expected false tags are ≈ 1 of 14 Vip, ≈ 2 of 72 Sst and ≈ 1 of 133 Pvalb (wild-type rate)
  - Tagging misses many Cre+ cells (light reaches only part of the cortex), so "untagged" contains unlabelled Pvalb / Sst / Vip cells
- **The per-cluster ridge summary has a few large failures:** a few held-out units fall into small clusters (some k = 16 clusters have 1 training unit, as in step 13), so two wild-type sessions lose 0.04–0.08. Means and medians are both reported; Wilcoxon (rank-based) and sessions-up are robust to this
- **The continuous-summary MLP amplifies the errors of a few non-stationary held-out units** (units whose own copy-last score is very negative) on spikes, turning the mean negative in 3 of 4 spike configurations while the median stays positive. On calcium it helps clearly
- **Label ÷ summary ratios are undefined where the summary gain is ≈ 0** (spikes FS / RS, spikes MLP), and their bootstrap CIs are then meaningless (marked "undefined")
- **Synthetic calcium is idealised:**
  - linear, no saturation or supralinearity
  - one kinetic per indicator: the median of the step 15 OASIS fits, not per-neuron kinetics
  - rise times approximate (Chen et al. 2013, taken as stated in the brief; the paper could not be opened from this environment)
  - white noise with an SNR drawn from the Allen distribution, no neuropil, motion or bleaching
  - Allen's real Pvalb mice also differ in reporter line and expression level, which this does not model
- **Copy-last baselines differ by signal** (score with no label: spikes 0.44, calcium 0.20–0.26), so absolute gains are not on a common scale. The ratios and the within-neuron spike → calcium changes are the comparisons least affected
- **Extraction failures, all resolved:**
  - one Pvalb session (746083955) has no `invalid_times` table, and four wild-type sessions have no optotagging module
  - both first raised errors and were marked failed in the manifest; the extractor was fixed (missing tables allowed) and the 5 sessions were re-extracted
  - a Content-Length check on every download was added; no download failed in the end
  - `results/step16/manifest_summary.csv` lists every session with its unit counts
- **Two bugs were found and fixed before any reported analysis:**
  - an invalid-time interval on a probe with no visual units was applied to all probes, which would have removed every unit of session 757216464
  - the labels phase failed on sessions without light pulses
- **Memory:** the all-VIS ridge holds the per-unit statistics in float32 (checked: ≤ 4e-7 change in session scores). Synthetic calcium is regenerated on demand and the ridge statistics are deleted after use (disk)
- **Size:** `neuron_scores.csv` holds per-unit scores for spikes (all configurations) and calcium VISp; the calcium all-VIS configurations are in `session_scores.csv` only (results/step16 is 14 MB)
- **The opto-tag analyses use the 28 Cre sessions (27 with VISp units);** FS / RS, genotype and summary analyses use all 58 (56 with VISp units)

### Code and ideas from other work
- **This repo:**
  - step 9: MLP, FiLM, optimiser / early stopping
  - step 11: S features
  - step 12: `neuron_stats` (re-implemented with contiguous products, checked against the original), `assemble`, `fit_weighted`, `balanced_weights`, `predict`, Sampler
  - step 13: `fit_group`, `fit_clusters`, `val_sse`, `unit_val_score`, `CondMixin` (continuous-summary FiLM), k on val
  - step 15: decay constants, comparison rows
  - `scripts/allen_optotag_session.py`: the tutorial tagging windows
- **AllenSDK optotagging tutorial:** windows and ratio criterion (re-implemented for the stored spike times)
- No code was copied from other repositories

## What I noticed
- On spikes, with every type mixed in every recording, the genetic label is useless for the 1 s forecast once the training weights are fair. The large-looking gain under the step 12 design came from reweighting a 3% minority. The activity summary helps a linear model; a well-trained MLP does not need it.
- Feeding the same spikes through a simulated indicator makes per-unit conditioning 6–18× more valuable, and an indicator that follows the mouse line hands a recording-level label an Allen-like, Pvalb-only gain. That fits the reading of steps 11–15: the Allen Cre-label gain reflects indicator and recording confounds, not cell-type dynamics.

## References
- Siegle, J.H. et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. Nature 592, 86–92
- AllenSDK optotagging example (ecephys_optotagging notebook): the 10 ms pulse, the [1, 9) ms / [−10, −2) ms windows and the ratio > 2 criterion
- Chen, T.-W. et al. (2013). Ultrasensitive fluorescent proteins for imaging neuronal activity. Nature 499, 295–300. Source of the approximate rise times (~50 ms GCaMP6f, ~150 ms GCaMP6s). The values were taken as stated in the step brief; the paper could not be opened from this environment (blocked network)
- Dana, H. et al. (2016). Sensitive red protein calcium indicators for imaging neural activity. eLife 5, e12727. Named in the step brief as a kinetics reference. **Not used here:** the decay constants come from the step 15 OASIS fits to the Allen data instead (see Task 3)
- Step 15 of this repo (`results/step15/oasis_params.csv`, `results/step15_result.md`): the GCaMP6f / 6s decay constants (median OASIS τ 0.427 / 0.832 s) and the noise estimates sn; step 10 (`results/step10/neuron_features.csv`): dF/F std for the SNR
- Steps 9, 11, 12, 13 of this repo: MLP / FiLM, S features, the ridge from summed per-unit statistics, k-means summary, continuous-summary FiLM, k chosen on val
- de Vries, S.E.J. et al. (2020). A large-scale standardized physiological survey reveals functional organization of the mouse visual cortex. Nat Neurosci 23, 138–151 (Allen 2P reporters per line: Pvalb = Ai162 / GCaMP6s)
- Bugeon, S. et al. (2022). A transcriptomic axis predicts state modulation of cortical interneurons. Nature 607, 330–338 (the Bugeon rows, via steps 12–15)

## Part 2 notes
**Re-extracting the data (nothing under `data/` is committed):**
```
uv venv -p python3.11 .venv && source .venv/bin/activate
uv pip install "setuptools<70" allensdk==2.16.2 statsmodels==0.13.5 scikit-learn==1.7.2 torch==2.14.1 h5py threadpoolctl matplotlib
mkdir -p data/step16/meta && cd data/step16/meta
for f in manifest.json sessions.csv units.csv probes.csv channels.csv; do
  curl -sSO https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache/$f; done
cd ../../..
python scripts/step16_extract.py                 # ~2.5 h; resumable; data/step16/manifest.csv; ≤ 2 NWB on disk
python scripts/step16_neuropixels.py labels      # task 1 -> data/step16/labels.csv, psth.pkl
python scripts/step16_neuropixels.py bin         # 10 Hz counts, stimulus inputs, windows -> data/step16/spk/<id>.npz
```

**Part 1 phases** (`<ds>` = spk, c6f, c6s, callen, callen_slow; `<scope>` = VISp, VIS; `<ls>` = tag, fs; each phase is resumable):
```
python scripts/step16_neuropixels.py gram <ds>                          # per-unit ridge statistics (~6 GB per dataset; delete after ridge)
python scripts/step16_neuropixels.py features <ds>
python scripts/step16_neuropixels.py cluster <ds> <scope> <ls>
python scripts/step16_neuropixels.py ridge <ds> <scope> <ls>
python scripts/step16_neuropixels.py mlp <ds> <scope> <ls> [uniform]   # balanced (default) or uniform sampling
STEP16_GRAM_DIR=gram_visp python scripts/step16_neuropixels.py gram <ds> VISp     # genotype check: VISp-only statistics
cp data/step16/<ds>/VISp_fs/clusters.pkl data/step16/<ds>/VISp_geno/
STEP16_GRAM_DIR=gram_visp python scripts/step16_neuropixels.py ridge <ds> VISp geno
python scripts/step16_neuropixels.py analyse && python scripts/step16_plots.py
```

**What is on disk for Part 2:**
- `data/step16/sessions/<id>.npz` (3.5 GB, 58 files), per session:
  - `spikes` (uint32, 30 kHz ticks) with `spike_offsets` and `tick`
  - `units`: unit id, area, probe, channel, probe vertical / horizontal position, surface channel, quality, waveform duration / half-width, firing rate, SNR, QC metrics
  - `stim`: every stimulus presentation with name, start, stop, orientation, contrast, TF, SF, frame
  - `run_t` / `run_speed`: running speed in cm/s at the native rate
  - `opto`: optogenetic epochs
  - `invalid_start` / `invalid_stop` / `invalid_tags`
  - `genotype`, `session_type`, `line`
- `data/step16/spk/<id>.npz` (2.3 GB):
  - kept units' 10 Hz counts `Ctr` / `Cte` with train-part `mean_count` / `sd_count`
  - stimulus inputs `Str` / `Ste`; `run` (binned running speed)
  - per-probe `fit_` / `val_` / `te_` window starts and `valid_` bin masks
  - `trial_rows` (grating / flash trials for the S features), `t0`, `t1`, `n_tr`, `n_fit`
- Synthetic calcium is regenerated on demand (`session_arrays(ds, sid)`, deterministic, ~4 s per session) and is not stored
- **Part 2 needs:** history length / atlas from the same windows; identity vs state (running speed is in both files; stimulus epochs in `stim`); horizons (re-bin from `spikes`); typing from `labels.csv` / `unit_labels.csv`

**Where each Part 1 result is:**
- **Task 1:** `results/step16/unit_labels.csv` (per unit: labels, tagging statistics, waveform), `label_counts.csv` (session × area × label), `optotag_summary.csv`, `optotag_psth*.csv`, `manifest_summary.csv`; plot `plots/step16_optotag.png`
- **Task 2:**
  - `gains.csv`: every gain, per dataset × scope × label set × unit subset × line × family, with mean, median, sem, sessions up, Wilcoxon p
  - `session_scores.csv`, `neuron_scores.csv` (per-unit scores of every variant), `label_vs_summary.csv` (ratios with bootstrap CI), `comparison.csv` (with the step 15 Allen / Bugeon rows)
  - `val_k.csv`, `ridge_alphas.csv`, `mlp_fits.csv`, `ridge_precision_check.csv`
  - plots `plots/step16_gains.png`, `plots/step16_neuropixels.png`
- **Task 3:** the same files with ds = c6f / c6s / callen / callen_slow, plus `calcium_vs_spikes.csv` (change of label gain and ratio vs spikes, paired bootstrap), `allen_snr.csv`, `example_traces.csv`; plot `plots/step16_calcium.png`

**Lessons for Part 2:**
- **Report the equal-weight / uniform designs alongside the label-balanced ones:** with a 3% tagged minority, label-balanced training creates an artificial label gain
- The 10 ms-pulse tag criterion needs the p < 0.001 threshold (wild-type false-positive rate)
- The per-cluster ridge summary has a few large failures (small clusters); use medians and session counts next to means
