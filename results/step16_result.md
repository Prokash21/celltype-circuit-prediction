# Step 16 (Part 1 of 2): Neuropixels spikes, no calcium indicator. Genetic label vs activity summary, and synthetic calcium from the same spikes
Date: 2026-10-19

## What was run
- `scripts/step16_extract.py`: resumable extraction of all 58 Allen Visual Coding Neuropixels sessions, one session at a time (download NWB → `data/step16/sessions/<id>.npz` → delete the NWB)
- `scripts/step16_neuropixels.py <phase>`: `labels` (task 1), `bin`, then per dataset `gram`, `features`, and per configuration `cluster`, `ridge`, `mlp` (tasks 2–3)
- `scripts/step16_analyse.py`: writes `results/step16/*.csv`. `scripts/step16_plots.py` makes every plot from those CSV files only
- **Environment:** Python 3.11 (`.venv`, rebuilt), numpy 1.23.5, pandas 1.5.3, scipy 1.10.1, scikit-learn 1.7.2, PyTorch 2.14.1 (CPU), allensdk 2.16.2 (needs setuptools < 70 and statsmodels 0.13.5 to import), h5py 3.16. These are the library versions of steps 11–15. AllenSDK is imported only because the step 3–13 modules import it; the NWB files are read directly with h5py

*Status of this file: task 1 is complete. Tasks 2 and 3 are running and their sections are filled in as each phase finishes.*

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
  - The equal-weight ridge showed no label information at all. The uniform-sampling MLPs (m1u…contu, running) test this directly
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
*Running.*

### Task 3: synthetic calcium from the same spikes (K5, C1)
*Running.*

## Plots
- `plots/step16_optotag.png`:
  - top: mean ± sem rate around the 10 ms pulse for tagged vs untagged units, per line (all VIS), with the baseline and evoked windows
  - bottom left: the criterion (log ratio vs −log10 p; red = tagged, blue = wild type)
  - bottom middle: waveform duration of the tagged units per line vs untagged, with the 0.4 ms FS threshold
  - bottom right: every tagged unit's baseline-normalised response

## Problems / errors
- **Extraction failures, all resolved:**
  - one Pvalb session (746083955) has no `invalid_times` table
  - four wild-type sessions have no optotagging module
  - both first raised errors and were marked failed in the manifest. The extractor was fixed (missing tables allowed) and the 5 sessions were re-extracted
  - a Content-Length check on every download was added. No download failed in the end
  - `results/step16/manifest_summary.csv` lists every session with its unit counts
- **An invalid-interval bug was found and fixed before any analysis:** an interval on a probe with no visual units was applied to all probes, which would have removed every unit of session 757216464

## References
- Siegle, J.H. et al. (2021). Survey of spiking in the mouse visual system reveals functional hierarchy. Nature 592, 86–92
- AllenSDK optotagging example (ecephys_optotagging notebook): the 10 ms pulse, the [1, 9) ms / [−10, −2) ms windows and the ratio > 2 criterion
