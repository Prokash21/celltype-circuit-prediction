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
