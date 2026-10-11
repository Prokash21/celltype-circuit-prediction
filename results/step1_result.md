# Step 1: Load one Allen Brain Observatory two-photon session
Date: 2026-10-12

## What was run
- `notebooks/step1_load_session.ipynb` (allensdk 2.16.2, Python 3.10, `.venv`)
- BrainObservatoryCache → Sst-IRES-Cre + VISp → lowest experiment id → dF/F traces

## Data
- Experiment id: 573720508 (three_session_A)
- Cre line / area / depth: Sst-IRES-Cre / VISp / 275 µm
- Neurons: 10
- Frames / frame rate / duration: 115,468 / 30.08 Hz / 3839.3 s (64.0 min)

## Results
- Matching Sst-IRES-Cre VISp experiments: 212
- Plotted neuron indices (seed 0): 4, 7, 2, 3, 5
- Max dF/F in first 60 s: 3.06, 2.70, 1.30, 1.17, 0.85

## Plots
- plots/step1_traces.png

## Problems / errors
- System Python 3.9 → made Python 3.10 env with uv
- allensdk import errors → pinned `setuptools<81`, `statsmodels<0.14.5`

## What I noticed
- Several neurons are co-active at ~26, 41, 47 and 56 s.
- After the gratings start at 24.7 s, activity rises with some 3 s grating trials (checked with the stimulus table).
- Events at ~41 s and ~47 s both match 225° grating trials (possible direction selectivity, to check later).
- The last neuron declines slowly at the start, likely baseline drift.
