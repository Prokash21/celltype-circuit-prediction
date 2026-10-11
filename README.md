# celltype-circuit-prediction

Can knowing a neuron's **cell type** help predict its **activity** in a circuit?

This project uses public mouse brain data to test that question step by step:
start with simple forecasting baselines on calcium imaging, then add cell-type
information and see whether predictions improve.

## Target

- Forecast near-future neural activity (dF/F) from past activity.
- Compare cell types, e.g. Sst (inhibitory) vs Cux2 (excitatory, upper layers) in visual cortex (VISp).
- Later: link cell types across datasets (Allen recordings ↔ single-cell atlases such as HypoMap).

## Connection to the BRAIN Initiative

The NIH [BRAIN Initiative](https://braininitiative.nih.gov/) aims to catalogue the
brain's cell types and understand how they work together in circuits. This project
uses openly shared data from that effort's ecosystem and asks a small, concrete
version of the same question: does cell type tell us something about circuit activity?

## Data

| Dataset | What it is | Used for |
| --- | --- | --- |
| [Allen Brain Observatory – Visual Coding 2P](https://observatory.brain-map.org/visualcoding) | Two-photon calcium imaging, Cre-line labeled neurons | Activity forecasting |
| [Allen Visual Coding – Neuropixels](https://portal.brain-map.org/explore/circuits/visual-coding-neuropixels) | Spikes with optotagging | Cell-type identity checks |
| [HypoMap](https://cellxgene.cziscience.com/collections/d86517f0-fa7e-4266-b82e-a521350d6d36) (mouse, human) | Hypothalamus snRNA-seq atlas | Cell-type reference |

Raw data goes to `data/` (not tracked by git).

## Progress

| Step | What | Result |
| --- | --- | --- |
| 1 | Load one 2P session (Sst, VISp) and plot dF/F | [step1_result.md](results/step1_result.md) |
| 2 | Copy-last and mean forecasting baselines (Sst, Cux2) | [step2_result.md](results/step2_result.md) |

## Repo layout

```text
notebooks/   step-by-step notebooks
scripts/     download, summary and analysis scripts
results/     short result notes and summary tables
plots/       figures
```

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python allensdk matplotlib ipykernel "setuptools<81" "statsmodels<0.14.5"
```
