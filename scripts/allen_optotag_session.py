"""Optotagging laser-response test on one Visual Coding Neuropixels session.

Follows the AllenSDK ecephys_optotagging tutorial: 10 ms light pulses,
binary 0.5 ms spike bins, and a unit is Cre+ if its evoked rate
(1-9 ms after onset) / (baseline rate (-10 to -2 ms) + 1) > 2.

Usage: python scripts/allen_optotag_session.py [session_id]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "allen_cache" / "ecephys"
OUT = ROOT / "results" / "allen_metadata"
OUT.mkdir(parents=True, exist_ok=True)

session_id = int(sys.argv[1]) if len(sys.argv) > 1 else 840012044

cache = EcephysProjectCache.from_warehouse(
    manifest=str(CACHE / "manifest.json"), timeout=60 * 60
)
session = cache.get_session_data(session_id)  # default unit QC filters applied
print(session.full_genotype)

opto = session.optogenetic_stimulation_epochs
trials = opto[(opto.duration > 0.009) & (opto.duration < 0.02)]
print(f"{len(trials)} 10 ms pulses")

bin_edges = np.arange(-0.01, 0.025, 0.0005)
units = session.units  # all regions; VIS* subset (the tutorial's selection) reported below


def optotagging_spike_counts(bin_edges, trials, units):
    # Verbatim from the tutorial: bins are binary (1 if any spike), edges exclusive.
    time_resolution = np.mean(np.diff(bin_edges))

    spike_matrix = np.zeros((len(trials), len(bin_edges), len(units)))

    for unit_idx, unit_id in enumerate(units.index.values):

        spike_times = session.spike_times[unit_id]

        for trial_idx, trial_start in enumerate(trials.start_time.values):

            in_range = (spike_times > (trial_start + bin_edges[0])) * \
                       (spike_times < (trial_start + bin_edges[-1]))

            binned_times = ((spike_times[in_range] - (trial_start + bin_edges[0])) / time_resolution).astype('int')
            spike_matrix[trial_idx, binned_times, unit_idx] = 1

    return xr.DataArray(
        name='spike_counts',
        data=spike_matrix,
        coords={
            'trial_id': trials.index.values,
            'time_relative_to_stimulus_onset': bin_edges,
            'unit_id': units.index.values
        },
        dims=['trial_id', 'time_relative_to_stimulus_onset', 'unit_id']
    )


da = optotagging_spike_counts(bin_edges, trials, units)

t = "time_relative_to_stimulus_onset"
baseline_rate = da.sel({t: slice(-0.01, -0.002)}).sum(dim=t).mean(dim="trial_id") / 0.008
evoked_rate = da.sel({t: slice(0.001, 0.009)}).sum(dim=t).mean(dim="trial_id") / 0.008
ratio = evoked_rate / (baseline_rate + 1)

result = units[["ecephys_structure_acronym"]].copy()
result["baseline_rate"] = baseline_rate.values
result["evoked_rate"] = evoked_rate.values
result["ratio"] = ratio.values
result["optotagged"] = result.ratio > 2
result.to_csv(OUT / f"optotag_units_{session_id}.csv")

in_vis = result.ecephys_structure_acronym.str.match("VIS")
summary = pd.DataFrame({
    "good_units": [len(result), in_vis.sum()],
    "optotagged": [result.optotagged.sum(), result.optotagged[in_vis].sum()],
}, index=["all regions", "visual cortex (VIS*)"])
summary["percent"] = (100 * summary.optotagged / summary.good_units).round(1)
summary.to_csv(OUT / f"optotag_summary_{session_id}.csv")
print(summary)
print("\nOptotagged units by region:")
print(result[result.optotagged].ecephys_structure_acronym.value_counts().to_string())
