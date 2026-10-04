"""Summarise Allen Visual Coding metadata (no NWB / movies / spikes downloaded).

Neuropixels: Cre-line sessions and QC-passing units per genotype/session.
Two-photon:  experiments and segmented neurons (cell specimens) per Cre line.

Caches go to data/allen_cache (gitignored); tables go to results/allen_metadata.
"""
from pathlib import Path

import pandas as pd
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache
from allensdk.core.brain_observatory_cache import BrainObservatoryCache

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "allen_cache"
OUT = ROOT / "results" / "allen_metadata"
OUT.mkdir(parents=True, exist_ok=True)

CRE_LINES = ["Pvalb", "Sst", "Vip"]

# --- Neuropixels -------------------------------------------------------------
ecephys = EcephysProjectCache.from_warehouse(manifest=str(CACHE / "ecephys" / "manifest.json"))
sessions = ecephys.get_session_table()

sessions["cre_line"] = None
for cre in CRE_LINES:
    sessions.loc[sessions.full_genotype.str.startswith(cre), "cre_line"] = f"{cre}-Cre"
cre_sessions = sessions.dropna(subset=["cre_line"])

# unit_count = units passing the default QC filters (isi, amplitude cutoff, presence ratio)
np_sessions = (
    cre_sessions[["cre_line", "full_genotype", "session_type", "unit_count"]]
    .rename(columns={"unit_count": "qc_units"})
    .sort_values(["cre_line", "session_type"])
)
# Optotagged counts need per-unit spike times around the laser pulses (NWB files);
# they cannot be derived from metadata, so the column is left empty here.
np_sessions["optotagged_units"] = pd.NA
np_sessions.index.name = "ecephys_session_id"
np_sessions.to_csv(OUT / "neuropixels_cre_sessions.csv")

np_by_line = np_sessions.groupby("cre_line").agg(
    sessions=("session_type", "size"),
    qc_units=("qc_units", "sum"),
)
np_by_line.loc["TOTAL"] = np_by_line.sum()
np_by_line["optotagged_units"] = pd.NA

# --- Two-photon --------------------------------------------------------------
boc = BrainObservatoryCache(manifest_file=str(CACHE / "ophys" / "manifest.json"))
experiments = pd.DataFrame(boc.get_ophys_experiments())
containers = pd.DataFrame(boc.get_experiment_containers())
cells = pd.DataFrame(boc.get_cell_specimens())

# Each cell_specimen_id is a unique neuron tracked across a container's sessions.
cells = cells.merge(
    containers[["id", "cre_line"]], left_on="experiment_container_id", right_on="id"
)
tp_by_line = pd.DataFrame({
    "containers": containers.groupby("cre_line").size(),
    "experiments": experiments.groupby("cre_line").size(),
    "neurons": cells.groupby("cre_line").cell_specimen_id.nunique(),
}).fillna(0).astype(int).sort_values("neurons", ascending=False)
tp_by_line.loc["TOTAL"] = tp_by_line.sum()
tp_by_line.to_csv(OUT / "twophoton_neurons_per_cre.csv")
np_by_line.to_csv(OUT / "neuropixels_units_per_cre.csv")

pd.set_option("display.width", 200)
print("Neuropixels Cre sessions\n", np_sessions.drop(columns="full_genotype").to_string(), "\n")
print("Neuropixels per Cre line\n", np_by_line, "\n")
print("Two-photon per Cre line\n", tp_by_line)
