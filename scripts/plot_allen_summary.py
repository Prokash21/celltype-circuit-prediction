"""Bar charts from results/allen_metadata/ CSVs (PNG at 300 DPI + PDF)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / "allen_metadata"

BLUE = "#4A7DB5"         # muted steel blue
TERRACOTTA = "#C2693D"   # muted terracotta
INK = "#000000"
GRID = "#E3E3E3"

# One session per genotype (most units), run with scripts/allen_optotag_session.py
SESSIONS = {"Pvalb": 840012044, "Sst": 794812542, "Vip": 751348571}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,  # editable text in Illustrator
})


def style_axes(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))


def save(fig, stem):
    fig.tight_layout()
    fig.savefig(RESULTS / f"{stem}.png", dpi=300)
    fig.savefig(RESULTS / f"{stem}.pdf")
    plt.close(fig)


# --- Chart 1: two-photon neurons per Cre line ---------------------------------
tp = pd.read_csv(RESULTS / "twophoton_neurons_per_cre.csv", index_col=0)
tp = tp.drop(index="TOTAL").sort_values("neurons", ascending=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(tp.index, tp.neurons, color=BLUE, width=0.7)
ax.set_ylabel("Number of neurons")
ax.set_title("Two-photon: neurons per Cre line (Allen Brain Observatory)")
ax.tick_params(axis="x", labelrotation=45)
for label in ax.get_xticklabels():
    label.set_horizontalalignment("right")
    label.set_rotation_mode("anchor")
style_axes(ax)
save(fig, "two_photon_neurons_by_creline")

# --- Chart 2: Neuropixels visual-cortex units vs optotagged -------------------
rows = []
for genotype, session in SESSIONS.items():
    s = pd.read_csv(RESULTS / f"optotag_summary_{session}.csv", index_col=0)
    vis = s.loc["visual cortex (VIS*)"]
    rows.append((genotype, vis.good_units, vis.optotagged, vis.percent))
np_df = pd.DataFrame(rows, columns=["genotype", "total", "optotagged", "percent"])

x = np.arange(len(np_df))
w = 0.38
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.bar(x - w / 2 - 0.01, np_df.total, w, color=BLUE, label="Visual-cortex units")
bars = ax.bar(x + w / 2 + 0.01, np_df.optotagged, w, color=TERRACOTTA, label="Optotagged units")
for bar, pct in zip(bars, np_df.percent):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6, f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10, color=INK)
ax.set_xticks(x, [f"{g}-Cre\n(session {SESSIONS[g]})" for g in np_df.genotype])
ax.set_ylabel("Units")
ax.set_title("Neuropixels: optotagged fraction by genotype (one session each)")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, np_df.total.max() * 1.12)
style_axes(ax)
save(fig, "neuropixels_optotagged_pct")
