"""Step 16 plots, from the CSV files in results/step16/ only.

plots/step16_neuropixels.png  overview: (a) unit counts per label, (b) label vs summary
                              gain next to Allen / Bugeon, (c) label / summary ratio across
                              all datasets, (d) synthetic-calcium result
plots/step16_optotag.png      task 1: tagged vs untagged light responses, criterion, waveforms
plots/step16_gains.png        task 2: per-session gains (label, summary, shuffled), per line
plots/step16_calcium.png      task 3: example traces, per-session label gain spikes vs calcium,
                              ratio per signal version
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results/step16"
PL = ROOT / "plots"
LC = {"Pvalb": "tab:red", "Sst": "tab:orange", "Vip": "tab:purple", "wt": "tab:gray", "untagged": "0.6"}
DSN = {"spk": "spikes", "c6f": "syn. 6f", "c6s": "syn. 6s", "callen": "syn. Allen-like", "callen_slow": "syn. Allen-like\n(6s 1.5 s)"}
CAL = ["spk", "c6f", "c6s", "callen", "callen_slow"]


def read(name):
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def overview():
    lab, G, Rt, comp, cvs = read("unit_labels.csv"), read("gains.csv"), read("label_vs_summary.csv"), read("comparison.csv"), read("calcium_vs_spikes.csv")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10.5))
    # (a) counts per label
    a = ax[0, 0]
    rows = []
    for scope, areas in [("VISp", ["VISp"]), ("all VIS", None)]:
        k = lab if areas is None else lab[lab.area.isin(areas)]
        for line in ["Pvalb", "Sst", "Vip"]:
            g = k[k.line == line]
            rows.append((scope, f"{line} mice", int(g.tagged.sum()), int((~g.tagged).sum())))
        rows.append((scope, "wt mice", 0, int((k.line == "wt").sum())))
        rows.append((scope, "FS (all mice)", int(k.fs.sum()), int((~k.fs).sum())))
    x = np.arange(len(rows))
    t = np.array([r[2] for r in rows]); u = np.array([r[3] for r in rows])
    a.bar(x, u, color="0.75", label="untagged / RS")
    a.bar(x, t, bottom=u, color="tab:red", label="tagged / FS")
    for i, r in enumerate(rows):
        a.text(i, u[i] + t[i], f"{t[i]}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(x, [f"{r[0]}\n{r[1]}" for r in rows], fontsize=7, rotation=45, ha="right")
    a.set_yscale("log"); a.set_ylabel("units (kept)"); a.legend(fontsize=8)
    a.set_title("(a) units per label (numbers = tagged or FS)")
    # (b) label vs summary gain, ridge and MLP
    a = ax[0, 1]
    if comp is not None:
        sel = comp[~comp.dataset.str.startswith("synthetic")]
        lbl = [f"{d}\n{s} {l}" if not d.startswith(("Allen", "Bugeon")) else d for d, s, l in zip(sel.dataset, sel.scope, sel.labelset)]
        x = np.arange(len(sel))
        w = 0.2
        for j, (col, nm, c) in enumerate([("label_ridge", "label, ridge", "tab:blue"), ("summary_ridge", "summary, ridge", "tab:cyan"),
                                          ("label_MLP", "label, MLP", "tab:green"), ("summary_MLP", "summary, MLP", "tab:olive")]):
            a.bar(x + (j - 1.5) * w, sel[col], w, color=c, label=nm)
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(x, lbl, fontsize=7, rotation=45, ha="right")
        a.set_ylabel("gain over no label (score)"); a.legend(fontsize=8)
        a.set_title("(b) label gain vs activity-summary gain")
    # (c) ratio
    a = ax[1, 0]
    if comp is not None:
        x = np.arange(len(comp))
        for j, (fam, c) in enumerate([("ridge", "tab:blue"), ("MLP", "tab:green")]):
            v, lo, hi = comp[f"ratio_{fam}"], comp[f"ratio_{fam}_lo"], comp[f"ratio_{fam}_hi"]
            a.errorbar(x + (j - 0.5) * 0.25, v, yerr=[v - lo, hi - v], fmt="o", color=c, label=fam, capsize=2)
        a.axhline(0, color="k", lw=0.5); a.axhline(1, color="k", lw=0.5, ls=":")
        lbl = [f"{d}\n{s} {l}" if not d.startswith(("Allen", "Bugeon")) else d for d, s, l in zip(comp.dataset, comp.scope, comp.labelset)]
        a.set_xticks(x, lbl, fontsize=6, rotation=55, ha="right")
        a.set_ylim(-1, 2.5); a.set_ylabel("label gain / summary gain (95% CI)"); a.legend(fontsize=8)
        a.set_title("(c) label / summary ratio across datasets")
    # (d) synthetic calcium
    a = ax[1, 1]
    if G is not None:
        g = G[(G.scope == "VISp") & (G.subset == "all units") & (G.line == "all") & (G.gain == "label")]
        for j, (ls, fam, mk) in enumerate([("tag", "ridge", "o"), ("tag", "MLP", "s"), ("fs", "ridge", "^"), ("fs", "MLP", "D")]):
            h = g[(g.labelset == ls) & (g.family == fam)].set_index("ds").reindex(CAL)
            a.errorbar(np.arange(len(CAL)) + (j - 1.5) * 0.12, h.gain_mean, yerr=h.gain_sem, fmt=mk + "-",
                       label=f"{'opto-tag' if ls == 'tag' else 'FS/RS'}, {fam}", capsize=2)
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(range(len(CAL)), [DSN[c] for c in CAL], fontsize=8)
        a.set_ylabel("label gain (mean ± sem over sessions)"); a.legend(fontsize=8)
        a.set_title("(d) same neurons, simulated indicator: label gain (VISp)")
    fig.tight_layout()
    fig.savefig(PL / "step16_neuropixels.png", dpi=130)
    plt.close(fig)


def optotag():
    ps, lab, pu = read("optotag_psth.csv"), read("unit_labels.csv"), read("optotag_psth_tagged_units.csv")
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    for j, line in enumerate(["Pvalb", "Sst", "Vip"]):
        a = ax[0, j]
        for tg, c in [(True, LC[line]), (False, "0.5")]:
            g = ps[(ps.line == line) & (ps.tagged == tg)]
            a.plot(g.t_ms, g.rate_mean, color=c, label=f"{'tagged' if tg else 'untagged'} (n = {g.n_units.iloc[0]})")
            a.fill_between(g.t_ms, g.rate_mean - g.rate_sem, g.rate_mean + g.rate_sem, color=c, alpha=0.3)
        a.axvspan(0, 10, color="tab:blue", alpha=0.08, label="10 ms light pulse")
        for w in [(-10, -2), (1, 9)]:
            a.axvline(w[0], color="k", ls=":", lw=0.7); a.axvline(w[1], color="k", ls=":", lw=0.7)
        a.set_title(f"{line} x Ai32: mean response to 10 ms pulses (VIS)")
        a.set_xlabel("time from pulse onset (ms)"); a.set_ylabel("rate (Hz)"); a.legend(fontsize=7)
    a = ax[1, 0]
    cre = lab[lab.line != "wt"]
    a.scatter(np.log10(cre.ratio.clip(1e-2)), -np.log10(cre.p_wilcoxon.clip(1e-12)), s=3, c=np.where(cre.tagged, "tab:red", "0.6"))
    w = lab[lab.line == "wt"]
    a.scatter(np.log10(w.ratio.clip(1e-2)), -np.log10(w.p_wilcoxon.clip(1e-12)), s=3, c="tab:blue", alpha=0.4, label="wild type")
    a.axvline(np.log10(2), color="k", ls=":"); a.axhline(3, color="k", ls=":")
    a.set_xlabel("log10 evoked / (baseline + 1 Hz)"); a.set_ylabel("-log10 p (Wilcoxon, evoked > baseline)")
    a.set_title("criterion: ratio > 2 and p < 0.001 (red = tagged, blue = wild type)"); a.legend(fontsize=7)
    a = ax[1, 1]
    bins = np.linspace(0.1, 1.2, 45)
    a.hist(lab[~lab.tagged].waveform_duration, bins, color="0.6", density=True, alpha=0.6, label="untagged")
    for line in ["Pvalb", "Sst", "Vip"]:
        g = lab[(lab.line == line) & lab.tagged]
        a.hist(g.waveform_duration, bins, histtype="step", density=True, color=LC[line], lw=2, label=f"{line} tagged ({len(g)})")
    a.axvline(0.4, color="k", ls=":")
    a.set_xlabel("waveform duration (ms)"); a.set_title("waveform class threshold 0.4 ms"); a.legend(fontsize=7)
    a = ax[1, 2]
    if pu is not None and len(pu):
        cols = [c for c in pu.columns if c not in ("unit_id", "session", "line", "area")]
        A = pu[cols].to_numpy()
        A = (A - A[:, :16].mean(1, keepdims=True)) / (A[:, :16].std(1, keepdims=True) + 1)
        o = np.lexsort((-A[:, 22:38].mean(1), pu.line.map({"Pvalb": 0, "Sst": 1, "Vip": 2}).to_numpy()))
        tt = np.array([float(c) for c in cols])
        a.imshow(A[o], aspect="auto", cmap="magma", vmin=0, vmax=np.percentile(A, 99), extent=[tt[0], tt[-1], len(A), 0])
        a.set_xlabel("time from pulse onset (ms)"); a.set_ylabel("tagged units (Pvalb, Sst, Vip)")
        a.set_title("tagged units: rate, baseline-normalised")
    fig.tight_layout()
    fig.savefig(PL / "step16_optotag.png", dpi=130)
    plt.close(fig)


def gains():
    sc = read("session_scores.csv")
    if sc is None:
        return
    s = sc[(sc.ds == "spk") & (sc.subset == "all units")]
    combos = [(sc_, ls) for sc_ in ["VISp", "VIS"] for ls in ["tag", "fs"]]
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    for j, (scope, ls) in enumerate(combos):
        g = s[(s.scope == scope) & (s.labelset == ls)]
        if g.empty:
            continue
        for i, (fam, none, lab_, summ, shuf) in enumerate([("ridge", "r_none", "r_true", "r_kmsel", "r_e"),
                                                          ("MLP", "m_none", "m_true", "m_cont", "m_m3")]):
            a = ax[i, j]
            if g[summ].isna().any() if summ in g else True:
                a.set_visible(False)
                continue
            for k_, (col, nm) in enumerate([(lab_, "label"), (summ, "summary"), (shuf, "shuffled")]):
                d = g[col] - g[none]
                jit = np.random.default_rng(k_).uniform(-0.15, 0.15, len(d))
                a.scatter(k_ + jit, d, s=12, c=[LC.get(l, "k") for l in g.line])
                a.errorbar(k_ + 0.3, d.mean(), yerr=d.std(ddof=1) / np.sqrt(len(d)), fmt="k_", capsize=4)
            a.axhline(0, color="k", lw=0.5)
            a.set_xticks([0, 1, 2], ["label", "summary", "shuffled"])
            a.set_title(f"{fam}: {scope}, {'opto-tag (i)' if ls == 'tag' else 'FS/RS (ii)'} ({len(g)} sessions)", fontsize=9)
            a.set_ylabel("gain over none (per session)")
    h = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in LC.items() if l != "untagged"]
    ax[0, 0].legend(handles=h, fontsize=7, title="session genotype")
    fig.tight_layout()
    fig.savefig(PL / "step16_gains.png", dpi=130)
    plt.close(fig)


def calcium():
    ex, sc, cvs = read("example_traces.csv"), read("session_scores.csv"), read("calcium_vs_spikes.csv")
    if sc is None or not sc.ds.isin(CAL[1:]).any():
        return
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    if ex is not None:
        a = ax[0, 0]
        off = 0
        for line in ex.line.unique():
            for sig, c in [("spk", "k"), ("c6f", "tab:green"), ("c6s", "tab:red")]:
                g = ex[(ex.line == line) & (ex.signal == sig)]
                a.plot(g.t, g.z / 4 + off, color=c, lw=0.7, label=sig if line == ex.line.unique()[0] else None)
            a.text(0, off + 0.6, line, fontsize=8)
            off -= 2.5
        a.set_xlabel("s (test part)"); a.set_yticks([]); a.legend(fontsize=7)
        a.set_title("example tagged VISp units: z spikes vs synthetic calcium (10 Hz)")
    s = sc[(sc.subset == "all units") & (sc.scope == "VISp")]
    for j, (ls, fam, none, lab_) in enumerate([("tag", "ridge", "r_none", "r_true"), ("tag", "MLP", "m_none", "m_true"),
                                               ("fs", "ridge", "r_none", "r_true"), ("fs", "MLP", "m_none", "m_true")]):
        a = ax.flat[1 + j]
        base = s[(s.ds == "spk") & (s.labelset == ls)].set_index("session")
        if base.empty or none not in base or base[none].isna().any():
            a.set_visible(False)
            continue
        for k_, ds in enumerate(CAL):
            c = s[(s.ds == ds) & (s.labelset == ls)].set_index("session")
            if c.empty or c[none].isna().any():
                continue
            d = (c[lab_] - c[none]).reindex(base.index)
            a.scatter(np.full(len(d), k_) + np.random.default_rng(k_).uniform(-0.15, 0.15, len(d)), d, s=10,
                      c=[LC.get(l, "k") for l in base.line])
            a.errorbar(k_ + 0.3, d.mean(), yerr=d.std(ddof=1) / np.sqrt(d.notna().sum()), fmt="k_", capsize=4)
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(range(len(CAL)), [DSN[c] for c in CAL], fontsize=7)
        a.set_title(f"label gain per session, {fam}, {'opto-tag (i)' if ls == 'tag' else 'FS/RS (ii)'} (VISp)", fontsize=9)
    a = ax[1, 2]
    if cvs is not None and len(cvs):
        for j, (ls, fam, mk) in enumerate([("tag", "ridge", "o"), ("tag", "MLP", "s"), ("fs", "ridge", "^"), ("fs", "MLP", "D")]):
            g = cvs[(cvs.scope == "VISp") & (cvs.labelset == ls) & (cvs.family == fam)].set_index("ds").reindex(CAL[1:])
            if g.ratio_spikes.isna().all():
                continue
            y = np.r_[g.ratio_spikes.dropna().iloc[0], g.ratio_calcium]
            a.plot(np.arange(len(CAL)) + (j - 1.5) * 0.08, y, mk + "-", label=f"{'opto-tag' if ls == 'tag' else 'FS/RS'}, {fam}")
        a.axhline(0, color="k", lw=0.5)
        a.set_xticks(range(len(CAL)), [DSN[c] for c in CAL], fontsize=7)
        a.set_ylabel("label gain / summary gain"); a.legend(fontsize=7)
        a.set_title("label / summary ratio per signal (VISp)")
    fig.tight_layout()
    fig.savefig(PL / "step16_calcium.png", dpi=130)
    plt.close(fig)


def main():
    PL.mkdir(exist_ok=True)
    if read("unit_labels.csv") is not None:
        optotag()
    overview()
    gains()
    calcium()
    print("plots written")


if __name__ == "__main__":
    main()
