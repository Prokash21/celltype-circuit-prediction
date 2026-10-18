"""Step 14 plots, remade from the CSV files in results/step14/ only.

plots/step14_atlas.png  main figure: gain over no label vs history length H for
                        every method (ridge and MLP families, Bugeon and Allen), and
                        recogniser accuracy vs window length
plots/step14_k4.png     task 3: session-centred and session-only summaries, and the
                        gain by behavioural state of the test window
plots/step14_k3.png     task 4 (Bugeon): summary from another session of the same cell
plots/step14_pairs.png  per-unit differences of the key comparisons at each H
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step14"
PLOTS = ROOT / "plots"
DS = {"bugeon": "Bugeon (17 sessions, 4 mice)", "allen": "Allen (46 mice)"}
HIST = ["3 min", "10 min", "30 min", "full"]
STYLE = {
    "true type": dict(color="black", ls="--", marker=""),
    "naive short summary": dict(color="tab:gray", ls="-", marker="o"),
    "naive short cluster": dict(color="silver", ls="-", marker="o"),
    "per-neuron ridge": dict(color="tab:red", ls="-", marker="s"),
    "per-neuron MLP fine-tune": dict(color="tab:red", ls="-", marker="s"),
    "E2 hard": dict(color="tab:cyan", ls="-", marker="^"),
    "E2 soft": dict(color="tab:blue", ls="-", marker="D"),
    "E2 continuous": dict(color="tab:purple", ls="-", marker="P"),
    "full-history summary": dict(color="tab:green", ls=":", marker=""),
    "full-history cluster": dict(color="yellowgreen", ls=":", marker=""),
}


def gain_panel(ax, g, ds, fam):
    d = g[(g.dataset == ds) & (g.family == fam)]
    xs = {h: d[d.history == h].minutes.iloc[0] for h in HIST}
    # the axis is set by the shared-model methods; very negative per-neuron / naive values are printed
    inside = d[~d.method.isin(["per-neuron ridge", "per-neuron MLP fine-tune", "naive short summary",
                               "E2 continuous"])]
    hi = d.gain_mean.max() * 1.15 + 1e-4
    lo = min(inside.gain_mean.min(), 0) - 0.25 * (hi - min(inside.gain_mean.min(), 0))
    n_off = {}
    for m, st in STYLE.items():
        e = d[d.method == m].set_index("history").reindex(HIST)
        if e.gain_mean.isna().all():
            continue
        x = np.array([xs[h] for h in HIST])
        y, s = e.gain_mean.to_numpy(), e.gain_sem.to_numpy()
        yc = np.clip(y, lo, hi)
        ax.errorbar(x, yc, yerr=np.where(y == yc, s, 0), label=m, capsize=2, lw=1.4, ms=5, **st)
        for xi, yi, yci in zip(x, y, yc):
            if yi != yci:
                k_ = n_off.get(xi, 0)
                n_off[xi] = k_ + 1
                ax.annotate(f"{yi:+.3f}", (xi, yci), xytext=(4, 4 + 9 * k_), textcoords="offset points",
                            fontsize=7, color=st["color"])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([xs[h] for h in HIST])
    ax.set_xticklabels([f"{h}\n({xs[h]:.0f} min)" if h == "full" else h for h in HIST], fontsize=8)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("history H of the held-out neuron")
    ax.set_ylabel("score gain over no label")
    ax.set_title(f"{DS[ds]}: {fam}", fontsize=10)
    ax.legend(fontsize=6.5, loc="best", ncol=2)


def recog_panel(ax, r, ds):
    d = r[(r.dataset == ds)]
    sw = d[d.kind == "single window"]
    for tgt, m_, st, lab in [("cluster", "acc", dict(color="tab:blue", marker="o"), "atlas cluster (16): accuracy"),
                             ("cluster", "bacc", dict(color="tab:blue", marker="o", ls="--"),
                              "atlas cluster: balanced acc."),
                             ("type", "bacc", dict(color="tab:orange", marker="s", ls="--"),
                              "genetic type: balanced acc.")]:
        e = sw[sw.target == tgt]
        ax.plot(e.minutes, e[m_], label=lab + " (one window)", **st)
    acc = d[(d.kind == "accumulated") & (d.window == "selected per fold")]
    hm = pd.read_csv(RES / "history_minutes.csv")
    hm = hm[hm.dataset == ds].set_index("history").minutes_mean
    for tgt, m_, col, lab in [("cluster", "acc", "tab:blue", "atlas cluster: accuracy"),
                              ("type", "bacc", "tab:orange", "genetic type: balanced acc.")]:
        e = acc[acc.target == tgt].set_index("history").reindex(HIST)
        ax.plot([hm[h] for h in HIST], e[m_], color=col, marker="*", ms=10, lw=0.8, ls=":",
                label=lab + " (accumulated over H)")
    nv = d[d.kind == "naive short summary (nearest centroid)"].set_index("history")
    ax.plot([hm[h] for h in HIST[:3]], [nv.acc[h] for h in HIST[:3]], color="tab:gray", marker="x", ms=8, lw=0.8,
            ls=":", label="atlas cluster: accuracy of the naive short summary (step 13)")
    ch = sw[sw.target == "cluster"].chance_acc.iloc[0]
    ax.axhline(1 / 16, color="tab:blue", lw=0.6, ls=":")
    ax.axhline(ch, color="tab:blue", lw=0.6, ls="-.")
    ax.axhline(sw[sw.target == "type"].chance_bacc.iloc[0], color="tab:orange", lw=0.6, ls=":")
    ax.text(0.45, ch, " largest cluster share", fontsize=6, va="bottom", color="tab:blue")
    ax.text(0.45, 1 / 16, " 1/16", fontsize=6, va="bottom", color="tab:blue")
    ax.text(5, sw[sw.target == "type"].chance_bacc.iloc[0], " type chance (bal. acc.)", fontsize=6, va="bottom",
            color="tab:orange")
    ax.set_xscale("log")
    ax.set_xlabel("window length / history (min)")
    ax.set_ylabel("held-out mice")
    ax.set_ylim(0, 1)
    ax.set_title(f"{DS[ds]}: recogniser", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left")


def main_figure():
    g = pd.read_csv(RES / "gains_by_history.csv")
    r = pd.read_csv(RES / "recogniser.csv")
    dss = [d for d in DS if d in set(g.dataset)]
    fig, axs = plt.subplots(len(dss), 3, figsize=(17, 5.2 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        gain_panel(axs[i, 0], g, ds, "ridge")
        gain_panel(axs[i, 1], g, ds, "MLP")
        recog_panel(axs[i, 2], r, ds)
    fig.suptitle("Step 14: atlas + recogniser (E2) vs per-neuron models vs naive short summary, by history length "
                 "(mean ± sem over units; off-scale values printed)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "step14_atlas.png", dpi=130)
    plt.close(fig)


def k4_figure():
    k4 = pd.read_csv(RES / "k4_session.csv")
    k4s = pd.read_csv(RES / "k4_state.csv")
    u = pd.read_csv(RES / "unit_scores.csv", low_memory=False)
    dss = [d for d in DS if d in set(k4.dataset)]
    fig, axs = plt.subplots(len(dss), 2, figsize=(13, 4.6 * len(dss)), squeeze=False)
    for i, ds in enumerate(dss):
        ax = axs[i, 0]
        lvl = "session" if ds == "bugeon" else "mouse"
        uu = u[(u.dataset == ds) & (u.level == lvl) & (u.state == "all")]
        x = 0
        ticks = []
        for fam, none in [("ridge", "r_none"), ("MLP", "m_none")]:
            d = k4[(k4.dataset == ds) & (k4.family == fam)]
            for _, r in d.iterrows():
                gu = uu[r.variant] - uu[none]
                ax.bar(x, r.gain_mean, color={"full summary": "tab:green", "(a) centred within session": "tab:blue",
                                              "(b) session mean only": "tab:gray"}[r.summary], alpha=0.7)
                ax.scatter(x + np.random.default_rng(int(10 * x)).uniform(-0.25, 0.25, len(gu)), gu, s=6, color="k", zorder=3)
                ax.text(x, 0.98, f"{r.frac_of_full_gain:.0%}\n{r.units_up}/{r.n_units}", ha="center", va="top",
                        fontsize=7, transform=ax.get_xaxis_transform())
                ticks.append((x, f"{fam}\n{r.summary.replace(' within session', '').replace(' only', '')}"))
                x += 1
            x += 0.6
        ax.axhline(0, color="k", lw=0.6)
        lo_, hi_ = ax.get_ylim()
        ax.set_ylim(lo_, hi_ + 0.18 * (hi_ - lo_))
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([l for _, l in ticks], fontsize=7)
        ax.set_ylabel("score gain over no label")
        ax.set_title(f"{DS[ds]}: summary with / without session information\n(% of the full-summary gain; units up)",
                     fontsize=10)
        ax = axs[i, 1]
        d = k4s[(k4s.dataset == ds) & (k4s.state.isin(["running", "stationary"]))]
        labels = list(dict.fromkeys(zip(d.family, d.summary)))
        w = 0.38
        for j, (st, col) in enumerate([("running", "tab:red"), ("stationary", "tab:blue")]):
            e = d[d.state == st].set_index(["family", "summary"]).reindex(labels)
            ax.bar(np.arange(len(labels)) + (j - 0.5) * w, e.gain_mean, w, yerr=e.gain_sem, color=col, alpha=0.7,
                   label=f"{st} test windows", capsize=2)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_xticklabels([f"{f}\n{s}" for f, s in labels], fontsize=7)
        ax.set_ylabel("score gain over no label (sessions)")
        ax.set_title(f"{DS[ds]}: gain by behavioural state of the test window", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "step14_k4.png", dpi=130)
    plt.close(fig)


def k3_figure():
    u = pd.read_csv(RES / "unit_scores.csv", low_memory=False)
    uu = u[(u.dataset == "bugeon") & (u.level == "session (task 4 cells)")]
    if uu.empty:
        return
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (fam, none, same, cross, true) in zip(axs, [("ridge", "r_none", "r_full", "r_cross", "r_true"),
                                                        ("MLP", "m_none", "m_contfull", "m_contcross", "m_true")]):
        names = ["true type", "same-session summary", "other-session summary"]
        for j, v in enumerate([true, same, cross]):
            g = uu[v] - uu[none]
            ax.scatter(np.full(len(g), j) + np.random.default_rng(j).uniform(-0.15, 0.15, len(g)), g, s=12, alpha=0.7)
            ax.errorbar(j + 0.3, g.mean(), yerr=g.std(ddof=1) / np.sqrt(len(g)), fmt="D", color="k", capsize=3)
            ax.text(j + 0.3, g.mean(), f" {g.mean():+.4f}\n {(g > 0).sum()}/{len(g)}", fontsize=7, va="center")
        for _, r in uu.iterrows():
            ax.plot([1, 2], [r[same] - r[none], r[cross] - r[none]], color="gray", lw=0.4, alpha=0.5)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(3))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel("score gain over no label")
        ax.set_title(f"Bugeon task 4 cells, {fam} (one dot per session)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step14_k3.png", dpi=130)
    plt.close(fig)


def pairs_figure():
    st = pd.read_csv(RES / "stats.csv")
    u = pd.read_csv(RES / "unit_scores.csv", low_memory=False)
    comps = [("ridge", "E2 soft vs per-neuron ridge"), ("ridge", "E2 soft vs naive short summary"),
             ("ridge", "E2 soft vs true type"), ("MLP", "E2 soft vs per-neuron MLP fine-tune"),
             ("MLP", "E2 continuous vs per-neuron MLP fine-tune"), ("MLP", "E2 continuous vs naive short summary"),
             ("MLP", "E2 continuous vs true type"), ("MLP", "per-neuron MLP fine-tune vs full-history summary")]
    dss = [d for d in DS if d in set(st.dataset)]
    fig, axs = plt.subplots(len(dss), len(comps), figsize=(2.6 * len(comps), 3.6 * len(dss)), squeeze=False,
                            sharey=False)
    for i, ds in enumerate(dss):
        lvl = "session" if ds == "bugeon" else "mouse"
        uu = u[(u.dataset == ds) & (u.level == lvl) & (u.state == "all")]
        for j, (fam, comp) in enumerate(comps):
            ax = axs[i, j]
            d = st[(st.dataset == ds) & (st.family == fam) & (st.comparison == comp)].set_index("history").reindex(HIST)
            for x, h in enumerate(HIST):
                r = d.loc[h]
                diff = uu[r.a] - uu[r.b]
                ax.scatter(np.full(len(diff), x) + np.random.default_rng(x).uniform(-0.2, 0.2, len(diff)), diff, s=5,
                           alpha=0.6)
                ax.text(x, np.nanmax(diff) if len(diff) else 0, f"{r.a_beats_b}/{r.n}\np={r.p:.1g}", fontsize=6,
                        ha="center", va="bottom")
            ax.axhline(0, color="k", lw=0.6)
            ax.set_xticks(range(len(HIST)))
            ax.set_xticklabels(HIST, fontsize=7)
            ax.set_title(f"{ds} {fam}\n{comp.replace(' vs ', ' − ')}", fontsize=7.5)
            if j == 0:
                ax.set_ylabel("score difference per unit")
    fig.tight_layout()
    fig.savefig(PLOTS / "step14_pairs.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    PLOTS.mkdir(exist_ok=True)
    main_figure()
    k4_figure()
    k3_figure()
    pairs_figure()
    print("plots/step14_atlas.png, step14_k4.png, step14_k3.png, step14_pairs.png")
