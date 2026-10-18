"""Step 13 plots, made only from the CSV files in results/step13/ (written by
`scripts/step13_dynamics.py both analyse`), so they can be remade without rerunning
any model.

plots/step13_dynamics.png  (a) ridge score gain over no label vs k (k-means, GMM, class-balanced
                           k-means, shuffled clusters), Bugeon and Allen, with the true-label gain;
                           (b) gain of true type / clusters / continuous summary / continuous + true
                           (MLP; ridge where it exists); (c) cluster vs type and cluster vs mouse
                           (Bugeon, fold 0, chosen k); (d) gain vs history length of the summary
plots/step13_types.png     gain within each true type (MLP and ridge), Bugeon and Allen
plots/step13_batch.png     adjusted mutual information of clusters with type, mouse and session vs k;
                           Allen cluster vs Cre line and cluster vs mouse (fold 0, chosen k)
plots/step13_valk.png      val score of the training sessions vs k per fold (the k selection)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step13"
PLOTS = ROOT / "plots"
DS = {"bugeon": "Bugeon (17 sessions, 4 mice)", "allen": "Allen (46 mice)"}
UNIT = {"bugeon": "session", "allen": "mouse"}
UNITS = {"bugeon": "sessions", "allen": "mice"}
TYPES = {"bugeon": ["EC", "Lamp5", "Pvalb", "Sncg", "Sst", "Vip"], "allen": ["Cux2", "Sst", "Vip", "Pvalb"]}
M_COL = {"km": "tab:blue", "gmm": "tab:purple", "kmbal": "tab:cyan", "kmshuf": "0.5"}
M_NAME = {"km": "k-means", "gmm": "Gaussian mixture", "kmbal": "k-means, class-balanced rows",
          "kmshuf": "shuffled clusters (k-means sizes)"}
V_COL = {"true": "tab:orange", "S": "tab:brown", "cl": "tab:blue", "clnt": "tab:cyan", "cont": "tab:green",
         "contT": "tab:olive", "clsh": "0.6", "k1": "0.8", "selshuf": "0.6", "sel": "tab:blue", "nt": "tab:cyan",
         "gmmsel": "tab:purple"}


def load():
    return {n: pd.read_csv(RES / f"{n}.csv") for n in ["k_sweep", "gains", "unit_scores", "history", "contingency",
                                                        "chosen_k", "type_gain", "ami", "val_k", "stats"]}


def p_of(st, ds, a, b):
    r = st[(st.dataset == ds) & (st.a == a) & (st.b == b) & (st.test != "mice (signs only)")]
    return r.iloc[0] if len(r) else None


def panel_k(ax, d, ds):
    ks = d["k_sweep"][d["k_sweep"].dataset == ds]
    g = d["gains"][d["gains"].dataset == ds].set_index("variant")
    for m in ["km", "gmm", "kmbal", "kmshuf"]:
        r = ks[ks.method == m].sort_values("k")
        ax.errorbar(r.k, r.gain_mean, yerr=r.gain_sem, color=M_COL[m], marker="o", ms=3.5, lw=1.3, capsize=2,
                    label=M_NAME[m])
    ax.axhline(g.loc["r_true", "gain_mean"], color=V_COL["true"], ls="--", lw=1.2, label="true type (ridge d)")
    ax.axhline(g.loc["r_S", "gain_mean"], color=V_COL["S"], ls=":", lw=1.2, label="S-classifier labels (ridge d)")
    ax.axhline(g.loc["r_k1", "gain_mean"], color="0.3", ls="-.", lw=0.9, label="k = 1, equal row weights")
    for v, mk in [("m_cl", "*"), ("m_clnt", "P")]:
        ck = d["chosen_k"]
        k = (ck[(ck.dataset == ds) & (ck.method == "km")].k.mode().iloc[0] if v == "m_cl" else len(TYPES[ds]))
        ax.plot(k, g.loc[v, "gain_mean"], mk, color="k", ms=9 if mk == "*" else 7, zorder=5,
                label=f"MLP FiLM cluster id (k = {k})" if v == "m_cl" else f"MLP FiLM cluster id (k = n types)")
    ax.axhline(g.loc["m_true", "gain_mean"], color=V_COL["true"], ls="--", lw=1.2, alpha=0.45)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvspan(20, 40, color="0.92", zorder=0)
    ax.text(28, ax.get_ylim()[0], "outside grid\n(exploratory)", ha="center", va="bottom", fontsize=6.5, color="0.4")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 3, 4, 6, 8, 12, 16, 24, 32]); ax.set_xticklabels([2, 3, 4, 6, 8, 12, 16, 24, 32], fontsize=7)
    ax.set_xlabel("number of clusters k (fitted on training mice)", fontsize=8)
    ax.set_ylabel(f"score gain over no label (mean ± sem over {UNITS[ds]})", fontsize=8)
    ck = d["chosen_k"][(d["chosen_k"].dataset == ds) & (d["chosen_k"].method == "km")]
    ax.set_title(f"(a) {DS[ds]}: ridge per-cluster models vs k\nval-chosen k per fold: {sorted(ck.k.tolist())}",
                 fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)


def panel_gains(ax, d, ds):
    u = d["unit_scores"]
    u = u[(u.dataset == ds) & (u.level == UNIT[ds])]
    st = d["stats"]
    bars = [("MLP", "m_true", "true type"), ("MLP", "m_cl", "clusters\n(chosen k)"), ("MLP", "m_clnt", "clusters\n(k = n types)"),
            ("MLP", "m_cont", "continuous\nsummary"), ("MLP", "m_contT", "continuous\n+ true type"),
            ("MLP", "m_clsh", "shuffled\nclusters"),
            ("ridge", "r_true", "true type"), ("ridge", "r_sel", "clusters\n(chosen k)"), ("ridge", "r_selshuf", "shuffled\nclusters")]
    x = 0
    ticks, labels = [], []
    rng = np.random.default_rng(0)
    for fam, v, lab in bars:
        none = "m_none" if fam == "MLP" else "r_none"
        g = (u[v] - u[none]).to_numpy()
        key = v.split("_", 1)[1]
        ax.bar(x, g.mean(), yerr=g.std(ddof=1) / np.sqrt(len(g)), color=V_COL.get(key, "0.5"), capsize=2,
               hatch="//" if fam == "ridge" else None, edgecolor="k", lw=0.4)
        ax.scatter(x + rng.uniform(-0.22, 0.22, len(g)), g, s=4, color="k", alpha=0.45, zorder=3, lw=0)
        r = p_of(st, ds, v, none)
        if r is not None:
            ax.text(x, max(g.max(), 0) + 0.0004, f"{int(r.a_beats_b)}/{int(r.n)}\np {r.p:.1g}", ha="center",
                    va="bottom", fontsize=5.5)
        ticks.append(x); labels.append(lab)
        x += 1 if fam == bars[min(len(bars) - 1, bars.index((fam, v, lab)) + 1)][0] else 1.6
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(ticks); ax.set_xticklabels([l.replace("\n", " ") for l in labels], fontsize=6.5, rotation=90)
    ax.set_ylabel(f"score gain over no label (per {UNIT[ds]})", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    xm = np.mean(ticks[:6]); xr = np.mean(ticks[6:])
    yl = ax.get_ylim()
    ax.set_ylim(yl[0], yl[1] + 0.12 * (yl[1] - yl[0]))
    ax.text(xm, ax.get_ylim()[1], "MLP (FiLM)", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(xr, ax.get_ylim()[1], "ridge (hatched)", ha="center", va="top", fontsize=7.5, weight="bold")
    ax.set_title(f"(b) {DS[ds]}: gain over no label\n(counts = {UNITS[ds]} up; Wilcoxon p)", fontsize=8.5)


def heat(ax, M, rows, cols, title, xlabel, fontsize=6, annotate=True, cmap="Blues"):
    ax.imshow(M, cmap=cmap, aspect="auto", vmin=0, vmax=max(M.max(), 1e-9))
    if annotate:
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i, j] >= 0.05:
                    ax.text(j, i, f"{M[i, j]:.2f}"[1:] if M[i, j] < 1 else "1", ha="center", va="center",
                            fontsize=fontsize - 1, color="white" if M[i, j] > 0.6 * M.max() else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=fontsize, rotation=90)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=7); ax.set_ylabel("cluster", fontsize=7)
    ax.set_title(title, fontsize=8)


def contingency(d, ds, by, fold=0):
    c = d["contingency"]
    ck = d["chosen_k"]
    k = int(ck[(ck.dataset == ds) & (ck.method == "km") & (ck.fold == fold)].k.iloc[0])
    c = c[(c.dataset == ds) & (c.fold == fold) & (c.k == k) & (c.by == by)]
    M = c.pivot_table(index="cluster", columns="group", values="count", aggfunc="sum", fill_value=0)
    return M, k, c


def panel_contingency(axs, d, ds="bugeon"):
    Mt, k, _ = contingency(d, ds, "type")
    Mt = Mt[TYPES[ds]]
    order = np.argsort(-(Mt["EC"] / Mt.sum(1)).to_numpy()) if ds == "bugeon" else np.arange(len(Mt))
    Mt = Mt.iloc[order]
    frac_type = (Mt / Mt.sum(0)).to_numpy()  # share of each type's neurons in each cluster
    heat(axs[0], frac_type, [f"{c} (n {n})" for c, n in zip(Mt.index, Mt.sum(1))], list(Mt.columns),
         f"(c) {ds.capitalize()}, fold 0, k = {k}: share of each\ntype's neurons in each cluster", "true type")
    Mm, _, c = contingency(d, ds, "mouse")
    Mm = Mm.loc[Mt.index]
    held = c.groupby("group").held_out.first()
    frac_mouse = (Mm / Mm.sum(0)).to_numpy()
    heat(axs[1], frac_mouse, [str(i) for i in Mm.index], [f"{m}{' (held out)' if held[m] else ''}" for m in Mm.columns],
         "share of each mouse's neurons\nin each cluster (rows as left)", "mouse")


def panel_history(ax, d, ds):
    h = d["history"][d["history"].dataset == ds]
    g = d["gains"][d["gains"].dataset == ds].set_index("variant")
    for v, col, lab in [("r_sel", "tab:blue", "ridge, clusters (chosen k)"), ("m_cl", "navy", "MLP, cluster id"),
                        ("m_cont", "tab:green", "MLP, continuous summary")]:
        r = h[h.variant == v].sort_values("minutes")
        ax.errorbar(r.minutes, r.gain_mean, yerr=r.gain_sem, marker="o", ms=4, color=col, capsize=2, lw=1.3,
                    ls="--" if v.startswith("r_") else "-", label=lab)
    ax.axhline(g.loc["m_true", "gain_mean"], color=V_COL["true"], lw=1.1, label="MLP, true type")
    ax.axhline(g.loc["r_true", "gain_mean"], color=V_COL["true"], lw=1.1, ls="--", label="ridge, true type")
    ax.axhline(0, color="k", lw=0.6)
    # the continuous summary from 3 min is far off scale: clip the axis and print the value
    lo = min(-0.01, h[h.variant != "m_cont"].gain_mean.min() - 0.01)
    hi = max(h.gain_mean.max(), g.loc["m_true", "gain_mean"]) * 1.25
    ax.set_ylim(lo, hi)
    for _, r in h[h.gain_mean < lo].iterrows():
        ax.annotate(f"{r.history}: {r.gain_mean:+.3f}\n(off scale)", (r.minutes, lo), xytext=(r.minutes * 1.4, lo + 0.25 * (hi - lo)),
                    fontsize=6.5, color="tab:green", arrowprops=dict(arrowstyle="->", color="tab:green", lw=0.8))
    ax.set_xscale("log")
    xs = sorted(h.minutes.unique())
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:.0f}" for x in xs], fontsize=7)
    ax.set_xlabel("minutes of activity behind the summary (held-out neurons)", fontsize=7.5)
    ax.set_ylabel(f"score gain over no label (mean ± sem over {UNITS[ds]})", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"(d) {DS[ds]}: gain vs history length\n(models trained with full-history summaries)", fontsize=8.5)
    ax.legend(fontsize=6)


def fig_main(d):
    fig = plt.figure(figsize=(21, 12))
    gs = fig.add_gridspec(2, 4, hspace=0.75, wspace=0.32)
    panel_k(fig.add_subplot(gs[0, 0]), d, "bugeon")
    panel_k(fig.add_subplot(gs[0, 1]), d, "allen")
    panel_gains(fig.add_subplot(gs[0, 2]), d, "bugeon")
    panel_gains(fig.add_subplot(gs[0, 3]), d, "allen")
    gc = gs[1, :2].subgridspec(1, 2, wspace=0.45)
    panel_contingency([fig.add_subplot(gc[0, 0]), fig.add_subplot(gc[0, 1])], d)
    panel_history(fig.add_subplot(gs[1, 2]), d, "bugeon")
    panel_history(fig.add_subplot(gs[1, 3]), d, "allen")
    fig.suptitle("Step 13: label-free dynamics summary (S features of the train part) vs genetic cell type for "
                 "forecasting; score = 1 - MSE / MSE copy-last on held-out mice", fontsize=11)
    fig.savefig(PLOTS / "step13_dynamics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_types(d):
    tg = d["type_gain"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for col, ds in enumerate(["bugeon", "allen"]):
        for row, (fam, vs) in enumerate([("MLP", [("m_true", "m_none", "true type"), ("m_cl", "m_none", "clusters"),
                                                  ("m_cont", "m_none", "continuous"),
                                                  ("m_contT", "m_none", "continuous + true"),
                                                  ("m_clsh", "m_none", "shuffled clusters")]),
                                         ("ridge", [("r_true", "r_none", "true type"), ("r_sel", "r_none", "clusters"),
                                                    ("r_S", "r_none", "S-classifier labels"),
                                                    ("r_selshuf", "r_none", "shuffled clusters")])]):
            ax = axes[row, col]
            types = TYPES[ds]
            w = 0.8 / len(vs)
            for j, (a, b, lab) in enumerate(vs):
                r = tg[(tg.dataset == ds) & (tg.a == a) & (tg.b == b)].set_index("type").loc[types]
                key = a.split("_", 1)[1]
                ax.bar(np.arange(len(types)) + (j - (len(vs) - 1) / 2) * w, r.pooled_gain, w,
                       color=V_COL.get(key, "0.5"), edgecolor="k", lw=0.3, label=lab)
                for i, t in enumerate(types):
                    ax.text(i + (j - (len(vs) - 1) / 2) * w, max(r.pooled_gain.iloc[i], 0) + 0.0003,
                            f"{int(r.a_beats_b.iloc[i])}/{int(r.n.iloc[i])}", ha="center", va="bottom", fontsize=5,
                            rotation=90)
            n = tg[(tg.dataset == ds) & (tg.a == vs[0][0])].set_index("type").n_neurons.loc[types]
            ax.set_xticks(range(len(types))); ax.set_xticklabels([f"{t}\n(n {v})" for t, v in zip(types, n)], fontsize=7)
            ax.axhline(0, color="k", lw=0.6)
            ax.set_ylabel("score gain over no label\n(pooled over the type's neurons)", fontsize=7.5)
            ax.set_title(f"{DS[ds]}, {fam}: gain within each true type (counts = {UNITS[ds]} up)", fontsize=8.5)
            ax.legend(fontsize=6.5, loc="best")
            ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(PLOTS / "step13_types.png", dpi=150)
    plt.close(fig)


def fig_batch(d):
    a = d["ami"]
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)
    for i, ds in enumerate(["bugeon", "allen"]):
        ax = fig.add_subplot(gs[0, i])
        r = a[(a.dataset == ds) & (a.method == "km")]
        for part, ls in [("train", "-"), ("held-out", "--")]:
            g = r[r.part == part].groupby("k")[["ami_type", "ami_session", "ami_mouse"]].mean()
            ax.plot(g.index, g.ami_type, ls, marker="o", ms=3, color="tab:orange", label=f"vs type, {part}")
            ax.plot(g.index, g.ami_session, ls, marker="s", ms=3, color="tab:green", label=f"vs session, {part}")
            if part == "train" and ds == "bugeon":
                ax.plot(g.index, g.ami_mouse, ls, marker="^", ms=3, color="tab:red", label="vs mouse, train")
        ref = r[r.part == "train"].ref_ami_type_mouse.mean() if ds == "allen" else np.nan
        if ds == "allen":
            ax.axhline(ref, color="tab:red", ls=":", lw=1, label="AMI(Cre line, mouse) = clusters = lines")
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
        ax.set_xlabel("k (k-means)", fontsize=8); ax.set_ylabel("adjusted mutual information (mean over folds)", fontsize=8)
        ax.set_title(f"{DS[ds]}: clusters vs type / session / mouse" + ("\n(Allen: session = mouse)" if ds == "allen" else ""),
                     fontsize=9)
        ax.legend(fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2])
    r = a[(a.method == "km") & (a.part == "train")].groupby(["dataset", "k"])[["top_mouse_share",
                                                                               "top_mouse_share_if_random"]].mean()
    for ds, col in [("bugeon", "tab:blue"), ("allen", "tab:red")]:
        g = r.loc[ds]
        ax.plot(g.index, g.top_mouse_share, marker="o", ms=3, color=col, label=f"{ds}: largest mouse share per cluster")
        ax.plot(g.index, g.top_mouse_share_if_random, ls=":", color=col, label=f"{ds}: if clusters ignored mice")
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
    ax.set_xlabel("k (k-means)", fontsize=8); ax.set_ylabel("share of a cluster's neurons from its largest mouse", fontsize=8)
    ax.set_title("Do clusters collect one mouse? (training neurons)", fontsize=9)
    ax.legend(fontsize=6.5)
    gc = gs[1, :].subgridspec(1, 3, wspace=0.35, width_ratios=[1, 3, 1])
    Mt, k, _ = contingency(d, "allen", "type")
    Mt = Mt[TYPES["allen"]]
    heat(fig.add_subplot(gc[0, 0]), (Mt / Mt.sum(0)).to_numpy(), [f"{c} (n {n})" for c, n in zip(Mt.index, Mt.sum(1))],
         list(Mt.columns), f"Allen, fold 0, k = {k}:\nshare of each line's neurons per cluster", "Cre line")
    Mm, _, c = contingency(d, "allen", "mouse")
    held = c.groupby("group").held_out.first()
    # mice ordered by Cre line, then id (line of each mouse from neuron_scores.csv)
    ns = pd.read_csv(RES / "neuron_scores.csv", usecols=["dataset", "session", "type"])
    line = ns[ns.dataset == "allen"].drop_duplicates("session").set_index("session").type
    cols = sorted(Mm.columns, key=lambda m: (TYPES["allen"].index(line[m]), m))
    Mm = Mm.loc[Mt.index, cols]
    heat(fig.add_subplot(gc[0, 1]), (Mm / Mm.sum(0)).to_numpy(), list(Mt.index.astype(str)),
         [f"{line[m]} {m}{'*' if held[m] else ''}" for m in Mm.columns], "share of each mouse's neurons per cluster (* = held out "
         "in fold 0)", "mouse (experiment id)", fontsize=5, annotate=False)
    ax = fig.add_subplot(gc[0, 2])
    r = a[(a.method == "km") & (a.part == "train")].groupby(["dataset", "k"]).top_type_share.mean()
    for ds, col in [("bugeon", "tab:blue"), ("allen", "tab:red")]:
        ax.plot(r.loc[ds].index, r.loc[ds].values, marker="o", ms=3, color=col, label=ds)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32]); ax.set_xticklabels([2, 4, 8, 16, 32])
    ax.set_xlabel("k", fontsize=8); ax.set_ylabel("type purity (largest type share)", fontsize=8)
    ax.set_title("cluster type purity (training neurons)", fontsize=9)
    ax.legend(fontsize=7)
    fig.savefig(PLOTS / "step13_batch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_valk(d):
    v = d["val_k"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, ds in zip(axes, ["bugeon", "allen"]):
        r = v[v.dataset == ds]
        for f in sorted(r.fold.unique()):
            for m, ls in [("km", "-"), ("gmm", ":")]:
                q = r[(r.fold == f) & (r.method == m)].sort_values("k")
                base = r[(r.fold == f) & (r.method == "b (none)")].val_score.iloc[0]
                ax.plot(q.k, q.val_score - base, ls, marker="o", ms=2.5, color=plt.cm.tab10(f),
                        label=f"fold {f} {m}" if m == "km" or f == 0 else None)
            k1 = r[(r.fold == f) & (r.method == "k=1 (one model, equal weights)")].val_score.iloc[0] - base
            ax.plot(1, k1, "x", color=plt.cm.tab10(f))
        ax.axvspan(20, 40, color="0.92", zorder=0)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32]); ax.set_xticklabels([1, 2, 4, 8, 16, 32])
        ax.set_xlabel("k (x at 1 = one model, equal row weights)", fontsize=8)
        ax.set_ylabel("val score gain over b (training sessions, fit-part models)", fontsize=8)
        ax.set_title(f"{DS[ds]}: k selection on val windows of the training mice\n(solid k-means, dotted GMM; grey = "
                     "outside grid, never chosen)", fontsize=8.5)
        ax.legend(fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(PLOTS / "step13_valk.png", dpi=150)
    plt.close(fig)


def main():
    d = load()
    fig_main(d)
    fig_types(d)
    fig_batch(d)
    fig_valk(d)
    print("Saved plots/step13_dynamics.png, step13_types.png, step13_batch.png, step13_valk.png")


if __name__ == "__main__":
    main()
