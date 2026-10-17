"""Step 11 plots, made only from the CSV files in results/step11/ (written by
scripts/step11_confound.py), so they can be remade without rerunning any model.

plots/step11_confound.png    (a) balanced accuracy of S, R, A, b, S+b (and A+b), (b) confusion matrix of S,
                             (c) fraction of the true-label forecasting gain recovered per label set,
                             (d) S classifier accuracy vs window length
plots/step11_confusions.png  neuron- and session-level confusion matrices of every feature set
plots/step11_sessions.png    recording-level summaries per line and the session-only classifier
plots/step11_features.png    held-out permutation importance of the S, R and A features
plots/step11_mice.png        per mouse: share of neurons the S classifier gets right, and forecasting gain
                             with S labels vs true labels (ridge, MLP)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step11"
PLOTS = ROOT / "plots"
LINES = ["Cux2", "Sst", "Vip", "Pvalb"]
COLORS = {"Cux2": "tab:blue", "Sst": "tab:orange", "Vip": "tab:green", "Pvalb": "tab:red"}
FS_COLORS = {"S": "tab:purple", "R": "0.45", "A": "tab:brown", "b": "tab:cyan", "S+b": "tab:pink", "A+b": "0.75"}
FS_NAME = {"S": "S\nsingle-neuron", "R": "R\nrecording", "A": "A\nall 42", "b": "b\nm1 emb.", "S+b": "S+b",
           "A+b": "A+b\n(step 10)"}
LABEL_SETS = ["S neuron", "S session", "S+b neuron", "S+b session", "R neuron", "R session", "A neuron",
              "A session", "step 10 (a+b) neuron", "shuffled"]


def confusion_from(conf, fs, level, clf="sel"):
    c = conf[(conf.featset == fs) & (conf.level == level) & (conf.clf == clf)]
    return c.pivot(index="true", columns="pred", values="count").loc[LINES, LINES].to_numpy()


def draw_confusion(ax, C, title, fontsize=8):
    R = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{C[i, j]}\n{R[i, j]:.2f}", ha="center", va="center", fontsize=fontsize - 1,
                    color="white" if R[i, j] > 0.6 else "black")
    ax.set_xticks(range(4)); ax.set_xticklabels(LINES, fontsize=fontsize)
    ax.set_yticks(range(4)); ax.set_yticklabels(LINES, fontsize=fontsize)
    ax.set_xlabel("predicted", fontsize=fontsize); ax.set_ylabel("true Cre line", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)


def panel_accuracy(ax, summ):
    sel = summ[summ.clf == "sel"].set_index("featset")
    fsets = [f for f in FS_NAME if f in sel.index]
    x = np.arange(len(fsets))
    ax.bar(x, sel.loc[fsets, "bacc_neuron"], color=[FS_COLORS[f] for f in fsets], alpha=0.8,
           label="neurons, pooled (selected clf)")
    for i, f in enumerate(fsets):
        folds = sel.loc[f, [f"bacc_fold{k}" for k in range(5)]].to_numpy(float)
        ax.scatter(np.full(5, i) + np.linspace(-0.2, 0.2, 5), folds, color="k", s=8, zorder=3,
                   label="per fold" if i == 0 else None)
        ax.text(i, 0.05, f"{sel.loc[f, 'bacc_neuron']:.3f}", ha="center", fontsize=8, color="w", weight="bold")
    ses = sel.loc[fsets, "session_correct"] / sel.loc[fsets, "n_sessions"]
    ax.scatter(x, ses, marker="D", color="w", edgecolor="k", s=30, zorder=4, label="session vote, fraction correct")
    for i, f in enumerate(fsets):
        ax.text(i, ses.iloc[i] + 0.035, f"{int(sel.loc[f, 'session_correct'])}/46", ha="center", fontsize=6.5)
    ax.axhline(0.25, color="k", ls=":", lw=0.8)
    ax.text(len(fsets) - 0.5, 0.26, "chance", fontsize=7, ha="right")
    ax.set_xticks(x); ax.set_xticklabels([FS_NAME[f] for f in fsets], fontsize=7.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("held-out balanced accuracy (4 lines)", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower left", bbox_to_anchor=(0.0, 0.3))
    ax.set_title("(a) Cre line from feature group, held-out mice", fontsize=9)


def panel_gain(ax, gr):
    g = gr[gr.line == "all lines"]
    sets = [s for s in LABEL_SETS if s in set(g.labels)]
    y = np.arange(len(sets))
    lo = -1.0  # axis floor; bars below it are cut and labelled with their value
    for off, fam, col in [(-0.18, "ridge", "0.4"), (0.18, "MLP", "tab:blue")]:
        v = g[g.family == fam].set_index("labels").loc[sets]
        ax.barh(y + off, v.frac.clip(lower=lo), height=0.34, color=col, alpha=0.8, label=fam)
        ok = v.frac > lo
        ax.errorbar(v.frac[ok], (y + off)[ok.to_numpy()], xerr=[(v.frac - v.ci_lo)[ok], (v.ci_hi - v.frac)[ok]],
                    fmt="none", ecolor="k", lw=0.7, capsize=2)
        for yi, fr in zip(y + off, v.frac):
            ax.text(1.3 if fr > 0 else lo + 0.03, yi, f"{fr:.2f}", va="center", fontsize=6.5,
                    color="k" if fr > 0 else "darkred")
    ax.set_xlim(lo, 1.45)
    ax.axvline(1, color="k", ls="--", lw=0.8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(sets, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("fraction of the true-label gain recovered (95% bootstrap CI over mice;\naxis cut at -1, red numbers = values below)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) forecasting gain recovered by each label set (46 mice)", fontsize=9)


def panel_window(ax, wr, summ):
    w = wr[wr.clf == "sel"].sort_values("window_s")
    ax.plot(w.window_s, w.bacc_window, "o-", color="tab:purple", label="one window (S, selected clf)")
    ax.errorbar(w.window_s, w.bacc_fold_mean, yerr=w.bacc_fold_std, fmt="none", ecolor="tab:purple", capsize=3,
                lw=0.8)
    short = w[w.windows_per_neuron > 1]
    ax.plot(short.window_s, short.bacc_neuron_pooled, "s--", color="tab:purple", alpha=0.5, mfc="w",
            label=f"{int(short.windows_per_neuron.iloc[0])} windows pooled per neuron")
    for _, r in w.iterrows():
        ax.text(r.window_s, r.bacc_window - 0.05, f"{r.bacc_window:.2f}", ha="center", fontsize=7)
    sel = summ[summ.clf == "sel"].set_index("featset")
    for f, ls in [("R", ":"), ("A", "-.")]:
        if f in sel.index:
            ax.axhline(sel.loc[f, "bacc_neuron"], color=FS_COLORS[f], ls=ls, lw=1,
                       label=f"{f}, full train part")
    ax.axhline(0.25, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(w.window_s)
    ax.set_xticklabels([n if n != "full train part" else f"full train part\n(~{s / 60:.0f} min)"
                        for n, s in zip(w.window, w.window_s)], fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("length of activity used for the S features", fontsize=8)
    ax.set_ylabel("held-out balanced accuracy", fontsize=8)
    ax.legend(fontsize=6.5, loc="lower right")
    ax.set_title("(d) how long must a neuron be watched?", fontsize=9)


def main():
    summ = pd.read_csv(RES / "classifier_summary.csv")
    conf = pd.read_csv(RES / "confusions.csv")
    gr = pd.read_csv(RES / "gain_recovered.csv")
    wr = pd.read_csv(RES / "window_accuracy.csv")
    imp = pd.read_csv(RES / "feature_importance.csv")
    st = pd.read_csv(RES / "session_features.csv")
    sc = pd.read_csv(RES / "session_classifier.csv")
    mice = pd.read_csv(RES / "assignment_mice_scores.csv")

    # Main figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panel_accuracy(axes[0, 0], summ)
    draw_confusion(axes[0, 1], confusion_from(conf, "S", "neuron"),
                   "(b) S features (single-neuron, scale-free): held-out neurons", fontsize=9)
    panel_gain(axes[1, 0], gr)
    panel_window(axes[1, 1], wr, summ)
    fig.suptitle("Step 11: does the activity classifier recognise the neuron or the recording?", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "step11_confound.png", dpi=150)
    plt.close(fig)

    # All confusion matrices
    fsets = [f for f in FS_NAME if f in set(conf.featset)]
    fig, axes = plt.subplots(2, len(fsets), figsize=(3.3 * len(fsets), 6.6))
    for j, f in enumerate(fsets):
        s = summ[(summ.featset == f) & (summ.clf == "sel")].iloc[0]
        draw_confusion(axes[0, j], confusion_from(conf, f, "neuron"),
                       f"{f}: neurons (bal. acc {s.bacc_neuron:.3f})", fontsize=7)
        draw_confusion(axes[1, j], confusion_from(conf, f, "session"),
                       f"{f}: session vote ({int(s.session_correct)}/46)", fontsize=7)
    fig.suptitle("Step 11: held-out confusion matrices, classifier selected per fold by inner CV", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step11_confusions.png", dpi=130)
    plt.close(fig)

    # Session-level summaries
    feats = ["n neurons", "depth", "pairwise corr mean", "pairwise corr median", "pop corr mean",
             "pop corr median", "dF/F std median", "dF/F mean median"]
    fig, axes = plt.subplots(2, 5, figsize=(17, 7))
    rng = np.random.default_rng(0)
    for ax, c in zip(axes.flat, feats):
        for i, l in enumerate(LINES):
            v = st[st.line == l][c]
            ax.scatter(np.full(len(v), i) + rng.uniform(-0.15, 0.15, len(v)), v, color=COLORS[l], s=18)
        ax.set_xticks(range(4)); ax.set_xticklabels(LINES, fontsize=8)
        ax.set_title(c, fontsize=9)
        if c == "n neurons":
            ax.set_yscale("log")
    ax = axes.flat[8]
    rep = st.groupby("line", sort=False).reporter.first().loc[LINES]
    ax.axis("off")
    ax.text(0, 1, "Reporter (GCaMP) per line\n(from ophys_experiments.json)\n\n"
            + "\n".join(f"{l}: {r}" for l, r in rep.items()), fontsize=8, va="top", transform=ax.transAxes)
    ax = axes.flat[9]
    s5 = sc[(sc.scheme == "5 folds") & (sc.clf == "sel")].set_index("features")
    s1 = sc[(sc.scheme == "leave one mouse out") & (sc.clf == "sel")].set_index("features")
    y = np.arange(len(s5))
    ax.barh(y - 0.18, s5.correct, height=0.34, color="0.4", label="5 outer folds")
    ax.barh(y + 0.18, s1.loc[s5.index].correct, height=0.34, color="tab:purple", label="leave one mouse out")
    for i, (a, b) in enumerate(zip(s5.correct, s1.loc[s5.index].correct)):
        ax.text(a + 0.5, i - 0.18, str(a), va="center", fontsize=7)
        ax.text(b + 0.5, i + 0.18, str(b), va="center", fontsize=7)
    ax.axvline(46, color="k", lw=0.6)
    ax.axvline(46 / 4, color="k", ls=":", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(s5.index, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 52)
    ax.set_xlabel("sessions with the right Cre line (of 46)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("session-only classifier", fontsize=9)
    fig.suptitle("Step 11: recording-level summaries per Cre line (one dot per session, train part)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step11_sessions.png", dpi=130)
    plt.close(fig)

    # Feature importance
    fig, axes = plt.subplots(1, 3, figsize=(16, 7.5), gridspec_kw=dict(width_ratios=[1.3, 0.8, 1.3]))
    for ax, f in zip(axes, ["S", "R", "A"]):
        d = imp[imp.featset == f]
        lr = d[d.clf == "logreg"].set_index("feature")
        gb = d[d.clf == "gboost"].set_index("feature")
        order = (lr["mean"] + gb.loc[lr.index, "mean"]).sort_values(ascending=False).index[:20]
        y = np.arange(len(order))
        ax.barh(y - 0.2, lr.loc[order, "mean"], height=0.4, xerr=lr.loc[order, "std"], color="tab:purple",
                label="logreg", error_kw=dict(lw=0.6))
        ax.barh(y + 0.2, gb.loc[order, "mean"], height=0.4, xerr=gb.loc[order, "std"], color="0.6",
                label="gradient boosting", error_kw=dict(lw=0.6))
        ax.set_yticks(y); ax.set_yticklabels(order, fontsize=7.5)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel("drop in held-out balanced accuracy when permuted\n(mean +- std over 5 folds)", fontsize=8)
        ax.set_title(f"{f} features (top {len(order)})", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("Step 11: held-out permutation importance", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step11_features.png", dpi=130)
    plt.close(fig)

    # Per mouse
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax = axes[0]
    m = mice[mice.family == "ridge"]
    nn = st.set_index("id").loc[m.id, "n neurons"].to_numpy()
    for l in LINES:
        sel = (m.line == l).to_numpy()
        ax.scatter(nn[sel], m["frac_correct_S neuron"][sel], color=COLORS[l], s=22, label=l)
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("neurons in the recording", fontsize=8)
    ax.set_ylabel("fraction of neurons labelled right (S classifier)", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("S classifier, per held-out mouse", fontsize=9)
    for ax, fam in zip(axes[1:], ["ridge", "MLP"]):
        m = mice[mice.family == fam]
        gt, gp = m.score_true - m.score_none, m["score_S neuron"] - m.score_none
        lim = [min(gt.min(), gp.min()) - 0.005, max(gt.max(), gp.max()) + 0.005]
        for l in LINES:
            sel = m.line == l
            ax.scatter(gt[sel], gp[sel], color=COLORS[l], s=22, label=l)
        ax.plot(lim, lim, "k--", lw=0.8)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
        ax.set_xlim(lim); ax.set_ylim(lim)
        name = "ridge d" if fam == "ridge" else "MLP m2"
        none = "b" if fam == "ridge" else "m1"
        ax.set_xlabel(f"score gain, true label ({name} true - {none})", fontsize=8)
        ax.set_ylabel(f"score gain, S per-neuron label ({name} S - {none})", fontsize=8)
        ax.set_title(f"{fam}: per-mouse gain with S labels vs true labels", fontsize=9)
    fig.suptitle("Step 11: per held-out mouse", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step11_mice.png", dpi=130)
    plt.close(fig)
    print("Saved plots/step11_confound.png, step11_confusions.png, step11_sessions.png, step11_features.png,"
          " step11_mice.png")


if __name__ == "__main__":
    main()
