"""Step 12 plots, made only from the CSV files in results/step12/ (written by
scripts/step12_bugeon.py), so they can be remade without rerunning any model.

plots/step12_bugeon.png      (a) labelled neurons per type and mouse, (b) score per model (sessions),
                             (c) per-session gain of the true type label (ridge, MLP) with controls,
                             (d) confusion matrix of the S classifier (6 classes, held-out mice)
plots/step12_types.png       per type: forecasting gain of the true label, of the shuffled-type control
                             and of S-inferred labels (ridge, MLP), pooled over sessions and per session
plots/step12_classifier.png  classifier accuracy per feature set vs NuCLR / NeuPRINT, confusion matrices
                             of S+state and of the inhibitory-only classifiers, top feature importances
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step12"
PLOTS = ROOT / "plots"
TYPES = ["EC", "Lamp5", "Pvalb", "Sncg", "Sst", "Vip"]
INH = TYPES[1:]
T_COLORS = {"EC": "0.55", "Lamp5": "tab:purple", "Pvalb": "tab:red", "Sncg": "tab:olive", "Sst": "tab:orange",
            "Vip": "tab:green"}
MOUSE_MARK = {"SB025": "o", "SB026": "s", "SB028": "^", "SB030": "D"}
FAM = {"ridge": ("b", "d", "e"), "MLP": ("m1", "m2", "m3")}


def confusion_from(conf, name, classes, clf="sel"):
    c = conf[(conf.set == name) & (conf.clf == clf)]
    return c.pivot(index="true", columns="pred", values="count").loc[classes, classes].to_numpy()


def draw_confusion(ax, C, classes, title, fontsize=8):
    R = C / np.maximum(C.sum(1, keepdims=True), 1)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{C[i, j]}\n{R[i, j]:.2f}", ha="center", va="center", fontsize=fontsize - 1.5,
                    color="white" if R[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, fontsize=fontsize)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, fontsize=fontsize)
    ax.set_xlabel("predicted", fontsize=fontsize); ax.set_ylabel("true type", fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize + 1)


def panel_counts(ax, neu):
    mice = sorted(neu.mouse.unique())
    x = np.arange(len(TYPES))
    bottom = np.zeros(len(TYPES))
    shades = dict(zip(mice, ["0.15", "0.4", "0.65", "0.85"]))
    for m in mice:
        c = neu[neu.mouse == m].type.value_counts().reindex(TYPES, fill_value=0).to_numpy()
        ax.bar(x, c, bottom=bottom, color=shades[m], edgecolor="k", lw=0.4, label=f"{m}")
        bottom += c
    uniq = {t: len(neu[(neu.type == t) & neu.uid.notna()].drop_duplicates(["mouse", "uid"])) for t in INH}
    for i, t in enumerate(TYPES):
        txt = f"{int(bottom[i])}" + (f"\n({uniq[t]} cells)" if t in uniq else "")
        ax.text(i, bottom[i] * 1.15, txt, ha="center", fontsize=6.5)
    ax.set_yscale("log")
    ax.set_ylim(10, bottom.max() * 4)
    ax.set_xticks(x); ax.set_xticklabels(TYPES, fontsize=8)
    for lab in ax.get_xticklabels():
        lab.set_color(T_COLORS[lab.get_text()] if lab.get_text() != "EC" else "k")
    ax.set_ylabel("neuron-sessions (log)", fontsize=8)
    ax.legend(fontsize=6.5, title="mouse", title_fontsize=6.5, loc="upper right", ncol=2)
    ax.set_title(f"(a) labelled neurons per type: {len(neu)} neuron-sessions,\n17 sessions, 4 mice"
                 " (unique matched cells in brackets)", fontsize=9)


def panel_models(ax, sc):
    ses = sc[sc.level == "session"]
    cols = [("ridge", "none", "b\nno label"), ("ridge", "true", "d\ntrue type"), ("ridge", "control", "e\nshuffled"),
            ("ridge", "S", "d\nS-inferred"), ("MLP", "none", "m1\nno label"), ("MLP", "true", "m2\ntrue type"),
            ("MLP", "control", "m3\nshuffled"), ("MLP", "S", "m2\nS-inferred")]
    x = np.arange(len(cols)) + np.array([0, 0, 0, 0, 1, 1, 1, 1])
    colors = ["0.6", "tab:red", "tab:olive", "tab:purple"] * 2
    for i, (fam, lab, _) in enumerate(cols):
        v = ses[ses.family == fam][lab].to_numpy()
        ax.bar(x[i], v.mean(), color=colors[i], alpha=0.75)
        ax.text(x[i], v.mean() + 0.003, f"{v.mean():.4f}", ha="center", fontsize=6, rotation=90, va="bottom")
    for fam, idx in [("ridge", range(4)), ("MLP", range(4, 8))]:
        d = ses[ses.family == fam]
        for _, r in d.iterrows():
            ax.plot([x[i] for i in idx], [r[cols[i][1]] for i in idx], color="k", lw=0.3, alpha=0.35,
                    marker=MOUSE_MARK[r.mouse], ms=2)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in cols], fontsize=6.5)
    lo = ses[["none", "true", "control", "S"]].min().min()
    hi = ses[["none", "true", "control", "S"]].max().max()
    ax.set_ylim(lo - 0.01, hi + 0.02)
    ax.set_ylabel("test score (1 - MSE / MSE copy-last)", fontsize=8)
    ax.set_title("(b) score per model: bars = mean over 17 sessions,\nlines = sessions (marker = mouse)", fontsize=9)


def panel_gain(ax, sc, st):
    ses = sc[sc.level == "session"]
    items = [("ridge", "true", "d - b\ntrue"), ("ridge", "control", "e - b\nshuffled"), ("ridge", "S", "d(S) - b\nS-inferred"),
             ("MLP", "true", "m2 - m1\ntrue"), ("MLP", "control", "m3 - m1\nshuffled"), ("MLP", "S", "m2(S) - m1\nS-inferred")]
    x = np.arange(len(items)) + np.array([0, 0, 0, 1, 1, 1])
    rng = np.random.default_rng(0)
    for i, (fam, lab, _) in enumerate(items):
        d = ses[ses.family == fam]
        g = (d[lab] - d["none"]).to_numpy()
        for m, mk in MOUSE_MARK.items():
            sel = (d.mouse == m).to_numpy()
            ax.scatter(x[i] + rng.uniform(-0.15, 0.15, sel.sum()), g[sel], marker=mk, s=14, facecolor="none",
                       edgecolor={"true": "tab:red", "control": "tab:olive", "S": "tab:purple"}[lab], lw=0.8,
                       label=m if i == 0 else None)
        ax.plot([x[i] - 0.25, x[i] + 0.25], [g.mean()] * 2, color="k", lw=1.5)
        r = st[(st.test == "sessions") & (st.type == "all") & (st.family == fam) & (st.a == lab) & (st.b == "none")]
        if len(r):
            r = r.iloc[0]
            ax.annotate(f"{r.a_beats_b}/17\np={r.p:.2g}", (x[i], 1.0), xycoords=("data", "axes fraction"),
                        ha="center", va="top", fontsize=6)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([c[2] for c in items], fontsize=6.5)
    ax.set_ylabel("score gain over the no-label model", fontsize=8)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + (hi - lo) * 0.25)
    ax.legend(fontsize=6, loc="lower right", ncol=2, title="mouse", title_fontsize=6)
    ax.set_title("(c) per-session gain of the type label (black bar = mean;\n"
                 "text: sessions with gain > 0, Wilcoxon p over 17 sessions)", fontsize=9)


def main():
    neu = pd.read_csv(RES / "neurons.csv")
    sc = pd.read_csv(RES / "scores.csv")
    st = pd.read_csv(RES / "stats.csv")
    conf = pd.read_csv(RES / "confusions.csv")
    summ = pd.read_csv(RES / "classifier_summary.csv")
    ts = pd.read_csv(RES / "type_session_scores.csv")
    imp = pd.read_csv(RES / "feature_importance.csv")

    # Main figure
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    panel_counts(axes[0, 0], neu)
    panel_models(axes[0, 1], sc)
    panel_gain(axes[1, 0], sc, st)
    s6 = summ[(summ.set == "S") & (summ.clf == "sel")].iloc[0]
    draw_confusion(axes[1, 1], confusion_from(conf, "S", TYPES), TYPES,
                   f"(d) S classifier, held-out mice (6 classes): balanced acc {s6.bacc:.3f},\n"
                   f"macro F1 {s6.macro_f1:.3f}; E vs I balanced acc {s6.bacc_EI:.3f}")
    fig.suptitle("Step 12: Bugeon et al. 2022, many cell types in one field of view and one indicator"
                 " (4 mice, 17 sessions, leave one mouse out)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "step12_bugeon.png", dpi=150)
    plt.close(fig)

    # Per type
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, fam in zip(axes, ["ridge", "MLP"]):
        tu = sc[(sc.level == "type") & (sc.family == fam)].set_index("unit").loc[TYPES]
        x = np.arange(len(TYPES))
        for off, lab, col, name in [(-0.27, "true", "tab:red", "true type"), (0, "control", "tab:olive", "shuffled control"),
                                    (0.27, "S", "tab:purple", "S-inferred type")]:
            ax.bar(x + off, tu[lab] - tu["none"], width=0.25, color=col, alpha=0.75, label=f"{name} (pooled over sessions)")
        rng = np.random.default_rng(0)
        d = ts[ts.family == fam]
        for i, t in enumerate(TYPES):
            dt = d[d.type == t]
            ax.scatter(i - 0.27 + rng.uniform(-0.08, 0.08, len(dt)), dt.true - dt.none, s=np.clip(dt.n_neurons, 3, 40),
                       color="k", alpha=0.5, lw=0, label="per session (size = neurons)" if i == 0 else None)
            r = st[(st.test == "neurons") & (st.type == t) & (st.family == fam) & (st.a == "true") & (st.b == "none")].iloc[0]
            ax.annotate(f"n={int(tu.loc[t, 'n_neurons'])}\n{r.a_beats_b}/{r.n} up\np={r.p:.2g}", (i, 1.0),
                        xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6.5)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(TYPES)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * 0.3)
        ax.set_ylabel("score gain over no label", fontsize=8)
        names = FAM[fam]
        ax.set_title(f"{fam}: {names[1]} - {names[0]} per type (text: neurons with gain > 0, Wilcoxon over neurons)",
                     fontsize=9)
        ax.legend(fontsize=6, loc="lower left", framealpha=0.6)
    fig.suptitle("Step 12: forecasting gain of the type label, per type", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step12_types.png", dpi=150)
    plt.close(fig)

    # Classifier
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(gs[0, 0])
    sets = ["S", "S+state", "S inh", "S+state inh"]
    sel = summ[summ.clf == "sel"].set_index("set").loc[sets]
    x = np.arange(len(sets))
    ax.bar(x - 0.2, sel.bacc, width=0.4, color="tab:purple", alpha=0.8, label="balanced accuracy")
    ax.bar(x + 0.2, sel.macro_f1, width=0.4, color="tab:blue", alpha=0.8, label="macro F1")
    for i in range(len(sets)):
        ax.text(i - 0.2, sel.bacc.iloc[i] + 0.01, f"{sel.bacc.iloc[i]:.2f}", ha="center", fontsize=7)
        ax.text(i + 0.2, sel.macro_f1.iloc[i] + 0.01, f"{sel.macro_f1.iloc[i]:.2f}", ha="center", fontsize=7)
    ax.axhline(0.39, color="tab:blue", ls="--", lw=0.8)
    ax.text(-0.45, 0.50, "NuCLR macro F1 0.39\n(5 subclasses)", fontsize=6.5, ha="left", color="tab:blue")
    ax.axhline(0.6, color="k", ls=":", lw=0.8)
    ax.text(-0.45, 0.62, "NeuPRINT ~0.6", fontsize=6.5, ha="left")
    ax.axhline(1 / 6, color="0.5", ls="-", lw=0.5)
    ax.axhline(1 / 5, color="0.5", ls="-", lw=0.5)
    ax.text(-0.45, 1 / 6 - 0.04, "chance 1/6, 1/5", fontsize=6.5, color="0.4")
    ax.set_xticks(x); ax.set_xticklabels(["S\n6 classes", "S+state\n6 classes", "S\n5 inh.", "S+state\n5 inh."], fontsize=7.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5, loc="upper left")
    ax.set_title("(a) held-out mice, selected classifier", fontsize=9)
    for k, (name, classes) in enumerate([("S+state", TYPES), ("S inh", INH), ("S+state inh", INH)]):
        r = summ[(summ.set == name) & (summ.clf == "sel")].iloc[0]
        draw_confusion(fig.add_subplot(gs[0, k + 1]), confusion_from(conf, name, classes), classes,
                       f"({'bcd'[k]}) {name}: bal. acc {r.bacc:.3f}, macro F1 {r.macro_f1:.3f}", fontsize=7)
    for k, name in enumerate(["S", "S+state"]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        top = imp[(imp.set == name) & (imp.clf == "logreg")].sort_values("mean", ascending=False).head(15)
        top2 = imp[(imp.set == name) & (imp.clf == "gboost")].set_index("feature")
        y = np.arange(len(top))[::-1]
        ax.barh(y + 0.2, top["mean"], height=0.4, xerr=top["std"], color="tab:purple", alpha=0.8, label="logreg")
        ax.barh(y - 0.2, top2.loc[top.feature, "mean"], height=0.4, xerr=top2.loc[top.feature, "std"], color="tab:cyan",
                alpha=0.8, label="gboost")
        ax.set_yticks(y); ax.set_yticklabels(top.feature, fontsize=7)
        ax.set_xlabel("drop in held-out balanced accuracy when permuted (mean +- std over 4 folds)", fontsize=7)
        ax.legend(fontsize=7)
        ax.set_title(f"({'ef'[k]}) {name} (6 classes): top 15 features by logreg permutation importance", fontsize=9)
    fig.suptitle("Step 12: cell type from single-neuron activity features, leave one mouse out", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step12_classifier.png", dpi=150)
    plt.close(fig)
    print("Saved plots/step12_bugeon.png, plots/step12_types.png, plots/step12_classifier.png")


if __name__ == "__main__":
    main()
