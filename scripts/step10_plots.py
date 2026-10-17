"""Step 10 plots, made only from the CSV files in results/step10/ (written by
scripts/step10_inferred.py), so they can be remade without rerunning any model.

plots/step10_inferred.png     (a) confusion matrix of the selected classifier, (b) score per model,
                              (c, d) per-mouse gain of the inferred vs the true label (MLP, ridge)
plots/step10_classifiers.png  held-out balanced accuracy per feature set x classifier (neuron, session)
plots/step10_confusions.png   confusion matrices of all feature sets x classifiers, neuron and session level
plots/step10_features.png     most useful hand features and their per-line distributions
plots/step10_mice.png         classifier accuracy per mouse and the forecasting cost of wrong labels
plots/step10_examples.png     example neurons: true trace and forecasts with the true vs predicted label
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step10"
PLOTS = ROOT / "plots"
LINES = ["Cux2", "Sst", "Vip", "Pvalb"]
COLORS = {"Cux2": "tab:blue", "Sst": "tab:orange", "Vip": "tab:green", "Pvalb": "tab:red"}
FEATSETS, CLFS = ["a", "b", "a+b"], ["logreg", "gboost"]
FS_NAME = {"a": "hand (a)", "b": "m1 emb. (b)", "a+b": "a+b"}
LABEL = {"b": "b", "d_true": "d true", "d_pred_neuron": "d pred\nneuron", "d_pred_session": "d pred\nsession",
         "e": "e shuf", "m1": "m1", "m2_true": "m2 true", "m2_pred_neuron": "m2 pred\nneuron",
         "m2_pred_session": "m2 pred\nsession", "m2_soft": "m2 soft", "m3": "m3 shuf"}
RIDGE = ["b", "d_true", "d_pred_neuron", "d_pred_session", "e"]
MLP = ["m1", "m2_true", "m2_pred_neuron", "m2_pred_session", "m2_soft", "m3"]


def confusion(true, pred):
    C = np.zeros((4, 4), int)
    np.add.at(C, (np.asarray(true), np.asarray(pred)), 1)
    return C


def bacc(true, pred):
    C = confusion(true, pred)
    rec = np.diag(C) / np.maximum(C.sum(1), 1)
    return rec[C.sum(1) > 0].mean()


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


def gain_panel(ax, mice, none, true, preds, title):
    gt = mice[f"score_{true}"] - mice[f"score_{none}"]
    allg = np.concatenate([gt] + [mice[f"score_{p}"] - mice[f"score_{none}"] for p, _ in preds])
    lim = [allg.min() - 0.005, allg.max() + 0.005]
    for p, marker in preds:
        gp = mice[f"score_{p}"] - mice[f"score_{none}"]
        for line in LINES:
            sel = mice.line == line
            if marker == "o":
                ax.scatter(gt[sel], gp[sel], s=22, marker="o", color=COLORS[line], alpha=0.85, label=line)
            else:
                ax.scatter(gt[sel], gp[sel], s=30, marker=marker, color=COLORS[line], alpha=0.85)
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"score gain with true label ({true} - {none})", fontsize=8)
    ax.set_ylabel(f"score gain with inferred label (pred - {none})", fontsize=8)
    ax.set_title(title, fontsize=9)


def main():
    mice = pd.read_csv(RES / "mice_scores.csv")
    pred = pd.read_csv(RES / "neuron_predictions.csv")
    fs = pd.read_csv(RES / "fold_scores.csv")
    clf = pd.read_csv(RES / "classifier_folds.csv")
    feat = pd.read_csv(RES / "features_summary.csv")
    nf = pd.read_csv(RES / "neuron_features.csv")
    ns = pd.read_csv(RES / "neuron_scores.csv")
    tr = pd.read_csv(RES / "example_traces.csv")
    gr = pd.read_csv(RES / "gain_recovered.csv")
    ses = pred.drop_duplicates("id")

    # ---- Main figure
    fig, axes = plt.subplots(1, 4, figsize=(21, 5))
    sel_tags = clf[clf.selected].apply(lambda r: f"{r.featset}/{r.clf}", axis=1).tolist()
    draw_confusion(axes[0], confusion(pred.true, pred.pred_sel),
                   f"(a) held-out neurons, selected classifier per fold\n({', '.join(sel_tags)})\n"
                   f"balanced acc {bacc(pred.true, pred.pred_sel):.2f}; session label correct "
                   f"{(ses.true == ses.session_sel).sum()}/{len(ses)} mice")
    ax = axes[1]
    models = RIDGE + MLP
    v = fs[fs.line == "all lines"].groupby("model").score
    mu, sd = v.mean()[models], v.std(ddof=0)[models]
    x = np.r_[np.arange(len(RIDGE)), np.arange(len(MLP)) + len(RIDGE) + 1]
    col = {"b": "0.6", "d_true": "0.25", "d_pred_neuron": "tab:purple", "d_pred_session": "tab:pink", "e": "tab:olive",
           "m1": "tab:blue", "m2_true": "tab:orange", "m2_pred_neuron": "tab:purple", "m2_pred_session": "tab:pink",
           "m2_soft": "tab:cyan", "m3": "tab:olive"}
    ax.bar(x, mu, yerr=sd, color=[col[m] for m in models], capsize=3)
    for k in range(5):
        f = fs[(fs.line == "all lines") & (fs.fold == k)].set_index("model").score[models]
        ax.plot(x[:len(RIDGE)], f.values[:len(RIDGE)], color="k", lw=0.4, alpha=0.4)
        ax.plot(x[len(RIDGE):], f.values[len(RIDGE):], color="k", lw=0.4, alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels([LABEL[m] for m in models], fontsize=7, rotation=90)
    ax.set_ylim(min(mu - sd) - 0.03, max(mu + sd) + 0.02)
    ax.set_ylabel("held-out score (1 - MSE / MSE copy-last)")
    ax.set_title("(b) all 46 mice: score per model\n(bars mean +- std over 5 folds, lines = folds; left ridge, right MLP)",
                 fontsize=9)
    g = gr[(gr.line == "all lines")].set_index("pred")
    gain_panel(axes[2], mice, "m1", "m2_true", [("m2_pred_neuron", "o"), ("m2_pred_session", "x")],
               f"(c) MLP, per mouse: o = pred-neuron, x = pred-session\nfraction of gain recovered: neuron "
               f"{g.loc['m2_pred_neuron', 'frac']:.2f}, session {g.loc['m2_pred_session', 'frac']:.2f},"
               f" soft {g.loc['m2_soft', 'frac']:.2f}")
    gain_panel(axes[3], mice, "b", "d_true", [("d_pred_neuron", "o"), ("d_pred_session", "x")],
               f"(d) ridge, per mouse: o = pred-neuron, x = pred-session\nfraction of gain recovered: neuron "
               f"{g.loc['d_pred_neuron', 'frac']:.2f}, session {g.loc['d_pred_session', 'frac']:.2f}")
    axes[2].legend(fontsize=7, loc="upper left")
    fig.suptitle("Step 10: cell type inferred from activity in place of the true Cre label (training always uses"
                 " the true label; only the test label changes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "step10_inferred.png", dpi=150)
    plt.close(fig)

    # ---- Classifier accuracy per feature set x classifier
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    combos = [(f, c) for f in FEATSETS for c in CLFS]
    for ax, level in zip(axes, ["neuron", "session"]):
        for i, (f, c) in enumerate(combos):
            sub = clf[(clf.featset == f) & (clf.clf == c)]
            tag = f"{f}_{c}"
            if level == "neuron":
                pooled = bacc(pred.true, pred[f"pred_{tag}"])
                per = sub.heldout_bacc_neuron
                ax.scatter(np.full(len(sub), i) + 0.15, sub.inner_bacc, marker="_", color="k", s=60,
                           label="inner CV (training mice)" if i == 0 else None)
            else:
                pooled = (ses.true == ses[f"session_{tag}"]).mean()
                per = sub.heldout_acc_session
            ax.bar(i, pooled, color="tab:blue" if c == "logreg" else "tab:green", alpha=0.6,
                   label=("logreg" if c == "logreg" else "gradient boosting") if i < 2 else None)
            ax.scatter(np.full(len(sub), i) - 0.15, per, color="k", s=10, label="held-out fold" if i == 0 else None)
            ax.text(i, pooled + 0.02, f"{pooled:.2f}", ha="center", fontsize=8)
        ax.axhline(0.25, color="k", ls=":", lw=0.8)
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels([f"{FS_NAME[f]}\n{c}" for f, c in combos], fontsize=7)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("balanced accuracy (neurons)" if level == "neuron" else "accuracy (mice, majority vote)")
        ax.set_title("neuron level, pooled over held-out mice (dotted = chance)" if level == "neuron" else
                     "session level: majority vote over the neurons of a recording", fontsize=9)
        ax.legend(fontsize=7)
    fig.suptitle("Step 10: held-out cell-type classification by feature set and classifier", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step10_classifiers.png", dpi=150)
    plt.close(fig)

    # ---- All confusion matrices
    fig, axes = plt.subplots(2, 6, figsize=(24, 8))
    for i, (f, c) in enumerate(combos):
        tag = f"{f}_{c}"
        draw_confusion(axes[0, i], confusion(pred.true, pred[f"pred_{tag}"]),
                       f"{FS_NAME[f]} / {c}, neurons\nbalanced acc {bacc(pred.true, pred[f'pred_{tag}']):.2f}", 7)
        draw_confusion(axes[1, i], confusion(ses.true, ses[f"session_{tag}"]),
                       f"{FS_NAME[f]} / {c}, mice (vote)\nacc {(ses.true == ses[f'session_{tag}']).mean():.2f}", 7)
    fig.suptitle("Step 10: held-out confusion matrices (rows = true line; count and row fraction)", fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOTS / "step10_confusions.png", dpi=130)
    plt.close(fig)

    # ---- Features: permutation importance and per-line distributions of the top features
    fig = plt.figure(figsize=(18, 8))
    gs = fig.add_gridspec(2, 6)
    top = feat.sort_values("perm_logreg_mean", ascending=False).head(15)
    for j, c in enumerate(CLFS):
        ax = fig.add_subplot(gs[j, 0:2])
        t = feat.sort_values(f"perm_{c}_mean", ascending=False).head(12)[::-1]
        ax.barh(range(len(t)), t[f"perm_{c}_mean"], xerr=t[f"perm_{c}_std"], color="tab:blue" if c == "logreg"
                else "tab:green", alpha=0.7)
        ax.set_yticks(range(len(t))); ax.set_yticklabels(t.feature, fontsize=7)
        ax.set_xlabel("drop in held-out balanced accuracy when shuffled", fontsize=7)
        ax.set_title(f"permutation importance, hand features / {c}\n(mean +- std over 5 folds)", fontsize=8)
    for i, nm in enumerate(top.feature.head(8)):
        ax = fig.add_subplot(gs[i // 4, 2 + i % 4])
        data = [nf[nf.cre.str.startswith(l)][nm].to_numpy() for l in LINES]
        lo, hi = np.percentile(np.concatenate(data), [1, 99])
        bp = ax.boxplot([np.clip(d, lo, hi) for d in data], showfliers=False, patch_artist=True)
        for patch, l in zip(bp["boxes"], LINES):
            patch.set_facecolor(COLORS[l]); patch.set_alpha(0.5)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(LINES, fontsize=7)
        H = feat.set_index("feature").loc[nm, "kruskal_H"]
        ax.set_title(f"{nm}\nKruskal-Wallis H {H:.0f}", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("Step 10: most useful hand features (left) and per-line distributions of the top 8 for logreg"
                 " (right; all neurons, whiskers 1.5 IQR)", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step10_features.png", dpi=140)
    plt.close(fig)

    # ---- Per mouse: classifier accuracy and cost of wrong labels
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    ax = axes[0]
    for line in LINES:
        m = mice[mice.line == line]
        ok = m.session_pred == m.line
        ax.scatter(m.n_neurons[ok], m.frac_neurons_correct[ok], color=COLORS[line], s=25, label=f"{line}")
        ax.scatter(m.n_neurons[~ok], m.frac_neurons_correct[~ok], facecolor="none", edgecolor=COLORS[line], s=40,
                   marker="s")
    ax.set_xscale("log")
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xlabel("neurons in the recording"); ax.set_ylabel("fraction of neurons labelled correctly")
    ax.set_title("per held-out mouse (filled = session vote correct,\nopen square = session vote wrong)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[1]
    for line in LINES:
        m = mice[mice.line == line]
        ax.scatter(m.frac_neurons_correct, m.score_m2_pred_neuron - m.score_m2_true, color=COLORS[line], s=25)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("fraction of neurons labelled correctly")
    ax.set_ylabel("score, m2 pred-neuron - m2 true")
    ax.set_title("per mouse: forecasting cost of the inferred label (MLP)", fontsize=9)
    ax = axes[2]
    data, ticks, cols = [], [], []
    for line in LINES:
        for ok in [True, False]:
            sub = ns[(ns.line == line) & ((ns.pred == ns.line) == ok)]
            data.append((sub.score_m2_pred_neuron - sub.score_m2_true).to_numpy())
            ticks.append(f"{line}\n{'right' if ok else 'wrong'}\nn={len(sub)}")
            cols.append(COLORS[line])
    bp = ax.boxplot(data, showfliers=False, patch_artist=True)
    for patch, c, k in zip(bp["boxes"], cols, range(len(cols))):
        patch.set_facecolor(c); patch.set_alpha(0.6 if k % 2 == 0 else 0.25)
    ax.set_xticks(range(1, len(ticks) + 1)); ax.set_xticklabels(ticks, fontsize=7)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("per-neuron score, m2 pred-neuron - m2 true")
    ax.set_title("per neuron: correctly vs wrongly labelled neurons (MLP)", fontsize=9)
    fig.suptitle("Step 10: classifier accuracy per held-out mouse and what wrong labels cost", fontsize=10)
    fig.tight_layout()
    fig.savefig(PLOTS / "step10_mice.png", dpi=150)
    plt.close(fig)

    # ---- Example neurons
    keys = tr[["id", "neuron", "line", "pred", "kind"]].drop_duplicates().reset_index(drop=True)
    fig, axes = plt.subplots(len(keys), 1, figsize=(15, 2.1 * len(keys)), squeeze=False)
    for ax, (_, k) in zip(axes[:, 0], keys.iterrows()):
        t = tr[(tr.id == k.id) & (tr.neuron == k.neuron)]
        sec = t.bin / 10
        ax.plot(sec, t.y, color="k", lw=1.0, label="true z dF/F")
        ax.plot(sec, t.m1, color="tab:blue", lw=0.8, label="m1 (no label)")
        ax.plot(sec, t.m2_true, color="tab:orange", lw=0.8, label="m2 true label")
        ax.plot(sec, t.m2_pred_neuron, color="tab:purple", lw=0.8, ls="--", label="m2 predicted label")
        ax.plot(sec, t.m2_soft, color="tab:cyan", lw=0.8, ls=":", label="m2 soft")
        n = ns[(ns.id == k.id) & (ns.neuron == k.neuron)].iloc[0]
        ax.set_title(f"{k.line} mouse {k.id} neuron {k.neuron}: predicted {k.pred} ({k.kind}); test score m1"
                     f" {n.score_m1:.3f}, m2 true {n.score_m2_true:.3f}, m2 pred {n.score_m2_pred_neuron:.3f},"
                     f" m2 soft {n.score_m2_soft:.3f}", fontsize=8)
        ax.set_ylabel("z", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7, ncol=5, loc="upper right")
    axes[-1, 0].set_xlabel("time in test part (s); forecasts = consecutive 1 s windows")
    fig.suptitle("Step 10: example held-out neurons, forecasts with the true vs the inferred label (mean of 3 seeds)",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(PLOTS / "step10_examples.png", dpi=130)
    plt.close(fig)
    print("Saved plots/step10_inferred.png, step10_classifiers.png, step10_confusions.png, step10_features.png,"
          " step10_mice.png, step10_examples.png")


if __name__ == "__main__":
    main()
