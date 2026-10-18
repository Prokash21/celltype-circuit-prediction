"""Step 15 plots, remade from the CSV files in results/step15/ only.

- plots/step15_signal.png: (a) example traces, dF/F vs deconvolved, per Cre line;
  (b) label gain and summary gain for Allen dF/F / deconvolved A / B / Bugeon;
  (c) per-line label gain, dF/F vs deconvolved; (d) S-classifier confusion matrix
  after deconvolution (A).
- plots/step15_mice.png: per-mouse gains, deconvolved vs dF/F (label and summary,
  ridge and MLP), with the fraction of the dF/F gain that survives.
- plots/step15_typing.png: S-classifier confusions (dF/F, A, B), per-line recall,
  and per-feature line separability, dF/F vs A.
- plots/step15_sparsity.png: fraction of zero bins (Allen deconvolved per line vs
  Bugeon per type) and OASIS decay time constants per line.
- plots/step15_lines.png: per-line summary gain and shuffled-label control.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
IN = ROOT / "results/step15"
LINES = ["Cux2", "Sst", "Vip", "Pvalb"]
LCOL = {"Cux2": "tab:blue", "Sst": "tab:orange", "Vip": "tab:green", "Pvalb": "tab:red"}
DSS = ["dff", "A", "B", "bugeon"]
DLAB = {"dff": "Allen dF/F\n(10 Hz)", "A": "Allen deconv. A\n(10 Hz)", "B": "Allen deconv. B\n(4.3 Hz)",
        "bugeon": "Bugeon\n(deconv., 4.3 Hz)"}
DCOL = {"dff": "0.25", "A": "tab:purple", "B": "tab:pink", "bugeon": "tab:olive"}
REPORTER = {"Cux2": "GCaMP6f", "Sst": "GCaMP6f", "Vip": "GCaMP6f", "Pvalb": "GCaMP6s"}


def read(name):
    return pd.read_csv(IN / name)


def panel_traces(fig, spec, ex):
    sub = spec.subgridspec(len(LINES), 1, hspace=0.35)
    for k, line in enumerate(LINES):
        ax = fig.add_subplot(sub[k])
        e = ex[ex.line == line]
        d = e[e.signal == "dF/F (30 Hz)"]
        s = e[e.signal == "deconvolved (30 Hz)"]
        ax.plot(d.t, d.value, color="k", lw=0.6, label="dF/F (30 Hz)")
        ax2 = ax.twinx()
        ax2.vlines(s.t[s.value > 0], 0, s.value[s.value > 0], color="tab:red", lw=0.6, label="OASIS (30 Hz)")
        ax2.set_ylim(0, None)
        ax2.set_yticks([])
        ax.set_xlim(0, d.t.max())
        ax.set_ylabel("dF/F", fontsize=7)
        ax.tick_params(labelsize=7)
        nid = int(e.neuron.iloc[0])
        ax.set_title(f"{line} ({REPORTER[line]}), mouse {e.id.iloc[0]} neuron {nid}", fontsize=8, loc="left")
        if k < len(LINES) - 1:
            ax.set_xticklabels([])
        if k == 0:
            h1, l1 = ax.get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax.legend(h1 + h2, l1 + l2, fontsize=6, loc="upper right", ncol=2)
    ax.set_xlabel("time from test-part start (s)", fontsize=8)
    return ax


def panel_gains(ax, gains):
    bars = [("label", "ridge"), ("label", "MLP"), ("summary", "ridge"), ("summary", "MLP")]
    w = 0.2
    for j, (g, fam) in enumerate(bars):
        for i, ds in enumerate(DSS):
            r = gains[(gains.ds == ds) & (gains.gain == g) & (gains.family == fam)].iloc[0]
            x = i + (j - 1.5) * w
            ax.bar(x, r.gain_mean, w, yerr=r.gain_sem, color=DCOL[ds], alpha=0.45 if g == "label" else 0.95,
                   hatch="//" if fam == "MLP" else None, edgecolor="k", lw=0.5, capsize=2)
            ax.text(x, r.gain_mean + r.gain_sem + 0.001, f"{r.gain_mean:+.3f}", ha="center", fontsize=5.5, rotation=90)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(DSS)))
    ax.set_xticklabels([DLAB[d] for d in DSS], fontsize=7)
    ax.set_ylabel("score gain over no label (mean ± sem over units)", fontsize=8)
    ax.set_ylim(None, ax.get_ylim()[1] * 1.15)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="0.6", alpha=0.45, edgecolor="k", label="genetic label, ridge (d - b)"),
                       Patch(facecolor="0.6", alpha=0.45, hatch="//", edgecolor="k", label="genetic label, MLP (m2 - m1)"),
                       Patch(facecolor="0.6", edgecolor="k", label="activity summary, ridge (km16 - b)"),
                       Patch(facecolor="0.6", hatch="//", edgecolor="k", label="activity summary, MLP (cont - m1)")],
              fontsize=6, loc="upper right")
    ax.set_title("(b) gain of the genetic label and of the activity summary\n(Allen: 46 mice; Bugeon: 17 sessions)",
                 fontsize=9, loc="left")


def panel_lines(axes, mice, lg):
    rng = np.random.default_rng(0)
    for ax, (fam, a, b) in zip(axes, [("ridge", "r_true", "r_none"), ("MLP", "m_true", "m_none")]):
        for i, line in enumerate(LINES):
            for j, ds in enumerate(["dff", "A", "B"]):
                m = mice[(mice.ds == ds) & (mice.line == line)]
                g = m[a] - m[b]
                x = i + (j - 1) * 0.27
                ax.bar(x, g.mean(), 0.25, color=DCOL[ds], alpha=0.6, edgecolor="k", lw=0.5,
                       label=DLAB[ds].replace("\n", " ") if i == 0 else None)
                ax.scatter(x + rng.uniform(-0.07, 0.07, len(g)), g, s=5, color="k", zorder=3)
                if ds != "dff":
                    r = lg[(lg.line == line) & (lg.gain == "label") & (lg.family == fam) & (lg.ds == ds)].iloc[0]
                    ax.text(x, ax.get_ylim()[1] if False else max(g.max(), 0) + 0.004, f"p {r.p_vs_dff:.2g}",
                            ha="center", fontsize=5, rotation=90)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(LINES)))
        ax.set_xticklabels([f"{l}\n({REPORTER[l]})" for l in LINES], fontsize=7)
        ax.set_title(f"{fam}: label gain per mouse", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("label gain (score)", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper left")


def confusion(ax, conf, ds, title):
    c = conf[(conf.ds == ds) & (conf.clf == "sel") & (conf.level == "neuron")]
    M = c.pivot(index="true", columns="pred", values="count").loc[LINES, LINES].to_numpy()
    R = M / M.sum(1, keepdims=True)
    ax.imshow(R, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(LINES)):
        for j in range(len(LINES)):
            ax.text(j, i, f"{M[i, j]}\n{R[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="w" if R[i, j] > 0.6 else "k")
    ax.set_xticks(range(len(LINES)))
    ax.set_yticks(range(len(LINES)))
    ax.set_xticklabels(LINES, fontsize=7)
    ax.set_yticklabels(LINES, fontsize=7)
    ax.set_xlabel("predicted", fontsize=7)
    ax.set_ylabel("true", fontsize=7)
    ax.set_title(title, fontsize=8)


def main():
    ex, gains, mice = read("example_traces.csv"), read("gains.csv"), read("mouse_scores.csv")
    lg, conf, cs = read("line_gains.csv"), read("confusions.csv"), read("classifier_summary.csv")
    surv, sp, op = read("survival.csv"), read("sparsity.csv"), read("oasis_params.csv")
    fe, sep = read("features_by_line.csv"), read("feature_separability.csv")

    # ---- main figure
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)
    panel_traces(fig, gs[0, 0], ex)
    fig.text(0.125, 0.905, "(a) example traces, first 60 s of the test part: dF/F vs OASIS-deconvolved",
             fontsize=9)
    panel_gains(fig.add_subplot(gs[0, 1]), gains)
    sub = gs[1, 0].subgridspec(1, 2, wspace=0.25)
    axl = [fig.add_subplot(sub[0]), fig.add_subplot(sub[1])]
    panel_lines(axl, mice, lg)
    axl[0].text(0, 1.1, "(c) genetic-label gain per Cre line, dF/F vs deconvolved (p: Wilcoxon vs dF/F over the"
                " line's mice)", transform=axl[0].transAxes, fontsize=9)
    sub = gs[1, 1].subgridspec(1, 2, wspace=0.35)
    r = cs[(cs.ds == "A") & (cs.clf == "sel")].iloc[0]
    r0 = cs[(cs.ds == "dff") & (cs.clf == "sel")].iloc[0]
    confusion(fig.add_subplot(sub[0]), conf, "A", f"deconvolved A: bal. acc {r.bacc_neuron:.3f},\n"
                                                  f"sessions {r.session_correct}/{r.n_sessions}")
    rb = cs[(cs.ds == "B") & (cs.clf == "sel")].iloc[0]
    confusion(fig.add_subplot(sub[1]), conf, "B", f"deconvolved B: bal. acc {rb.bacc_neuron:.3f},\n"
                                                  f"sessions {rb.session_correct}/{rb.n_sessions}")
    fig.text(0.53, 0.47, f"(d) step 11 S classifier (single-neuron features) after deconvolution; held-out mice"
                         f" (dF/F: bal. acc {r0.bacc_neuron:.3f})", fontsize=9)
    fig.suptitle("Step 15: does the signal type (dF/F vs deconvolved) explain the small cell-type label gain in"
                 " Bugeon?", fontsize=11)
    fig.savefig(ROOT / "plots/step15_signal.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- per-mouse gains, deconvolved vs dF/F
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5))
    for col, (gname, fam, a, b) in enumerate([("label", "ridge", "r_true", "r_none"), ("label", "MLP", "m_true", "m_none"),
                                              ("summary", "ridge", "r_km16", "r_none"),
                                              ("summary", "MLP", "m_cont", "m_none")]):
        m0 = mice[mice.ds == "dff"].set_index("id")
        for row, ds in enumerate(["A", "B"]):
            ax = axes[row, col]
            mx = mice[mice.ds == ds].set_index("id").loc[m0.index]
            g0, gx = m0[a] - m0[b], mx[a] - mx[b]
            for line in LINES:
                s = m0.line == line
                ax.scatter(g0[s], gx[s], s=14, color=LCOL[line], label=line)
            lim = [min(g0.min(), gx.min(), 0) - 0.005, max(g0.max(), gx.max()) + 0.005]
            ax.plot(lim, lim, "k--", lw=0.6)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            s_ = surv[(surv.gain == gname) & (surv.family == fam) & (surv.ds == ds)].iloc[0]
            ax.set_title(f"{gname} gain, {fam}: {DLAB[ds].splitlines()[0]}\nsurviving {s_.frac_surviving:.2f}"
                         f" [{s_.ci_lo:.2f}, {s_.ci_hi:.2f}], Wilcoxon p {s_.p_wilcoxon:.2g}", fontsize=8)
            ax.set_xlabel("gain on dF/F (per mouse)", fontsize=7)
            ax.set_ylabel(f"gain on deconvolved {ds}", fontsize=7)
            ax.tick_params(labelsize=7)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Step 15: per-mouse gains on deconvolved vs dF/F traces (46 mice; dashed = equal gain)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step15_mice.png", dpi=150)
    plt.close(fig)

    # ---- typing
    fig = plt.figure(figsize=(17, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.35)
    for k, ds in enumerate(["dff", "A", "B"]):
        r = cs[(cs.ds == ds) & (cs.clf == "sel")].iloc[0]
        confusion(fig.add_subplot(gs[0, k]), conf, ds, f"{DLAB[ds].replace(chr(10), ' ')}: bal. acc {r.bacc_neuron:.3f}"
                                                        f"\nsessions {r.session_correct}/{r.n_sessions} ({r.selected})")
    ax = fig.add_subplot(gs[0, 3])
    for j, ds in enumerate(["dff", "A", "B"]):
        r = cs[(cs.ds == ds) & (cs.clf == "sel")].iloc[0]
        ax.bar(np.arange(len(LINES)) + (j - 1) * 0.27, [r[f"recall_{l}"] for l in LINES], 0.25, color=DCOL[ds],
               label=DLAB[ds].replace("\n", " "))
    ax.axhline(0.25, color="k", ls=":", lw=0.6)
    ax.set_xticks(range(len(LINES)))
    ax.set_xticklabels(LINES, fontsize=7)
    ax.set_ylabel("recall (held-out neurons)", fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.legend(fontsize=6, loc="upper center", ncol=2)
    ax.set_title("per-line recall, selected classifier", fontsize=8)
    s0 = sep[sep.ds == "dff"].set_index("feature")
    sA = sep[sep.ds == "A"].set_index("feature").loc[s0.index]
    for k, (col, lab) in enumerate([("eps2", "Kruskal-Wallis $\\epsilon^2$ over the 4 lines"),
                                    ("auc_Pvalb_vs_rest", "AUC Pvalb vs rest")]):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        v0, vA = s0[col], sA[col]
        if col.startswith("auc"):
            v0, vA = (v0 - 0.5).abs() + 0.5, (vA - 0.5).abs() + 0.5
        order = (v0 - vA).sort_values(ascending=False).index
        x = np.arange(len(order))
        ax.bar(x - 0.2, v0[order], 0.4, color=DCOL["dff"], label="dF/F")
        ax.bar(x + 0.2, vA[order], 0.4, color=DCOL["A"], label="deconvolved A")
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=90, fontsize=6)
        ax.set_ylabel(lab + (" (folded: max(AUC, 1 - AUC))" if col.startswith("auc") else ""), fontsize=7)
        ax.set_title(f"per-feature {lab}, sorted by the drop from dF/F to deconvolved A", fontsize=8)
        ax.legend(fontsize=7)
        if col.startswith("auc"):
            ax.set_ylim(0.5, 1)
    fig.suptitle("Step 15: single-neuron S classifier (step 11) on dF/F vs deconvolved Allen traces, 46 held-out mice",
                 fontsize=10)
    fig.savefig(ROOT / "plots/step15_typing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- sparsity and decay constants
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    al = sp[(sp.dataset == "Allen deconvolved") & (sp.group != "all")].set_index("group").loc[LINES]
    bg = sp[(sp.dataset == "Bugeon frame.neuralActivity") & (sp.group != "all")]
    x = np.arange(len(LINES))
    for j, (c, lab) in enumerate([("zeros_30Hz_train", "Allen OASIS, 30 Hz frames"),
                                  ("zeros_10Hz_train", "Allen A, 10 Hz bins"), ("zeros_4.3Hz_train", "Allen B, 4.3 Hz bins")]):
        ax.bar(x + (j - 1) * 0.27, al[c], 0.25, label=lab, color=["0.75", DCOL["A"], DCOL["B"]][j])
    xb = len(LINES) + 0.5 + np.arange(len(bg))
    ax.bar(xb, bg["zeros_4.3Hz_train"], 0.6, color=DCOL["bugeon"], label="Bugeon, 4.3 Hz frames")
    ax.set_xticks(list(x) + list(xb))
    ax.set_xticklabels(LINES + list(bg.group), rotation=45, fontsize=7)
    ax.set_ylabel("fraction of exact zeros (train part, mean over neurons)", fontsize=7)
    ax.legend(fontsize=6, loc="lower right")
    ax.set_ylim(0, 1)
    bss = sp[sp.dataset == "Bugeon frame.neuralActivity, per session"]
    ax.text(xb.mean(), 0.93, f"Bugeon per session: {bss['zeros_4.3Hz_train'].min():.2f}-"
            f"{bss['zeros_4.3Hz_train'].max():.2f}\n(SB025 first 4 sessions ~0.79)", ha="center", fontsize=6.5)
    ax.set_title("sparsity of the deconvolved signal", fontsize=9)
    ax = axes[1]
    for line in LINES:
        v = op[op.line == line].tau_s
        ax.hist(np.clip(v, 0, 3), bins=np.linspace(0, 3, 41), histtype="step", color=LCOL[line], density=True,
                label=f"{line} ({REPORTER[line]}): median {v.median():.2f} s")
    ax.set_xlabel("OASIS AR(1) decay time constant (s), train part", fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("estimated indicator decay per neuron", fontsize=9)
    ax = axes[2]
    for line in LINES:
        v = op[op.line == line]
        ax.scatter(v["zeros_4.3Hz_train"], v.tau_s.clip(0, 3), s=3, color=LCOL[line], alpha=0.5, label=line)
    ax.set_xlabel("fraction of zero 4.3 Hz bins (B)", fontsize=8)
    ax.set_ylabel("decay time constant (s)", fontsize=8)
    ax.legend(fontsize=7, markerscale=3)
    ax.set_title("sparsity vs decay, per neuron", fontsize=9)
    fig.suptitle("Step 15: OASIS deconvolution of Allen dF/F (AR(1), L1, parameters from the train part)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step15_sparsity.png", dpi=150)
    plt.close(fig)

    # ---- per-line summary gain and controls
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for ax, (gname, fam) in zip(axes, [("summary", "ridge"), ("summary", "MLP"), ("shuffled label", "ridge"),
                                       ("shuffled label", "MLP")]):
        for j, ds in enumerate(["dff", "A", "B"]):
            v = lg[(lg.gain == gname) & (lg.family == fam) & (lg.ds == ds)].set_index("line").loc[LINES]
            ax.bar(np.arange(len(LINES)) + (j - 1) * 0.27, v.gain_mean, 0.25, yerr=v.gain_sem, color=DCOL[ds],
                   capsize=2, label=DLAB[ds].replace("\n", " "))
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xticks(range(len(LINES)))
        ax.set_xticklabels(LINES, fontsize=7)
        ax.set_title(f"{gname} gain, {fam} (mean ± sem over mice)", fontsize=8)
    axes[0].legend(fontsize=6)
    axes[0].set_ylabel("score gain over no label", fontsize=8)
    fig.suptitle("Step 15: per-line activity-summary gain and shuffled-label control, dF/F vs deconvolved", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step15_lines.png", dpi=150)
    plt.close(fig)
    print("Saved plots/step15_signal.png, step15_mice.png, step15_typing.png, step15_sparsity.png, step15_lines.png")


if __name__ == "__main__":
    main()
