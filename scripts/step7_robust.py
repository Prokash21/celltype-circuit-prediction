"""Step 7: is the step 6 result robust? Leave-one-mouse-out over 5 mice per line.

Sessions: VISp, three_session_A, Cux2-CreERT2 / Sst-IRES-Cre / Vip-IRES-Cre;
lowest experiment id per mouse, then the 5 mice with the lowest such ids per
line (15 sessions). Preprocessing, stimulus features, model a and the batched
X'X fitting are those of step 5 (scripts/step5_shared.py); models as in step 6
(scripts/step6_celltype.py). Inputs per neuron: own past 30 bins + 480 stimulus
inputs -> next 10 bins.

Folds: fold k holds out the k-th mouse (by experiment id) of every line, so each
fold trains on the other 4 mice of every line (12 sessions) and tests on 3
held-out sessions (one per line). Test windows = last 20% of each held-out
session, as in steps 5-6.

Models (all with stimulus):
- (a) ridge per neuron + stim, fit on the held-out session's own train part
  (reference only; same in every fold)
- (b) shared, no label, line-balanced: every row of a session is weighted by
  w_line = N / (n_lines * N_line) (N = training rows), so each Cre line
  contributes the same total weight to X'X, X'Y and to the val MSE used to
  pick alpha. Refit on train + val with the same weights
- (d) one shared model per Cre line (step 6 d; one line per model, so no
  weighting needed)
- (e) shuffled control for d: the 12 training sessions are split into 3 random
  groups of 4, each mixing lines; held-out sessions get a random group. Each
  group's model is fit line-balanced as b (within the group). 5 seeds per fold

Summary per line: mean +- std over the 5 folds, number of folds where d beats
b and d beats e (mean over seeds) in held-out MSE, and a paired two-sided
Wilcoxon signed-rank test over folds (d vs b, d vs e). Also over all 15
line-folds pooled.

Data files are read from the allensdk cache layout under data/brain_observatory/,
fetched from the Allen public S3 mirror (see scripts/step2_baselines.py).
"""
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_linear import ALPHAS, CACHE, FUTURE, ROOT
from step5_shared import CRE_LINES, Gram, neuron_r, predict, process_session
from step6_celltype import N_GROUPS, fit_grouped, pooled, shuffled_groups

warnings.filterwarnings("ignore")

N_MICE = 5
N_SEEDS = 5
MODELS = ["copy-last", "a", "b", "d"]
NAMES = {"copy-last": "copy-last", "a": "(a) ridge per neuron + stim (ref)",
         "b": "(b) shared, no label, balanced", "d": "(d) shared per Cre line",
         "e": "(e) shuffled-label control, balanced"}


def pick_sessions(api, exps):
    """Per Cre line: lowest id per mouse, then the N_MICE lowest of those."""
    out = []
    for cre in CRE_LINES:
        es = api.filter_ophys_experiments(exps, cre_lines=[cre], targeted_structures=["VISp"],
                                          session_types=["three_session_A"])
        first = {}
        for e in sorted(es, key=lambda e: e["id"]):
            first.setdefault(e["specimen_id"], e)
        for k, e in enumerate(sorted(first.values(), key=lambda e: e["id"])[:N_MICE]):
            out.append(dict(cre=cre, id=e["id"], mouse=e["specimen"]["external_specimen_name"],
                            depth=e["imaging_depth"], fold=k, held_out=False))
    return out


def scaled(g, w):
    return Gram(w * g.n, w * g.sx, w * g.sy, w * g.xx, w * g.xy)


def line_weights(sessions):
    """Per-session row weight so that each Cre line present gets equal total weight."""
    n_line = {}
    for s in sessions:
        n_line[s["info"]["cre"]] = n_line.get(s["info"]["cre"], 0) + s["gfit"].n + s["gval"].n
    total = sum(n_line.values())
    return [total / (len(n_line) * n_line[s["info"]["cre"]]) for s in sessions]


def fit_balanced(sessions):
    """Step 5 b with line-balanced row weights: alpha by weighted val MSE, then
    refit on train + val with the same weights."""
    w = line_weights(sessions)
    coef = sum((scaled(s["gfit"], ws) for s, ws in zip(sessions, w)), Gram()).solver()
    n_val = sum(ws * s["Yv"].size for s, ws in zip(sessions, w))
    val_mse = []
    for a in ALPHAS:
        W, b = coef(a)
        val_mse.append(sum(ws * ((predict(W, b, s["Xv"], s["Sv"]) - s["Yv"]) ** 2).sum()
                           for s, ws in zip(sessions, w)) / n_val)
    alpha = ALPHAS[int(np.argmin(val_mse))]
    W, b = sum((scaled(s["gfit"] + s["gval"], ws) for s, ws in zip(sessions, w)),
               Gram()).solver()(alpha)
    return W, b, alpha


def fit_grouped_balanced(sessions, labels):
    return {k: fit_balanced([s for s in sessions if labels[s["info"]["id"]] == k])
            for k in range(N_GROUPS)}


def row(s, P):
    Y = s["Yte"]
    return dict(info=s["info"], n_neurons=s["n_neurons"], n_el=Y.size,
                sse={m: float(((p - Y) ** 2).sum()) for m, p in P.items()},
                r_neuron={m: neuron_r(Y, p) for m, p in P.items()})


def summarize(rows):
    """rows of one or more held-out sessions -> metrics for each model, e as mean/std."""
    res = {m: pooled(rows, m) for m in MODELS}
    ev = [pooled(rows, f"e{k}") for k in range(N_SEEDS)]
    res["e"] = {q: float(np.mean([v[q] for v in ev])) for q in ["mse", "r", "score"]}
    res["e_std"] = {q: float(np.std([v[q] for v in ev])) for q in ["mse", "r", "score"]}
    res["e_seeds"] = [v["mse"] for v in ev]
    return res


def main():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    infos = pick_sessions(api, exps)
    sessions = []
    for info in infos:
        s = process_session(info)
        s.pop("zte"); s.pop("Ste_all"); s.pop("te_starts")
        sessions.append(s)
        print(f"{info['cre']:13s} fold {info['fold']} exp {info['id']} mouse {info['mouse']}"
              f" depth {info['depth']} | {s['n_neurons']} neurons | windows train/val/test"
              f" {s['n_train_win']}/{s['n_val_win']}/{len(s['Xte'])}", flush=True)
    true = {s["info"]["id"]: CRE_LINES.index(s["info"]["cre"]) for s in sessions}

    per_fold = {}  # (fold, cre or "all lines") -> summary
    for k in range(N_MICE):
        held = [s for s in sessions if s["info"]["fold"] == k]
        train = [s for s in sessions if s["info"]["fold"] != k]
        Wb, bb, alpha_b = fit_balanced(train)
        fd = fit_grouped(train, true)
        w = line_weights(train)
        print(f"\n== Fold {k}: held out {[s['info']['id'] for s in held]} ==")
        print(f"    line weights (Cux2, Sst, Vip): "
              f"{[round(w[[s['info']['cre'] for s in train].index(c)], 3) for c in CRE_LINES]}"
              f" | alpha b {alpha_b:g}, d " + ", ".join(f"{CRE_LINES[g].split('-')[0]} {v[2]:g}"
                                                         for g, v in fd.items()))
        shuffles = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            lab = shuffled_groups(train, rng)
            for s in held:
                lab[s["info"]["id"]] = int(rng.integers(N_GROUPS))
            shuffles.append((lab, fit_grouped_balanced(train, lab)))
            print(f"    e seed {seed}: groups "
                  f"{[[s['info']['cre'][0] + str(s['info']['fold']) for s in train if lab[s['info']['id']] == g] for g in range(N_GROUPS)]}"
                  f" held-out -> {[lab[s['info']['id']] for s in held]}"
                  f" | alphas {[v[2] for v in shuffles[-1][1].values()]}", flush=True)

        rows = []
        for s in held:
            X, S, sid = s["Xte"], s["Ste"], s["info"]["id"]
            Wd, bd, _ = fd[true[sid]]
            P = {"copy-last": np.repeat(X[:, :, -1:], FUTURE, axis=2), "a": s["Pa"],
                 "b": predict(Wb, bb, X, S), "d": predict(Wd, bd, X, S)}
            for j, (lab, fe) in enumerate(shuffles):
                We, be, _ = fe[lab[sid]]
                P[f"e{j}"] = predict(We, be, X, S)
            rows.append(row(s, P))
        for cre in CRE_LINES + ["all lines"]:
            rs = [r for r in rows if cre == "all lines" or r["info"]["cre"] == cre]
            res = summarize(rs)
            per_fold[k, cre] = res
            ids = ",".join(str(r["info"]["id"]) for r in rs)
            print(f"  {cre} ({ids}, {sum(r['n_neurons'] for r in rs)} neurons)")
            for m in MODELS:
                v = res[m]
                print(f"    {NAMES[m]:38s} MSE {v['mse']:.4f}  r {v['r']:.3f}  score {v['score']:+.3f}")
            v, sd = res["e"], res["e_std"]
            print(f"    {NAMES['e']:38s} MSE {v['mse']:.4f}+-{sd['mse']:.4f}  r {v['r']:.3f}+-{sd['r']:.3f}"
                  f"  score {v['score']:+.3f}+-{sd['score']:.3f}  (seeds {np.round(res['e_seeds'], 4)})")
            print(f"    d - b {res['d']['mse'] - res['b']['mse']:+.4f}, d - e {res['d']['mse'] - v['mse']:+.4f}")

    # Summary over folds
    print("\n== Summary over 5 folds (held-out mouse test) ==")
    summary = {}
    for cre in CRE_LINES + ["all lines"]:
        print(f"{cre}")
        for m in MODELS + ["e"]:
            vals = {q: np.array([per_fold[k, cre][m][q] for k in range(N_MICE)]) for q in ["mse", "r", "score"]}
            print(f"    {NAMES[m]:38s} MSE {vals['mse'].mean():.4f}+-{vals['mse'].std():.4f}"
                  f"  r {vals['r'].mean():.3f}+-{vals['r'].std():.3f}"
                  f"  score {vals['score'].mean():+.3f}+-{vals['score'].std():.3f}")
        d = np.array([per_fold[k, cre]["d"]["mse"] for k in range(N_MICE)])
        b = np.array([per_fold[k, cre]["b"]["mse"] for k in range(N_MICE)])
        e = np.array([per_fold[k, cre]["e"]["mse"] for k in range(N_MICE)])
        p_b, p_e = wilcoxon(d, b).pvalue, wilcoxon(d, e).pvalue
        summary[cre] = dict(db=d - b, de=d - e)
        print(f"    d - b per fold {np.round(d - b, 4)}; d beats b in {(d < b).sum()}/{N_MICE},"
              f" Wilcoxon p {p_b:.4f}")
        print(f"    d - e per fold {np.round(d - e, 4)}; d beats e in {(d < e).sum()}/{N_MICE},"
              f" Wilcoxon p {p_e:.4f}")

    # All 15 line-folds as pairs
    d = np.concatenate([[per_fold[k, c]["d"]["mse"] for k in range(N_MICE)] for c in CRE_LINES])
    b = np.concatenate([[per_fold[k, c]["b"]["mse"] for k in range(N_MICE)] for c in CRE_LINES])
    e = np.concatenate([[per_fold[k, c]["e"]["mse"] for k in range(N_MICE)] for c in CRE_LINES])
    print(f"\nAll 15 line-folds as pairs: d beats b in {(d < b).sum()}/15 (Wilcoxon p "
          f"{wilcoxon(d, b).pvalue:.4f}), d beats e in {(d < e).sum()}/15 (p {wilcoxon(d, e).pvalue:.4f})")

    # Plot: per line, d - b held-out MSE per fold
    ids = {(s["info"]["cre"], s["info"]["fold"]): s["info"]["id"] for s in sessions}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, cre in zip(axes, CRE_LINES):
        db = summary[cre]["db"]
        ax.bar(range(N_MICE), db, color=["tab:blue" if v < 0 else "tab:red" for v in db])
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(N_MICE))
        ax.set_xticklabels([f"fold {k}\n{ids[cre, k]}" for k in range(N_MICE)], fontsize=7, rotation=30)
        ax.set_title(f"{cre}: d beats b in {(db < 0).sum()}/{N_MICE}", fontsize=10)
        ax.set_ylabel("held-out MSE, d - b")
    fig.suptitle("Step 7: per-line model (d) minus shared balanced model (b), held-out mouse test MSE"
                 " (below 0 = d better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step7_robust.png", dpi=150)
    print("Saved plots/step7_robust.png")


if __name__ == "__main__":
    main()
