"""Step 6: does the Cre-line label help the shared model?

Same 9 sessions (VISp, three_session_A, Cux2-CreERT2 / Sst-IRES-Cre / Vip-IRES-Cre,
3 mice per line), preprocessing, stimulus features and batched X'X fitting as
step 5 (scripts/step5_shared.py). Inputs per neuron: own past 30 bins + 480
stimulus inputs -> next 10 bins.

Models (all shared across neurons, all with stimulus):
- (a) ridge per neuron + stim (step 5 reference row)
- (b) shared, no label (step 5 b)
- (c) shared + Cre one-hot (3 extra inputs: only an offset per line). The one-hot
  is constant within a session, so its Gram terms come from the session's
  n, sum x and sum y; one alpha by pooled val MSE, then refit on train + val
- (d) Cre interaction = one shared model per Cre line (Cre one-hot x all inputs,
  incl. the intercept). Each line's model is step 5 b fit on that line's
  sessions only, with its own alpha picked on that line's val windows
- (e) shuffled control for d: the training sessions are split into 3 random
  groups of equal size, each group mixing lines (no group is all one line);
  d is fit with these fake labels. Within-session, each session is tested with
  its own fake group; held-out mice get a random group. 10 seeds, mean +- std

Tests (test windows = last 20% of each session, same as step 5):
- held-out mouse (main): fit on mice 1-2 per line (6 sessions), test on the 3rd
- within-session: fit on the train parts of all 9 sessions
Decision: the label helps if d beats b AND d beats e (mean over seeds) in
held-out-mouse test MSE.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_linear import ALPHAS, CACHE, FUTURE, PAST, ROOT
from step5_shared import (CRE_LINES, Gram, fit_shared, neuron_r, pick_sessions,
                          predict, process_session)

warnings.filterwarnings("ignore")

N_SEEDS = 10
N_GROUPS = len(CRE_LINES)
MAIN = ["copy-last", "a", "b", "c", "d"]
NAMES = {"copy-last": "copy-last", "a": "(a) ridge per neuron + stim (ref)",
         "b": "(b) shared, no label", "c": "(c) shared + Cre one-hot",
         "d": "(d) shared per Cre line", "e": "(e) shuffled-label control"}


def onehot(k):
    e = np.zeros(N_GROUPS)
    e[k] = 1.0
    return e


def add_onehot(g, e):
    """Gram of the design [D, e] where e (one-hot) is constant over the rows of g."""
    p = len(g.sx)
    xx = np.empty((p + len(e),) * 2)
    xx[:p, :p] = g.xx
    xx[:p, p:] = np.outer(g.sx, e)
    xx[p:, :p] = xx[:p, p:].T
    xx[p:, p:] = g.n * np.outer(e, e)
    return Gram(g.n, np.concatenate([g.sx, g.n * e]), g.sy, xx,
                np.vstack([g.xy, np.outer(e, g.sy)]))


def predict_c(W, b, X, S, e):
    return predict(W[:-N_GROUPS], b + e @ W[-N_GROUPS:], X, S)


def fit_onehot(sessions, labels):
    """Model c: shared ridge + one-hot label per session. Alpha by pooled val
    MSE, then refit on train + val (as step5_shared.fit_shared)."""
    E = [onehot(labels[s["info"]["id"]]) for s in sessions]
    coef = sum((add_onehot(s["gfit"], e) for s, e in zip(sessions, E)), Gram()).solver()
    n_val = sum(s["Yv"].size for s in sessions)
    val_mse = []
    for a in ALPHAS:
        W, b = coef(a)
        val_mse.append(sum(((predict_c(W, b, s["Xv"], s["Sv"], e) - s["Yv"]) ** 2).sum()
                           for s, e in zip(sessions, E)) / n_val)
    alpha = ALPHAS[int(np.argmin(val_mse))]
    W, b = sum((add_onehot(s["gfit"] + s["gval"], e) for s, e in zip(sessions, E)),
               Gram()).solver()(alpha)
    return W, b, alpha


def fit_grouped(sessions, labels):
    """Models d / e: one step 5 b model per group. Returns {group: (W, b, alpha)}."""
    out = {}
    for k in range(N_GROUPS):
        W, b, alpha, _, _ = fit_shared([s for s in sessions if labels[s["info"]["id"]] == k])
        out[k] = (W, b, alpha)
    return out


def shuffled_groups(train, rng):
    """Random equal-size groups of the training sessions; redraw until every group
    mixes Cre lines, so no fake group equals (or sits inside) a true line."""
    while True:
        perm = rng.permutation(len(train))
        groups = np.array_split(perm, N_GROUPS)
        if all(len({train[i]["info"]["cre"] for i in g}) > 1 for g in groups):
            return {train[i]["info"]["id"]: k for k, g in enumerate(groups) for i in g}


def pooled(rows, model):
    """As step 5: MSE over all neurons x windows x steps, mean per-neuron r,
    score vs copy-last on the same elements."""
    se = sum(r["sse"][model] for r in rows)
    n = sum(r["n_el"] for r in rows)
    rs = np.concatenate([r["r_neuron"][model] for r in rows])
    return dict(mse=se / n, r=float(np.nanmean(rs)),
                score=1 - se / sum(r["sse"]["copy-last"] for r in rows))


def main():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    infos = pick_sessions(api, exps)
    sessions = []
    for info in infos:
        s = process_session(info)
        sessions.append(s)
        print(f"{info['cre']:13s} exp {info['id']} mouse {info['mouse']} | {s['n_neurons']} neurons"
              f" | windows train/val/test {s['n_train_win']}/{s['n_val_win']}/{len(s['Xte'])}"
              f"{' | HELD-OUT' if info['held_out'] else ''}", flush=True)
    true = {s["info"]["id"]: CRE_LINES.index(s["info"]["cre"]) for s in sessions}
    held = [s for s in sessions if s["info"]["held_out"]]
    train6 = [s for s in sessions if not s["info"]["held_out"]]

    # Fits: "ho" = trained on 6 sessions (held-out test), "all" = all 9 (within-session)
    fits = {}
    for tag, tr in [("all", sessions), ("ho", train6)]:
        W, b, alpha, _, _ = fit_shared(tr)
        fits[tag, "b"] = (W, b, alpha)
        fits[tag, "c"] = fit_onehot(tr, true)
        fits[tag, "d"] = fit_grouped(tr, true)
        print(f"\nfit on {len(tr)} sessions: alpha b {fits[tag, 'b'][2]:g}, c {fits[tag, 'c'][2]:g},"
              f" d " + ", ".join(f"{CRE_LINES[k].split('-')[0]} {v[2]:g}"
                                 for k, v in fits[tag, "d"].items()))
        print(f"    c one-hot weights on step 1 (Cux2, Sst, Vip):"
              f" {np.round(fits[tag, 'c'][0][-N_GROUPS:, 0], 3)}")

    # Shuffled control: per seed, fake groups for the training sessions and a
    # random fake group for each held-out session
    shuffles = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        lab_all = shuffled_groups(sessions, rng)
        lab_ho = shuffled_groups(train6, rng)
        for s in held:
            lab_ho[s["info"]["id"]] = int(rng.integers(N_GROUPS))
        shuffles.append(dict(lab_all=lab_all, lab_ho=lab_ho,
                             fit_all=fit_grouped(sessions, lab_all),
                             fit_ho=fit_grouped(train6, lab_ho)))
        groups = lambda lab, ss: [[f"{s['info']['cre'].split('-')[0]}{s['info']['id'] % 1000}"
                                   for s in ss if lab[s["info"]["id"]] == k] for k in range(N_GROUPS)]
        print(f"seed {seed}: groups (6) {groups(lab_ho, train6)} held-out -> "
              f"{[lab_ho[s['info']['id']] for s in held]} | alphas "
              f"{[v[2] for v in shuffles[-1]['fit_ho'].values()]}", flush=True)

    # Test predictions and metrics
    rows = {"all": [], "ho": []}
    for s in sessions:
        X, Y, S, sid = s["Xte"], s["Yte"], s["Ste"], s["info"]["id"]
        for tag in ["all", "ho"]:
            if tag == "ho" and not s["info"]["held_out"]:
                continue
            Wb, bb, _ = fits[tag, "b"]
            Wc, bc, _ = fits[tag, "c"]
            Wd, bd, _ = fits[tag, "d"][true[sid]]
            P = {"copy-last": np.repeat(X[:, :, -1:], FUTURE, axis=2), "a": s["Pa"],
                 "b": predict(Wb, bb, X, S), "c": predict_c(Wc, bc, X, S, onehot(true[sid])),
                 "d": predict(Wd, bd, X, S)}
            for k, sh in enumerate(shuffles):
                We, be, _ = sh[f"fit_{tag}"][sh[f"lab_{tag}"][sid]]
                P[f"e{k}"] = predict(We, be, X, S)
            rows[tag].append(dict(info=s["info"], n_neurons=s["n_neurons"], n_el=Y.size,
                                  sse={m: float(((p - Y) ** 2).sum()) for m, p in P.items()},
                                  r_neuron={m: neuron_r(Y, p) for m, p in P.items()}))
            if tag == "ho":
                s["P"] = {m: P[m] for m in ["b", "d"]}

    results = {}
    for tag, title in [("ho", "Held-out mouse (3rd mouse per line) - MAIN TEST"),
                       ("within", "Within-session test (all 9 sessions)")]:
        rs_tag = rows["ho" if tag == "ho" else "all"]
        print(f"\n== {title} ==")
        for cre in CRE_LINES + ["all lines"]:
            rs = [r for r in rs_tag if cre == "all lines" or r["info"]["cre"] == cre]
            nn = sum(r["n_neurons"] for r in rs)
            print(f"{cre} ({len(rs)} sessions, {nn} neurons)")
            res = {m: pooled(rs, m) for m in MAIN}
            ev = [pooled(rs, f"e{k}") for k in range(N_SEEDS)]
            for m in MAIN:
                v = res[m]
                print(f"    {NAMES[m]:36s} MSE {v['mse']:.4f}  r {v['r']:.3f}  score {v['score']:+.3f}")
            agg = {q: (np.mean([v[q] for v in ev]), np.std([v[q] for v in ev])) for q in ["mse", "r", "score"]}
            print(f"    {NAMES['e']:36s} MSE {agg['mse'][0]:.4f}+-{agg['mse'][1]:.4f}"
                  f"  r {agg['r'][0]:.3f}+-{agg['r'][1]:.3f}  score {agg['score'][0]:+.3f}+-{agg['score'][1]:.3f}")
            print(f"    e per seed MSE: {np.round([v['mse'] for v in ev], 4)}")
            n_d_beats_e = sum(res["d"]["mse"] < v["mse"] for v in ev)
            helps = res["d"]["mse"] < res["b"]["mse"] and res["d"]["mse"] < agg["mse"][0]
            print(f"    d - b MSE {res['d']['mse'] - res['b']['mse']:+.4f}, d - e(mean) MSE "
                  f"{res['d']['mse'] - agg['mse'][0]:+.4f}, d beats e in {n_d_beats_e}/{N_SEEDS} seeds"
                  f" -> label helps: {'YES' if helps else 'NO'}")
            results[tag, cre] = (res, agg, n_d_beats_e, helps)

    # Per session, held-out
    print("\n== Held-out sessions: per-neuron test MSE d - b ==")
    for s in held:
        Y = s["Yte"]
        dm = ((s["P"]["d"] - Y) ** 2).mean(axis=(0, 2)) - ((s["P"]["b"] - Y) ** 2).mean(axis=(0, 2))
        print(f"    {s['info']['cre']:13s} d better for {(dm < 0).sum()}/{len(dm)} neurons,"
              f" median {np.median(dm):+.4f}")

    # Weights of the per-line models (held-out fit): own past last bins and intercept, step 1
    for k, (W, b, _) in fits["ho", "d"].items():
        print(f"d {CRE_LINES[k]:13s} own-past weights (last 5 bins -> step 1) "
              f"{np.round(W[PAST - 5:PAST, 0], 3)}, intercept step 1 {b[0]:+.3f}")

    # Plot: same held-out neurons as step 5 (same seeded pick), b, d and true trace
    rng = np.random.default_rng(0)
    colors = {"b": "tab:green", "d": "tab:orange"}
    fig, axes = plt.subplots(6, 1, figsize=(11, 14))
    ax_i = 0
    plotted = {}
    for s in held:
        zte, S, starts = s["zte"], s["Ste_all"], s["te_starts"]
        tt = np.arange(zte.shape[1]) / s["bin_hz"]
        t0 = tt[np.argmax(S[0] > 0.5)] - 5
        ws = np.where((tt[starts] >= t0) & (tt[starts] + 1 <= t0 + 30))[0]
        lo, hi = starts[ws[0]] - PAST, starts[ws[-1]] + FUTURE
        neurons = np.sort(rng.choice(s["n_neurons"], size=2, replace=False))
        plotted[s["info"]["id"]] = neurons.tolist()
        for n in neurons:
            ax = axes[ax_i]; ax_i += 1
            ax.fill_between(tt[lo:hi], 0, 1, where=S[3, lo:hi] > 0.5,
                            transform=ax.get_xaxis_transform(), color="0.85", step="mid",
                            label="grating on")
            ax.plot(tt[lo:hi], zte[n, lo:hi], color="k", lw=0.9, label="true")
            for m, c in colors.items():
                for k, w in enumerate(ws):
                    st = starts[w]
                    ax.plot(tt[st:st + FUTURE], s["P"][m][w, n], color=c, lw=1.4,
                            label=NAMES[m] if k == 0 else None)
            ax.set_title(f"{s['info']['cre']} exp {s['info']['id']} (held-out mouse "
                         f"{s['info']['mouse']}) | neuron {n}", fontsize=9, loc="left")
            ax.set_ylabel("z dF/F")
    axes[0].legend(fontsize=8, loc="upper right", ncol=4)
    axes[-1].set_xlabel("Time from test start (s)")
    fig.suptitle("Step 6: shared ridge without label (b) vs one shared model per Cre line (d),"
                 " held-out mice (test set)")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step6_celltype.png", dpi=150)
    print("\nPlotted neurons:", plotted)
    print("Saved plots/step6_celltype.png")


if __name__ == "__main__":
    main()
