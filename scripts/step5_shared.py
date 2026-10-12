"""Step 5: one shared model for all neurons of several sessions (no cell-type label).

Sessions: VISp, three_session_A, Cre lines Cux2-CreERT2, Sst-IRES-Cre and
Vip-IRES-Cre; per line the 3 lowest experiment ids from 3 different mice
(lowest id per mouse, then the 3 mice with the lowest such ids).

Preprocessing as in steps 3-4: first 80% train (last 10% of train = val),
last 20% test, z-score with train stats, 10 Hz bins, past 3 s -> next 1 s,
12 stimulus features for the past 3 s and the next 1 s (scripts/step4_stimulus.py).
Every model sees the same inputs for a neuron: its own past 30 bins + 480
stimulus inputs (510 inputs), and predicts its next 10 bins.

Models:
- ridge-neuron+stim (a): one ridge per neuron, alpha picked per neuron on val,
  then refit on train + val.
- ridge-shared+stim (b): ONE ridge for all neurons of all training sessions
  (neurons are pooled as samples; no neuron or Cre-line identity in the input).
  Fit from X'X and X'Y summed over (session, neuron) batches, so the stacked
  design matrix is never built. Alpha picked on the pooled val windows of the
  training sessions, then refit on train + val.

Tests (always on the test windows of the last 20% of a session):
- within-session: a, and b trained on the train parts of all 9 sessions
- held-out mouse: b trained on 2 mice per line (6 sessions), tested on the 3rd
  mouse per line (highest of the 3 ids). The held-out mouse is still z-scored
  with its own train-part mean/std; none of its data is used to fit b.

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
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_linear import (ALPHAS, BIN, CACHE, FUTURE, PAST, ROOT, TRAIN_FRAC,
                          VAL_FRAC, downsample, mean_pearson, window_starts, windows)
from step4_stimulus import stim_windows, stimulus_features

warnings.filterwarnings("ignore")

CRE_LINES = ["Cux2-CreERT2", "Sst-IRES-Cre", "Vip-IRES-Cre"]
N_PER_LINE = 3
MODELS = ["copy-last", "ridge-neuron+stim", "ridge-shared+stim (all 9)",
          "ridge-shared+stim (held-out)"]


def pick_sessions(api, exps):
    """Per Cre line: lowest id per mouse, then the N_PER_LINE lowest of those."""
    out = []
    for cre in CRE_LINES:
        es = api.filter_ophys_experiments(exps, cre_lines=[cre], targeted_structures=["VISp"],
                                          session_types=["three_session_A"])
        first = {}
        for e in sorted(es, key=lambda e: e["id"]):
            first.setdefault(e["specimen_id"], e)
        for k, e in enumerate(sorted(first.values(), key=lambda e: e["id"])[:N_PER_LINE]):
            out.append(dict(cre=cre, id=e["id"], mouse=e["specimen"]["external_specimen_name"],
                            depth=e["imaging_depth"], held_out=k == N_PER_LINE - 1))
    return out


class Gram:
    """Sufficient statistics for ridge with intercept: n, sum x, sum y, X'X, X'Y."""

    def __init__(self, n=0, sx=0.0, sy=0.0, xx=0.0, xy=0.0):
        self.n, self.sx, self.sy, self.xx, self.xy = n, sx, sy, xx, xy

    def __add__(self, o):
        return Gram(self.n + o.n, self.sx + o.sx, self.sy + o.sy, self.xx + o.xx, self.xy + o.xy)

    def solver(self):
        """Returns coef(alpha) -> (W, b) for the centred ridge problem."""
        xm, ym = self.sx / self.n, self.sy / self.n
        C = self.xx - self.n * np.outer(xm, xm)
        Cxy = self.xy - self.n * np.outer(xm, ym)
        s, V = np.linalg.eigh(C)
        VtXtY = V.T @ Cxy

        def coef(alpha):
            W = V @ (VtXtY / (s + alpha)[:, None])
            return W, ym - xm @ W
        return coef


def stim_stats(S):
    return dict(n=len(S), s=S.sum(0), ss=S.T @ S)


def neuron_gram(x, y, S, st):
    """Gram of the design [x, S] for one neuron, reusing the session's S'S."""
    p = x.shape[1]
    xx = np.empty((p + S.shape[1],) * 2)
    xS = x.T @ S
    xx[:p, :p], xx[:p, p:], xx[p:, :p], xx[p:, p:] = x.T @ x, xS, xS.T, st["ss"]
    xy = np.vstack([x.T @ y, S.T @ y])
    return Gram(len(x), np.concatenate([x.sum(0), st["s"]]), y.sum(0), xx, xy)


def predict(W, b, X, S):
    """X: (n_win, n_neurons, PAST), S: (n_win, n_stim) -> (n_win, n_neurons, FUTURE)."""
    return X @ W[:PAST] + (S @ W[PAST:] + b)[:, None, :]


def train_windows(z):
    """All stride-1 windows as views (no copy): X (n, PAST), Y (n, FUTURE) per neuron."""
    v = sliding_window_view(z, PAST + FUTURE, axis=1)  # (n_neurons, n_win, PAST+FUTURE)
    return v[:, :, :PAST], v[:, :, PAST:]


def load_session(info):
    ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{info['id']}.nwb"))
    t, dff = ds.get_dff_traces()
    fr = 1 / np.median(np.diff(t))
    F, _ = stimulus_features(ds, dff.shape[1])
    n_train = int(TRAIN_FRAC * dff.shape[1])
    mu = dff[:, :n_train].mean(axis=1, keepdims=True)
    sd = dff[:, :n_train].std(axis=1, keepdims=True)
    z = (dff - mu) / sd
    ztr, zte = downsample(z[:, :n_train], BIN), downsample(z[:, n_train:], BIN)
    Str_all, Ste_all = downsample(F[:, :n_train], BIN), downsample(F[:, n_train:], BIN)
    n_fit = ztr.shape[1] - int(round(VAL_FRAC * ztr.shape[1]))
    return dict(fr=fr, n_neurons=dff.shape[0], ztr=ztr, zte=zte, Str_all=Str_all,
                Ste_all=Ste_all, n_fit=n_fit)


def process_session(info):
    """Fits model a (per neuron, alpha per neuron) and returns the session's
    Gram sums for model b plus val/test windows."""
    d = load_session(info)
    ztr, n_fit = d["ztr"], d["n_fit"]
    # Train windows: stride 1 bin over the fit part; val = last 10% of train
    Xf, Yf = train_windows(ztr[:, :n_fit])
    Xv, Yv = train_windows(ztr[:, n_fit:])
    Sf = stim_windows(d["Str_all"][:, :n_fit], window_starts(n_fit, 1))
    Sv = stim_windows(d["Str_all"][:, n_fit:], window_starts(ztr.shape[1] - n_fit, 1))
    stf, stv = stim_stats(Sf), stim_stats(Sv)

    te_starts = window_starts(d["zte"].shape[1], FUTURE)
    Xte, Yte = windows(d["zte"], te_starts)
    Ste = stim_windows(d["Ste_all"], te_starts)

    gfit, gval = Gram(), Gram()
    Wa, ba, alpha_a = [], [], []
    for i in range(d["n_neurons"]):
        x, y, xv, yv = Xf[i], Yf[i], Xv[i], Yv[i]
        gf, gv = neuron_gram(x, y, Sf, stf), neuron_gram(xv, yv, Sv, stv)
        gfit, gval = gfit + gf, gval + gv
        # Model a: alpha for this neuron by val MSE, then refit on train + val
        coef = gf.solver()
        Dv = np.hstack([xv, Sv])
        val_mse = []
        for a in ALPHAS:
            W, b = coef(a)
            val_mse.append(((Dv @ W + b - yv) ** 2).mean())
        alpha = ALPHAS[int(np.argmin(val_mse))]
        W, b = (gf + gv).solver()(alpha)
        Wa.append(W); ba.append(b); alpha_a.append(alpha)

    # Test predictions of model a
    Pa = np.stack([np.hstack([Xte[:, i], Ste]) @ Wa[i] + ba[i]
                   for i in range(d["n_neurons"])], axis=1)
    return dict(info=info, fr=d["fr"], n_neurons=d["n_neurons"], bin_hz=d["fr"] / BIN,
                n_train_win=Xf.shape[1], n_val_win=Xv.shape[1],
                gfit=gfit, gval=gval, Xv=np.ascontiguousarray(Xv.transpose(1, 0, 2)),
                Yv=np.ascontiguousarray(Yv.transpose(1, 0, 2)), Sv=Sv,
                Xte=Xte, Yte=Yte, Ste=Ste, te_starts=te_starts, zte=d["zte"],
                Ste_all=d["Ste_all"], alpha_a=np.array(alpha_a), Pa=Pa)


def fit_shared(sessions):
    """Model b on the given sessions: alpha by pooled val MSE (over all neurons
    x windows x steps), then refit on train + val."""
    coef = sum((s["gfit"] for s in sessions), Gram()).solver()
    val_mse = []
    n_val = sum(s["Yv"].size for s in sessions)
    for a in ALPHAS:
        W, b = coef(a)
        val_mse.append(sum(((predict(W, b, s["Xv"], s["Sv"]) - s["Yv"]) ** 2).sum()
                           for s in sessions) / n_val)
    alpha = ALPHAS[int(np.argmin(val_mse))]
    W, b = sum((s["gfit"] + s["gval"] for s in sessions), Gram()).solver()(alpha)
    n_rows = sum(s["gfit"].n + s["gval"].n for s in sessions)
    return W, b, alpha, np.array(val_mse), n_rows


def metrics(Y, P, Pcopy):
    mse = float(((P - Y) ** 2).mean())
    return dict(mse=mse, r=mean_pearson(Y, P), score=1 - mse / float(((Pcopy - Y) ** 2).mean()))


def pooled(rows, model):
    """Pool the neurons of several sessions: MSE over all neurons x windows x steps
    (sessions have different window counts, so weight by element count), mean r over
    neurons, score vs copy-last on the same elements."""
    se = sum(r["sse"][model] for r in rows)
    se_copy = sum(r["sse"]["copy-last"] for r in rows)
    n = sum(r["n_el"] for r in rows)
    rs = np.concatenate([r["r_neuron"][model] for r in rows])
    return dict(mse=se / n, r=float(np.nanmean(rs)), score=1 - se / se_copy)


def neuron_r(Y, P):
    yt = Y.transpose(1, 0, 2).reshape(Y.shape[1], -1)
    pt = P.transpose(1, 0, 2).reshape(P.shape[1], -1)
    return np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(yt, pt)])


def main():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    infos = pick_sessions(api, exps)
    sessions = []
    for info in infos:
        s = process_session(info)
        sessions.append(s)
        print(f"{info['cre']:13s} exp {info['id']} mouse {info['mouse']} depth {info['depth']} um"
              f" | {s['n_neurons']} neurons | {s['fr']:.2f} Hz | windows train/val/test"
              f" {s['n_train_win']}/{s['n_val_win']}/{len(s['Xte'])}"
              f"{' | HELD-OUT' if info['held_out'] else ''}")
        a = s["alpha_a"]
        print(f"    model a alpha per neuron: median {np.median(a):g}, at grid min"
              f" {(a == ALPHAS[0]).sum()}, at grid max {(a == ALPHAS[-1]).sum()}")

    W_all, b_all, alpha_all, _, nrow_all = fit_shared(sessions)
    train_s = [s for s in sessions if not s["info"]["held_out"]]
    W_ho, b_ho, alpha_ho, _, nrow_ho = fit_shared(train_s)
    print(f"\nshared (all 9): alpha {alpha_all:g}, {nrow_all} training rows (neuron x window)")
    print(f"shared (6, held-out test): alpha {alpha_ho:g}, {nrow_ho} training rows")

    rows = []
    for s in sessions:
        X, Y, S = s["Xte"], s["Yte"], s["Ste"]
        P = {"copy-last": np.repeat(X[:, :, -1:], FUTURE, axis=2),
             "ridge-neuron+stim": s["Pa"],
             "ridge-shared+stim (all 9)": predict(W_all, b_all, X, S),
             "ridge-shared+stim (held-out)": predict(W_ho, b_ho, X, S)}
        s["P"] = P
        rows.append(dict(info=s["info"], n_neurons=s["n_neurons"], n_el=Y.size,
                         sse={m: float(((P[m] - Y) ** 2).sum()) for m in MODELS},
                         r_neuron={m: neuron_r(Y, P[m]) for m in MODELS},
                         per={m: metrics(Y, P[m], P["copy-last"]) for m in MODELS}))

    def show(title, sel, models):
        print(f"\n== {title} ==")
        for cre in CRE_LINES:
            rs = [r for r in rows if r["info"]["cre"] == cre and sel(r)]
            nn = sum(r["n_neurons"] for r in rs)
            print(f"{cre} ({len(rs)} sessions, {nn} neurons)")
            for m in models:
                v = pooled(rs, m)
                print(f"    {m:30s} MSE {v['mse']:.3f}  r {v['r']:.3f}  score {v['score']:+.3f}")

    show("Within-session test (all 9 sessions)", lambda r: True,
         ["copy-last", "ridge-neuron+stim", "ridge-shared+stim (all 9)"])
    show("Held-out mouse (3rd mouse per line)", lambda r: r["info"]["held_out"], MODELS)
    print("\n== Per session ==")
    for r in rows:
        print(f"{r['info']['cre']:13s} {r['info']['id']}{' (held-out)' if r['info']['held_out'] else ''}")
        for m in MODELS:
            if m.endswith("(held-out)") and not r["info"]["held_out"]:
                continue
            v = r["per"][m]
            print(f"    {m:30s} MSE {v['mse']:.3f}  r {v['r']:.3f}  score {v['score']:+.3f}")

    # Per-neuron comparison on held-out mice: shared (held-out) vs per-neuron a
    print("\n== Held-out mice, per neuron: test MSE shared(held-out) - ridge-neuron+stim ==")
    for s in sessions:
        if not s["info"]["held_out"]:
            continue
        Y, P = s["Yte"], s["P"]
        dm = (((P["ridge-shared+stim (held-out)"] - Y) ** 2).mean(axis=(0, 2))
              - ((P["ridge-neuron+stim"] - Y) ** 2).mean(axis=(0, 2)))
        print(f"    {s['info']['cre']:13s} shared better for {(dm < 0).sum()}/{len(dm)} neurons,"
              f" median {np.median(dm):+.3f}")

    # Shared weights: own-past part and stimulus part
    print("\nshared (all 9) weights on own past, last 5 bins -> step 1:",
          np.round(W_all[PAST - 5:PAST, 0], 3), " intercept", np.round(b_all, 3))

    # Plot: 2 neurons per Cre line from the held-out mice (seeded random pick),
    # 30 s from 5 s before the first grating epoch in test, as in step 4
    rng = np.random.default_rng(0)
    colors = {"copy-last": "tab:red", "ridge-neuron+stim": "tab:purple",
              "ridge-shared+stim (held-out)": "tab:green"}
    held = [s for s in sessions if s["info"]["held_out"]]
    fig, axes = plt.subplots(6, 1, figsize=(11, 14))
    ax_i = 0
    for s in held:
        zte, S, starts = s["zte"], s["Ste_all"], s["te_starts"]
        tt = np.arange(zte.shape[1]) / s["bin_hz"]
        t0 = tt[np.argmax(S[0] > 0.5)] - 5
        ws = np.where((tt[starts] >= t0) & (tt[starts] + 1 <= t0 + 30))[0]
        lo, hi = starts[ws[0]] - PAST, starts[ws[-1]] + FUTURE
        neurons = np.sort(rng.choice(s["n_neurons"], size=2, replace=False))
        s["plot_neurons"] = neurons.tolist()
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
                            label=m if k == 0 else None)
            ax.set_title(f"{s['info']['cre']} exp {s['info']['id']} (held-out mouse "
                         f"{s['info']['mouse']}) | neuron {n}", fontsize=9, loc="left")
            ax.set_ylabel("z dF/F")
    axes[0].legend(fontsize=8, loc="upper right", ncol=5)
    axes[-1].set_xlabel("Time from test start (s)")
    fig.suptitle("Step 5: shared ridge (trained on other mice) vs per-neuron ridge, held-out mice (test set)")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step5_shared.png", dpi=150)
    print("\nPlotted neurons:", {s["info"]["id"]: s["plot_neurons"] for s in held})
    print("Saved plots/step5_shared.png")


if __name__ == "__main__":
    main()
