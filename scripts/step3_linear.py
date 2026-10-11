"""Step 3: linear forecasting models on Allen Brain Observatory dF/F traces.

Same sessions, 80/20 split and z-score as step 2 (z-score stats from train only).
Traces are downsampled to 10 Hz by averaging every 3 frames.
Task: past 3 s (30 bins) -> next 1 s (10 bins).

Models:
- copy-last, mean (as in step 2, at 10 Hz)
- ridge per neuron: each neuron predicted from its own past only
- ridge population: each neuron predicted from all neurons' past
Ridge alpha is picked on the last 10% of train (val), then the model is refit
on the full train part. Ridge is solved in closed form with numpy.

Data files are read from the allensdk cache layout under data/brain_observatory/,
fetched from the Allen public S3 mirror (see scripts/step2_baselines.py).
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data/brain_observatory"
SESSIONS = [("Sst-IRES-Cre", 573720508), ("Cux2-CreERT2", 501271265)]
PLOT_NEURONS = {573720508: [7, 6], 501271265: [66, 57]}  # same as step 2
BIN, PAST, FUTURE = 3, 30, 10      # 3 frames/bin (~10 Hz), 3 s past, 1 s future
TRAIN_FRAC, VAL_FRAC = 0.8, 0.1    # val = last 10% of train
ALPHAS = np.logspace(-1, 7, 17)
MODELS = ["copy-last", "mean", "ridge-neuron", "ridge-pop"]


def downsample(x, k):
    n = x.shape[1] // k * k
    return x[:, :n].reshape(x.shape[0], -1, k).mean(axis=2)


def window_starts(n_bins, stride):
    return np.arange(PAST, n_bins - FUTURE + 1, stride)


def windows(z, starts):
    """X: (n_windows, n_neurons, PAST), Y: (n_windows, n_neurons, FUTURE)."""
    X = np.stack([z[:, s - PAST:s] for s in starts])
    Y = np.stack([z[:, s:s + FUTURE] for s in starts])
    return X, Y


class Ridge:
    """Multi-output ridge with an intercept, for many alphas at once.
    Uses an eigendecomposition of X'X so each alpha is cheap."""

    def fit(self, X, Y):
        self.xm, self.ym = X.mean(0), Y.mean(0)
        Xc, Yc = X - self.xm, Y - self.ym
        self.s, self.V = np.linalg.eigh(Xc.T @ Xc)
        self.VtXtY = self.V.T @ (Xc.T @ Yc)
        return self

    def coef(self, alpha):
        return self.V @ (self.VtXtY / (self.s + alpha)[:, None])

    def predict(self, X, alpha):
        return (X - self.xm) @ self.coef(alpha) + self.ym


def fit_ridge_neuron(Xtr, Ytr, Xva, Yva):
    """One ridge per neuron, own past only. One alpha shared by all neurons
    (picked on val). Returns predict(X) for the model refit on train+val."""
    n = Xtr.shape[1]
    val_mse = np.zeros(len(ALPHAS))
    for i in range(n):
        r = Ridge().fit(Xtr[:, i], Ytr[:, i])
        for a, alpha in enumerate(ALPHAS):
            val_mse[a] += ((r.predict(Xva[:, i], alpha) - Yva[:, i]) ** 2).mean() / n
    alpha = ALPHAS[val_mse.argmin()]
    Xall, Yall = np.concatenate([Xtr, Xva]), np.concatenate([Ytr, Yva])
    fits = [Ridge().fit(Xall[:, i], Yall[:, i]) for i in range(n)]

    def predict(X):
        return np.stack([fits[i].predict(X[:, i], alpha) for i in range(n)], axis=1)
    return predict, alpha, val_mse


def fit_ridge_pop(Xtr, Ytr, Xva, Yva):
    """One ridge on all neurons' past -> all neurons' future."""
    flat = lambda A: A.reshape(len(A), -1)
    shape = Ytr.shape[1:]
    r = Ridge().fit(flat(Xtr), flat(Ytr))
    val_mse = np.array([((r.predict(flat(Xva), a) - flat(Yva)) ** 2).mean() for a in ALPHAS])
    alpha = ALPHAS[val_mse.argmin()]
    r = Ridge().fit(flat(np.concatenate([Xtr, Xva])), flat(np.concatenate([Ytr, Yva])))

    def predict(X):
        return r.predict(flat(X), alpha).reshape(len(X), *shape)
    return predict, alpha, val_mse


def mean_pearson(Y, P):
    """Per-neuron Pearson r over all test windows x steps, then mean."""
    yt = Y.transpose(1, 0, 2).reshape(Y.shape[1], -1)
    pt = P.transpose(1, 0, 2).reshape(P.shape[1], -1)
    return float(np.nanmean([np.corrcoef(a, b)[0, 1] for a, b in zip(yt, pt)]))


def main():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    results, plot_data = [], []
    for label, eid in SESSIONS:
        exp = api.filter_ophys_experiments(exps, ids=[eid])[0]
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{eid}.nwb"))
        t, dff = ds.get_dff_traces()
        fr = 1 / np.median(np.diff(t))

        # Same split and z-score as step 2, then 10 Hz bins within each part
        n_train = int(TRAIN_FRAC * dff.shape[1])
        mu = dff[:, :n_train].mean(axis=1, keepdims=True)
        sd = dff[:, :n_train].std(axis=1, keepdims=True)
        z = (dff - mu) / sd
        ztr, zte = downsample(z[:, :n_train], BIN), downsample(z[:, n_train:], BIN)

        # Train windows: stride 1 bin. Val = last 10% of train bins.
        n_fit = ztr.shape[1] - int(round(VAL_FRAC * ztr.shape[1]))
        Xtr, Ytr = windows(ztr[:, :n_fit], window_starts(n_fit, 1))
        Xva, Yva = windows(ztr[:, n_fit:], window_starts(ztr.shape[1] - n_fit, 1))
        # Test windows: stride 1 s (no overlap in targets), as in step 2
        te_starts = window_starts(zte.shape[1], FUTURE)
        Xte, Yte = windows(zte, te_starts)

        pred_neuron, a_neuron, _ = fit_ridge_neuron(Xtr, Ytr, Xva, Yva)
        pred_pop, a_pop, _ = fit_ridge_pop(Xtr, Ytr, Xva, Yva)
        preds = {
            "copy-last": np.repeat(Xte[:, :, -1:], FUTURE, axis=2),
            "mean": np.repeat(Xte.mean(axis=2, keepdims=True), FUTURE, axis=2),
            "ridge-neuron": pred_neuron(Xte),
            "ridge-pop": pred_pop(Xte),
        }

        row = dict(label=label, id=eid, depth=exp["imaging_depth"], n_neurons=dff.shape[0],
                   fr=fr, bin_hz=fr / BIN, n_train_bins=ztr.shape[1], n_fit_bins=n_fit,
                   n_test_bins=zte.shape[1], n_train_win=len(Xtr), n_val_win=len(Xva),
                   n_test_win=len(Xte), alpha_neuron=a_neuron, alpha_pop=a_pop)
        mse_copy = float(((preds["copy-last"] - Yte) ** 2).mean())
        for m in MODELS:
            err = (preds[m] - Yte) ** 2
            mse = float(err.mean())
            row[m] = dict(mse=mse, r=mean_pearson(Yte, preds[m]), score=1 - mse / mse_copy,
                          mse_first=float(err[:, :, 0].mean()), mse_last=float(err[:, :, -1].mean()))
        results.append(row)
        plot_data.append((row, zte, te_starts, preds))

        print(f"\n{label} | experiment {eid} | depth {row['depth']} um | {row['n_neurons']} neurons")
        print(f"  {fr:.2f} Hz -> {row['bin_hz']:.2f} Hz bins; train/val/test windows "
              f"{len(Xtr)}/{len(Xva)}/{len(Xte)}")
        print(f"  alpha: ridge-neuron {a_neuron:g}, ridge-pop {a_pop:g}")
        for m in MODELS:
            v = row[m]
            print(f"  {m:12s} MSE {v['mse']:.3f}  r {v['r']:.3f}  score {v['score']:+.3f}"
                  f"  (MSE step1 {v['mse_first']:.3f}, step10 {v['mse_last']:.3f})")

    # Plot: same 4 neurons as step 2, first 30 s of test
    colors = {"copy-last": "tab:red", "mean": "tab:blue",
              "ridge-neuron": "tab:orange", "ridge-pop": "tab:green"}
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    ax_i = 0
    for row, zte, starts, preds in plot_data:
        tt = np.arange(zte.shape[1]) * BIN / row["fr"]
        n_show = int(np.searchsorted(starts, 30 * row["bin_hz"]))
        end = starts[n_show - 1] + FUTURE
        for n in PLOT_NEURONS[row["id"]]:
            ax = axes[ax_i]; ax_i += 1
            ax.plot(tt[:end], zte[n, :end], color="k", lw=0.9, label="true")
            for m in MODELS:
                for w in range(n_show):
                    s = starts[w]
                    ax.plot(tt[s:s + FUTURE], preds[m][w, n], color=colors[m], lw=1.3,
                            label=m if w == 0 else None)
            ax.set_title(f"{row['label']} exp {row['id']} | neuron {n}", fontsize=9, loc="left")
            ax.set_ylabel("z dF/F")
    axes[0].legend(fontsize=8, loc="upper right", ncol=5)
    axes[-1].set_xlabel("Time from test start (s)")
    fig.suptitle("Step 3 linear models: past 3 s -> next 1 s at 10 Hz (test set)")
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "plots/step3_linear.png", dpi=150)
    print("\nSaved plots/step3_linear.png")


if __name__ == "__main__":
    main()
