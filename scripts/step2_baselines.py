"""Step 2: simple forecasting baselines on Allen Brain Observatory dF/F traces.

Task: from the past 3 s of a neuron's z-scored dF/F, predict the next 1 s.
Baselines: copy-last (repeat last value) and mean (mean of the past 3 s).
Split: first 80% of frames train, last 20% test (z-score stats from train only).

Data files are read from the allensdk cache layout under data/brain_observatory/
(ophys_experiments.json + ophys_experiment_data/<id>.nwb). In this environment
api.brain-map.org was not reachable, so the files were fetched from the Allen
public S3 mirror (allen-brain-observatory/visual-coding-2p/).
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
PAST_S, FUTURE_S, TRAIN_FRAC = 3.0, 1.0, 0.8


def pick_experiments():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    sst = api.filter_ophys_experiments(exps, ids=[573720508])[0]
    cux2 = api.filter_ophys_experiments(
        exps, cre_lines=["Cux2-CreERT2"], targeted_structures=["VISp"],
        session_types=["three_session_A"])
    print("Cux2-CreERT2 VISp three_session_A experiments:", len(cux2))
    cux2 = min(cux2, key=lambda e: e["id"])  # same rule as step 1: lowest id
    return [("Sst-IRES-Cre", sst), ("Cux2-CreERT2", cux2)]


def make_windows(z, past, future):
    """Non-overlapping targets: stride = future. Returns (X, Y) with shapes
    (n_windows, n_neurons, past) and (n_windows, n_neurons, future)."""
    starts = np.arange(past, z.shape[1] - future + 1, future)
    X = np.stack([z[:, s - past:s] for s in starts])
    Y = np.stack([z[:, s:s + future] for s in starts])
    return X, Y, starts


def predict(X, future, kind):
    if kind == "copy-last":
        v = X[:, :, -1:]
    else:
        v = X.mean(axis=2, keepdims=True)
    return np.repeat(v, future, axis=2)


def mean_pearson(Y, P):
    """Per-neuron Pearson r over all test windows x horizon steps, then mean."""
    yt = Y.transpose(1, 0, 2).reshape(Y.shape[1], -1)
    pt = P.transpose(1, 0, 2).reshape(P.shape[1], -1)
    rs = [np.corrcoef(a, b)[0, 1] for a, b in zip(yt, pt)]
    return float(np.nanmean(rs)), rs


def main():
    results, plot_data = [], []
    for label, exp in pick_experiments():
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{exp['id']}.nwb"))
        t, dff = ds.get_dff_traces()
        fr = 1 / np.median(np.diff(t))
        past, future = int(round(PAST_S * fr)), int(round(FUTURE_S * fr))

        n_train = int(TRAIN_FRAC * dff.shape[1])
        mu = dff[:, :n_train].mean(axis=1, keepdims=True)
        sd = dff[:, :n_train].std(axis=1, keepdims=True)
        z = (dff - mu) / sd
        X, Y, starts = make_windows(z[:, n_train:], past, future)

        row = dict(label=label, id=exp["id"], depth=exp["imaging_depth"],
                   n_neurons=dff.shape[0], n_frames=dff.shape[1], fr=fr,
                   past=past, future=future, n_train=n_train,
                   n_test=dff.shape[1] - n_train, n_windows=len(starts))
        preds = {}
        for kind in ["copy-last", "mean"]:
            P = predict(X, future, kind)
            preds[kind] = P
            mse = float(((P - Y) ** 2).mean())
            # MSE at first and last horizon step, to see how error grows
            mse_h = ((P - Y) ** 2).mean(axis=(0, 1))
            r, _ = mean_pearson(Y, P)
            row[kind] = dict(mse=mse, r=r, mse_first=float(mse_h[0]),
                             mse_last=float(mse_h[-1]))
        results.append(row)
        plot_data.append((row, z[:, n_train:], starts, preds, t[n_train:] - t[n_train]))

        print(f"\n{label} | experiment {exp['id']} | depth {exp['imaging_depth']} um")
        print(f"  neurons {row['n_neurons']}, frames {row['n_frames']}, {fr:.2f} Hz")
        print(f"  past {past} frames, future {future} frames, test windows {len(starts)}")
        for kind in ["copy-last", "mean"]:
            m = row[kind]
            print(f"  {kind:9s}  MSE {m['mse']:.3f}  mean r {m['r']:.3f}"
                  f"  (MSE step1 {m['mse_first']:.3f}, step{future} {m['mse_last']:.3f})")

    # Plot: 2 neurons per session (seeded random pick), first 30 s of test
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    ax_i = 0
    for row, zt, starts, preds, tt in plot_data:
        neurons = rng.choice(row["n_neurons"], size=2, replace=False)
        row["plot_neurons"] = neurons.tolist()
        n_show = int(np.searchsorted(starts, 30 * row["fr"]))
        for n in neurons:
            ax = axes[ax_i]; ax_i += 1
            end = starts[n_show - 1] + row["future"]
            ax.plot(tt[:end], zt[n, :end], color="k", lw=0.8, label="true")
            for kind, c in [("copy-last", "tab:red"), ("mean", "tab:blue")]:
                for w in range(n_show):
                    s = starts[w]
                    ax.plot(tt[s:s + row["future"]], preds[kind][w, n], color=c, lw=1.5,
                            label=kind if w == 0 else None)
            ax.set_title(f"{row['label']} exp {row['id']} | neuron {n}", fontsize=9, loc="left")
            ax.set_ylabel("z dF/F")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("Time from test start (s)")
    fig.suptitle("Step 2 baselines: past 3 s -> next 1 s (test set)")
    fig.tight_layout()
    (ROOT / "plots").mkdir(exist_ok=True)
    fig.savefig(ROOT / "plots/step2_baselines.png", dpi=150)
    print("\nPlotted neurons:", {r["label"]: r["plot_neurons"] for r, *_ in plot_data})


if __name__ == "__main__":
    main()
