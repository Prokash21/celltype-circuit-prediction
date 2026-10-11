"""Step 4: add stimulus information to the per-neuron ridge model.

Same sessions, split, z-score and 10 Hz bins as scripts/step3_linear.py.
Stimulus features per bin (12), built from the NWB stimulus tables:
- stimulus type one-hot: drifting gratings, natural movie (one + three), spontaneous
- grating on/off (non-blank grating trial)
- grating direction one-hot, 8 directions (0 when no grating)
The stimulus is set by the experiment, so the model sees stimulus features for
the past 3 s AND the next 1 s (40 bins x 12 = 480 inputs), plus the neuron's
own past 30 bins.

Compared: ridge per neuron without vs with stimulus features (alpha picked as
in step 3), plus copy-last and mean for the score. Extra evaluation: windows
whose 1 s target starts at a grating onset (non-blank trial start) in test.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_linear import (ALPHAS, BIN, CACHE, FUTURE, PAST, PLOT_NEURONS, ROOT,
                          SESSIONS, TRAIN_FRAC, VAL_FRAC, Ridge, downsample,
                          mean_pearson, window_starts, windows)

warnings.filterwarnings("ignore")

DIRECTIONS = np.arange(0, 360, 45)
FEATURES = (["type: gratings", "type: natural movie", "type: spontaneous", "grating on"]
            + [f"dir {d}" for d in DIRECTIONS])
MAX_GAP = 60  # frames; gaps between trials up to this are filled as the same epoch
MODELS = ["copy-last", "mean", "ridge-neuron", "ridge-neuron+stim"]


def fill_epochs(starts, ends, n_frames):
    """Frames covered by trials, with short gaps between trials filled."""
    mask = np.zeros(n_frames, bool)
    order = np.argsort(starts)
    starts, ends = np.asarray(starts)[order], np.asarray(ends)[order]
    for i, (s, e) in enumerate(zip(starts, ends)):
        mask[s:e] = True
        if i + 1 < len(starts) and starts[i + 1] - e <= MAX_GAP:
            mask[e:starts[i + 1]] = True
    return mask


def stimulus_features(ds, n_frames):
    """(12, n_frames) stimulus features at imaging-frame resolution, plus the
    start frames of non-blank grating trials."""
    F = np.zeros((len(FEATURES), n_frames))
    dg = ds.get_stimulus_table("drifting_gratings")
    F[0] = fill_epochs(dg.start, dg.end, n_frames)
    for name in ["natural_movie_one", "natural_movie_three"]:
        nm = ds.get_stimulus_table(name)
        F[1] = np.maximum(F[1], fill_epochs(nm.start, nm.end, n_frames))
    sp = ds.get_stimulus_table("spontaneous")
    F[2] = fill_epochs(sp.start, sp.end, n_frames)
    on = dg[dg.blank_sweep == 0]
    for _, tr in on.iterrows():
        s, e = int(tr.start), int(tr.end)
        F[3, s:e] = 1
        F[4 + int(np.where(DIRECTIONS == tr.orientation)[0][0]), s:e] = 1
    return F, on.start.values.astype(int)


def stim_windows(S, starts):
    """Stimulus features for past PAST bins and next FUTURE bins:
    (n_windows, n_features * (PAST + FUTURE))."""
    return np.stack([S[:, s - PAST:s + FUTURE].ravel() for s in starts])


def fit_ridge_neuron(Xtr, Ytr, Xva, Yva, Str=None, Sva=None):
    """One ridge per neuron on its own past (+ shared stimulus inputs if given).
    One alpha for all neurons, picked on val; then refit on train + val."""
    def design(X, S, i):
        return X[:, i] if S is None else np.hstack([X[:, i], S])

    n = Xtr.shape[1]
    val_mse = np.zeros(len(ALPHAS))
    for i in range(n):
        r = Ridge().fit(design(Xtr, Str, i), Ytr[:, i])
        Xv = design(Xva, Sva, i)
        for a, alpha in enumerate(ALPHAS):
            val_mse[a] += ((r.predict(Xv, alpha) - Yva[:, i]) ** 2).mean() / n
    alpha = ALPHAS[val_mse.argmin()]
    Xall, Yall = np.concatenate([Xtr, Xva]), np.concatenate([Ytr, Yva])
    Sall = None if Str is None else np.concatenate([Str, Sva])
    fits = [Ridge().fit(design(Xall, Sall, i), Yall[:, i]) for i in range(n)]

    def predict(X, S=None):
        return np.stack([fits[i].predict(design(X, S, i), alpha) for i in range(n)], axis=1)
    return predict, alpha


def predict_all(X, S, pred_plain, pred_stim):
    return {
        "copy-last": np.repeat(X[:, :, -1:], FUTURE, axis=2),
        "mean": np.repeat(X.mean(axis=2, keepdims=True), FUTURE, axis=2),
        "ridge-neuron": pred_plain(X),
        "ridge-neuron+stim": pred_stim(X, S),
    }


def scores(Y, preds):
    mse_copy = float(((preds["copy-last"] - Y) ** 2).mean())
    out = {}
    for m in MODELS:
        err = (preds[m] - Y) ** 2
        out[m] = dict(mse=float(err.mean()), r=mean_pearson(Y, preds[m]),
                      score=1 - float(err.mean()) / mse_copy,
                      mse_first=float(err[:, :, 0].mean()), mse_last=float(err[:, :, -1].mean()))
    return out


def main():
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    results, plot_data = [], []
    for label, eid in SESSIONS:
        exp = api.filter_ophys_experiments(exps, ids=[eid])[0]
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{eid}.nwb"))
        t, dff = ds.get_dff_traces()
        fr = 1 / np.median(np.diff(t))
        F, onsets = stimulus_features(ds, dff.shape[1])

        # Same split / z-score / bins as step 3; stimulus binned the same way
        n_train = int(TRAIN_FRAC * dff.shape[1])
        mu = dff[:, :n_train].mean(axis=1, keepdims=True)
        sd = dff[:, :n_train].std(axis=1, keepdims=True)
        z = (dff - mu) / sd
        ztr, zte = downsample(z[:, :n_train], BIN), downsample(z[:, n_train:], BIN)
        Str_all, Ste_all = downsample(F[:, :n_train], BIN), downsample(F[:, n_train:], BIN)

        n_fit = ztr.shape[1] - int(round(VAL_FRAC * ztr.shape[1]))
        tr_starts = window_starts(n_fit, 1)
        va_starts = window_starts(ztr.shape[1] - n_fit, 1)
        Xtr, Ytr = windows(ztr[:, :n_fit], tr_starts)
        Xva, Yva = windows(ztr[:, n_fit:], va_starts)
        Str = stim_windows(Str_all[:, :n_fit], tr_starts)
        Sva = stim_windows(Str_all[:, n_fit:], va_starts)

        te_starts = window_starts(zte.shape[1], FUTURE)
        Xte, Yte = windows(zte, te_starts)
        Ste = stim_windows(Ste_all, te_starts)

        # Grating-onset windows in test: target starts at the bin holding the onset
        on_bins = (onsets[onsets >= n_train] - n_train) // BIN
        on_bins = on_bins[(on_bins >= PAST) & (on_bins + FUTURE <= zte.shape[1])]
        Xon, Yon = windows(zte, on_bins)
        Son = stim_windows(Ste_all, on_bins)

        pred_plain, a_plain = fit_ridge_neuron(Xtr, Ytr, Xva, Yva)
        pred_stim, a_stim = fit_ridge_neuron(Xtr, Ytr, Xva, Yva, Str, Sva)

        preds = predict_all(Xte, Ste, pred_plain, pred_stim)
        preds_on = predict_all(Xon, Son, pred_plain, pred_stim)
        row = dict(label=label, id=eid, depth=exp["imaging_depth"], n_neurons=dff.shape[0],
                   fr=fr, bin_hz=fr / BIN, n_train_win=len(Xtr), n_val_win=len(Xva),
                   n_test_win=len(Xte), n_onset_win=len(Xon),
                   frac_test_grating=float(Ste_all[0].mean()),
                   alpha_plain=a_plain, alpha_stim=a_stim,
                   test=scores(Yte, preds), onset=scores(Yon, preds_on))
        # Per-neuron change in test MSE from adding stimulus (negative = better)
        d = (((preds["ridge-neuron+stim"] - Yte) ** 2).mean(axis=(0, 2))
             - ((preds["ridge-neuron"] - Yte) ** 2).mean(axis=(0, 2)))
        row["neuron_dmse"] = d
        results.append(row)
        plot_data.append((row, zte, Ste_all, te_starts, preds))

        print(f"\n{label} | experiment {eid} | depth {row['depth']} um | {row['n_neurons']} neurons")
        print(f"  windows train/val/test/onset {len(Xtr)}/{len(Xva)}/{len(Xte)}/{len(Xon)};"
              f" grating epoch = {100 * row['frac_test_grating']:.0f}% of test bins")
        print(f"  alpha: ridge-neuron {a_plain:g}, ridge-neuron+stim {a_stim:g}")
        d = row["neuron_dmse"]
        print(f"  stim lowers test MSE for {(d < 0).sum()}/{len(d)} neurons;"
              f" median change {np.median(d):+.3f}, best {d.min():+.3f} (neuron {d.argmin()})")
        for part in ["test", "onset"]:
            print(f"  [{part}]")
            for m in MODELS:
                v = row[part][m]
                print(f"    {m:18s} MSE {v['mse']:.3f}  r {v['r']:.3f}  score {v['score']:+.3f}"
                      f"  (MSE step1 {v['mse_first']:.3f}, step10 {v['mse_last']:.3f})")

    # Plot: same 4 neurons as step 3. Test starts with natural movie, so show
    # 30 s starting 5 s before the first grating epoch in test.
    colors = {"ridge-neuron": "tab:orange", "ridge-neuron+stim": "tab:purple"}
    fig, axes = plt.subplots(4, 1, figsize=(11, 10))
    ax_i = 0
    for row, zte, S, starts, preds in plot_data:
        tt = np.arange(zte.shape[1]) / row["bin_hz"]
        t0 = tt[np.argmax(S[0] > 0.5)] - 5
        ws = np.where((tt[starts] >= t0) & (tt[starts] + 1 <= t0 + 30))[0]
        lo, hi = starts[ws[0]] - PAST, starts[ws[-1]] + FUTURE
        for n in PLOT_NEURONS[row["id"]]:
            ax = axes[ax_i]; ax_i += 1
            on = S[3, lo:hi] > 0.5
            ax.fill_between(tt[lo:hi], 0, 1, where=on, transform=ax.get_xaxis_transform(),
                            color="0.85", step="mid", label="grating on")
            ax.plot(tt[lo:hi], zte[n, lo:hi], color="k", lw=0.9, label="true")
            for m, c in colors.items():
                for k, w in enumerate(ws):
                    s = starts[w]
                    ax.plot(tt[s:s + FUTURE], preds[m][w, n], color=c, lw=1.5,
                            label=m if k == 0 else None)
            ax.set_title(f"{row['label']} exp {row['id']} | neuron {n}", fontsize=9, loc="left")
            ax.set_ylabel("z dF/F")
    axes[0].legend(fontsize=8, loc="upper right", ncol=4)
    axes[-1].set_xlabel("Time from test start (s)")
    fig.suptitle("Step 4: ridge per neuron with vs without stimulus features (test set)")
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step4_stimulus.png", dpi=150)
    print("\nSaved plots/step4_stimulus.png")


if __name__ == "__main__":
    main()
