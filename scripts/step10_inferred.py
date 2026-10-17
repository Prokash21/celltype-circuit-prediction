"""Step 10: can a cell type inferred from activity replace the true Cre label?

Same 46 sessions (VISp, three_session_A, Cux2 / Sst / Vip / Pvalb, one per mouse),
5 leave-mice-out folds, inputs (own past 30 bins + 480 stimulus inputs -> next 10
bins) and test windows (last 20% of each held-out session, stride 10) as steps 8-9.

Phases (run in order, `python scripts/step10_inferred.py <phase>`; intermediate files
go to data/step10/, not committed; run `ridge` with OMP_NUM_THREADS=1, its workers
are slow with oversubscribed BLAS threads):
- `extract`: step 8 extraction (scripts/step8_medium.py `extract`, unchanged) plus a
  side file data/cache/<id>_raw.npz with per-neuron raw dF/F statistics over the
  train part (the step 8 cache keeps only z-scored traces).
- `ridge`: step 8 ridge models per fold, with the step 8 functions and seeds:
  b (shared, line-balanced), d (one model per true Cre line), e (shuffled groups,
  5 seeds).
- `mlp`: step 9 MLPs per fold with step 9's `run_job` and seeds: m1 (3 seeds),
  m2 = Cre FiLM trained with the true labels (3 seeds), m3 = shuffled FiLM (5
  seeds). The trained weights are kept, and the m1 seed-0 model of each fold gives
  the embedding features (b) of every neuron.
- `classify`: cell-type classifier per fold, fitted on the training mice only.
  Features per neuron, from the train part (first 80%) of each recording only:
    (a) hand features (see hand_features), (b) the m1 embedding = mean activity of
    the 2nd hidden layer (256 units) over all train-part windows (stride 10),
    (a+b) both. Classifiers: multinomial logistic regression (quantile -> normal
    transform, C in LR_C) and histogram gradient boosting, both with class weights
    balanced by line. Model and C are picked per fold by a 4-fold inner CV over the
    training mice (StratifiedGroupKFold, groups = mice), score = neuron-level
    balanced accuracy. Session label = majority vote over the neurons of a
    recording (ties: highest mean probability).
- `forecast`: test-time labels change, training labels never do:
    ridge b, d-true, d-pred-neuron, d-pred-session, e;
    MLP m1, m2-true, m2-pred-neuron, m2-pred-session, m2-soft (FiLM embedding =
    class probabilities x embedding table, i.e. gamma and beta are the
    probability-weighted mix of the line gammas and betas), m3.
  Per-mouse score = 1 - SSE / SSE_copylast; Wilcoxon signed-rank tests over mice;
  fraction of the true-label gain recovered = mean(pred - none) / mean(true - none).

LOLCAT (Schneider et al. 2023, Cell Reports; code: github.com/nerdslab/lolcat,
read but not run): the idea of classifying the Cre line of single neurons from
their activity alone, the 4 classes of its Allen ophys dataset (Pvalb, Vip, Sst,
excitatory), features computed per stimulus condition, inter-event-interval
distributions as features, and class-balanced training (its
DynamicBalancedSampler). Its attention-over-snippets network is not used here.
"""
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import welch
from scipy.stats import kruskal, kurtosis, skew, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5_shared
import step8_medium
import step9_mlp as s9
from step3_linear import BIN, CACHE, FUTURE, PAST, ROOT, TRAIN_FRAC, window_starts, windows
from step4_stimulus import DIRECTIONS, stim_windows
from step5_shared import predict
from step7_robust import N_SEEDS, fit_balanced
from step8_medium import (CRE_LINES, N_FOLDS, NPZ, SHORT, fit_grouped, fit_grouped_balanced,
                          pick_sessions, process_cached, shuffled_groups)

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step10"
OUT = ROOT / "results/step10"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
N_LINES = len(CRE_LINES)
LINES = [SHORT[c] for c in CRE_LINES]

# Hand features
EVENT_Z = 2.0                        # event = upward crossing of z = 2 (10 Hz bins)
AC_LAGS = [1, 2, 5, 10, 20, 50]      # bins (~0.1-5 s)
BANDS = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]  # Hz
IEI_EDGES = [0.0, 0.3, 1.0, 3.0, 10.0, 30.0, np.inf]                   # s
EMB_STRIDE = 10

# Classifiers
FEATSETS = ["a", "b", "a+b"]
CLFS = ["logreg", "gboost"]
LR_C = [0.01, 0.1, 1.0]
N_INNER = 4

MLP_KINDS = ["m1", "m2", "m3"]
RIDGE_MODELS = ["b", "d_true", "d_pred_neuron", "d_pred_session", "e"]
MLP_MODELS = ["m1", "m2_true", "m2_pred_neuron", "m2_pred_session", "m2_soft", "m3"]
MODELS = RIDGE_MODELS + MLP_MODELS


def raw_path(info):
    return NPZ / f"{info['id']}_raw.npz"


# ---------------------------------------------------------------- extract
def extract(infos):
    """Step 8 extraction, plus raw dF/F statistics per neuron over the train part
    (first 80% of frames, 30 Hz), saved while the NWB file is still on disk."""
    from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
    load = step5_shared.load_session

    def load_and_raw(info):
        d = load(info)
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{info['id']}.nwb"))
        _, dff = ds.get_dff_traces()
        tr = dff[:, :int(TRAIN_FRAC * dff.shape[1])]
        np.savez(raw_path(info), mean=tr.mean(1), std=tr.std(1), median=np.median(tr, 1),
                 p99=np.percentile(tr, 99, axis=1))
        return d

    step5_shared.load_session = load_and_raw
    for info in infos:
        if (NPZ / f"{info['id']}.npz").exists() and not raw_path(info).exists():
            (NPZ / f"{info['id']}.npz").unlink()  # rebuild, so the raw stats get saved
    step8_medium.extract(infos)


# ---------------------------------------------------------------- ridge (step 8)
def ridge(infos):
    """Step 8 b, d and e per fold (same functions, seeds and draws as step 8 `fit`).
    Keeps only the weights and the e group of each held-out session."""
    with Pool(N_WORKERS) as pool:
        sessions = pool.map(process_cached, infos, chunksize=1)
    true = {s["info"]["id"]: CRE_LINES.index(s["info"]["cre"]) for s in sessions}
    out = {}
    for k in range(N_FOLDS):
        held = [s for s in sessions if s["info"]["fold"] == k]
        train = [s for s in sessions if s["info"]["fold"] != k]
        b = fit_balanced(train)
        d = fit_grouped(train, true)
        e = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            lab = shuffled_groups(train, rng)
            for s in held:
                lab[s["info"]["id"]] = int(rng.integers(N_LINES))
            e.append(({s["info"]["id"]: lab[s["info"]["id"]] for s in held},
                      fit_grouped_balanced(train, lab)))
        out[k] = dict(b=b, d=d, e=e)
        print(f"fold {k}: alpha b {b[2]:g}, d {[d[g][2] for g in range(N_LINES)]}", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(WORK / "ridge.pkl", "wb"))


# ---------------------------------------------------------------- MLP (step 9)
CAPTURED = []


class CapturingMLP(s9.MLP):
    """step9_mlp.MLP that registers itself, so the model trained inside run_job
    (best epoch restored) can be kept."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        CAPTURED.append(self)


def embed(model, si):
    """Mean 2nd-hidden-layer activity of m1 per neuron over the train-part windows
    (first 80% of the recording, stride EMB_STRIDE). The stimulus part of layer 1 is
    shared by all neurons of a window, so it is computed once per window."""
    s = s9.SESS[si]
    z, S_all = s["ztr"], s["Str"]
    starts = window_starts(z.shape[1], EMB_STRIDE)
    W1, b1 = model.l1.weight.detach(), model.l1.bias.detach()
    acc = torch.zeros(s["n"], s9.HIDDEN, dtype=torch.float64)
    with torch.no_grad():
        for a in range(0, len(starts), 128):
            st = starts[a:a + 128]
            X, _ = windows(z, st)
            S = torch.from_numpy(stim_windows(S_all, st).astype(np.float32))
            pre = torch.from_numpy(X.astype(np.float32)) @ W1[:, :PAST].T + (S @ W1[:, PAST:].T + b1)[:, None, :]
            h2 = torch.relu(model.l2(torch.relu(pre)))
            acc += h2.sum(0).double()
    return (acc / len(starts)).numpy().astype(np.float32)


def mlp_job(job):
    CAPTURED.clear()
    r = s9.run_job(job)
    model = CAPTURED[-1]
    r["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if job["kind"] == "m1" and job["seed"] == 0:
        r["emb"] = [embed(model, si) for si in range(len(s9.SESS))]
    return r


def load_sess():
    if not s9.SESS:
        s9.load()


def mlp():
    s9.MLP = CapturingMLP
    load_sess()
    jobs = [j for j in s9.make_jobs() if j["kind"] in MLP_KINDS]
    print(f"{len(jobs)} MLP fits in {N_WORKERS} workers", flush=True)
    results = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(mlp_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}/{len(jobs)}] fold {j['fold']} {j['kind']} seed {j['seed']}:"
                  f" {r['epochs']} epochs, best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    pickle.dump(results, open(WORK / "mlp.pkl", "wb"))


# ---------------------------------------------------------------- features
def hand_features(info):
    """Per-neuron features from the train part only (ztr = first 80%, z-scored with
    train stats, 10 Hz; Str_all = stimulus features of the same bins; raw dF/F stats)."""
    c = np.load(NPZ / f"{info['id']}.npz")
    raw = np.load(raw_path(info))
    z, S = c["ztr"].astype(np.float64), c["Str_all"].astype(np.float64)
    hz = float(c["fr"]) / BIN
    n, T = z.shape
    f = {"dF/F mean": raw["mean"], "dF/F std": raw["std"], "dF/F median": raw["median"],
         "dF/F p99": raw["p99"], "z std (10 Hz)": z.std(1), "skew": skew(z, axis=1),
         "kurtosis": kurtosis(z, axis=1)}
    # Events (threshold crossings) and inter-event intervals (LOLCAT-style IEI histogram)
    above = z > EVENT_Z
    onset = above[:, 1:] & ~above[:, :-1]
    f["event rate (/min)"] = onset.sum(1) / (T / hz / 60)
    f["time above z=2"] = above.mean(1)
    iei = np.zeros((n, len(IEI_EDGES) - 1))
    for i in range(n):
        d = np.diff(np.where(onset[i])[0]) / hz
        if len(d):
            iei[i] = np.histogram(d, IEI_EDGES)[0] / len(d)
    for j in range(len(IEI_EDGES) - 1):
        f[f"IEI {IEI_EDGES[j]:g}-{IEI_EDGES[j + 1]:g} s"] = iei[:, j]
    # Autocorrelation
    zc = z - z.mean(1, keepdims=True)
    var = (zc ** 2).mean(1)
    for lag in AC_LAGS:
        f[f"autocorr {lag / 10:g} s"] = (zc[:, lag:] * zc[:, :-lag]).mean(1) / var
    # Power in frequency bands (fraction of total, log10)
    fq, P = welch(z, fs=hz, nperseg=256, axis=1)
    tot = P[:, fq > 0].sum(1)
    for lo, hi in BANDS:
        sel = (fq > lo) & (fq <= hi)
        f[f"power {lo:g}-{hi:g} Hz"] = np.log10(P[:, sel].sum(1) / tot + 1e-12)
    # Mean response to each stimulus type (weights = fraction of the bin in that epoch)
    for j, name in enumerate(["gratings", "natural movie", "spontaneous"]):
        f[f"mean z: {name}"] = z @ S[j] / S[j].sum()
    blank = S[0] * (1 - S[3])  # inside the grating epoch, no grating shown
    base = z @ blank / blank.sum()
    f["grating on - blank"] = z @ S[3] / S[3].sum() - base
    # Direction tuning: mean z per direction minus blank, plus DSI / OSI (vector sums)
    R = np.stack([z @ S[4 + k] / S[4 + k].sum() - base for k in range(len(DIRECTIONS))], axis=1)
    for k, deg in enumerate(DIRECTIONS):
        f[f"dir {deg}"] = R[:, k]
    th = np.deg2rad(DIRECTIONS)
    Rp = np.clip(R, 0, None)
    norm = Rp.sum(1) + 1e-9
    f["tuning peak"] = R.max(1)
    f["DSI"] = np.abs(Rp @ np.exp(1j * th)) / norm
    f["OSI"] = np.abs(Rp @ np.exp(2j * th)) / norm
    # Correlation with the population mean of the other neurons of the recording
    if n > 1:
        loo = (z.sum(0)[None] - z) / (n - 1)
        lc = loo - loo.mean(1, keepdims=True)
        f["corr with pop mean"] = (zc * lc).mean(1) / np.sqrt(var * (lc ** 2).mean(1))
    else:
        f["corr with pop mean"] = np.zeros(n)
    df = pd.DataFrame(f)
    df.insert(0, "neuron", np.arange(n))
    df.insert(0, "id", info["id"])
    return df


# ---------------------------------------------------------------- classify
def make_clf(clf, C=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    if clf == "logreg":
        return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal"),
                             LogisticRegression(C=C, class_weight="balanced", max_iter=5000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def candidates():
    for fs in FEATSETS:
        for C in LR_C:
            yield fs, "logreg", C
        yield fs, "gboost", None


def classify_fold(args):
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import balanced_accuracy_score as bacc
    from sklearn.model_selection import StratifiedGroupKFold
    from threadpoolctl import threadpool_limits
    k, A, B, y, mouse, fold = args
    X = {"a": A, "b": B, "a+b": np.hstack([A, B])}
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    t0 = time.time()
    with threadpool_limits(1):
        inner = list(StratifiedGroupKFold(N_INNER, shuffle=True, random_state=k)
                     .split(tr, y[tr], mouse[tr]))
        inner_score = {}
        for cand in candidates():
            oof = np.empty(len(tr), int)
            for itr, iva in inner:
                m = make_clf(cand[1], cand[2]).fit(X[cand[0]][tr[itr]], y[tr[itr]])
                oof[iva] = m.predict(X[cand[0]][tr[iva]])
            inner_score[cand] = bacc(y[tr], oof)
        # Per feature set x classifier: logreg C picked by inner CV; fit on all training mice
        res = {}
        for fs in FEATSETS:
            for clf in CLFS:
                cands = [c for c in inner_score if c[0] == fs and c[1] == clf]
                best = max(cands, key=lambda c: inner_score[c])
                m = make_clf(clf, best[2]).fit(X[fs][tr], y[tr])
                r = dict(C=best[2], inner=inner_score[best], proba=m.predict_proba(X[fs][te]))
                if fs == "a":
                    pi = permutation_importance(m, X[fs][te], y[te], scoring="balanced_accuracy",
                                                n_repeats=10, random_state=0)
                    r["perm"] = pi.importances_mean
                    if clf == "logreg":
                        r["coef"] = m[-1].coef_
                res[fs, clf] = r
    sel = max(res, key=lambda key: res[key]["inner"])
    print(f"fold {k}: inner-CV balanced acc " + ", ".join(f"{fs}/{c} {r['inner']:.3f}" for (fs, c), r in res.items())
          + f" -> selected {sel} ({time.time() - t0:.0f} s)", flush=True)
    return k, te, res, sel


def session_vote(pred, proba):
    """Majority vote over the neurons of a recording; ties -> highest mean probability."""
    cnt = np.bincount(pred, minlength=N_LINES)
    tied = np.where(cnt == cnt.max())[0]
    return int(tied[np.argmax(proba.mean(0)[tied])])


def classify(infos):
    from sklearn.metrics import balanced_accuracy_score as bacc
    from sklearn.metrics import confusion_matrix
    load_sess()
    with Pool(N_WORKERS) as pool:
        feats = pd.concat(pool.map(hand_features, infos), ignore_index=True)
    info_by_id = {i["id"]: i for i in infos}
    feats.insert(2, "cre", [info_by_id[i]["cre"] for i in feats.id])
    feats.insert(3, "fold", [info_by_id[i]["fold"] for i in feats.id])
    names = [c for c in feats.columns if c not in ["id", "neuron", "cre", "fold"]]
    A = feats[names].to_numpy(np.float64)
    assert np.isfinite(A).all(), "non-finite hand feature"
    y = np.array([CRE_LINES.index(c) for c in feats.cre])
    mouse, fold = feats.id.to_numpy(), feats.fold.to_numpy()
    # sessions in SESS order == pick_sessions order == feats order
    assert list(feats.drop_duplicates("id").id) == [s["info"]["id"] for s in s9.SESS]

    mlp_res = pickle.load(open(WORK / "mlp.pkl", "rb"))
    emb = {r["job"]["fold"]: np.vstack(r["emb"]) for r in mlp_res if "emb" in r}
    with Pool(min(N_WORKERS, N_FOLDS)) as pool:
        outs = pool.map(classify_fold, [(k, A, emb[k], y, mouse, fold) for k in range(N_FOLDS)])

    # Per-neuron predictions (each neuron is held out exactly once)
    pred = feats[["id", "neuron", "cre", "fold"]].copy()
    pred["true"] = y
    fold_rows, perm, coef = [], [], []
    for k, te, res, sel in outs:
        for (fs, clf), r in res.items():
            tag = f"{fs}_{clf}"
            p = r["proba"]
            for g in range(N_LINES):
                pred.loc[te, f"p_{tag}_{LINES[g]}"] = p[:, g]
            pred.loc[te, f"pred_{tag}"] = p.argmax(1)
            if "perm" in r:
                perm.append(pd.DataFrame(dict(feature=names, fold=k, clf=clf, importance=r["perm"])))
            if "coef" in r:
                for g in range(N_LINES):
                    coef.append(pd.DataFrame(dict(feature=names, fold=k, line=LINES[g], coef=r["coef"][g])))
            fold_rows.append(dict(fold=k, featset=fs, clf=clf, C=r["C"], inner_bacc=r["inner"],
                                  selected=(fs, clf) == sel))
        pred.loc[te, "selected"] = f"{sel[0]}_{sel[1]}"
    for c in pred.columns:
        if c.startswith("pred_"):
            pred[c] = pred[c].astype(int)
    for fs in FEATSETS:
        for clf in CLFS:
            tag = f"{fs}_{clf}"
            P = pred[[f"p_{tag}_{l}" for l in LINES]].to_numpy()
            pred[f"session_{tag}"] = 0
            for mid, g in pred.groupby("id"):
                pred.loc[g.index, f"session_{tag}"] = session_vote(g[f"pred_{tag}"].to_numpy(), P[g.index])
    # Selected classifier of each fold -> the labels used for forecasting
    pred["pred_sel"] = [pred.loc[i, f"pred_{t}"] for i, t in zip(pred.index, pred.selected)]
    pred["session_sel"] = [pred.loc[i, f"session_{t}"] for i, t in zip(pred.index, pred.selected)]
    for g in range(N_LINES):
        pred[f"p_sel_{LINES[g]}"] = [pred.loc[i, f"p_{t}_{LINES[g]}"] for i, t in zip(pred.index, pred.selected)]

    # Held-out accuracy per fold and pooled
    fr = pd.DataFrame(fold_rows)
    for i, r in fr.iterrows():
        sub = pred[pred.fold == r.fold]
        ses = sub.drop_duplicates("id")
        tag = f"{r.featset}_{r.clf}"
        fr.loc[i, "heldout_bacc_neuron"] = bacc(sub.true, sub[f"pred_{tag}"])
        fr.loc[i, "heldout_acc_session"] = (ses.true == ses[f"session_{tag}"]).mean()
    fr.to_csv(OUT / "classifier_folds.csv", index=False, float_format="%.5g")
    print("\n== Held-out classifier accuracy (pooled over the 46 held-out mice) ==")
    ses = pred.drop_duplicates("id")
    for tag in [f"{fs}_{clf}" for fs in FEATSETS for clf in CLFS] + ["sel"]:
        bn = bacc(pred.true, pred[f"pred_{tag}"])
        bs = bacc(ses.true, ses[f"session_{tag}"])
        print(f"  {tag:14s} neuron balanced acc {bn:.3f} | session acc {(ses.true == ses[f'session_{tag}']).mean():.3f}"
              f" ({(ses.true == ses[f'session_{tag}']).sum()}/{len(ses)}), session balanced acc {bs:.3f}")
        print("    confusion (rows true, cols pred, neurons):",
              confusion_matrix(pred.true, pred[f"pred_{tag}"], labels=range(N_LINES)).tolist())
    print("selected per fold:", fr[fr.selected].set_index("fold")[["featset", "clf", "C"]].to_dict("index"))

    # Feature summary: per-line mean / std, Kruskal-Wallis across lines, held-out
    # permutation importance (mean over folds), logreg coefficients (mean over folds)
    rows = []
    perm, coef = pd.concat(perm), pd.concat(coef)
    for j, nm in enumerate(names):
        d = dict(feature=nm, kruskal_H=kruskal(*[A[y == g, j] for g in range(N_LINES)]).statistic)
        for g in range(N_LINES):
            d[f"mean_{LINES[g]}"] = A[y == g, j].mean()
            d[f"std_{LINES[g]}"] = A[y == g, j].std()
        for clf in CLFS:
            v = perm[(perm.feature == nm) & (perm.clf == clf)].importance
            d[f"perm_{clf}_mean"], d[f"perm_{clf}_std"] = v.mean(), v.std()
        for g in range(N_LINES):
            d[f"coef_{LINES[g]}"] = coef[(coef.feature == nm) & (coef.line == LINES[g])].coef.mean()
        rows.append(d)
    pd.DataFrame(rows).to_csv(OUT / "features_summary.csv", index=False, float_format="%.5g")
    feats.to_csv(OUT / "neuron_features.csv", index=False, float_format="%.5g")
    pred.to_csv(OUT / "neuron_predictions.csv", index=False, float_format="%.4g")
    print(f"Saved {OUT}/classifier_folds.csv, features_summary.csv, neuron_features.csv, neuron_predictions.csv")


# ---------------------------------------------------------------- forecast
def film_forward(model, x, e):
    """step9_mlp.MLP.forward with the label embedding given directly (rows of e)."""
    g1, b1, g2, b2 = model.gb(e).chunk(4, dim=1)
    h1 = torch.relu(model.l1(x)) * (1 + g1) + b1
    h2 = torch.relu(model.l2(h1)) * (1 + g2) + b2
    return model.out(h2)


def mlp_forecast(model, X, S, E=None):
    """X (nw, n, PAST), S (nw, 480), E (n, EMB) per-neuron FiLM embedding or None."""
    nw, n = X.shape[:2]
    P = np.empty((nw, n, FUTURE), np.float32)
    with torch.no_grad():
        for a in range(0, nw, 64):
            Xc, Sc = X[a:a + 64], S[a:a + 64]
            D = torch.from_numpy(np.concatenate([Xc.reshape(-1, PAST), np.repeat(Sc, n, axis=0)], axis=1))
            out = model(D) if E is None else film_forward(model, D, E.repeat(len(Xc), 1))
            P[a:a + 64] = out.numpy().reshape(len(Xc), n, FUTURE)
    return P


def load_model(state, film):
    m = s9.MLP(N_LINES if film else 0)
    m.load_state_dict(state)
    return m.eval()


def compare(df, a, b):
    """Paired comparison of per-mouse scores: A beats B count, median diff, Wilcoxon p."""
    diff = df[f"score_{a}"] - df[f"score_{b}"]
    try:
        p = wilcoxon(df[f"score_{a}"], df[f"score_{b}"]).pvalue
    except ValueError:  # all differences zero
        p = np.nan
    return dict(n=len(df), a_beats_b=int((diff > 0).sum()), ties=int((diff == 0).sum()),
                median_diff=diff.median(), mean_diff=diff.mean(), p=p)


def recovered(df, pred, true, none, rng, n_boot=2000):
    g_p, g_t = (df[f"score_{pred}"] - df[f"score_{none}"]).to_numpy(), (df[f"score_{true}"] - df[f"score_{none}"]).to_numpy()
    frac = g_p.mean() / g_t.mean()
    idx = rng.integers(len(df), size=(n_boot, len(df)))
    boot = g_p[idx].mean(1) / g_t[idx].mean(1)
    return frac, np.percentile(boot, 2.5), np.percentile(boot, 97.5), g_p.mean(), g_t.mean()


def forecast(infos):
    torch.set_num_threads(N_WORKERS)
    load_sess()
    ridge_m = pickle.load(open(WORK / "ridge.pkl", "rb"))
    mlp_res = pickle.load(open(WORK / "mlp.pkl", "rb"))
    pred = pd.read_csv(OUT / "neuron_predictions.csv")
    m2 = {}  # fold -> list of (seed, model)
    run_out = {}  # (mouse id, kind) -> list of run_job outputs (step 9 test evaluation)
    for r in mlp_res:
        j = r["job"]
        if j["kind"] == "m2":
            m2.setdefault(j["fold"], []).append((j["seed"], load_model(r["state"], True)))
        for mid, v in r["out"].items():
            run_out.setdefault((mid, j["kind"]), []).append(v["sse"])
    keep = ["copy", "b", "d_true", "d_pred_neuron", "m1", "m2_true", "m2_pred_neuron", "m2_soft"]
    m1_models = {r["job"]["fold"]: [] for r in mlp_res}
    for r in mlp_res:
        if r["job"]["kind"] == "m1":
            m1_models[r["job"]["fold"]].append(load_model(r["state"], False))

    rows, neuron_rows, traces = [], [], {}
    for s in s9.SESS:
        info, mid, k = s["info"], s["info"]["id"], s["info"]["fold"]
        starts = window_starts(s["zte"].shape[1], FUTURE)
        X, Y = windows(s["zte"], starts)
        S = stim_windows(s["Ste"], starts)
        X64, S64, Y64 = X.astype(np.float64), S.astype(np.float64), Y.astype(np.float64)
        p = pred[pred.id == mid].sort_values("neuron")
        lab_n = p.pred_sel.to_numpy()
        lab_s = int(p.session_sel.iloc[0])
        prob = p[[f"p_sel_{l}" for l in LINES]].to_numpy()
        true = s["line"]
        P = {"copy": np.repeat(X64[:, :, -1:], FUTURE, axis=2)}
        rm = ridge_m[k]
        P["b"] = predict(rm["b"][0], rm["b"][1], X64, S64)
        Pd = [predict(rm["d"][g][0], rm["d"][g][1], X64, S64) for g in range(N_LINES)]
        P["d_true"], P["d_pred_session"] = Pd[true], Pd[lab_s]
        P["d_pred_neuron"] = np.stack([Pd[lab_n[i]][:, i] for i in range(s["n"])], axis=1)
        sse = {m: ((v - Y64) ** 2).sum(axis=(0, 2)) for m, v in P.items()}  # per neuron
        sse["e"] = np.mean([((predict(fe[lab[mid]][0], fe[lab[mid]][1], X64, S64) - Y64) ** 2).sum(axis=(0, 2))
                            for lab, fe in rm["e"]], axis=0)
        # MLPs: m1 / m3 test SSE from run_job (step 9 evaluation); m1 also recomputed per neuron
        Pm1 = [mlp_forecast(m, X, S) for m in m1_models[k]]
        sse["m1"] = np.mean([((v - Y) ** 2).sum(axis=(0, 2)).astype(np.float64) for v in Pm1], axis=0)
        P["m1"] = np.mean(Pm1, axis=0)
        variants = {"m2_true": np.full(s["n"], true), "m2_pred_neuron": lab_n,
                    "m2_pred_session": np.full(s["n"], lab_s), "m2_soft": prob}
        for name, lab in variants.items():
            preds, sses = [], []
            for seed, m in sorted(m2[k], key=lambda t: t[0]):
                W = m.emb.weight.detach()
                E = torch.from_numpy(lab.astype(np.float32)) @ W if lab.ndim == 2 else W[torch.from_numpy(lab)]
                Pv = mlp_forecast(m, X, S, E)
                preds.append(Pv)
                sses.append(((Pv - Y) ** 2).sum(axis=(0, 2)).astype(np.float64))
            sse[name] = np.mean(sses, axis=0)
            if name in keep:
                P[name] = np.mean(preds, axis=0)
        # Check: per-mouse SSE equals the step 9 run_job evaluation (true label m2, m1)
        for kind, name in [("m1", "m1"), ("m2", "m2_true")]:
            ref = np.mean(run_out[mid, kind])
            assert abs(sse[name].sum() - ref) < 1e-3 * ref, (mid, kind, sse[name].sum(), ref)
        sse["m3"] = np.full(s["n"], np.mean(run_out[mid, "m3"]) / s["n"])  # mouse-level only
        n_el = Y.size
        d = dict(cre=info["cre"], line=LINES[true], fold=k, id=mid, depth=info["depth"], n_neurons=s["n"],
                 n_el=n_el, sse_copy=sse["copy"].sum(), session_pred=LINES[lab_s],
                 frac_neurons_correct=(lab_n == true).mean())
        for m in MODELS:
            d[f"sse_{m}"] = sse[m].sum()
            d[f"score_{m}"] = 1 - d[f"sse_{m}"] / d["sse_copy"]
        rows.append(d)
        nel = Y.shape[0] * FUTURE
        for i in range(s["n"]):
            nr = dict(id=mid, neuron=i, line=LINES[true], pred=LINES[lab_n[i]], session_pred=LINES[lab_s],
                      mse_copy=sse["copy"][i] / nel)
            for m in ["b", "d_true", "d_pred_neuron", "d_pred_session", "e", "m1", "m2_true", "m2_pred_neuron",
                      "m2_pred_session", "m2_soft"]:
                nr[f"score_{m}"] = 1 - sse[m][i] / sse["copy"][i]
            neuron_rows.append(nr)
        traces[mid] = {m: P[m].astype(np.float32) for m in keep} | {"y": Y}
        print(f"{info['cre']:14s} fold {k} {mid} ({s['n']:3d} neurons) session pred {LINES[lab_s]:5s}"
              f" neurons correct {d['frac_neurons_correct']:.2f} | score b {d['score_b']:.3f} d {d['score_d_true']:.3f}"
              f" d-pn {d['score_d_pred_neuron']:.3f} d-ps {d['score_d_pred_session']:.3f} | m1 {d['score_m1']:.3f}"
              f" m2 {d['score_m2_true']:.3f} m2-pn {d['score_m2_pred_neuron']:.3f} m2-ps {d['score_m2_pred_session']:.3f}"
              f" m2-soft {d['score_m2_soft']:.3f} m3 {d['score_m3']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "mice_scores.csv", index=False, float_format="%.6g")
    nd = pd.DataFrame(neuron_rows)
    nd.to_csv(OUT / "neuron_scores.csv", index=False, float_format="%.5g")

    # Fold-pooled scores (as steps 8-9): score from summed SSE over the held-out mice of a line in a fold
    groups = CRE_LINES + ["all lines"]
    fold_rows = []
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        for kf in range(N_FOLDS):
            f = sub[sub.fold == kf]
            for m in MODELS:
                fold_rows.append(dict(line=cre, fold=kf, model=m, score=1 - f[f"sse_{m}"].sum() / f.sse_copy.sum(),
                                      mse=f[f"sse_{m}"].sum() / f.n_el.sum()))
    fs = pd.DataFrame(fold_rows)
    fs.to_csv(OUT / "fold_scores.csv", index=False, float_format="%.6g")
    print("\n== Fold-pooled score (mean +- std over 5 folds) | per-mouse mean score ==")
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        print(f"  {cre} ({len(sub)} mice)")
        for m in MODELS:
            v = fs[(fs.line == cre) & (fs.model == m)]
            print(f"    {m:16s} fold score {v.score.mean():.3f} +- {v.score.std(ddof=0):.3f}"
                  f" | MSE {v.mse.mean():.4f} +- {v.mse.std(ddof=0):.4f} | mouse mean score {sub[f'score_{m}'].mean():.4f}")

    # Paired tests over mice
    pairs = [("d_true", "b"), ("d_pred_neuron", "b"), ("d_pred_session", "b"), ("d_pred_neuron", "d_true"),
             ("d_pred_session", "d_true"), ("d_pred_neuron", "e"), ("d_pred_session", "e"), ("d_true", "e"),
             ("m2_true", "m1"), ("m2_pred_neuron", "m1"), ("m2_pred_session", "m1"), ("m2_soft", "m1"),
             ("m2_pred_neuron", "m2_true"), ("m2_pred_session", "m2_true"), ("m2_soft", "m2_true"),
             ("m2_pred_neuron", "m3"), ("m2_pred_session", "m3"), ("m2_soft", "m3"), ("m2_true", "m3")]
    st = []
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        for a, b in pairs:
            st.append(dict(line=cre, a=a, b=b, **compare(sub, a, b)))
    st = pd.DataFrame(st)
    st.to_csv(OUT / "stats.csv", index=False, float_format="%.4g")
    print("\n== Paired Wilcoxon over mice (score; A beats B = higher score) ==")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # Fraction of the true-label gain recovered
    rng = np.random.default_rng(0)
    gr = []
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        for fam, none, true, preds in [("ridge", "b", "d_true", ["d_pred_neuron", "d_pred_session"]),
                                       ("MLP", "m1", "m2_true", ["m2_pred_neuron", "m2_pred_session", "m2_soft"])]:
            for pm in preds:
                frac, lo, hi, gp, gt = recovered(sub, pm, true, none, rng)
                v = {m: fs[(fs.line == cre) & (fs.model == m)].score.mean() for m in [none, true, pm]}
                gr.append(dict(line=cre, family=fam, pred=pm, gain_pred=gp, gain_true=gt, frac=frac, ci_lo=lo, ci_hi=hi,
                               frac_foldpooled=(v[pm] - v[none]) / (v[true] - v[none])))
    gr = pd.DataFrame(gr)
    gr.to_csv(OUT / "gain_recovered.csv", index=False, float_format="%.4g")
    print("\n== Fraction of the true-label gain recovered: mean(pred - none) / mean(true - none) over mice ==")
    print(gr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # Example neurons: per line, the correctly labelled neuron with the largest true-label
    # gain (m1 -> m2_true) and the mislabelled neuron with the largest loss (m2_true -> m2_pred_neuron),
    # each over the 60 s of test with the highest variance of the true trace
    ex = []
    for line in LINES:
        sub = nd[nd.line == line]
        ok, bad = sub[sub.pred == line], sub[sub.pred != line]
        if len(ok):
            ex.append(("correct", ok.loc[(ok.score_m2_true - ok.score_m1).idxmax()]))
        if len(bad):
            ex.append(("mislabelled", bad.loc[(bad.score_m2_true - bad.score_m2_pred_neuron).idxmax()]))
    tr_rows = []
    seg = 600  # bins
    for kind, r in ex:
        T = traces[r.id]
        i = int(r.neuron)
        series = {m: T[m][:, i].reshape(-1) for m in ["y"] + keep}
        nb = len(series["y"])
        a = max(range(0, max(nb - seg, 0) + 1, FUTURE), key=lambda a: series["y"][a:a + seg].var())
        for t in range(a, min(a + seg, nb)):
            # bin = index in the test part (forecast windows start at bin PAST, stride FUTURE)
            tr_rows.append(dict(id=r.id, neuron=i, line=r.line, pred=r.pred, kind=kind, bin=PAST + t,
                                **{m: series[m][t] for m in series}))
    pd.DataFrame(tr_rows).to_csv(OUT / "example_traces.csv", index=False, float_format="%.4g")
    print(f"Saved {OUT}/mice_scores.csv, neuron_scores.csv, fold_scores.csv, stats.csv, gain_recovered.csv,"
          " example_traces.csv")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    infos = pick_sessions()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    phases = ["extract", "ridge", "mlp", "classify", "forecast"] if phase == "all" else [phase]
    for ph in phases:
        print(f"\n######## phase {ph} ########", flush=True)
        t0 = time.time()
        {"extract": lambda: extract(infos), "ridge": lambda: ridge(infos), "mlp": mlp,
         "classify": lambda: classify(infos), "forecast": lambda: forecast(infos)}[ph]()
        print(f"phase {ph} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
