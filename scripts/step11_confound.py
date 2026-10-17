"""Step 11: does the activity classifier recognise the neuron, or the recording?

Each Allen Visual Coding 2P recording holds a single Cre line, so the step 10
classifier (92.5% balanced accuracy per neuron, 46/46 sessions) may read
recording-level cues (dF/F scale, population synchrony, neuron count, depth)
instead of the neuron's own behaviour. This step splits the features into
single-neuron, scale-free (S) and recording-level (R) groups and repeats the
step 10 classifier and forecasting tests with them.

Same 46 sessions (VISp, three_session_A, Cux2 / Sst / Vip / Pvalb, one per mouse),
5 leave-mice-out folds, chronological 80/20 split, inputs and test windows as
steps 8-10. Intermediate files go to data/step11/ (not committed).

Phases (`python scripts/step11_confound.py <phase>`, in this order; `all` runs
them all; run `ridge` with OMP_NUM_THREADS=1):
- `extract`: step 10's extract (step 8 cache + raw dF/F statistics), only if
  data/cache is missing sessions.
- `ridge`: step 10's `ridge` (step 8 b, d and e, same functions, seeds and draws)
  -> data/step11/ridge.pkl. Step 8's per-neuron model a, which b / d / e never
  use, is skipped: `process_grams` builds the same Gram sums and val windows as
  step5_shared.process_session in the same order, without fitting a.
- `mlp`: step 10's `mlp` (step 9 run_job, m1 / m2 / m3, same seeds) ->
  data/step11/mlp.pkl, with the m1 seed-0 embedding (feature set b).
- `classify` (task 1): feature groups from results/step10/neuron_features.csv
    S = single-neuron, scale-free (37 features: z std, skew, kurtosis, event rate,
        time above z = 2, IEI histogram, autocorrelations, power bands, mean z per
        stimulus, grating on - blank, 8 directions, tuning peak, DSI, OSI)
    R = recording-level (dF/F mean, std, median, p99, corr with pop mean, number of
        neurons in the session, imaging depth)
    A = all 42 hand features (step 10 set a)
    b = m1 embedding (step 10 set b; sees only the neuron's own z-scored past and
        the stimulus, so it counts as single-neuron), S+b, and A+b (step 10's
        best set, as a reproduction check)
  Same classifiers (quantile -> logistic regression with C in {0.01, 0.1, 1};
  histogram gradient boosting), balanced class weights and 4-fold inner CV over
  the training mice (StratifiedGroupKFold, groups = mice) as step 10. Per feature
  set, inner CV picks C and the classifier per outer fold.
- `sessions` (task 2): one row per session with recording summaries only (number
  of neurons, depth, mean / median pairwise correlation, mean / median correlation
  with the population mean, median dF/F std and mean); same classifiers, tested
  on the 5 outer folds and leave-one-mouse-out.
- `forecast` (task 3): per neuron, test SSE under each of the 4 line labels
  (ridge d with each line's model; MLP m2 with each line's FiLM embedding, mean
  over 3 seeds) and with no label (b, m1), plus the trained-shuffled controls
  (e, m3) -> results/step11/neuron_scores_all_labels.csv. Checks that b, d, e,
  m1, m2 and m3 reproduce step 10's per-mouse scores.
- `assign` (task 3): scores label assignments from CSV files only: true,
  per-neuron and session-vote labels of the S, S+b, R, A classifiers, step 10's
  selected classifier (a+b), and shuffled labels (true labels permuted across
  all neurons, 20 seeds). Fraction of the true-label gain recovered =
  mean(pred - none) / mean(true - none) over mice, 95% bootstrap CI over mice
  (2,000 resamples), Wilcoxon signed-rank over mice (pred vs none, pred vs true).
- `windows` (task 4): S features from windows of 3 s, 30 s, 3 min (10
  non-overlapping random windows per neuron) and the full train part, train part
  only; the S classifier is trained and tested per window length.

Ideas from the literature (not run): LOLCAT (Schneider et al. 2023) classifies
3 s snippets of activity and pools them, which motivates the window-length test;
NeuPRINT (Mi et al. 2023) separates a neuron's own time-invariant representation
from population-level statistics given as context, which is the S / R split here;
NuCLR and NeuPRINT report much lower accuracy on mixed-type recordings (Bugeon
et al. 2022), where the recording cannot reveal the type; de Vries et al. 2020
describe the dataset design (one Cre line and reporter per recording).
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
from scipy.stats import kurtosis, skew, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step10_inferred as s10
import step9_mlp as s9
from step3_linear import FUTURE, ROOT, window_starts, windows
from step4_stimulus import DIRECTIONS, stim_windows
from step5_shared import Gram, neuron_gram, predict, stim_stats, train_windows
from step8_medium import CRE_LINES, N_FOLDS, NPZ, load_cached, pick_sessions
from step10_inferred import (AC_LAGS, BANDS, EVENT_Z, IEI_EDGES, LINES, LR_C, N_INNER, N_LINES,
                             compare, load_model, mlp_forecast, recovered, session_vote)

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step11"
OUT = ROOT / "results/step11"
STEP10 = ROOT / "results/step10"
s10.WORK = WORK  # step 10's ridge / mlp phases write their pickles here
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
CLFS = ["logreg", "gboost"]
META = ["id", "neuron", "cre", "fold"]

R_HAND = ["dF/F mean", "dF/F std", "dF/F median", "dF/F p99", "corr with pop mean"]
R_META = ["n neurons", "depth"]
FEATSETS = ["S", "R", "A", "b", "S+b", "A+b"]

# Task 2: session summaries
SESS_FEATS = ["n neurons", "depth", "pairwise corr mean", "pairwise corr median", "pop corr mean",
              "pop corr median", "dF/F std median", "dF/F mean median"]
SESS_SUBSETS = {"all 8": SESS_FEATS, "n neurons + depth": ["n neurons", "depth"],
                "correlations": SESS_FEATS[2:6], "dF/F scale": SESS_FEATS[6:]}

# Task 4: window lengths in 10 Hz bins (None = full train part) and windows per neuron
WINDOWS = [("3 s", 30), ("30 s", 300), ("3 min", 1800), ("full train part", None)]
N_WIN = 10

N_SHUFFLE = 20
N_BOOT = 2000


# ---------------------------------------------------------------- ridge / mlp (step 10)
def process_grams(info):
    """step5_shared.process_session on the step 8 cache without model a: the same
    Gram sums (added in the same order) and val windows, which is all b, d, e use."""
    d = load_cached(info)
    ztr, n_fit = d["ztr"], d["n_fit"]
    Xf, Yf = train_windows(ztr[:, :n_fit])
    Xv, Yv = train_windows(ztr[:, n_fit:])
    Sf = stim_windows(d["Str_all"][:, :n_fit], window_starts(n_fit, 1))
    Sv = stim_windows(d["Str_all"][:, n_fit:], window_starts(ztr.shape[1] - n_fit, 1))
    stf, stv = stim_stats(Sf), stim_stats(Sv)
    gfit, gval = Gram(), Gram()
    for i in range(d["n_neurons"]):
        gf, gv = neuron_gram(Xf[i], Yf[i], Sf, stf), neuron_gram(Xv[i], Yv[i], Sv, stv)
        gfit, gval = gfit + gf, gval + gv
    return dict(info=info, n_neurons=d["n_neurons"], gfit=gfit, gval=gval,
                Xv=np.ascontiguousarray(Xv.transpose(1, 0, 2)),
                Yv=np.ascontiguousarray(Yv.transpose(1, 0, 2)), Sv=Sv)


def ridge(infos):
    s10.process_cached = process_grams
    s10.ridge(infos)


# ---------------------------------------------------------------- classifiers
def make_clf(clf, C=None, impute=False):
    """Step 10 classifiers; with impute, logreg first fills missing features with
    the training median and adds missing-indicator columns (gboost handles NaN)."""
    m = s10.make_clf(clf, C)
    if impute and clf == "logreg":
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), *[s for _, s in m.steps])
    return m


def candidates():
    for C in LR_C:
        yield "logreg", C
    yield "gboost", None


def fit_featset(X, y, groups, tr, te, seed, impute=False, perm=False):
    """Step 10 model selection for one feature set: 4-fold inner CV over the
    training groups (mice) picks C and the classifier by balanced accuracy; both
    classifiers are refitted on all training rows and predict the test rows."""
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import balanced_accuracy_score as bacc
    from sklearn.model_selection import StratifiedGroupKFold
    inner = list(StratifiedGroupKFold(N_INNER, shuffle=True, random_state=seed).split(tr, y[tr], groups[tr]))
    score = {}
    for cand in candidates():
        oof = np.empty(len(tr), int)
        for itr, iva in inner:
            oof[iva] = make_clf(*cand, impute).fit(X[tr[itr]], y[tr[itr]]).predict(X[tr[iva]])
        score[cand] = bacc(y[tr], oof)
    res = {}
    for clf in CLFS:
        best = max([c for c in score if c[0] == clf], key=lambda c: score[c])
        m = make_clf(*best, impute).fit(X[tr], y[tr])
        r = dict(C=best[1], inner=score[best], proba=m.predict_proba(X[te]))
        if perm:
            r["perm"] = permutation_importance(m, X[te], y[te], scoring="balanced_accuracy", n_repeats=10,
                                               random_state=0).importances_mean
        res[clf] = r
    sel = max(CLFS, key=lambda c: res[c]["inner"])
    return res, sel


def bacc(true, pred):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(true, pred)


def confusion(true, pred):
    C = np.zeros((N_LINES, N_LINES), int)
    np.add.at(C, (np.asarray(true), np.asarray(pred)), 1)
    return C


# ---------------------------------------------------------------- task 1
def load_features(infos):
    feats = pd.read_csv(STEP10 / "neuron_features.csv")
    by_id = {i["id"]: i for i in infos}
    feats["n neurons"] = feats.id.map(feats.groupby("id").size())
    feats["depth"] = [by_id[i]["depth"] for i in feats.id]
    hand = [c for c in feats.columns if c not in META + R_META]
    assert len(hand) == 42, len(hand)
    return feats, hand


def load_embedding(feats):
    """m1 seed-0 embedding of every neuron, per fold (rows in feats order)."""
    s10.load_sess()
    assert list(feats.drop_duplicates("id").id) == [s["info"]["id"] for s in s9.SESS]
    mlp_res = pickle.load(open(WORK / "mlp.pkl", "rb"))
    return {r["job"]["fold"]: np.vstack(r["emb"]) for r in mlp_res if "emb" in r}


def classify_fold(args):
    from threadpoolctl import threadpool_limits
    k, X, Bk, y, mouse, fold = args
    X = dict(X)
    if Bk is not None:
        X["b"], X["S+b"], X["A+b"] = Bk, np.hstack([X["S"], Bk]), np.hstack([X["A"], Bk])
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    out, t0 = {}, time.time()
    with threadpool_limits(1):
        for fs in [f for f in FEATSETS if f in X]:
            out[fs] = fit_featset(X[fs], y, mouse, tr, te, seed=k, perm=fs in ["S", "R", "A"])
    print(f"fold {k}: " + ", ".join(f"{fs} {r[1]} (inner {r[0][r[1]]['inner']:.3f})" for fs, r in out.items())
          + f" ({time.time() - t0:.0f} s)", flush=True)
    return k, te, out


def classify(infos):
    feats, hand = load_features(infos)
    S = [c for c in hand if c not in R_HAND]
    groups = {"S": S, "R": R_HAND + R_META, "A": hand}
    assert len(S) == 37
    X = {fs: feats[cols].to_numpy(np.float64) for fs, cols in groups.items()}
    y = np.array([CRE_LINES.index(c) for c in feats.cre])
    mouse, fold = feats.id.to_numpy(), feats.fold.to_numpy()
    emb = load_embedding(feats) if (WORK / "mlp.pkl").exists() else None
    if emb is None:
        print("WARNING: data/step11/mlp.pkl missing, feature sets b, S+b and A+b skipped", flush=True)
    with Pool(min(N_WORKERS, N_FOLDS)) as pool:
        outs = pool.map(classify_fold, [(k, X, None if emb is None else emb[k], y, mouse, fold)
                                        for k in range(N_FOLDS)])
    fsets = list(outs[0][2])

    pred = feats[META].copy()
    pred["true"] = y
    fold_rows, imp = [], []
    for k, te, out in outs:
        for fs, (res, sel) in out.items():
            for clf, r in res.items():
                for g in range(N_LINES):
                    pred.loc[te, f"p_{fs}_{clf}_{LINES[g]}"] = r["proba"][:, g]
                pred.loc[te, f"pred_{fs}_{clf}"] = r["proba"].argmax(1)
                fold_rows.append(dict(fold=k, featset=fs, clf=clf, C=r["C"], inner_bacc=r["inner"],
                                      selected=clf == sel))
                if "perm" in r:
                    imp.append(pd.DataFrame(dict(featset=fs, clf=clf, fold=k, feature=groups[fs],
                                                 importance=r["perm"])))
            pred.loc[te, f"sel_{fs}"] = sel
    for fs in fsets:
        tags = [f"{fs}_{c}" for c in CLFS] + [f"{fs}_sel"]
        for g in range(N_LINES):
            pred[f"p_{fs}_sel_{LINES[g]}"] = np.where(pred[f"sel_{fs}"] == "logreg", pred[f"p_{fs}_logreg_{LINES[g]}"],
                                                      pred[f"p_{fs}_gboost_{LINES[g]}"])
        pred[f"pred_{fs}_sel"] = pred[[f"p_{fs}_sel_{l}" for l in LINES]].to_numpy().argmax(1)
        for tag in tags:
            pred[f"pred_{tag}"] = pred[f"pred_{tag}"].astype(int)
            P = pred[[f"p_{tag}_{l}" for l in LINES]].to_numpy()
            pred[f"session_{tag}"] = 0
            for _, g in pred.groupby("id"):
                pred.loc[g.index, f"session_{tag}"] = session_vote(g[f"pred_{tag}"].to_numpy(), P[g.index])

    # Summary: pooled held-out balanced accuracy (neurons), per fold, session vote
    fr = pd.DataFrame(fold_rows)
    rows, conf = [], []
    ses = pred.drop_duplicates("id")
    for fs in fsets:
        for clf in CLFS + ["sel"]:
            tag = f"{fs}_{clf}"
            per_fold = [bacc(pred.true[pred.fold == k], pred[f"pred_{tag}"][pred.fold == k]) for k in range(N_FOLDS)]
            inner = fr[(fr.featset == fs) & (fr.selected if clf == "sel" else fr.clf == clf)].inner_bacc
            frac = pred.assign(ok=pred.true == pred[f"pred_{tag}"]).groupby("id").ok.mean()
            rows.append(dict(featset=fs, clf=clf, inner_bacc_mean=inner.mean(),
                             bacc_neuron=bacc(pred.true, pred[f"pred_{tag}"]),
                             bacc_fold_mean=np.mean(per_fold), bacc_fold_std=np.std(per_fold),
                             **{f"bacc_fold{k}": v for k, v in enumerate(per_fold)},
                             session_correct=int((ses.true == ses[f"session_{tag}"]).sum()), n_sessions=len(ses),
                             mouse_frac_correct_median=frac.median(), mouse_frac_correct_min=frac.min(),
                             **{f"recall_{LINES[g]}": (pred[f"pred_{tag}"][pred.true == g] == g).mean()
                                for g in range(N_LINES)}))
            for level, d in [("neuron", pred), ("session", ses)]:
                col = f"pred_{tag}" if level == "neuron" else f"session_{tag}"
                C = confusion(d.true, d[col])
                for i in range(N_LINES):
                    for j in range(N_LINES):
                        conf.append(dict(featset=fs, clf=clf, level=level, true=LINES[i], pred=LINES[j],
                                         count=int(C[i, j])))
    summ = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summ.to_csv(OUT / "classifier_summary.csv", index=False, float_format="%.5g")
    pd.DataFrame(conf).to_csv(OUT / "confusions.csv", index=False)
    fr.to_csv(OUT / "classifier_folds.csv", index=False, float_format="%.5g")
    imp = pd.concat(imp).groupby(["featset", "clf", "feature"], sort=False).importance.agg(["mean", "std"])
    imp.reset_index().to_csv(OUT / "feature_importance.csv", index=False, float_format="%.5g")
    keep = META + ["true"] + [c for c in pred.columns if c.startswith(("sel_", "pred_", "session_", "p_"))
                              and ("_sel" in c or c.startswith(("pred_", "sel_", "session_")))]
    pred[keep].to_csv(OUT / "neuron_predictions.csv", index=False, float_format="%.4g")
    print("\n== Task 1: held-out classifier accuracy (46 mice) ==")
    print(summ[["featset", "clf", "inner_bacc_mean", "bacc_neuron", "bacc_fold_mean", "bacc_fold_std",
                "session_correct", "mouse_frac_correct_median"] + [f"recall_{l}" for l in LINES]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for fs in fsets:
        print(f"  {fs} selected per fold: {fr[(fr.featset == fs) & fr.selected].clf.tolist()}"
              f" | neuron confusion {confusion(pred.true, pred[f'pred_{fs}_sel']).tolist()}")
    print(f"Saved {OUT}/classifier_summary.csv, confusions.csv, classifier_folds.csv, feature_importance.csv,"
          " neuron_predictions.csv")


# ---------------------------------------------------------------- task 2
def session_table(infos):
    import json
    from step3_linear import CACHE
    feats, _ = load_features(infos)
    exps = {e["id"]: e for e in json.load(open(CACHE / "ophys_experiments.json"))}
    rows = []
    for info in infos:
        z = np.load(NPZ / f"{info['id']}.npz")["ztr"].astype(np.float64)
        f = feats[feats.id == info["id"]]
        cc = np.corrcoef(z)[np.triu_indices(len(z), 1)]
        tl = exps[info["id"]]["specimen"]["donor"]["transgenic_lines"]
        rows.append({"id": info["id"], "cre": info["cre"], "line": LINES[CRE_LINES.index(info["cre"])],
                     "fold": info["fold"], "reporter": ";".join(t["name"] for t in tl
                                                                 if t["transgenic_line_type_name"] == "reporter"),
                     "n neurons": len(z), "depth": info["depth"], "pairwise corr mean": cc.mean(),
                     "pairwise corr median": np.median(cc), "pop corr mean": f["corr with pop mean"].mean(),
                     "pop corr median": f["corr with pop mean"].median(),
                     "dF/F std median": f["dF/F std"].median(), "dF/F mean median": f["dF/F mean"].median()})
    return pd.DataFrame(rows)


def sessions(infos):
    from threadpoolctl import threadpool_limits
    st = session_table(infos)
    st.to_csv(OUT / "session_features.csv", index=False, float_format="%.5g")
    y = np.array([CRE_LINES.index(c) for c in st.cre])
    mouse, fold = st.id.to_numpy(), st.fold.to_numpy()
    rows, preds = [], []
    with threadpool_limits(1):
        for name, cols in SESS_SUBSETS.items():
            X = st[cols].to_numpy(np.float64)
            for scheme, splits in [("5 folds", [(np.where(fold != k)[0], np.where(fold == k)[0], k)
                                                for k in range(N_FOLDS)]),
                                   ("leave one mouse out", [(np.where(np.arange(len(st)) != i)[0], np.array([i]), i)
                                                            for i in range(len(st))])]:
                p = {c: np.empty(len(st), int) for c in CLFS + ["sel"]}
                for tr, te, seed in splits:
                    res, sel = fit_featset(X, y, mouse, tr, te, seed=int(seed) % 1000)
                    for c in CLFS:
                        p[c][te] = res[c]["proba"].argmax(1)
                    p["sel"][te] = res[sel]["proba"].argmax(1)
                for c, v in p.items():
                    C = confusion(y, v)
                    rows.append(dict(features=name, scheme=scheme, clf=c, correct=int((v == y).sum()), n=len(y),
                                     bacc=bacc(y, v), **{f"correct_{LINES[g]}": int(C[g, g]) for g in range(N_LINES)},
                                     **{f"n_{LINES[g]}": int(C[g].sum()) for g in range(N_LINES)},
                                     confusion=str(C.tolist())))
                    preds.append(pd.DataFrame(dict(id=st.id, line=st.line, features=name, scheme=scheme, clf=c,
                                                   pred=[LINES[g] for g in v])))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "session_classifier.csv", index=False, float_format="%.4g")
    pd.concat(preds).to_csv(OUT / "session_predictions.csv", index=False)
    print("\n== Task 2: session-level features per line (median [min, max]) ==")
    for c in SESS_FEATS:
        print(f"  {c:22s} " + " | ".join(f"{l} {g[c].median():.3g} [{g[c].min():.3g}, {g[c].max():.3g}]"
                                          for l, g in st.groupby("line", sort=False)))
    print("  reporters:", st.groupby("line", sort=False).reporter.unique().to_dict())
    print("\n== Task 2: session classifier, correct of 46 ==")
    print(summ[["features", "scheme", "clf", "correct", "bacc"] + [f"correct_{l}" for l in LINES] + ["confusion"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- task 3
def forecast(infos):
    torch.set_num_threads(N_WORKERS)
    s10.load_sess()
    ridge_m = pickle.load(open(WORK / "ridge.pkl", "rb"))
    mlp_res = pickle.load(open(WORK / "mlp.pkl", "rb"))
    models, run_out = {}, {}
    for r in mlp_res:
        j = r["job"]
        models.setdefault((j["fold"], j["kind"]), []).append((j["seed"], load_model(r["state"], j["kind"] != "m1")))
        for mid, v in r["out"].items():
            run_out.setdefault((mid, j["kind"]), []).append(v["sse"])
    for key in models:
        models[key].sort(key=lambda t: t[0])
    m3_lab = {(j["fold"], j["seed"]): j["labels"] for j in s9.make_jobs() if j["kind"] == "m3"}

    rows = []
    for si, s in enumerate(s9.SESS):
        info, mid, k, n = s["info"], s["info"]["id"], s["info"]["fold"], s["n"]
        starts = window_starts(s["zte"].shape[1], FUTURE)
        X, Y = windows(s["zte"], starts)
        S = stim_windows(s["Ste"], starts)
        X64, S64, Y64 = X.astype(np.float64), S.astype(np.float64), Y.astype(np.float64)

        def err64(P):
            return ((P - Y64) ** 2).sum(axis=(0, 2))

        def err32(P):
            return ((P - Y) ** 2).sum(axis=(0, 2)).astype(np.float64)

        rm = ridge_m[k]
        sse = {"copy": err64(np.repeat(X64[:, :, -1:], FUTURE, axis=2)), "b": err64(predict(*rm["b"][:2], X64, S64))}
        for g in range(N_LINES):
            sse[f"d_{LINES[g]}"] = err64(predict(*rm["d"][g][:2], X64, S64))
        sse["e"] = np.mean([err64(predict(*fe[lab[mid]][:2], X64, S64)) for lab, fe in rm["e"]], axis=0)
        sse["m1"] = np.mean([err32(mlp_forecast(m, X, S)) for _, m in models[k, "m1"]], axis=0)
        for g in range(N_LINES):
            sse[f"m2_{LINES[g]}"] = np.mean([err32(mlp_forecast(m, X, S, m.emb.weight.detach()[torch.full((n,), g)]))
                                             for _, m in models[k, "m2"]], axis=0)
        sse["m3"] = np.mean([err32(mlp_forecast(m, X, S, m.emb.weight.detach()[torch.full((n,), m3_lab[k, seed][si])]))
                             for seed, m in models[k, "m3"]], axis=0)
        # Check against step 9's run_job test evaluation of the same fits
        for kind, name in [("m1", "m1"), ("m2", f"m2_{LINES[s['line']]}"), ("m3", "m3")]:
            ref = np.mean(run_out[mid, kind])
            assert abs(sse[name].sum() - ref) < 1e-3 * ref, (mid, kind, sse[name].sum(), ref)
        nel = Y.shape[0] * FUTURE
        for i in range(n):
            r = dict(id=mid, neuron=i, cre=info["cre"], line=LINES[s["line"]], fold=k, n_el=nel)
            r.update({f"sse_{m}": v[i] for m, v in sse.items()})
            r.update({f"score_{m}": 1 - v[i] / sse["copy"][i] for m, v in sse.items() if m != "copy"})
            rows.append(r)
        sc = {m: 1 - sse[m].sum() / sse["copy"].sum() for m in sse}
        t = LINES[s["line"]]
        print(f"{info['cre']:14s} fold {k} {mid} ({n:3d} neurons) | score b {sc['b']:.3f} d-true {sc[f'd_{t}']:.3f}"
              f" e {sc['e']:.3f} | m1 {sc['m1']:.3f} m2-true {sc[f'm2_{t}']:.3f} m3 {sc['m3']:.3f}", flush=True)
    ns = pd.DataFrame(rows)
    ns.to_csv(OUT / "neuron_scores_all_labels.csv", index=False, float_format="%.7g")
    print(f"Saved {OUT}/neuron_scores_all_labels.csv ({len(ns)} neurons)")
    reproduction_check()


def mouse_scores(ns, family, lab):
    """Per-mouse score of one model family with per-neuron labels lab (line names,
    or None for the no-label model): 1 - sum SSE / sum SSE_copylast over the
    neurons of the mouse."""
    if lab is None:
        sse = ns[f"sse_{'b' if family == 'ridge' else 'm1'}"].to_numpy()
    else:
        pre = "d" if family == "ridge" else "m2"
        M = ns[[f"sse_{pre}_{l}" for l in LINES]].to_numpy()
        sse = M[np.arange(len(ns)), pd.Index(LINES).get_indexer(lab)]
    g = pd.DataFrame(dict(id=ns.id, sse=sse, sse_copy=ns.sse_copy)).groupby("id", sort=False).sum()
    return 1 - g.sse / g.sse_copy


def reproduction_check():
    ns = pd.read_csv(OUT / "neuron_scores_all_labels.csv")
    ref = pd.read_csv(STEP10 / "mice_scores.csv").set_index("id")
    rows = []
    for fam, model, ref_col in [("ridge", None, "b"), ("ridge", "true", "d_true"), ("MLP", None, "m1"),
                                ("MLP", "true", "m2_true")]:
        new = mouse_scores(ns, fam, None if model is None else ns.line.to_numpy())
        rows.append(dict(model=ref_col, max_abs_diff=(new - ref.loc[new.index, f"score_{ref_col}"]).abs().max(),
                         mean_step11=new.mean(), mean_step10=ref[f"score_{ref_col}"].mean()))
    for m in ["e", "m3"]:
        g = ns.groupby("id", sort=False)[[f"sse_{m}", "sse_copy"]].sum()
        new = 1 - g[f"sse_{m}"] / g.sse_copy
        rows.append(dict(model=m, max_abs_diff=(new - ref.loc[new.index, f"score_{m}"]).abs().max(),
                         mean_step11=new.mean(), mean_step10=ref[f"score_{m}"].mean()))
    rc = pd.DataFrame(rows)
    rc.to_csv(OUT / "reproduction_check.csv", index=False, float_format="%.6g")
    print("\n== Reproduction of step 10 per-mouse scores (46 mice) ==")
    print(rc.to_string(index=False, float_format=lambda v: f"{v:.6f}"))


def assignments(ns):
    """Label assignments (per-neuron line names, in ns row order), from CSV files only."""
    p11 = pd.read_csv(OUT / "neuron_predictions.csv")
    p10 = pd.read_csv(STEP10 / "neuron_predictions.csv")
    key = ns[["id", "neuron"]]
    p11 = key.merge(p11, on=["id", "neuron"], how="left", validate="1:1")
    p10 = key.merge(p10, on=["id", "neuron"], how="left", validate="1:1")
    names = np.array(LINES)
    out = {"true": ns.line.to_numpy()}
    for fs in ["S", "S+b", "R", "A"]:
        if f"pred_{fs}_sel" in p11:
            out[f"{fs} neuron"] = names[p11[f"pred_{fs}_sel"].to_numpy()]
            out[f"{fs} session"] = names[p11[f"session_{fs}_sel"].to_numpy()]
    out["step 10 (a+b) neuron"] = names[p10.pred_sel.to_numpy()]
    return out


def assign():
    ns = pd.read_csv(OUT / "neuron_scores_all_labels.csv")
    labs = assignments(ns)
    rng = np.random.default_rng(0)
    shuffles = [rng.permutation(ns.line.to_numpy()) for _ in range(N_SHUFFLE)]
    info = ns.drop_duplicates("id").set_index("id")[["cre", "line", "fold"]]
    groups = CRE_LINES + ["all lines"]
    mice, gr, st = [], [], []
    for fam in ["ridge", "MLP"]:
        df = info.copy()
        df["score_none"] = mouse_scores(ns, fam, None)
        for name, lab in labs.items():
            df[f"score_{name}"] = mouse_scores(ns, fam, lab)
        df["score_shuffled"] = np.mean([mouse_scores(ns, fam, lab) for lab in shuffles], axis=0)
        df["frac_correct_S neuron"] = pd.Series(labs["S neuron"] == ns.line.to_numpy()).groupby(ns.id.to_numpy(),
                                                                                                 sort=False).mean()
        mice.append(df.reset_index().assign(family=fam))
        preds = [n for n in labs if n != "true"] + ["shuffled"]
        brng = np.random.default_rng(0)
        for cre in groups:
            sub = df if cre == "all lines" else df[df.cre == cre]
            for pm in preds:
                frac, lo, hi, gp, gt = recovered(sub, pm, "true", "none", brng, N_BOOT)
                gr.append(dict(line=cre, family=fam, labels=pm, gain_pred=gp, gain_true=gt, frac=frac, ci_lo=lo,
                               ci_hi=hi, score_none=sub.score_none.mean(), score_true=sub.score_true.mean(),
                               score_pred=sub[f"score_{pm}"].mean()))
                for b in ["none", "true"]:
                    st.append(dict(line=cre, family=fam, a=pm, b=b, **compare(sub, pm, b)))
            st.append(dict(line=cre, family=fam, a="true", b="none", **compare(sub, "true", "none")))
    pd.concat(mice).to_csv(OUT / "assignment_mice_scores.csv", index=False, float_format="%.6g")
    gr, st = pd.DataFrame(gr), pd.DataFrame(st)
    gr.to_csv(OUT / "gain_recovered.csv", index=False, float_format="%.4g")
    st.to_csv(OUT / "stats.csv", index=False, float_format="%.4g")
    print("\n== Task 3: fraction of the true-label gain recovered (all lines; 95% bootstrap CI over mice) ==")
    print(gr[gr.line == "all lines"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n== Task 3: per line ==")
    print(gr[gr.line != "all lines"][["line", "family", "labels", "gain_pred", "gain_true", "frac", "ci_lo", "ci_hi"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n== Task 3: Wilcoxon over mice (score; a_beats_b = mice where a has the higher score) ==")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


# ---------------------------------------------------------------- task 4
def s_features(z, S, hz):
    """S features (step 10 hand features without the recording-level ones) of the
    z-scored 10 Hz traces z (n, L) with stimulus features S (12, L). Features that
    a short window cannot define (a stimulus absent from it, an autocorrelation lag
    or a frequency band beyond its length) are NaN."""
    n, L = z.shape
    with np.errstate(all="ignore"):
        f = {"z std (10 Hz)": z.std(1), "skew": skew(z, axis=1), "kurtosis": kurtosis(z, axis=1)}
        above = z > EVENT_Z
        onset = above[:, 1:] & ~above[:, :-1]
        f["event rate (/min)"] = onset.sum(1) / (L / hz / 60)
        f["time above z=2"] = above.mean(1)
        iei = np.zeros((n, len(IEI_EDGES) - 1))
        for i in range(n):
            d = np.diff(np.where(onset[i])[0]) / hz
            if len(d):
                iei[i] = np.histogram(d, IEI_EDGES)[0] / len(d)
        for j in range(len(IEI_EDGES) - 1):
            f[f"IEI {IEI_EDGES[j]:g}-{IEI_EDGES[j + 1]:g} s"] = iei[:, j]
        zc = z - z.mean(1, keepdims=True)
        var = (zc ** 2).mean(1)
        for lag in AC_LAGS:
            f[f"autocorr {lag / 10:g} s"] = (zc[:, lag:] * zc[:, :-lag]).mean(1) / var if lag < L else np.full(n, np.nan)
        fq, P = welch(z, fs=hz, nperseg=min(256, L), axis=1)
        tot = P[:, fq > 0].sum(1)
        for lo, hi in BANDS:
            sel = (fq > lo) & (fq <= hi)
            f[f"power {lo:g}-{hi:g} Hz"] = np.log10(P[:, sel].sum(1) / tot + 1e-12) if sel.any() else np.full(n, np.nan)

        def mean_in(w):
            return z @ w / w.sum() if w.sum() > 0 else np.full(n, np.nan)

        for j, name in enumerate(["gratings", "natural movie", "spontaneous"]):
            f[f"mean z: {name}"] = mean_in(S[j])
        base = mean_in(S[0] * (1 - S[3]))
        f["grating on - blank"] = mean_in(S[3]) - base
        R = np.stack([mean_in(S[4 + k]) - base for k in range(len(DIRECTIONS))], axis=1)
        for k, deg in enumerate(DIRECTIONS):
            f[f"dir {deg}"] = R[:, k]
        th = np.deg2rad(DIRECTIONS)
        ok = np.isfinite(R)
        Rp = np.clip(np.where(ok, R, 0), 0, None)
        norm = Rp.sum(1) + 1e-9
        has = ok.any(1)
        f["tuning peak"] = np.where(has, np.where(ok, R, -np.inf).max(1), np.nan)
        f["DSI"] = np.where(has, np.abs(Rp @ np.exp(1j * th)) / norm, np.nan)
        f["OSI"] = np.where(has, np.abs(Rp @ np.exp(2j * th)) / norm, np.nan)
    return f


def window_features(info):
    """S features per neuron for each window length: N_WIN non-overlapping
    windows at random positions of the train part (same windows for all neurons
    of a session), or the whole train part."""
    c = np.load(NPZ / f"{info['id']}.npz")
    z, S = c["ztr"].astype(np.float64), c["Str_all"].astype(np.float64)
    hz = float(c["fr"]) / 3
    rng = np.random.default_rng(info["id"])
    out = []
    for wname, L in WINDOWS:
        if L is None:
            starts, L = [0], z.shape[1]
        else:
            starts = rng.choice(z.shape[1] // L, N_WIN, replace=False) * L
        for w, a in enumerate(starts):
            df = pd.DataFrame(s_features(z[:, a:a + L], S[:, a:a + L], hz))
            df.insert(0, "start_bin", a)
            df.insert(0, "win", w)
            df.insert(0, "window", wname)
            df.insert(0, "neuron", np.arange(len(z)))
            df.insert(0, "id", info["id"])
            out.append(df)
    return pd.concat(out, ignore_index=True)


def window_fold(args):
    from threadpoolctl import threadpool_limits
    k, X, y, mouse, fold, nid = args
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    with threadpool_limits(1):
        res, sel = fit_featset(X, y, mouse, tr, te, seed=k, impute=True)
    return k, te, res, sel


def window_test(infos):
    by_id = {i["id"]: i for i in infos}
    with Pool(N_WORKERS) as pool:
        wf = pd.concat(pool.map(window_features, infos), ignore_index=True)
    wf.insert(2, "cre", [by_id[i]["cre"] for i in wf.id])
    wf.insert(3, "fold", [by_id[i]["fold"] for i in wf.id])
    names = [c for c in wf.columns if c not in META + ["window", "win", "start_bin"]]
    assert len(names) == 37
    # Check: full-train-part features equal step 10's (CSV, 5 significant digits)
    full = wf[wf.window == "full train part"].sort_values(["id", "neuron"])
    ref = pd.read_csv(STEP10 / "neuron_features.csv").sort_values(["id", "neuron"])
    rel = (np.abs(full[names].to_numpy() - ref[names].to_numpy()) / (np.abs(ref[names].to_numpy()) + 1e-3)).max()
    print(f"full-window S features vs step 10 neuron_features.csv: max relative difference {rel:.2g}", flush=True)
    rows, preds = [], []
    for wname, L in WINDOWS:
        d = wf[wf.window == wname].reset_index(drop=True)
        X = d[names].to_numpy(np.float64)
        y = np.array([CRE_LINES.index(c) for c in d.cre])
        mouse, fold = d.id.to_numpy(), d.fold.to_numpy()
        nid = (d.id.astype(str) + "_" + d.neuron.astype(str)).to_numpy()
        t0 = time.time()
        with Pool(min(N_WORKERS, N_FOLDS)) as pool:
            outs = pool.map(window_fold, [(k, X, y, mouse, fold, nid) for k in range(N_FOLDS)])
        P = {c: np.empty((len(d), N_LINES)) for c in CLFS + ["sel"]}
        sels = {}
        for k, te, res, sel in outs:
            for c in CLFS:
                P[c][te] = res[c]["proba"]
            P["sel"][te] = res[sel]["proba"]
            sels[k] = sel
        for c, p in P.items():
            pw = p.argmax(1)
            # Neuron level: mean log-probability over the neuron's windows
            lp = pd.DataFrame(np.log(np.clip(p, 1e-9, 1)), columns=LINES).groupby(nid, sort=False).mean()
            yn = pd.Series(y).groupby(nid, sort=False).first().loc[lp.index].to_numpy()
            per_fold = [bacc(y[fold == k], pw[fold == k]) for k in range(N_FOLDS)]
            C = confusion(y, pw)
            rows.append(dict(window=wname, window_s=(L or 0) / 10 if L else np.nan, clf=c,
                             n_rows=len(d), windows_per_neuron=d.groupby(["id", "neuron"]).size().iloc[0],
                             bacc_window=bacc(y, pw), bacc_fold_mean=np.mean(per_fold), bacc_fold_std=np.std(per_fold),
                             bacc_neuron_pooled=bacc(yn, lp.to_numpy().argmax(1)),
                             selected=";".join(sels[k] for k in range(N_FOLDS)) if c == "sel" else "",
                             nan_frac=float(np.isnan(X).mean()),
                             **{f"recall_{LINES[g]}": C[g, g] / C[g].sum() for g in range(N_LINES)},
                             confusion=str(C.tolist())))
        print(f"{wname}: {len(d)} rows, NaN fraction {np.isnan(X).mean():.3f}, selected {sels},"
              f" balanced acc (window) " + ", ".join(f"{r['clf']} {r['bacc_window']:.3f}" for r in rows[-3:])
              + f", neuron-pooled sel {rows[-1]['bacc_neuron_pooled']:.3f} ({time.time() - t0:.0f} s)", flush=True)
    wr = pd.DataFrame(rows)
    full_len = wf[wf.window == "full train part"]
    wr.loc[wr.window == "full train part", "window_s"] = float(
        np.median([np.load(NPZ / f"{i}.npz", mmap_mode="r")["ztr"].shape[1] for i in full_len.id.unique()]) / 10)
    wr.to_csv(OUT / "window_accuracy.csv", index=False, float_format="%.5g")
    print("\n== Task 4: S classifier vs window length ==")
    print(wr.drop(columns=["confusion"]).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- main
def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    infos = pick_sessions()
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    phases = ["extract", "ridge", "mlp", "classify", "sessions", "forecast", "assign", "windows"] \
        if phase == "all" else [phase]
    for ph in phases:
        print(f"\n######## phase {ph} ########", flush=True)
        t0 = time.time()
        if ph == "extract":
            if all((NPZ / f"{i['id']}.npz").exists() and s10.raw_path(i).exists() for i in infos):
                print("data/cache complete, skip")
            else:
                s10.extract(infos)
        else:
            {"ridge": lambda: ridge(infos), "mlp": s10.mlp, "classify": lambda: classify(infos),
             "sessions": lambda: sessions(infos), "forecast": lambda: forecast(infos), "assign": assign,
             "windows": lambda: window_test(infos)}[ph]()
        print(f"phase {ph} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
