"""Step 13: label-free dynamics groups for forecasting (Bugeon et al. 2022 and Allen).

Hypothesis: a summary of a neuron's long-run dynamics, with no label at all, helps
the 3 s forecaster at least as much as the genetic cell type (step 12: S-classifier
labels, poor as cell typing, beat the true transcriptomic label in 17/17 sessions).
Analogy: clustering cells in single-cell RNA-seq before any downstream analysis.

Same data, splits, windows, folds, seeds and model settings as step 11 (Allen: 46
VISp three_session_A mice, 4 Cre lines, 5 leave-mice-out folds, 10 Hz, past 30 ->
next 10 bins) and step 12 (Bugeon: 17 sessions of 4 mice, leave one mouse out,
4.3 Hz, past 13 -> next 4 frames, 6 types EC / Lamp5 / Pvalb / Sncg / Sst / Vip).

Run `python scripts/step13_dynamics.py <dataset> <phase>`, dataset bugeon or allen,
phases in this order (`all` runs them all); then `python scripts/step13_dynamics.py
both analyse` writes results/step13/ from the two datasets. Intermediate files go
to data/step13/<dataset>/ (not committed).
- `prep`: Bugeon: step 12's extract (data/step12/, its tables are written to
  data/step13/bugeon_step12_out/ and must equal results/step12/*.csv). Allen: the
  step 8/10 cache must exist (step 11 `extract`); per-neuron ridge sufficient
  statistics with step 12's `neuron_stats` on the Allen windows (the same stride-1
  fit and val windows as step 5 / 8 `process_session`).
- `features`: S summary per neuron (step 11 `s_features` for Allen, 37 features;
  step 12 `s_features` for Bugeon, 39 features), from the train part only: the
  full train part, and (task 4) its first 3 and 30 minutes. Bugeon's blocks
  are separate recordings, so "first N minutes" takes the first part of every
  block in proportion to the block's train length (at least 128 frames per block,
  the Welch segment); Allen's are the first N minutes of the recording.
- `cluster` (task 1), per fold, training mice only: z-score each feature with the
  training-neuron mean / std, PCA keeping 95% of the variance, then k-means
  (n_init 10) and a Gaussian mixture (full covariance, reg 1e-3, n_init 3) for k
  in KS. Held-out neurons get the nearest centroid (k-means) or the most likely
  component (GMM). Missing features (short histories only) = training mean.
  No type label and no held-out data are used.
- `ridge` (tasks 1-2): step 12 ridge from summed per-neuron statistics (alpha on
  val, refit on train + val). b (no label, class-balanced) and d (one model per
  true type) reproduce steps 11 / 12. Per cluster, one model per cluster fitted on
  the cluster's training neurons with equal row weights, as d is per type ("km",
  "gmm"); the same with class-balanced rows inside each cluster ("kmbal"); and the
  shuffled-cluster control ("kmshuf": cluster ids permuted over the training
  neurons and, separately, over the held-out neurons, same sizes, 5 seeds). k is
  chosen per fold on the val windows of the training mice: the fit-part models at
  the val-picked alpha give each training session a val score (1 - SSE / SSE
  copy-last), and k maximises the mean over training sessions. The chosen k is
  used for every "sel" variant. Task 4: held-out neurons are reassigned from their
  3 / 30 min summaries with the fold's PCA and k-means and scored with the same
  models.
- `mlp` (task 2): step 9 / 12 FiLM MLP, same optimiser, batches (class-balanced
  over the true types, as m1 / m2), epochs, early stopping and seeding formula,
  3 seeds. All FiLM variants use m2's seeds (same initial weights and batches; only
  the FiLM input differs):
    cl     FiLM from the cluster id at the chosen k (16-dim embedding)
    clnt   the same at k = number of true types (fits shared with cl where equal)
    cont   FiLM from the standardised S vector through a linear layer (-> 16 dims,
           N(0, 0.02) init), no clustering
    contT  continuous summary and true-type embedding concatenated (32 dims)
    clsh   shuffled cluster ids at the chosen k (permutation as kmshuf, seeds 0-2)
  Plus fold-0 refits of m1 and m2 as a reproduction check (all other m1 / m2 SSE
  are read from results/step11 and results/step12 neuron_scores_all_labels.csv).
  cl and cont are also scored with the 3 / 30 min summaries (task 4).
- `collect`: per-neuron SSE of the MLP variants, mean over seeds.
- `analyse` (both datasets, CSV outputs only): results/step13/*.csv, see the result
  file. Scores per unit: Bugeon sessions (and mice), Allen mice (= sessions).

Ideas from other work (no code copied): NeuPRINT (Mi et al. 2023) learns a
time-invariant per-neuron representation from activity and feeds it to a
forecaster, which is the continuous-summary variant here; NuCLR (Arora et al. 2025)
learns neuron representations from activity alone and reads cell type from them,
the basis for asking whether a label-free summary carries the useful type
information; POCO's zero-initialised FiLM conditioning (via step 9); LOLCAT
(Schneider et al. 2023) for the inter-event-interval features and class-balanced
sampling (via step 10) and for asking how much activity a summary needs (task 4).
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
from scipy.stats import wilcoxon
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step11_confound as s11
import step12_bugeon as s12
import step9_mlp as s9
from step3_linear import ALPHAS, ROOT, window_starts, windows
from step4_stimulus import stim_windows
from step8_medium import CRE_LINES, NPZ, pick_sessions
from step10_inferred import LINES

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step13"
OUT = ROOT / "results/step13"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
KS = [2, 3, 4, 6, 8, 12, 16]
KS_EXTRA = [24, 32]                         # exploratory, ridge k-means / GMM only, never chosen
METHODS = ["km", "gmm"]
HIST = [("3 min", 3.0), ("30 min", 30.0)]   # plus the full train part
HTAG = {"3 min": "h3", "30 min": "h30"}
MIN_BLOCK = 128                             # Bugeon: frames per block in a short history
N_SHUF = 5                                  # ridge shuffled-cluster seeds (as step 12 e)
N_SEEDS = 3
MLP_KINDS = ["cl", "clnt", "cont", "contT", "clsh"]
DSN = None                                  # dataset of this process
P = F = None                                # PAST, FUTURE of the dataset
TYPES = None


def configure(ds):
    """Dataset constants. Step 12's ridge helpers (neuron_stats, assemble, val_mse,
    predict) read PAST / FUTURE / OFF from their module, so they are set here."""
    global DSN, P, F, TYPES
    DSN = ds
    if ds == "allen":
        s12.PAST, s12.FUTURE, s12.OFF, s12.CHUNK = 30, 10, np.arange(-30, 10), 16
        TYPES = LINES
    else:
        TYPES = s12.TYPES
    P, F = s12.PAST, s12.FUTURE
    (WORK / ds).mkdir(parents=True, exist_ok=True)


def wd():
    return WORK / DSN


# ---------------------------------------------------------------- neurons and data
def neurons():
    """One row per neuron (Bugeon: neuron-session) in the order of the step 11 / 12
    tables. Allen: session = mouse = experiment id."""
    if DSN == "bugeon":
        neu = s12.load_neurons()
    else:
        rows = []
        for info in pick_sessions():
            n = int(np.load(NPZ / f"{info['id']}.npz")["n_neurons"])
            rows.append(pd.DataFrame(dict(session=str(info["id"]), id=info["id"], mouse=str(info["id"]),
                                          mouse_name=info["mouse"], fold=info["fold"], neuron=np.arange(n),
                                          type=LINES[CRE_LINES.index(info["cre"])], depth=info["depth"])))
        neu = pd.concat(rows, ignore_index=True)
        neu["t"] = neu.type.map({t: k for k, t in enumerate(LINES)})
        neu["si"] = neu.session.map({s: k for k, s in enumerate(neu.session.unique())})
    return neu


def allen_gram(info):
    """Per-neuron ridge statistics for the step 5 / 8 Allen windows: fit = stride-1
    windows of the first n_fit bins of the train part, val = those of the rest."""
    out = wd() / f"{info['id']}_gram.npz"
    if out.exists():
        return
    c = np.load(NPZ / f"{info['id']}.npz")
    Z, S = c["ztr"].astype(np.float64), c["Str_all"].astype(np.float64)
    n_fit, n_tr = int(c["n_fit"]), Z.shape[1]
    g = {}
    for part, st in [("f", np.arange(P, n_fit - F + 1)), ("v", np.arange(n_fit + P, n_tr - F + 1))]:
        for k, v in s12.neuron_stats(Z, S, st, np.arange(len(Z))).items():
            g[f"{part}_{k}"] = v
    np.savez(out, **g)
    print(f"{info['id']}: gram of {len(Z)} neurons", flush=True)


def load_grams(neu):
    """step12_bugeon.load_grams for either dataset."""
    d = s12.WORK if DSN == "bugeon" else wd()
    G, sess = {}, []
    for s in neu.session.unique():
        g = np.load(d / f"{s}_gram.npz")
        sess.append({k: g[k] for k in g.files if k.split("_", 1)[1] in ["SS", "sS", "n"]})
        for k in g.files:
            if k.split("_", 1)[1] not in ["SS", "sS", "n"]:
                G.setdefault(k, []).append(g[k])
    G = {k: np.concatenate(v) for k, v in G.items()}
    for k in sess[0]:
        G[k] = np.stack([s[k] for s in sess])
    return G


def test_data(s):
    """Test windows of a session in float64: X (nw, n, P), Y (nw, n, F), S (nw, n_stim)."""
    if DSN == "bugeon":
        return s12.test_windows(np.load(s12.WORK / f"{s}.npz"))
    c = np.load(NPZ / f"{s}.npz")
    st = window_starts(c["zte"].shape[1], F)
    X, Y = windows(c["zte"], st)
    return X.astype(np.float64), Y.astype(np.float64), stim_windows(c["Ste_all"], st).astype(np.float64)


def prep():
    if DSN == "bugeon":
        if not all((s12.WORK / f"{s}_gram.npz").exists() for s in neurons().session.unique()):
            s12.OUT = WORK / "bugeon_step12_out"
            s12.OUT.mkdir(parents=True, exist_ok=True)
            s12.extract()
        for f in ["neurons", "sessions", "blocks"]:
            p = WORK / "bugeon_step12_out" / f"{f}.csv"
            if p.exists():
                same = open(p).read() == open(ROOT / f"results/step12/{f}.csv").read()
                print(f"step 12 {f}.csv re-extracted: {'identical' if same else 'DIFFERENT'}")
                assert same
        return
    infos = pick_sessions()
    assert all((NPZ / f"{i['id']}.npz").exists() for i in infos), "run scripts/step11_confound.py extract first"
    with Pool(N_WORKERS) as pool:
        pool.map(allen_gram, infos, chunksize=1)


# ---------------------------------------------------------------- features (S summary)
def feats_bugeon(s):
    z = np.load(s12.WORK / f"{s}.npz")
    Z, S, blk, hz = z["Ztr"].astype(np.float64), z["Str"].astype(np.float64), z["blk"], float(z["hz"])
    out = {"full": s12.s_features(Z, S, blk, hz), "full train part frames": len(blk), "hz": hz}
    for name, minutes in HIST:
        sel = []
        for b in np.unique(blk):
            idx = np.where(blk == b)[0]
            m = int(round(minutes * 60 * hz * len(idx) / len(blk)))
            sel.append(idx[:min(len(idx), max(MIN_BLOCK, m))])
        sel = np.concatenate(sel)
        with np.errstate(all="ignore"):
            out[name] = s12.s_features(Z[:, sel], S[:, sel], blk[sel], hz)
        out[f"{name} frames"] = len(sel)
    return s, out


def feats_allen(s):
    c = np.load(NPZ / f"{s}.npz")
    z, S, hz = c["ztr"].astype(np.float64), c["Str_all"].astype(np.float64), float(c["fr"]) / 3
    out = {"full": s11.s_features(z, S, hz), "full train part frames": z.shape[1], "hz": hz}
    for name, minutes in HIST:
        L = int(round(minutes * 60 * hz))
        out[name] = s11.s_features(z[:, :L], S[:, :L], hz)
        out[f"{name} frames"] = min(L, z.shape[1])
    return s, out


def features():
    neu = neurons()
    with Pool(N_WORKERS) as pool:
        res = dict(pool.map(feats_bugeon if DSN == "bugeon" else feats_allen, list(neu.session.unique())))
    names = list(res[neu.session.iloc[0]]["full"])
    Fm = {}
    for h in ["full"] + [n for n, _ in HIST]:
        A = np.vstack([np.column_stack([res[s][h][c] for c in names]) for s in neu.session.unique()])
        A[~np.isfinite(A)] = np.nan
        Fm[h] = A
    assert np.isfinite(Fm["full"]).all(), "non-finite S feature on the full train part"
    lens = pd.DataFrame([dict(session=s, hz=res[s]["hz"], **{f"{n} frames": res[s][f"{n} frames"]
                                                            for n in [n for n, _ in HIST] + ["full train part"]})
                         for s in res])
    np.savez(wd() / "features.npz", names=np.array(names), **{h: v for h, v in Fm.items()})
    lens.to_csv(wd() / "history_frames.csv", index=False)
    print(f"{len(names)} S features x {len(neu)} neurons; NaN fraction " +
          ", ".join(f"{h} {np.isnan(v).mean():.4f}" for h, v in Fm.items()))


def load_features():
    z = np.load(wd() / "features.npz")
    return list(z["names"]), {h: z[h] for h in ["full"] + [n for n, _ in HIST]}


# ---------------------------------------------------------------- clusters (task 1)
def cluster_fold(args):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from threadpoolctl import threadpool_limits
    k_fold, Fm, tr = args
    with threadpool_limits(1):
        X = Fm["full"]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1

        def standardise(A):
            return np.nan_to_num((A - mu) / sd, nan=0.0)  # missing feature -> training mean

        Z = {h: standardise(A) for h, A in Fm.items()}
        pca = PCA(n_components=0.95, svd_solver="full").fit(Z["full"][tr])
        Pc = {h: pca.transform(A) for h, A in Z.items()}
        lab, lab_hist, info = {}, {}, []
        for k in KS + KS_EXTRA:
            km = KMeans(n_clusters=k, n_init=10, random_state=k_fold).fit(Pc["full"][tr])
            gm = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-3, n_init=3,
                                 random_state=k_fold).fit(Pc["full"][tr])
            lab["km", k], lab["gmm", k] = km.predict(Pc["full"]), gm.predict(Pc["full"])
            for h in Fm:
                if h != "full":
                    lab_hist["km", k, h] = km.predict(Pc[h])
            for m, v in [("km", lab["km", k]), ("gmm", lab["gmm", k])]:
                info.append(dict(fold=k_fold, method=m, k=k, n_empty_train=k - len(np.unique(v[tr])),
                                 min_train_size=np.bincount(v[tr], minlength=k).min(),
                                 inertia=km.inertia_ if m == "km" else np.nan))
    print(f"fold {k_fold}: {pca.n_components_} PCs ({pca.explained_variance_ratio_.sum():.3f} of the variance)",
          flush=True)
    return k_fold, dict(mu=mu, sd=sd, n_pc=pca.n_components_, evr=pca.explained_variance_ratio_, lab=lab,
                        lab_hist=lab_hist, Z={h: v.astype(np.float32) for h, v in Z.items()}, info=info)


def cluster():
    neu = neurons()
    _, Fm = load_features()
    folds = sorted(neu.fold.unique())
    with Pool(min(N_WORKERS, len(folds))) as pool:
        res = dict(pool.map(cluster_fold, [(k, Fm, (neu.fold != k).to_numpy()) for k in folds]))
    pickle.dump(res, open(wd() / "clusters.pkl", "wb"))
    info = pd.DataFrame([r for k in folds for r in res[k]["info"]])
    info.to_csv(wd() / "cluster_info.csv", index=False)
    print(info.groupby(["method", "k"])[["n_empty_train", "min_train_size"]].agg(["min", "max"]).to_string())


def load_clusters():
    return pickle.load(open(wd() / "clusters.pkl", "rb"))


# ---------------------------------------------------------------- ridge (tasks 1, 2, 4)
def fit_group(G, si, w):
    """step12_bugeon.fit_weighted, also returning the fit-part model at the
    val-picked alpha (for choosing k on val)."""
    gf, _ = s12.assemble(G, si, w, "f")
    gv, yyv = s12.assemble(G, si, w, "v")
    coef = gf.solver()
    mse = [s12.val_mse(*coef(a), gv, yyv) for a in ALPHAS]
    j = int(np.argmin(mse))
    Wf, bf = coef(ALPHAS[j])
    W, b = (gf + gv).solver()(ALPHAS[j])
    return dict(W=W, b=b, alpha=ALPHAS[j], Wf=Wf, bf=bf)


def val_sse(G, si, idx, W, b):
    """Per-neuron SSE of (W, b) on the val windows of neurons idx, from their
    sufficient statistics (no data needed)."""
    v = lambda k: G[f"v_{k}"]
    Wx, Ws = W[:P], W[P:]
    s = si[idx]
    out = (np.einsum("npq,pf,qf->n", v("xx")[idx], Wx, Wx, optimize=True)
           + 2 * np.einsum("npf,pf->n", v("xS")[idx] @ Ws, Wx, optimize=True)
           + 2 * (v("sx")[idx] @ Wx @ b) - 2 * np.einsum("npf,pf->n", v("xy")[idx], Wx, optimize=True)
           - 2 * np.einsum("nsf,sf->n", v("Sy")[idx], Ws, optimize=True) - 2 * (v("sy")[idx] @ b)
           + np.trace(v("yy")[idx], axis1=1, axis2=2))
    per_sess = np.array([np.einsum("sf,st,tf->", Ws, SS, Ws) + 2 * sS @ Ws @ b + n * b @ b
                         for SS, sS, n in zip(v("SS"), v("sS"), v("n"))])
    return out + per_sess[s]


def val_copy_sse(G, idx):
    v = lambda k: G[f"v_{k}"][idx]
    return np.trace(v("yy"), axis1=1, axis2=2) - 2 * v("xy")[:, P - 1, :].sum(1) + F * v("xx")[:, P - 1, P - 1]


def unit_val_score(neu, idx, sse, copy):
    g = pd.DataFrame(dict(u=neu.session.to_numpy()[idx], sse=sse, cp=copy)).groupby("u").sum()
    return (1 - g.sse / g.cp).mean()


def fit_clusters(G, si, neu, lab, mask, balanced=False):
    """One model per cluster on the training neurons (mask) of that cluster; equal
    row weights (as d), or class-balanced over the true types inside the cluster."""
    out = {}
    for c in range(lab.max() + 1):
        m = mask & (lab == c)
        if m.any():
            out[c] = fit_group(G, si, s12.balanced_weights(neu, m) if balanced else m.astype(float))
    return out


def fake_clusters(lab, held, seed):
    """Shuffled-cluster control: ids permuted over the training neurons (same sizes)
    and, separately, over the held-out neurons (step 12 fake_labels on clusters)."""
    rng = np.random.default_rng(seed)
    f = lab.copy()
    for m in [~held, held]:
        f[m] = rng.permutation(f[m])
    return f


def ridge():
    neu = neurons()
    G = load_grams(neu)
    si = neu.si.to_numpy()
    neu["rows"] = G["f_n"][si] + G["v_n"][si]
    cl = load_clusters()
    t_arr = neu.t.to_numpy()
    rows, val_rows, chosen, alphas = [], [], [], []
    for k in sorted(neu.fold.unique()):
        t0 = time.time()
        held = (neu.fold == k).to_numpy()
        tr = ~held
        C = cl[k]
        b = s12.fit_weighted(G, si, s12.balanced_weights(neu, tr))
        d = [s12.fit_weighted(G, si, (tr & (t_arr == c)).astype(float)) for c in range(len(TYPES))]
        # Val scores of the training sessions (fit-part models), for choosing k
        itr = np.where(tr)[0]
        cp = val_copy_sse(G, itr)
        bb = fit_group(G, si, s12.balanced_weights(neu, tr))
        one = fit_group(G, si, tr.astype(float))
        ref = {"b (none)": unit_val_score(neu, itr, val_sse(G, si, itr, bb["Wf"], bb["bf"]), cp),
               "k=1 (one model, equal weights)": unit_val_score(neu, itr, val_sse(G, si, itr, one["Wf"], one["bf"]), cp)}
        # Check: summed per-neuron val SSE = step 12 val_mse of the same model and weights
        gv, yyv = s12.assemble(G, si, tr.astype(float), "v")
        chk = s12.val_mse(one["Wf"], one["bf"], gv, yyv) * gv.n * F
        assert abs(val_sse(G, si, itr, one["Wf"], one["bf"]).sum() / chk - 1) < 1e-6
        for name, v in ref.items():
            val_rows.append(dict(fold=k, method=name, k=1, val_score=v))
        models, sse_cols = {}, {}
        for m in METHODS:
            for kk in KS + KS_EXTRA:
                lab = C["lab"][m, kk]
                ms = fit_clusters(G, si, neu, lab, tr)
                models[m, kk] = ms
                sse = np.zeros(len(itr))
                for c, mod in ms.items():
                    sel = lab[itr] == c
                    sse[sel] = val_sse(G, si, itr[sel], mod["Wf"], mod["bf"])
                val_rows.append(dict(fold=k, method=m, k=kk, val_score=unit_val_score(neu, itr, sse, cp),
                                     in_grid=kk in KS))
            vr = [r for r in val_rows if r["fold"] == k and r["method"] == m and r["in_grid"]]
            chosen.append(dict(fold=k, method=m, k=max(vr, key=lambda r: r["val_score"])["k"]))
        k_sel = [c["k"] for c in chosen if c["fold"] == k and c["method"] == "km"][0]
        k_gsel = [c["k"] for c in chosen if c["fold"] == k and c["method"] == "gmm"][0]
        for kk in KS:
            lab = C["lab"]["km", kk]
            models["kmbal", kk] = fit_clusters(G, si, neu, lab, tr, balanced=True)
            for seed in range(N_SHUF):
                fk = fake_clusters(lab, held, seed)
                models["kmshuf", kk, seed] = (fk, fit_clusters(G, si, neu, fk, tr))
        models["k1"] = {0: one}
        alphas.append(dict(fold=k, b=b[2], **{f"d_{TYPES[c]}": d[c][2] for c in range(len(TYPES))},
                           k_sel=k_sel, k_gmm_sel=k_gsel,
                           **{f"km{kk}": ";".join(f"{models['km', kk][c]['alpha']:g}" for c in models["km", kk])
                              for kk in KS}))

        # Test SSE per held-out neuron
        for s in neu[held].session.unique():
            X, Y, S = test_data(s)
            rr = np.where((neu.session == s).to_numpy())[0]
            err = lambda Pr: ((Pr - Y) ** 2).sum(axis=(0, 2))
            cache = {}

            def sse_of(ms, labs, key):
                """Per-neuron SSE with neuron i forecast by model ms[labs[i]] (b if the
                cluster has no training neuron)."""
                out = np.empty(len(rr))
                for c in np.unique(labs):
                    if (key, c) not in cache:
                        mod = ms.get(c)
                        W_, b_ = (mod["W"], mod["b"]) if mod else b[:2]
                        cache[key, c] = err(s12.predict(W_, b_, X, S))
                    out[labs == c] = cache[key, c][labs == c]
                return out

            r = dict(copy=err(np.repeat(X[:, :, -1:], F, axis=2)), r_none=err(s12.predict(b[0], b[1], X, S)))
            for c in range(len(TYPES)):
                r[f"d_{TYPES[c]}"] = err(s12.predict(d[c][0], d[c][1], X, S))
            r["r_k1"] = sse_of(models["k1"], np.zeros(len(rr), int), "k1")
            for m in METHODS + ["kmbal"]:
                for kk in KS + (KS_EXTRA if m != "kmbal" else []):
                    base = "km" if m == "kmbal" else m
                    r[f"r_{m}{kk}"] = sse_of(models[m, kk], C["lab"][base, kk][rr], (m, kk))
            for kk in KS:
                r[f"r_kmshuf{kk}"] = np.mean([sse_of(models["kmshuf", kk, sd][1], models["kmshuf", kk, sd][0][rr],
                                                     ("kmshuf", kk, sd)) for sd in range(N_SHUF)], axis=0)
            r["r_sel"], r["r_selshuf"] = r[f"r_km{k_sel}"], r[f"r_kmshuf{k_sel}"]
            r["r_gmmsel"] = r[f"r_gmm{k_gsel}"]
            for h, _ in HIST:
                r[f"r_sel_{HTAG[h]}"] = sse_of(models["km", k_sel], C["lab_hist"]["km", k_sel, h][rr], ("km", k_sel))
            rows.append(pd.DataFrame(dict(session=s, neuron=neu.neuron.to_numpy()[rr], n_el=Y.shape[0] * F,
                                          k_sel=k_sel, k_gmm_sel=k_gsel, **{f"sse_{m}": v for m, v in r.items()})))
        print(f"fold {k}: alpha b {b[2]:g}; val score " + ", ".join(
            f"{r['method']} k={r['k']} {r['val_score']:.4f}" for r in val_rows if r["fold"] == k and r["method"] == "km")
              + f" -> k {k_sel} (GMM {k_gsel}) ({time.time() - t0:.0f} s)", flush=True)
    pd.concat(rows).to_csv(wd() / "ridge_sse.csv", index=False, float_format="%.9g")
    pd.DataFrame(val_rows).to_csv(wd() / "val_k.csv", index=False, float_format="%.6g")
    pd.DataFrame(chosen).to_csv(wd() / "chosen_k.csv", index=False)
    pd.DataFrame(alphas).to_csv(wd() / "ridge_alphas.csv", index=False)
    print(f"Saved {wd()}/ridge_sse.csv, val_k.csv, chosen_k.csv, ridge_alphas.csv")


# ---------------------------------------------------------------- MLP (tasks 2, 4)
NEU = T = None
COND = {}


class CondMixin:
    """FiLM from a continuous per-neuron vector through a linear layer (16 dims,
    N(0, 0.02) init like the label embedding), optionally concatenated with a label
    embedding; zero-initialised gamma / beta as step 9, so it starts as m1."""

    def __init__(self, n_labels, n_cont):
        super().__init__(n_labels)
        self.film, self.n_labels = True, n_labels
        self.cont = nn.Linear(n_cont, s9.EMB)
        nn.init.normal_(self.cont.weight, 0, 0.02); nn.init.zeros_(self.cont.bias)
        self.gb = nn.Linear(s9.EMB * (2 if n_labels else 1), 4 * s9.HIDDEN)
        nn.init.zeros_(self.gb.weight); nn.init.zeros_(self.gb.bias)

    def forward(self, x, lab=None, cont=None):
        e = self.cont(cont) if lab is None else torch.cat([self.cont(cont), self.emb(lab)], 1)
        g1, b1, g2, b2 = self.gb(e).chunk(4, dim=1)
        h1 = torch.relu(self.l1(x)) * (1 + g1) + b1
        h2 = torch.relu(self.l2(h1)) * (1 + g2) + b2
        return self.out(h2)


class CondBugeon(CondMixin, s12.MLP):
    pass


class CondAllen(CondMixin, s9.MLP):
    pass


class AllenSampler(s9.Sampler):
    """step9_mlp.Sampler.draw with the same random draws, also returning the global
    neuron row of every sample (for per-neuron FiLM inputs)."""

    def draw(self, n_per_line):
        xs, ys, rs = [], [], []
        for l in self.lines:
            ids, p = self.by_line[l]
            pick = self.rng.choice(len(ids), size=n_per_line, p=p)
            for k in np.unique(pick):
                m = int((pick == k).sum())
                si = int(ids[k])
                lo, hi = s9.SESS[si][self.part]
                neur = self.rng.integers(s9.SESS[si]["n"], size=m)
                x, y = s9.gather(si, neur, self.rng.integers(lo, hi, size=m))
                xs.append(x); ys.append(y); rs.append(OFFS[si] + neur)
        return np.vstack(xs), np.vstack(ys), np.concatenate(rs)


OFFS = None


def load_mlp_data():
    global NEU, T, OFFS
    if DSN == "bugeon":
        s12.load_sess()
        NEU = s12.NEU
    else:
        s9.load()
        NEU = neurons()
        assert [str(s["info"]["id"]) for s in s9.SESS] == list(NEU.session.unique())
        OFFS = np.cumsum([0] + [s["n"] for s in s9.SESS])[:-1]
    T = NEU.t.to_numpy()
    cl = load_clusters()
    ck = pd.read_csv(wd() / "chosen_k.csv")
    for k, C in cl.items():
        ksel = int(ck[(ck.fold == k) & (ck.method == "km")].k.iloc[0])
        COND[k] = dict(k_sel=ksel, lab_sel=C["lab"]["km", ksel], lab_nt=C["lab"]["km", len(TYPES)],
                       Z=C["Z"]["full"], lab_hist={h: C["lab_hist"]["km", ksel, h] for h, _ in HIST},
                       Z_hist={h: C["Z"][h] for h, _ in HIST})


def cond_inputs(kind, fold, seed):
    """Per-neuron FiLM inputs (rows of NEU): label ids or None, continuous or None,
    and the number of label ids."""
    C = COND[fold]
    held = NEU.fold.to_numpy() == fold
    nt, ks = len(TYPES), C["k_sel"]
    return {"m1": (None, None, 0), "m2": (T, None, nt), "cl": (C["lab_sel"], None, ks),
            "clnt": (C["lab_nt"], None, nt), "clsh": (fake_clusters(C["lab_sel"], held, seed), None, ks),
            "cont": (None, C["Z"], 0), "contT": (T, C["Z"], nt)}[kind]


def make_model(n_lab, cont):
    if cont is None:
        return (s12.MLP if DSN == "bugeon" else s9.MLP)(n_lab)
    return (CondBugeon if DSN == "bugeon" else CondAllen)(n_lab, cont.shape[1])


def call(model, x, lab, cont, rows):
    L = None if lab is None else torch.from_numpy(lab[rows].astype(np.int64))
    if cont is None:
        return model(x, L)
    return model(x, L, torch.from_numpy(cont[rows]))


def held_windows(si):
    """Test windows of held-out session si as run_job evaluates them (float32)."""
    if DSN == "bugeon":
        s = s12.SESS[si]
        W = s["Zte"][:, s["te"][:, None] + s12.OFF].transpose(1, 0, 2)
        S = s["Ste"][:, s["te"][:, None] + s12.OFF].transpose(1, 0, 2).reshape(len(s["te"]), -1)
        return W[..., :P], W[..., P:], S
    s = s9.SESS[si]
    st = window_starts(s["zte"].shape[1], F)
    X, Y = windows(s["zte"], st)
    return X, Y, stim_windows(s["Ste"], st)


def forecast(model, si, lab, cont):
    rows = np.where(NEU.si.to_numpy() == si)[0]
    X, Y, S = held_windows(si)
    nw, n = X.shape[:2]
    Pr = np.empty_like(Y)
    with torch.no_grad():
        for a in range(0, nw, 64):
            Xc, Sc = X[a:a + 64], S[a:a + 64]
            D = torch.from_numpy(np.concatenate([Xc.reshape(-1, P), np.repeat(Sc, n, axis=0)], axis=1))
            Pr[a:a + 64] = call(model, D, lab, cont, np.tile(rows, len(Xc))).numpy().reshape(len(Xc), n, F)
    return ((Pr - Y) ** 2).sum(axis=(0, 2)).astype(np.float64)


def run_job(job):
    """step 9 / 12 run_job (same optimiser, batches, epochs, early stopping, seeding)
    with per-neuron FiLM inputs; returns the per-neuron test SSE of the held-out
    neurons (and, for cl / cont, with the 3 / 30 min summaries)."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, kind, seed = job["fold"], job["kind"], job["seed"]
    rng = np.random.default_rng(1000 * fold + seed + (0 if kind == "m1" else 1) * 10)
    torch.manual_seed(int(rng.integers(2 ** 31)))
    lab, cont, n_lab = cond_inputs(kind, fold, seed)
    model = make_model(n_lab, cont)
    opt = torch.optim.Adam(model.parameters(), lr=s9.LR)
    if DSN == "bugeon":
        tr = np.where(NEU.fold.to_numpy() != fold)[0]
        smp = s12.Sampler(tr, "fit", rng)
        n_t = len(smp.by_type)
        xv, yv, rv = s12.Sampler(tr, "val", np.random.default_rng(fold)).draw(s12.N_VAL)
        groups = list(smp.by_type)
    else:
        train = [i for i, s in enumerate(s9.SESS) if s["info"]["fold"] != fold]
        smp = AllenSampler(train, "tr", rng)
        n_t = len(smp.lines)
        xv, yv, rv = AllenSampler(train, "va", np.random.default_rng(fold)).draw(s9.N_VAL)
        groups = smp.lines
    xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    v_type = T[rv]

    def val_loss():
        model.eval()
        with torch.no_grad():
            err = ((call(model, xv, lab, cont, rv) - yv) ** 2).mean(1).numpy()
        model.train()
        return float(np.mean([err[v_type == t].mean() for t in groups]))

    best, best_state, bad, hist = val_loss(), None, 0, []
    for ep in range(s9.MAX_EPOCHS):
        for _ in range(s9.EPOCH_BATCHES):
            x, y, r = smp.draw(s9.BATCH // n_t)
            loss = ((call(model, torch.from_numpy(x), lab, cont, r) - torch.from_numpy(y)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = val_loss()
        hist.append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= s9.PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    held_s = sorted(NEU[NEU.fold == fold].si.unique())
    out = {si: forecast(model, si, lab, cont) for si in held_s}
    out_h = {}
    if kind in ["cl", "cont"]:
        for h, _ in HIST:
            lh = COND[fold]["lab_hist"][h] if kind == "cl" else None
            zh = COND[fold]["Z_hist"][h] if kind == "cont" else None
            out_h[h] = {si: forecast(model, si, lh, zh) for si in held_s}
    return dict(job=job, out=out, out_hist=out_h, epochs=len(hist), best_val=best, hist=hist,
                secs=time.time() - t0, state={k: v.clone() for k, v in model.state_dict().items()})


def mlp_jobs():
    jobs = [dict(fold=0, kind=kind, seed=seed) for kind in ["m1", "m2"] for seed in range(N_SEEDS)]
    for k in sorted(NEU.fold.unique()):
        for kind in MLP_KINDS:
            if kind == "clnt" and COND[k]["k_sel"] == len(TYPES):
                continue  # same fits as cl
            jobs += [dict(fold=k, kind=kind, seed=seed) for seed in range(N_SEEDS)]
    return jobs


def mlp():
    load_mlp_data()
    path = wd() / "mlp.pkl"
    done = pickle.load(open(path, "rb")) if path.exists() else []
    have = {tuple(r["job"].values()) for r in done}
    jobs = [j for j in mlp_jobs() if tuple(j.values()) not in have]
    print(f"{len(jobs)} MLP fits to run ({len(have)} done) in {N_WORKERS} workers; chosen k "
          + str({k: c["k_sel"] for k, c in COND.items()}), flush=True)
    results = list(done)
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}/{len(jobs) + len(done)}] fold {j['fold']} {j['kind']} seed {j['seed']}:"
                  f" {r['epochs']} epochs, best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)
            pickle.dump(results, open(path, "wb"))  # checkpoint, so an interrupted run resumes


def collect():
    neu = neurons()
    res = pickle.load(open(wd() / "mlp.pkl", "rb"))
    acc = {}
    for r in res:
        j = r["job"]
        for si, v in r["out"].items():
            acc.setdefault((j["kind"], si), []).append(v)
        for h, d in r["out_hist"].items():
            for si, v in d.items():
                acc.setdefault((f"{j['kind']}_{HTAG[h]}", si), []).append(v)
    ns = neu[["session", "neuron"]].copy()
    ck = pd.read_csv(wd() / "chosen_k.csv")
    for (name, si), v in acc.items():
        assert len(v) == N_SEEDS, (name, si, len(v))
        ns.loc[neu.si == si, f"sse_m_{name}"] = np.mean(v, axis=0)
    for k in sorted(neu.fold.unique()):  # clnt = cl where the chosen k is the number of types
        if int(ck[(ck.fold == k) & (ck.method == "km")].k.iloc[0]) == len(TYPES):
            ns.loc[neu.fold == k, "sse_m_clnt"] = ns.loc[neu.fold == k, "sse_m_cl"]
    ns = ns.rename(columns={"sse_m_m1": "sse_m_none_rerun", "sse_m_m2": "sse_m_true_rerun"})
    ns.to_csv(wd() / "mlp_sse.csv", index=False, float_format="%.9g")
    pd.DataFrame([dict(**r["job"], epochs=r["epochs"], best_val=r["best_val"], secs=r["secs"]) for r in res]) \
        .sort_values(["fold", "kind", "seed"]).to_csv(wd() / "mlp_fits.csv", index=False, float_format="%.5g")
    print(f"Saved {wd()}/mlp_sse.csv ({ns.shape[1] - 2} variants), mlp_fits.csv")


# ---------------------------------------------------------------- analyse (CSV only)
RIDGE_MAIN = ["r_none", "r_true", "r_S", "r_k1", "r_sel", "r_nt", "r_selshuf", "r_gmmsel"]
MLP_MAIN = ["m_none", "m_true", "m_S", "m_cl", "m_clnt", "m_cont", "m_contT", "m_clsh"]
NAMES = {"r_none": "none (b)", "r_true": "true type (d)", "r_S": "S-classifier labels", "r_k1": "k = 1, equal weights",
         "r_sel": "clusters, chosen k", "r_nt": "clusters, k = n types", "r_selshuf": "shuffled clusters, chosen k",
         "r_gmmsel": "GMM clusters, chosen k", "m_none": "none (m1)", "m_true": "true type (m2)",
         "m_S": "S-classifier labels", "m_cl": "clusters, chosen k", "m_clnt": "clusters, k = n types",
         "m_cont": "continuous summary", "m_contT": "continuous + true type", "m_clsh": "shuffled clusters"}
PAIRS = [("r_sel", "r_none"), ("r_sel", "r_true"), ("r_sel", "r_selshuf"), ("r_true", "r_none"), ("r_S", "r_none"),
         ("r_sel", "r_S"), ("r_nt", "r_none"), ("r_nt", "r_true"), ("r_k1", "r_none"), ("r_gmmsel", "r_none"),
         ("r_selshuf", "r_none"),
         ("m_cl", "m_none"), ("m_cl", "m_true"), ("m_cl", "m_clsh"), ("m_clnt", "m_none"), ("m_clnt", "m_true"),
         ("m_cont", "m_none"), ("m_cont", "m_true"), ("m_contT", "m_cont"), ("m_contT", "m_true"),
         ("m_contT", "m_none"), ("m_true", "m_none"), ("m_S", "m_none"), ("m_cont", "m_cl"), ("m_clsh", "m_none"),
         ("m_cont", "m_S"), ("m_cl", "m_S")]


def wil(a, b):
    return s12.wil(a, b)


def dataset_table(ds):
    """Per-neuron SSE of every variant for one dataset (rows in step 11 / 12 order)."""
    configure(ds)
    neu = neurons()
    meta = ["session", "mouse", "fold", "neuron", "type"] + (["uid", "depth"] if ds == "bugeon" else ["depth"])
    ns = neu[meta].copy()
    ns.insert(0, "dataset", ds)
    ns = ns.merge(pd.read_csv(wd() / "ridge_sse.csv").astype({"session": str}), on=["session", "neuron"],
                  how="left", validate="1:1")
    ns = ns.merge(pd.read_csv(wd() / "mlp_sse.csv").astype({"session": str}), on=["session", "neuron"],
                  how="left", validate="1:1")
    D = ns[[f"sse_d_{t}" for t in TYPES]].to_numpy()
    tix = pd.Index(TYPES).get_indexer(ns.type)
    ns["sse_r_true"] = D[np.arange(len(ns)), tix]
    ns["sse_r_nt"] = ns[f"sse_r_km{len(TYPES)}"]
    # Earlier steps: m1 / m2 SSE (all folds) and S-classifier labels
    if ds == "bugeon":
        old = pd.read_csv(ROOT / "results/step12/neuron_scores_all_labels.csv")
        pr = pd.read_csv(ROOT / "results/step12/neuron_predictions.csv")
        key = ["session", "neuron"]
        s_lab = ns[key].merge(pr, on=key, how="left", validate="1:1")["pred_S_sel"].to_numpy()
    else:
        old = pd.read_csv(ROOT / "results/step11/neuron_scores_all_labels.csv").assign(session=lambda d: d.id.astype(str))
        pr = pd.read_csv(ROOT / "results/step11/neuron_predictions.csv").assign(session=lambda d: d.id.astype(str))
        key = ["session", "neuron"]
        s_lab = np.array(LINES)[ns[key].merge(pr, on=key, how="left", validate="1:1")["pred_S_sel"].to_numpy()]
    old = ns[key].merge(old, on=key, how="left", validate="1:1")
    M2 = old[[f"sse_m2_{t}" for t in TYPES]].to_numpy()
    six = pd.Index(TYPES).get_indexer(s_lab)
    assert (six >= 0).all()
    ns["sse_m_none"], ns["sse_m_true"], ns["sse_m_S"] = old.sse_m1, M2[np.arange(len(ns)), tix], M2[np.arange(len(ns)), six]
    ns["sse_r_S"] = D[np.arange(len(ns)), six]
    ns["S_label"] = s_lab
    # Reproduction: ridge b / d recomputed here vs steps 11 / 12; MLP fold-0 refits vs steps 11 / 12
    rep = []
    for new, ref in [("r_none", old.sse_b), ("r_true", pd.Series([old[f"sse_d_{t}"][i] for i, t in enumerate(ns.type)]))]:
        rep.append(dict(dataset=ds, model=new, folds="all", **repro(ns, ns[f"sse_{new}"], ref.to_numpy())))
    f0 = (ns.fold == 0).to_numpy()
    for new, ref in [("m_none_rerun", ns.sse_m_none), ("m_true_rerun", ns.sse_m_true)]:
        rep.append(dict(dataset=ds, model=new, folds="0", **repro(ns[f0], ns[f"sse_{new}"][f0], ref[f0].to_numpy())))
    copy_diff = np.abs(ns.sse_copy - old.sse_copy).max() / old.sse_copy.max()
    rep.append(dict(dataset=ds, model="copy-last", folds="all", max_abs_unit_score_diff=0.0, max_rel_neuron_sse_diff=copy_diff))
    return ns.drop(columns=[f"sse_d_{t}" for t in TYPES]), pd.DataFrame(rep)


def repro(ns, new, ref):
    u = pd.DataFrame(dict(u=ns.session.to_numpy(), a=np.asarray(new), b=np.asarray(ref), c=ns.sse_copy.to_numpy()))
    g = u.groupby("u").sum()
    return dict(max_abs_unit_score_diff=float(np.abs(g.a / g.c - g.b / g.c).max()),
                max_rel_neuron_sse_diff=float(np.nanmax(np.abs(u.a - u.b) / u.b)),
                mean_score_new=float((1 - g.a / g.c).mean()), mean_score_ref=float((1 - g.b / g.c).mean()))


def unit_scores(ns, variants, by):
    g = ns.groupby(by, sort=False)
    cp = g.sse_copy.sum()
    return pd.DataFrame({v: 1 - g[f"sse_{v}"].sum() / cp for v in variants})


def ami_rows(ds, cl, neu, k_sel):
    from sklearn.metrics import adjusted_mutual_info_score as ami
    rows, cont = [], []
    for k, C in cl.items():
        held = (neu.fold == k).to_numpy()
        for (m, kk), lab in C["lab"].items():
            for part, mask in [("train", ~held), ("held-out", held)]:
                sub = neu[mask]
                r = dict(dataset=ds, fold=k, method=m, k=kk, part=part, n=int(mask.sum()),
                         ami_type=ami(sub.type, lab[mask]), ami_session=ami(sub.session, lab[mask]),
                         ami_mouse=ami(sub.mouse, lab[mask]) if sub.mouse.nunique() > 1 else np.nan,
                         ref_ami_type_mouse=ami(sub.mouse, sub.type) if sub.mouse.nunique() > 1 else np.nan,
                         ref_ami_type_session=ami(sub.session, sub.type))
                # mouse / session purity: share of each cluster from its largest mouse, weighted by cluster size
                ct = pd.crosstab(lab[mask], sub.mouse.to_numpy())
                r["top_mouse_share"] = float(ct.max(1).sum() / ct.to_numpy().sum())
                r["top_mouse_share_if_random"] = float(sub.mouse.value_counts(normalize=True).max())
                ct = pd.crosstab(lab[mask], sub.type.to_numpy())
                r["top_type_share"] = float(ct.max(1).sum() / ct.to_numpy().sum())
                r["top_type_share_if_random"] = float(sub.type.value_counts(normalize=True).max())
                rows.append(r)
            if m == "km" and kk in (k_sel[k], len(TYPES)):  # contingency at the chosen k and k = n types
                for col in ["type", "mouse"]:
                    ct = pd.crosstab(lab, neu[col].to_numpy())
                    for c in ct.index:
                        for g in ct.columns:
                            if ct.loc[c, g]:
                                cont.append(dict(dataset=ds, fold=k, k=kk, cluster=int(c), by=col, group=g,
                                                 count=int(ct.loc[c, g]),
                                                 held_out=bool(neu[neu[col] == g].fold.iloc[0] == k) if col == "mouse"
                                                 else np.nan))
    return rows, cont


def analyse():
    tables, reps, units, stats, kswe, typ, hist, ami, cont, prof, valk, chosen, fits = ([] for _ in range(13))
    for ds in ["bugeon", "allen"]:
        if not (WORK / ds / "mlp_sse.csv").exists():
            print(f"{ds}: no mlp_sse.csv, skipped")
            continue
        ns, rep = dataset_table(ds)
        tables.append(ns)
        reps.append(rep)
        neu = neurons()
        cl = load_clusters()
        names, _ = load_features()
        ridge_k = [f"r_{m}{k}" for m in ["km", "gmm", "kmbal", "kmshuf"]
                   for k in KS + (KS_EXTRA if m in METHODS else [])]
        hv = [f"{v}_{HTAG[h]}" for v in ["r_sel", "m_cl", "m_cont"] for h, _ in HIST]
        var = RIDGE_MAIN + MLP_MAIN + ridge_k + hv
        levels = [("session", "session"), ("mouse", "mouse")] if ds == "bugeon" else [("mouse", "session")]
        for level, by in levels:
            u = unit_scores(ns, var, by)
            u["n_neurons"] = ns.groupby(by, sort=False).size()
            u["mouse"] = ns.groupby(by, sort=False).mouse.first()
            units.append(u.reset_index(names="unit").assign(dataset=ds, level=level))
        main_level = "session" if ds == "bugeon" else "mouse"
        U = units[-len(levels)]
        Um = units[-1]
        for a, b in PAIRS:
            fam = "ridge" if a.startswith("r_") else "MLP"
            stats.append(dict(dataset=ds, test={"session": "sessions", "mouse": "mice"}[main_level], type="all", family=fam, a=a, b=b,
                              **wil(U[a], U[b])))
            if ds == "bugeon":
                d = (Um[a] - Um[b]).to_numpy()
                stats.append(dict(dataset=ds, test="mice (signs only)", type="all", family=fam, a=a, b=b, n=len(d),
                                  a_beats_b=int((d > 0).sum()), ties=int((d == 0).sum()), median_diff=np.median(d),
                                  mean_diff=d.mean(), p=wil(Um[a], Um[b])["p"],
                                  per_mouse=";".join(f"{m} {x:+.4f}" for m, x in zip(Um.unit, d))))
        # k sweep (ridge): mean over units of the gain over none; val score per fold
        for m in ["km", "gmm", "kmbal", "kmshuf"]:
            for k in KS + (KS_EXTRA if m in METHODS else []):
                g = U[f"r_{m}{k}"] - U.r_none
                kswe.append(dict(dataset=ds, family="ridge", method=m, k=k, in_grid=k in KS, gain_mean=g.mean(),
                                 gain_sem=g.std(ddof=1) / np.sqrt(len(g)), units_up=int((g > 0).sum()), n_units=len(g),
                                 gain_vs_true=(U[f"r_{m}{k}"] - U.r_true).mean(),
                                 p_vs_none=wil(U[f"r_{m}{k}"], U.r_none)["p"]))
        vk = pd.read_csv(wd() / "val_k.csv").assign(dataset=ds)
        valk.append(vk)
        chosen.append(pd.read_csv(wd() / "chosen_k.csv").assign(dataset=ds))
        fits.append(pd.read_csv(wd() / "mlp_fits.csv").assign(dataset=ds))
        # Within each true type: score pooled over the type's neurons per unit, Wilcoxon over units
        tv = ["r_none", "r_true", "r_S", "r_sel", "r_nt", "r_selshuf", "m_none", "m_true", "m_S", "m_cl", "m_clnt",
              "m_cont", "m_contT", "m_clsh"]
        by = "session"
        for t in TYPES:
            sub = ns[ns.type == t]
            ut = unit_scores(sub, tv, by)
            pooled = unit_scores(sub.assign(all="all"), tv, "all").iloc[0]
            for a, b in [("r_true", "r_none"), ("r_sel", "r_none"), ("r_nt", "r_none"), ("r_S", "r_none"),
                         ("r_sel", "r_true"), ("r_selshuf", "r_none"), ("m_true", "m_none"), ("m_cl", "m_none"),
                         ("m_clnt", "m_none"), ("m_cont", "m_none"), ("m_contT", "m_none"), ("m_S", "m_none"),
                         ("m_cl", "m_true"), ("m_cont", "m_true"), ("m_contT", "m_cont"), ("m_clsh", "m_none")]:
                typ.append(dict(dataset=ds, type=t, n_neurons=len(sub), a=a, b=b, pooled_gain=pooled[a] - pooled[b],
                                **wil(ut[a], ut[b])))
        # History length (task 4); minutes = mean over units of the summary's length
        hf = pd.read_csv(wd() / "history_frames.csv")
        minutes_of = {h: (hf[f"{h} frames"] / hf.hz / 60).mean() for h in [n for n, _ in HIST] + ["full train part"]}
        cf = cl
        for v in ["r_sel", "m_cl", "m_cont"]:
            none = "r_none" if v.startswith("r_") else "m_none"
            for h, mins in HIST + [("full train part", None)]:
                col = v if mins is None else f"{v}_{HTAG[h]}"
                g = U[col] - U[none]
                r = dict(dataset=ds, variant=v, history=h, minutes=minutes_of[h], gain_mean=g.mean(), gain_sem=g.std(ddof=1) / np.sqrt(len(g)),
                         units_up=int((g > 0).sum()), n_units=len(g), p_vs_none=wil(U[col], U[none])["p"],
                         gain_vs_full=(U[col] - U[v]).mean(), p_vs_full=wil(U[col], U[v])["p"] if mins else np.nan)
                if v != "m_cont" and mins is not None:
                    from sklearn.metrics import adjusted_rand_score
                    same, ari = [], []
                    for k, C in cf.items():
                        held = (neu.fold == k).to_numpy()
                        ks = int(ns.loc[ns.fold == k, "k_sel"].iloc[0])
                        a_, b_ = C["lab"]["km", ks][held], C["lab_hist"]["km", ks, h][held]
                        same.append((a_ == b_).sum()); ari.append(adjusted_rand_score(a_, b_) * held.sum())
                    r["frac_same_cluster_as_full"] = sum(same) / len(ns)
                    r["ari_vs_full"] = sum(ari) / len(ns)
                hist.append(r)
        # Cluster vs mouse / session / type (task 3)
        a_, c_ = ami_rows(ds, cl, neu, ns.groupby("fold").k_sel.first().to_dict())
        ami += a_
        cont += c_
        # Cluster profiles at the chosen k (fold 0): mean standardised S feature per cluster
        C0 = cl[0]
        ks = int(ns.loc[ns.fold == 0, "k_sel"].iloc[0])
        lab = C0["lab"]["km", ks]
        Z = C0["Z"]["full"]
        for c in range(ks):
            m = lab == c
            prof.append(dict(dataset=ds, fold=0, k=ks, cluster=c, n=int(m.sum()),
                             **{f: float(Z[m, j].mean()) for j, f in enumerate(names)}))
    OUT.mkdir(parents=True, exist_ok=True)
    ns = pd.concat(tables, ignore_index=True)
    sse_cols = [c for c in ns.columns if c.startswith("sse_") and c != "sse_copy"]
    for c in sse_cols:
        ns[c.replace("sse_", "score_")] = 1 - ns[c] / ns.sse_copy
    score_cols = [c.replace("sse_", "score_") for c in sse_cols]
    # scores rounded more coarsely to keep the file small
    out = ns[[c for c in ns.columns if not c.startswith("score_")]].copy()
    for c in score_cols:
        out[c] = ns[c].round(6)
    out.to_csv(OUT / "neuron_scores.csv", index=False, float_format="%.7g")
    pd.concat(reps).to_csv(OUT / "reproduction_check.csv", index=False, float_format="%.6g")
    U = pd.concat(units, ignore_index=True)
    U.to_csv(OUT / "unit_scores.csv", index=False, float_format="%.6g")
    st = pd.DataFrame(stats)
    st.to_csv(OUT / "stats.csv", index=False, float_format="%.4g")
    pd.DataFrame(kswe).to_csv(OUT / "k_sweep.csv", index=False, float_format="%.5g")
    pd.concat(valk).to_csv(OUT / "val_k.csv", index=False, float_format="%.6g")
    pd.concat(chosen).to_csv(OUT / "chosen_k.csv", index=False)
    pd.concat(fits).to_csv(OUT / "mlp_fits.csv", index=False, float_format="%.5g")
    pd.DataFrame(typ).to_csv(OUT / "type_gain.csv", index=False, float_format="%.4g")
    pd.DataFrame(hist).to_csv(OUT / "history.csv", index=False, float_format="%.5g")
    pd.DataFrame(ami).to_csv(OUT / "ami.csv", index=False, float_format="%.4g")
    pd.DataFrame(cont).to_csv(OUT / "contingency.csv", index=False)
    pd.DataFrame(prof).to_csv(OUT / "cluster_profiles.csv", index=False, float_format="%.3g")
    # Gains summary (plot b)
    gs = []
    for (ds, level), u in U.groupby(["dataset", "level"], sort=False):
        if (ds, level) == ("bugeon", "mouse"):
            continue
        for v in RIDGE_MAIN + MLP_MAIN:
            none = "r_none" if v.startswith("r_") else "m_none"
            g = u[v] - u[none]
            gs.append(dict(dataset=ds, unit=level, family="ridge" if v.startswith("r_") else "MLP", variant=v,
                           name=NAMES[v], score_mean=u[v].mean(), gain_mean=g.mean(),
                           gain_sem=g.std(ddof=1) / np.sqrt(len(g)), units_up=int((g > 0).sum()), n_units=len(g)))
    pd.DataFrame(gs).to_csv(OUT / "gains.csv", index=False, float_format="%.5g")
    pd.set_option("display.width", 250)
    print(pd.concat(reps).to_string(index=False))
    print(pd.DataFrame(gs).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(st[st.test != "mice (signs only)"].drop(columns=["type"]).to_string(index=False,
                                                                               float_format=lambda v: f"{v:.4g}"))
    sizes = {p.name: p.stat().st_size / 1e6 for p in OUT.iterdir()}
    print(f"results/step13: {sum(sizes.values()):.1f} MB;", {k: round(v, 2) for k, v in sizes.items()})


# ---------------------------------------------------------------- main
def main():
    ds = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else "all"
    if ds == "both":
        assert phase == "analyse"
        return analyse()
    configure(ds)
    phases = ["prep", "features", "cluster", "ridge", "mlp", "collect"] if phase == "all" else [phase]
    for ph in phases:
        print(f"\n######## {ds}: phase {ph} ########", flush=True)
        t0 = time.time()
        {"prep": prep, "features": features, "cluster": cluster, "ridge": ridge, "mlp": mlp, "collect": collect}[ph]()
        print(f"phase {ph} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
