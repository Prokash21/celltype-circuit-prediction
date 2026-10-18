"""Step 15: does the signal type (dF/F vs deconvolved) explain the small cell-type
label gain in Bugeon?

Step 11 / 13 (Allen, dF/F, 10 Hz, one Cre line per recording): genetic label gain
+0.024 (MLP) / +0.025 (ridge), activity-summary gain +0.044 / +0.047. Steps 12 / 13
(Bugeon, deconvolved, 4.3 Hz, types mixed in a recording): label gain +0.0018 /
+0.0014, summary gain +0.017 / +0.013. Alternative to the confound explanation
(K5): type information lives in the smooth fluorescence shape (rise / decay,
indicator kinetics), which deconvolution removes. Here the Allen dF/F is
deconvolved and the key comparisons are rerun.

Same 46 Allen sessions (VISp, three_session_A, Cux2 / Sst / Vip / Pvalb, one per
mouse), 5 leave-mice-out folds, chronological 80/20 split, seeds and model settings
as steps 8-13. Datasets (`ds`):
- `dff`: the step 8 cache (10 Hz z-scored dF/F; reproduction of steps 11 / 13).
- `A`: OASIS-deconvolved activity, binned to 10 Hz exactly as the dF/F (z-score
  per neuron at the native ~30 Hz with the train-part mean / std, then 3-frame
  means inside the train and test parts); past 30 -> next 10 bins.
- `B`: the same deconvolved activity in 7-frame means (~4.3 Hz, as Bugeon), then
  z-scored with the train-part statistics of the bins; past 13 -> next 4 frames
  (3.0 s -> 0.93 s), test stride 4.

Deconvolution (`deconv`): OASIS (Friedrich, Zhou & Paninski 2017; pip package
oasis-deconv 0.3.2), AR(1) with an L1 penalty. Per neuron, on the train part only
(first 80% of the 30 Hz frames): `oasis.functions.deconvolve(y_train, penalty=1)`
estimates the decay g (autocovariance, fudge factor 0.98) and the noise sn (PSD),
then optimises the baseline b and the sparsity penalty lam under the noise
constraint |c - y|^2 = sn^2 T. The whole trace is then deconvolved with these
fixed g, b, lam (`oasisAR1(y - b, g, lam)`), so no parameter sees the test part.

Phases (`python scripts/step15_signal.py <ds> <phase>`; the dataset-free ones as
`python scripts/step15_signal.py all <phase>`). Intermediate files go to
data/step15/ (not committed).
- `all extract`: step 10's extract (data/cache, step 8 format, Allen S3 mirror),
  also keeping the native-rate dF/F and stimulus features of each session
  (data/step15/raw30/) while the NWB file is on disk.
- `all deconv`: OASIS per neuron -> data/step15/A/<id>.npz, data/step15/B/<id>.npz
  (same keys as the step 8 cache), OASIS parameters, sparsity (and Bugeon's).
  `all deconv_bfree`: sensitivity version A with a free (possibly negative)
  baseline, b_nonneg=False -> data/step15/A_bfree/ (ridge only).
- `<ds> gram`: per-neuron ridge sufficient statistics (step 13 `allen_gram`).
- `<ds> features`: S summary per neuron from the train part (step 11 `s_features`,
  37 features; for B the autocorrelation lags and power bands are step 12's 4.3 Hz
  ones, S_LAGS / S_BANDS).
- `<ds> cluster`: per fold, training mice only: z-score, PCA 95%, k-means k = 16
  (n_init 10, seed = fold) as step 13 (k = 16 was its val choice in every fold).
- `<ds> ridge`: step 12 / 13 ridge from summed per-neuron statistics: b (none,
  line-balanced), d (one model per Cre line), e (step 8 shuffled session groups,
  line-balanced, 5 seeds), k1 (one model, equal row weights) and km16 (one model
  per cluster, equal row weights, as step 13).
- `<ds> mlp`: step 9 / 13 FiLM MLP (same optimiser, line-balanced batches, epochs,
  early stopping and seeding): m1 none, m2 Cre FiLM, m3 shuffled FiLM (step 9
  groups, 5 seeds), cont = continuous S summary FiLM (step 13). dff: only fold 0
  of m1 / m2 / cont is refit, as a reproduction check; the other SSE are read from
  results/step11 and results/step13.
- `<ds> collect`: per-neuron SSE of every variant -> data/step15/<ds>/sse.csv.
- `all classify` (task 3): step 11's S classifier (s11.fit_featset: quantile ->
  logistic regression or gradient boosting, inner CV over training mice) on the S
  features of dff (full precision), A and B.
- `all analyse`: results/step15/*.csv (scores, gains, per-line gains, fraction of
  the dF/F label gain that survives, Wilcoxon, bootstrap, Bugeon reference rows,
  Bugeon file inventory).
"""
import json
import os
import pickle
import sys
import time
import warnings
import zipfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step3_linear
import step4_stimulus
import step5_shared
import step8_medium as s8
import step9_mlp as s9
import step10_inferred as s10
import step11_confound as s11
import step12_bugeon as s12
import step13_dynamics as s13
from step3_linear import ALPHAS, CACHE, ROOT, TRAIN_FRAC, VAL_FRAC, downsample
from step8_medium import CRE_LINES, N_FOLDS, pick_sessions
from step10_inferred import LINES, recovered

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step15"
RAW30 = WORK / "raw30"
OUT = ROOT / "results/step15"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
DATASETS = ["dff", "A", "B"]
BIN_OF = {"A": 3, "B": 7}       # 30 Hz frames per bin
K = 16                           # clusters (step 13's val choice in every fold)
N_E = 5                          # ridge e / MLP m3 seeds (step 8 / 9)
N_SEEDS = {"m1": 3, "m2": 3, "m3": 5, "cont": 3}
SEED_OFF = {"m1": 0, "m2": 1, "m3": 2, "cont": 1}  # step 9 formula; cont uses m2's seeds (step 13)
N_BOOT = 2000
DS = None
P = F = None


# ---------------------------------------------------------------- configuration
def npz_dir(ds):
    return s8.NPZ if ds == "dff" else WORK / ds


def configure(ds):
    """Point the step 3-13 helpers at the cache and window sizes of dataset ds. They
    read these module globals at call time; each process configures once, before
    any worker pool forks."""
    global DS, P, F
    DS = ds
    P, F = (13, 4) if ds == "B" else (30, 10)
    for m in [step3_linear, step4_stimulus, step5_shared, s9]:
        m.PAST, m.FUTURE = P, F
    s9.OFF = np.arange(-P, F)
    s12.PAST, s12.FUTURE, s12.OFF, s12.CHUNK = P, F, np.arange(-P, F), 16
    s13.P, s13.F, s13.TYPES = P, F, LINES
    s9.NPZ = s13.NPZ = npz_dir(ds)
    s13.WORK, s13.DSN = WORK, ds
    (WORK / ds).mkdir(parents=True, exist_ok=True)


def wd():
    return WORK / DS


# ---------------------------------------------------------------- extract
def extract(infos):
    """Step 10 extract (step 8 cache + raw dF/F statistics); while the NWB is on
    disk, also save the native-rate dF/F and stimulus features (step 4)."""
    from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
    RAW30.mkdir(parents=True, exist_ok=True)
    load = step5_shared.load_session

    def load_keep(info):
        d = load(info)
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{info['id']}.nwb"))
        t, dff = ds.get_dff_traces()
        Fs, _ = step4_stimulus.stimulus_features(ds, dff.shape[1])
        np.savez(RAW30 / f"{info['id']}.npz", dff=dff.astype(np.float32), F=Fs.astype(np.float32), t=t,
                 fr=d["fr"], n_train=int(TRAIN_FRAC * dff.shape[1]))
        return d

    for info in infos:
        if not (RAW30 / f"{info['id']}.npz").exists() and (s8.NPZ / f"{info['id']}.npz").exists():
            (s8.NPZ / f"{info['id']}.npz").unlink()  # rebuild, so the 30 Hz traces get saved
    step5_shared.load_session = load_keep
    s10.extract(infos)


# ---------------------------------------------------------------- deconvolution (task 1)
def oasis_neuron(y, n_tr, b_nonneg=True):
    """OASIS AR(1) / L1 with every parameter from the train part y[:n_tr]: decay g and
    noise sn estimated, baseline b and penalty lam optimised under the noise
    constraint; the full trace is then deconvolved with these fixed g, b, lam."""
    from oasis.functions import GetSn, deconvolve
    from oasis.oasis_methods import oasisAR1
    _, s_tr, b, g, lam = deconvolve(y[:n_tr], penalty=1, b_nonneg=b_nonneg)
    g = float(np.ravel(g)[0])
    c, s = oasisAR1(y - b, g, lam=lam)
    sn = GetSn(y[:n_tr])
    rss = ((y[:n_tr] - b - c[:n_tr]) ** 2).mean() / sn ** 2
    # max_rel_diff_train: constrained fit vs the fixed-parameter rerun on the train part; where it is
    # large, the difference is at the last train frame only (OASIS's end-of-trace pool correction)
    return s, dict(g=g, b=b, lam=lam, sn=sn, rss_over_sn2_train=rss,
                   max_rel_diff_train=float(np.abs(s[:n_tr] - s_tr).max() / max(s_tr.max(), 1e-12)))


def binned(s, n_tr, k):
    """k-frame means inside the train and the test part (step 3 downsample)."""
    return downsample(s[:, :n_tr], k), downsample(s[:, n_tr:], k)


def deconv_session(args):
    info, variant = args
    r = np.load(RAW30 / f"{info['id']}.npz")
    dff, Fs, n_tr, fr = r["dff"].astype(np.float64), r["F"].astype(np.float64), int(r["n_train"]), float(r["fr"])
    assert np.isfinite(dff).all()
    c8 = np.load(s8.NPZ / f"{info['id']}.npz")
    tag = "" if variant == "" else f"_{variant}"
    done = WORK / "oasis" / f"{info['id']}{tag}.csv"
    out = {}
    for name in ["A", "B"] if variant == "" else [variant]:
        if not (WORK / name / f"{info['id']}.npz").exists():
            out[name] = True
    if not out and done.exists():
        return
    S30, rows = [], []
    for i in range(len(dff)):
        s, p = oasis_neuron(dff[i], n_tr, b_nonneg=variant != "A_bfree")
        S30.append(s)
        rows.append(dict(id=info["id"], neuron=i, line=LINES[CRE_LINES.index(info["cre"])], **p,
                         tau_s=-1 / np.log(p["g"]) / fr))
    S30 = np.array(S30)
    for name in out:
        k = BIN_OF[name[0]]
        if name.startswith("A"):
            # As step 5 load_session: z-score at the native rate with the train-part statistics, then bin
            mu, sd = S30[:, :n_tr].mean(1, keepdims=True), S30[:, :n_tr].std(1, keepdims=True)
            ztr, zte = binned((S30 - mu) / np.where(sd > 0, sd, 1), n_tr, k)
        else:
            # Bugeon-like: bin first, then z-score with the train-part statistics of the bins
            btr, bte = binned(S30, n_tr, k)
            mu, sd = btr.mean(1, keepdims=True), btr.std(1, keepdims=True)
            sd = np.where(sd > 0, sd, 1)
            ztr, zte = (btr - mu) / sd, (bte - mu) / sd
        Str, Ste = binned(Fs, n_tr, k)
        if k == 3:  # same stimulus bins as the dF/F cache
            assert np.allclose(Str, c8["Str_all"], atol=1e-6) and np.allclose(Ste, c8["Ste_all"], atol=1e-6)
        n_fit = ztr.shape[1] - int(round(VAL_FRAC * ztr.shape[1]))
        if k == 3:
            assert n_fit == int(c8["n_fit"]) and ztr.shape == c8["ztr"].shape and zte.shape == c8["zte"].shape
        (WORK / name).mkdir(parents=True, exist_ok=True)
        tmp = WORK / name / f"{info['id']}.tmp.npz"
        np.savez(tmp, ztr=ztr.astype(np.float32), zte=zte.astype(np.float32), Str_all=Str.astype(np.float32),
                 Ste_all=Ste.astype(np.float32), n_fit=n_fit, fr=fr, hz=fr / k, n_neurons=len(dff),
                 depth=info["depth"], n_zero_var=int((S30[:, :n_tr].std(1) == 0).sum()))
        tmp.rename(WORK / name / f"{info['id']}.npz")
    df = pd.DataFrame(rows)
    # Sparsity: fraction of exact zeros, train and test part, native rate and bins
    for part, sl in [("train", slice(0, n_tr)), ("test", slice(n_tr, None))]:
        df[f"zeros_30Hz_{part}"] = (S30[:, sl] == 0).mean(1)
        for name, k in [("10Hz", 3), ("4.3Hz", 7)]:
            df[f"zeros_{name}_{part}"] = (downsample(S30[:, sl], k) == 0).mean(1)
    df["n_frames"], df["n_train_frames"], df["fr"] = dff.shape[1], n_tr, fr
    df.to_csv(done, index=False, float_format="%.6g")
    print(f"{info['cre']:14s} {info['id']}: {len(dff)} neurons, tau median {df.tau_s.median():.2f} s, lam > 0 in"
          f" {(df.lam > 0).sum()}, zeros train 30 Hz {df.zeros_30Hz_train.mean():.3f} / 4.3 Hz"
          f" {df['zeros_4.3Hz_train'].mean():.3f}", flush=True)


def examples(infos):
    """Example traces for plot (a): per line, the neuron (over all the line's mice) with
    the largest train-part dF/F 99th percentile (clear transients); first 60 s of the
    test part as dF/F (30 Hz), deconvolved (30 Hz, same OASIS fit as `deconv`), A
    (10 Hz) and B (4.3 Hz) z."""
    def p99(info):
        r = np.load(RAW30 / f"{info['id']}.npz")
        return np.percentile(r["dff"][:, :int(r["n_train"])], 99, axis=1)

    out = []
    for line in LINES:
        info = max((i for i in infos if LINES[CRE_LINES.index(i["cre"])] == line), key=lambda i: p99(i).max())
        r = np.load(RAW30 / f"{info['id']}.npz")
        dff, n_tr, fr = r["dff"].astype(np.float64), int(r["n_train"]), float(r["fr"])
        i = int(np.argmax(np.percentile(dff[:, :n_tr], 99, axis=1)))
        s, _ = oasis_neuron(dff[i], n_tr)
        L = int(60 * fr)
        A = np.load(WORK / "A" / f"{info['id']}.npz")["zte"][i]
        B = np.load(WORK / "B" / f"{info['id']}.npz")["zte"][i]
        ex = pd.concat([pd.DataFrame(dict(signal=sig, t=np.arange(len(v)) / hz, value=v)) for sig, v, hz in
                        [("dF/F (30 Hz)", dff[i, n_tr:n_tr + L], fr), ("deconvolved (30 Hz)", s[n_tr:n_tr + L], fr),
                         ("deconvolved A, z (10 Hz)", A[:L // 3], fr / 3), ("deconvolved B, z (4.3 Hz)", B[:L // 7], fr / 7)]])
        ex.insert(0, "neuron", i)
        ex.insert(0, "id", info["id"])
        ex.insert(0, "line", line)
        out.append(ex)
    pd.concat(out).to_csv(OUT / "example_traces.csv", index=False, float_format="%.5g")
    print(f"Saved {OUT}/example_traces.csv")


def bugeon_sparsity():
    """Fraction of exact zeros of Bugeon's frame.neuralActivity for the labelled neurons
    of step 12 (all blocks; train part = first 80% of each block)."""
    neu = pd.read_csv(ROOT / "results/step12/neurons.csv")
    rows = []
    for s, g in neu.groupby("session", sort=False):
        mouse, date = s.split("_", 1)
        z_tr = z_all = n_tr = n_all = 0
        for bdir in sorted(p for p in (s12.RAW / mouse / date).glob("*/*") if p.is_dir()):
            A = np.load(bdir / "frame.neuralActivity.npy")[:, g.orig.to_numpy()]
            t = int(TRAIN_FRAC * len(A))
            z_tr, z_all = z_tr + (A[:t] == 0).sum(0), z_all + (A == 0).sum(0)
            n_tr, n_all = n_tr + t, n_all + len(A)
        rows.append(pd.DataFrame(dict(session=s, neuron=g.neuron, type=g.type, zeros_train=z_tr / n_tr,
                                      zeros_all=z_all / n_all, min_value_ok=True)))
    return pd.concat(rows, ignore_index=True)


def deconv(infos, variant=""):
    (WORK / "oasis").mkdir(parents=True, exist_ok=True)
    with Pool(N_WORKERS) as pool:
        pool.map(deconv_session, [(i, variant) for i in infos], chunksize=1)
    tag = "" if variant == "" else f"_{variant}"
    pd.concat([pd.read_csv(WORK / "oasis" / f"{i['id']}{tag}.csv") for i in infos]) \
        .to_csv(WORK / f"oasis_params{tag}.csv", index=False, float_format="%.6g")
    if variant == "":
        bugeon_sparsity().to_csv(WORK / "bugeon_sparsity.csv", index=False, float_format="%.5g")


# ---------------------------------------------------------------- ridge statistics and S summary
def gram(infos):
    with Pool(N_WORKERS) as pool:
        pool.map(s13.allen_gram, infos, chunksize=1)


def s_features_ds(z, S, hz):
    """Step 11 S features. dff / A: as step 11 (10 Hz). B (4.3 Hz): step 12's
    autocorrelation lags (frames) and power bands; the train-part z std is 1 by
    construction there, so it is dropped (as step 12)."""
    if DS != "B":
        return s11.s_features(z, S, hz)
    lags, bands = s11.AC_LAGS, s11.BANDS
    s11.AC_LAGS, s11.BANDS = s12.S_LAGS, s12.S_BANDS
    try:
        f = s11.s_features(z, S, hz)
    finally:
        s11.AC_LAGS, s11.BANDS = lags, bands
    f.pop("z std (10 Hz)")
    return {(f"autocorr {int(round(float(k.split()[1]) * 10))} fr ({float(k.split()[1]) * 10 / 4.3:.2g} s)"
             if k.startswith("autocorr") else k): v for k, v in f.items()}  # lag names at the nominal 4.3 Hz (step 12)


def feats_session(info):
    c = np.load(npz_dir(DS) / f"{info['id']}.npz")
    hz = float(c["fr"]) / (BIN_OF.get(DS, 3))
    return s_features_ds(c["ztr"].astype(np.float64), c["Str_all"].astype(np.float64), hz)


def features(infos):
    with Pool(N_WORKERS) as pool:
        res = pool.map(feats_session, infos)
    names = list(res[0])
    A = np.vstack([np.column_stack([r[c] for c in names]) for r in res])
    A[~np.isfinite(A)] = np.nan
    assert np.isfinite(A).all(), "non-finite S feature on the full train part"
    np.savez(wd() / "features.npz", names=np.array(names), full=A)
    print(f"{len(names)} S features x {len(A)} neurons")


def load_features():
    z = np.load(wd() / "features.npz")
    return list(z["names"]), z["full"]


def cluster_fold(args):
    """step13_dynamics.cluster_fold at k = 16 only (same standardisation, PCA and
    k-means seeds, so the same labels as step 13's k = 16)."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    k_fold, X, tr = args
    with threadpool_limits(1):
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
        pca = PCA(n_components=0.95, svd_solver="full").fit(Z[tr])
        Pc = pca.transform(Z)
        km = KMeans(n_clusters=K, n_init=10, random_state=k_fold).fit(Pc[tr])
        lab = km.predict(Pc)
    print(f"fold {k_fold}: {pca.n_components_} PCs, smallest training cluster {np.bincount(lab[tr], minlength=K).min()}",
          flush=True)
    return k_fold, dict(lab=lab, Z=Z.astype(np.float32), n_pc=pca.n_components_)


def cluster():
    neu = s13.neurons()
    _, X = load_features()
    with Pool(min(N_WORKERS, N_FOLDS)) as pool:
        res = dict(pool.map(cluster_fold, [(k, X, (neu.fold != k).to_numpy()) for k in range(N_FOLDS)]))
    pickle.dump(res, open(wd() / "clusters.pkl", "wb"))


def load_clusters():
    return pickle.load(open(wd() / "clusters.pkl", "rb"))


# ---------------------------------------------------------------- ridge (task 2)
def shuffled_session_groups(infos, k, seed):
    """Step 8 / 10 e grouping: training sessions in 4 random equal-size groups that
    each mix lines (same draws), then a random group per held-out session."""
    rng = np.random.default_rng(seed)
    lab = s8.shuffled_groups([dict(info=i) for i in infos if i["fold"] != k], rng)
    for i in infos:
        if i["fold"] == k:
            lab[i["id"]] = int(rng.integers(len(LINES)))
    return lab


def ridge(infos):
    neu = s13.neurons()
    G = s13.load_grams(neu)
    si = neu.si.to_numpy()
    neu["rows"] = G["f_n"][si] + G["v_n"][si]
    t_arr, ids = neu.t.to_numpy(), neu.id.to_numpy()
    cl = load_clusters() if (wd() / "clusters.pkl").exists() else None
    rows, alphas = [], []
    for k in range(N_FOLDS):
        t0 = time.time()
        held = (neu.fold == k).to_numpy()
        tr = ~held
        b = s12.fit_weighted(G, si, s12.balanced_weights(neu, tr))
        d = [s12.fit_weighted(G, si, (tr & (t_arr == c)).astype(float)) for c in range(len(LINES))]
        e = []
        for seed in range(N_E):
            lab = shuffled_session_groups(infos, k, seed)
            grp = np.array([lab[i] for i in ids])
            e.append((grp, [s12.fit_weighted(G, si, s12.balanced_weights(neu, tr & (grp == g)))
                            for g in range(len(LINES))]))
        one = s13.fit_group(G, si, tr.astype(float))
        km = None
        if cl is not None:
            lab16 = cl[k]["lab"]
            km = s13.fit_clusters(G, si, neu, lab16, tr)
        alphas.append(dict(fold=k, b=b[2], **{f"d_{LINES[c]}": d[c][2] for c in range(len(LINES))},
                           e=";".join(f"{m[2]:g}" for _, ms in e for m in ms), k1=one["alpha"],
                           km16=";".join(f"{km[c]['alpha']:g}" for c in sorted(km)) if km else ""))
        for s in neu[held].session.unique():
            X, Y, S = s13.test_data(s)
            rr = np.where((neu.session == s).to_numpy())[0]
            err = lambda Pr: ((Pr - Y) ** 2).sum(axis=(0, 2))
            r = dict(copy=err(np.repeat(X[:, :, -1:], F, axis=2)), r_none=err(s12.predict(b[0], b[1], X, S)))
            for c in range(len(LINES)):
                r[f"d_{LINES[c]}"] = err(s12.predict(d[c][0], d[c][1], X, S))
            r["r_e"] = np.mean([err(s12.predict(*ms[grp[rr[0]]][:2], X, S)) for grp, ms in e], axis=0)
            r["r_k1"] = err(s12.predict(one["W"], one["b"], X, S))
            if km is not None:
                out = np.empty(len(rr))
                for c in np.unique(lab16[rr]):
                    mod = km.get(c)
                    W_, b_ = (mod["W"], mod["b"]) if mod else b[:2]
                    m = lab16[rr] == c
                    out[m] = err(s12.predict(W_, b_, X, S))[m]
                r["r_km16"] = out
            rows.append(pd.DataFrame(dict(session=s, neuron=neu.neuron.to_numpy()[rr], n_el=Y.shape[0] * F,
                                          **{f"sse_{m}": v for m, v in r.items()})))
        print(f"fold {k}: alpha b {b[2]:g}, d {[d[c][2] for c in range(len(LINES))]} ({time.time() - t0:.0f} s)",
              flush=True)
    pd.concat(rows).to_csv(wd() / "ridge_sse.csv", index=False, float_format="%.9g")
    pd.DataFrame(alphas).to_csv(wd() / "ridge_alphas.csv", index=False)
    print(f"Saved {wd()}/ridge_sse.csv, ridge_alphas.csv")


# ---------------------------------------------------------------- MLP (task 2)
COND, M3 = {}, {}


def load_mlp_data():
    s9.load()
    neu = s13.neurons()
    assert [str(s["info"]["id"]) for s in s9.SESS] == list(neu.session.unique())
    s13.NEU, s13.T = neu, neu.t.to_numpy()
    s13.OFFS = np.cumsum([0] + [s["n"] for s in s9.SESS])[:-1]
    for k, C in load_clusters().items():
        COND[k] = C["Z"]
    si = neu.si.to_numpy()
    for j in s9.make_jobs():  # step 9 m3 labels (session -> fake group), per neuron
        if j["kind"] == "m3":
            M3[j["fold"], j["seed"]] = np.array([j["labels"][i] for i in si])


def run_job(job):
    """step13_dynamics.run_job (Allen branch: step 9 optimiser, line-balanced batches,
    epochs, early stopping and seeding) for m1 / m2 / m3 / cont."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, kind, seed = job["fold"], job["kind"], job["seed"]
    rng = np.random.default_rng(1000 * fold + seed + SEED_OFF[kind] * 10)
    torch.manual_seed(int(rng.integers(2 ** 31)))
    lab, cont, n_lab = {"m1": (None, None, 0), "m2": (s13.T, None, len(LINES)),
                        "m3": (M3.get((fold, seed)), None, len(LINES)), "cont": (None, COND[fold], 0)}[kind]
    model = s13.make_model(n_lab, cont)
    opt = torch.optim.Adam(model.parameters(), lr=s9.LR)
    train = [i for i, s in enumerate(s9.SESS) if s["info"]["fold"] != fold]
    smp = s13.AllenSampler(train, "tr", rng)
    n_t = len(smp.lines)
    xv, yv, rv = s13.AllenSampler(train, "va", np.random.default_rng(fold)).draw(s9.N_VAL)
    xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    v_type = s13.T[rv]

    def val_loss():
        model.eval()
        with torch.no_grad():
            err = ((s13.call(model, xv, lab, cont, rv) - yv) ** 2).mean(1).numpy()
        model.train()
        return float(np.mean([err[v_type == t].mean() for t in smp.lines]))

    best, best_state, bad, hist = val_loss(), None, 0, []
    for ep in range(s9.MAX_EPOCHS):
        for _ in range(s9.EPOCH_BATCHES):
            x, y, r = smp.draw(s9.BATCH // n_t)
            loss = ((s13.call(model, torch.from_numpy(x), lab, cont, r) - torch.from_numpy(y)) ** 2).mean()
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
    held_s = sorted(s13.NEU[s13.NEU.fold == fold].si.unique())
    out = {si: s13.forecast(model, si, lab, cont) for si in held_s}
    return dict(job=job, out=out, epochs=len(hist), best_val=best, hist=hist, secs=time.time() - t0)


def mlp_jobs():
    if DS == "dff":  # reproduction check of steps 11 / 13: fold 0 only
        return [dict(fold=0, kind=kind, seed=seed) for kind in ["m1", "m2", "cont"] for seed in range(N_SEEDS[kind])]
    return [dict(fold=k, kind=kind, seed=seed) for k in range(N_FOLDS) for kind in ["m1", "m2", "m3", "cont"]
            for seed in range(N_SEEDS[kind])]


def mlp():
    load_mlp_data()
    path = wd() / "mlp.pkl"
    done = pickle.load(open(path, "rb")) if path.exists() else []
    have = {tuple(r["job"].values()) for r in done}
    jobs = [j for j in mlp_jobs() if tuple(j.values()) not in have]
    jobs.sort(key=lambda j: j["kind"] == "m3")  # m3 last
    print(f"{DS}: {len(jobs)} MLP fits to run ({len(have)} done) in {N_WORKERS} workers; P {P}, F {F},"
          f" inputs {P + 12 * (P + F)}", flush=True)
    results = list(done)
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}/{len(jobs) + len(done)}] fold {j['fold']} {j['kind']} seed {j['seed']}:"
                  f" {r['epochs']} epochs, best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)
            pickle.dump(results, open(path, "wb"))  # checkpoint, so an interrupted run resumes


def collect():
    """Per-neuron SSE of every variant (MLP: mean over seeds) -> data/step15/<ds>/sse.csv."""
    neu = s13.neurons()
    ns = neu[["session", "id", "fold", "neuron", "type", "depth"]].copy()
    ns = ns.merge(pd.read_csv(wd() / "ridge_sse.csv").astype({"session": str}), on=["session", "neuron"],
                  how="left", validate="1:1")
    D = ns[[f"sse_d_{t}" for t in LINES]].to_numpy()
    ns["sse_r_true"] = D[np.arange(len(ns)), pd.Index(LINES).get_indexer(ns.type)]
    acc = {}
    res = pickle.load(open(wd() / "mlp.pkl", "rb")) if (wd() / "mlp.pkl").exists() else []  # A_bfree: ridge only
    for r in res:
        for si, v in r["out"].items():
            acc.setdefault((r["job"]["kind"], si), []).append(v)
    tag = {"m1": "m_none", "m2": "m_true", "m3": "m_m3", "cont": "m_cont"}
    for (kind, si), v in acc.items():
        assert len(v) == N_SEEDS[kind], (kind, si, len(v))
        ns.loc[neu.si == si, f"sse_{tag[kind]}{'_rerun' if DS == 'dff' else ''}"] = np.mean(v, axis=0)
    if DS == "dff":  # all folds of m1 / m2 / m3 from step 11, cont from step 13
        key = ["session", "neuron"]
        o11 = ns[key].merge(pd.read_csv(ROOT / "results/step11/neuron_scores_all_labels.csv")
                            .assign(session=lambda d: d.id.astype(str)), on=key, how="left", validate="1:1")
        o13 = ns[key].merge(pd.read_csv(ROOT / "results/step13/neuron_scores.csv")
                            .query("dataset == 'allen'").astype({"session": str}), on=key, how="left", validate="1:1")
        M2 = o11[[f"sse_m2_{t}" for t in LINES]].to_numpy()
        ns["sse_m_none"], ns["sse_m_m3"] = o11.sse_m1.to_numpy(), o11.sse_m3.to_numpy()
        ns["sse_m_true"] = M2[np.arange(len(ns)), pd.Index(LINES).get_indexer(ns.type)]
        ns["sse_m_cont"] = o13.sse_m_cont.to_numpy()
        ns["ref11_sse_b"], ns["ref11_sse_e"], ns["ref11_sse_copy"] = o11.sse_b, o11.sse_e, o11.sse_copy
        ns["ref11_sse_d"] = o11[[f"sse_d_{t}" for t in LINES]].to_numpy()[np.arange(len(ns)),
                                                                          pd.Index(LINES).get_indexer(ns.type)]
        ns["ref13_sse_km16"], ns["ref13_sse_k1"] = o13.sse_r_km16.to_numpy(), o13.sse_r_k1.to_numpy()
    ns.to_csv(wd() / "sse.csv", index=False, float_format="%.9g")
    if res:
        pd.DataFrame([dict(**r["job"], epochs=r["epochs"], best_val=r["best_val"], secs=r["secs"]) for r in res]) \
            .sort_values(["fold", "kind", "seed"]).to_csv(wd() / "mlp_fits.csv", index=False, float_format="%.5g")
    print(f"Saved {wd()}/sse.csv ({len(ns)} neurons), mlp_fits.csv")


# ---------------------------------------------------------------- typing after deconvolution (task 3)
def classify_fold(args):
    from threadpoolctl import threadpool_limits
    k, X, y, mouse, fold = args
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    with threadpool_limits(1):
        res, sel = s11.fit_featset(X, y, mouse, tr, te, seed=k, perm=True)
    print(f"  fold {k}: {sel} (inner {res[sel]['inner']:.3f})", flush=True)
    return k, te, res, sel


def separability(X, y, names):
    """Per feature: Kruskal-Wallis epsilon^2 over the 4 lines and one-vs-rest AUC per line."""
    from scipy.stats import kruskal
    from sklearn.metrics import roc_auc_score
    rows = []
    for j, f in enumerate(names):
        x = X[:, j]
        H = kruskal(*[x[y == g] for g in range(len(LINES))]).statistic if np.ptp(x) > 0 else 0.0
        r = dict(feature=f, eps2=H / (len(x) - 1))
        for g, l in enumerate(LINES):
            r[f"auc_{l}_vs_rest"] = roc_auc_score(y == g, x) if np.ptp(x) > 0 else 0.5
        rows.append(r)
    return rows


def classify(infos):
    summ, conf, imp, preds, feat_rows, sep = [], [], [], None, [], []
    for ds in DATASETS:
        configure(ds)
        neu = s13.neurons()
        names, X = load_features()
        y, mouse, fold = neu.t.to_numpy(), neu.id.to_numpy(), neu.fold.to_numpy()
        print(f"{ds}: {X.shape[1]} S features", flush=True)
        with Pool(min(N_WORKERS, N_FOLDS)) as pool:
            outs = pool.map(classify_fold, [(k, X, y, mouse, fold) for k in range(N_FOLDS)])
        Pr = {c: np.empty((len(y), len(LINES))) for c in s11.CLFS + ["sel"]}
        sels = {}
        for k, te, res, sel in outs:
            for c in s11.CLFS:
                Pr[c][te] = res[c]["proba"]
                imp.append(pd.DataFrame(dict(ds=ds, clf=c, fold=k, feature=names, importance=res[c]["perm"])))
            Pr["sel"][te] = res[sel]["proba"]
            sels[k] = sel
        if preds is None:
            preds = neu[["id", "neuron", "type", "fold"]].rename(columns={"type": "line"})
        for c, p in Pr.items():
            pr = p.argmax(1)
            sv = np.empty(len(y), int)
            for _, g in pd.Series(np.arange(len(y))).groupby(mouse, sort=False):
                sv[g.to_numpy()] = s10.session_vote(pr[g.to_numpy()], p[g.to_numpy()])
            ses = pd.DataFrame(dict(id=mouse, y=y, v=sv)).drop_duplicates("id")
            per_fold = [s11.bacc(y[fold == k], pr[fold == k]) for k in range(N_FOLDS)]
            C = s11.confusion(y, pr)
            frac = pd.Series(pr == y).groupby(mouse, sort=False).mean()
            summ.append(dict(ds=ds, clf=c, n_features=X.shape[1], bacc_neuron=s11.bacc(y, pr),
                             bacc_fold_mean=np.mean(per_fold), bacc_fold_std=np.std(per_fold),
                             selected=";".join(sels[k] for k in range(N_FOLDS)) if c == "sel" else "",
                             inner_bacc_mean=np.mean([o[2][c if c != "sel" else o[3]]["inner"] for o in outs]),
                             session_correct=int((ses.y == ses.v).sum()), n_sessions=len(ses),
                             mouse_frac_correct_median=frac.median(), mouse_frac_correct_min=frac.min(),
                             **{f"recall_{LINES[g]}": C[g, g] / C[g].sum() for g in range(len(LINES))},
                             **{f"session_correct_{LINES[g]}": int(((ses.y == g) & (ses.v == g)).sum())
                                for g in range(len(LINES))}))
            for level, (tt, pp) in [("neuron", (y, pr)), ("session", (ses.y.to_numpy(), ses.v.to_numpy()))]:
                Cm = s11.confusion(tt, pp)
                conf += [dict(ds=ds, clf=c, level=level, true=LINES[i], pred=LINES[j], count=int(Cm[i, j]))
                         for i in range(len(LINES)) for j in range(len(LINES))]
            if c == "sel":
                preds[f"pred_{ds}"] = np.array(LINES)[pr]
                for g, l in enumerate(LINES):
                    preds[f"p_{ds}_{l}"] = p[:, g]
        df = pd.DataFrame(X, columns=names).assign(line=neu.type.to_numpy())
        for l, g in df.groupby("line", sort=False):
            for f in names:
                feat_rows.append(dict(ds=ds, feature=f, line=l, mean=g[f].mean(), std=g[f].std(), median=g[f].median()))
        sep += [dict(ds=ds, **r) for r in separability(X, y, names)]
        print(pd.DataFrame(summ)[lambda d: d.ds == ds].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    pd.DataFrame(summ).to_csv(OUT / "classifier_summary.csv", index=False, float_format="%.5g")
    pd.DataFrame(conf).to_csv(OUT / "confusions.csv", index=False)
    pd.concat(imp).groupby(["ds", "clf", "feature"], sort=False).importance.agg(["mean", "std"]).reset_index() \
        .to_csv(OUT / "feature_importance.csv", index=False, float_format="%.5g")
    preds.to_csv(OUT / "neuron_predictions.csv", index=False, float_format="%.4g")
    pd.DataFrame(feat_rows).to_csv(OUT / "features_by_line.csv", index=False, float_format="%.5g")
    pd.DataFrame(sep).to_csv(OUT / "feature_separability.csv", index=False, float_format="%.5g")
    print(f"Saved {OUT}/classifier_summary.csv, confusions.csv, feature_importance.csv, neuron_predictions.csv,"
          " features_by_line.csv, feature_separability.csv")


# ---------------------------------------------------------------- analyse (CSV outputs)
VARIANTS = ["r_none", "r_true", "r_e", "r_k1", "r_km16", "m_none", "m_true", "m_m3", "m_cont"]
GAINS = {  # name: (family, a, b)
    "label": [("ridge", "r_true", "r_none"), ("MLP", "m_true", "m_none")],
    "summary": [("ridge", "r_km16", "r_none"), ("MLP", "m_cont", "m_none")],
    "shuffled label": [("ridge", "r_e", "r_none"), ("MLP", "m_m3", "m_none")],
    "summary - label": [("ridge", "r_km16", "r_true"), ("MLP", "m_cont", "m_true")],
    "k1 (equal weights)": [("ridge", "r_k1", "r_none")],
}
DS_NAME = {"dff": "Allen dF/F", "A": "Allen deconvolved A (10 Hz)", "B": "Allen deconvolved B (4.3 Hz)",
           "A_bfree": "Allen deconvolved A, free baseline (sensitivity)", "bugeon": "Bugeon (steps 12-13)"}


def mouse_scores(ns, variants):
    g = ns.groupby("id", sort=False)
    cp = g.sse_copy.sum()
    out = pd.DataFrame({v: 1 - g[f"sse_{v}"].sum() / cp for v in variants if f"sse_{v}" in ns})
    out["line"] = g.type.first()
    out["fold"] = g.fold.first()
    out["n_neurons"] = g.size()
    return out


def bugeon_units():
    """Bugeon session scores (steps 12 / 13): none / true / km16 / cont from step 13,
    shuffled controls e / m3 from step 12."""
    u = pd.read_csv(ROOT / "results/step13/unit_scores.csv")
    u = u[(u.dataset == "bugeon") & (u.level == "session")].set_index("unit")
    sc = pd.read_csv(ROOT / "results/step12/scores.csv")
    sc = sc[sc.level == "session"]
    out = u[["r_none", "r_true", "r_km16", "m_none", "m_true", "m_cont", "mouse", "n_neurons"]].copy()
    for fam, col in [("ridge", "r_e"), ("MLP", "m_m3")]:
        s = sc[sc.family == fam].set_index("unit")
        assert np.allclose(s.loc[out.index, "none"], out["r_none" if fam == "ridge" else "m_none"], atol=2e-6)
        out[col] = s.loc[out.index, "control"]
    return out


def wil(a, b):
    return s12.wil(a, b)


def boot_frac(gx, g0, rng, n=N_BOOT):
    """Ratio of mean gains (paired units) with a percentile bootstrap CI."""
    idx = rng.integers(len(gx), size=(n, len(gx)))
    b = gx[idx].mean(1) / g0[idx].mean(1)
    return gx.mean() / g0.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def boot_gap(g0, gx, gbug, rng, n=N_BOOT):
    """Share of the Allen dF/F - Bugeon gap closed by dataset x: (G0 - Gx) / (G0 - Gbug),
    Allen mice (paired) and Bugeon sessions resampled independently."""
    i = rng.integers(len(g0), size=(n, len(g0)))
    j = rng.integers(len(gbug), size=(n, len(gbug)))
    b = (g0[i].mean(1) - gx[i].mean(1)) / (g0[i].mean(1) - gbug[j].mean(1))
    return (g0.mean() - gx.mean()) / (g0.mean() - gbug.mean()), np.percentile(b, 2.5), np.percentile(b, 97.5)


def bugeon_inventory():
    """Every file type in the Bugeon zip, with the dataset's own description
    (DatasetTypesV2.xlsx, figshare file 38046648) and whether it could hold raw
    fluorescence or dF/F."""
    import re
    z = zipfile.ZipFile(ROOT / "data/bugeon/bugeon_v2.zip")
    info = [i for i in z.infolist() if not i.is_dir()]
    desc = {}
    xl = ROOT / "data/bugeon/meta/DatasetTypesV2.xlsx"
    if xl.exists():
        d = pd.read_excel(xl, header=None).iloc[1:, :3].dropna(subset=[0])
        desc = {r[0]: f"{r[2]} [{r[1]}]" for r in d.itertuples(index=False)}
    rows = {}
    for i in info:
        base = i.filename.rsplit("/", 1)[-1]
        r = rows.setdefault(base, dict(file=base, n_files=0, total_mb=0.0, example=i.filename))
        r["n_files"] += 1
        r["total_mb"] += i.file_size / 1e6
    act = []
    for p in sorted((ROOT / "data/bugeon/raw").glob("SB*/*/*/*/frame.neuralActivity.npy")):
        A = np.load(p, mmap_mode="r")
        act.append((float(A.min()), float((A == 0).mean())))
    act = np.array(act)
    out = []
    for base, r in sorted(rows.items()):
        key = base.replace(".npy", "").replace(".txt", "")
        d = desc.get(base) or next((v for k, v in desc.items() if k.split(".")[:2] == base.split(".")[:2]), "")
        is_frame = base.startswith("frame.")
        cand = base == "frame.neuralActivity.npy"
        if cand:
            note = (f"only per-frame neural signal in the zip; described as deconvolved; min value over the"
                    f" {len(act)} block files {act[:, 0].min():.3g}, fraction of zeros {act[:, 1].min():.2f}-"
                    f"{act[:, 1].max():.2f} (all ROIs) -> deconvolved, not fluorescence")
        elif is_frame:
            note = "per-frame, but times or cortical state, not neural"
        elif re.match(r"neuron\.", base):
            note = "one value (or map) per neuron, no time axis"
        else:
            note = "stimulus, behaviour or metadata"
        out.append(dict(**r, description=d, could_hold_fluorescence="no (deconvolved)" if cand else "no", note=note))
    return pd.DataFrame(out)


def analyse(infos):
    tabs, ms = {}, {}
    for ds in DATASETS + ["A_bfree"]:
        p = WORK / ds / "sse.csv"
        if not p.exists():
            print(f"{ds}: no sse.csv, skipped")
            continue
        tabs[ds] = pd.read_csv(p)
        ms[ds] = mouse_scores(tabs[ds], VARIANTS)
    bug = bugeon_units()
    ref = ms["dff"]
    # Reproduction check (dff vs steps 11 / 13)
    ns = tabs["dff"]
    rep = []
    for new, old, folds in [("r_none", "ref11_sse_b", "all"), ("r_true", "ref11_sse_d", "all"), ("r_e", "ref11_sse_e", "all"),
                            ("r_k1", "ref13_sse_k1", "all"), ("r_km16", "ref13_sse_km16", "all"),
                            ("copy", "ref11_sse_copy", "all"), ("m_none_rerun", "sse_m_none", "0"),
                            ("m_true_rerun", "sse_m_true", "0"), ("m_cont_rerun", "sse_m_cont", "0")]:
        sub = ns if folds == "all" else ns[ns.fold == 0]
        g = sub.groupby("id")
        a = 1 - g[f"sse_{new}"].sum() / g.sse_copy.sum()
        b = 1 - g[old].sum() / g.sse_copy.sum()
        rep.append(dict(model=new, reference=old.replace("ref11_", "step 11 ").replace("ref13_", "step 13 "),
                        folds=folds, n_mice=len(a), max_abs_mouse_score_diff=float((a - b).abs().max()),
                        max_rel_neuron_sse_diff=float((np.abs(sub[f"sse_{new}"] - sub[old]) / sub[old]).max()),
                        mean_score_here=a.mean(), mean_score_reference=b.mean()))
    rep = pd.DataFrame(rep)
    rep.to_csv(OUT / "reproduction_check.csv", index=False, float_format="%.6g")

    # Per-neuron scores (every dataset), per-mouse scores
    cols = [f"sse_{v}" for v in VARIANTS] + ["sse_copy"]
    nsc = []
    for ds, t in tabs.items():
        d = t[["id", "fold", "neuron", "type", "depth", "n_el"] + [c for c in cols if c in t]].copy()
        for c in [c for c in cols if c in t and c != "sse_copy"]:
            d[c.replace("sse_", "score_")] = (1 - d[c] / d.sse_copy).round(6)
        nsc.append(d.rename(columns={"type": "line"}).assign(ds=ds))
    pd.concat(nsc)[["ds"] + [c for c in pd.concat(nsc).columns if c != "ds"]] \
        .to_csv(OUT / "neuron_scores.csv", index=False, float_format="%.7g")
    pd.concat([m.reset_index().assign(ds=ds) for ds, m in ms.items()]) \
        .to_csv(OUT / "mouse_scores.csv", index=False, float_format="%.6g")
    bug.reset_index(names="session").to_csv(OUT / "bugeon_session_scores.csv", index=False, float_format="%.6g")

    # Main table: gains per dataset (mean over units, sem, units up, Wilcoxon vs the baseline)
    rows = []
    units = dict(ms, bugeon=bug)
    for ds, u in units.items():
        for gname, specs in GAINS.items():
            for fam, a, b in specs:
                if a not in u or b not in u:
                    continue
                g = (u[a] - u[b]).to_numpy()
                rows.append(dict(ds=ds, dataset=DS_NAME[ds], unit="session" if ds == "bugeon" else "mouse",
                                 gain=gname, family=fam, a=a, b=b, score_b=u[b].mean(), score_a=u[a].mean(),
                                 gain_mean=g.mean(), gain_sem=g.std(ddof=1) / np.sqrt(len(g)),
                                 gain_rel=(g / (1 - u[b].to_numpy())).mean(), units_up=int((g > 0).sum()),
                                 n_units=len(g), p=wil(u[a], u[b])["p"]))
    gains = pd.DataFrame(rows)
    gains.to_csv(OUT / "gains.csv", index=False, float_format="%.5g")

    # Fraction of the dF/F gain that survives deconvolution; share of the Allen - Bugeon gap
    surv, rng = [], np.random.default_rng(0)
    for gname in ["label", "summary"]:
        for fam, a, b in GAINS[gname]:
            g0 = (ref[a] - ref[b]).to_numpy()
            gb = (bug[a] - bug[b]).to_numpy()
            for ds in [d for d in ms if d != "dff"]:
                if a not in ms[ds]:
                    continue
                gx = (ms[ds][a] - ms[ds][b]).loc[ref.index].to_numpy()
                f, lo, hi = boot_frac(gx, g0, rng)
                gap, glo, ghi = boot_gap(g0, gx, gb, rng)
                w = wil(gx, g0)
                surv.append(dict(gain=gname, family=fam, ds=ds, gain_dff=g0.mean(), gain_ds=gx.mean(), gain_bugeon=gb.mean(),
                                 frac_surviving=f, ci_lo=lo, ci_hi=hi, mice_ds_gain_higher=w["a_beats_b"], n_mice=w["n"],
                                 median_diff=w["median_diff"], p_wilcoxon=w["p"], gap_share_signal=gap, gap_ci_lo=glo,
                                 gap_ci_hi=ghi))
    surv = pd.DataFrame(surv)
    surv.to_csv(OUT / "survival.csv", index=False, float_format="%.5g")

    # Label gain relative to the summary gain of the same dataset and family (how much of what
    # per-neuron conditioning can give at all the genetic label gives), bootstrap over units
    lvs = []
    for ds, u in units.items():
        for (fam, a, b), (_, c, _) in zip(GAINS["label"], GAINS["summary"]):
            if a not in u or c not in u:
                continue
            gl, gs = (u[a] - u[b]).to_numpy(), (u[c] - u[b]).to_numpy()
            f, lo, hi = boot_frac(gl, gs, rng)
            lvs.append(dict(ds=ds, family=fam, n_units=len(gl), gain_label=gl.mean(), gain_summary=gs.mean(),
                            label_over_summary=f, ci_lo=lo, ci_hi=hi))
    lvs = pd.DataFrame(lvs)
    lvs.to_csv(OUT / "label_vs_summary.csv", index=False, float_format="%.5g")

    # Per Cre line: gains per dataset, deconvolved vs dF/F
    pl = []
    for line in LINES:
        for gname in ["label", "summary", "shuffled label", "summary - label"]:
            for fam, a, b in GAINS[gname]:
                m0 = ref.line == line
                g0 = (ref[a] - ref[b])[m0]
                for ds, u in ms.items():
                    if a not in u:
                        continue
                    gx = (u[a] - u[b]).loc[g0.index]
                    w0 = wil(u.loc[g0.index, a], u.loc[g0.index, b])
                    r = dict(line=line, gain=gname, family=fam, ds=ds, n_mice=len(gx), gain_mean=gx.mean(),
                             gain_sem=gx.std(ddof=1) / np.sqrt(len(gx)), mice_up=int((gx > 0).sum()), p_vs_zero=w0["p"])
                    if ds != "dff":
                        w = wil(gx, g0)
                        r.update(gain_dff=g0.mean(), frac_surviving=gx.mean() / g0.mean(),
                                 mice_higher_than_dff=w["a_beats_b"], p_vs_dff=w["p"])
                    pl.append(r)
    pl = pd.DataFrame(pl)
    pl.to_csv(OUT / "line_gains.csv", index=False, float_format="%.5g")

    # Sparsity and OASIS parameters
    op = pd.read_csv(WORK / "oasis_params.csv")
    op.to_csv(OUT / "oasis_params.csv", index=False, float_format="%.5g")
    bs = pd.read_csv(WORK / "bugeon_sparsity.csv")
    sp = []
    for line, g in list(op.groupby("line", sort=False)) + [("all", op)]:
        sp.append(dict(dataset="Allen deconvolved", group=line, n_neurons=len(g),
                       **{c: g[c].mean() for c in op.columns if c.startswith("zeros_")},
                       tau_s_median=g.tau_s.median(), tau_s_q25=g.tau_s.quantile(0.25), tau_s_q75=g.tau_s.quantile(0.75),
                       frac_lam_pos=(g.lam > 0).mean(), rss_over_sn2_median=g.rss_over_sn2_train.median()))
    for t, g in list(bs.groupby("type")) + [("all", bs)]:
        sp.append(dict(dataset="Bugeon frame.neuralActivity", group=t, n_neurons=len(g), **{"zeros_4.3Hz_train": g.zeros_train.mean(),
                                                                                              "zeros_4.3Hz_all": g.zeros_all.mean()}))
    for t, g in bs.groupby("session"):
        sp.append(dict(dataset="Bugeon frame.neuralActivity, per session", group=t, n_neurons=len(g),
                       **{"zeros_4.3Hz_train": g.zeros_train.mean(), "zeros_4.3Hz_all": g.zeros_all.mean(),
                          "zeros_4.3Hz_train_min": g.zeros_train.min(), "zeros_4.3Hz_train_max": g.zeros_train.max()}))
    pb = WORK / "oasis_params_A_bfree.csv"
    if pb.exists():
        ob = pd.read_csv(pb)
        for line, g in list(ob.groupby("line", sort=False)) + [("all", ob)]:
            sp.append(dict(dataset="Allen deconvolved, free baseline", group=line, n_neurons=len(g),
                           **{c: g[c].mean() for c in ob.columns if c.startswith("zeros_")}, tau_s_median=g.tau_s.median(),
                           frac_lam_pos=(g.lam > 0).mean(), rss_over_sn2_median=g.rss_over_sn2_train.median()))
    pd.DataFrame(sp).to_csv(OUT / "sparsity.csv", index=False, float_format="%.4g")
    examples(infos)
    bugeon_inventory().to_csv(OUT / "bugeon_files.csv", index=False, float_format="%.4g")
    fits = [pd.read_csv(WORK / ds / "mlp_fits.csv").assign(ds=ds) for ds in DATASETS if (WORK / ds / "mlp_fits.csv").exists()]
    pd.concat(fits).to_csv(OUT / "mlp_fits.csv", index=False, float_format="%.5g")
    pd.concat([pd.read_csv(WORK / ds / "ridge_alphas.csv").assign(ds=ds) for ds in tabs]) \
        .to_csv(OUT / "ridge_alphas.csv", index=False)

    pd.set_option("display.width", 250)
    print(rep.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(gains.drop(columns=["dataset", "a", "b"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(surv.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(lvs.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pl[pl.gain == "label"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(sp).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sizes = {p.name: p.stat().st_size / 1e6 for p in OUT.iterdir()}
    print(f"results/step15: {sum(sizes.values()):.1f} MB;", {k: round(v, 2) for k, v in sizes.items()})


# ---------------------------------------------------------------- main
def main():
    ds, phase = sys.argv[1], sys.argv[2]
    infos = pick_sessions()
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if ds != "all":
        configure(ds)
    t0 = time.time()
    print(f"\n######## {ds}: phase {phase} ########", flush=True)
    if ds == "all":
        {"extract": lambda: extract(infos), "deconv": lambda: deconv(infos),
         "deconv_bfree": lambda: deconv(infos, "A_bfree"), "classify": lambda: classify(infos),
         "analyse": lambda: analyse(infos)}[phase]()
    else:
        {"gram": lambda: gram(infos), "features": lambda: features(infos), "cluster": cluster,
         "ridge": lambda: ridge(infos), "mlp": mlp, "collect": collect}[phase]()
    print(f"phase {phase} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
