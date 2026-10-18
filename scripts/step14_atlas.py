"""Step 14: atlas of dynamics types + fast recogniser, vs per-neuron models, across
history lengths (Bugeon et al. 2022 and Allen).

Step 13 showed that a label-free S summary of a neuron's long-run dynamics (from the
train part, 51-92 min) helps the forecaster far more than the genetic type, but that
with 3 min of history the summary fails (the models only ever saw full-history
summaries). This step asks, for a held-out neuron with only H minutes of activity
(H = 3, 10, 30 min or the full train part, always from the start of the train part;
for Bugeon proportionally from every block, as step 13):
- K1: is a shared model + summary better than a model fitted to that neuron alone?
- E2: does mapping a short stretch onto an atlas of dynamics types learned from long
  recordings of other mice (reference mapping, as scArches / Azimuth do for cells)
  forecast better than the naive short summary?
- K4: is the summary's gain the neuron's identity or its session's brain state?
- K3 (Bugeon): does a summary from another session of the same cell work?

Same data, folds, splits, windows, seeds and model settings as step 13 (and so as
steps 11 / 12): Allen 46 VISp three_session_A mice, 5 leave-mice-out folds, 10 Hz,
past 30 -> next 10 bins; Bugeon 17 sessions of 4 mice, leave one mouse out, 4.3 Hz,
past 13 -> next 4 frames. The forecast test set is unchanged (last 20%).

Run `python scripts/step14_atlas.py <dataset> <phase>` (dataset bugeon or allen), in
this order (`all` runs them all), then `python scripts/step14_atlas.py both analyse`.
Intermediate files go to data/step14/<dataset>/ (not committed).
- `extract` (allen only): step 11 `extract` unchanged (S3 mirror), plus the running
  speed of every 10 Hz bin, saved while the NWB is on disk (task 3).
- `step13`: step 13's `prep`, `features` and `cluster` phases if their outputs are
  missing (data/step13/<dataset>/). The chosen k (16 in every fold of both datasets
  in step 13) is read from step 13's ridge phase or from results/step13/chosen_k.csv.
- `features`: S summaries from the first 3 / 10 / 30 min and the full train part
  (step 13 `feats_*`, with a 10 min history added). Per fold (training mice only): the
  atlas = step 13's z-score, PCA (95%) and k-means (k = 16, same seed; the labels are
  checked against step 13's). Held-out neurons: naive cluster of each short summary.
  Task 3 summaries: (a) every raw S feature centred within its session (session mean
  over neurons subtracted), (b) the session's mean S vector for every neuron; each is
  z-scored on the training neurons, PCA 95%, k-means (k = 16; for (b) k = min(16,
  number of training sessions)). Task 4: for Bugeon cells (UniqueID) seen in >= 2
  sessions of a mouse, the other session nearest in date. Test-window behavioural
  state (Bugeon frame.states: running vs stationary; Allen: mean running speed over the
  window > 1 cm/s).
- `windows`: recogniser inputs. S features (same function) of short windows of 30 s,
  3, 10 and 30 min: 20 random windows per neuron per length (same positions for all
  neurons of a session), and non-overlapping tiles from the start of the train part
  (each tile marked with the histories H that contain it). Allen: contiguous windows.
  Bugeon: 30 s = 128 contiguous frames inside one block (the Welch segment); 3 / 10 /
  30 min = a piece of every block in proportion to its length (>= 128 frames), as H.
- `recognise` (task 2), per fold, training mice only. Teacher = the neuron's
  full-history atlas cluster, student = a short window. Classifiers: logistic
  regression (C 1; quantile-normal transform, missing values = training median
  + missing indicators) and gradient boosting (NaN native); log10 window length is an
  input. Picked by inner CV over the training mice (grouped by mouse) on log-loss.
  Evidence accumulation: for history H, the log-probabilities are averaged over the
  non-overlapping tiles of one length inside H; the tile length is picked per H on the
  inner-CV log-loss of the accumulated prediction. The same for the genetic type as a
  comparison (the classifier family picked for the atlas, balanced class weights, tile
  length picked on balanced accuracy), and a
  "short -> long" ridge from the window features to the full-history z-scored summary
  (alpha and tile length on inner CV; predictions averaged over tiles).
- `ridge` (tasks 1-4): step 12 ridge from summed per-neuron statistics (alpha on val,
  refit on train + val): b (none), d (true type), the atlas per-cluster models
  (= step 13 r_km16), and per-cluster models for the task 3 summaries. Scored per held-out
  neuron: none, true, atlas cluster from the full history (upper bound), naive short
  cluster, E2 hard (recognised cluster), E2 soft (cluster models' predictions weighted
  by the recogniser probabilities), task 3 (a) / (b), task 4. Per-neuron ridge (step 8
  model a: own past + stimulus): fit on the neuron's first H only, alpha (ALPHAS of step
  3) on the last 10% of H (per block for Bugeon), refit on all of H. Its fit windows have
  targets in the first 90% and val windows targets in the last 10% (past may reach back;
  needed for 3 min of Bugeon).
- `mlp`: step 13 `run_job` (same optimiser, batches, epochs, early stopping, seeds):
  m1 (none), cl (atlas cluster FiLM, k = 16), cont (continuous summary FiLM), and
  cont on the task 3 summaries, 3 seeds each.
- `finetune` (task 1, POCO-style unit embedding): the trained cont network is frozen;
  each held-out neuron gets its own 16-dim FiLM embedding (initialised at the
  population mean, the layer's bias), fitted with Adam (lr 0.05) on its first H (fit
  windows, batches of 32 random windows, at most 500 steps) with early stopping on its
  last 10% of H (up to 256 val windows, checked every 10 steps; each neuron keeps its
  best embedding; stop when no neuron improved for 5 checks).
- `forecast`: MLP test SSE for every variant (as ridge, plus E2 continuous = the
  cont network fed the summary predicted by the short -> long regressor, and the
  fine-tuned embeddings), mean over seeds.
- `analyse` (both datasets, CSV only): results/step14/*.csv.

All test SSE are kept split by the state of the test window (running / stationary /
unknown). Units: Bugeon sessions (and mice), Allen mice (= sessions).

Ideas from other work (no code copied): POCO (unit-embedding fine-tuning of a frozen
pretrained forecaster for a new recording; zero-initialised FiLM via step 9);
NeuPRINT (Mi et al. 2023; a time-invariant per-neuron representation fed to a
forecaster: the continuous summary); NuCLR (Arora et al. 2025; neuron identity from
activity alone); LOLCAT (Schneider et al. 2023; classify a neuron from many short
windows and pool them, inter-event-interval features); single-cell reference mapping
(scArches, Lotfollahi et al. 2022; Azimuth / Seurat v4, Hao et al. 2021: build a
reference atlas once from rich data, then map a new, sparse query onto it with a
light-weight model and transfer its labels with probabilities).
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step11_confound as s11
import step12_bugeon as s12
import step13_dynamics as s13
import step9_mlp as s9
from step3_linear import ALPHAS, BIN, CACHE, ROOT, TRAIN_FRAC, VAL_FRAC, downsample
from step5_shared import Gram
from step8_medium import NPZ

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step14"
OUT = ROOT / "results/step14"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
K = 16                                         # atlas size (step 13's chosen k)
HS = [("3 min", 3.0), ("10 min", 10.0), ("30 min", 30.0), ("full", None)]
HT = {"3 min": "h3", "10 min": "h10", "30 min": "h30", "full": "full"}
WLENS = [("30 s", 0.5), ("3 min", 3.0), ("10 min", 10.0), ("30 min", 30.0)]
N_RAND = 20                                    # random windows per neuron per length
N_RAND_FIT = {"bugeon": 10, "allen": 20}       # of which used to train the recogniser
MIN_BLOCK = s13.MIN_BLOCK                      # 128 frames (Bugeon)
RUN_CM_S = 1.0                                 # Allen running threshold
N_SEEDS = 3
MLP_KINDS = ["cl", "cont", "m1", "cont_ctr", "cont_sess"]
FT_BATCH, FT_MAX_STEPS, FT_EVAL, FT_PATIENCE, FT_LR, FT_NVAL = 32, 500, 10, 5, 0.05, 256
DS = None
P = F = None


def configure(ds):
    global DS, P, F
    DS = ds
    s13.configure(ds)
    P, F = s13.P, s13.F
    (WORK / ds).mkdir(parents=True, exist_ok=True)


def wd():
    return WORK / DS


def save(obj, name):
    pickle.dump(obj, open(wd() / name, "wb"), protocol=4)


def load(name):
    return pickle.load(open(wd() / name, "rb"))


# ---------------------------------------------------------------- extract (Allen)
def allen_extract():
    """Step 11 `extract` (step 10 / 8 extraction, unchanged), plus the running speed
    of every 10 Hz bin of the train and test parts, saved while the NWB is on disk."""
    import step5_shared
    import step10_inferred as s10
    from allensdk.core.brain_observatory_nwb_data_set import BrainObservatoryNwbDataSet
    from step8_medium import pick_sessions
    (WORK / "allen_run").mkdir(parents=True, exist_ok=True)
    orig = step5_shared.load_session

    def load_and_run(info):
        d = orig(info)
        ds = BrainObservatoryNwbDataSet(str(CACHE / f"ophys_experiment_data/{info['id']}.nwb"))
        speed, _ = ds.get_running_speed()
        n_train = int(TRAIN_FRAC * len(speed))
        ok = np.isfinite(speed)
        v = np.where(ok, speed, 0.0)[None]
        c = ok.astype(float)[None]
        out = {}
        for part, sl in [("tr", slice(None, n_train)), ("te", slice(n_train, None))]:
            n = downsample(c[:, sl], BIN)[0]
            with np.errstate(invalid="ignore"):
                out[part] = downsample(v[:, sl], BIN)[0] / n  # NaN where no valid sample
        assert len(out["tr"]) == d["ztr"].shape[1] and len(out["te"]) == d["zte"].shape[1]
        np.savez(WORK / "allen_run" / f"{info['id']}.npz", **out)
        return d

    step5_shared.load_session = load_and_run  # s10.extract wraps this one
    infos = pick_sessions()
    for info in infos:  # a cache file without its running speed is rebuilt
        if not (WORK / "allen_run" / f"{info['id']}.npz").exists():
            (NPZ / f"{info['id']}.npz").unlink(missing_ok=True)
    s10.extract(infos)


# ---------------------------------------------------------------- step 13 inputs
def step13():
    d = s13.wd()
    if not all((d / f).exists() for f in ["features.npz", "clusters.pkl"]) or \
            (DS == "allen" and not all((d / f"{i}_gram.npz").exists() for i in s13.neurons().session.unique())):
        for ph in [s13.prep, s13.features, s13.cluster]:
            ph()
    if not (d / "chosen_k.csv").exists():
        ck = pd.read_csv(ROOT / "results/step13/chosen_k.csv")
        ck[ck.dataset == DS].drop(columns="dataset").to_csv(d / "chosen_k.csv", index=False)
        print("chosen_k.csv taken from results/step13 (step 13 ridge phase not rerun)")
    ck = pd.read_csv(d / "chosen_k.csv")
    assert (ck[ck.method == "km"].k == K).all(), ck


# ---------------------------------------------------------------- sessions, histories
def sess_npz(s):
    return np.load(s12.WORK / f"{s}.npz") if DS == "bugeon" else np.load(NPZ / f"{s}.npz")


def train_arrays(z):
    """Train part (n, T) float64, stimulus (n_stim, T), block of each frame, Hz."""
    if DS == "bugeon":
        return z["Ztr"].astype(np.float64), z["Str"].astype(np.float64), z["blk"], float(z["hz"])
    T = z["ztr"].shape[1]
    return z["ztr"].astype(np.float64), z["Str_all"].astype(np.float64), np.zeros(T, int), float(z["fr"]) / 3


def segments(blk, hz, minutes):
    """History of `minutes` (None = full) as (start, length, block length) per block,
    in train-part frames: step 13's rule (Bugeon: a piece of every block in proportion
    to its length, >= MIN_BLOCK frames; Allen: one block, the first N minutes)."""
    out = []
    for b in np.unique(blk):
        idx = np.where(blk == b)[0]
        if minutes is None:
            m = len(idx)
        elif DS == "bugeon":
            m = min(len(idx), max(MIN_BLOCK, int(round(minutes * 60 * hz * len(idx) / len(blk)))))
        else:
            m = min(len(idx), int(round(minutes * 60 * hz)))
        out.append((int(idx[0]), m, len(idx)))
    return out


def fit_val_starts(segs):
    """Window starts inside a history: per block piece, fit = targets in its first 90%,
    val = targets in its last 10% (the past may reach into the first 90%)."""
    f, v = [], []
    for s0, m, _ in segs:
        nf = m - int(round(VAL_FRAC * m))
        f.append(s0 + np.arange(P, nf - F + 1))
        v.append(s0 + np.arange(max(nf, P), m - F + 1))
    return np.concatenate(f), np.concatenate(v)


# ---------------------------------------------------------------- features (atlas)
def feats_session(s):
    s13.HIST = [(h, m) for h, m in HS if m is not None]
    return (s13.feats_bugeon if DS == "bugeon" else s13.feats_allen)(s)


def test_states(s):
    """State of every test window: 0 running, 1 stationary, 2 unknown."""
    if DS == "bugeon":
        z = sess_npz(s)
        mouse, date = s.split("_", 1)
        sdir = s12.RAW / mouse / date
        st = []
        for bdir in sorted(p for p in sdir.glob("*/*") if p.is_dir()):
            x = np.load(bdir / "frame.states.npy").ravel().astype(int)
            st.append(x[int(TRAIN_FRAC * len(x)):])
        st = np.concatenate(st)
        assert len(st) == z["Zte"].shape[1]
        run = (st[z["te"][:, None] + s12.OFF] == 0).mean(1)
        return np.where(run > 0.5, 0, 1)
    from step3_linear import window_starts
    sp = np.load(WORK / "allen_run" / f"{s}.npz")["te"]
    st = window_starts(len(sp), F)
    with np.errstate(all="ignore"):
        v = np.nanmean(sp[st[:, None] + np.arange(-P, F)], axis=1)
    return np.where(np.isnan(v), 2, np.where(v > RUN_CM_S, 0, 1))


def features():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    neu = s13.neurons()
    sess = list(neu.session.unique())
    with Pool(N_WORKERS) as pool:
        res = dict(pool.map(feats_session, sess))
        states = dict(zip(sess, pool.map(test_states, sess)))
    names = list(res[sess[0]]["full"])
    Fm = {}
    for h, _ in HS:
        A = np.vstack([np.column_stack([res[s][h][c] for c in names]) for s in sess])
        A[~np.isfinite(A)] = np.nan
        Fm[h] = A
    old_names, old = s13.load_features()
    assert old_names == names
    for h in ["full", "3 min", "30 min"]:  # same as step 13's features
        assert np.allclose(np.nan_to_num(old[h], nan=7.7), np.nan_to_num(Fm[h], nan=7.7), rtol=1e-9, atol=1e-12), h
    frames = pd.DataFrame([dict(session=s, hz=res[s]["hz"], **{f"{h} frames": res[s][f"{h} frames" if h != "full" else
                                                                                        "full train part frames"]
                                                               for h, _ in HS}) for s in sess])
    # Task 3 summaries: centred within session / session mean
    A = Fm["full"]
    smean = pd.DataFrame(A).groupby(neu.session.to_numpy(), sort=False).transform("mean").to_numpy()
    summ = {"ctr": A - smean, "sess": smean}
    # Task 4 (Bugeon): the same cell in another session of the mouse, nearest in date
    cross = np.full(len(neu), -1)
    if DS == "bugeon":
        d = pd.to_datetime(neu.date)
        for (m, u), g in neu[neu.uid.notna()].groupby(["mouse", "uid"]):
            for i in g.index:
                oth = g.index[g.session != neu.session[i]]
                if len(oth):
                    dd = np.abs((d[oth] - d[i]).dt.days.to_numpy())
                    cross[i] = oth[np.lexsort((d[oth].to_numpy(), dd))[0]]
    cl13 = s13.load_clusters()
    atlas = {}
    for k in sorted(neu.fold.unique()):
        tr = (neu.fold != k).to_numpy()
        mu, sd = A[tr].mean(0), A[tr].std(0)
        sd[sd == 0] = 1
        Z = {h: np.nan_to_num((Fm[h] - mu) / sd, nan=0.0) for h, _ in HS}
        pca = PCA(n_components=0.95, svd_solver="full").fit(Z["full"][tr])
        km = KMeans(n_clusters=K, n_init=10, random_state=k).fit(pca.transform(Z["full"][tr]))
        lab = {h: km.predict(pca.transform(Z[h])) for h, _ in HS}
        assert (lab["full"] == cl13[k]["lab"]["km", K]).all(), "atlas differs from step 13"
        for h in ["3 min", "30 min"]:
            assert (lab[h] == cl13[k]["lab_hist"]["km", K, h]).all()
            assert np.allclose(Z[h], cl13[k]["Z"][h], atol=1e-5)
        a = dict(mu=mu, sd=sd, pca=pca, km=km, lab=lab, Z={h: Z[h].astype(np.float32) for h in Z},
                 n_pc=pca.n_components_)
        for name, M in summ.items():
            m_, s_ = M[tr].mean(0), M[tr].std(0)
            s_[s_ == 0] = 1
            Zs = (M - m_) / s_
            p = PCA(n_components=0.95, svd_solver="full").fit(Zs[tr])
            kk = min(K, len(np.unique(Zs[tr], axis=0)))
            c = KMeans(n_clusters=kk, n_init=10, random_state=k).fit(p.transform(Zs[tr]))
            a[f"Z_{name}"], a[f"lab_{name}"], a[f"k_{name}"] = Zs.astype(np.float32), c.predict(p.transform(Zs)), kk
            a[f"npc_{name}"] = p.n_components_
        atlas[k] = a
        print(f"fold {k}: atlas {pca.n_components_} PCs; naive short-history clusters equal to full: " +
              ", ".join(f"{h} {(lab[h][~tr] == lab['full'][~tr]).mean():.2f}" for h, _ in HS[:3]) +
              f"; centred {a['npc_ctr']} PCs, session-mean {a['npc_sess']} PCs, k {a['k_sess']}", flush=True)
    save(dict(names=names, F=Fm, atlas=atlas, cross=cross, states=states), "atlas.pkl")
    frames.to_csv(wd() / "history_frames.csv", index=False)
    st = np.concatenate([states[s] for s in sess])
    print(f"{len(names)} S features x {len(neu)} neurons; test windows running / stationary / unknown: "
          f"{np.bincount(st, minlength=3)}; task 4 cells with another session: {(cross >= 0).sum()}")


def load_atlas():
    return load("atlas.pkl")


# ---------------------------------------------------------------- windows (recogniser inputs)
def window_frames(blk, hz, minutes, rng=None, j=None):
    """Frames of one window of `minutes`: random (rng) or tile j from the start.
    Returns the frames and, for tiles, the per-block end offsets (to test H membership)."""
    blocks = [np.where(blk == b)[0] for b in np.unique(blk)]
    if DS == "bugeon" and minutes < 1:  # 30 s: 128 contiguous frames inside one block
        if rng is not None:
            lens = np.array([len(b) for b in blocks], float)
            b = rng.choice(len(blocks), p=lens / lens.sum())
            a = rng.integers(len(blocks[b]) - MIN_BLOCK + 1)
            return blocks[b][a:a + MIN_BLOCK]
        return None
    if DS == "bugeon":
        ls = [min(len(b), max(MIN_BLOCK, int(round(minutes * 60 * hz * len(b) / len(blk))))) for b in blocks]
    else:
        ls = [min(len(blocks[0]), int(round(minutes * 60 * hz)))]
    if rng is not None:
        return np.concatenate([b[(a := rng.integers(len(b) - l + 1)):a + l] for b, l in zip(blocks, ls)])
    return [(b, l) for b, l in zip(blocks, ls)]


def windows_session(args):
    s, si, row0 = args
    z = sess_npz(s)
    Z, S, blk, hz = train_arrays(z)
    fn = (lambda fr: s12.s_features(Z[:, fr], S[:, fr], blk[fr], hz)) if DS == "bugeon" else \
        (lambda fr: s11.s_features(Z[:, fr], S[:, fr], hz))
    rng = np.random.default_rng(14000 + si)
    n = len(Z)
    H_segs = [segments(blk, hz, m) for _, m in HS]
    rand, tiles = [], []
    with np.errstate(all="ignore"):
        for li, (_, mins) in enumerate(WLENS):
            for r in range(N_RAND):
                fr = window_frames(blk, hz, mins, rng=rng)
                f = fn(fr)
                rand.append((li, r, len(fr), np.column_stack(list(f.values()))))
            # tiles from the start of the train part
            if DS == "bugeon" and mins < 1:
                for b, (s0, _, lb) in enumerate(H_segs[-1]):
                    for j in range(lb // MIN_BLOCK):
                        fr = s0 + j * MIN_BLOCK + np.arange(MIN_BLOCK)
                        inH = [(j + 1) * MIN_BLOCK <= segs[b][1] for segs in H_segs]
                        tiles.append((li, len(fr), inH, np.column_stack(list(fn(fr).values()))))
            else:
                pieces = window_frames(blk, hz, mins)
                nt = min(len(b) // l for b, l in pieces)
                for j in range(nt):
                    fr = np.concatenate([b[j * l:(j + 1) * l] for b, l in pieces])
                    inH = [all((j + 1) * l <= segs[b][1] for b, (_, l) in enumerate(pieces)) for segs in H_segs]
                    tiles.append((li, len(fr), inH, np.column_stack(list(fn(fr).values()))))
    rows = row0 + np.arange(n)
    out = dict(
        rand_feat=np.vstack([x[3] for x in rand]).astype(np.float32),
        rand_row=np.concatenate([rows for _ in rand]), rand_len=np.repeat([x[0] for x in rand], n),
        rand_rep=np.repeat([x[1] for x in rand], n), rand_min=np.repeat([x[2] / hz / 60 for x in rand], n),
        tile_feat=np.vstack([x[3] for x in tiles]).astype(np.float32),
        tile_row=np.concatenate([rows for _ in tiles]), tile_len=np.repeat([x[0] for x in tiles], n),
        tile_min=np.repeat([x[1] / hz / 60 for x in tiles], n),
        tile_inH=np.repeat(np.array([x[2] for x in tiles]), n, axis=0))
    print(f"{s}: {len(rand)} random windows, {len(tiles)} tiles "
          f"({', '.join(str(sum(t[0] == li for t in tiles)) for li in range(len(WLENS)))} per length)", flush=True)
    return out


def windows():
    neu = s13.neurons()
    sess = list(neu.session.unique())
    row0 = neu.groupby("session", sort=False).size().cumsum().shift(fill_value=0)
    with Pool(N_WORKERS) as pool:
        res = pool.map(windows_session, [(s, i, int(row0[s])) for i, s in enumerate(sess)], chunksize=1)
    out = {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    np.savez(wd() / "windows.npz", **out)
    print(f"{len(out['rand_row'])} random-window rows, {len(out['tile_row'])} tile rows; NaN fraction "
          f"random {np.isnan(out['rand_feat']).mean():.3f}, tiles {np.isnan(out['tile_feat']).mean():.3f}")


# ---------------------------------------------------------------- recogniser (task 2)
class Design:
    """Inputs of the recogniser: (linear) training median for missing values plus
    missing indicators, then a quantile-normal transform; (gboost) raw, NaN native.
    log10 window length (minutes) is appended in both."""

    def __init__(self, X):
        self.med = np.nanmedian(X, axis=0)
        self.med[~np.isfinite(self.med)] = 0
        self.miss = np.where(np.isnan(X).any(0))[0]

    def lin(self, X, mins):
        Xf = np.where(np.isnan(X), self.med, X)
        return np.hstack([Xf, np.isnan(X[:, self.miss]), np.log10(mins)[:, None]]).astype(np.float32)

    @staticmethod
    def gb(X, mins):
        return np.hstack([X, np.log10(mins)[:, None]]).astype(np.float32)


def make_clf(cand, balanced, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer
    cw = "balanced" if balanced else None
    if cand[0] == "logreg":
        return make_pipeline(QuantileTransformer(n_quantiles=200, output_distribution="normal", subsample=200000,
                                                 random_state=seed),
                             LogisticRegression(C=cand[1], class_weight=cw, max_iter=1000))
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_leaf_nodes=31, l2_regularization=1.0,
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=10,
                                          class_weight=cw, random_state=seed)


CANDS = [("logreg", 1.0), ("gboost", None)]
REG_ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def full_proba(m, X, n_cls):
    """predict_proba with a column for every class (a class may be absent in a fit)."""
    p = np.zeros((len(X), n_cls))
    p[:, m.classes_] = m.predict_proba(X)
    return p


def accumulate(rows, vals, n_rows):
    """Mean of vals (M, C) per row id -> (n_rows, C), NaN where a row has no value."""
    acc = np.zeros((n_rows, vals.shape[1]))
    cnt = np.bincount(rows, minlength=n_rows).astype(float)
    for c in range(vals.shape[1]):
        acc[:, c] = np.bincount(rows, vals[:, c], minlength=n_rows)
    with np.errstate(invalid="ignore"):
        return acc / cnt[:, None], cnt


def logloss(P, y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-6, None))))


def softmax_mean_log(LP):
    LP = LP - LP.max(1, keepdims=True)
    E = np.exp(LP)
    return E / E.sum(1, keepdims=True)


def bacc(y, p):
    from sklearn.metrics import balanced_accuracy_score
    return balanced_accuracy_score(y, p)


def tile_accum(W, sel_rows, P_tiles, n_all):
    """Accumulated probabilities for every (H, tile length): softmax of the mean
    log-probability over the tiles of that length inside H. P_tiles: probabilities of
    the tiles W rows sel_rows. Returns {(h, li): (n_all, C) with NaN rows if none}."""
    out = {}
    LP = np.log(np.clip(P_tiles, 1e-6, None))
    for hi, (h, hm) in enumerate(HS):
        for li, (_, lm) in enumerate(WLENS):
            if hm is not None and lm > hm:
                continue
            m = W["tile_inH"][sel_rows, hi] & (W["tile_len"][sel_rows] == li)
            if not m.any():
                continue
            mlp, cnt = accumulate(W["tile_row"][sel_rows][m], LP[m], n_all)
            P = np.full_like(mlp, np.nan)
            ok = cnt > 0
            P[ok] = softmax_mean_log(mlp[ok])
            out[h, li] = P, cnt
    return out


def recognise_fold(k, W, neu, atlas):
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    held = (neu.fold == k).to_numpy()
    mouse = neu.mouse.to_numpy()
    n_all = len(neu)
    A = atlas[k]
    targets = {"cluster": (A["lab"]["full"], K, False), "type": (neu.t.to_numpy(), len(s13.TYPES), True)}
    rr = W["rand_row"]
    fit_m = (~held[rr]) & (W["rand_rep"] < N_RAND_FIT[DS])
    ev_m = held[rr]
    tr_rows = np.where(fit_m)[0]
    design = Design(W["rand_feat"][tr_rows])
    Xr = {"lin": design.lin(W["rand_feat"], W["rand_min"]), "gb": Design.gb(W["rand_feat"], W["rand_min"])}
    tile_tr = np.where(~held[W["tile_row"]])[0]
    tile_he = np.where(held[W["tile_row"]])[0]
    Xt = {"lin": design.lin(W["tile_feat"], W["tile_min"]), "gb": Design.gb(W["tile_feat"], W["tile_min"])}
    n_inner = 3 if DS == "bugeon" else 4
    inner = list(GroupKFold(n_inner).split(tr_rows[:, None], groups=mouse[rr[tr_rows]]))
    inner_mice = [np.unique(mouse[rr[tr_rows[iva]]]) for _, iva in inner]
    res = dict(rows=[], P={}, Zpred={}, choice=[])
    for tname, (y_all, C, bal) in targets.items():
        y = y_all[rr]
        cand_res = {}
        # the type recogniser (a comparison only) reuses the classifier family picked for the atlas
        for cand in (CANDS if tname == "cluster" else [best_cluster]):
            xk = "lin" if cand[0] == "logreg" else "gb"
            oof = np.zeros((len(tr_rows), C))
            models = []
            for itr, iva in inner:
                m = make_clf(cand, bal, k).fit(Xr[xk][tr_rows[itr]], y[tr_rows[itr]])
                oof[iva] = full_proba(m, Xr[xk][tr_rows[iva]], C)
                models.append(m)
            yt = y[tr_rows]
            cand_res[cand] = dict(models=models, logloss=logloss(oof, yt), bacc=bacc(yt, oof.argmax(1)),
                                  acc=float((oof.argmax(1) == yt).mean()))
            res["choice"].append(dict(fold=k, target=tname, stage="inner CV, random windows", cand=f"{cand[0]} {cand[1]}",
                                      **{m_: cand_res[cand][m_] for m_ in ["logloss", "bacc", "acc"]}))
        key = "logloss" if tname == "cluster" else "bacc"
        best = (min if key == "logloss" else max)(cand_res, key=lambda c: cand_res[c][key])
        if tname == "cluster":
            best_cluster = best
        xk = "lin" if best[0] == "logreg" else "gb"
        # Out-of-fold tile predictions of the training neurons -> tile length per H
        Ptr = np.zeros((len(tile_tr), C))
        trm = mouse[W["tile_row"][tile_tr]]
        for (itr, iva), mice, m in zip(inner, inner_mice, cand_res[best]["models"]):
            sel = np.isin(trm, mice)
            Ptr[sel] = full_proba(m, Xt[xk][tile_tr[sel]], C)
        acc_in = tile_accum(W, tile_tr, Ptr, n_all)
        lsel = {}
        for h, _ in HS:
            sc = {}
            for (hh, li), (Pa, _) in acc_in.items():
                if hh == h:
                    ok = ~np.isnan(Pa[:, 0])
                    sc[li] = logloss(Pa[ok], y_all[ok]) if key == "logloss" else -bacc(y_all[ok], Pa[ok].argmax(1))
                    res["choice"].append(dict(fold=k, target=tname, stage=f"inner CV, accumulated, H {h}",
                                              cand=f"tiles {WLENS[li][0]}", logloss=logloss(Pa[ok], y_all[ok]),
                                              bacc=bacc(y_all[ok], Pa[ok].argmax(1)),
                                              acc=float((Pa[ok].argmax(1) == y_all[ok]).mean())))
            lsel[h] = min(sc, key=sc.get)
        # Refit on all training rows; held-out mice
        m = make_clf(best, bal, k).fit(Xr[xk][tr_rows], y[tr_rows])
        ev = np.where(ev_m)[0]
        Pev = full_proba(m, Xr[xk][ev], C)
        for li, (lname, _) in enumerate(WLENS):
            s_ = W["rand_len"][ev] == li
            yy, pp = y[ev][s_], Pev[s_]
            res["rows"].append(dict(fold=k, target=tname, kind="single window", window=lname,
                                    minutes=float(np.mean(W["rand_min"][ev][s_])), clf=f"{best[0]} {best[1]}",
                                    n=int(s_.sum()), acc=float((pp.argmax(1) == yy).mean()), bacc=bacc(yy, pp.argmax(1)),
                                    logloss=logloss(pp, yy), chance_acc=float(np.bincount(y_all[~held]).max() / (~held).sum()),
                                    chance_bacc=1 / C))
        Phe = full_proba(m, Xt[xk][tile_he], C)
        acc_he = tile_accum(W, tile_he, Phe, n_all)
        for (h, li), (Pa, cnt) in acc_he.items():
            ok = held & ~np.isnan(Pa[:, 0])
            yy, pp = y_all[ok], Pa[ok]
            res["rows"].append(dict(fold=k, target=tname, kind="accumulated", history=h, window=WLENS[li][0],
                                    selected=lsel[h] == li, clf=f"{best[0]} {best[1]}", n=int(ok.sum()),
                                    n_tiles=float(cnt[ok].mean()), acc=float((pp.argmax(1) == yy).mean()), bacc=bacc(yy, pp.argmax(1)), logloss=logloss(pp, yy)))
            if lsel[h] == li:
                res["P"][tname, h] = Pa[held]
        res["choice"].append(dict(fold=k, target=tname, stage="selected", cand=f"{best[0]} {best[1]}",
                                  **{f"tiles {h}": WLENS[lsel[h]][0] for h, _ in HS}))
    # Short -> long regressor: window features -> full-history z-scored summary
    Zt = A["Z"]["full"].astype(np.float64)
    Y = Zt[rr]
    yt = Y[tr_rows]
    mse = {}
    oofs = {}
    for a in REG_ALPHAS:
        oof = np.zeros_like(yt)
        for itr, iva in inner:
            oof[iva] = Ridge(alpha=a).fit(Xr["lin"][tr_rows[itr]], yt[itr]).predict(Xr["lin"][tr_rows[iva]])
        mse[a] = float(((oof - yt) ** 2).mean())
    a_best = min(mse, key=mse.get)
    for a in REG_ALPHAS:
        res["choice"].append(dict(fold=k, target="summary", stage="inner CV, random windows", cand=f"ridge {a:g}",
                                  mse=mse[a], r2=1 - mse[a] / yt.var(0).mean(), selected=a == a_best))
    Ptr = np.zeros((len(tile_tr), Zt.shape[1]))
    trm = mouse[W["tile_row"][tile_tr]]
    for (itr, iva), mice in zip(inner, inner_mice):
        sel = np.isin(trm, mice)
        Ptr[sel] = Ridge(alpha=a_best).fit(Xr["lin"][tr_rows[itr]], yt[itr]).predict(Xt["lin"][tile_tr[sel]])
    reg = Ridge(alpha=a_best).fit(Xr["lin"][tr_rows], yt)
    Phe = reg.predict(Xt["lin"][tile_he])
    ev = np.where(ev_m)[0]
    Pev = reg.predict(Xr["lin"][ev])
    for li, (lname, _) in enumerate(WLENS):
        s_ = W["rand_len"][ev] == li
        e = ((Pev[s_] - Y[ev][s_]) ** 2).mean()
        res["rows"].append(dict(fold=k, target="summary", kind="single window", window=lname, n=int(s_.sum()),
                                mse=e, r2=1 - e / Zt[held].var(0).mean()))
    for hi, (h, hm) in enumerate(HS):
        best_l, best_e = None, np.inf
        for li, (lname, lm) in enumerate(WLENS):
            if hm is not None and lm > hm:
                continue
            for part, rows_, PP in [("inner", tile_tr, Ptr), ("held", tile_he, Phe)]:
                m_ = W["tile_inH"][rows_, hi] & (W["tile_len"][rows_] == li)
                Za, cnt = accumulate(W["tile_row"][rows_][m_], PP[m_], n_all)
                ok = cnt > 0
                e = float(((Za[ok] - Zt[ok]) ** 2).mean())
                if part == "inner":
                    res["choice"].append(dict(fold=k, target="summary", stage=f"inner CV, accumulated, H {h}",
                                              cand=f"tiles {lname}", mse=e, r2=1 - e / Zt[ok].var(0).mean()))
                    if e < best_e:
                        best_l, best_e = li, e
                else:
                    res["rows"].append(dict(fold=k, target="summary", kind="accumulated", history=h, window=lname,
                                            n=int(ok.sum()), mse=e, r2=1 - e / Zt[ok].var(0).mean(), Za=Za))
        for r in res["rows"]:
            if r.get("target") == "summary" and r.get("kind") == "accumulated" and r["history"] == h:
                r["selected"] = r["window"] == WLENS[best_l][0]
                if r["selected"]:
                    res["Zpred"][h] = r["Za"][held]
                del r["Za"]
    print(f"fold {k}: recogniser " + "; ".join(f"{c['target']} {c['cand']} tiles " +
                                                 ", ".join(f"{h} {c[f'tiles {h}']}" for h, _ in HS)
                                                 for c in res["choice"] if c["stage"] == "selected") +
          f"; summary ridge alpha {a_best:g} ({time.time() - t0:.0f} s)", flush=True)
    return k, res


def recognise():
    neu = s13.neurons()
    W = dict(np.load(wd() / "windows.npz"))
    A = load_atlas()
    out = {}
    path = wd() / "recognise.pkl"
    if path.exists():
        out = load("recognise.pkl")
    for k in sorted(neu.fold.unique()):
        if k in out:
            continue
        out[k] = recognise_fold(k, W, neu, A["atlas"])[1]
        save(out, "recognise.pkl")


# ---------------------------------------------------------------- ridge (tasks 1-4)
def single_gram(g, i):
    """Gram of the design [own past, stimulus] of neuron i from per-neuron statistics."""
    xx = np.block([[g["xx"][i], g["xS"][i]], [g["xS"][i].T, g["SS"]]])
    return Gram(g["n"], np.concatenate([g["sx"][i], g["sS"]]), g["sy"][i], xx,
                np.vstack([g["xy"][i], g["Sy"][i]])), g["yy"][i]


def per_neuron_ridge(Z, S, segs):
    """Step 8 model a on a history: alpha by val MSE, refit on fit + val windows."""
    fs, vs = fit_val_starts(segs)
    idx = np.arange(len(Z))
    gf, gv = s12.neuron_stats(Z, S, fs, idx), s12.neuron_stats(Z, S, vs, idx)
    W = np.empty((len(Z), gf["xx"].shape[1] + gf["SS"].shape[0], F))
    b = np.empty((len(Z), F))
    al = np.empty(len(Z))
    for i in idx:
        g1, _ = single_gram(gf, i)
        g2, yy2 = single_gram(gv, i)
        coef = g1.solver()
        mse = [s12.val_mse(*coef(a), g2, yy2) for a in ALPHAS]
        al[i] = ALPHAS[int(np.argmin(mse))]
        W[i], b[i] = (g1 + g2).solver()(al[i])
    return W, b, al, len(fs), len(vs)


def split_sse(E, st):
    """E (n_windows, n_neurons) squared errors -> (n_neurons, 3) by window state."""
    return np.stack([E[st == c].sum(0) for c in range(3)], axis=1)


def ridge_session(args):
    """All ridge variants of one held-out session (models fitted on the other mice)."""
    s, rows, models, Pc, lab, cross_lab = args
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        X, Y, S = s13.test_data(s)
        st = STATES[s]
        n = len(rows)
        err = lambda Pr: split_sse(((Pr - Y) ** 2).sum(2), st)
        out = {"copy": err(np.repeat(X[:, :, -1:], F, axis=2)), "r_none": err(s12.predict(*models["b"][:2], X, S))}
        tr = TYPES_ARR[rows]
        E = np.zeros((n, 3))
        for c in np.unique(tr):
            E[tr == c] = err(s12.predict(*models["d"][c][:2], X, S))[tr == c]
        out["r_true"] = E
        # atlas cluster models: every neuron forecast by every cluster model once
        sse_c = np.zeros((K, n, 3))
        mix = {h: 0.0 for h, _ in HS}
        for c in range(K):
            mod = models["km"].get(c)
            Pr = s12.predict(mod["W"], mod["b"], X, S) if mod else s12.predict(*models["b"][:2], X, S)
            sse_c[c] = err(Pr)
            for h, _ in HS:
                mix[h] = mix[h] + Pc[h][:, c][None, :, None] * Pr
        pick = lambda L: sse_c[L, np.arange(n)]
        out["r_full"] = pick(lab["full"])
        for h, _ in HS:
            if h != "full":
                out[f"r_naive_{HT[h]}"] = pick(lab[h])
            out[f"r_hard_{HT[h]}"] = pick(Pc[h].argmax(1))
            out[f"r_soft_{HT[h]}"] = err(mix[h])
        if (cross_lab >= 0).any():
            v = pick(np.maximum(cross_lab, 0))
            v[cross_lab < 0] = np.nan
            out["r_cross"] = v
        for name in ["ctr", "sess"]:
            L = lab[name]
            v = np.zeros((n, 3))
            for c in np.unique(L):
                mod = models[name].get(c)
                W_, b_ = (mod["W"], mod["b"]) if mod else models["b"][:2]
                v[L == c] = err(s12.predict(W_, b_, X, S))[L == c]
            out[f"r_{name}"] = v
        # per-neuron ridge on the first H of the train part
        z = sess_npz(s)
        Z, St, blk, hz = train_arrays(z)
        info = []
        for h, m in HS:
            W, b, al, nf, nv = per_neuron_ridge(Z, St, segments(blk, hz, m))
            Pr = np.einsum("wnp,npf->wnf", X, W[:, :P]) + np.einsum("ws,nsf->wnf", S, W[:, P:]) + b[None]
            out[f"r_pn_{HT[h]}"] = err(Pr)
            info.append(dict(session=s, history=h, fit_windows=nf, val_windows=nv,
                             **{f"alpha_q{q}": np.quantile(al, q / 100) for q in [10, 50, 90]}))
    print(f"{s}: {n} neurons", flush=True)
    return s, out, info


STATES = TYPES_ARR = None


def ridge():
    global STATES, TYPES_ARR
    neu = s13.neurons()
    A = load_atlas()
    R = load("recognise.pkl")
    STATES, TYPES_ARR = A["states"], neu.t.to_numpy()
    G = s13.load_grams(neu)
    si = neu.si.to_numpy()
    neu["rows"] = G["f_n"][si] + G["v_n"][si]
    res, infos = {}, []
    for k in sorted(neu.fold.unique()):
        t0 = time.time()
        held = (neu.fold == k).to_numpy()
        tr = ~held
        a = A["atlas"][k]
        models = {"b": s12.fit_weighted(G, si, s12.balanced_weights(neu, tr)),
                  "d": {c: s12.fit_weighted(G, si, (tr & (TYPES_ARR == c)).astype(float))
                        for c in range(len(s13.TYPES))},
                  "km": s13.fit_clusters(G, si, neu, a["lab"]["full"], tr),
                  "ctr": s13.fit_clusters(G, si, neu, a["lab_ctr"], tr),
                  "sess": s13.fit_clusters(G, si, neu, a["lab_sess"], tr)}
        print(f"fold {k}: models fitted ({time.time() - t0:.0f} s)", flush=True)
        hrows = np.where(held)[0]
        Pfull = {h: np.zeros((len(neu), K)) for h, _ in HS}
        for h, _ in HS:
            Pfull[h][hrows] = R[k]["P"]["cluster", h]
        cross = A["cross"]
        jobs = []
        for s in neu[held].session.unique():
            rows = np.where((neu.session == s).to_numpy())[0]
            cl = np.where(cross[rows] >= 0, a["lab"]["full"][np.maximum(cross[rows], 0)], -1)
            jobs.append((s, rows, models, {h: Pfull[h][rows] for h, _ in HS},
                         {**{h: a["lab"][h][rows] for h, _ in HS}, "ctr": a["lab_ctr"][rows],
                          "sess": a["lab_sess"][rows]}, cl))
        with Pool(N_WORKERS) as pool:
            for s, out, info in pool.imap_unordered(ridge_session, jobs):
                res[s] = out
                infos += info
        print(f"fold {k} done ({time.time() - t0:.0f} s)", flush=True)
    save(res, "ridge_sse.pkl")
    pd.DataFrame(infos).to_csv(wd() / "pn_ridge_info.csv", index=False, float_format="%.4g")


# ---------------------------------------------------------------- MLP (training)
ATLAS = None


def cond_inputs(kind, fold, seed):
    if kind == "cont_ctr":
        return None, ATLAS[fold]["Z_ctr"], 0
    if kind == "cont_sess":
        return None, ATLAS[fold]["Z_sess"], 0
    return COND_ORIG(kind, fold, seed)


COND_ORIG = s13.cond_inputs


def load_mlp():
    global ATLAS
    s13.load_mlp_data()
    for k, C in s13.COND.items():
        assert C["k_sel"] == K
    ATLAS = load_atlas()["atlas"]
    s13.cond_inputs = cond_inputs


def run_job(job):
    r = s13.run_job(job)
    r.pop("hist", None)
    return r


def mlp():
    load_mlp()
    path = wd() / "mlp.pkl"
    done = load("mlp.pkl") if path.exists() else []
    have = {(r["job"]["fold"], r["job"]["kind"], r["job"]["seed"]) for r in done}
    jobs = [dict(fold=k, kind=kind, seed=seed) for kind in MLP_KINDS for k in sorted(s13.NEU.fold.unique())
            for seed in range(N_SEEDS)]
    jobs = [j for j in jobs if (j["fold"], j["kind"], j["seed"]) not in have]
    print(f"{len(jobs)} MLP fits to run ({len(have)} done) in {N_WORKERS} workers", flush=True)
    results = list(done)
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}] fold {j['fold']} {j['kind']} seed {j['seed']}: {r['epochs']} epochs, "
                  f"best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)
            save(results, "mlp.pkl")


def get_model(r):
    kind = r["job"]["kind"]
    if kind == "m1":
        m = s13.make_model(0, None)
    elif kind == "cl":
        m = s13.make_model(K, None)
    else:
        m = s13.make_model(0, np.zeros((1, len(ATLAS[0]["mu"]))))
    m.load_state_dict(r["state"])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def train_part(si):
    if DS == "bugeon":
        return s12.SESS[si]["Ztr"], s12.SESS[si]["Str"]
    return s9.SESS[si]["ztr"], s9.SESS[si]["Str"]


def stim_rows(Str, t):
    """Stimulus inputs of windows starting at t, in gather's layout."""
    return Str[:, t[:, None] + np.arange(-P, F)].transpose(1, 0, 2).reshape(len(t), -1)


def film_tail(model, h1, g):
    g1, b1, g2, b2 = g
    h2 = torch.relu(model.l2(h1 * (1 + g1) + b1)) * (1 + g2) + b2
    return model.out(h2)


# ---------------------------------------------------------------- finetune (task 1)
def finetune_job(job):
    """Per-neuron FiLM embedding of the frozen cont network on the neuron's first H."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, seed = job["fold"], job["seed"]
    model = get_model(job["r"])
    Wx, Ws, b1 = model.l1.weight[:, :P], model.l1.weight[:, P:], model.l1.bias
    e0 = model.cont.bias.detach()
    out, info = {}, []
    held = sorted(s13.NEU[s13.NEU.fold == fold].si.unique())
    for si in held:
        Z, Str = train_part(si)
        s = s13.NEU[s13.NEU.si == si].session.iloc[0]
        z = sess_npz(s)
        _, _, blk, hz = train_arrays(z)
        n = len(Z)
        for hi, (h, m) in enumerate(HS):
            rng = np.random.default_rng(100000 * fold + 1000 * seed + 10 * si + hi)
            fs, vs = fit_val_starts(segments(blk, hz, m))
            vsub = np.sort(rng.choice(vs, FT_NVAL, replace=False)) if len(vs) > FT_NVAL else vs
            with torch.no_grad():
                Wv = torch.from_numpy(Z[:, vsub[:, None] + np.arange(-P, F)])
                hv = torch.relu(Wv[..., :P] @ Wx.T + (torch.from_numpy(stim_rows(Str, vsub)) @ Ws.T + b1)[None])
                yv = Wv[..., P:]
            E = torch.nn.Parameter(e0.repeat(n, 1).clone())
            opt = torch.optim.Adam([E], lr=FT_LR)

            def val():
                with torch.no_grad():
                    g = [x[:, None, :] for x in model.gb(E).chunk(4, dim=1)]
                    return ((film_tail(model, hv, g) - yv) ** 2).mean((1, 2)).numpy()

            best = val()
            v0 = best.copy()
            bestE = E.detach().clone()
            bad, steps = 0, 0
            for step in range(1, FT_MAX_STEPS + 1):
                t = rng.choice(fs, FT_BATCH)
                Wt = torch.from_numpy(Z[:, t[:, None] + np.arange(-P, F)])
                with torch.no_grad():
                    h1 = torch.relu(Wt[..., :P] @ Wx.T + (torch.from_numpy(stim_rows(Str, t)) @ Ws.T + b1)[None])
                g = [x[:, None, :] for x in model.gb(E).chunk(4, dim=1)]
                loss = ((film_tail(model, h1, g) - Wt[..., P:]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                steps = step
                if step % FT_EVAL == 0:
                    v = val()
                    imp = v < best - 1e-6
                    best[imp] = v[imp]
                    bestE[torch.from_numpy(imp)] = E.detach()[torch.from_numpy(imp)]
                    bad = 0 if imp.any() else bad + 1
                    if bad >= FT_PATIENCE:
                        break
            out[si, h] = bestE.numpy().copy()
            info.append(dict(fold=fold, seed=seed, si=si, history=h, n=n, fit_windows=len(fs), val_windows=len(vsub),
                             steps=steps, val_init=float(v0.mean()), val_best=float(best.mean()),
                             frac_moved=float((best < v0).mean()),
                             emb_shift=float((bestE - e0).norm(dim=1).mean())))
    print(f"fold {fold} seed {seed}: fine-tuned {len(held)} sessions ({time.time() - t0:.0f} s)", flush=True)
    return (fold, seed), out, info


def finetune():
    load_mlp()
    res = load("mlp.pkl")
    jobs = [dict(fold=r["job"]["fold"], seed=r["job"]["seed"], r=r) for r in res if r["job"]["kind"] == "cont"]
    path = wd() / "finetune.pkl"
    done = load("finetune.pkl") if path.exists() else {"E": {}, "info": []}
    jobs = [j for j in jobs if (j["fold"], j["seed"]) not in {k[:2] for k in done["E"]}]
    print(f"{len(jobs)} fine-tune jobs", flush=True)
    with Pool(N_WORKERS) as pool:
        for key, out, info in pool.imap_unordered(finetune_job, jobs):
            for (si, h), E in out.items():
                done["E"][key + (si, h)] = E
            done["info"] += info
            save(done, "finetune.pkl")


# ---------------------------------------------------------------- forecast (MLP)
def forecast_multi(model, si, E_dict, film=True):
    """Per-neuron test SSE (n, 3 states) of one held-out session for several FiLM
    embeddings (name -> (n, 16)); the first layer is computed once."""
    X, Y, S = s13.held_windows(si)
    st = STATES[s13.NEU[s13.NEU.si == si].session.iloc[0]]
    nw, n = X.shape[:2]
    Wx, Ws, b1 = model.l1.weight[:, :P], model.l1.weight[:, P:], model.l1.bias
    out = {name: np.zeros((n, 3)) for name in E_dict}
    with torch.no_grad():
        gs = {name: [x[None] for x in model.gb(torch.from_numpy(np.asarray(E, np.float32))).chunk(4, dim=1)]
              for name, E in E_dict.items()} if film else {}
        for a in range(0, nw, 32):
            Xc, Yc = torch.from_numpy(X[a:a + 32]), torch.from_numpy(Y[a:a + 32])
            h1 = torch.relu(Xc @ Wx.T + (torch.from_numpy(S[a:a + 32]) @ Ws.T + b1)[:, None, :])
            for name in E_dict:
                pr = film_tail(model, h1, gs[name]) if film else model.out(torch.relu(model.l2(h1)))
                e = ((pr - Yc) ** 2).sum(2).numpy().astype(np.float64)
                for c in range(3):
                    out[name][:, c] += e[st[a:a + 32] == c].sum(0)
    return out


def forecast_job(job):
    torch.set_num_threads(1)
    t0 = time.time()
    fold, seed = job["fold"], job["seed"]
    A, R, FT = job["atlas"], job["rec"], job["ft"]
    models = {kind: get_model(r) for kind, r in job["rs"].items()}
    held = sorted(s13.NEU[s13.NEU.fold == fold].si.unique())
    hrows = np.where((s13.NEU.fold == fold).to_numpy())[0]
    out, check = {}, []
    for si in held:
        rows = np.where(s13.NEU.si.to_numpy() == si)[0]
        loc = np.searchsorted(hrows, rows)
        o = forecast_multi(models["m1"], si, {"m_none": None}, film=False)
        # atlas cluster network: full, naive, hard, soft
        emb = models["cl"].emb.weight.detach().numpy()
        Ec = {"m_clfull": emb[A["lab"]["full"][rows]]}
        for h, _ in HS:
            Pc = R["P"]["cluster", h][loc]
            if h != "full":
                Ec[f"m_clnaive_{HT[h]}"] = emb[A["lab"][h][rows]]
            Ec[f"m_hard_{HT[h]}"] = emb[Pc.argmax(1)]
            Ec[f"m_soft_{HT[h]}"] = Pc @ emb
        cross = job["cross"][rows]
        if (cross >= 0).any():
            Ec["m_clcross"] = emb[A["lab"]["full"][np.maximum(cross, 0)]]
        o.update(forecast_multi(models["cl"], si, Ec))
        # continuous network: full, naive, E2 continuous, fine-tuned, task 4
        cn = models["cont"]
        enc = lambda Zm: cn.cont(torch.from_numpy(np.asarray(Zm, np.float32))).detach().numpy()
        Ek = {"m_contfull": enc(A["Z"]["full"][rows])}
        for h, _ in HS:
            if h != "full":
                Ek[f"m_contnaive_{HT[h]}"] = enc(A["Z"][h][rows])
            Ek[f"m_e2cont_{HT[h]}"] = enc(R["Zpred"][h][loc])
            Ek[f"m_ft_{HT[h]}"] = FT[si, h]
        if (cross >= 0).any():
            Ek["m_contcross"] = enc(A["Z"]["full"][np.maximum(cross, 0)])
        o.update(forecast_multi(cn, si, Ek))
        for kind, Zk in [("cont_ctr", "Z_ctr"), ("cont_sess", "Z_sess")]:
            m = models[kind]
            o.update(forecast_multi(m, si, {f"m_{kind[5:]}": m.cont(torch.from_numpy(A[Zk][rows])).detach().numpy()}))
        for name in ["m_clcross", "m_contcross"]:
            if name in o:
                o[name][cross < 0] = np.nan
        out[si] = o
        # check: same SSE as step 13's run_job forecast of the same model
        for kind, name in [("m1", "m_none"), ("cl", "m_clfull"), ("cont", "m_contfull")]:
            ref = job["rs"][kind]["out"][si]
            check.append(dict(fold=fold, seed=seed, si=si, variant=name,
                              max_rel_diff=float(np.max(np.abs(o[name].sum(1) - ref) / ref))))
    print(f"fold {fold} seed {seed}: {len(held)} sessions ({time.time() - t0:.0f} s)", flush=True)
    return (fold, seed), out, check


def forecast():
    global STATES
    load_mlp()
    A = load_atlas()
    STATES = A["states"]
    R = load("recognise.pkl")
    res = load("mlp.pkl")
    FT = load("finetune.pkl")["E"]
    by = {(r["job"]["fold"], r["job"]["seed"], r["job"]["kind"]): r for r in res}
    jobs = []
    for k in sorted(s13.NEU.fold.unique()):
        for seed in range(N_SEEDS):
            jobs.append(dict(fold=k, seed=seed, atlas=A["atlas"][k], rec=R[k], cross=A["cross"],
                             ft={(si, h): E for (f, sd, si, h), E in FT.items() if f == k and sd == seed},
                             rs={kind: by[k, seed, kind] for kind in MLP_KINDS}))
    acc, checks = {}, []
    with Pool(N_WORKERS) as pool:
        for key, out, check in pool.imap_unordered(forecast_job, jobs):
            checks += check
            for si, o in out.items():
                for name, v in o.items():
                    acc.setdefault((si, name), []).append(v)
    sse = {}
    for (si, name), v in acc.items():
        assert len(v) == N_SEEDS
        sse.setdefault(si, {})[name] = np.mean(v, axis=0)
    save(sse, "mlp_sse.pkl")
    ch = pd.DataFrame(checks)
    ch.to_csv(wd() / "forecast_check.csv", index=False)
    print(ch.groupby("variant").max_rel_diff.max())
    pd.DataFrame([dict(**r["job"], epochs=r["epochs"], best_val=r["best_val"], secs=r["secs"]) for r in res]) \
        .sort_values(["kind", "fold", "seed"]).to_csv(wd() / "mlp_fits.csv", index=False, float_format="%.5g")


# ---------------------------------------------------------------- analyse (CSV only)
HV = [HT[h] for h, _ in HS]
# method -> (family, column pattern per H tag; "{h}" = history tag)
METHODS = {
    "ridge": {"true type": "r_true", "naive short summary": "r_naive_{h}", "per-neuron ridge": "r_pn_{h}",
              "E2 hard": "r_hard_{h}", "E2 soft": "r_soft_{h}", "full-history summary": "r_full"},
    "MLP": {"true type": "m_true", "naive short summary": "m_contnaive_{h}", "naive short cluster": "m_clnaive_{h}",
            "per-neuron MLP fine-tune": "m_ft_{h}", "E2 hard": "m_hard_{h}", "E2 soft": "m_soft_{h}",
            "E2 continuous": "m_e2cont_{h}", "full-history summary": "m_contfull", "full-history cluster": "m_clfull"}}
NONE = {"ridge": "r_none", "MLP": "m_none"}
KEY_PAIRS = {"ridge": [("E2 soft", "per-neuron ridge"), ("E2 soft", "naive short summary"), ("E2 soft", "true type"),
                       ("E2 hard", "true type"), ("E2 soft", "E2 hard"), ("per-neuron ridge", "full-history summary"),
                       ("E2 soft", "full-history summary"), ("per-neuron ridge", "true type")],
             "MLP": [("E2 soft", "per-neuron MLP fine-tune"), ("E2 continuous", "per-neuron MLP fine-tune"),
                     ("E2 soft", "naive short cluster"), ("E2 continuous", "naive short summary"),
                     ("E2 soft", "true type"), ("E2 continuous", "true type"), ("E2 hard", "true type"),
                     ("E2 continuous", "E2 soft"), ("per-neuron MLP fine-tune", "full-history summary"),
                     ("E2 continuous", "full-history summary"), ("per-neuron MLP fine-tune", "true type")]}


def col(pattern, h):
    if "{h}" not in pattern:
        return pattern
    if h == "full" and ("naive" in pattern):
        return {"r_naive_{h}": "r_full", "m_contnaive_{h}": "m_contfull", "m_clnaive_{h}": "m_clfull"}[pattern]
    return pattern.format(h=h)


def dataset_tables(ds):
    """Per-neuron SSE (total and by state) of every variant of one dataset."""
    configure(ds)
    neu = s13.neurons()
    meta = ["session", "mouse", "fold", "neuron", "type"] + (["uid", "date"] if ds == "bugeon" else [])
    rid, mlp_ = load("ridge_sse.pkl"), load("mlp_sse.pkl")
    sess = list(neu.session.unique())
    names = sorted({v for s in rid.values() for v in s} | {v for s in mlp_.values() for v in s})
    n = len(neu)
    T = {v: np.full((n, 3), np.nan) for v in names}
    for i, s in enumerate(sess):
        rows = np.where((neu.session == s).to_numpy())[0]
        for v, x in {**rid[s], **mlp_[i]}.items():
            T[v][rows] = x
    ns = neu[meta].copy()
    ns.insert(0, "dataset", ds)
    for v in names:
        ns[f"sse_{v}"] = T[v].sum(1)
    # true type MLP (m2) and none (m1) of steps 11 / 12, as step 13
    if ds == "bugeon":
        old = pd.read_csv(ROOT / "results/step12/neuron_scores_all_labels.csv")
    else:
        old = pd.read_csv(ROOT / "results/step11/neuron_scores_all_labels.csv").assign(session=lambda d: d.id.astype(str))
    old = ns[["session", "neuron"]].merge(old, on=["session", "neuron"], how="left", validate="1:1")
    tix = pd.Index(s13.TYPES).get_indexer(ns.type)
    ns["sse_m_true"] = old[[f"sse_m2_{t}" for t in s13.TYPES]].to_numpy()[np.arange(n), tix]
    ns["sse_m_none_step11_12"] = old.sse_m1
    # reproduction of step 13
    s13n = pd.read_csv(ROOT / "results/step13/neuron_scores.csv", low_memory=False)
    s13n = ns[["session", "neuron"]].merge(s13n[s13n.dataset == ds].astype({"session": str}), on=["session", "neuron"],
                                           how="left", validate="1:1")
    rep = []
    for new, ref in [("r_none", "r_none"), ("r_true", "r_true"), ("r_full", "r_km16"), ("m_none", "m_none"),
                     ("m_true", "m_true"), ("m_clfull", "m_cl"), ("m_contfull", "m_cont"), ("r_naive_h3", "r_sel_h3"),
                     ("r_naive_h30", "r_sel_h30"), ("m_clnaive_h3", "m_cl_h3"), ("m_clnaive_h30", "m_cl_h30"),
                     ("m_contnaive_h3", "m_cont_h3"), ("m_contnaive_h30", "m_cont_h30")]:
        rep.append(dict(dataset=ds, here=new, step13=ref, **s13.repro(ns, ns[f"sse_{new}"], s13n[f"sse_{ref}"].to_numpy())))
    rep.append(dict(dataset=ds, here="copy", step13="copy",
                    max_rel_neuron_sse_diff=float(np.max(np.abs(ns.sse_copy - s13n.sse_copy) / s13n.sse_copy))))
    return ns, T, pd.DataFrame(rep), neu


def unit_table(ns, T, variants, by, state=None):
    """Score per unit: 1 - sum SSE / sum SSE_copy (all windows, or one state);
    neurons with a missing value of a variant are left out of that variant's score
    and of its copy-last sum."""
    out = {}
    cp = T["copy"].sum(1) if state is None else T["copy"][:, state]
    g = ns[by].to_numpy()
    for v in variants:
        x = (ns[f"sse_{v}"].to_numpy() if state is None or v not in T else T[v][:, state])
        ok = np.isfinite(x)
        d = pd.DataFrame(dict(u=g[ok], a=x[ok], c=cp[ok])).groupby("u", sort=False).sum()
        out[v] = 1 - d.a / d.c
    return pd.DataFrame(out)


def wil(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return s12.wil(a[ok], b[ok])


def analyse():
    OUT.mkdir(parents=True, exist_ok=True)
    tabs, reps, units, gains, stats, k4, k4s, k3, rec, choice, ftinfo, pninfo, fits, frames = ([] for _ in range(14))
    for ds in ["bugeon", "allen"]:
        if not (WORK / ds / "mlp_sse.pkl").exists():
            print(f"{ds}: no mlp_sse.pkl, skipped")
            continue
        ns, T, rep, neu = dataset_tables(ds)
        reps.append(rep)
        tabs.append(ns)
        T["m_true"] = np.column_stack([ns.sse_m_true, np.zeros(len(ns)), np.zeros(len(ns))])  # total only
        variants = [c[4:] for c in ns.columns if c.startswith("sse_") and c != "sse_copy"]
        levels = [("session", "session"), ("mouse", "mouse")] if ds == "bugeon" else [("mouse", "session")]
        U = {}
        for level, by in levels:
            u = unit_table(ns, T, variants, by)
            u["n_neurons"] = ns.groupby(by, sort=False).size()
            u["mouse"] = ns.groupby(by, sort=False).mouse.first()
            U[level] = u
            units.append(u.reset_index(names="unit").assign(dataset=ds, level=level, state="all"))
        main = "session" if ds == "bugeon" else "mouse"
        Um = U[main]
        states = load_atlas()["states"]

        def test(a, b, fam, h, what):
            r = dict(dataset=ds, family=fam, history=h, comparison=what, a=a, b=b,
                     test={"session": "sessions", "mouse": "mice"}[main], **wil(Um[a], Um[b]))
            if ds == "bugeon":
                d = (U["mouse"][a] - U["mouse"][b]).to_numpy()
                r.update(mice_up=int((d > 0).sum()), mice_n=len(d),
                         per_mouse=";".join(f"{m} {x:+.4f}" for m, x in zip(U["mouse"].index, d)))
            stats.append(r)

        # gains vs H (main figure) and key comparisons
        hf = pd.read_csv(wd() / "history_frames.csv")
        mins = {h: float((hf[f"{h} frames"] / hf.hz / 60).mean()) for h, _ in HS}
        for fam, meths in METHODS.items():
            for h in HV:
                hname = [x for x, _ in HS if HT[x] == h][0]
                for mname, pat in meths.items():
                    v = col(pat, h)
                    g = Um[v] - Um[NONE[fam]]
                    gains.append(dict(dataset=ds, family=fam, method=mname, history=hname, minutes=mins[hname],
                                      variant=v, score_mean=Um[v].mean(), gain_mean=g.mean(),
                                      gain_sem=g.std(ddof=1) / np.sqrt(len(g)), units_up=int((g > 0).sum()),
                                      n_units=len(g), p_vs_none=wil(Um[v], Um[NONE[fam]])["p"]))
                    test(v, NONE[fam], fam, hname, f"{mname} vs none")
                for a, b in KEY_PAIRS[fam]:
                    test(col(meths[a], h), col(meths[b], h), fam, hname, f"{a} vs {b}")
        # K4: session / state confound
        for fam, (full, ctr, sess) in {"ridge": ("r_full", "r_ctr", "r_sess"),
                                       "MLP": ("m_contfull", "m_ctr", "m_sess")}.items():
            for v, name in [(full, "full summary"), (ctr, "(a) centred within session"), (sess, "(b) session mean only")]:
                g = Um[v] - Um[NONE[fam]]
                k4.append(dict(dataset=ds, family=fam, summary=name, variant=v, gain_mean=g.mean(),
                               gain_sem=g.std(ddof=1) / np.sqrt(len(g)), units_up=int((g > 0).sum()), n_units=len(g),
                               frac_of_full_gain=g.mean() / (Um[full] - Um[NONE[fam]]).mean(),
                               p_vs_none=wil(Um[v], Um[NONE[fam]])["p"]))
                test(v, NONE[fam], fam, "full", f"K4 {name} vs none")
            test(ctr, full, fam, "full", "K4 (a) centred vs full summary")
            test(sess, full, fam, "full", "K4 (b) session mean vs full summary")
            test(ctr, sess, fam, "full", "K4 (a) centred vs (b) session mean")
            # by behavioural state of the test window
            for st_i, st_name in [(0, "running"), (1, "stationary")]:
                us = unit_table(ns, T, [NONE[fam], full, ctr, sess, "r_true"], "session", state=st_i)
                cp = pd.DataFrame(dict(u=ns.session, c=T["copy"][:, st_i])).groupby("u", sort=False).c.sum()
                us = us[cp > 0]
                units.append(us.reset_index(names="unit").assign(dataset=ds, level="session", state=st_name))
                for v, name in [(full, "full summary"), (ctr, "(a) centred"), (sess, "(b) session mean")] + \
                        ([("r_true", "true type")] if fam == "ridge" else []):
                    g = us[v] - us[NONE[fam]]
                    k4s.append(dict(dataset=ds, family=fam, state=st_name, summary=name, gain_mean=g.mean(),
                                    gain_sem=g.std(ddof=1) / np.sqrt(len(g)), units_up=int((g > 0).sum()),
                                    n_units=len(g), p_vs_none=wil(us[v], us[NONE[fam]])["p"],
                                    windows=int(sum((x == st_i).sum() for x in states.values()))))
            # paired running vs stationary gain (sessions with both)
            ur = unit_table(ns, T, [NONE[fam], full], "session", state=0)
            ust = unit_table(ns, T, [NONE[fam], full], "session", state=1)
            both = ur.index.intersection(ust.index)
            cpr = pd.DataFrame(dict(u=ns.session, c=T["copy"][:, 0])).groupby("u", sort=False).c.sum()
            both = [u for u in both if cpr[u] > 0]
            gr, gs = (ur[full] - ur[NONE[fam]])[both], (ust[full] - ust[NONE[fam]])[both]
            k4s.append(dict(dataset=ds, family=fam, state="running - stationary (paired)", summary="full summary",
                            gain_mean=(gr - gs).mean(), n_units=len(both), units_up=int((gr > gs).sum()),
                            p_vs_none=wil(gr, gs)["p"]))
        # K3 (Bugeon): summary from another session of the same cell
        if ds == "bugeon":
            m = ns.sse_r_cross.notna().to_numpy()
            sub, Ts = ns[m], {v: T[v][m] for v in T}
            vv = ["r_none", "r_full", "r_cross", "m_none", "m_contfull", "m_contcross", "m_clfull", "m_clcross", "m_true",
                  "r_true"]
            for level in ["session", "mouse"]:
                u = unit_table(sub, Ts, vv, level)
                u["n_neurons"] = sub.groupby(level, sort=False).size()
                units.append(u.reset_index(names="unit").assign(dataset=ds, level=f"{level} (task 4 cells)", state="all"))
                for a, b in [("r_cross", "r_none"), ("r_full", "r_none"), ("r_cross", "r_full"), ("r_true", "r_none"),
                             ("m_contcross", "m_none"), ("m_contfull", "m_none"), ("m_contcross", "m_contfull"),
                             ("m_clcross", "m_none"), ("m_clcross", "m_clfull"), ("m_true", "m_none"),
                             ("m_contcross", "m_true")]:
                    d = u[a] - u[b]
                    k3.append(dict(dataset=ds, level=level, a=a, b=b, n_cells=int(m.sum()),
                                   n_unique_cells=int(sub.drop_duplicates(["mouse", "uid"]).shape[0]),
                                   **wil(u[a], u[b]), per_unit=";".join(f"{i} {x:+.4f}" for i, x in d.items())
                                   if level == "mouse" else ""))
        if ds == "bugeon":  # stability of each S feature between two sessions of the same cell
            A = load_atlas()
            c = A["cross"]
            m = np.where(c >= 0)[0]
            Fz = A["atlas"][0]["Z"]["full"].astype(np.float64)
            rng = np.random.default_rng(0)
            rnd = []
            for i in m:  # a random other cell of the same type in the session the summary came from
                pool = np.where((neu.session == neu.session[c[i]]).to_numpy() & (neu.type == neu.type[i]).to_numpy()
                                & (np.arange(len(neu)) != c[i]))[0]
                rnd.append(np.linalg.norm(Fz[i] - Fz[rng.choice(pool)]) if len(pool) else np.nan)
            d_same = np.linalg.norm(Fz[m] - Fz[c[m]], axis=1)
            k3.append(dict(dataset=ds, level="summary distance (fold-0 z-scores)", a="same cell, other session",
                           b="random same-type cell, other session", n_cells=len(m), median_diff=np.median(d_same),
                           mean_diff=float(np.nanmedian(rnd)), a_beats_b=int(np.nansum(d_same < np.array(rnd))),
                           per_unit="median distance same cell (median_diff) vs random cell (mean_diff); "
                                    "a_beats_b = cells closer to themselves"))
            Fr = A["F"]["full"]
            for j, name in enumerate(A["names"]):
                k3.append(dict(dataset=ds, level="feature stability across sessions", a=name, b="same cell",
                               n_cells=len(m), mean_diff=np.corrcoef(Fr[m, j], Fr[c[m], j])[0, 1],
                               per_unit="Pearson r of the feature between two sessions of the same cell"))
        # naive short summary: held-out neurons whose nearest atlas centroid is their full-history cluster
        A = load_atlas()
        for h, _ in HS[:3]:
            same = sum(int((a["lab"][h] == a["lab"]["full"])[(neu.fold == k).to_numpy()].sum())
                       for k, a in A["atlas"].items())
            rec.append(dict(dataset=ds, target="cluster", kind="naive short summary (nearest centroid)", history=h,
                            window="history", n=len(neu), acc=same / len(neu)))
        # recogniser
        R = load("recognise.pkl")
        for k, r in R.items():
            rec += [dict(dataset=ds, **x) for x in r["rows"]]
            choice += [dict(dataset=ds, **x) for x in r["choice"]]
        fi = pd.DataFrame(load("finetune.pkl")["info"]).assign(dataset=ds)
        ftinfo.append(fi.groupby(["dataset", "history"], sort=False).agg(
            n_jobs=("n", "size"), neurons=("n", "sum"), fit_windows=("fit_windows", "mean"),
            val_windows=("val_windows", "mean"), steps_mean=("steps", "mean"), steps_max=("steps", "max"),
            val_init=("val_init", "mean"), val_best=("val_best", "mean"), frac_neurons_moved=("frac_moved", "mean"),
            emb_shift=("emb_shift", "mean")).reset_index())
        pi = pd.read_csv(wd() / "pn_ridge_info.csv").assign(dataset=ds)
        pninfo.append(pi.groupby(["dataset", "history"], sort=False)[["fit_windows", "val_windows", "alpha_q10",
                                                                      "alpha_q50", "alpha_q90"]].mean().reset_index())
        fits.append(pd.read_csv(wd() / "mlp_fits.csv").assign(dataset=ds))
        frames.append(pd.DataFrame([dict(dataset=ds, history=h, minutes_mean=v) for h, v in mins.items()]))
    ns = pd.concat(tabs, ignore_index=True)
    keep = [c for c in ns.columns if not c.startswith("sse_")] + ["sse_copy"]
    sc = {c.replace("sse_", "score_"): (1 - ns[c] / ns.sse_copy).round(5) for c in ns.columns
          if c.startswith("sse_") and c not in ("sse_copy",)}
    out = pd.concat([ns[keep], pd.DataFrame(sc)], axis=1)
    out.to_csv(OUT / "neuron_scores.csv", index=False, float_format="%.6g")
    pd.concat(reps).to_csv(OUT / "reproduction_check.csv", index=False, float_format="%.6g")
    U = pd.concat(units, ignore_index=True)
    U.to_csv(OUT / "unit_scores.csv", index=False, float_format="%.6g")
    pd.DataFrame(gains).to_csv(OUT / "gains_by_history.csv", index=False, float_format="%.6g")
    pd.DataFrame(stats).to_csv(OUT / "stats.csv", index=False, float_format="%.4g")
    pd.DataFrame(k4).to_csv(OUT / "k4_session.csv", index=False, float_format="%.5g")
    pd.DataFrame(k4s).to_csv(OUT / "k4_state.csv", index=False, float_format="%.5g")
    pd.DataFrame(k3).to_csv(OUT / "k3_cross_session.csv", index=False, float_format="%.5g")
    rec = pd.DataFrame(rec)
    rec.to_csv(OUT / "recogniser_folds.csv", index=False, float_format="%.5g")
    # recogniser summary: pooled over folds (weighted by rows)
    rs = []
    for key, g in rec.groupby(["dataset", "target", "kind", "history", "window"], dropna=False, sort=False):
        w = g.n.to_numpy(float)
        r = dict(zip(["dataset", "target", "kind", "history", "window"], key), n=int(w.sum()),
                 selected=bool(g.selected.mean() > 0.5) if "selected" in g and g.selected.notna().any() else np.nan,
                 frac_folds_selected=float(g.selected.mean()) if "selected" in g and g.selected.notna().any() else np.nan)
        for m_ in ["acc", "bacc", "logloss", "r2", "mse", "chance_acc", "chance_bacc", "n_tiles", "minutes"]:
            if m_ in g and g[m_].notna().any():
                r[m_] = float(np.average(g[m_], weights=w))
        rs.append(r)
    # accumulated recognition with the tile length each fold picked on inner CV (what forecasting used)
    for key, g in rec[(rec.kind == "accumulated") & (rec.selected == True)].groupby(  # noqa: E712
            ["dataset", "target", "history"], sort=False):
        w = g.n.to_numpy(float)
        rs.append(dict(zip(["dataset", "target", "history"], key), kind="accumulated", window="selected per fold",
                       n=int(w.sum()), windows_picked=";".join(g.window), selected=True,
                       **{m_: float(np.average(g[m_], weights=w)) for m_ in ["acc", "bacc", "logloss", "r2", "mse",
                                                                             "n_tiles"]
                          if m_ in g and g[m_].notna().any()}))
    pd.DataFrame(rs).to_csv(OUT / "recogniser.csv", index=False, float_format="%.5g")
    pd.DataFrame(choice).to_csv(OUT / "recogniser_selection.csv", index=False, float_format="%.5g")
    pd.concat(ftinfo).to_csv(OUT / "finetune_info.csv", index=False, float_format="%.5g")
    pd.concat(pninfo).to_csv(OUT / "per_neuron_ridge_info.csv", index=False, float_format="%.5g")
    pd.concat(fits).to_csv(OUT / "mlp_fits.csv", index=False, float_format="%.5g")
    pd.concat(frames).to_csv(OUT / "history_minutes.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(pd.concat(reps).to_string(index=False))
    print(pd.DataFrame(gains).drop(columns=["variant"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(k4).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(k4s).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if k3:
        print(pd.DataFrame(k3).drop(columns=["per_unit"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(pd.DataFrame(rs).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    sizes = {p.name: p.stat().st_size / 1e6 for p in OUT.iterdir()}
    print(f"results/step14: {sum(sizes.values()):.1f} MB;", {k: round(v, 2) for k, v in sizes.items()})


# ---------------------------------------------------------------- main
PHASES = ["step13", "features", "windows", "recognise", "ridge", "mlp", "finetune", "forecast"]


def main():
    ds = sys.argv[1]
    phase = sys.argv[2] if len(sys.argv) > 2 else "all"
    if ds == "both":
        assert phase == "analyse"
        return analyse()
    if phase == "extract":
        assert ds == "allen"
        return allen_extract()
    configure(ds)
    for ph in PHASES if phase == "all" else phase.split(","):
        print(f"\n######## {ds}: phase {ph} ########", flush=True)
        t0 = time.time()
        globals()[ph]()
        print(f"phase {ph} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
