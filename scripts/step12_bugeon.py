"""Step 12: does the cell-type label help when many types share one recording?

Data: Bugeon et al. 2022 (Nature), figshare 10.6084/m9.figshare.19448531 (v2,
"Bugeon et al 2022 v2.zip"), unpacked to data/bugeon/raw/. 4 mice, 17 sessions of
mouse V1, 7-plane two-photon imaging at ~4.3 Hz (volume rate), the same GCaMP
indicator in every neuron. Activity =
deconvolved calcium trace per frame. Every session has separate blocks:
Blank (grey screen), Drifting Gratings, Natural Scenes 01 / 02, each with its own
clock. Types (neuron.ttype.txt): EC (excitatory), IN (inhibitory, not matched to
the transcriptomics) or a Tasic et al. 2018 t-type. Labels here: EC and the 5
inhibitory subclasses (first token of the t-type) Lamp5, Pvalb, Sncg, Sst, Vip;
Serpinf1-Aqp5-Vip (15 neuron-sessions) is merged into Vip; IN cells have no type
and are left out of every model and score.

Preparation (task 1), per session:
- No resampling: the native ~4.3 Hz frames are the bins (below the ~10 Hz of
  steps 3-11). Past 13 frames (3.0 s) -> next 4 frames (0.93 s).
- Chronological 80/20 split inside every block (the blocks are separate
  recordings), last 10% of each block's train part = val. z-score per neuron with
  mean / std over the train parts of all blocks. Windows never cross a block edge.
  Test windows: last 20% of each block, stride 4 frames (non-overlapping targets,
  as steps 5-11).
- 17 stimulus features per frame (step 4 style; fraction of the frame covered):
  block type (gratings, natural scenes, blank), grating on (contrast > 0), 12
  grating directions (one-hot, while on), natural image on (non-blank image).
  Inputs per neuron: own past 13 frames + 17 x (13 + 4) stimulus inputs = 302.
- Score = 1 - MSE / MSE_copylast, as before.
- Folds: leave one mouse out (4). All sessions of a mouse are in the same fold, so
  a neuron recorded in several sessions (neuron.UniqueID) is never in train and
  test at once.

Phases (`python scripts/step12_bugeon.py <phase>`, in this order; `all` runs all):
- `extract`: raw files -> data/step12/<session>.npz (z-scored traces, stimulus,
  behaviour, window starts) and <session>_gram.npz (per-neuron ridge sufficient
  statistics), plus results/step12/sessions.csv and neurons.csv.
- `ridge` (task 2): step 5 ridge from summed per-neuron Gram statistics, alpha
  picked on val (ALPHAS of step 3), refit on train + val:
    b  shared, no label, class-balanced (each type gets equal total row weight,
       as step 7's line balance);
    d  one model per type (the 6 models are all applied to every held-out neuron,
       so any label can be scored later);
    e  shuffled-type control: the type labels of the training neurons are
       permuted (same group sizes as the true types), one class-balanced model
       per fake group; held-out neurons get a permutation of their true labels;
       5 seeds.
- `mlp` (task 2): step 9 MLP (510 -> 302 inputs, 10 -> 4 outputs; same hidden
  sizes, FiLM, zero-init, optimiser, batch, epochs, early stopping and seeds),
  class-balanced batches over the 6 types:
    m1 no label (3 seeds), m2 type FiLM (3 seeds; every held-out neuron is
    evaluated under all 6 labels), m3 shuffled FiLM with the same permuted
    labels as ridge e (5 seeds).
- `forecast`: per neuron test SSE and score under every label and no label ->
  results/step12/neuron_scores_all_labels.csv.
- `classify` (task 3): step 11 "S" features (single neuron, scale-free, train part
  of every block only) adapted to 4.3 Hz and to this stimulus set; step 10 / 11
  classifiers and selection (s11.fit_featset), inner CV over the 3 training mice.
  Also "S+state" (S plus correlation with running speed and pupil size and the
  running - stationary difference) and 5-class inhibitory-only versions for the
  NuCLR / NeuPRINT comparison.
- `assign`: everything scored from the CSV files only: task 2 tables (per mouse,
  session, type), Wilcoxon over sessions and neurons, and task 3 forecasting with
  inferred labels (fraction of the true-label gain recovered, step 10 / 11 style,
  bootstrap over sessions).

Code read but not copied: NuCLR (github.com/nerdslab/nuclr,
preprocess/bugeon/prepare_data.py, src/evaluators/bugeon.py) for the file layout,
the subclass = first token of the t-type rule and its exclusion of IN / Serpinf1
cells from the subclass task; NeuPRINT (github.com/lumimim/NeuPRINT,
data_preprocess.py, label_generation.py) for the same layout, the 5 subclasses
and how running speed / pupil are aligned to frames; NetFormer
(github.com/NeuroAIHub/NetFormer, NetFormer/data.py) only loads the same files.
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
from sklearn.metrics import f1_score
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step11_confound as s11
import step9_mlp as s9
from step3_linear import ALPHAS, ROOT, TRAIN_FRAC, VAL_FRAC
from step5_shared import Gram
from step10_inferred import EVENT_Z, IEI_EDGES, recovered

warnings.filterwarnings("ignore")

RAW = ROOT / "data/bugeon/raw"
WORK = ROOT / "data/step12"
OUT = ROOT / "results/step12"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))

TYPES = ["EC", "Lamp5", "Pvalb", "Sncg", "Sst", "Vip"]
INH = TYPES[1:]
MERGE = {"Serpinf1": "Vip"}
N_TYPES = len(TYPES)
PAST, FUTURE = 13, 4
OFF = np.arange(-PAST, FUTURE)
DIRS = np.arange(0, 360, 30)
STIM = (["block: gratings", "block: natural scenes", "block: blank", "grating on"]
        + [f"dir {d}" for d in DIRS] + ["image on"])
N_STIM = len(STIM)
N_IN = PAST + N_STIM * (PAST + FUTURE)
N_SEEDS = {"m1": 3, "m2": 3, "m3": 5}
N_E = 5          # ridge e seeds (= m3 seeds, same permutations)
N_VAL = 20000    # MLP val rows per type
CHUNK = 64       # neurons per Gram batch

# S features at 4.3 Hz: autocorrelation lags in frames (~0.23, 0.47, 0.93, 2.1, 4.9 s;
# step 10's AC_LAGS were 10 Hz bins) and bands up to the Nyquist frequency
S_LAGS = [1, 2, 4, 9, 21]
S_BANDS = [(0.0, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 2.2)]
N_SHUFFLE = 20
N_BOOT = 2000


def sessions():
    return [(m.name, d.name) for m in sorted(RAW.glob("SB*")) for d in sorted(m.iterdir()) if d.is_dir()]


def sid(mouse, date):
    return f"{mouse}_{date}"


# ---------------------------------------------------------------- extract
def on_fraction(t, dt, iv):
    """Fraction of each frame [t, t + dt) covered by the intervals iv (k, 2)."""
    if len(iv) == 0:
        return np.zeros(len(t))
    iv = iv[np.argsort(iv[:, 0])]
    s, e = iv[:, 0], iv[:, 1]
    assert (s[1:] >= e[:-1] - 1e-6).all(), "overlapping intervals"
    cum_e = np.cumsum(e - s)
    xp = np.column_stack([s, e]).ravel()
    fp = np.column_stack([cum_e - (e - s), cum_e]).ravel()
    return (np.interp(t + dt, xp, fp) - np.interp(t, xp, fp)) / dt


def frame_mean(t, ts, v):
    """Mean of samples v (times ts) within each frame [t_i, t_i+1); NaN if none."""
    idx = np.searchsorted(t, ts, side="right") - 1
    ok = (idx >= 0) & np.isfinite(v)
    n = np.bincount(idx[ok], minlength=len(t))
    with np.errstate(invalid="ignore"):
        return np.bincount(idx[ok], v[ok], minlength=len(t)) / n


def load_block(bdir, kind):
    t = np.load(bdir / "frame.times.npy").ravel()
    dt = float(np.median(np.diff(t)))
    A = np.load(bdir / "frame.neuralActivity.npy").T.astype(np.float64)  # (n, T)
    F = np.zeros((N_STIM, len(t)))
    F[{"Drifting Gratings": 0, "Natural Scenes": 1, "Blank": 2}[kind]] = 1
    if kind == "Drifting Gratings":
        iv = np.load(bdir / "ori.intervals.npy")
        con = np.load(bdir / "ori.contrast.npy").ravel()
        dr = np.load(bdir / "ori.stimDirection.npy").ravel()
        F[3] = on_fraction(t, dt, iv[con > 0])
        for k, d in enumerate(DIRS):
            F[4 + k] = on_fraction(t, dt, iv[(con > 0) & (dr == d)])
        assert np.isin(dr[con > 0], DIRS).all()
    if kind == "Natural Scenes":
        iv = np.load(bdir / "nat.intervals.npy")
        bl = np.load(bdir / "nat.blank.npy").ravel()
        F[16] = on_fraction(t, dt, iv[bl == 0])
    run = frame_mean(t, np.load(bdir / "running.times.npy").ravel(), np.load(bdir / "running.speed.npy").ravel())
    pup = frame_mean(t, np.load(bdir / "eye.times.npy").ravel(), np.load(bdir / "eye.size.npy").ravel())
    state = np.load(bdir / "frame.states.npy").ravel().astype(int)
    return dict(t=t, dt=dt, A=A, F=F, run=run, pup=pup, state=state)


def neuron_stats(Z, S_all, starts, idx):
    """Per-neuron ridge sufficient statistics of the design [own past, stimulus]
    for the windows starting at `starts`: X'X split in own-past / stimulus blocks,
    X'Y, Y'Y and sums. The stimulus-stimulus block is shared by the session."""
    S = S_all[:, starts[:, None] + OFF].transpose(1, 0, 2).reshape(len(starts), -1)  # as s9.gather
    nw, ns = len(starts), S.shape[1]
    out = {k: [] for k in ["xx", "xS", "xy", "Sy", "yy", "sx", "sy"]}
    for a in range(0, len(idx), CHUNK):
        W = Z[idx[a:a + CHUNK]][:, starts[:, None] + OFF]  # (c, nw, PAST + FUTURE)
        X, Y = W[..., :PAST], W[..., PAST:]
        c = len(W)
        out["xx"].append(np.einsum("ntp,ntq->npq", X, X))
        out["xS"].append((X.transpose(0, 2, 1).reshape(c * PAST, nw) @ S).reshape(c, PAST, ns))
        out["xy"].append(np.einsum("ntp,ntq->npq", X, Y))
        out["Sy"].append((S.T @ Y.transpose(1, 0, 2).reshape(nw, c * FUTURE)).reshape(ns, c, FUTURE).transpose(1, 0, 2))
        out["yy"].append(np.einsum("ntp,ntq->npq", Y, Y))
        out["sx"].append(X.sum(1))
        out["sy"].append(Y.sum(1))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out.update(SS=S.T @ S, sS=S.sum(0), n=nw)
    return out


def extract_session(args):
    mouse, date = args
    sdir = RAW / mouse / date
    tt = open(sdir / "neuron.ttype.txt").read().strip().split("\n")
    uid = np.load(sdir / "neuron.UniqueID.npy").ravel()
    depth = np.load(sdir / "neuron.depth.npy").ravel()
    plane = np.load(sdir / "neuron.plane.npy").ravel().astype(int)
    sub = [t if t in ("EC", "IN") else t.split("-")[0] for t in tt]
    sub = [MERGE.get(s, s) for s in sub]
    blocks = sorted(p for p in sdir.glob("*/*") if p.is_dir())
    parts = {k: [] for k in ["Ztr", "Zte", "Str", "Ste", "run", "pup", "state", "blk"]}
    fit, val, te, brow = [], [], [], []
    o_tr = o_te = 0
    for b, bdir in enumerate(blocks):
        d = load_block(bdir, bdir.parent.name)
        T = d["A"].shape[1]
        n_tr = int(TRAIN_FRAC * T)
        n_fit = n_tr - int(round(VAL_FRAC * n_tr))
        n_te = T - n_tr
        fit.append(o_tr + np.arange(PAST, n_fit - FUTURE + 1))
        val.append(o_tr + np.arange(n_fit + PAST, n_tr - FUTURE + 1))
        te.append(o_te + np.arange(PAST, n_te - FUTURE + 1, FUTURE))
        parts["Ztr"].append(d["A"][:, :n_tr]); parts["Zte"].append(d["A"][:, n_tr:])
        parts["Str"].append(d["F"][:, :n_tr]); parts["Ste"].append(d["F"][:, n_tr:])
        for k in ["run", "pup", "state"]:
            parts[k].append(d[k][:n_tr])
        parts["blk"].append(np.full(n_tr, b))
        brow.append(dict(block=f"{bdir.parent.name}/{bdir.name}", frames=T, minutes=(d["t"][-1] - d["t"][0]) / 60,
                         dt=d["dt"], n_train=n_tr, n_fit=n_fit, n_test=n_te, n_test_windows=len(te[-1]),
                         running_frac=(d["state"] == 0).mean(), stim_frames=int((d["F"][3] + d["F"][16] > 0).sum()),
                         pupil_nan=float(np.isnan(d["pup"]).mean())))
        o_tr += n_tr
        o_te += n_te
    cat = {k: np.concatenate(v, axis=-1) for k, v in parts.items()}
    mu, sd = cat["Ztr"].mean(1), cat["Ztr"].std(1)
    keep = np.array([s in TYPES for s in sub]) & (sd > 0)
    idx = np.where(keep)[0]
    Ztr = (cat["Ztr"][idx] - mu[idx, None]) / sd[idx, None]
    Zte = (cat["Zte"][idx] - mu[idx, None]) / sd[idx, None]
    fit, val, te = np.concatenate(fit), np.concatenate(val), np.concatenate(te)
    name = sid(mouse, date)
    np.savez(WORK / f"{name}.npz", Ztr=Ztr.astype(np.float32), Zte=Zte.astype(np.float32),
             Str=cat["Str"].astype(np.float32), Ste=cat["Ste"].astype(np.float32), fit=fit, val=val, te=te,
             blk=cat["blk"], run=cat["run"], pup=cat["pup"], state=cat["state"],
             hz=1 / np.median([r["dt"] for r in brow]))
    g = {}
    for part, st in [("f", fit), ("v", val)]:
        for k, v in neuron_stats(Ztr, cat["Str"], st, np.arange(len(idx))).items():
            g[f"{part}_{k}"] = v
    np.savez(WORK / f"{name}_gram.npz", **g)
    neu = pd.DataFrame(dict(session=name, mouse=mouse, date=date, orig=idx, neuron=np.arange(len(idx)),
                            ttype=np.array(tt)[idx], type=np.array(sub)[idx], uid=uid[idx], depth=depth[idx],
                            plane=plane[idx]))
    counts = pd.Series(sub).value_counts()
    srow = dict(session=name, mouse=mouse, date=date, n_rois=len(tt), n_kept=len(idx),
                n_zero_var=int((np.array([s in TYPES for s in sub]) & (sd == 0)).sum()),
                n_IN_unmatched=int(counts.get("IN", 0)),
                n_serpinf1=int(sum(t.startswith("Serpinf1") for t in tt)),
                n_uid=int(np.isfinite(uid).sum()), depth_min=depth.min(), depth_max=depth.max(),
                n_planes=len(np.unique(plane)), hz=1 / np.median([r["dt"] for r in brow]),
                frames=sum(r["frames"] for r in brow), minutes=sum(r["minutes"] for r in brow),
                n_blocks=len(blocks), blocks=";".join(r["block"] for r in brow),
                fit_windows=len(fit), val_windows=len(val), test_windows=len(te),
                **{f"n_{t}": int((neu.type == t).sum()) for t in TYPES})
    print(f"{name}: {len(tt)} ROIs -> {len(idx)} labelled neurons ("
          + ", ".join(f"{t} {srow[f'n_{t}']}" for t in TYPES) + f"), {srow['frames']} frames"
          f" ({srow['minutes']:.0f} min, {srow['hz']:.2f} Hz), windows fit/val/test {len(fit)}/{len(val)}/{len(te)}",
          flush=True)
    return neu, srow, [dict(session=name, **r) for r in brow]


def extract():
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(N_WORKERS) as pool:
        res = pool.map(extract_session, sessions(), chunksize=1)
    mice = sorted({r[1]["mouse"] for r in res})
    neu = pd.concat([r[0] for r in res], ignore_index=True)
    neu["fold"] = neu.mouse.map({m: k for k, m in enumerate(mice)})
    neu.to_csv(OUT / "neurons.csv", index=False, float_format="%.6g")
    pd.DataFrame([r[1] for r in res]).to_csv(OUT / "sessions.csv", index=False, float_format="%.5g")
    pd.DataFrame([b for r in res for b in r[2]]).to_csv(OUT / "blocks.csv", index=False, float_format="%.5g")
    print(f"{len(neu)} labelled neurons in {len(res)} sessions:", neu.type.value_counts().to_dict())


# ---------------------------------------------------------------- shared helpers
def load_neurons():
    neu = pd.read_csv(OUT / "neurons.csv")
    neu["t"] = neu.type.map({t: k for k, t in enumerate(TYPES)})
    neu["si"] = neu.session.map({s: k for k, s in enumerate(neu.session.unique())})
    return neu


def fake_labels(neu, seed, fold):
    """Shuffled-type control: true labels permuted over the training neurons (same
    group sizes as the true types) and, separately, over the held-out neurons."""
    rng = np.random.default_rng(seed)
    lab = neu.t.to_numpy().copy()
    for m in [neu.fold.to_numpy() != fold, neu.fold.to_numpy() == fold]:
        lab[m] = rng.permutation(lab[m])
    return lab


def predict(W, b, X, S):
    """step5_shared.predict with this step's PAST (13 frames)."""
    return X @ W[:PAST] + (S @ W[PAST:] + b)[:, None, :]


def test_windows(z):
    """X (nw, n, PAST), Y (nw, n, FUTURE), S (nw, N_STIM * (PAST + FUTURE)) of a session."""
    W = z["Zte"][:, z["te"][:, None] + OFF].transpose(1, 0, 2).astype(np.float64)
    S = z["Ste"][:, z["te"][:, None] + OFF].transpose(1, 0, 2).reshape(len(z["te"]), -1).astype(np.float64)
    return W[..., :PAST], W[..., PAST:], S


# ---------------------------------------------------------------- ridge (task 2)
def load_grams(neu):
    G, sess = {}, []
    for s in neu.session.unique():
        g = np.load(WORK / f"{s}_gram.npz")
        sess.append({k: g[k] for k in g.files if k.split("_", 1)[1] in ["SS", "sS", "n"]})
        for k in g.files:
            if k.split("_", 1)[1] not in ["SS", "sS", "n"]:
                G.setdefault(k, []).append(g[k])
    G = {k: np.concatenate(v) for k, v in G.items()}
    for k in sess[0]:
        G[k] = np.stack([s[k] for s in sess])
    return G


def assemble(G, si, w, part):
    """Weighted sum of per-neuron statistics -> step 5 Gram of the full design, plus Y'Y."""
    p = lambda k: G[f"{part}_{k}"]
    ws = np.bincount(si, w, minlength=len(p("n")))
    xS = np.tensordot(w, p("xS"), 1)
    XX = np.block([[np.tensordot(w, p("xx"), 1), xS], [xS.T, np.tensordot(ws, p("SS"), 1)]])
    XY = np.vstack([np.tensordot(w, p("xy"), 1), np.tensordot(w, p("Sy"), 1)])
    sx = np.concatenate([w @ p("sx"), ws @ p("sS")])
    return Gram(ws @ p("n"), sx, w @ p("sy"), XX, XY), np.tensordot(w, p("yy"), 1)


def val_mse(W, b, g, yy):
    """Weighted MSE of (W, b) on the windows summarised by Gram g (no data needed)."""
    sse = (np.trace(W.T @ g.xx @ W) + 2 * b @ (W.T @ g.sx) + g.n * b @ b - 2 * np.trace(W.T @ g.xy)
           - 2 * b @ g.sy + np.trace(yy))
    return sse / (g.n * FUTURE)


def balanced_weights(neu, mask):
    """Row weights giving each true type the same total weight among neurons in mask
    (step 7 line_weights, with rows = fit + val windows of the neuron's session)."""
    rows = neu.rows.to_numpy()
    w = np.zeros(len(neu))
    types = np.unique(neu.t[mask])
    tot = rows[mask].sum()
    for t in types:
        m = mask & (neu.t.to_numpy() == t)
        w[m] = tot / (len(types) * rows[m].sum())
    return w


def fit_weighted(G, si, w):
    """Alpha by weighted val MSE (fit part), then refit on fit + val (as step 7 fit_balanced)."""
    gf, _ = assemble(G, si, w, "f")
    gv, yyv = assemble(G, si, w, "v")
    coef = gf.solver()
    mse = [val_mse(*coef(a), gv, yyv) for a in ALPHAS]
    alpha = ALPHAS[int(np.argmin(mse))]
    W, b = (gf + gv).solver()(alpha)
    return W, b, alpha


def ridge():
    neu = load_neurons()
    G = load_grams(neu)
    si = neu.si.to_numpy()
    nf = G["f_n"][si] + G["v_n"][si]
    neu["rows"] = nf
    rows, alphas = [], []
    for k in sorted(neu.fold.unique()):
        t0 = time.time()
        tr = neu.fold.to_numpy() != k
        b = fit_weighted(G, si, balanced_weights(neu, tr))
        d = [fit_weighted(G, si, (tr & (neu.t.to_numpy() == c)).astype(float)) for c in range(N_TYPES)]
        e = []
        for seed in range(N_E):
            fake = fake_labels(neu, seed, k)
            e.append((fake, [fit_weighted(G, si, balanced_weights(neu, tr & (fake == g))) for g in range(N_TYPES)]))
        alphas.append(dict(fold=k, mouse=neu[~tr].mouse.iloc[0], b=b[2], **{f"d_{TYPES[c]}": d[c][2] for c in range(N_TYPES)},
                           e=";".join(f"{m[2]:g}" for _, ms in e for m in ms)))
        print(f"fold {k} ({alphas[-1]['mouse']}): alpha b {b[2]:g}, d "
              + ", ".join(f"{TYPES[c]} {d[c][2]:g}" for c in range(N_TYPES)) + f" ({time.time() - t0:.0f} s)", flush=True)
        for s in neu[~tr].session.unique():
            z = np.load(WORK / f"{s}.npz")
            X, Y, S = test_windows(z)
            nn_ = neu[neu.session == s]
            err = lambda P: ((P - Y) ** 2).sum(axis=(0, 2))
            r = dict(copy=err(np.repeat(X[:, :, -1:], FUTURE, axis=2)), b=err(predict(b[0], b[1], X, S)))
            for c in range(N_TYPES):
                r[f"d_{TYPES[c]}"] = err(predict(d[c][0], d[c][1], X, S))
            se = []
            for fake, ms in e:
                f = fake[nn_.index.to_numpy()]
                Pg = [predict(m[0], m[1], X, S) for m in ms]
                se.append(np.choose(f, [err(P) for P in Pg]))
            r["e"] = np.mean(se, axis=0)
            rows.append(pd.DataFrame(dict(session=s, neuron=nn_.neuron.to_numpy(), n_el=Y.shape[0] * FUTURE,
                                          **{f"sse_{m}": v for m, v in r.items()})))
    pd.concat(rows).to_csv(WORK / "ridge_sse.csv", index=False, float_format="%.9g")
    pd.DataFrame(alphas).to_csv(OUT / "ridge_alphas.csv", index=False)
    print(f"Saved {WORK}/ridge_sse.csv, {OUT}/ridge_alphas.csv")


# ---------------------------------------------------------------- MLP (task 2)
SESS = []  # filled before the worker pool forks
NEU = None


class MLP(s9.MLP):
    """step9_mlp.MLP (2 x 256 hidden, zero-initialised FiLM from a label embedding)
    with this step's input (302) and output (4) sizes."""

    def __init__(self, n_labels=0):
        super().__init__(n_labels)
        self.l1, self.out = nn.Linear(N_IN, s9.HIDDEN), nn.Linear(s9.HIDDEN, FUTURE)


def load_sess():
    global NEU
    if SESS:
        return
    NEU = load_neurons()
    for s in NEU.session.unique():
        z = np.load(WORK / f"{s}.npz")
        SESS.append({k: z[k] for k in ["Ztr", "Zte", "Str", "Ste", "fit", "val", "te"]})


def gather(si, neur, t):
    s, idx = SESS[si], t[:, None] + OFF
    z = s["Ztr"][neur[:, None], idx]
    st = s["Str"][:, idx].transpose(1, 0, 2).reshape(len(t), -1)
    return np.hstack([z[:, :PAST], st]), z[:, PAST:]


class Sampler:
    """step9_mlp.Sampler with types in place of Cre lines: every batch has the same
    number of rows of each true type; within a type, rows are uniform over
    (neuron, window) of the given neurons (rows = neuron-session indices in NEU)."""

    def __init__(self, rows, part, rng):
        self.rng, self.part = rng, part
        sub = NEU.iloc[rows]
        self.by_type = {}
        for t in sorted(sub.t.unique()):
            r = sub[sub.t == t]
            w = np.array([len(SESS[si][part]) for si in r.si], float)
            self.by_type[t] = (r.si.to_numpy(), r.neuron.to_numpy(), r.index.to_numpy(), w / w.sum())

    def draw(self, n_per_type):
        xs, ys, rs = [], [], []
        for t, (si, ne, ix, p) in self.by_type.items():
            pick = self.rng.choice(len(si), size=n_per_type, p=p)
            for s in np.unique(si[pick]):
                m = pick[si[pick] == s]
                st = SESS[s][self.part]
                x, y = gather(s, ne[m], st[self.rng.integers(len(st), size=len(m))])
                xs.append(x); ys.append(y); rs.append(ix[m])
        return np.vstack(xs), np.vstack(ys), np.concatenate(rs)


def forecast_session(model, si, lab=None):
    """Test-window predictions of one held-out session; lab: per-neuron labels or None."""
    s = SESS[si]
    X = s["Zte"][:, s["te"][:, None] + OFF].transpose(1, 0, 2)
    S = s["Ste"][:, s["te"][:, None] + OFF].transpose(1, 0, 2).reshape(len(s["te"]), -1)
    Xp, Y = X[..., :PAST], X[..., PAST:]
    nw, n = Xp.shape[:2]
    P = np.empty_like(Y)
    with torch.no_grad():
        for a in range(0, nw, 64):
            Xc, Sc = Xp[a:a + 64], S[a:a + 64]
            D = torch.from_numpy(np.concatenate([Xc.reshape(-1, PAST), np.repeat(Sc, n, axis=0)], axis=1))
            L = None if lab is None else torch.from_numpy(np.tile(lab, len(Xc)).astype(np.int64))
            P[a:a + 64] = model(D, L).numpy().reshape(len(Xc), n, FUTURE)
    return ((P - Y) ** 2).sum(axis=(0, 2)).astype(np.float64)


def run_job(job):
    """step9_mlp.run_job for one fit (same optimiser, batches, epochs, early stopping);
    returns the per-neuron test SSE of every held-out neuron (m2: under all 6 labels)."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, kind, seed = job["fold"], job["kind"], job["seed"]
    rng = np.random.default_rng(1000 * fold + seed + {"m1": 0, "m2": 1, "m3": 2}[kind] * 10)
    torch.manual_seed(int(rng.integers(2 ** 31)))
    tr = np.where(NEU.fold.to_numpy() != fold)[0]
    held = NEU.fold.to_numpy() == fold
    lab = {"m1": None, "m2": NEU.t.to_numpy(), "m3": fake_labels(NEU, seed, fold) if kind == "m3" else None}[kind]
    model = MLP(N_TYPES if lab is not None else 0)
    opt = torch.optim.Adam(model.parameters(), lr=s9.LR)
    smp = Sampler(tr, "fit", rng)
    n_t = len(smp.by_type)
    xv, yv, rv = Sampler(tr, "val", np.random.default_rng(fold)).draw(N_VAL)
    xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    lv = None if lab is None else torch.from_numpy(lab[rv])
    v_type = NEU.t.to_numpy()[rv]

    def val_loss():
        model.eval()
        with torch.no_grad():
            err = ((model(xv, lv) - yv) ** 2).mean(1).numpy()
        model.train()
        return float(np.mean([err[v_type == t].mean() for t in smp.by_type]))

    best, best_state, bad, hist = val_loss(), None, 0, []
    for ep in range(s9.MAX_EPOCHS):
        for _ in range(s9.EPOCH_BATCHES):
            x, y, r = smp.draw(s9.BATCH // n_t)
            l = None if lab is None else torch.from_numpy(lab[r])
            loss = ((model(torch.from_numpy(x), l) - torch.from_numpy(y)) ** 2).mean()
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

    out = {}
    for si in sorted(NEU[held].si.unique()):
        rows = NEU[NEU.si == si]
        if kind == "m2":
            out[si] = np.stack([forecast_session(model, si, np.full(len(rows), c)) for c in range(N_TYPES)], axis=1)
        else:
            out[si] = forecast_session(model, si, None if lab is None else lab[rows.index.to_numpy()])
    return dict(job=job, out=out, epochs=len(hist), best_val=best, hist=hist, secs=time.time() - t0,
                state={k: v.clone() for k, v in model.state_dict().items()})


def mlp():
    load_sess()
    jobs = [dict(fold=k, kind=kind, seed=seed) for k in sorted(NEU.fold.unique())
            for kind in ["m1", "m2", "m3"] for seed in range(N_SEEDS[kind])]
    print(f"{len(jobs)} MLP fits in {N_WORKERS} workers; {len(NEU)} neurons, {N_IN} inputs", flush=True)
    results = []
    with Pool(N_WORKERS) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}/{len(jobs)}] fold {j['fold']} {j['kind']} seed {j['seed']}: {r['epochs']} epochs,"
                  f" best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)
    pickle.dump(results, open(WORK / "mlp.pkl", "wb"))
    pd.DataFrame([dict(**r["job"], epochs=r["epochs"], best_val=r["best_val"], secs=r["secs"]) for r in results]) \
        .sort_values(["fold", "kind", "seed"]).to_csv(OUT / "mlp_fits.csv", index=False, float_format="%.5g")


# ---------------------------------------------------------------- forecast table
def forecast():
    neu = load_neurons()
    ns = neu[["session", "mouse", "fold", "neuron", "orig", "ttype", "type", "uid", "depth", "plane"]].copy()
    ns = ns.merge(pd.read_csv(WORK / "ridge_sse.csv"), on=["session", "neuron"], how="left", validate="1:1")
    res = pickle.load(open(WORK / "mlp.pkl", "rb"))
    acc = {}
    for r in res:
        for si, v in r["out"].items():
            acc.setdefault((r["job"]["kind"], si), []).append(v)
    for kind, n in N_SEEDS.items():
        cols = [f"sse_{kind}_{t}" for t in TYPES] if kind == "m2" else [f"sse_{kind}"]
        for si in neu.si.unique():
            v = acc[kind, si]
            assert len(v) == n, (kind, si, len(v))
            ns.loc[neu.index[neu.si == si], cols] = np.mean(v, axis=0).reshape(-1, len(cols))
    for c in [c for c in ns.columns if c.startswith("sse_") and c != "sse_copy"]:
        ns[c.replace("sse_", "score_")] = 1 - ns[c] / ns.sse_copy
    ns.to_csv(OUT / "neuron_scores_all_labels.csv", index=False, float_format="%.7g")
    print(f"Saved {OUT}/neuron_scores_all_labels.csv ({len(ns)} neurons, {ns.shape[1]} columns)")


# ---------------------------------------------------------------- classify (task 3)
def s_features(z, S, blk, hz):
    """Step 11 S features (single neuron, scale-free) on the train part z (n, L) of a
    session made of several blocks (blk: block of each frame). Temporal features
    use only frame pairs inside one block. The 10 Hz z std of step 10 is dropped:
    at the native rate it is 1 by construction of the z-score."""
    n, L = z.shape
    same = blk[1:] == blk[:-1]
    f = {"skew": skew(z, axis=1), "kurtosis": kurtosis(z, axis=1)}
    above = z > EVENT_Z
    onset = np.zeros_like(above)
    onset[:, 1:] = above[:, 1:] & ~above[:, :-1] & same
    f["event rate (/min)"] = onset.sum(1) / (L / hz / 60)
    f["time above z=2"] = above.mean(1)
    iei = np.zeros((n, len(IEI_EDGES) - 1))
    for i in range(n):
        ev = np.where(onset[i])[0]
        d = np.diff(ev)[blk[ev[1:]] == blk[ev[:-1]]] / hz if len(ev) > 1 else []
        if len(d):
            iei[i] = np.histogram(d, IEI_EDGES)[0] / len(d)
    for j in range(len(IEI_EDGES) - 1):
        f[f"IEI {IEI_EDGES[j]:g}-{IEI_EDGES[j + 1]:g} s"] = iei[:, j]
    zc = z - z.mean(1, keepdims=True)
    var = (zc ** 2).mean(1)
    for lag in S_LAGS:
        ok = blk[lag:] == blk[:-lag]
        f[f"autocorr {lag} fr ({lag / 4.3:.2g} s)"] = (zc[:, lag:][:, ok] * zc[:, :-lag][:, ok]).mean(1) / var
    P, wsum = 0, 0
    for b in np.unique(blk):
        fq, Pb = welch(z[:, blk == b], fs=hz, nperseg=128, axis=1)
        P, wsum = P + Pb * (blk == b).sum(), wsum + (blk == b).sum()
    P = P / wsum
    tot = P[:, fq > 0].sum(1)
    for lo, hi in S_BANDS:
        sel = (fq > lo) & (fq <= hi)
        f[f"power {lo:g}-{hi:g} Hz"] = np.log10(P[:, sel].sum(1) / tot + 1e-12)

    def mean_in(w):
        return z @ w / w.sum()

    for j, name in enumerate(["gratings", "natural scenes", "blank"]):
        f[f"mean z: {name}"] = mean_in(S[j])
    base = mean_in(S[0] * (1 - S[3]))  # grating block, no grating on
    f["grating on - off"] = mean_in(S[3]) - base
    f["image on - off"] = mean_in(S[16]) - mean_in(S[1] * (1 - S[16]))
    R = np.stack([mean_in(S[4 + k]) - base for k in range(len(DIRS))], axis=1)
    for k, deg in enumerate(DIRS):
        f[f"dir {deg}"] = R[:, k]
    th = np.deg2rad(DIRS)
    Rp = np.clip(R, 0, None)
    norm = Rp.sum(1) + 1e-9
    f["tuning peak"] = R.max(1)
    f["DSI"] = np.abs(Rp @ np.exp(1j * th)) / norm
    f["OSI"] = np.abs(Rp @ np.exp(2j * th)) / norm
    return f


def state_features(z, run, pup, state):
    """Behaviour-state features (not in step 11's S): correlation of the neuron with
    running speed and pupil area, and mean z running - mean z stationary."""
    def corr(v):
        ok = np.isfinite(v)
        a, b = z[:, ok] - z[:, ok].mean(1, keepdims=True), v[ok] - v[ok].mean()
        return a @ b / np.sqrt((a ** 2).sum(1) * (b ** 2).sum())
    return {"corr running speed": corr(run), "corr pupil area": corr(pup),
            "z running - stationary": z[:, state == 0].mean(1) - z[:, state > 0].mean(1)}


def features_session(s):
    z = np.load(WORK / f"{s}.npz")
    Z = z["Ztr"].astype(np.float64)
    f = s_features(Z, z["Str"].astype(np.float64), z["blk"], float(z["hz"]))
    g = state_features(Z, z["run"], z["pup"], z["state"])
    df = pd.DataFrame({**f, **g})
    df.insert(0, "neuron", np.arange(len(Z)))
    df.insert(0, "session", s)
    return df, list(f), list(g)


CLF_SETS = {"S": ("S", TYPES), "S+state": ("S+state", TYPES), "S inh": ("S", INH), "S+state inh": ("S+state", INH)}


def classify_fold(args):
    from threadpoolctl import threadpool_limits
    k, X, y, mouse, fold, inh = args
    out = {}
    with threadpool_limits(1):
        for name, (fs, classes) in CLF_SETS.items():
            rows = np.where(inh)[0] if classes == INH else np.arange(len(y))
            yy = y[rows] - (1 if classes == INH else 0)
            tr, te = np.where(fold[rows] != k)[0], np.where(fold[rows] == k)[0]
            res, sel = s11.fit_featset(X[fs][rows], yy, mouse[rows], tr, te, seed=k, perm=classes == TYPES)
            out[name] = (rows[te], res, sel)
    print(f"fold {k}: " + ", ".join(f"{n} {o[2]} (inner {o[1][o[2]]['inner']:.3f})" for n, o in out.items()), flush=True)
    return k, out


def macro_f1(true, pred, k):
    return f1_score(true, pred, labels=range(k), average="macro")


def classify():
    s11.N_INNER = 3  # inner CV over the 3 training mice
    neu = load_neurons()
    with Pool(N_WORKERS) as pool:
        res = pool.map(features_session, list(neu.session.unique()))
    feats = pd.concat([r[0] for r in res], ignore_index=True)
    s_names, g_names = res[0][1], res[0][2]
    assert (feats.session.to_numpy() == neu.session.to_numpy()).all() and (feats.neuron == neu.neuron).all()
    X = {"S": feats[s_names].to_numpy(), "S+state": feats[s_names + g_names].to_numpy()}
    bad = ~np.isfinite(X["S+state"]).all(0)
    assert not bad.any(), np.array(s_names + g_names)[bad]
    y, mouse, fold = neu.t.to_numpy(), neu.mouse.to_numpy(), neu.fold.to_numpy()
    inh = y > 0
    with Pool(N_WORKERS) as pool:
        outs = pool.map(classify_fold, [(k, X, y, mouse, fold, inh) for k in sorted(neu.fold.unique())])

    pred = neu[["session", "mouse", "fold", "neuron", "type"]].copy()
    fold_rows, imp, conf, rows = [], [], [], []
    for k, out in outs:
        for name, (te, r, sel) in out.items():
            classes = CLF_SETS[name][1]
            for clf, v in r.items():
                for j, c in enumerate(classes):
                    pred.loc[te, f"p_{name}_{clf}_{c}"] = v["proba"][:, j]
                fold_rows.append(dict(fold=k, set=name, clf=clf, C=v["C"], inner_bacc=v["inner"], selected=clf == sel))
                if "perm" in v:
                    names = s_names + (g_names if CLF_SETS[name][0] == "S+state" else [])
                    imp.append(pd.DataFrame(dict(set=name, clf=clf, fold=k, feature=names, importance=v["perm"])))
            pred.loc[te, f"sel_{name}"] = sel
    for name, (fs, classes) in CLF_SETS.items():
        for g, c in enumerate(classes):
            pred[f"p_{name}_sel_{c}"] = np.where(pred[f"sel_{name}"] == "logreg", pred[f"p_{name}_logreg_{c}"],
                                                 pred[f"p_{name}_gboost_{c}"])
        rr = pred[f"sel_{name}"].notna().to_numpy()
        for clf in ["logreg", "gboost", "sel"]:
            P = pred.loc[rr, [f"p_{name}_{clf}_{c}" for c in classes]].to_numpy()
            pred.loc[rr, f"pred_{name}_{clf}"] = np.array(classes)[P.argmax(1)]
            true = pred.loc[rr, "type"].to_numpy()
            pr = pred.loc[rr, f"pred_{name}_{clf}"].to_numpy()
            ti, pi = pd.Index(classes).get_indexer(true), pd.Index(classes).get_indexer(pr)
            per_fold = [s11.bacc(ti[pred.fold[rr] == k], pi[pred.fold[rr] == k]) for k in sorted(neu.fold.unique())]
            d = dict(set=name, clf=clf, n=int(rr.sum()), bacc=s11.bacc(ti, pi), macro_f1=macro_f1(ti, pi, len(classes)),
                     bacc_fold_mean=np.mean(per_fold), bacc_fold_std=np.std(per_fold),
                     **{f"bacc_fold{k}": v for k, v in enumerate(per_fold)},
                     **{f"recall_{c}": (pr[true == c] == c).mean() for c in classes},
                     **{f"f1_{c}": v for c, v in zip(classes, f1_score(ti, pi, labels=range(len(classes)), average=None))})
            if classes == TYPES:  # E vs I from the 6-class prediction
                d["bacc_EI"] = s11.bacc(true == "EC", pr == "EC")
                d["recall_E"], d["recall_I"] = (pr[true == "EC"] == "EC").mean(), (pr[true != "EC"] != "EC").mean()
                # macro F1 over the 5 inhibitory subclasses, on the inhibitory neurons only
                m = true != "EC"
                d["macro_f1_inh_of_6class"] = f1_score(ti[m], pi[m], labels=range(1, len(classes)), average="macro")
            rows.append(d)
            C = np.zeros((len(classes), len(classes)), int)
            np.add.at(C, (ti, pi), 1)
            for i, a in enumerate(classes):
                for j, b in enumerate(classes):
                    conf.append(dict(set=name, clf=clf, true=a, pred=b, count=int(C[i, j])))
    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "classifier_summary.csv", index=False, float_format="%.5g")
    pd.DataFrame(conf).to_csv(OUT / "confusions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "classifier_folds.csv", index=False, float_format="%.5g")
    if imp:
        pd.concat(imp).groupby(["set", "clf", "feature"], sort=False).importance.agg(["mean", "std"]).reset_index() \
            .to_csv(OUT / "feature_importance.csv", index=False, float_format="%.5g")
    feats.insert(2, "type", neu.type)
    feats.insert(2, "mouse", neu.mouse)
    feats.to_csv(WORK / "neuron_features.csv", index=False, float_format="%.6g")
    fsum = feats.groupby("type")[s_names + g_names].agg(["mean", "std"])
    fsum.columns = [f"{a}|{b}" for a, b in fsum.columns]
    fsum.T.to_csv(OUT / "features_by_type.csv", float_format="%.4g")
    keep = ["session", "mouse", "fold", "neuron", "type"] + [c for c in pred.columns if c.startswith(("pred_", "sel_"))
                                                             or "_sel_" in c]
    pred[keep].to_csv(OUT / "neuron_predictions.csv", index=False, float_format="%.4g")
    print("\n== Task 3: held-out classifier (leave one mouse out) ==")
    print(summ.drop(columns=[c for c in summ.columns if c.startswith(("bacc_fold", "f1_"))])
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


# ---------------------------------------------------------------- assign (CSV only)
def scores_for(ns, family, lab, by):
    """Score per group (session / mouse / type) of one model family with per-neuron
    labels lab (type names; None = no-label model): 1 - sum SSE / sum SSE_copy."""
    if lab is None:
        sse = ns[f"sse_{'b' if family == 'ridge' else 'm1'}"].to_numpy()
    else:
        pre = "d" if family == "ridge" else "m2"
        M = ns[[f"sse_{pre}_{t}" for t in TYPES]].to_numpy()
        sse = M[np.arange(len(ns)), pd.Index(TYPES).get_indexer(lab)]
    g = pd.DataFrame({"g": ns[by].to_numpy(), "sse": sse, "cp": ns.sse_copy.to_numpy()}).groupby("g", sort=False).sum()
    return 1 - g.sse / g.cp


def neuron_sse(ns, family, lab):
    if lab is None:
        return ns[f"sse_{'b' if family == 'ridge' else 'm1'}"].to_numpy()
    M = ns[[f"sse_{'d' if family == 'ridge' else 'm2'}_{t}" for t in TYPES]].to_numpy()
    return M[np.arange(len(ns)), pd.Index(TYPES).get_indexer(lab)]


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        p = wilcoxon(a, b).pvalue
    except ValueError:
        p = np.nan
    return dict(n=len(d), a_beats_b=int((d > 0).sum()), ties=int((d == 0).sum()), median_diff=np.median(d),
                mean_diff=d.mean(), p=p)


def recovered_over(df, pred, rng):
    """step10_inferred.recovered over the rows of df (sessions or mice)."""
    return recovered(df, pred, "true", "none", rng, N_BOOT)


def assign():
    ns = pd.read_csv(OUT / "neuron_scores_all_labels.csv")
    pr = pd.read_csv(OUT / "neuron_predictions.csv")
    pr = ns[["session", "neuron"]].merge(pr, on=["session", "neuron"], how="left", validate="1:1")
    true = ns.type.to_numpy()
    rng = np.random.default_rng(0)
    shuffles = []
    for _ in range(N_SHUFFLE):  # true labels permuted within each session (keeps each session's type mix)
        lab = true.copy()
        for s, g in ns.groupby("session").indices.items():
            lab[g] = rng.permutation(lab[g])
        shuffles.append(lab)
    labs = {"true": true, "S": pr["pred_S_sel"].to_numpy(), "S+state": pr["pred_S+state_sel"].to_numpy()}
    fams = {"ridge": ("b", "d", "e"), "MLP": ("m1", "m2", "m3")}

    # Task 2: scores per session / mouse / type for every model (true labels)
    units = []
    for by in ["session", "mouse", "type", "all"]:
        ns["all"] = "all"
        for fam, (none, lab_m, ctrl) in fams.items():
            sc = {"none": scores_for(ns, fam, None, by), "true": scores_for(ns, fam, true, by)}
            g = ns.groupby(by, sort=False)
            sc["control"] = 1 - g[f"sse_{ctrl}"].sum() / g.sse_copy.sum()
            for name in ["S", "S+state"]:
                sc[name] = scores_for(ns, fam, labs[name], by)
            sc["shuffled"] = np.mean([scores_for(ns, fam, l, by) for l in shuffles], axis=0)
            df = pd.DataFrame(sc)
            df["n_neurons"] = g.size()
            if by == "session":
                df["mouse"] = g.mouse.first()
            units.append(df.reset_index(names="unit").assign(level=by, family=fam))
    units = pd.concat(units, ignore_index=True)
    units.to_csv(OUT / "scores.csv", index=False, float_format="%.6g")

    # Per-type x per-session (for type-level tests over sessions)
    ts = []
    for fam in fams:
        ns["st"] = ns.session + "|" + ns.type
        d = pd.DataFrame({"none": scores_for(ns, fam, None, "st"), "true": scores_for(ns, fam, true, "st"),
                          "S": scores_for(ns, fam, labs["S"], "st")})
        d["control"] = 1 - ns.groupby("st", sort=False)[f"sse_{fams[fam][2]}"].sum() / ns.groupby("st", sort=False).sse_copy.sum()
        d["n_neurons"] = ns.groupby("st", sort=False).size()
        d = d.reset_index(names="st")
        d[["session", "type"]] = d.st.str.split("|", expand=True)
        ts.append(d.drop(columns="st").assign(family=fam))
    ts = pd.concat(ts, ignore_index=True)
    ts.to_csv(OUT / "type_session_scores.csv", index=False, float_format="%.6g")

    # Statistics
    st = []
    pairs = [("true", "none"), ("true", "control"), ("control", "none"), ("S", "none"), ("S", "true"),
             ("S+state", "none"), ("S+state", "true"), ("shuffled", "none"), ("shuffled", "true")]
    for fam in fams:
        ses = units[(units.level == "session") & (units.family == fam)]
        mic = units[(units.level == "mouse") & (units.family == fam)]
        for a, b in pairs:
            st.append(dict(test="sessions", type="all", family=fam, a=a, b=b, **wil(ses[a], ses[b])))
            d = (mic[a] - mic[b]).to_numpy()
            st.append(dict(test="mice (signs only)", type="all", family=fam, a=a, b=b, n=len(d),
                           a_beats_b=int((d > 0).sum()), ties=int((d == 0).sum()), median_diff=np.median(d),
                           mean_diff=d.mean(), p=wil(mic[a], mic[b])["p"]))
        # neurons: per-neuron score under each label set
        sc = {"none": 1 - neuron_sse(ns, fam, None) / ns.sse_copy, "true": 1 - neuron_sse(ns, fam, true) / ns.sse_copy,
              "S": 1 - neuron_sse(ns, fam, labs["S"]) / ns.sse_copy,
              "control": 1 - ns[f"sse_{fams[fam][2]}"] / ns.sse_copy}
        for typ in ["all"] + TYPES:
            m = np.ones(len(ns), bool) if typ == "all" else (ns.type == typ).to_numpy()
            for a, b in [("true", "none"), ("true", "control"), ("S", "none"), ("S", "true")]:
                st.append(dict(test="neurons", type=typ, family=fam, a=a, b=b, **wil(sc[a][m], sc[b][m])))
                # same, one value per unique cell (neurons recorded in several sessions averaged)
                key = np.where(ns.uid.notna(), ns.mouse + "_u" + ns.uid.astype("Int64").astype(str),
                               ns.session + "_n" + ns.neuron.astype(str))[m]
                g = pd.DataFrame(dict(k=key, a=np.asarray(sc[a])[m], b=np.asarray(sc[b])[m])).groupby("k").mean()
                st.append(dict(test="unique cells", type=typ, family=fam, a=a, b=b, **wil(g.a, g.b)))
            if typ != "all":
                d = ts[(ts.family == fam) & (ts.type == typ)]
                for a, b in [("true", "none"), ("true", "control"), ("S", "none")]:
                    st.append(dict(test="sessions", type=typ, family=fam, a=a, b=b, **wil(d[a], d[b])))
    st = pd.DataFrame(st)
    st.to_csv(OUT / "stats.csv", index=False, float_format="%.4g")

    # Task 3: fraction of the true-label gain recovered
    gr = []
    for fam in fams:
        for level in ["session", "mouse"]:
            df = units[(units.level == level) & (units.family == fam)]
            brng = np.random.default_rng(0)
            for pm in ["S", "S+state", "shuffled", "control"]:
                frac, lo, hi, gp, gt = recovered_over(df.rename(columns=lambda c: f"score_{c}"), pm, brng)
                gr.append(dict(level=level, type="all", family=fam, labels=pm, gain_pred=gp, gain_true=gt, frac=frac,
                               ci_lo=lo, ci_hi=hi))
        brng = np.random.default_rng(0)
        for typ in TYPES:
            df = ts[(ts.family == fam) & (ts.type == typ)]
            frac, lo, hi, gp, gt = recovered_over(df.rename(columns=lambda c: f"score_{c}"), "S", brng)
            gr.append(dict(level="session", type=typ, family=fam, labels="S", gain_pred=gp, gain_true=gt, frac=frac,
                           ci_lo=lo, ci_hi=hi))
    gr = pd.DataFrame(gr)
    gr.to_csv(OUT / "gain_recovered.csv", index=False, float_format="%.4g")

    # Why inferred labels can beat true ones: per-neuron gain for each (true, S-predicted) pair,
    # and the best of the 6 labels per neuron (picked on the test data: an upper bound, not a method)
    pairs_rows = []
    for fam, (none, pre, _) in fams.items():
        M = ns[[f"sse_{pre}_{t}" for t in TYPES]].to_numpy()
        g_pred = (ns[f"sse_{none}"] - neuron_sse(ns, fam, labs["S"])) / ns.sse_copy
        for (t, p), v in g_pred.groupby([ns.type, labs["S"]]):
            pairs_rows.append(dict(family=fam, true=t, pred=p, n=len(v), mean_gain=v.mean()))
        best = np.array(TYPES)[M.argmin(1)]
        for t in TYPES + ["all"]:
            m = np.ones(len(ns), bool) if t == "all" else (ns.type == t).to_numpy()
            pairs_rows.append(dict(family=fam, true=t, pred="best of 6 (oracle)", n=int(m.sum()),
                                   mean_gain=((ns[f"sse_{none}"] - M.min(1)) / ns.sse_copy)[m].mean(),
                                   **{f"frac_best_{c}": (best[m] == c).mean() for c in TYPES}))
        ns["oracle"] = M.min(1)
        g = ns.groupby("session")
        pairs_rows.append(dict(family=fam, true="all", pred="best of 6 (oracle), session mean score",
                               n=len(ns), mean_gain=(1 - g.oracle.sum() / g.sse_copy.sum()).mean()))
    pd.DataFrame(pairs_rows).to_csv(OUT / "label_pairs.csv", index=False, float_format="%.5g")

    pd.set_option("display.width", 250)
    for level in ["mouse", "type", "all"]:
        print(f"\n== Scores per {level} ==")
        print(units[units.level == level].drop(columns=["level"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n== Statistics ==")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n== Fraction of the true-label gain recovered ==")
    print(gr.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    phases = ["extract", "ridge", "mlp", "forecast", "classify", "assign"] if phase == "all" else [phase]
    for ph in phases:
        print(f"\n######## phase {ph} ########", flush=True)
        t0 = time.time()
        {"extract": extract, "ridge": ridge, "mlp": mlp, "forecast": forecast, "classify": classify,
         "assign": assign}[ph]()
        print(f"phase {ph} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
