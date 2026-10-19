"""Step 16 (Part 1): Neuropixels spikes, no calcium indicator. Genetic label
(opto-tagging) vs waveform class vs label-free activity summary for the 1 s
forecast, and synthetic calcium made from the same spikes.

Data: Allen Visual Coding Neuropixels (Siegle et al. 2021), 58 sessions, extracted
by scripts/step16_extract.py into data/step16/sessions/<id>.npz (not committed).
In the Cre x Ai32 mice the opto-tagged Pvalb / Sst / Vip units are recorded
together with untagged units in the same session, so a recording does not reveal
the label (unlike Allen 2P, one Cre line per recording).

Units: VISp, VISl, VISal, VISrl, VISam, VISpm (peak channel), isi_violations < 0.5,
amplitude_cutoff < 0.1, presence_ratio > 0.9, quality == "good" (the noise
clusters AllenSDK also drops). Units on a probe whose invalid-time intervals cover
> 20% of the analysed period are dropped. Scopes: "VISp" (main) and "VIS" (all six
areas pooled).

Phases (`python scripts/step16_neuropixels.py <phase> [args]`):
- `labels` (task 1): opto-tagging of every unit from the 10 ms light pulses at the
  end of the session. Per trial, spikes in an 8 ms baseline [-10, -2) ms and an
  8 ms evoked window [1, 9) ms from pulse onset (AllenSDK optotagging example /
  Siegle et al. 2021). Tagged = (evoked rate / (baseline rate + 1 Hz) > 2, the
  tutorial's criterion) AND (one-sided Wilcoxon signed-rank test of the per-trial
  evoked vs baseline counts, p < 0.001). The p threshold was set from the wild-type
  sessions (no ChR2), where p < 0.01 still flags 0.40% of units (as many as 40% of
  the Vip tags) and p < 0.001 flags 0.05%. The short evoked window is the latency
  criterion (first spike within 9 ms). Waveform class: fast-spiking if
  waveform_duration < 0.4 ms, else regular-spiking. Wild-type sessions get the
  same test as a false-positive check.
- `bin`: per session, spike counts in 100 ms bins from the first stimulus onset to
  1 s before the first light pulse (the optotagging epoch is excluded), z-scored
  per unit with the train-part mean / std; chronological 80/20 split, val = last
  10% of the train part; stimulus inputs; window starts per probe that avoid
  invalid-time intervals. Also builds the synthetic-calcium datasets (task 3).
- `gram <ds>`, `features <ds>`: per-unit ridge sufficient statistics (step 12
  `neuron_stats`) and S summary features (step 11 `s_features`, adapted to the
  Neuropixels stimuli) from the train part.
- `cluster|ridge|mlp <ds> <scope> <labelset>`: labelset "tag" = (i) tagged
  type (Pvalb / Sst / Vip) vs untagged, Cre sessions only; "fs" = (ii) FS vs RS,
  all sessions.
- `analyse`: results/step16/*.csv.

Windows: past 30 -> next 10 bins (3 s -> 1 s) at 10 Hz. Inputs per unit: own past
30 bins + stimulus inputs = 13 stimulus-type rows (fraction of the bin inside a
block of that stimulus type; types = the interval tables of the NWB files) over
the past 30 and next 10 bins, + running speed (cm/s / 10, mean over the bin) over
the past 40 bins (the running row is shifted by 10 bins, so its 40-bin window ends
at the last past bin and no future running is used) = 30 + 14 x 40 = 590 inputs.
Score = 1 - SSE / SSE_copylast per session (sum over the session's units).

Models and settings as steps 9-15: ridge closed form from summed per-unit
statistics (alpha grid 1e-1..1e7, chosen on val, refit on train + val): b none
(class-balanced over the label groups), d one model per label group, e shuffled
labels (permuted within each session, same counts, 5 seeds; one model per fake
group, balanced over the true groups inside it), k1 one model with equal row
weights, km16 / kmsel activity summary (k-means on the S features of the training
sessions' units: z-score, PCA 95%, n_init 10, seed = fold; one model per cluster
with equal row weights; k chosen per fold on the val windows of the training
sessions among 2-16 as step 13). MLP (step 9: 590 -> 256 -> 256 -> 10, Adam 1e-3,
batch 1024 balanced over the label groups, 400 batches / epoch, <= 40 epochs,
early stopping patience 5 on 20,000 val rows per group): m1 none, m2 label FiLM,
m3 shuffled-label FiLM (5 seeds), cont continuous-summary FiLM (step 13); 3 seeds
(m2's seeds for cont). 5 leave-sessions-out folds (fold = rank of the session id
within its genotype, mod 5).

Equal-weight references (added here): the tagged groups are ~3% of the units, so the
label-balanced no-label models (b, m1) are themselves strongly reweighted towards the
tagged units. Ridge k1 (one model, equal row weights) and e_eq (shuffled labels,
equal row weights within each fake group) and the MLP variants m1u / m2u / m3u / contu
(`mlp <ds> <scope> <ls> uniform`: rows sampled uniformly over all units' windows, val
loss uniform over rows) give the same comparisons without label-balanced training.
"""
import json
import os
for _v in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(_v, "1")   # N_WORKERS single-threaded processes
import pickle
import re
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import lfilter, welch
from scipy.stats import kurtosis, skew, wilcoxon
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step12_bugeon as s12
import step13_dynamics as s13
import step9_mlp as s9
from step3_linear import ALPHAS, ROOT, TRAIN_FRAC, VAL_FRAC
from step10_inferred import AC_LAGS, BANDS, EVENT_Z, IEI_EDGES

warnings.filterwarnings("ignore")

WORK = ROOT / "data/step16"
SESS_DIR = WORK / "sessions"
OUT = ROOT / "results/step16"
N_WORKERS = int(os.environ.get("N_WORKERS", 4))
P, F = 30, 10
OFF = np.arange(-P, F)
BIN = 0.1
LINES = ["Pvalb", "Sst", "Vip"]
GENOS = LINES + ["wt"]
N_FOLDS = 5
AREAS = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]
SCOPES = {"VISp": ["VISp"], "VIS": AREAS}
LABELSETS = {"tag": ["Pvalb", "Sst", "Vip", "untagged"], "fs": ["FS", "RS"]}
FS_MS = 0.4
MAX_INVALID = 0.2
MAX_GAP = 2.0                       # s; gaps between presentations up to this belong to the same block
RUN_SCALE = 10.0                    # running speed input = cm/s / 10
KS = [2, 3, 4, 6, 8, 12, 16]
K_MAIN = 16
N_E = 5
N_SEEDS = {"m1": 3, "m2": 3, "m3": 5, "cont": 3, "m1u": 3, "m2u": 3, "m3u": 5, "contu": 3}
SEED_OFF = {"m1": 0, "m2": 1, "m3": 2, "cont": 1,   # step 9 / 12 formula; cont uses m2's seeds (step 13)
            "m1u": 3, "m2u": 4, "m3u": 5, "contu": 4}
# "u" variants: the same models trained with uniform row sampling over all units (equal row weights, as
# ridge k1 / d) instead of label-balanced batches; added because the tagged groups are ~3% of the units,
# so label-balanced training of the no-label model is itself a large reweighting (see the result file)
BALANCED_KINDS = ["m1", "m2", "cont", "m3"]
UNIFORM_KINDS = ["m1u", "m2u", "contu", "m3u"]
N_BOOT = 2000
# Opto-tagging (AllenSDK optotagging example / Siegle et al. 2021)
PULSE = (0.009, 0.02)               # 10 ms square pulses
BASE_WIN, EVOKED_WIN = (-0.010, -0.002), (0.001, 0.009)
RATIO_THR, P_THR = 2.0, 0.001     # p threshold set from the wild-type false-positive rate (task 1)
PSTH_EDGES = np.arange(-0.010, 0.0250001, 0.0005)
# Synthetic calcium (task 3): double-exponential kernel, rise / decay time constants (s).
# Decay: median OASIS decay of the Allen 2P neurons fitted in step 15 (results/step15/oasis_params.csv):
# GCaMP6f lines (Cux2 / Sst / Vip) 0.427 s, GCaMP6s (Pvalb) 0.832 s. Rise: approximate values from
# Chen et al. 2013 (~50 ms 6f, ~150 ms 6s). Sensitivity: 6s with a 1.5 s decay.
KERNELS = {"6f": (0.050, 0.427), "6s": (0.150, 0.832), "6s_slow": (0.150, 1.5)}
FRAME_HZ = 30.0
CALCIUM = {  # dataset -> indicator per genotype
    "c6f": {g: "6f" for g in GENOS},
    "c6s": {g: "6s" for g in GENOS},
    "callen": {"Pvalb": "6s", "Sst": "6f", "Vip": "6f", "wt": "6f"},
    "callen_slow": {"Pvalb": "6s_slow", "Sst": "6f", "Vip": "6f", "wt": "6f"},
}
DATASETS = ["spk"] + list(CALCIUM)

# step 12 / 13 helpers read these module globals
s12.PAST, s12.FUTURE, s12.OFF, s12.CHUNK = P, F, OFF, 16
s13.P, s13.F = P, F


# ---------------------------------------------------------------- sessions
def sessions():
    s = pd.read_csv(WORK / "meta/sessions.csv")
    s["line"] = s.genotype.map(lambda g: g.split("-")[0] if "Cre" in g else "wt")
    s = s.sort_values("id").reset_index(drop=True)
    s["fold"] = s.groupby("line").cumcount() % N_FOLDS
    m = pd.read_csv(WORK / "manifest.csv")
    done = set(m[m.status == "done"].session)
    s["extracted"] = s.id.isin(done)
    return s


def load_raw(sid):
    z = np.load(SESS_DIR / f"{sid}.npz", allow_pickle=True)
    units = pd.DataFrame(z["units"])
    off = z["spike_offsets"]
    spk = z["spikes"]
    times = [spk[off[i]:off[i + 1]].astype(np.float64) / float(z["tick"]) for i in range(len(units))]
    return z, units, times


# ---------------------------------------------------------------- task 1: labels
def optotag_session(sid):
    z, units, times = load_raw(sid)
    opto = pd.DataFrame(z["opto"])
    if not len(opto):
        opto = pd.DataFrame(dict(name=[], duration=[], start=[]))
    pulses = opto[(opto.name == "pulse") & (opto.duration > PULSE[0]) & (opto.duration < PULSE[1])].start.to_numpy()
    rows, psth = [], []
    if not len(pulses):
        nan = np.full(len(units), np.nan)
        return sid, pd.DataFrame(dict(unit_id=units.unit_id.to_numpy(), baseline_rate=nan, evoked_rate=nan, ratio=nan,
                                      p_wilcoxon=nan, frac_trials_evoked_spike=nan, median_first_spike_ms=nan, n_pulses=0)), \
            np.full((len(units), len(PSTH_EDGES) - 1), np.nan)
    for i, t in enumerate(times):
        def counts(w):
            return np.searchsorted(t, pulses + w[1], "left") - np.searchsorted(t, pulses + w[0], "left")
        cb, ce = counts(BASE_WIN), counts(EVOKED_WIN)
        dur_b, dur_e = BASE_WIN[1] - BASE_WIN[0], EVOKED_WIN[1] - EVOKED_WIN[0]
        rb, re_ = cb.mean() / dur_b, ce.mean() / dur_e
        d = ce - cb
        p = wilcoxon(ce, cb, alternative="greater").pvalue if np.any(d != 0) else 1.0
        lo = np.searchsorted(t, pulses)
        first = np.array([t[j] - s if j < len(t) and t[j] - s < 0.025 else np.nan for j, s in zip(lo, pulses)])
        h = np.stack([np.histogram(t[np.searchsorted(t, s + PSTH_EDGES[0]):np.searchsorted(t, s + PSTH_EDGES[-1])] - s,
                                   PSTH_EDGES)[0] for s in pulses]).mean(0) / np.diff(PSTH_EDGES)
        psth.append(h)
        rows.append(dict(baseline_rate=rb, evoked_rate=re_, ratio=re_ / (rb + 1), p_wilcoxon=p,
                         frac_trials_evoked_spike=(ce > 0).mean(),
                         median_first_spike_ms=np.nanmedian(first) * 1e3 if np.isfinite(first).any() else np.nan))
    r = pd.DataFrame(rows)
    r.insert(0, "unit_id", units.unit_id.to_numpy())
    r["n_pulses"] = len(pulses)
    return sid, r, np.array(psth)


def labels():
    S = sessions()
    S = S[S.extracted]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(N_WORKERS) as pool:
        res = pool.map(optotag_session, S.id.tolist(), chunksize=1)
    tabs, psths = [], {}
    for sid, r, h in res:
        z = np.load(SESS_DIR / f"{sid}.npz", allow_pickle=True)
        units = pd.DataFrame(z["units"])
        info = S[S.id == sid].iloc[0]
        r = units[["unit_id", "area", "probe", "local_channel", "probe_vertical_position", "quality",
                   "waveform_duration", "waveform_halfwidth", "firing_rate", "snr"]].merge(r, on="unit_id")
        r.insert(0, "session", sid)
        r.insert(1, "line", info.line)
        r.insert(2, "session_type", info.session_type)
        r.insert(3, "fold", info.fold)
        r["tagged_tutorial"] = r.ratio > RATIO_THR
        r["tagged"] = (r.ratio > RATIO_THR) & (r.p_wilcoxon < P_THR)
        r["tagged_p01"] = (r.ratio > RATIO_THR) & (r.p_wilcoxon < 0.01)   # sensitivity: looser p
        r["fs"] = r.waveform_duration < FS_MS
        r["label_fs"] = np.where(r.fs, "FS", "RS")
        r["label_tag"] = np.where(info.line == "wt", "", np.where(r.tagged, info.line, "untagged"))
        tabs.append(r)
        psths[sid] = h
    lab = pd.concat(tabs, ignore_index=True)
    lab.to_csv(WORK / "labels.csv", index=False)
    pickle.dump(psths, open(WORK / "psth.pkl", "wb"))
    print(lab[lab.quality == "good"].groupby("line")[["tagged", "tagged_tutorial", "fs"]].agg(["sum", "mean"]))


# ---------------------------------------------------------------- binning (spikes and synthetic calcium)
def blocks(st, edges):
    """Fraction of every bin covered by each stimulus type's blocks (presentations
    with gaps <= MAX_GAP merged); returns {type: (n_bins,)}."""
    out = {}
    for name, g in st.groupby("stimulus"):
        a, b = g.start.to_numpy(), g.stop.to_numpy()
        o = np.argsort(a)
        a, b = a[o], b[o]
        merged = []
        for s, e in zip(a, b):
            if merged and s - merged[-1][1] <= MAX_GAP:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out[name] = coverage(np.array(merged), edges)
    return out


def coverage(iv, edges):
    """Fraction of each bin [edges[k], edges[k+1]) covered by the intervals iv (m, 2)."""
    cov = np.zeros(len(edges) - 1)
    for s, e in iv:
        k0, k1 = np.searchsorted(edges, s, "right") - 1, np.searchsorted(edges, e, "left")
        for k in range(max(k0, 0), min(k1, len(cov))):
            cov[k] += max(0.0, min(e, edges[k + 1]) - max(s, edges[k]))
    return np.clip(cov / np.diff(edges), 0, 1)


def coverage_fast(starts, stops, edges):
    """coverage() for many short intervals (trials), vectorised over intervals no
    longer than a few bins."""
    cov = np.zeros(len(edges) - 1)
    k0 = np.clip(np.searchsorted(edges, starts, "right") - 1, 0, len(cov) - 1)
    span = int(np.ceil((stops - starts).max() / BIN)) + 2 if len(starts) else 0
    for j in range(span):
        k = np.clip(k0 + j, 0, len(cov) - 1)
        ov = np.clip(np.minimum(stops, edges[k + 1]) - np.maximum(starts, edges[k]), 0, None)
        np.add.at(cov, k, ov)
    return np.clip(cov / np.diff(edges), 0, 1)


def stim_types():
    p = WORK / "stim_types.json"
    if p.exists():
        return json.load(open(p))
    names = set()
    for sid in sessions().query("extracted").id:
        names |= set(pd.DataFrame(np.load(SESS_DIR / f"{sid}.npz", allow_pickle=True)["stim"]).stimulus)
    names = sorted(names)
    json.dump(names, open(p, "w"))
    return names


def trial_rows(st, edges):
    """Trial-level stimulus rows used only by the S summary: drifting-grating block /
    grating on / grating on per direction (0-315 by 45), flash block / flash on."""
    dga = st[st.stimulus.str.startswith("drifting_gratings")]
    dg = dga[np.isfinite(dga.orientation)]          # blank sweeps (no orientation) count as "off"
    fl = st[st.stimulus == "flashes"]
    R = {"dg block": blocks(dga.assign(stimulus="dg"), edges)["dg"] if len(dga) else np.zeros(len(edges) - 1)}
    R["dg on"] = coverage_fast(dg.start.to_numpy(), dg.stop.to_numpy(), edges)
    for d in range(0, 360, 45):
        m = dg.orientation == d
        R[f"dg dir {d}"] = coverage_fast(dg.start[m].to_numpy(), dg.stop[m].to_numpy(), edges) if m.any() else np.zeros(len(edges) - 1)
    R["flash block"] = blocks(fl.assign(stimulus="f"), edges)["f"] if len(fl) else np.zeros(len(edges) - 1)
    R["flash on"] = coverage_fast(fl.start.to_numpy(), fl.stop.to_numpy(), edges)
    return R


def windows_ok(valid, lo, hi, stride=1):
    """Window starts s in [lo, hi] (step stride) whose bins s - P .. s + F - 1 are all valid."""
    st = np.arange(lo, hi + 1, stride)
    bad = np.concatenate([[0], np.cumsum(~valid)])
    return st[(bad[st + F] - bad[st - P]) == 0]


def bin_session(sid):
    out = WORK / "spk" / f"{sid}.npz"
    if out.exists():
        return
    z, units, times = load_raw(sid)
    st = pd.DataFrame(z["stim"])
    opto = pd.DataFrame(z["opto"])
    t0 = float(st.start.min())
    # end: 1 s before the first light pulse; sessions without an optotagging epoch: end of the last stimulus
    t1 = float(opto.start.min()) - 1.0 if len(opto) else float(st.stop.max())
    N = int((t1 - t0) / BIN)
    edges = t0 + BIN * np.arange(N + 1)
    n_tr = int(TRAIN_FRAC * N)
    n_fit = n_tr - int(round(VAL_FRAC * n_tr))
    n_te = N - n_tr
    # invalid intervals per probe
    probes = sorted(units.probe.unique())
    inv = {p: np.zeros(N, bool) for p in probes}
    for a, b, tag in zip(z["invalid_start"], z["invalid_stop"], z["invalid_tags"]):
        named = re.findall(r"probe[A-F]", str(tag))       # interval tagged with a probe: that probe only
        hit = [p for p in probes if p in named] if named else probes
        k0, k1 = max(np.searchsorted(edges, a, "right") - 1, 0), min(np.searchsorted(edges, b, "left"), N)
        for p in hit:
            inv[p][k0:k1] = True
    inv_frac = {p: float(v.mean()) for p, v in inv.items()}
    good = (units.quality == "good").to_numpy() & units.area.isin(AREAS).to_numpy()
    good &= np.array([inv_frac[p] <= MAX_INVALID for p in units.probe])
    C = np.stack([np.diff(np.searchsorted(t, edges)) for t in times]).astype(np.float64)
    valid_u = np.stack([~inv[p] for p in units.probe])
    tr = valid_u[:, :n_tr]
    mu = (C[:, :n_tr] * tr).sum(1) / tr.sum(1)
    sd = np.sqrt((((C[:, :n_tr] - mu[:, None]) ** 2) * tr).sum(1) / tr.sum(1))
    good &= sd > 0
    idx = np.where(good)[0]
    Ci = C[idx]
    dt_ = np.uint8 if Ci.max() <= 255 else np.uint16      # counts on disk; z = (count - mean) / sd on load
    assert Ci.max() <= np.iinfo(dt_).max
    Ci = Ci.astype(dt_)
    # stimulus inputs
    types = stim_types()
    cov = blocks(st, edges)
    Srows = [cov.get(n, np.zeros(N)) for n in types]
    rt, rs = z["run_t"], z["run_speed"].astype(np.float64)
    ok = np.isfinite(rs)
    rt, rs = rt[ok], rs[ok]
    k = np.searchsorted(edges, rt, "right") - 1
    m = (k >= 0) & (k < N)
    sums, cnt = np.bincount(k[m], rs[m], N), np.bincount(k[m], None, N)
    centers = edges[:-1] + BIN / 2
    run = np.where(cnt > 0, sums / np.maximum(cnt, 1), np.interp(centers, rt, rs))
    run_shift = np.concatenate([np.full(F, run[0]), run[:-F]]) / RUN_SCALE  # bin j holds running of bin j - F
    S = np.vstack(Srows + [run_shift]).astype(np.float32)
    T = trial_rows(st, edges)
    gr = []
    for p in probes:
        v = ~inv[p]
        gr.append(dict(probe=p, fit=windows_ok(v[:n_tr], P, n_fit - F), val=windows_ok(v[:n_tr], n_fit + P, n_tr - F),
                       te=windows_ok(v[n_tr:], P, n_te - F, F), valid=v))
    (WORK / "spk").mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, Ctr=Ci[:, :n_tr], Cte=Ci[:, n_tr:], Str=S[:, :n_tr], Ste=S[:, n_tr:], unit_id=units.unit_id.to_numpy()[idx],
             unit_probe=units.probe.to_numpy()[idx].astype(str), probes=np.array(probes),
             **{f"{k_}_{g['probe']}": g[k_] for g in gr for k_ in ["fit", "val", "te", "valid"]},
             trial_names=np.array(list(T)), trial_rows=np.vstack(list(T.values())).astype(np.float32),
             t0=t0, t1=t1, n_tr=n_tr, n_fit=n_fit, stim_names=np.array(types + ["running (past 40 bins)"]),
             mean_count=mu[idx], sd_count=sd[idx], run=run.astype(np.float32))
    tmp.rename(out)
    pd.DataFrame(dict(session=sid, probe=probes, invalid_frac=[inv_frac[p] for p in probes],
                      n_units_kept=[int((units.probe.to_numpy()[idx] == p).sum()) for p in probes],
                      n_good_dropped_invalid=[int(((units.probe == p) & (units.quality == "good") & units.area.isin(AREAS)).sum()
                                                  - (units.probe.to_numpy()[idx] == p).sum()) if inv_frac[p] > MAX_INVALID else 0
                                              for p in probes],
                      fit_windows=[len(g["fit"]) for g in gr], val_windows=[len(g["val"]) for g in gr],
                      test_windows=[len(g["te"]) for g in gr], n_bins=N, minutes=(t1 - t0) / 60)) \
        .to_csv(WORK / "spk" / f"{sid}_probes.csv", index=False)
    print(f"{sid}: {len(idx)} units, {N} bins ({(t1 - t0) / 60:.0f} min), invalid "
          + ", ".join(f"{p} {v:.3f}" for p, v in inv_frac.items() if v > 0), flush=True)


# ---------------------------------------------------------------- synthetic calcium (task 3)
def allen_snr():
    """Signal-to-noise of the Allen 2P neurons (VISp, step 8 / 15 sessions): (train
    dF/F variance - sn^2) / sn^2, sn = OASIS noise (30 Hz) from step 15, dF/F std from
    step 10 (both on the train part). Per indicator: 6f = Cux2 / Sst / Vip, 6s = Pvalb."""
    f = pd.read_csv(ROOT / "results/step10/neuron_features.csv", usecols=["id", "neuron", "cre", "dF/F std"])
    o = pd.read_csv(ROOT / "results/step15/oasis_params.csv", usecols=["id", "neuron", "line", "sn"])
    d = o.merge(f, on=["id", "neuron"], validate="1:1")
    d["snr"] = np.clip((d["dF/F std"] ** 2 - d.sn ** 2) / d.sn ** 2, 0.01, None)
    d["indicator"] = np.where(d.line == "Pvalb", "6s", "6f")
    return d


def kernel_peak(tr, td):
    tp = tr * td / (td - tr) * np.log(td / tr)
    return np.exp(-tp / td) - np.exp(-tp / tr)


def calcium_trace(t, frames, tr, td):
    """Noise-free double-exponential calcium (peak 1 per spike) at the frame times,
    exact sums over the native spike times: c(t_k) = sum_{s <= t_k} exp(-(t_k-s)/td) - exp(-(t_k-s)/tr)."""
    dt = frames[1] - frames[0]
    t = t[(t >= frames[0] - 20 * td) & (t <= frames[-1])]
    lead = int(np.ceil(20 * td / dt))
    fr0 = frames[0] - lead * dt
    k = np.ceil((t - fr0) / dt - 1e-9).astype(int)
    tk = fr0 + k * dt
    n = len(frames) + lead
    out = 0
    for tau, sign in [(td, 1), (tr, -1)]:
        u = np.bincount(k, np.exp(-(tk - t) / tau), n)
        out = out + sign * lfilter([1.0], [1.0, -np.exp(-dt / tau)], u)
    return out[lead:] / kernel_peak(tr, td)


SNR_Q = {}


def calcium_arrays(ds, sid, b):
    """Synthetic calcium of dataset ds for the kept units of session sid, generated
    from the native spike times (deterministic: the SNR quantile and the noise of a
    unit are seeded by its unit id, the same in every version). Not stored on disk
    (4 versions x 58 sessions would not fit); about 1 s per 50 units."""
    if not SNR_Q:
        snr = allen_snr()
        SNR_Q.update({ind: np.sort(snr[snr.indicator == ind].snr.to_numpy()) for ind in ["6f", "6s"]})
    line = sessions().set_index("id").loc[sid, "line"]
    z, units, times = load_raw(sid)
    uid = b["unit_id"]
    pos = pd.Series(np.arange(len(units)), index=units.unit_id).loc[uid].to_numpy()
    n_tr, N = int(b["n_tr"]), b["Ctr"].shape[1] + b["Cte"].shape[1]
    frames = float(b["t0"]) + np.arange(3 * N) / FRAME_HZ + 0.5 / FRAME_HZ  # 3 frames per 100 ms bin
    valid = {p: b[f"valid_{p}"] for p in b["probes"]}
    if True:
        ind = CALCIUM[ds][line]
        tr_, td = KERNELS[ind]
        dist = SNR_Q["6s" if ind.startswith("6s") else "6f"]
        Z = np.empty((len(uid), N), np.float32)
        snr_used = np.empty(len(uid))
        for j, (u, i) in enumerate(zip(uid, pos)):
            c = calcium_trace(times[i], frames, tr_, td)
            rng = np.random.default_rng(int(u))           # same SNR quantile and noise per unit in every version
            qu = rng.random()
            s = dist[min(int(qu * len(dist)), len(dist) - 1)]
            vb = np.repeat(valid[b["unit_probe"][j]][:n_tr], 3)
            ctr = c[:3 * n_tr][vb]
            sig = np.sqrt(ctr.var() / s)
            y = c + sig * rng.standard_normal(len(c))
            # As Allen (step 5 load_session): z-score at 30 Hz with the train-part statistics, then 3-frame means
            yt = y[:3 * n_tr][vb]
            y = (y - yt.mean()) / yt.std()
            Z[j] = y.reshape(N, 3).mean(1)
            snr_used[j] = s
    return Z[:, :n_tr], Z[:, n_tr:], snr_used


def binning():
    S = sessions()
    ids = S[S.extracted].id.tolist()
    stim_types()
    with Pool(N_WORKERS) as pool:
        pool.map(bin_session, ids, chunksize=1)
    pd.concat([pd.read_csv(WORK / "spk" / f"{i}_probes.csv") for i in ids]).to_csv(WORK / "probes_binned.csv", index=False)


# ---------------------------------------------------------------- per-dataset data access
_CACHE = {}


def session_arrays(ds, sid):
    """Z train / test (float32) of dataset ds plus the spike file (stimulus inputs,
    windows). Spikes: counts z-scored with the train-part mean / sd. Calcium:
    generated (calcium_arrays). The last session is cached."""
    key = (ds, sid)
    if key in _CACHE:
        return _CACHE[key]
    b = dict(np.load(WORK / "spk" / f"{sid}.npz"))
    if ds == "spk":
        mu, sd = b["mean_count"][:, None], b["sd_count"][:, None]
        Ztr = ((b["Ctr"] - mu) / sd).astype(np.float32)
        Zte = ((b["Cte"] - mu) / sd).astype(np.float32)
    else:
        Ztr, Zte, snr = calcium_arrays(ds, sid, b)
        b["snr"] = snr
    _CACHE.clear()
    _CACHE[key] = (b, Ztr, Zte)
    return _CACHE[key]


def neuron_stats(Z, S_all, starts, idx, chunk=16):
    """step12_bugeon.neuron_stats with the products done on contiguous transposed
    windows (same statistics to float rounding; checked against step 12 in gram_session)."""
    S = np.ascontiguousarray(S_all[:, starts[:, None] + OFF].transpose(1, 0, 2).reshape(len(starts), -1))
    nw, ns = len(starts), S.shape[1]
    out = {k: [] for k in ["xx", "xS", "xy", "Sy", "yy", "sx", "sy"]}
    for a in range(0, len(idx), chunk):
        W = Z[idx[a:a + chunk]][:, starts[:, None] + OFF]          # (c, nw, P + F)
        c = len(W)
        Xt = np.ascontiguousarray(W[..., :P].transpose(0, 2, 1))   # (c, P, nw)
        Yt = np.ascontiguousarray(W[..., P:].transpose(0, 2, 1))   # (c, F, nw)
        out["xx"].append(np.einsum("cpn,cqn->cpq", Xt, Xt, optimize=True))
        out["xy"].append(np.einsum("cpn,cqn->cpq", Xt, Yt, optimize=True))
        out["yy"].append(np.einsum("cpn,cqn->cpq", Yt, Yt, optimize=True))
        out["xS"].append((Xt.reshape(c * P, nw) @ S).reshape(c, P, ns))
        out["Sy"].append((Yt.reshape(c * F, nw) @ S).reshape(c, F, ns).transpose(0, 2, 1))
        out["sx"].append(Xt.sum(2))
        out["sy"].append(Yt.sum(2))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out.update(SS=S.T @ S, sS=S.sum(0), n=nw)
    return out


def gram_session(args):
    ds, sid = args
    out = WORK / ds / "gram" / f"{sid}.npz"
    if out.exists():
        return
    b, Ztr, _ = session_arrays(ds, sid)
    S = b["Str"].astype(np.float64)
    Z = Ztr.astype(np.float64)
    g = {}
    for p in b["probes"]:
        idx = np.where(b["unit_probe"] == p)[0]
        if not len(idx):
            continue
        for part, st in [("f", b[f"fit_{p}"]), ("v", b[f"val_{p}"])]:
            for k, v in neuron_stats(Z, S, st, idx).items():
                g[f"{p}|{part}_{k}"] = v
            if part == "v" and p == b["probes"][0]:  # check against step 12's implementation on 4 units
                ref = s12.neuron_stats(Z, S, st, idx[:4])
                for k, v in ref.items():
                    new = g[f"{p}|v_{k}"] if np.ndim(v) == 0 or k in ("SS", "sS") else g[f"{p}|v_{k}"][:4]
                    assert np.allclose(new, v, rtol=1e-9, atol=1e-6), (sid, k)
    (WORK / ds / "gram").mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **g)
    tmp.rename(out)
    print(f"{ds} {sid}: gram", flush=True)


def gram(ds):
    ids = sessions().query("extracted").id.tolist()
    with Pool(N_WORKERS) as pool:
        pool.map(gram_session, [(ds, i) for i in ids], chunksize=1)


# ---------------------------------------------------------------- S summary (step 11 features, Neuropixels stimuli)
S_MEAN_TYPES = {"spontaneous": ["spontaneous"], "gabors": ["gabors"], "flashes": ["flashes"],
                "drifting gratings": ["drifting_gratings", "drifting_gratings_75_repeats", "drifting_gratings_contrast"],
                "natural movie one": ["natural_movie_one", "natural_movie_one_more_repeats"]}


def s_features(z, S, names, T, tnames, hz):
    """Step 11 s_features on the z-scored 10 Hz train part z (n, L) (the same
    single-unit statistics: z std, skew, kurtosis, event rate and time above z = 2,
    inter-event intervals, autocorrelation at 0.1-5 s, relative power in 5 bands),
    with the stimulus features of the Neuropixels sessions: mean z in the stimulus
    types shown in both session types, grating on - off and flash on - off, tuning
    peak and orientation selectivity from the drifting-grating directions present.
    A stimulus absent from a session's train part gives NaN (-> training mean in the
    clustering, as step 13)."""
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
            f[f"autocorr {lag / 10:g} s"] = (zc[:, lag:] * zc[:, :-lag]).mean(1) / var
        fq, Pw = welch(z, fs=hz, nperseg=256, axis=1)
        tot = Pw[:, fq > 0].sum(1)
        for lo, hi in BANDS:
            sel = (fq > lo) & (fq <= hi)
            f[f"power {lo:g}-{hi:g} Hz"] = np.log10(Pw[:, sel].sum(1) / tot + 1e-12)

        def mean_in(w):
            return z @ w / w.sum() if w.sum() > 0 else np.full(n, np.nan)

        nm = list(names)
        for lab, typ in S_MEAN_TYPES.items():
            w = sum(S[nm.index(t)] for t in typ if t in nm)
            f[f"mean z: {lab}"] = mean_in(np.asarray(w, float) if not np.isscalar(w) else np.zeros(L))
        tn = list(tnames)
        row = lambda k: T[tn.index(k)]
        f["grating on - off"] = mean_in(row("dg on")) - mean_in(row("dg block") * (1 - row("dg on")))
        f["flash on - off"] = mean_in(row("flash on")) - mean_in(row("flash block") * (1 - row("flash on")))
        base = mean_in(row("dg block") * (1 - row("dg on")))
        dirs = [d for d in range(0, 360, 45) if row(f"dg dir {d}").sum() > 0]
        if dirs:
            R = np.stack([mean_in(row(f"dg dir {d}")) - base for d in dirs], 1)
            th = np.deg2rad(dirs)
            Rp = np.clip(R, 0, None)
            f["tuning peak"] = R.max(1)
            f["OSI"] = np.abs(Rp @ np.exp(2j * th)) / (Rp.sum(1) + 1e-9)
        else:
            f["tuning peak"] = f["OSI"] = np.full(n, np.nan)
    return f


def feats_session(args):
    ds, sid = args
    b, Ztr, _ = session_arrays(ds, sid)
    out = []
    for p in b["probes"]:
        idx = np.where(b["unit_probe"] == p)[0]
        if not len(idx):
            continue
        n_tr = Ztr.shape[1]
        v = b[f"valid_{p}"][:n_tr]
        f = s_features(Ztr[idx][:, v].astype(np.float64), b["Str"][:, v].astype(np.float64), b["stim_names"],
                       b["trial_rows"][:, :n_tr][:, v].astype(np.float64), b["trial_names"], 1 / BIN)
        df = pd.DataFrame(dict(session=sid, unit_id=b["unit_id"][idx], **f))
        if "snr" in b:
            df["snr_used"] = b["snr"][idx]
        out.append(df)
    return pd.concat(out)


def features(ds):
    ids = sessions().query("extracted").id.tolist()
    with Pool(N_WORKERS) as pool:
        df = pd.concat(pool.map(feats_session, [(ds, i) for i in ids], chunksize=1), ignore_index=True)
    (WORK / ds).mkdir(parents=True, exist_ok=True)
    if "snr_used" in df:  # synthetic calcium: the Allen-matched SNR drawn per unit
        df[["session", "unit_id", "snr_used"]].to_csv(WORK / ds / "snr.csv", index=False, float_format="%.5g")
        df = df.drop(columns="snr_used")
    df.to_csv(WORK / ds / "features.csv", index=False, float_format="%.7g")
    X = df.iloc[:, 2:]
    print(f"{ds}: {X.shape[1]} S features x {len(df)} units; NaN fraction per feature:",
          X.isna().mean()[X.isna().mean() > 0].round(3).to_dict())


# ---------------------------------------------------------------- configurations (dataset, scope, label set)
def neurons(scope, ls):
    """One row per unit of the configuration, ordered by (session, probe, unit) as the
    data files; gi = (session, probe) group index (the "si" of the step 12 / 13
    ridge helpers, which share the stimulus statistics within a group)."""
    S = sessions().query("extracted").set_index("id")
    lab = pd.read_csv(WORK / "labels.csv")
    rows = []
    for sid in S.index:
        if ls == "tag" and S.loc[sid, "line"] == "wt":
            continue
        b = np.load(WORK / "spk" / f"{sid}.npz")
        for p in b["probes"]:
            idx = np.where(b["unit_probe"] == p)[0]
            rows.append(pd.DataFrame(dict(session=sid, probe=p, unit_id=b["unit_id"][idx], row=idx)))
    neu = pd.concat(rows, ignore_index=True).merge(
        lab[["unit_id", "area", "line", "fold", "tagged", "fs", "label_tag", "label_fs", "waveform_duration",
             "ratio", "p_wilcoxon", "probe_vertical_position"]], on="unit_id", how="left", validate="1:1")
    neu = neu[neu.area.isin(SCOPES[scope])].reset_index(drop=True)
    neu["label"] = neu[f"label_{ls}"]
    neu["t"] = neu.label.map({l: k for k, l in enumerate(LABELSETS[ls])})
    assert neu.t.notna().all()
    neu["t"] = neu.t.astype(int)
    neu["gi"] = pd.factorize(list(zip(neu.session, neu.probe)))[0]
    neu["si"] = neu.gi
    neu["neuron"] = neu.groupby("gi").cumcount()
    return neu


def cdir(ds, scope, ls):
    d = WORK / ds / f"{scope}_{ls}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def fake_labels(neu, seed):
    """Shuffled-label control: true labels permuted within each session (same counts)."""
    rng = np.random.default_rng(seed)
    lab = neu.t.to_numpy().copy()
    for s in neu.session.unique():
        m = (neu.session == s).to_numpy()
        lab[m] = rng.permutation(lab[m])
    return lab


def load_grams(ds, neu):
    G, per = {}, {}
    for (sid, p), g in neu.groupby(["session", "probe"], sort=False):
        z = np.load(WORK / ds / "gram" / f"{sid}.npz")
        b = np.load(WORK / "spk" / f"{sid}.npz")
        idx_p = np.where(b["unit_probe"] == p)[0]
        pos = pd.Series(np.arange(len(idx_p)), index=b["unit_id"][idx_p]).loc[g.unit_id].to_numpy()
        for k in z.files:
            pp, key = k.split("|")
            if pp != p:
                continue
            if key.split("_", 1)[1] in ["SS", "sS", "n"]:
                per.setdefault(key, []).append(z[k])
            else:
                G.setdefault(key, []).append(z[k][pos].astype(np.float32))  # per-unit statistics in float32
    G = {k: np.concatenate(v) for k, v in G.items()}
    for k, v in per.items():
        G[k] = np.stack(v)
    return G


def test_data(ds, neu, gi):
    g = neu[neu.gi == gi]
    sid, p = g.session.iloc[0], g.probe.iloc[0]
    b, _, Zte = session_arrays(ds, sid)
    te = b[f"te_{p}"]
    Z = Zte[g.row.to_numpy()][:, te[:, None] + OFF].transpose(1, 0, 2).astype(np.float64)
    S = b["Ste"][:, te[:, None] + OFF].transpose(1, 0, 2).reshape(len(te), -1).astype(np.float64)
    return Z[..., :P], Z[..., P:], S


# ---------------------------------------------------------------- clusters
def cluster_fold(args):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    k_fold, X, tr = args
    with threadpool_limits(1):
        mu, sd = np.nanmean(X[tr], 0), np.nanstd(X[tr], 0)
        sd[~(sd > 0)] = 1
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
        pca = PCA(n_components=0.95, svd_solver="full").fit(Z[tr])
        Pc = pca.transform(Z)
        lab = {k: KMeans(n_clusters=k, n_init=10, random_state=k_fold).fit(Pc[tr]).predict(Pc) for k in KS}
    print(f"fold {k_fold}: {pca.n_components_} PCs; smallest training cluster at k=16:"
          f" {np.bincount(lab[16][tr], minlength=16).min()}", flush=True)
    return k_fold, dict(lab=lab, Z=Z.astype(np.float32), n_pc=pca.n_components_)


def cluster(ds, scope, ls):
    neu = neurons(scope, ls)
    f = pd.read_csv(WORK / ds / "features.csv")
    X = neu[["unit_id"]].merge(f, on="unit_id", how="left", validate="1:1").iloc[:, 2:]
    names, X = list(X.columns), X.to_numpy(np.float64)
    with Pool(N_FOLDS) as pool:
        res = dict(pool.map(cluster_fold, [(k, X, (neu.fold != k).to_numpy()) for k in range(N_FOLDS)]))
    pickle.dump(dict(res=res, names=names, nan_frac=np.isnan(X).mean(0)), open(cdir(ds, scope, ls) / "clusters.pkl", "wb"))


# ---------------------------------------------------------------- ridge (task 2)
PER_UNIT = ["xx", "xS", "xy", "Sy", "yy", "sx", "sy"]


def assemble_chunked(G, si, w, part):
    """step12_bugeon.assemble with per-unit statistics stored in float32 and the
    weighted sums accumulated in float64 over chunks of units (memory: the all-area
    configurations hold ~17k units)."""
    p = lambda k: G[f"{part}_{k}"]
    ws = np.bincount(si, w, minlength=len(p("n")))
    nz = np.where(w != 0)[0]
    acc = {k: 0.0 for k in PER_UNIT}
    for a in range(0, len(nz), 2048):
        ii = nz[a:a + 2048]
        for k in PER_UNIT:
            acc[k] = acc[k] + np.tensordot(w[ii], p(k)[ii].astype(np.float64), 1)
    xS = acc["xS"]
    XX = np.block([[acc["xx"], xS], [xS.T, np.tensordot(ws, p("SS"), 1)]])
    XY = np.vstack([acc["xy"], acc["Sy"]])
    sx = np.concatenate([acc["sx"], ws @ p("sS")])
    return s12.Gram(ws @ p("n"), sx, acc["sy"], XX, XY), acc["yy"]


def val_sse_chunked(G, si, idx, W, b):
    """step13 val_sse (same formula, per unit, so chunking over units is exact) on
    float64 copies of the float32 statistics."""
    out = []
    for a in range(0, len(idx), 2048):
        ii = idx[a:a + 2048]
        sub = {k: (v[ii].astype(np.float64) if k[2:] in PER_UNIT else v) for k, v in G.items() if k.startswith("v_")}
        out.append(_val_sse_sub(sub, si[ii], W, b))
    return np.concatenate(out)


def _val_sse_sub(v_, s, W, b):
    v = lambda k: v_[f"v_{k}"]
    Wx, Ws = W[:P], W[P:]
    out = (np.einsum("npq,pf,qf->n", v("xx"), Wx, Wx, optimize=True)
           + 2 * np.einsum("npf,pf->n", v("xS") @ Ws, Wx, optimize=True)
           + 2 * (v("sx") @ Wx @ b) - 2 * np.einsum("npf,pf->n", v("xy"), Wx, optimize=True)
           - 2 * np.einsum("nsf,sf->n", v("Sy"), Ws, optimize=True) - 2 * (v("sy") @ b)
           + np.trace(v("yy"), axis1=1, axis2=2))
    per_sess = np.array([np.einsum("sf,st,tf->", Ws, SS, Ws) + 2 * sS @ Ws @ b + n * b @ b
                         for SS, sS, n in zip(v("SS"), v("sS"), v("n"))])
    return out + per_sess[s]


def val_copy_sse_chunked(G, idx):
    out = []
    for a in range(0, len(idx), 2048):
        ii = idx[a:a + 2048]
        v = lambda k: G[f"v_{k}"][ii].astype(np.float64)
        out.append(np.trace(v("yy"), axis1=1, axis2=2) - 2 * v("xy")[:, P - 1, :].sum(1) + F * v("xx")[:, P - 1, P - 1])
    return np.concatenate(out)


def ridge(ds, scope, ls):
    neu = neurons(scope, ls)
    d = cdir(ds, scope, ls)
    G = load_grams(ds, neu)
    for k in list(G):
        if k[2:] in PER_UNIT:
            G[k] = G[k].astype(np.float32)
    s12.assemble, s13.val_sse, s13.val_copy_sse = assemble_chunked, val_sse_chunked, val_copy_sse_chunked
    si = neu.si.to_numpy()
    neu["rows"] = G["f_n"][si] + G["v_n"][si]
    t_arr = neu.t.to_numpy()
    groups = sorted(np.unique(t_arr))
    CL = pickle.load(open(d / "clusters.pkl", "rb"))["res"]
    fakes = [fake_labels(neu, seed) for seed in range(N_E)]
    for k in range(N_FOLDS):
        if (d / f"ridge_sse_fold{k}.csv").exists():
            continue
        t0 = time.time()
        held = (neu.fold == k).to_numpy()
        tr = ~held
        b = s12.fit_weighted(G, si, s12.balanced_weights(neu, tr))
        dm = {c: s12.fit_weighted(G, si, (tr & (t_arr == c)).astype(float)) for c in groups if (tr & (t_arr == c)).any()}
        e = [(fk, {g: s12.fit_weighted(G, si, s12.balanced_weights(neu, tr & (fk == g))) for g in groups
                   if (tr & (fk == g)).any()}) for fk in fakes]
        # equal-weight shuffled control (one model per fake group, equal row weights, as d / k1)
        e_eq = [(fk, {g: s12.fit_weighted(G, si, (tr & (fk == g)).astype(float)) for g in groups
                      if (tr & (fk == g)).any()}) for fk in fakes]
        one = s13.fit_group(G, si, tr.astype(float))
        # k on val (step 13): fit-part cluster models, val score of the training sessions
        itr = np.where(tr)[0]
        cp = s13.val_copy_sse(G, itr)
        bb = s13.fit_group(G, si, s12.balanced_weights(neu, tr))
        val_rows = [dict(fold=k, k="b (none)", val_score=s13.unit_val_score(neu, itr, s13.val_sse(G, si, itr, bb["Wf"], bb["bf"]), cp)),
                    dict(fold=k, k="1 (one model, equal weights)",
                         val_score=s13.unit_val_score(neu, itr, s13.val_sse(G, si, itr, one["Wf"], one["bf"]), cp))]
        models = {}
        for kk in KS:
            lab = CL[k]["lab"][kk]
            ms = s13.fit_clusters(G, si, neu, lab, tr)
            models[kk] = ms
            sse = np.zeros(len(itr))
            for c, mod in ms.items():
                sel = lab[itr] == c
                sse[sel] = s13.val_sse(G, si, itr[sel], mod["Wf"], mod["bf"])
            val_rows.append(dict(fold=k, k=kk, val_score=s13.unit_val_score(neu, itr, sse, cp)))
        vr = pd.DataFrame(val_rows)
        kv = vr[vr.k.isin(KS)]
        k_sel = int(kv.k.iloc[int(np.argmax(kv.val_score.to_numpy()))])
        vr["chosen"] = vr.k == k_sel
        rows = []
        for gi in sorted(neu[held].gi.unique()):
            X, Y, S = test_data(ds, neu, gi)
            rr = np.where(si == gi)[0]
            err = lambda Pr: ((Pr - Y) ** 2).sum(axis=(0, 2))
            r = dict(copy=err(np.repeat(X[:, :, -1:], F, axis=2)), r_none=err(s12.predict(b[0], b[1], X, S)))
            Dm = {c: err(s12.predict(m[0], m[1], X, S)) for c, m in dm.items()}
            for c in groups:
                r[f"d_{LABELSETS[ls][c]}"] = Dm.get(c, r["r_none"])
            r["r_true"] = np.choose(t_arr[rr], [Dm.get(c, r["r_none"]) for c in range(len(LABELSETS[ls]))])
            se = []
            for fk, ms in e:
                Pg = {g: err(s12.predict(m[0], m[1], X, S)) for g, m in ms.items()}
                se.append(np.choose(fk[rr], [Pg.get(g, r["r_none"]) for g in range(len(LABELSETS[ls]))]))
            r["r_e"] = np.mean(se, axis=0)
            se = []
            for fk, ms in e_eq:
                Pg = {g: err(s12.predict(m[0], m[1], X, S)) for g, m in ms.items()}
                se.append(np.choose(fk[rr], [Pg.get(g, r["r_none"]) for g in range(len(LABELSETS[ls]))]))
            r["r_e_eq"] = np.mean(se, axis=0)
            r["r_k1"] = err(s12.predict(one["W"], one["b"], X, S))
            for name, kk in [("r_km16", K_MAIN), ("r_kmsel", k_sel)]:
                lab = CL[k]["lab"][kk][rr]
                o = np.empty(len(rr))
                for c in np.unique(lab):
                    mod = models[kk].get(c)
                    W_, b_ = (mod["W"], mod["b"]) if mod else b[:2]
                    o[lab == c] = err(s12.predict(W_, b_, X, S))[lab == c]
                r[name] = o
            rows.append(pd.DataFrame(dict(session=neu.session.to_numpy()[rr], unit_id=neu.unit_id.to_numpy()[rr],
                                          n_el=Y.shape[0] * F, **{f"sse_{m}": v for m, v in r.items()})))
        al = dict(fold=k, k_sel=k_sel, b=b[2], k1=one["alpha"], **{f"d_{LABELSETS[ls][c]}": m[2] for c, m in dm.items()},
                  e=";".join(f"{m[2]:g}" for _, ms in e for m in ms.values()))
        pd.DataFrame([al]).to_csv(d / f"ridge_alphas_fold{k}.csv", index=False)
        vr.to_csv(d / f"val_k_fold{k}.csv", index=False)
        pd.concat(rows).to_csv(d / f"ridge_sse_fold{k}.csv", index=False, float_format="%.9g")
        print(f"{ds} {scope} {ls} fold {k}: alpha b {b[2]:g}, k_sel {k_sel} ({time.time() - t0:.0f} s)", flush=True)


# ---------------------------------------------------------------- MLP (task 2)
N_IN = None
GRP = []          # per (session, probe) group: Ztr / Zte rows of the configuration, Str / Ste, windows
NEU = None
COND = {}


class MLP16(s9.MLP):
    """step9_mlp.MLP (2 x 256, zero-initialised FiLM from a label embedding) with this
    step's input size (590) and output size (10)."""

    def __init__(self, n_labels=0):
        super().__init__(n_labels)
        self.l1, self.out = nn.Linear(N_IN, s9.HIDDEN), nn.Linear(s9.HIDDEN, F)


class Cond16(s13.CondMixin, MLP16):
    pass


def load_mlp(ds, scope, ls):
    global NEU, N_IN
    NEU = neurons(scope, ls)
    for gi, g in NEU.groupby("gi", sort=True):
        b, Ztr, Zte = session_arrays(ds, g.session.iloc[0])
        p, r = g.probe.iloc[0], g.row.to_numpy()
        GRP.append(dict(Ztr=np.ascontiguousarray(Ztr[r]), Zte=np.ascontiguousarray(Zte[r]), Str=b["Str"], Ste=b["Ste"],
                        fit=b[f"fit_{p}"], val=b[f"val_{p}"], te=b[f"te_{p}"]))
    N_IN = P + GRP[0]["Str"].shape[0] * (P + F)
    CL = pickle.load(open(cdir(ds, scope, ls) / "clusters.pkl", "rb"))["res"]
    for k, C in CL.items():
        COND[k] = C["Z"]


def gather(gi, neur, t):
    g, idx = GRP[gi], t[:, None] + OFF
    z = g["Ztr"][neur[:, None], idx]
    st = g["Str"][:, idx].transpose(1, 0, 2).reshape(len(t), -1)
    return np.hstack([z[:, :P], st]), z[:, P:]


class Sampler:
    """step 12 Sampler: every batch has the same number of rows of each label group;
    within a group rows are uniform over (unit, window) of the given units. With
    uniform=True there is one group (rows uniform over all units' windows)."""

    def __init__(self, rows, part, rng, uniform=False):
        self.rng, self.part = rng, part
        sub = NEU.iloc[rows]
        self.by_type = {}
        for t in ([None] if uniform else sorted(sub.t.unique())):
            r = sub if t is None else sub[sub.t == t]
            w = np.array([len(GRP[gi][part]) for gi in r.gi], float)
            self.by_type[t] = (r.gi.to_numpy(), r.neuron.to_numpy(), r.index.to_numpy(), w / w.sum())

    def draw(self, n_per_type):
        xs, ys, rs = [], [], []
        for t, (gis, ne, ix, p) in self.by_type.items():
            pick = self.rng.choice(len(gis), size=n_per_type, p=p)
            for g in np.unique(gis[pick]):
                m = pick[gis[pick] == g]
                st = GRP[g][self.part]
                x, y = gather(g, ne[m], st[self.rng.integers(len(st), size=len(m))])
                xs.append(x); ys.append(y); rs.append(ix[m])
        return np.vstack(xs), np.vstack(ys), np.concatenate(rs)


def call(model, x, lab, cont, rows):
    L = None if lab is None else torch.from_numpy(lab[rows].astype(np.int64))
    if cont is None:
        return model(x, L)
    return model(x, L, torch.from_numpy(cont[rows]))


def forecast(model, gi, lab, cont):
    g = GRP[gi]
    rows = np.where(NEU.gi.to_numpy() == gi)[0]
    W = g["Zte"][:, g["te"][:, None] + OFF].transpose(1, 0, 2)
    S = g["Ste"][:, g["te"][:, None] + OFF].transpose(1, 0, 2).reshape(len(g["te"]), -1)
    X, Y = W[..., :P], W[..., P:]
    nw, n = X.shape[:2]
    Pr = np.empty_like(Y)
    with torch.no_grad():
        for a in range(0, nw, 64):
            Xc, Sc = X[a:a + 64], S[a:a + 64]
            D = torch.from_numpy(np.concatenate([Xc.reshape(-1, P), np.repeat(Sc, n, axis=0)], axis=1))
            Pr[a:a + 64] = call(model, D, lab, cont, np.tile(rows, len(Xc))).numpy().reshape(len(Xc), n, F)
    return ((Pr - Y) ** 2).sum(axis=(0, 2)).astype(np.float64)


def run_job(job):
    """step 9 / 12 / 13 run_job (optimiser, label-balanced batches, epochs, early
    stopping, seeding formula); returns the per-unit test SSE of the held-out units
    (m2: under every label value)."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, kind, seed = job["fold"], job["kind"], job["seed"]
    rng = np.random.default_rng(1000 * fold + seed + SEED_OFF[kind] * 10)
    torch.manual_seed(int(rng.integers(2 ** 31)))
    T = NEU.t.to_numpy()
    nl = int(T.max()) + 1
    base = kind.rstrip("u")
    uni = kind.endswith("u")
    lab, cont, n_lab = {"m1": (None, None, 0), "m2": (T, None, nl),
                        "m3": (fake_labels(NEU, seed) if base == "m3" else None, None, nl),
                        "cont": (None, COND[fold], 0)}[base]
    model = MLP16(n_lab) if cont is None else Cond16(n_lab, cont.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=s9.LR)
    tr = np.where(NEU.fold.to_numpy() != fold)[0]
    smp = Sampler(tr, "fit", rng, uni)
    n_t = len(smp.by_type)
    # val rows: label-balanced (as step 12) or, for the uniform variants, uniform (same total size)
    xv, yv, rv = Sampler(tr, "val", np.random.default_rng(fold), uni).draw(s9.N_VAL * (nl if uni else 1))
    xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    v_type = T[rv] if not uni else np.zeros(len(rv), int)

    def val_loss():
        model.eval()
        with torch.no_grad():
            err = ((call(model, xv, lab, cont, rv) - yv) ** 2).mean(1).numpy()
        model.train()
        return float(np.mean([err[v_type == t].mean() for t in np.unique(v_type)]))

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
    out = []
    for gi in sorted(NEU[NEU.fold == fold].gi.unique()):
        rows = np.where(NEU.gi.to_numpy() == gi)[0]
        r = dict(unit_id=NEU.unit_id.to_numpy()[rows], sse=forecast(model, gi, lab, cont))
        if base == "m2":
            for c in range(nl):
                r[f"sse_as_{c}"] = forecast(model, gi, np.full(len(NEU), c), None)
        out.append(pd.DataFrame(r))
    out = pd.concat(out)
    for k_, v in job.items():
        out.insert(0, k_, v)
    return dict(job=job, out=out, epochs=len(hist), best_val=best, hist=hist, secs=time.time() - t0)


def mlp(ds, scope, ls, which="balanced"):
    d = cdir(ds, scope, ls)
    kinds = BALANCED_KINDS if which == "balanced" else UNIFORM_KINDS
    load_mlp(ds, scope, ls)
    fits = d / "mlp_fits.csv"
    have = set()
    if fits.exists():
        fd = pd.read_csv(fits)
        have = set(zip(fd.fold, fd.kind, fd.seed))
    jobs = [dict(fold=k, kind=kind, seed=seed) for k in range(N_FOLDS) for kind in kinds
            for seed in range(N_SEEDS[kind]) if (k, kind, seed) not in have]
    print(f"{ds} {scope} {ls}: {len(jobs)} MLP fits to run ({len(have)} done), {len(NEU)} units, {N_IN} inputs,"
          f" groups {NEU.label.value_counts().to_dict()}", flush=True)
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            j = r["job"]
            sse = d / f"mlp_sse_{j['kind']}.csv"
            r["out"].to_csv(sse, mode="a", header=not sse.exists(), index=False, float_format="%.9g")
            pd.DataFrame([dict(**j, epochs=r["epochs"], best_val=r["best_val"], secs=r["secs"],
                               hist=";".join(f"{h:.5f}" for h in r["hist"]))]) \
                .to_csv(fits, mode="a", header=not fits.exists(), index=False)
            print(f"[{i + 1}/{len(jobs)}] fold {j['fold']} {j['kind']} seed {j['seed']}: {r['epochs']} epochs,"
                  f" best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)


# ---------------------------------------------------------------- main
def main():
    a = sys.argv[1:]
    ph = a[0]
    if ph == "labels":
        labels()
    elif ph == "bin":
        binning()
    elif ph == "gram":
        gram(a[1])
    elif ph == "features":
        features(a[1])
    elif ph in ("cluster", "ridge"):
        {"cluster": cluster, "ridge": ridge}[ph](a[1], a[2], a[3])
    elif ph == "mlp":
        mlp(a[1], a[2], a[3], a[4] if len(a) > 4 else "balanced")
    elif ph == "analyse":
        import step16_analyse
        step16_analyse.main()
    else:
        raise SystemExit(f"unknown phase {ph}")


if __name__ == "__main__":
    main()
