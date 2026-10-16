"""Step 9: does a neural network learn cell type by itself?

Same 46 sessions (VISp, three_session_A, Cux2 / Sst / Vip / Pvalb, one per mouse),
same 5 folds, inputs and test windows as step 8 (scripts/step8_medium.py), read
from the step 8 cache in data/cache/. Inputs per neuron: own past 30 bins + 480
stimulus inputs (12 features x past 30 + next 10 bins) -> next 10 bins.

Models (PyTorch, CPU), all an MLP 510 -> 256 -> 256 -> 10 with ReLU:
- m1: shared, no label
- m2: + Cre line via FiLM. A learned Cre embedding (dim 16, N(0, 0.02) init as
  POYO's unit/session embeddings) gives a scale and shift for each hidden layer
  (after the ReLU): h * (1 + gamma) + beta, gamma/beta linear in the embedding
  and zero-initialised as in POCO, so m2 starts as m1
- m3: as m2 with shuffled labels (control): training sessions in 4 random
  equal-size groups that each mix lines (step 8 e grouping, same seeds); a
  held-out mouse gets a random group. 5 seeds
- m4: one MLP per Cre line (trained on that line's mice only)
m1, m2, m4: 3 seeds (init and sampling). Reference rows: step 8 ridge b
(line-balanced shared) and d (per line), read from data/cache/step8_mice.csv.

Training: rows (session, neuron, window start) are sampled on the fly from the
train part (first 90% of the 80% train split, stride 1, as step 8). Line
balance as step 8 b: every batch has the same number of rows from each Cre line,
rows uniform within a line (m4: uniform over its line). Adam lr 1e-3, batch
1024, epoch = 400 batches, at most 40 epochs; early stopping on the val part
(last 10% of train; fixed sample of 20,000 val rows per line from the training
sessions, line-balanced MSE), patience 5, best epoch restored. Test windows =
last 20% of each held-out session, stride 10 bins, as steps 5-8.

Metrics as step 8: MSE, mean per-neuron r, score = 1 - MSE / MSE_copylast,
pooled over the held-out mice of a line in a fold, mean +- std over folds.
Seeds: per mouse and per fold the MSE is averaged over seeds.
Jobs (one model fit each) run in N_WORKERS processes with 1 torch thread each.
"""
import os
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from step3_linear import FUTURE, PAST, ROOT, window_starts, windows
from step4_stimulus import stim_windows
from step5_shared import neuron_r
from step8_medium import CRE_LINES, N_FOLDS, NPZ, SHORT, pick_sessions

warnings.filterwarnings("ignore")

N_LINES = len(CRE_LINES)
SEEDS = {"m1": 3, "m2": 3, "m3": 5, "m4": 3}
HIDDEN, EMB = 256, 16
BATCH, EPOCH_BATCHES, MAX_EPOCHS, PATIENCE, LR = 1024, 400, 40, 5, 1e-3
N_VAL = 20000  # val rows per line
NAMES = {"rb": "ridge b (step 8, balanced)", "rd": "ridge d (step 8, per line)",
         "m1": "m1 MLP shared", "m2": "m2 MLP + Cre FiLM", "m3": "m3 MLP + shuffled FiLM",
         "m4": "m4 MLP per Cre line"}
MODELS = ["rb", "rd", "m1", "m2", "m3", "m4"]
OFF = np.arange(-PAST, FUTURE)
SESS = []  # filled before the worker pool forks


def load():
    for info in pick_sessions():
        z = np.load(NPZ / f"{info['id']}.npz")
        n_fit, ztr = int(z["n_fit"]), z["ztr"]
        SESS.append(dict(info=info, line=CRE_LINES.index(info["cre"]), ztr=ztr, Str=z["Str_all"],
                         zte=z["zte"], Ste=z["Ste_all"], n_fit=n_fit, n=int(z["n_neurons"]),
                         # window starts: train part [PAST, n_fit - FUTURE], val part
                         # [n_fit + PAST, n_tr - FUTURE] (val windows lie inside the val part)
                         tr=(PAST, n_fit - FUTURE + 1), va=(n_fit + PAST, ztr.shape[1] - FUTURE + 1)))


def gather(si, neur, t):
    """Rows for session si, neurons neur and window starts t -> x (B, 510), y (B, 10)."""
    s, idx = SESS[si], t[:, None] + OFF
    z = s["ztr"][neur[:, None], idx]
    st = s["Str"][:, idx].transpose(1, 0, 2).reshape(len(t), -1)
    return np.hstack([z[:, :PAST], st]), z[:, PAST:]


class Sampler:
    """Draws rows from the given sessions; per batch equal counts per Cre line,
    uniform within a line (session chosen in proportion to neurons x windows)."""

    def __init__(self, sess_idx, part, rng):
        self.rng, self.part = rng, part
        self.lines = sorted({SESS[i]["line"] for i in sess_idx})
        self.by_line = {}
        for l in self.lines:
            ids = np.array([i for i in sess_idx if SESS[i]["line"] == l])
            w = np.array([SESS[i]["n"] * (SESS[i][part][1] - SESS[i][part][0]) for i in ids], float)
            self.by_line[l] = (ids, w / w.sum())

    def draw(self, n_per_line):
        xs, ys, ss = [], [], []
        for l in self.lines:
            ids, p = self.by_line[l]
            pick = self.rng.choice(len(ids), size=n_per_line, p=p)
            for k in np.unique(pick):
                m = int((pick == k).sum())
                si = int(ids[k])
                lo, hi = SESS[si][self.part]
                x, y = gather(si, self.rng.integers(SESS[si]["n"], size=m), self.rng.integers(lo, hi, size=m))
                xs.append(x); ys.append(y); ss.append(np.full(m, si))
        return np.vstack(xs), np.vstack(ys), np.concatenate(ss)


class MLP(nn.Module):
    """2 hidden layers; optional FiLM (POCO-style, zero-initialised) from a label embedding."""

    def __init__(self, n_labels=0):
        super().__init__()
        self.l1, self.l2 = nn.Linear(PAST + 12 * (PAST + FUTURE), HIDDEN), nn.Linear(HIDDEN, HIDDEN)
        self.out = nn.Linear(HIDDEN, FUTURE)
        self.film = n_labels > 0
        if self.film:
            self.emb = nn.Embedding(n_labels, EMB)
            nn.init.normal_(self.emb.weight, 0, 0.02)
            self.gb = nn.Linear(EMB, 4 * HIDDEN)  # gamma1, beta1, gamma2, beta2
            nn.init.zeros_(self.gb.weight); nn.init.zeros_(self.gb.bias)

    def forward(self, x, lab=None):
        h1, h2 = torch.relu(self.l1(x)), None
        if self.film:
            g1, b1, g2, b2 = self.gb(self.emb(lab)).chunk(4, dim=1)
            h1 = h1 * (1 + g1) + b1
            h2 = torch.relu(self.l2(h1)) * (1 + g2) + b2
        else:
            h2 = torch.relu(self.l2(h1))
        return self.out(h2)


def run_job(job):
    """Fit one model; returns per held-out mouse SSE and per-neuron r on the test windows."""
    torch.set_num_threads(1)
    t0 = time.time()
    fold, kind, seed, line = job["fold"], job["kind"], job["seed"], job.get("line")
    rng = np.random.default_rng(1000 * fold + seed + 100 * (line or 0) + {"m1": 0, "m2": 1, "m3": 2, "m4": 3}[kind] * 10)
    torch.manual_seed(int(rng.integers(2 ** 31)))
    train = [i for i, s in enumerate(SESS) if s["info"]["fold"] != fold and (line is None or s["line"] == line)]
    held = [i for i, s in enumerate(SESS) if s["info"]["fold"] == fold and (line is None or s["line"] == line)]
    lab = job.get("labels")  # session index -> label for FiLM (m2: true line, m3: fake group)
    model = MLP(N_LINES if lab is not None else 0)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    tr = Sampler(train, "tr", rng)
    n_line = len(tr.lines)
    xv, yv, sv = Sampler(train, "va", np.random.default_rng(fold)).draw(N_VAL)
    xv, yv = torch.from_numpy(xv), torch.from_numpy(yv)
    lv = torch.from_numpy(np.array([lab[i] for i in sv])) if lab is not None else None
    v_line = np.array([SESS[i]["line"] for i in sv])

    def val_loss():
        model.eval()
        with torch.no_grad():
            err = ((model(xv, lv) - yv) ** 2).mean(1).numpy()
        model.train()
        return float(np.mean([err[v_line == l].mean() for l in tr.lines]))

    best, best_state, bad, hist = val_loss(), None, 0, []
    for ep in range(MAX_EPOCHS):
        for _ in range(EPOCH_BATCHES):
            x, y, s = tr.draw(BATCH // n_line)
            l = torch.from_numpy(np.array([lab[i] for i in s])) if lab is not None else None
            loss = ((model(torch.from_numpy(x), l) - torch.from_numpy(y)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        v = val_loss()
        hist.append(v)
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    out = {}
    for si in held:
        s = SESS[si]
        starts = window_starts(s["zte"].shape[1], FUTURE)
        X, Y = windows(s["zte"], starts)
        S = stim_windows(s["Ste"], starts)
        nw, nn_ = X.shape[:2]
        P = np.empty_like(Y)
        with torch.no_grad():
            for a in range(0, nw, 64):  # chunks of 64 windows x all neurons
                Xc, Sc = X[a:a + 64], S[a:a + 64]
                D = np.concatenate([Xc.reshape(-1, PAST), np.repeat(Sc, nn_, axis=0)], axis=1)
                L = torch.full((len(D),), lab[si], dtype=torch.long) if lab is not None else None
                P[a:a + 64] = model(torch.from_numpy(D), L).numpy().reshape(len(Xc), nn_, FUTURE)
        out[s["info"]["id"]] = dict(sse=float(((P - Y) ** 2).sum()), r=neuron_r(Y, P))
    return dict(job={k: v for k, v in job.items() if k != "labels"}, out=out, epochs=len(hist),
                best_val=best, secs=time.time() - t0)


def shuffled_groups(train, rng):
    """Step 8 e grouping: random equal-size groups, each mixing Cre lines."""
    while True:
        groups = np.array_split(rng.permutation(len(train)), N_LINES)
        if all(len({SESS[train[i]]["line"] for i in g}) > 1 for g in groups):
            return {train[i]: k for k, g in enumerate(groups) for i in g}


def make_jobs():
    jobs = []
    true = {i: s["line"] for i, s in enumerate(SESS)}
    for fold in range(N_FOLDS):
        train = [i for i, s in enumerate(SESS) if s["info"]["fold"] != fold]
        held = [i for i, s in enumerate(SESS) if s["info"]["fold"] == fold]
        for seed in range(SEEDS["m1"]):
            jobs.append(dict(fold=fold, kind="m1", seed=seed))
        for seed in range(SEEDS["m2"]):
            jobs.append(dict(fold=fold, kind="m2", seed=seed, labels=true))
        for seed in range(SEEDS["m3"]):
            # same draws as step 8 e: train groups, then a random group per held-out session
            rng = np.random.default_rng(seed)
            lab = shuffled_groups(train, rng)
            for i in held:
                lab[i] = int(rng.integers(N_LINES))
            jobs.append(dict(fold=fold, kind="m3", seed=seed, labels=lab))
        for line in range(N_LINES):
            for seed in range(SEEDS["m4"]):
                jobs.append(dict(fold=fold, kind="m4", seed=seed, line=line))
    return jobs


def main():
    load()
    print(f"{len(SESS)} sessions, {sum(s['n'] for s in SESS)} neurons; "
          + ", ".join(f"{SHORT[c]} {sum(s['info']['cre'] == c for s in SESS)}" for c in CRE_LINES), flush=True)
    jobs = make_jobs()
    # Longest jobs first (m1-m3 on all lines), so the pool finishes evenly
    jobs.sort(key=lambda j: j["kind"] == "m4")
    print(f"{len(jobs)} model fits in {os.environ.get('N_WORKERS', 4)} workers", flush=True)
    results = []
    with Pool(int(os.environ.get("N_WORKERS", 4))) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            results.append(r)
            j = r["job"]
            print(f"[{len(results)}/{len(jobs)}] fold {j['fold']} {j['kind']} seed {j['seed']}"
                  f"{' line ' + SHORT[CRE_LINES[j['line']]] if j.get('line') is not None else ''}:"
                  f" {r['epochs']} epochs, best val {r['best_val']:.4f}, {r['secs']:.0f} s", flush=True)

    # Per held-out mouse: SSE per model (mean over seeds), copy-last SSE, n elements, r
    sid = {s["info"]["id"]: s for s in SESS}
    base = {}
    for s in SESS:
        starts = window_starts(s["zte"].shape[1], FUTURE)
        X, Y = windows(s["zte"], starts)
        base[s["info"]["id"]] = dict(n_el=Y.size, sse_copy=float(((X[:, :, -1:] - Y) ** 2).sum()))
    per = {}  # (mouse id, model) -> list over seeds of (sse, r array)
    for r in results:
        for mid, v in r["out"].items():
            per.setdefault((mid, r["job"]["kind"]), []).append(v)
    step8 = pd.read_csv(NPZ / "step8_mice.csv").set_index("id")
    rows = []
    for mid, s in sid.items():
        b = base[mid]
        d = dict(cre=s["info"]["cre"], fold=s["info"]["fold"], id=mid, depth=s["info"]["depth"],
                 n_neurons=s["n"], n_el=b["n_el"], sse_copy=b["sse_copy"])
        for m in ["m1", "m2", "m3", "m4"]:
            vs = per[mid, m]
            assert len(vs) == SEEDS[m], (mid, m, len(vs))
            d[f"sse_{m}"] = float(np.mean([v["sse"] for v in vs]))
            d[f"r_{m}"] = float(np.mean([np.nanmean(v["r"]) for v in vs]))
            d[f"mse_{m}_seeds"] = ";".join(f"{v['sse'] / b['n_el']:.5f}" for v in vs)
        for m, col in [("rb", "b"), ("rd", "d")]:
            d[f"sse_{m}"] = step8.loc[mid, f"mse_{col}"] * b["n_el"]
            d[f"r_{m}"] = step8.loc[mid, f"r_{col}"]
        for m in MODELS:
            d[f"mse_{m}"] = d[f"sse_{m}"] / b["n_el"]
            d[f"score_{m}"] = 1 - d[f"sse_{m}"] / b["sse_copy"]
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(NPZ / "step9_mice.csv", index=False)

    # Fold-level pooled metrics per line (score from summed SSE over held-out mice)
    groups = CRE_LINES + ["all lines"]
    fold_score = {}
    print(f"\n== Summary over {N_FOLDS} folds (mean +- std; held-out mice pooled per fold) ==")
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        print(cre)
        for m in MODELS:
            sc = np.array([1 - sub[sub.fold == k][f"sse_{m}"].sum() / sub[sub.fold == k].sse_copy.sum()
                           for k in range(N_FOLDS)])
            mse = np.array([sub[sub.fold == k][f"sse_{m}"].sum() / sub[sub.fold == k].n_el.sum()
                            for k in range(N_FOLDS)])
            rr = np.array([np.average(sub[sub.fold == k][f"r_{m}"], weights=sub[sub.fold == k].n_neurons)
                           for k in range(N_FOLDS)])
            fold_score[cre, m] = sc
            print(f"    {NAMES[m]:28s} MSE {mse.mean():.4f}+-{mse.std():.4f}  r {rr.mean():.3f}+-{rr.std():.3f}"
                  f"  score {sc.mean():+.3f}+-{sc.std():.3f}  per fold {np.round(sc, 3)}")

    print("\n== Paired comparisons over held-out mice (MSE; 'A beats B' = lower MSE) ==")
    pairs = [("m2", "m1"), ("m2", "m3"), ("m4", "m1"), ("m4", "m2"), ("m1", "rb"), ("rd", "rb")]
    stats = {}
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        line = []
        for a, b in pairs:
            diff = sub[f"mse_{a}"] - sub[f"mse_{b}"]
            p = wilcoxon(sub[f"mse_{a}"], sub[f"mse_{b}"]).pvalue
            stats[cre, a, b] = ((diff < 0).sum(), len(sub), diff.median(), p)
            line.append(f"{a} vs {b}: {(diff < 0).sum()}/{len(sub)}, median {diff.median():+.4f}, p {p:.2g}")
        print(f"  {cre}\n    " + "\n    ".join(line))

    # Key question: cell-type gain with MLP vs with ridge (per mouse, MSE reduction)
    print("\n== Cell-type gain per mouse: ridge (b - d) vs MLP (m1 - m2) and (m1 - m4) ==")
    for cre in groups:
        sub = df if cre == "all lines" else df[df.cre == cre]
        g_r = sub.mse_rb - sub.mse_rd
        g_r_rel = g_r / sub.mse_rb
        out = [f"ridge b-d median {g_r.median():+.4f} (rel {g_r_rel.median():+.3f})"]
        for m in ["m2", "m4"]:
            g = sub.mse_m1 - sub[f"mse_{m}"]
            g_rel = g / sub.mse_m1
            p = wilcoxon(g, g_r).pvalue
            out.append(f"MLP m1-{m} median {g.median():+.4f} (rel {g_rel.median():+.3f}); MLP gain > ridge gain in"
                       f" {(g > g_r).sum()}/{len(sub)}, Wilcoxon p {p:.2g}")
        print(f"  {cre}: " + " | ".join(out))

    print("\nPer held-out mouse (MSE):")
    print(df[["cre", "fold", "id", "depth", "n_neurons"] + [f"mse_{m}" for m in MODELS]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ep = pd.DataFrame([dict(kind=r["job"]["kind"], epochs=r["epochs"]) for r in results])
    print("\nEpochs used (early stopping):", ep.groupby("kind").epochs.agg(["min", "median", "max"]).to_dict("index"))

    # Plot: per line, score (mean +- std over folds) of each model
    colors = {"rb": "0.6", "rd": "0.3", "m1": "tab:blue", "m2": "tab:orange", "m3": "tab:olive", "m4": "tab:red"}
    fig, axes = plt.subplots(1, len(groups), figsize=(17, 4.2))
    for ax, cre in zip(axes, groups):
        mu = [fold_score[cre, m].mean() for m in MODELS]
        sd = [fold_score[cre, m].std() for m in MODELS]
        ax.bar(range(len(MODELS)), mu, yerr=sd, color=[colors[m] for m in MODELS], capsize=3)
        for k in range(N_FOLDS):
            ax.plot(range(len(MODELS)), [fold_score[cre, m][k] for m in MODELS], color="k", lw=0.4, alpha=0.4)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(["ridge b", "ridge d", "m1", "m2", "m3", "m4"], fontsize=8)
        lo = min(min(fold_score[cre, m]) for m in MODELS)
        ax.set_ylim(max(0, lo - 0.05), None)
        n = len(df) if cre == "all lines" else int((df.cre == cre).sum())
        ax.set_title(f"{cre} ({n} mice)", fontsize=9)
        ax.set_ylabel("held-out score (1 - MSE / MSE copy-last)")
    fig.suptitle("Step 9: ridge (step 8) vs MLP: shared (m1), Cre FiLM (m2), shuffled FiLM (m3), per line (m4);"
                 " bars mean +- std over 5 folds, lines = folds", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step9_mlp.png", dpi=150)
    print("Saved plots/step9_mlp.png")


if __name__ == "__main__":
    main()
