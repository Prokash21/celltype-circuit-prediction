"""Step 8: medium-scale test. All VISp three_session_A mice of 4 Cre lines, plus depth.

Sessions: VISp, three_session_A, Cux2-CreERT2 / Sst-IRES-Cre / Vip-IRES-Cre /
Pvalb-IRES-Cre; one experiment per mouse (lowest id), all mice.

Two phases (run separately, so the long download can run in the background):
- `extract`: one session at a time: download the NWB from the Allen public S3
  mirror -> z-scored 10 Hz dF/F (train/test parts), 10 Hz stimulus features,
  frame rate, depth and neuron count (load_session of scripts/step5_shared.py)
  -> data/cache/<id>.npz (float32) -> delete the NWB. Cached sessions are
  skipped, so the phase can be restarted after an interruption.
- `fit`: models and metrics from the cached sessions only.

Preprocessing, stimulus features, model a and the batched X'X fitting are those
of step 5; models b (line-balanced), d and e as in step 7 (scripts/step7_robust.py),
with 4 lines / 4 groups. Inputs per neuron: own past 30 bins + 480 stimulus
inputs -> next 10 bins. Test windows = last 20% of each held-out session.

Folds: per line, mice sorted by experiment id are dealt round-robin into 5
groups (group = rank mod 5). Fold k holds out group k of every line and trains
on the other groups of every line.

Models: (a) per-neuron ridge + stim on the held-out session's own train part
(reference), (b) shared, no label, line-balanced, (d) one shared model per
Cre line, (e) shuffled control for d: training sessions in 4 random groups of
equal size, each mixing lines, each group fit line-balanced as b; held-out
sessions get a random group; 5 seeds per fold.

Summary per line: mean +- std over the 5 folds (metrics pooled over the held-out
mice of the line in the fold), d beats b / d beats e counts over folds and over
held-out mice, and a paired two-sided Wilcoxon test over held-out mice.
Depth: per held-out mouse (d - b) MSE vs imaging depth.
"""
import json
import os
import subprocess
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf
import pandas as pd
from allensdk.api.queries.brain_observatory_api import BrainObservatoryApi
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step5_shared
from step3_linear import CACHE, FUTURE, ROOT
from step5_shared import fit_shared, predict, process_session
from step6_celltype import pooled
from step7_robust import N_SEEDS, NAMES, fit_balanced, line_weights, row, summarize

warnings.filterwarnings("ignore")

CRE_LINES = ["Cux2-CreERT2", "Sst-IRES-Cre", "Vip-IRES-Cre", "Pvalb-IRES-Cre"]
N_GROUPS = len(CRE_LINES)
N_FOLDS = 5
MODELS = ["copy-last", "a", "b", "d"]
S3 = "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-2p"
NPZ = ROOT / "data/cache"
COLORS = {"Cux2-CreERT2": "tab:blue", "Sst-IRES-Cre": "tab:orange",
          "Vip-IRES-Cre": "tab:green", "Pvalb-IRES-Cre": "tab:red"}
SHORT = {c: c.split("-")[0] for c in CRE_LINES}


def pick_sessions():
    """Per Cre line: lowest experiment id per mouse, all mice; fold = rank mod 5."""
    api = BrainObservatoryApi()
    exps = json.load(open(CACHE / "ophys_experiments.json"))
    out = []
    for cre in CRE_LINES:
        es = api.filter_ophys_experiments(exps, cre_lines=[cre], targeted_structures=["VISp"],
                                          session_types=["three_session_A"])
        first = {}
        for e in sorted(es, key=lambda e: e["id"]):
            first.setdefault(e["specimen_id"], e)
        for k, e in enumerate(sorted(first.values(), key=lambda e: e["id"])):
            out.append(dict(cre=cre, id=e["id"], mouse=e["specimen"]["external_specimen_name"],
                            depth=e["imaging_depth"], fold=k % N_FOLDS, held_out=False))
    return out


def extract(infos):
    NPZ.mkdir(parents=True, exist_ok=True)
    nwb_dir = CACHE / "ophys_experiment_data"
    nwb_dir.mkdir(parents=True, exist_ok=True)
    for i, info in enumerate(infos):
        out = NPZ / f"{info['id']}.npz"
        tag = f"[{i + 1}/{len(infos)}] {info['cre']} exp {info['id']}"
        if out.exists():
            print(f"{tag}: cached, skip", flush=True)
            continue
        nwb = nwb_dir / f"{info['id']}.nwb"
        url = f"{S3}/ophys_experiment_data/{info['id']}.nwb"
        print(f"{tag}: downloading", flush=True)
        subprocess.run(["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "10",
                        "--retry-delay", "5", "-o", str(nwb), url], check=True)
        d = step5_shared.load_session(info)
        tmp = NPZ / f"{info['id']}.tmp.npz"
        np.savez_compressed(tmp, ztr=d["ztr"].astype(np.float32), zte=d["zte"].astype(np.float32),
                            Str_all=d["Str_all"].astype(np.float32),
                            Ste_all=d["Ste_all"].astype(np.float32), n_fit=d["n_fit"], fr=d["fr"],
                            n_neurons=d["n_neurons"], depth=info["depth"])
        tmp.rename(out)
        nwb.unlink()
        print(f"{tag}: {d['n_neurons']} neurons, {d['fr']:.2f} Hz, depth {info['depth']} um,"
              f" {d['ztr'].shape[1] + d['zte'].shape[1]} bins -> {out.name}"
              f" ({out.stat().st_size / 1e6:.1f} MB), NWB deleted", flush=True)


def load_cached(info):
    """Replaces step5_shared.load_session: same dict, read from data/cache/<id>.npz."""
    z = np.load(NPZ / f"{info['id']}.npz")
    return dict(fr=float(z["fr"]), n_neurons=int(z["n_neurons"]), n_fit=int(z["n_fit"]),
                **{k: z[k].astype(np.float64) for k in ["ztr", "zte", "Str_all", "Ste_all"]})


def fit_grouped(sessions, labels):
    """d: one step 5 b model per group (no weighting)."""
    return {k: fit_shared([s for s in sessions if labels[s["info"]["id"]] == k])[:3]
            for k in range(N_GROUPS)}


def fit_grouped_balanced(sessions, labels):
    """e: one line-balanced b model per (fake) group."""
    return {k: fit_balanced([s for s in sessions if labels[s["info"]["id"]] == k])
            for k in range(N_GROUPS)}


def shuffled_groups(train, rng):
    """Random equal-size groups of the training sessions; redraw until every group
    mixes Cre lines."""
    while True:
        groups = np.array_split(rng.permutation(len(train)), N_GROUPS)
        if all(len({train[i]["info"]["cre"] for i in g}) > 1 for g in groups):
            return {train[i]["info"]["id"]: k for k, g in enumerate(groups) for i in g}


def process_cached(info):
    step5_shared.load_session = load_cached
    s = process_session(info)
    s.pop("zte"); s.pop("Ste_all"); s.pop("te_starts")
    return s


def fit(infos):
    # Sessions are independent: model a and the Gram sums run in N_WORKERS processes
    with Pool(int(os.environ.get("N_WORKERS", 4))) as pool:
        sessions = pool.map(process_cached, infos, chunksize=1)
    for s in sessions:
        info = s["info"]
        print(f"{info['cre']:14s} fold {info['fold']} exp {info['id']} mouse {info['mouse']}"
              f" depth {info['depth']} | {s['n_neurons']} neurons | windows train/val/test"
              f" {s['n_train_win']}/{s['n_val_win']}/{len(s['Xte'])}", flush=True)
    true = {s["info"]["id"]: CRE_LINES.index(s["info"]["cre"]) for s in sessions}

    per_fold, mice = {}, []
    for k in range(N_FOLDS):
        held = [s for s in sessions if s["info"]["fold"] == k]
        train = [s for s in sessions if s["info"]["fold"] != k]
        Wb, bb, alpha_b = fit_balanced(train)
        fd = fit_grouped(train, true)
        w = line_weights(train)
        cres = [s["info"]["cre"] for s in train]
        print(f"\n== Fold {k}: held out {len(held)} mice, train {len(train)} ==")
        print(f"    line weights {[f'{SHORT[c]} {w[cres.index(c)]:.3f}' for c in CRE_LINES]}"
              f" | alpha b {alpha_b:g}, d " + ", ".join(f"{SHORT[CRE_LINES[g]]} {v[2]:g}"
                                                         for g, v in fd.items()), flush=True)
        shuffles = []
        for seed in range(N_SEEDS):
            rng = np.random.default_rng(seed)
            lab = shuffled_groups(train, rng)
            for s in held:
                lab[s["info"]["id"]] = int(rng.integers(N_GROUPS))
            shuffles.append((lab, fit_grouped_balanced(train, lab)))
            comp = [{SHORT[c]: sum(1 for s in train if lab[s["info"]["id"]] == g and s["info"]["cre"] == c)
                     for c in CRE_LINES} for g in range(N_GROUPS)]
            print(f"    e seed {seed}: group composition {comp} | alphas"
                  f" {[v[2] for v in shuffles[-1][1].values()]}", flush=True)

        rows = []
        for s in held:
            X, S, sid = s["Xte"], s["Ste"], s["info"]["id"]
            Wd, bd, _ = fd[true[sid]]
            P = {"copy-last": np.repeat(X[:, :, -1:], FUTURE, axis=2), "a": s["Pa"],
                 "b": predict(Wb, bb, X, S), "d": predict(Wd, bd, X, S)}
            for j, (lab, fe) in enumerate(shuffles):
                We, be, _ = fe[lab[sid]]
                P[f"e{j}"] = predict(We, be, X, S)
            r = row(s, P)
            rows.append(r)
            m = summarize([r])
            mice.append(dict(**s["info"], n_neurons=s["n_neurons"],
                             **{f"{q}_{x}": m[x][q] for x in MODELS + ["e"] for q in ["mse", "r", "score"]}))
            print(f"  {s['info']['cre']:14s} {sid} depth {s['info']['depth']} ({s['n_neurons']} neurons):"
                  f" MSE a {m['a']['mse']:.4f} b {m['b']['mse']:.4f} d {m['d']['mse']:.4f}"
                  f" e {m['e']['mse']:.4f}+-{m['e_std']['mse']:.4f} | d-b {m['d']['mse'] - m['b']['mse']:+.4f}"
                  f" d-e {m['d']['mse'] - m['e']['mse']:+.4f}", flush=True)
        for cre in CRE_LINES + ["all lines"]:
            rs = [r for r in rows if cre == "all lines" or r["info"]["cre"] == cre]
            res = summarize(rs)
            per_fold[k, cre] = res
            print(f"  {cre} ({len(rs)} mice, {sum(r['n_neurons'] for r in rs)} neurons): "
                  + " | ".join(f"{x} MSE {res[x]['mse']:.4f} score {res[x]['score']:+.3f}"
                               for x in ["a", "b", "d", "e"]))

    # Summary over folds and over held-out mice
    df = pd.DataFrame(mice)
    df["db"] = df["mse_d"] - df["mse_b"]
    df["de"] = df["mse_d"] - df["mse_e"]
    df["db_rel"] = df["db"] / df["mse_b"]
    print(f"\n== Summary over {N_FOLDS} folds (mean +- std, held-out mice pooled per fold) ==")
    for cre in CRE_LINES + ["all lines"]:
        print(cre)
        for x in MODELS + ["e"]:
            v = {q: np.array([per_fold[k, cre][x][q] for k in range(N_FOLDS)]) for q in ["mse", "r", "score"]}
            print(f"    {NAMES[x]:38s} MSE {v['mse'].mean():.4f}+-{v['mse'].std():.4f}"
                  f"  r {v['r'].mean():.3f}+-{v['r'].std():.3f}"
                  f"  score {v['score'].mean():+.3f}+-{v['score'].std():.3f}")
        fd_, fb, fe = (np.array([per_fold[k, cre][x]["mse"] for k in range(N_FOLDS)]) for x in "dbe")
        sub = df if cre == "all lines" else df[df.cre == cre]
        p_b, p_e = wilcoxon(sub.mse_d, sub.mse_b).pvalue, wilcoxon(sub.mse_d, sub.mse_e).pvalue
        print(f"    folds: d - b {np.round(fd_ - fb, 4)}, d beats b {(fd_ < fb).sum()}/{N_FOLDS};"
              f" d - e {np.round(fd_ - fe, 4)}, d beats e {(fd_ < fe).sum()}/{N_FOLDS}")
        print(f"    mice (n={len(sub)}): d beats b {(sub.db < 0).sum()}/{len(sub)}, median d-b"
              f" {sub.db.median():+.4f}, Wilcoxon p {p_b:.2g}; d beats e {(sub.de < 0).sum()}/{len(sub)},"
              f" median d-e {sub.de.median():+.4f}, Wilcoxon p {p_e:.2g}")

    # Depth
    print("\n== Depth: per held-out mouse (d - b) MSE vs imaging depth ==")
    for cre in CRE_LINES:
        sub = df[df.cre == cre]
        print(f"  {cre}: " + ", ".join(f"{dep} um n={len(g)} median d-b {g.db.median():+.4f}"
                                       for dep, g in sub.groupby("depth")))
        if sub.depth.nunique() > 1:
            rho, p = spearmanr(sub.depth, sub.db)
            print(f"    Spearman depth vs d-b rho {rho:+.2f} p {p:.2g}")
    rho, p = spearmanr(df.depth, df.db)
    rho_r, p_r = spearmanr(df.depth, df.db_rel)
    print(f"  all mice: Spearman depth vs d-b rho {rho:+.2f} p {p:.2g}; vs (d-b)/b rho {rho_r:+.2f} p {p_r:.2g}")
    ols = smf.ols("db ~ depth + C(cre)", data=df).fit()
    print(f"  OLS d-b ~ depth + C(cre): depth slope {ols.params['depth'] * 100:+.4f} per 100 um,"
          f" p {ols.pvalues['depth']:.2g}, R2 {ols.rsquared:.2f}")
    ols_r = smf.ols("db_rel ~ depth + C(cre)", data=df).fit()
    print(f"  OLS (d-b)/b ~ depth + C(cre): depth slope {ols_r.params['depth'] * 100:+.4f} per 100 um,"
          f" p {ols_r.pvalues['depth']:.2g}")
    shallow, deep = df[df.depth < 250], df[df.depth >= 250]
    print(f"  <250 um: n={len(shallow)}, median d-b {shallow.db.median():+.4f}, d beats b"
          f" {(shallow.db < 0).sum()}/{len(shallow)}; >=250 um: n={len(deep)}, median d-b"
          f" {deep.db.median():+.4f}, d beats b {(deep.db < 0).sum()}/{len(deep)}")
    print("\nPer held-out mouse:")
    print(df[["cre", "fold", "id", "depth", "n_neurons", "mse_a", "mse_b", "mse_d", "mse_e",
              "score_b", "score_d", "db", "de"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    df.to_csv(NPZ / "step8_mice.csv", index=False)

    # Plot 1: per line, d - b per fold (bars, pooled) and per held-out mouse (dots)
    fig, axes = plt.subplots(1, N_GROUPS, figsize=(15, 4), sharey=False)
    for ax, cre in zip(axes, CRE_LINES):
        db = np.array([per_fold[k, cre]["d"]["mse"] - per_fold[k, cre]["b"]["mse"] for k in range(N_FOLDS)])
        ax.bar(range(N_FOLDS), db, color=["tab:blue" if v < 0 else "tab:red" for v in db], alpha=0.5,
               label="fold (pooled)")
        sub = df[df.cre == cre]
        ax.scatter(sub.fold + np.random.default_rng(0).uniform(-0.15, 0.15, len(sub)), sub.db,
                   color="k", s=12, zorder=3, label="held-out mouse")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(N_FOLDS))
        ax.set_xticklabels([f"fold {k}" for k in range(N_FOLDS)], fontsize=8)
        ax.set_title(f"{cre} ({len(sub)} mice)\nd beats b: {(db < 0).sum()}/{N_FOLDS} folds,"
                     f" {(sub.db < 0).sum()}/{len(sub)} mice", fontsize=9)
        ax.set_ylabel("held-out MSE, d - b")
    axes[0].legend(fontsize=7)
    fig.suptitle("Step 8: per-line model (d) minus shared balanced model (b), held-out test MSE"
                 " (below 0 = d better)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step8_folds.png", dpi=150)

    # Plot 2: d - b vs depth per held-out mouse
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rng = np.random.default_rng(1)
    for cre in CRE_LINES:
        sub = df[df.cre == cre]
        ax.scatter(sub.depth + rng.uniform(-4, 4, len(sub)), sub.db, color=COLORS[cre], s=25,
                   label=f"{cre} (n={len(sub)})", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("imaging depth (um, jittered +-4)")
    ax.set_ylabel("held-out MSE, d - b")
    ax.set_title(f"Step 8: gain of per-line model vs depth (below 0 = d better)\n"
                 f"Spearman rho {rho:+.2f} (p {p:.2g}); OLS with line: depth p {ols.pvalues['depth']:.2g}",
                 fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "plots/step8_depth.png", dpi=150)
    print("Saved plots/step8_folds.png, plots/step8_depth.png")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "fit"
    infos = pick_sessions()
    print("Mice per line (VISp, three_session_A, one experiment per mouse):")
    for cre in CRE_LINES:
        sub = [i for i in infos if i["cre"] == cre]
        print(f"    {cre:14s} {len(sub)} mice, folds {[sum(i['fold'] == k for i in sub) for k in range(N_FOLDS)]},"
              f" depths {sorted({i['depth'] for i in sub})}")
    print(f"    total {len(infos)} sessions", flush=True)
    if phase == "extract":
        extract(infos)
    else:
        fit(infos)


if __name__ == "__main__":
    main()
