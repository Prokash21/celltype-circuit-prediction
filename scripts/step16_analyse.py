"""Step 16 analysis: results/step16/*.csv from the intermediate files in data/step16
(run as `python scripts/step16_neuropixels.py analyse`, after the model phases).

Scores per session (the unit of every test): 1 - sum SSE / sum SSE_copylast over the
session's units of the configuration (dataset x scope x label set). Gains are
variant - none; p = two-sided Wilcoxon signed-rank over sessions (paired); the label
/ summary ratio is mean label gain / mean summary gain over sessions with a 95%
percentile bootstrap over sessions (2,000 resamples, paired). MLP SSE = mean over
seeds. Reference rows (Allen dF/F, Allen deconvolved A / B, Bugeon) are read from
results/step15/gains.csv and label_vs_summary.csv.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import step16_neuropixels as m

OUT = m.OUT
WORK = m.WORK
DS_NAME = {"spk": "Neuropixels spikes", "c6f": "synthetic GCaMP6f", "c6s": "synthetic GCaMP6s",
           "callen": "synthetic Allen-like (Pvalb 6s, others 6f)", "callen_slow": "synthetic Allen-like, 6s decay 1.5 s"}
RIDGE = ["r_none", "r_true", "r_e", "r_e_eq", "r_k1", "r_km16", "r_kmsel"]
MLPV = ["m_none", "m_true", "m_m3", "m_cont"]
MLPU = ["m_none_u", "m_true_u", "m_m3_u", "m_cont_u"]
KIND = {"m1": "m_none", "m2": "m_true", "m3": "m_m3", "cont": "m_cont",
        "m1u": "m_none_u", "m2u": "m_true_u", "m3u": "m_m3_u", "contu": "m_cont_u"}
SETS = {"balanced": ["m1", "m2", "m3", "cont"], "uniform": ["m1u", "m2u", "m3u", "contu"]}
GAINS = [  # (gain, family, a, b)
    ("label", "ridge", "r_true", "r_none"), ("label", "MLP", "m_true", "m_none"),
    ("summary", "ridge", "r_kmsel", "r_none"), ("summary", "MLP", "m_cont", "m_none"),
    ("summary (k=16)", "ridge", "r_km16", "r_none"),
    ("shuffled label", "ridge", "r_e", "r_none"), ("shuffled label", "MLP", "m_m3", "m_none"),
    ("label - shuffled", "ridge", "r_true", "r_e"), ("label - shuffled", "MLP", "m_true", "m_m3"),
    ("summary - label", "ridge", "r_kmsel", "r_true"), ("summary - label", "MLP", "m_cont", "m_true"),
    ("summary (k=16) - label", "ridge", "r_km16", "r_true"),
    ("k1 (equal weights) - none", "ridge", "r_k1", "r_none"),
    ("label - k1", "ridge", "r_true", "r_k1"), ("summary - k1", "ridge", "r_kmsel", "r_k1"),
    # equal-weight / uniform-sampling design: references without label-balanced training
    ("label", "ridge (equal weights)", "r_true", "r_k1"), ("summary", "ridge (equal weights)", "r_kmsel", "r_k1"),
    ("shuffled label", "ridge (equal weights)", "r_e_eq", "r_k1"), ("label - shuffled", "ridge (equal weights)", "r_true", "r_e_eq"),
    ("summary - label", "ridge (equal weights)", "r_kmsel", "r_true"),
    ("label", "MLP (uniform)", "m_true_u", "m_none_u"), ("summary", "MLP (uniform)", "m_cont_u", "m_none_u"),
    ("shuffled label", "MLP (uniform)", "m_m3_u", "m_none_u"), ("label - shuffled", "MLP (uniform)", "m_true_u", "m_m3_u"),
    ("summary - label", "MLP (uniform)", "m_cont_u", "m_true_u"),
    ("uniform - balanced, no label", "MLP", "m_none_u", "m_none"),
]


def configs():
    out = []
    for ds in m.DATASETS:
        for scope in m.SCOPES:
            for ls in m.LABELSETS:
                d = WORK / ds / f"{scope}_{ls}"
                if all((d / f"ridge_sse_fold{k}.csv").exists() for k in range(m.N_FOLDS)):
                    out.append((ds, scope, ls))
    return out


def neuron_table(ds, scope, ls):
    d = WORK / ds / f"{scope}_{ls}"
    neu = m.neurons(scope, ls)
    r = pd.concat([pd.read_csv(d / f"ridge_sse_fold{k}.csv") for k in range(m.N_FOLDS)])
    ns = neu[["session", "unit_id", "probe", "area", "line", "fold", "label", "tagged", "fs", "waveform_duration"]] \
        .merge(r.drop(columns="session"), on="unit_id", how="left", validate="1:1")
    assert ns.sse_copy.notna().all()
    has_mlp = []
    if (d / "mlp_fits.csv").exists():
        fits = pd.read_csv(d / "mlp_fits.csv")
        for set_, kinds in SETS.items():
            complete = all(len(fits[(fits.kind == k) & (fits.fold == f)].seed.unique()) == m.N_SEEDS[k]
                           for k in kinds for f in range(m.N_FOLDS))
            if not complete:
                continue
            has_mlp.append(set_)
            for kind in kinds:
                name = KIND[kind]
                s = pd.read_csv(d / f"mlp_sse_{kind}.csv").drop_duplicates(["fold", "seed", "unit_id"])
                assert (s.groupby("unit_id").seed.nunique() == m.N_SEEDS[kind]).all()
                g = s.groupby("unit_id").mean(numeric_only=True)
                ns[f"sse_{name}"] = ns.unit_id.map(g.sse)
                if kind in ("m2", "m2u"):
                    for c, lab in enumerate(m.LABELSETS[ls]):
                        ns[f"sse_{kind}_as_{lab}"] = ns.unit_id.map(g[f"sse_as_{c}"])
    ns.insert(0, "labelset", ls)
    ns.insert(0, "scope", scope)
    ns.insert(0, "ds", ds)
    return ns, has_mlp


def session_scores(ns, variants, subset=None):
    d = ns if subset is None else ns[subset(ns)]
    g = d.groupby("session")
    out = pd.DataFrame({"n_units": g.size()})
    cp = g.sse_copy.sum()
    for v in variants:
        out[v] = 1 - g[f"sse_{v}"].sum() / cp
    first = g.first()
    out["line"] = first.line
    return out


def wil(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        p = wilcoxon(a, b).pvalue if np.any(d != 0) else np.nan
    except ValueError:
        p = np.nan
    return p


def gain_rows(sc, keys):
    rows = []
    for gain, fam, a, b in GAINS:
        if a not in sc or b not in sc or sc[a].isna().any() or sc[b].isna().any():
            continue
        d = (sc[a] - sc[b]).to_numpy()
        rows.append(dict(**keys, gain=gain, family=fam, a=a, b=b, n_sessions=len(d), score_b=sc[b].mean(),
                         score_a=sc[a].mean(), gain_mean=d.mean(), gain_sem=d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan,
                         gain_median=np.median(d), sessions_up=int((d > 0).sum()), p=wil(sc[a], sc[b]) if len(d) > 5 else np.nan))
    return rows


def ratio_rows(sc, keys, rng_seed=0):
    rows = []
    for fam, (lab, summ, none) in RATIO_FAMS.items():
        if summ not in sc or sc[summ].isna().any() or len(sc) < 3:
            continue
        gl, gs = (sc[lab] - sc[none]).to_numpy(), (sc[summ] - sc[none]).to_numpy()
        rng = np.random.default_rng(rng_seed)
        bs = []
        for _ in range(m.N_BOOT):
            i = rng.integers(len(gl), size=len(gl))
            bs.append(gl[i].mean() / gs[i].mean())
        rows.append(dict(**keys, family=fam, n_sessions=len(gl), gain_label=gl.mean(), gain_summary=gs.mean(),
                         label_over_summary=gl.mean() / gs.mean(), ci_lo=np.percentile(bs, 2.5), ci_hi=np.percentile(bs, 97.5)))
    return rows


RATIO_FAMS = {"ridge": ("r_true", "r_kmsel", "r_none"), "MLP": ("m_true", "m_cont", "m_none"),
              "ridge (k=16)": ("r_true", "r_km16", "r_none"), "ridge (equal weights)": ("r_true", "r_kmsel", "r_k1"),
              "MLP (uniform)": ("m_true_u", "m_cont_u", "m_none_u")}


def subsets(ls):
    s = {"all units": None}
    if ls == "geno":
        return s
    if ls == "tag":
        s["tagged units"] = lambda d: d.tagged.astype(bool)
        s["untagged units"] = lambda d: ~d.tagged.astype(bool)
    else:
        s["FS units"] = lambda d: d.fs.astype(bool)
        s["RS units"] = lambda d: ~d.fs.astype(bool)
    return s


def calcium_vs_spikes(sc_all):
    """Task 3: per calcium version, the change of the label gain and of the label /
    summary ratio relative to spikes on the same sessions and units."""
    rows = []
    for (scope, ls), g in sc_all[sc_all.subset == "all units"].groupby(["scope", "labelset"]):
        base = g[g.ds == "spk"].set_index("session")
        if base.empty:
            continue
        for ds in m.CALCIUM:
            c = g[g.ds == ds].set_index("session")
            if c.empty:
                continue
            for fam, (lab, summ, none) in RATIO_FAMS.items():
                if summ not in c or summ not in base or c[summ].isna().any() or base[summ].isna().any():
                    continue
                s_ = base.index.intersection(c.index)
                gl_s, gs_s = (base[lab] - base[none])[s_].to_numpy(), (base[summ] - base[none])[s_].to_numpy()
                gl_c, gs_c = (c[lab] - c[none])[s_].to_numpy(), (c[summ] - c[none])[s_].to_numpy()
                rng = np.random.default_rng(0)
                bd, br = [], []
                for _ in range(m.N_BOOT):
                    i = rng.integers(len(s_), size=len(s_))
                    bd.append(gl_c[i].mean() - gl_s[i].mean())
                    br.append(gl_c[i].mean() / gs_c[i].mean() - gl_s[i].mean() / gs_s[i].mean())
                rows.append(dict(scope=scope, labelset=ls, ds=ds, family=fam, n_sessions=len(s_),
                                 label_gain_spikes=gl_s.mean(), label_gain_calcium=gl_c.mean(),
                                 diff=gl_c.mean() - gl_s.mean(), diff_ci_lo=np.percentile(bd, 2.5), diff_ci_hi=np.percentile(bd, 97.5),
                                 sessions_higher=int((gl_c > gl_s).sum()), p_label_gain=wil(gl_c, gl_s),
                                 summary_gain_spikes=gs_s.mean(), summary_gain_calcium=gs_c.mean(), p_summary_gain=wil(gs_c, gs_s),
                                 ratio_spikes=gl_s.mean() / gs_s.mean(), ratio_calcium=gl_c.mean() / gs_c.mean(),
                                 ratio_diff_ci_lo=np.percentile(br, 2.5), ratio_diff_ci_hi=np.percentile(br, 97.5)))
    return pd.DataFrame(rows)


def task1_tables():
    lab = pd.read_csv(WORK / "labels.csv")
    S = m.sessions()
    man = pd.read_csv(WORK / "manifest.csv")
    pb = pd.read_csv(WORK / "probes_binned.csv")
    kept = set()
    for sid in S[S.extracted].id:
        kept |= set(np.load(WORK / "spk" / f"{sid}.npz")["unit_id"])
    lab["kept"] = lab.unit_id.isin(kept)
    good = lab[(lab.quality == "good")]
    # manifest summary (no data)
    ms = S[["id", "line", "session_type", "genotype", "fold"]].rename(columns={"id": "session"}) \
        .merge(man, on="session", how="left")
    cnt = good.groupby("session").agg(n_good_vis=("unit_id", "size"), n_kept_vis=("kept", "sum"))
    k = lab[lab.kept]
    cnt["n_kept_visp"] = k[k.area == "VISp"].groupby("session").size()
    cnt["n_tagged_vis"] = k[k.tagged & (k.line != "wt")].groupby("session").size()
    cnt["n_tagged_visp"] = k[k.tagged & (k.line != "wt") & (k.area == "VISp")].groupby("session").size()
    cnt["n_fs_vis"] = k[k.fs].groupby("session").size()
    cnt["n_wt_false_tag_vis"] = k[k.tagged & (k.line == "wt")].groupby("session").size()
    ms = ms.merge(cnt.fillna(0).astype(int).reset_index(), on="session", how="left")
    inv = pb.groupby("session").agg(minutes=("minutes", "first"), max_probe_invalid_frac=("invalid_frac", "max"),
                                    n_good_dropped_invalid=("n_good_dropped_invalid", "sum"))
    ms = ms.merge(inv.reset_index(), on="session", how="left")
    ms.to_csv(OUT / "manifest_summary.csv", index=False, float_format="%.4g")
    # unit labels (kept units = the units of every analysis)
    cols = ["session", "line", "session_type", "fold", "unit_id", "area", "probe", "local_channel", "probe_vertical_position",
            "waveform_duration", "waveform_halfwidth", "firing_rate", "snr", "n_pulses", "baseline_rate", "evoked_rate",
            "ratio", "p_wilcoxon", "frac_trials_evoked_spike", "median_first_spike_ms", "tagged_tutorial", "tagged_p01", "tagged", "fs",
            "label_tag", "label_fs"]
    k[cols].to_csv(OUT / "unit_labels.csv", index=False, float_format="%.5g")
    # counts per session x area x label
    rows = []
    for (sid, area), g in k.groupby(["session", "area"]):
        r = dict(session=sid, line=g.line.iloc[0], area=area, n_units=len(g), FS=int(g.fs.sum()), RS=int((~g.fs).sum()))
        for l in m.LINES:
            r[f"tagged_{l}"] = int((g.tagged & (g.line == l)).sum())
        r["untagged_cre"] = int((~g.tagged & (g.line != "wt")).sum())
        r["tagged_in_wt"] = int((g.tagged & (g.line == "wt")).sum())
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "label_counts.csv", index=False)
    # summary per line and scope
    rows = []
    for scope, areas in m.SCOPES.items():
        kk = k[k.area.isin(areas)]
        for line, g in kk.groupby("line"):
            t = g[g.tagged]
            rows.append(dict(scope=scope, line=line, sessions=g.session.nunique(), units=len(g),
                             tagged=len(t), tagged_pct=100 * len(t) / len(g), tagged_tutorial_only=int(g.tagged_tutorial.sum()),
                             tagged_p01=int(g.tagged_p01.sum()),
                             sessions_with_tagged=t.session.nunique(), tagged_per_session_median=t.groupby("session").size().reindex(g.session.unique(), fill_value=0).median(),
                             tagged_fs_pct=100 * t.fs.mean() if len(t) else np.nan, untagged_fs_pct=100 * g[~g.tagged].fs.mean(),
                             tagged_waveform_ms_median=t.waveform_duration.median() if len(t) else np.nan,
                             tagged_latency_ms_median=t.median_first_spike_ms.median() if len(t) else np.nan,
                             tagged_ratio_median=t.ratio.median() if len(t) else np.nan, fs_pct=100 * g.fs.mean()))
    pd.DataFrame(rows).to_csv(OUT / "optotag_summary.csv", index=False, float_format="%.4g")
    # PSTHs: mean +- sem per line and tagged / untagged (kept VIS units); per tagged unit
    ps = pickle.load(open(WORK / "psth.pkl", "rb"))
    cen = (m.PSTH_EDGES[:-1] + m.PSTH_EDGES[1:]) / 2 * 1e3
    allu = []
    for sid, h in ps.items():
        z = np.load(m.SESS_DIR / f"{sid}.npz", allow_pickle=True)
        allu.append(pd.DataFrame(h, columns=[f"{c:.2f}" for c in cen]).assign(unit_id=pd.DataFrame(z["units"]).unit_id.to_numpy()))
    H = pd.concat(allu).set_index("unit_id")
    rows = []
    for (line, tg), g in k.groupby(["line", "tagged"]):
        A = H.loc[g.unit_id].to_numpy()
        A = A[np.isfinite(A).all(1)]
        for j, c in enumerate(cen):
            rows.append(dict(line=line, tagged=tg, n_units=len(A), t_ms=c, rate_mean=A[:, j].mean(), rate_sem=A[:, j].std() / np.sqrt(len(A))))
    pd.DataFrame(rows).to_csv(OUT / "optotag_psth.csv", index=False, float_format="%.5g")
    t = k[k.tagged]
    H.loc[t.unit_id].reset_index().merge(t[["unit_id", "session", "line", "area"]], on="unit_id") \
        .to_csv(OUT / "optotag_psth_tagged_units.csv", index=False, float_format="%.4g")
    print(pd.read_csv(OUT / "optotag_summary.csv").to_string())


def example_traces():
    """First 60 s of the test part of one tagged VISp unit per line (the one with the
    highest train-part event rate), as z-scored 10 Hz spike counts and synthetic
    calcium (6f, 6s)."""
    lab = pd.read_csv(OUT / "unit_labels.csv")
    f = pd.read_csv(WORK / "spk" / "features.csv")[["unit_id", "event rate (/min)"]]
    out = []
    for line in m.LINES:
        t = lab[(lab.line == line) & lab.tagged & (lab.area == "VISp")].merge(f, on="unit_id")
        if t.empty:
            continue
        u = t.sort_values("event rate (/min)").iloc[-1]
        for ds in ["spk", "c6f", "c6s"]:
            b, _, Zte = m.session_arrays(ds, u.session)
            j = int(np.where(b["unit_id"] == u.unit_id)[0][0])
            Z = Zte[j, :600]
            out.append(pd.DataFrame(dict(line=line, session=u.session, unit_id=u.unit_id, signal=ds, t=np.arange(len(Z)) / 10, z=Z)))
    pd.concat(out).to_csv(OUT / "example_traces.csv", index=False, float_format="%.4g")


def reference_rows():
    g = pd.read_csv(m.ROOT / "results/step15/gains.csv")
    r = pd.read_csv(m.ROOT / "results/step15/label_vs_summary.csv")
    g = g[g.ds.isin(["dff", "A", "B", "bugeon"]) & g.gain.isin(["label", "summary"])]
    return g, r[r.ds.isin(["dff", "A", "B", "bugeon"])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    task1_tables()
    nst, sc_rows, gains, ratios, alphas, valk, fits = [], [], [], [], [], [], []
    for ds, scope, ls in configs():
        ns, has_mlp = neuron_table(ds, scope, ls)
        d = WORK / ds / f"{scope}_{ls}"
        variants = RIDGE + (MLPV if "balanced" in has_mlp else []) + (MLPU if "uniform" in has_mlp else [])
        for c in [c for c in ns.columns if c.startswith("sse_") and c != "sse_copy"]:
            ns[c.replace("sse_", "score_")] = 1 - ns[c] / ns.sse_copy
        nst.append(ns)
        for name, sub in subsets(ls).items():
            sc = session_scores(ns, variants, sub)
            keys = dict(ds=ds, dataset=DS_NAME[ds], scope=scope, labelset=ls, subset=name)
            sc_rows.append(sc.reset_index().assign(**keys))
            gains += gain_rows(sc, dict(**keys, line="all"))
            ratios += ratio_rows(sc, dict(**keys, line="all"))
            for line, g in sc.groupby("line"):
                gains += gain_rows(g, dict(**keys, line=line))
                ratios += ratio_rows(g, dict(**keys, line=line))
        a = pd.concat([pd.read_csv(d / f"ridge_alphas_fold{k}.csv") for k in range(m.N_FOLDS)])
        alphas.append(a.assign(ds=ds, scope=scope, labelset=ls))
        valk.append(pd.concat([pd.read_csv(d / f"val_k_fold{k}.csv") for k in range(m.N_FOLDS)]).assign(ds=ds, scope=scope, labelset=ls))
        if has_mlp:
            fits.append(pd.read_csv(d / "mlp_fits.csv").drop(columns="hist").assign(ds=ds, scope=scope, labelset=ls))
    ns = pd.concat(nst, ignore_index=True)
    keep = ["ds", "scope", "labelset", "session", "unit_id", "area", "line", "fold", "label", "tagged", "fs", "n_el", "sse_copy"] + \
        [c for c in ns.columns if c.startswith("score_")]
    ns[keep].to_csv(OUT / "neuron_scores.csv", index=False, float_format="%.5g")
    sc_all = pd.concat(sc_rows, ignore_index=True)
    sc_all.to_csv(OUT / "session_scores.csv", index=False, float_format="%.6g")
    G = pd.DataFrame(gains)
    G.to_csv(OUT / "gains.csv", index=False, float_format="%.5g")
    R = pd.DataFrame(ratios)
    R.to_csv(OUT / "label_vs_summary.csv", index=False, float_format="%.5g")
    pd.concat(alphas).to_csv(OUT / "ridge_alphas.csv", index=False)
    vk = pd.concat(valk)
    vk.to_csv(OUT / "val_k.csv", index=False, float_format="%.6g")
    if fits:
        pd.concat(fits).to_csv(OUT / "mlp_fits.csv", index=False, float_format="%.5g")
    cvs = calcium_vs_spikes(sc_all)
    cvs.to_csv(OUT / "calcium_vs_spikes.csv", index=False, float_format="%.5g")
    # comparison table: Neuropixels configurations next to Allen dF/F / A / B and Bugeon (step 15)
    gref, rref = reference_rows()
    rows = []
    name15 = {"dff": "Allen dF/F (10 Hz)", "A": "Allen deconvolved A (10 Hz)", "B": "Allen deconvolved B (4.3 Hz)",
              "bugeon": "Bugeon (deconvolved, 4.3 Hz)"}
    for ds15, nm in name15.items():
        r = dict(dataset=nm, scope="VISp", labelset="Cre line" if ds15 != "bugeon" else "t-type",
                 units=f"{int(gref[gref.ds == ds15].n_units.iloc[0])} {gref[gref.ds == ds15].unit.iloc[0]}s")
        for fam in ["ridge", "MLP"]:
            for gain in ["label", "summary"]:
                x = gref[(gref.ds == ds15) & (gref.family == fam) & (gref.gain == gain)]
                r[f"{gain}_{fam}"] = x.gain_mean.iloc[0] if len(x) else np.nan
            x = rref[(rref.ds == ds15) & (rref.family == fam)]
            r[f"ratio_{fam}"], r[f"ratio_{fam}_lo"], r[f"ratio_{fam}_hi"] = (x.label_over_summary.iloc[0], x.ci_lo.iloc[0], x.ci_hi.iloc[0]) if len(x) else (np.nan,) * 3
        rows.append(r)
    for (ds, scope, ls), g in G[(G.subset == "all units") & (G.line == "all")].groupby(["ds", "scope", "labelset"], sort=False):
        r = dict(dataset=DS_NAME[ds], scope=scope, labelset={"tag": "opto-tag (i)", "fs": "FS / RS (ii)", "geno": "mouse genotype (recording-level)"}[ls],
                 units=f"{int(g.n_sessions.iloc[0])} sessions")
        for fam in ["ridge", "MLP", "ridge (equal weights)", "MLP (uniform)"]:
            for gain in ["label", "summary"]:
                x = g[(g.family == fam) & (g.gain == gain)]
                r[f"{gain}_{fam}"] = x.gain_mean.iloc[0] if len(x) else np.nan
                r[f"{gain}_{fam}_p"] = x.p.iloc[0] if len(x) else np.nan
            x = R[(R.ds == ds) & (R.scope == scope) & (R.labelset == ls) & (R.subset == "all units") & (R.line == "all") & (R.family == fam)]
            r[f"ratio_{fam}"], r[f"ratio_{fam}_lo"], r[f"ratio_{fam}_hi"] = (x.label_over_summary.iloc[0], x.ci_lo.iloc[0], x.ci_hi.iloc[0]) if len(x) else (np.nan,) * 3
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False, float_format="%.5g")
    if any(c[0] == "c6f" for c in configs()):
        example_traces()
    snr = m.allen_snr()
    snr.groupby("indicator").snr.describe().to_csv(OUT / "allen_snr.csv", float_format="%.4g")
    print(pd.read_csv(OUT / "comparison.csv").to_string())


if __name__ == "__main__":
    main()
