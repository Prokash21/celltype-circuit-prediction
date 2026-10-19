"""Step 16 extraction: Allen Visual Coding Neuropixels (Siegle et al. 2021), one
session at a time, resumable.

For each of the 58 sessions (brain_observatory_1.1 and functional_connectivity;
Pvalb / Sst / Vip x Ai32 first, then wild type):
  download data/step16/nwb/session_<id>.nwb from the public S3 bucket
  (allen-brain-observatory/visual-coding-neuropixels/ecephys-cache; session NWB
  only, never the LFP files) -> extract data/step16/sessions/<id>.npz -> delete
  the NWB. data/step16/manifest.csv (session, status, n_units, error) is updated
  after every session; sessions marked done are skipped on restart. At most 2 NWB
  files are on disk (the one being extracted and the next being downloaded); free
  disk space is checked before each download; a failed download is retried 3 times
  and then marked failed.

Kept per unit (visual cortex: VISp, VISl, VISal, VISrl, VISam, VISpm, by the
peak channel's CCF structure; isi_violations < 0.5, amplitude_cutoff < 0.1,
presence_ratio > 0.9; the quality column is kept so the noise clusters, which
AllenSDK's units table also drops, can be excluded downstream): spike times
(uint32, 30 kHz ticks, 33 us resolution), waveform_duration, waveform_halfwidth,
firing_rate, the QC metrics, area, peak channel id, probe, local channel index,
probe vertical / horizontal position. Per session: every stimulus presentation
(name, start, stop, orientation, contrast, temporal / spatial frequency, frame),
running speed (timestamps, cm/s), optogenetic stimulation epochs (start, stop,
duration, level, name), invalid-time intervals (start, stop, probe tags) and the
genotype / session type.

Usage: python scripts/step16_extract.py [--only <id>]
"""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data/step16"
META, NWB, SESS = WORK / "meta", WORK / "nwb", WORK / "sessions"
MANIFEST = WORK / "manifest.csv"
BUCKET = "https://allen-brain-observatory.s3.amazonaws.com/visual-coding-neuropixels/ecephys-cache"
AREAS = ["VISp", "VISl", "VISal", "VISrl", "VISam", "VISpm"]
QC = dict(isi_violations=0.5, amplitude_cutoff=0.1, presence_ratio=0.9)
TICK = 30000.0                    # spike-time resolution of the stored files (Hz)
MIN_FREE_GB = 8                   # needed before a download (<= 3.3 GB file + extraction headroom)
LINE_ORDER = ["Pvalb", "Sst", "Vip", "wt"]
STIM_COLS = ["orientation", "contrast", "temporal_frequency", "spatial_frequency", "frame"]


def genotype_line(g):
    return g.split("-")[0] if "Cre" in g else "wt"


def session_table():
    """All 58 sessions in extraction order: Cre x Ai32 (Pvalb, Sst, Vip) first, then wild type."""
    s = pd.read_csv(META / "sessions.csv")
    s["line"] = s.genotype.map(genotype_line)
    s["order"] = s.line.map(LINE_ORDER.index)
    return s.sort_values(["order", "id"]).reset_index(drop=True)


def read_manifest():
    if MANIFEST.exists():
        return pd.read_csv(MANIFEST, dtype={"error": str}).set_index("session")
    return pd.DataFrame(columns=["status", "n_units", "n_units_good", "n_visp_good", "error"]).rename_axis("session")


def write_manifest(m):
    tmp = MANIFEST.with_suffix(".tmp")
    m.reset_index().to_csv(tmp, index=False)
    tmp.rename(MANIFEST)


def free_gb():
    return shutil.disk_usage(WORK).free / 1e9


def download(sid):
    """curl the session NWB (retries: 3 after the first attempt). Returns the path or raises."""
    path = NWB / f"session_{sid}.nwb"
    url = f"{BUCKET}/session_{sid}/session_{sid}.nwb"
    if path.exists():  # complete file from an earlier run (downloads land as .part first)
        return path
    err = ""
    for attempt in range(4):
        while free_gb() < MIN_FREE_GB:
            print(f"  {sid}: waiting for disk ({free_gb():.1f} GB free)", flush=True)
            time.sleep(30)
        part = path.with_suffix(".part")
        h = subprocess.run(["curl", "-sSI", "--fail", url], capture_output=True, text=True)
        size = [int(l.split(":")[1]) for l in h.stdout.lower().splitlines() if l.startswith("content-length")]
        r = subprocess.run(["curl", "-sS", "--fail", "-o", str(part), url], capture_output=True, text=True)
        if r.returncode == 0 and part.exists() and size and part.stat().st_size == size[-1]:
            part.rename(path)
            return path
        err = (r.stderr or f"curl exit {r.returncode}, {part.stat().st_size if part.exists() else 0} of"
                            f" {size[-1] if size else '?'} bytes").strip()[:200]
        part.unlink(missing_ok=True)
        print(f"  {sid}: download attempt {attempt + 1} failed ({err})", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"download failed after 4 attempts: {err}")


def dec(a):
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def ragged(f, name, idx_name):
    end = f[idx_name][()]
    start = np.concatenate([[0], end[:-1]])
    return start, end


def extract(sid, path, info):
    f = h5py.File(path, "r")
    u = f["units"]
    e = f["general/extracellular_ephys/electrodes"]
    loc = dict(zip(e["id"][()], dec(e["location"][()])))
    vpos = dict(zip(e["id"][()], e["probe_vertical_position"][()]))
    hpos = dict(zip(e["id"][()], e["probe_horizontal_position"][()]))
    lidx = dict(zip(e["id"][()], e["local_index"][()]))
    pid = dict(zip(e["id"][()], e["probe_id"][()]))
    probes = pd.read_csv(META / "probes.csv").set_index("id")
    ch = u["peak_channel_id"][()]
    area = np.array([loc[c] for c in ch])
    qcol = {k: u[k][()] for k in QC}
    keep = np.isin(area, AREAS)
    for k, thr in QC.items():
        keep &= (qcol[k] > thr) if k == "presence_ratio" else (qcol[k] < thr)
    idx = np.where(keep)[0]
    st_start, st_end = ragged(f, "spike_times", "units/spike_times_index")
    spikes, counts = [], []
    allspk = u["spike_times"]
    for i in idx:
        t = allspk[st_start[i]:st_end[i]]
        assert np.all(np.diff(t) >= 0) and t.min() >= 0
        spikes.append(np.round(t * TICK).astype(np.uint32))
        counts.append(len(t))
    units = pd.DataFrame(dict(
        unit_id=u["id"][()][idx], area=area[idx], peak_channel_id=ch[idx],
        probe_id=[pid[c] for c in ch[idx]], probe=[probes.loc[pid[c], "name"] for c in ch[idx]],
        local_channel=[lidx[c] for c in ch[idx]], probe_vertical_position=[vpos[c] for c in ch[idx]],
        probe_horizontal_position=[hpos[c] for c in ch[idx]],
        surface_channel_index=[probes.loc[pid[c], "surface_channel_index"] for c in ch[idx]],
        quality=dec(u["quality"][()][idx]), n_spikes=counts,
        **{k: u[k][()][idx] for k in ["waveform_duration", "waveform_halfwidth", "firing_rate", "snr",
                                      "isi_violations", "amplitude_cutoff", "presence_ratio", "amplitude",
                                      "PT_ratio", "repolarization_slope", "recovery_slope"]}))
    # Stimulus presentations
    rows = []
    for k in f["intervals"]:
        g = f["intervals"][k]
        if not k.endswith("_presentations"):
            continue
        n = len(g["start_time"])
        d = dict(stimulus=k[:-len("_presentations")], start=g["start_time"][()], stop=g["stop_time"][()],
                 name=dec(g["stimulus_name"][()]) if "stimulus_name" in g else np.full(n, k))
        for c in STIM_COLS:
            v = np.full(n, np.nan)
            if c in g:
                raw = g[c][()]
                if raw.dtype.kind in "fiu":
                    v = raw.astype(float)
                else:  # e.g. '[250.0, 250.0]' or 'null'
                    v = pd.to_numeric(pd.Series(dec(raw)), errors="coerce").to_numpy()
            d[c] = v
        rows.append(pd.DataFrame(d))
    stim = pd.concat(rows).sort_values("start").reset_index(drop=True)
    og = f["processing/optotagging"] if "optotagging" in f["processing"] else {}
    tabs = [k for k in og if isinstance(og[k], h5py.Group) and "start_time" in og[k]]  # name varies across releases
    assert len(tabs) <= 1, f"optotagging tables: {list(og)}"
    if tabs:
        o = og[tabs[0]]
        opto = pd.DataFrame(dict(start=o["start_time"][()], stop=o["stop_time"][()], duration=o["duration"][()],
                                 level=o["level"][()], name=dec(o["stimulus_name"][()]), condition=dec(o["condition"][()])))
    else:  # 4 wild-type sessions have no optotagging epoch
        opto = pd.DataFrame(dict(start=np.zeros(0), stop=np.zeros(0), duration=np.zeros(0), level=np.zeros(0),
                                 name=np.array([], dtype="<U1"), condition=np.array([], dtype="<U1")))
    inv = f["intervals/invalid_times"] if "invalid_times" in f["intervals"] else {"id": []}  # absent in some files
    tags = dec(inv["tags"][()]) if len(inv["id"]) else np.array([])
    tix = inv["tags_index"][()] if len(inv["id"]) else np.array([], int)
    t0 = np.concatenate([[0], tix[:-1]]) if len(tix) else np.array([], int)
    inv_tags = [";".join(tags[a:b]) for a, b in zip(t0, tix)]
    run = f["processing/running/running_speed"]
    out = SESS / f"{sid}.tmp.npz"
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    np.savez_compressed(
        out, spikes=np.concatenate(spikes) if spikes else np.zeros(0, np.uint32), spike_offsets=off, tick=TICK,
        units=units.to_records(index=False), stim=stim.to_records(index=False), opto=opto.to_records(index=False),
        invalid_start=inv["start_time"][()] if len(inv["id"]) else np.zeros(0),
        invalid_stop=inv["stop_time"][()] if len(inv["id"]) else np.zeros(0), invalid_tags=np.array(inv_tags, dtype=object),
        run_t=run["timestamps"][()], run_speed=run["data"][()].astype(np.float32),
        genotype=dec([f["general/subject/genotype"][()]])[0], session_type=info["session_type"], line=info["line"],
        session_id=sid, specimen_id=info["specimen_id"])
    f.close()
    out.rename(SESS / f"{sid}.npz")
    return units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int)
    a = ap.parse_args()
    for d in [NWB, SESS]:
        d.mkdir(parents=True, exist_ok=True)
    sess = session_table()
    if a.only:
        sess = sess[sess.id == a.only]
    m = read_manifest()
    todo = [r for _, r in sess.iterrows() if not (r.id in m.index and m.loc[r.id, "status"] == "done"
                                                   and (SESS / f"{r.id}.npz").exists())]
    print(f"{len(sess)} sessions, {len(todo)} to extract; {free_gb():.1f} GB free", flush=True)
    # Prefetch: the next NWB downloads while the current one is extracted (<= 2 on disk)
    for p in NWB.glob("*.part"):
        p.unlink()
    nxt = {}

    def fetch(sid):
        try:
            nxt[sid] = download(sid)
        except Exception as ex:  # noqa: BLE001
            nxt[sid] = ex

    th = None
    for j, r in enumerate(todo):
        sid = int(r.id)
        t0 = time.time()
        if th is None:
            th = threading.Thread(target=fetch, args=(sid,))
            th.start()
        th.join()
        res = nxt.pop(sid)
        th = None
        if j + 1 < len(todo):
            th = threading.Thread(target=fetch, args=(int(todo[j + 1].id),))
            th.start()
        if isinstance(res, Exception):
            m.loc[sid, ["status", "n_units", "n_units_good", "n_visp_good", "error"]] = ["failed", np.nan, np.nan, np.nan, str(res)]
            write_manifest(m)
            print(f"[{j + 1}/{len(todo)}] {sid} FAILED: {res}", flush=True)
            continue
        try:
            try:
                units = extract(sid, res, r)
            except (KeyError, OSError) as ex:  # unreadable HDF5 (e.g. a truncated transfer): download once more
                print(f"  {sid}: read error ({ex!r}), downloading again", flush=True)
                res.unlink(missing_ok=True)
                res = download(sid)
                units = extract(sid, res, r)
            good = units.quality == "good"
            m.loc[sid, ["status", "n_units", "n_units_good", "n_visp_good", "error"]] = [
                "done", len(units), int(good.sum()), int((good & (units.area == "VISp")).sum()), ""]
            print(f"[{j + 1}/{len(todo)}] {sid} {r.line} {r.session_type}: {len(units)} VIS units ({good.sum()} good,"
                  f" VISp {(good & (units.area == 'VISp')).sum()}), {time.time() - t0:.0f} s,"
                  f" {free_gb():.1f} GB free", flush=True)
        except Exception as ex:  # noqa: BLE001
            m.loc[sid, ["status", "n_units", "n_units_good", "n_visp_good", "error"]] = ["failed", np.nan, np.nan, np.nan, repr(ex)[:200]]
            print(f"[{j + 1}/{len(todo)}] {sid} extraction FAILED: {ex!r}", flush=True)
        finally:
            write_manifest(m)
            res.unlink(missing_ok=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
