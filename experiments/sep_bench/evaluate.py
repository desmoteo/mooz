"""Score separation models and ensembles on the benchmark (experiments/sep_bench/build.py).

Every model runs through the pipeline's own separation code (rehearsal_master.separate: the
same chunking, scaling and model calls), on the signal the pipeline separates (the mix after
its rumble filter; the references get the same filter). Per song three runs, cached in
processed/experiments/sep_bench/stems/<run>/<song>/ (each model's outputs in models/):

  superset  every candidate model on the whole mix (SW, five vocal models, htdemucs_ft),
            then both DrumSep models on SW's drums
  oracle    both DrumSep models on the true drums (what DrumSep alone loses)
  cascade   the earlier plan (vocals model -> SW on its instrumental -> DrumSep 6 stems)

Scores, per stem, averaged over the songs where it plays (reference within 40 dB of the mix):
  SDR        10 log10(sum ref^2 / sum (ref - est)^2), both channels (MVSep / ZFTurbo "SDR")
  bleedless  / fullness: ZFTurbo's metrics (utils/metrics.py), mel spectrograms in dB:
             the mean excess of the estimate over the reference (other sounds in it) and
             the mean shortfall (parts of it missing), as 100 / (1 + dB); higher is better
Ensembles: per stem, the weighted average of its candidates with the weights (>= 0, sum 1)
that maximise the mean SDR over the songs; "loo" = each song scored with the weights fitted
on the other songs (what to expect on new recordings).

  uv run python experiments/sep_bench/evaluate.py [song-substring ...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import optimize, signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from rehearsal_master import config, separate  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import custom_models  # noqa: E402

OUT = ROOT / "processed" / "experiments" / "sep_bench"
BENCH = Path(os.environ.get("SEP_BENCH_DATA", OUT / "data"))  # built by build.py
KIT = ["kick", "snare", "toms", "hihat", "cymbals"]

SW = "BS-Roformer-SW.ckpt"
V317 = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
DS6 = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
DS5 = "mdx23c_drumsep_5stem_aufr33_jarredou.ckpt"
VOCALS = {"317": V317, "kj": "vocals_mel_band_roformer.ckpt", "becruily": "mel_band_roformer_vocals_becruily.ckpt",
          "revive": "bs_roformer_vocals_revive_v2_unwa.ckpt"}
KIT6 = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "ride": "cymbals", "crash": "cymbals"}
KIT5 = {"kick": "kick", "snare": "snare", "toms": "toms", "hh": "hihat", "cymbals": "cymbals"}


def kit_step(inp: str) -> dict:
    return {"input": inp, "models": [{"model": DS6, "outputs": {k: f"{v}~6" for k, v in KIT6.items()}},
                                     {"model": DS5, "outputs": {k: f"{v}~5" for k, v in KIT5.items()}}]}


RUNS = {
    "superset": [
        {"input": "mix", "models": [
            {"model": SW, "outputs": {"drums": "drums", "bass": "bass", "guitar": "guitar", "piano": "other",
                                      "other": "other", "vocals": "vocals~sw"}},
            *[{"model": m, "outputs": {"vocals": f"vocals~{k}"}} for k, m in VOCALS.items()],
            {"model": "htdemucs_ft.yaml", "outputs": {"drums": "drums~htdemucs", "bass": "bass~htdemucs"}},
        ]},
        kit_step("drums"),
    ],
    "oracle": [kit_step("mix")],
    "cascade": [s for s in config.load(None)["stems"]["plans"]["cascade"]["steps"]
                if "dereverb" not in s["model"] and s.get("enabled", True)],
    # the mirror's models (candidates.py) on the whole mix; the 53-stem model's stems overlap
    # (drums and kick, vocal and lead-vocal), so each goes to its own name
    "extra": [{"input": "mix", "models": [
        {"model": "bs_mega_53stem_full_mvsep.ckpt", "segment_size": 512, "chunk_s": 20.0, "outputs": {
            "tambourine": "perc~mega", "percussion": "perc~megapc", "harmonica": "harmonica~mega",
            "kick": "kick~mega", "snare": "snare~mega", "toms": "toms~mega", "hh": "hihat~mega", "drums": "drums~mega",
            "bass": "bass~mega", "guitar": "guitar~mega", "electric-guitar": "guitar~megae",
            "vocal": "vocals~mega", "lead-vocal": "vocals~megalb", "back-vocal": "vocals~megalb"}},
        {"model": "mbr_percussion_yolkispaliks.ckpt", "outputs": {"percussions": "perc~yolk"}},
        {"model": "bs_drums_xlancer.ckpt", "outputs": {"drums": "drums~xl"}},
        {"model": "bs_drums2_xlancer.ckpt", "outputs": {"drums": "drums~xl2"}},
        {"model": "bs_drums_gilliaaan.ckpt", "outputs": {"drums": "drums~gill"}},
        {"model": "bs_bass_xlancer.ckpt", "outputs": {"bass": "bass~xl"}},
        {"model": "bs_vocals_large1_unwa.ckpt", "outputs": {"vocals": "vocals~large1"}},
    ]}],
}
# per stem: the candidates of its ensemble (estimate names in the runs' models/ folders)
CANDIDATES = {
    "vocals": ["vocals~sw"] + [f"vocals~{k}" for k in VOCALS] + ["vocals~large1", "vocals~mega", "vocals~megalb"],
    "drums": ["drums", "drums~htdemucs", "drums~xl", "drums~xl2", "drums~gill", "drums~mega"],
    "bass": ["bass", "bass~htdemucs", "bass~xl", "bass~mega"],
    "guitar": ["guitar", "guitar~mega", "guitar~megae"],
    "kick": ["kick~5", "kick~6", "kick~mega"],
    "snare": ["snare~5", "snare~6", "snare~mega"],
    "toms": ["toms~5", "toms~6", "toms~mega"],
    "hh+cym": ["hh+cym~5", "hh+cym~6", "hh+cym~mega"],
    "perc": ["other", "perc~mega", "perc~megapc", "perc~yolk"],
}


def read(f: Path, dtype: str = "float32") -> np.ndarray:
    return sf.read(f, dtype=dtype, always_2d=True)[0]


def separate_run(name: str, steps: list[dict], x: np.ndarray, fs: int, song: str) -> None:
    """Separate (or find in the cache) one run of a song, every model's outputs kept."""
    scfg = config.load(ROOT / "rehearsal.toml")["stems"]
    scfg = dict(scfg, plan="bench", plans={"bench": {"steps": steps}}, cache=True, keep_models=True,
                custom_models=scfg["custom_models"] | custom_models())
    key = {"bench": song, "steps": steps, "input": round(float((x ** 2).sum()), 6)}  # a new rendering: new stems
    separate.separate(x, fs, scfg, OUT / "stems" / name / song, key)


def load_run(name: str, song: str) -> tuple[dict, dict]:
    """Each model's own outputs by stem name (back at the recording's level), and the
    run's final stems."""
    cache = OUT / "stems" / name / song
    final = {f.stem: read(f) for f in sorted(cache.glob("*.wav")) if not f.stem.startswith("_")}
    idx = json.loads((cache / "models" / "index.json").read_text())["files"]
    out = {}
    for e in idx:
        if e["model"] == "ensemble":
            continue
        out[f"{e['step']}:{e['stem']}" if e["stem"] in out else e["stem"]] = read(cache / "models" / e["file"]) * 10 ** (-e["level_db"] / 20)
    return out, final


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = ref.astype(np.float64), est.astype(np.float64)
    return float(10 * np.log10(((ref ** 2).sum() + 1e-12) / (((ref - est) ** 2).sum() + 1e-12)))


_MEL = {}


def bleed_full(ref: np.ndarray, est: np.ndarray, n_fft: int = 4096, hop: int = 1024) -> tuple[float, float]:
    """ZFTurbo's bleedless / fullness (utils/metrics.py bleed_full: torch.stft, centred,
    zero-padded, periodic Hann; 512 mel bands at sr 44100; dB with top_db 80)."""
    import librosa

    if n_fft not in _MEL:
        _MEL[n_fft] = librosa.filters.mel(sr=44100, n_fft=n_fft, n_mels=512).astype(np.float32)
    win = signal.get_window("hann", n_fft).astype(np.float32)

    def mel_db(y):
        y = np.pad(y.T.astype(np.float32), ((0, 0), (n_fft // 2, n_fft // 2)))
        fr = np.lib.stride_tricks.sliding_window_view(y, n_fft, axis=1)[:, ::hop]  # (ch, T, n_fft)
        S = np.abs(np.fft.rfft(fr * win, axis=-1)).transpose(0, 2, 1)  # (ch, F, T)
        db = 20 * np.log10(np.maximum(_MEL[n_fft] @ S, 1e-10))
        return np.maximum(db, db.max() - 80)

    d = mel_db(est) - mel_db(ref)
    pos, neg = d[d > 0], d[d < 0]
    return (float(100 / ((pos.mean() if pos.size else 0.0) + 1)), float(100 / ((-neg.mean() if neg.size else 0.0) + 1)))


def fit(G: list[np.ndarray], b: list[np.ndarray], rr: list[float]) -> np.ndarray:
    """Weights (>= 0, sum 1) maximising the mean SDR over songs, from each song's Gram
    matrix of the candidates (G), their products with the reference (b) and its energy."""
    k = len(b[0])

    def neg_mean_sdr(w):
        return -np.mean([10 * np.log10(r / max(r - 2 * w @ bb + w @ g @ w, 1e-12 * r)) for g, bb, r in zip(G, b, rr)])

    best = None
    for start in [np.eye(k)[i] * 0.7 + 0.3 / k for i in range(k)] + [np.full(k, 1 / k)]:
        res = optimize.minimize(neg_mean_sdr, start, method="SLSQP", bounds=[(0, 1)] * k,
                                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
        if best is None or res.fun < best.fun:
            best = res
    w = np.clip(best.x, 0, None)
    return w / w.sum()


def song_sdr(g, bb, r, w) -> float:
    return float(10 * np.log10(r / max(r - 2 * w @ bb + w @ g @ w, 1e-12 * r)))


def score_song(d: Path, hp: dict) -> dict:
    """Separate (or find cached) and score one song; everything it loads is freed on return."""
    fs = sf.info(d / "mix.flac").samplerate
    sos = signal.butter(hp["order"], hp["freq_hz"], "high", fs=fs, output="sos")
    filt = (lambda y: signal.sosfilt(sos, y, axis=0)) if hp["enabled"] else (lambda y: y)
    # the separations first (each in a subprocess), with nothing else held in memory
    mix = filt(read(d / "mix.flac", "float64"))
    separate_run("superset", RUNS["superset"], mix, fs, d.name)
    separate_run("cascade", RUNS["cascade"], mix, fs, d.name)
    separate_run("extra", RUNS["extra"], mix, fs, d.name)
    drums = sum(filt(read(d / f"{k}.flac", "float64")) for k in KIT if (d / f"{k}.flac").exists())
    separate_run("oracle", RUNS["oracle"], drums, fs, d.name)
    del drums
    mix = mix.astype(np.float32)
    refs = {k: filt(read(d / f"{k}.flac", "float64")).astype(np.float32)
            for k in KIT + ["vocals", "guitar", "bass", "perc"] if (d / f"{k}.flac").exists()}
    refs["drums"] = sum(refs[k] for k in KIT if k in refs)
    refs["hh+cym"] = refs.get("hihat", 0) + refs.get("cymbals", 0)
    mix_e = float((mix.astype(np.float64) ** 2).sum())
    sup, _ = load_run("superset", d.name)
    sup |= load_run("extra", d.name)[0]  # one pool of candidates
    if all(f"{k}~mega" in sup for k in ("drums", "kick", "snare", "toms")):  # its cymbals and hi-hat: the rest of its kit
        sup["hh+cym~mega"] = sup["drums~mega"] - sup["kick~mega"] - sup["snare~mega"] - sup["toms~mega"]
    orc, _ = load_run("oracle", d.name)
    cas_models, cas = load_run("cascade", d.name)
    est = {"superset": sup, "oracle": orc, "cascade": cas | {"drums": cas_models["drums"]}}
    for e in est.values():  # hi-hat + cymbals, per DrumSep model (the references cannot split them)
        for tag in ("~5", "~6", ""):
            if f"hihat{tag}" in e or f"cymbals{tag}" in e:
                e[f"hh+cym{tag}"] = e.get(f"hihat{tag}", 0) + e.get(f"cymbals{tag}", 0)
    rec: dict = {"single": {}, "gram": {}}
    for run_name, e in est.items():
        for name, y in e.items():
            stem = name.split("~")[0]
            if stem not in refs or float((refs[stem] ** 2).sum()) < mix_e * 1e-4:
                continue
            r = refs[stem][: len(y)]
            bl, fu = bleed_full(r, y)
            rec["single"][f"{run_name}:{name}"] = {"sdr": sdr(r, y), "bleedless": bl, "fullness": fu}
    for stem, cands in CANDIDATES.items():
        e = est["superset"]
        if stem not in refs or float((refs[stem] ** 2).sum()) < mix_e * 1e-4 or not all(c in e for c in cands):
            continue
        r = refs[stem]
        Y = np.stack([e[c][: len(r)].ravel() for c in cands]).astype(np.float64)
        rv = r.ravel().astype(np.float64)
        rec["gram"][stem] = {"cands": cands, "G": (Y @ Y.T).tolist(), "b": (Y @ rv).tolist(), "rr": float(rv @ rv)}
        del Y, rv
    return rec


def main() -> None:
    cfg = config.load(ROOT / "rehearsal.toml")
    hp = cfg["highpass"]
    songs = sorted(d for d in BENCH.iterdir() if (d / "meta.json").exists())
    if sys.argv[1:]:
        songs = [d for d in songs if any(a.lower() in d.name.lower() for a in sys.argv[1:])]
    OUT.mkdir(parents=True, exist_ok=True)
    cache_f = OUT / "scores.json"
    store = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    for d in songs:
        t0 = time.time()
        store[d.name] = score_song(d, hp)
        cache_f.write_text(json.dumps(store))
        print(f"{d.name} ({time.time() - t0:.0f}s)", flush=True)
    report(store)


def report(store: dict) -> None:
    songs = list(store)
    print(f"\n=== {len(songs)} songs; mean over the songs where the stem plays (n) ===")
    names = sorted({k for s in store.values() for k in s["single"]})
    rows = {}
    for k in names:
        v = [store[s]["single"][k] for s in songs if k in store[s]["single"]]
        rows[k] = {m: float(np.mean([x[m] for x in v])) for m in ("sdr", "bleedless", "fullness")} | {"n": len(v)}
    print(f"{'estimate':34s} {'SDR':>6s} {'bleedl':>7s} {'fullns':>7s}  n")
    for k in sorted(rows, key=lambda k: (k.split(":")[1].split("~")[0], k)):
        r = rows[k]
        print(f"{k:34s} {r['sdr']:6.2f} {r['bleedless']:7.2f} {r['fullness']:7.2f}  {r['n']}")
    print("\nensembles: weights fitted on all songs, SDR in-sample and leave-one-song-out")
    weights = {}
    for stem, cands in CANDIDATES.items():
        have = [s for s in songs if store[s]["gram"].get(stem, {}).get("cands") == cands]
        if len(have) < 2:
            continue
        G = [np.array(store[s]["gram"][stem]["G"]) for s in have]
        b = [np.array(store[s]["gram"][stem]["b"]) for s in have]
        rr = [store[s]["gram"][stem]["rr"] for s in have]
        w = fit(G, b, rr)
        ins = np.mean([song_sdr(g, bb, r, w) for g, bb, r in zip(G, b, rr)])
        loo = np.mean([song_sdr(G[i], b[i], rr[i], fit(G[:i] + G[i + 1:], b[:i] + b[i + 1:], rr[:i] + rr[i + 1:]))
                       for i in range(len(have))])
        best1 = max(range(len(cands)), key=lambda j: np.mean([song_sdr(g, bb, r, np.eye(len(cands))[j]) for g, bb, r in zip(G, b, rr)]))
        b1 = np.mean([song_sdr(g, bb, r, np.eye(len(cands))[best1]) for g, bb, r in zip(G, b, rr)])
        weights[stem] = dict(zip(cands, [round(float(x), 3) for x in w]))
        print(f"  {stem:7s} best single {cands[best1]:16s} {b1:6.2f} | ensemble {ins:6.2f} (loo {loo:6.2f}, n {len(have)})  "
              + "  ".join(f"{c} {x:.2f}" for c, x in zip(cands, w)))
    (OUT / "results.json").write_text(json.dumps({"songs": songs, "mean": rows, "weights": weights}, indent=1))
    print(f"\nwritten {OUT / 'results.json'}")


if __name__ == "__main__":
    if sys.argv[1:] == ["--report"]:
        report(json.loads((OUT / "scores.json").read_text()))
    else:
        main()
