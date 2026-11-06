"""Build the separation benchmark: rehearsal-like room recordings with known stems.

Each item is a 45 s excerpt of a Cambridge-MT raw multitrack (via RawStems, as
tools/build_raw_references.py) with drums, bass, electric guitars and vocals (and a
tambourine when there is one). Every source is placed in a simulated rehearsal room
(pyroomacoustics image sources, RT60 0.4-0.8 s) and picked up by an X/Y pair of cardioids,
like the Zoom H4essential. The kit is built from its close mics (kick, snare, toms, hi-hat,
cymbal spots) and the overheads high-passed as the cymbals; the vocals come out of two PA
speakers; two guitar amps, one bass amp. Levels follow this band's own balance (the learnt
group targets, the kit pieces' levels in its separated stems), randomised a little.

The reference stems are what each source contributes at the recorder (room included): the
mix is their exact sum plus a low noise floor, so a perfect separation scores infinite SDR.
The overheads, used as the cymbals, first lose what the kick, snare and tom mics predict in
them below about 3 kHz (least squares per STFT bin), so the snare's body leaves the cymbals.
Known limits: close mics carry bleed (snare top: hi-hat; toms: cymbals, reduced by gating
the toms), the overheads keep the snare's top octaves and the hi-hat, so the hi-hat /
cymbals split (and to a lesser degree the snare) are the least exact references.

The fetched tracks are kept, cut to the excerpt, in <song>/_tracks/: a new rendering
(RENDER) only redoes the room and the mix.

Written to processed/experiments/sep_bench/data/<song>/ (FLAC 24-bit, 48 kHz stereo, all
files of a song at one common gain); the multitracks may be used, not redistributed.

  uv run --with pyroomacoustics --with requests python experiments/sep_bench/build.py [n_songs]
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import build_raw_references as raw  # noqa: E402

from rehearsal_master import loudness  # noqa: E402

OUT = ROOT / "processed" / "experiments" / "sep_bench" / "data"  # the data drive: the system disk is nearly full
INDEX = OUT / "rawstems_index.json"
FS = 48000
WIN_S = 45.0
FETCH_MB = 28  # compressed bytes per track: 2-3 minutes of a dense track
raw.FETCH = FETCH_MB << 20

# role -> (folder pattern, name pattern, exclude pattern, max tracks)
NOT_DRUM = r"sample|trig|fx|room|amb|squash|crush|loop|break|verb|comp"
ROLES = {
    "kick": (r"Rhy/DK", r"kick|bd\b|bassdrum", NOT_DRUM + r"|sub|side|thump", 1),
    "snare": (r"Rhy/DK", r"snare|sn\b", NOT_DRUM + r"|bot|down|under|side|btm", 1),
    "toms": (r"Rhy/DK", r"tom", NOT_DRUM + r"|snare|bott|custom", 3),
    "hihat": (r"Rhy/DK", r"hi[ _-]?hat|hh\b|hat", NOT_DRUM, 1),
    "spots": (r"Rhy/DK", r"ride|crash|cymbal|splash|china", NOT_DRUM, 2),
    "overheads": (r"Rhy/DK", r"overhead|ovhd|(^|[^a-z])oh([^a-z]|$)", NOT_DRUM + r"|mono", 2),
    "bass": (r"Bass|Bss", r".", r"synth|sub|fx|keys|moog|di\b|di\.flac|direct", 1),
    "bass_di": (r"Bass|Bss", r"di\b|di\.flac|direct", r"synth|sub|fx", 1),
    "guitar": (r"Gtr/EG", r".", r"fx|room|amb|di\b|di\.flac|direct|verb|delay", 4),
    "lead": (r"Voc/LV", r".", r"fx|room|amb|verb|delay|dt|double|dbl|harm|whisper|adlib", 1),
    "backing": (r"Voc/BV", r".", r"fx|room|amb|verb|delay", 3),
    "perc": (r"Rhy/PERC", r"tamb|shaker", r"fx|room|amb", 1),
}
REQUIRED = ["kick", "snare", "toms", "overheads", "lead", "guitar"]
NOT_BAND = r"^(Synth|Kbs|Orch|Gtr/AG)"  # this band has none of these


def pick(tracks: list[str], index: dict, song: str) -> dict[str, list[str]]:
    out = {}
    for role, (fpat, npat, excl, n) in ROLES.items():
        c = [t for t in tracks if re.match(fpat, t.rpartition("/")[0])
             and re.search(npat, t.rpartition("/")[2], re.I) and not re.search(excl, t.rpartition("/")[2], re.I)]
        c.sort(key=lambda t: -index[f"{song}/{t}"]["size"])  # most content first
        if role in ("toms", "overheads", "spots"):
            c = sorted(c[:n])
        out[role] = c[:n]
    if not out["bass"]:
        out["bass"] = out["bass_di"]
    del out["bass_di"]
    return out


def fetch(path: str, index: dict) -> np.ndarray | None:
    for _ in range(3):
        try:
            data = raw._member_prefix(index[path])
            break
        except (requests.RequestException, zlib.error) as e:
            print(f"    {path}: {e}", file=sys.stderr)
    else:
        return None
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ar", str(FS), "-f", "f32le", "pipe:1"],
                       input=data, capture_output=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=channels", "-of", "csv=p=0", "pipe:0"],
                           input=data[: 1 << 20], capture_output=True).stdout
    ch = int(probe.split()[0]) if probe.split() else 1
    x = np.frombuffer(p.stdout, dtype=np.float32)
    x = x[: len(x) // ch * ch].reshape(-1, ch)
    return x if len(x) > FS * 20 else None


def frames_db(x: np.ndarray, hop: int = FS // 10) -> np.ndarray:
    m = x if x.ndim == 1 else x.mean(axis=1)
    n = len(m) // hop * hop
    return 10 * np.log10((m[:n].reshape(-1, hop) ** 2).mean(axis=1) + 1e-20)


def gate(x: np.ndarray, depth_db: float = 18.0, hold_s: float = 0.3) -> np.ndarray:
    """Toms: keep only around their own hits (their mics are full of cymbals)."""
    hop = FS // 200
    e = frames_db(x, hop)
    on = e > np.percentile(e, 99.9) - depth_db
    k = int(hold_s * FS / hop)
    on = np.convolve(on.astype(float), np.ones(k), "full")[: len(on)] > 0
    g = np.repeat(on.astype(float), hop)
    g = np.pad(g, (0, max(0, len(x) - len(g))))[: len(x)]
    g = np.convolve(g, np.ones(FS // 200) / (FS // 200), "same")  # 5 ms ramps
    return x * (g[:, None] if x.ndim == 2 else g)


def p95(x: np.ndarray) -> float:
    return float(np.percentile(frames_db(x), 95))


def audible(x: np.ndarray) -> bool:
    return p95(x) > -90.0


def place(rnd: random.Random):
    """Room, recorder and source positions (metres)."""
    L, W, H = rnd.uniform(5, 8), rnd.uniform(4, 6), rnd.uniform(2.6, 3.2)
    rt60 = rnd.uniform(0.4, 0.8)
    cx, cy = L / 2 + rnd.uniform(-0.6, 0.6), W - 1.3
    clamp = lambda p: [min(max(p[0], 0.3), L - 0.3), min(max(p[1], 0.3), W - 0.3), min(max(p[2], 0.3), H - 0.3)]
    pos = {
        "kick": [cx, cy, 0.35], "snare": [cx - 0.3, cy - 0.35, 0.65], "hihat": [cx - 0.6, cy - 0.3, 0.95],
        "toms": [[cx - 0.15, cy, 0.8], [cx + 0.15, cy, 0.8], [cx + 0.5, cy - 0.3, 0.6]],
        "spots": [[cx + 0.6, cy - 0.2, 1.1], [cx - 0.5, cy + 0.1, 1.3]],
        "overheads": [[cx - 0.6, cy, 1.6], [cx + 0.6, cy, 1.6]],
        "bass": [L * 0.8, W - 1.0, 0.5],
        "amps": [[L * 0.15, W * 0.55, 0.6], [L * 0.85, W * 0.45, 0.6]],
        "pa": [[0.6, 0.6, 1.8], [L - 0.6, 0.6, 1.8]],
        "perc": [cx - 1.3, cy - 1.0, 1.2],
    }
    d = rnd.uniform(2.0, 3.5)
    mic = clamp([cx + rnd.uniform(-1.0, 1.0), cy - d, rnd.uniform(1.2, 1.5)])
    walk = lambda v: [walk(u) for u in v] if isinstance(v[0], list) else clamp(v)
    return {"dims": [L, W, H], "rt60": rt60, "mic": mic, "pos": {k: walk(v) for k, v in pos.items()}}


def rirs(geo: dict, points: list[list[float]]) -> list[np.ndarray]:
    """(n, 2) impulse responses from each point to the X/Y pair (left capsule first)."""
    import pyroomacoustics as pra
    from pyroomacoustics.directivities import Cardioid, DirectionVector

    absorb, order = pra.inverse_sabine(geo["rt60"], geo["dims"])
    room = pra.ShoeBox(geo["dims"], fs=FS, materials=pra.Material(absorb), max_order=min(order, 60),
                       air_absorption=True)
    for p in points:
        room.add_source(p)
    m = np.array(geo["mic"])[:, None].repeat(2, axis=1)
    room.add_microphone_array(m, directivity=[Cardioid(DirectionVector(azimuth=a, colatitude=90, degrees=True))
                                              for a in (135, 45)])  # facing the band (+y), 90 degrees apart
    room.compute_rir()
    out = []
    for s in range(len(points)):
        n = max(len(room.rir[0][s]), len(room.rir[1][s]))
        out.append(np.stack([np.pad(room.rir[c][s], (0, n - len(room.rir[c][s]))) for c in range(2)], axis=1))
    return out


def best_window(groups: dict[str, np.ndarray], n: int) -> tuple[int, float]:
    """Start of the WIN_S window where the least active group is most active."""
    act = {}
    for g, x in groups.items():
        e = frames_db(x)
        act[g] = (e > np.percentile(e, 95) - 25).astype(float)
    w = int(WIN_S * 10)
    m = min(len(a) for a in act.values())
    best, score = 0, -1.0
    for s in range(0, max(1, min(m, n // (FS // 10)) - w), 5):
        sc = min(a[s:s + w].mean() for a in act.values())
        if sc > score:
            best, score = s, sc
    return best * (FS // 10), score


RENDER = 2  # bump when the rendering changes: songs are re-rendered from their kept tracks


def fetch_tracks(song: str, sel: dict, index: dict, d: Path) -> dict | None:
    """The chosen tracks of the song, cut to the window (plus the room's lead-in), kept in
    <song>/_tracks/ so a new rendering never downloads again."""
    tdir = d / "_tracks"
    if (tdir / "tracks.json").exists():
        info = json.loads((tdir / "tracks.json").read_text())
        if info.get("skip"):
            return None
        info["tr"] = {r: [(p, sf.read(tdir / f, dtype="float32", always_2d=True)[0]) for p, f in v]
                      for r, v in info["files"].items()}
        return info
    tr = {}
    for role, paths in sel.items():
        for p in paths:
            x = fetch(f"{song}/{p}", index)
            if x is not None:
                tr.setdefault(role, []).append((p, x))
    tdir.mkdir(parents=True, exist_ok=True)
    if any(r not in tr for r in REQUIRED) or "bass" not in tr:
        print(f"  {d.name}: missing {[r for r in REQUIRED + ['bass'] if r not in tr]}", flush=True)
        (tdir / "tracks.json").write_text(json.dumps({"skip": "missing parts"}))
        return None
    n = min(len(x) for v in tr.values() for _, x in v)  # tracks are song-length: the shortest fetch
    mono = lambda v: sum(x[:n].mean(axis=1) for _, x in v)
    start, score = best_window({"vocals": mono(tr["lead"]), "guitar": mono(tr["guitar"]), "bass": mono(tr["bass"]),
                                "drums": mono(tr["kick"]) + mono(tr["snare"])}, n)
    if score < 0.5:
        print(f"  {d.name}: no window where everything plays ({score:.2f})", flush=True)
        (tdir / "tracks.json").write_text(json.dumps({"skip": f"activity {score:.2f}"}))
        return None
    pre = int(1.5 * FS)  # the room's tail from before the window
    a, b = max(0, start - pre), start + int(WIN_S * FS)
    cut = lambda x: np.pad(x[a:b], ((0, max(0, b - a - len(x[a:b]))), (0, 0)))
    files = {}
    for role, v in tr.items():
        files[role] = []
        for i, (p, x) in enumerate(v):
            f = f"{role}_{i}.flac"
            sf.write(tdir / f, cut(x), FS, subtype="PCM_24")
            files[role].append((p, f))
    info = {"start": start - a, "window_start_s": round(start / FS, 2), "activity": round(score, 2), "files": files}
    (tdir / "tracks.json").write_text(json.dumps(info, indent=1))
    info["tr"] = {r: [(p, sf.read(tdir / f, dtype="float32", always_2d=True)[0]) for p, f in v] for r, v in files.items()}
    return info


def clean_overheads(oh: np.ndarray, close: list[np.ndarray], lo: float = 150.0, hi: float = 3500.0) -> np.ndarray:
    """The overheads minus what the kick, snare and tom mics predict in them (per STFT bin,
    least squares), below about 3 kHz where the drums dominate: the snare's body and crack
    leave the "cymbals" reference. Above, the cymbals would be cancelled too, so it stops."""
    nf, hop = 4096, 1024
    stft = lambda y: signal.stft(y, FS, nperseg=nf, noverlap=nf - hop)[2].astype(np.complex64)
    X = np.stack([stft(c) for c in close])  # (K, F, T)
    f = np.fft.rfftfreq(nf, 1 / FS)
    w = np.clip((f - lo * 0.7) / (lo * 0.3), 0, 1) * np.clip((hi - f) / (hi * 0.3), 0, 1)  # soft band edges
    out = np.empty_like(oh)
    for c in range(oh.shape[1]):
        Y = stft(oh[:, c])
        for k in np.nonzero(w)[0]:
            A = X[:, k, :].T  # (T, K)
            G = A.conj().T @ A
            h = np.linalg.solve(G + 1e-3 * np.trace(G).real / len(close) * np.eye(len(close)), A.conj().T @ Y[k])
            Y[k] -= w[k] * (A @ h)
        y = signal.istft(Y, FS, nperseg=nf, noverlap=nf - hop)[1]
        out[:, c] = np.pad(y, (0, max(0, len(oh) - len(y))))[: len(oh)]
    return out


def build(song: str, sel: dict, index: dict, rnd: random.Random) -> bool:
    name = re.sub(r"[^A-Za-z0-9]+", "_", song)[:60]
    d = OUT / name
    if (d / "meta.json").exists() and json.loads((d / "meta.json").read_text()).get("render") == RENDER:
        print(f"  {name}: exists", flush=True)
        return True
    info = fetch_tracks(song, sel, index, d)
    if info is None:
        return False
    tr, s0 = info["tr"], info["start"]
    geo = place(rnd)
    P = geo["pos"]
    # sources: (reference stem, mono signal, position)
    src: list[tuple[str, np.ndarray, list[float]]] = []
    jit = lambda: rnd.uniform(-2.0, 2.0)
    kick = tr["kick"][0][1].mean(axis=1)
    k95, k999 = p95(kick), float(np.percentile(frames_db(kick), 99.9))
    # levels re the kick as in this band's separated stems: snare +2.5 (P95), tom hits -2 (P99.9),
    # hi-hat -22, cymbals -9 (P95)
    src.append(("kick", kick, P["kick"]))
    sn = tr["snare"][0][1].mean(axis=1)
    src.append(("snare", sn * 10 ** ((k95 + 2.5 + jit() - p95(sn)) / 20), P["snare"]))
    for i, (_, x) in enumerate(tr["toms"]):
        t = x.mean(axis=1)
        e = frames_db(t)
        if np.percentile(e, 99.9) - np.percentile(e, 50) < 20:  # no hits here, only bleed
            continue
        t = gate(t)
        src.append(("toms", t * 10 ** ((k999 - 2.0 + jit() - float(np.percentile(frames_db(t), 99.9))) / 20), P["toms"][i]))
    if "hihat" in tr and audible(h := tr["hihat"][0][1].mean(axis=1)):
        src.append(("hihat", h * 10 ** ((k95 - 22.0 + jit() - p95(h)) / 20), P["hihat"]))
    for i, (_, x) in enumerate(tr.get("spots", [])):
        s = x.mean(axis=1)
        if audible(s):
            src.append(("cymbals", s * 10 ** ((k95 - 14.0 + jit() - p95(s)) / 20), P["spots"][i]))
    oh = np.concatenate([x for _, x in tr["overheads"]], axis=1)[:, :2].astype(np.float64)
    oh = clean_overheads(oh, [kick, sn] + [x.mean(axis=1) for _, x in tr["toms"]])
    oh = signal.sosfilt(signal.butter(4, 400, "high", fs=FS, output="sos"), oh, axis=0)
    g = 10 ** ((k95 - 9.0 + jit() - p95(oh)) / 20)
    if oh.shape[1] == 2:
        src += [("cymbals", oh[:, c] * g, P["overheads"][c]) for c in range(2)]
    else:
        src.append(("cymbals", oh[:, 0] * g, list(np.mean(P["overheads"], axis=0))))
    src.append(("bass", tr["bass"][0][1].mean(axis=1), P["bass"]))
    gl = [y for y in (x.mean(axis=1) for _, x in tr["guitar"]) if audible(y)]
    if not gl:
        print(f"  {name}: guitars silent in the window", flush=True)
        return False
    ref = max(p95(x) for x in gl)
    for i, x in enumerate(gl):
        src.append(("guitar", x * 10 ** ((ref + jit() - p95(x)) / 20), P["amps"][i % 2]))
    lead = tr["lead"][0][1].mean(axis=1)
    voc = [lead] + [x * 10 ** ((p95(lead) - 8.0 + jit() - p95(x)) / 20)
                    for x in (y.mean(axis=1) for _, y in tr.get("backing", [])) if audible(x)]
    for x in voc:
        for sp in P["pa"]:
            src.append(("vocals", x * 0.7071, sp))
    if "perc" in tr and audible(tr["perc"][0][1].mean(axis=1)):
        src.append(("perc", tr["perc"][0][1].mean(axis=1), P["perc"]))
    # the room
    hs = rirs(geo, [p for _, _, p in src])
    n = len(kick)
    stems: dict[str, np.ndarray] = {}
    for (stem, x, _), h in zip(src, hs):
        y = signal.fftconvolve(x[:, None], h, axes=0)[:n]
        stems[stem] = stems.get(stem, 0) + y
    stems = {k: v[s0:] for k, v in stems.items()}  # drop the lead-in
    # group levels: this band's balance (LU re the remix, the learnt targets), randomised
    target = {"vocals": -8.5, "guitar": -8.2, "bass": -7.3, "perc": -16.0}
    drums_lu = loudness.integrated(sum(stems[k] for k in ("kick", "snare", "toms", "hihat", "cymbals") if k in stems), FS)
    gains = {}
    for k in target:
        if k in stems:
            gains[k] = (drums_lu + target[k] + 3.1 + jit()) - loudness.integrated(stems[k], FS)
            stems[k] = stems[k] * 10 ** (gains[k] / 20)
    mix = sum(stems.values())
    noise = np.cumsum(np.random.default_rng(rnd.randrange(1 << 30)).standard_normal(mix.shape), axis=0)
    noise = signal.sosfilt(signal.butter(1, 20, "high", fs=FS, output="sos"), noise, axis=0)  # brown, minus DC
    noise *= 10 ** ((loudness.integrated(mix, FS) - 58.0 - loudness.integrated(noise, FS)) / 20)
    mix = mix + noise
    scale = min(0.5 / float(np.abs(mix).max()), 0.95 / max(float(np.abs(v).max()) for v in stems.values()))
    for f in d.glob("*.flac"):  # an older rendering
        f.unlink()
    sf.write(d / "mix.flac", (mix * scale).astype(np.float32), FS, subtype="PCM_24")
    for k, v in stems.items():
        sf.write(d / f"{k}.flac", (v * scale).astype(np.float32), FS, subtype="PCM_24")
    meta = {"song": song, "render": RENDER, "start_s": info["window_start_s"], "activity": info["activity"], "room": geo,
            "tracks": {r: [p for p, _ in v] for r, v in tr.items()}, "gains_db": {k: round(v, 1) for k, v in gains.items()},
            "lufs": {k: round(loudness.integrated(v * scale, FS), 1) for k, v in stems.items()} | {
                "mix": round(loudness.integrated(mix * scale, FS), 1)}}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {name}: window {info['window_start_s']:.0f}s (activity {info['activity']:.2f}), RT60 {geo['rt60']:.2f}s, "
          + ", ".join(f"{k} {v:.1f}" for k, v in meta["lufs"].items()), flush=True)
    return True


def main(n_songs: int = 12) -> None:
    index = json.loads(INDEX.read_text())
    songs: dict[str, list[str]] = {}
    for p in index:
        if not p.endswith("/"):
            s, _, rest = p.partition("/")
            songs.setdefault(s, []).append(rest)
    cands = []
    for s, t in sorted(songs.items()):
        if sum(bool(re.match(NOT_BAND, x)) for x in t) > 2:
            continue
        sel = pick(t, index, s)
        if all(sel[r] for r in REQUIRED) and sel["bass"]:
            cands.append((s, sel))
    print(f"{len(cands)} band-like songs with all the parts", flush=True)
    rnd = random.Random(2026)
    rnd.shuffle(cands)
    done = 0
    for s, sel in cands:
        if done >= n_songs:
            break
        print(f"[{done + 1}/{n_songs}] {s}", flush=True)
        done += build(s, sel, index, random.Random(s))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
