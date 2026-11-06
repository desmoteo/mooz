"""Build the studio drum sample library: kick, snare, tom and (closed) hi-hat one-shots from
raw close-mic tracks of the Cambridge-MT multitracks (via RawStems, as
tools/build_raw_references.py), for the "library" part of the drum augmentation.

From each track's first minute the clean hits (no other hit of it within the sample's
length) are aligned and averaged, in two velocity layers (loud: the top quarter; soft:
around the median). Hi-hat hits are mostly noise, which averaging would cancel (a dull,
blurred hat): for them one real hit per layer (the clean hit nearest the 85th / 50th
velocity percentile). Written to ~/.cache/rehearsal-master/drum_samples/<drum>/ as FLAC with
an index.json (pitch, brightness, source). Kept out of the repository: the multitracks may
be used, not redistributed.

  uv run --with remotezip --with requests python tools/build_drum_samples.py [per_drum] [index.json] [drums]

drums: e.g. "hihat" to build only those (merged into the existing index.json).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_raw_references import FS, PICK, URL, _decode, _member_prefix  # noqa: E402

from rehearsal_master import augment  # noqa: E402

OUT = Path.home() / ".cache" / "rehearsal-master" / "drum_samples"
DRUMS = {"kick": (40.0, 150.0, 0.45), "snare": (150.0, 400.0, 0.4), "toms": (70.0, 400.0, 0.7),
         "hihat": (5000.0, 14000.0, 0.2)}  # band, length s
SINGLE = {"hihat"}  # one real hit per layer, not an average
# cymbals: from crash / ride spot mics, single real hits, each as long as it rings before the next hit (at most
# length_s, at least min_s); "bell": the ride tracks' clearly tonal hits (bell strokes: a peaky spectrum)
CYMBALS = {"crash": (r"crash", 2000.0, 14000.0, 2.0, 0.8), "ride": (r"ride", 2000.0, 14000.0, 1.0, 0.35),
           "bell": (r"ride", 2000.0, 14000.0, 1.2, 0.5)}  # file pattern, band, length_s, min_s


def flatness(seg: np.ndarray, lo: float = 3000.0, hi: float = 14000.0) -> float:
    """Spectral flatness in the cymbal band (geometric / arithmetic mean of the power): low = tonal (a bell);
    the same measure augment.cymbal_augment classifies the song's hits by."""
    f, p = signal.periodogram(seg * np.hanning(len(seg)), FS, nfft=1 << 13)
    m = (f >= lo) & (f <= hi)
    q = p[m] + 1e-20
    return float(np.exp(np.mean(np.log(q))) / np.mean(q))


def cymbal_samples(x: np.ndarray, kind: str) -> dict[str, np.ndarray]:
    """Loud and soft single hits of a crash / ride (bow) / bell from a spot-mic track."""
    _, lo, hi, length_s, min_s = CYMBALS[kind]
    starts, vel = augment.cymbal_hits(x[:, None], FS, lo, hi)  # rises, not peaks in a crash's noisy decay
    keep = vel >= vel.max(initial=0) * 10 ** (-20 / 20)  # real hits, not bleed or noise in the gaps
    starts, vel = starts[keep], vel[keep]
    pre = int(0.002 * FS)
    if len(starts) < 4:
        return {}
    gap = np.diff(np.append(starts, len(x))) / FS
    fl = np.array([flatness(x[s : s + int(0.08 * FS)]) for s in starts])
    med = np.median(fl)
    tonal = fl < 0.6 * med
    ok = [i for i in range(len(starts)) if starts[i] > pre and gap[i] >= min_s and (tonal[i] if kind == "bell" else not tonal[i])]
    if len(ok) < (1 if kind == "bell" else 3):
        return {}
    v = vel[ok]
    out = {}
    for name, q in (("loud", 85), ("soft", 50)):
        i = min(ok, key=lambda j: abs(vel[j] - np.percentile(v, q)))
        L = int(min(length_s, gap[i] - 0.01) * FS)
        fade = np.ones(L)
        k = int(0.3 * L)
        fade[-k:] = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
        s = x[starts[i] - pre : starts[i] - pre + L] * fade
        out[name] = (s / (np.abs(s).max() + 1e-12) * 0.5).astype(np.float32)
    return out


def samples_of(x: np.ndarray, lo: float, hi: float, length_s: float, single: bool = False) -> dict[str, np.ndarray]:
    """Loud and soft one-shots (mono) from a raw close-mic track (single: one real hit each)."""
    xs = x[:, None]
    starts, vel = augment.drum_hits(xs, FS, lo, hi)
    L, pre = int(length_s * FS), int(0.002 * FS)
    ok = [i for i in range(len(starts)) if starts[i] > pre and starts[i] + L < len(x)
          and not (i + 1 < len(starts) and starts[i + 1] < starts[i] + L)]
    if len(ok) < 6:
        return {}
    v = vel[ok]
    layers = {"loud": [i for i in ok if vel[i] >= np.percentile(v, 75)],
              "soft": [i for i in ok if np.percentile(v, 35) <= vel[i] <= np.percentile(v, 65)]}
    fade = np.ones(L)
    k = int(0.3 * L)
    fade[-k:] = np.cos(np.linspace(0, np.pi / 2, k)) ** 2
    out = {}
    if single:
        for name, q in (("loud", 85), ("soft", 50)):
            i = min(ok, key=lambda j: abs(vel[j] - np.percentile(v, q)))
            s = x[starts[i] - pre : starts[i] - pre + L] * fade
            out[name] = (s / (np.abs(s).max() + 1e-12) * 0.5).astype(np.float32)
        return out
    for name, members in layers.items():
        if len(members) < 3:
            continue
        ref = x[starts[members[-1]] - pre : starts[members[-1]] - pre + L][:, None]
        acc = np.zeros(L)
        for i in members:
            lag = augment._lag(ref[: int(0.03 * FS)], xs, starts[i] - pre, int(0.002 * FS))
            seg = x[starts[i] - pre + lag : starts[i] - pre + lag + L]
            if len(seg) == L:
                acc += seg / (vel[i] + 1e-12)
        s = acc / len(members) * fade
        out[name] = (s / (np.abs(s).max() + 1e-12) * 0.5).astype(np.float32)
    return out


def describe(s: np.ndarray) -> dict:
    f, p = signal.periodogram(s[: int(0.15 * FS)] * np.hanning(int(0.15 * FS)), FS, nfft=1 << 15)
    m = (f >= 40) & (f <= 400)
    return {"pitch_hz": round(float(f[m][np.argmax(p[m])]), 1),
            "centroid_hz": round(float(np.sum(f * p) / (np.sum(p) + 1e-20)), 0)}


def main(per_drum: int = 12) -> None:
    index = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    only = sys.argv[3].split(",") if len(sys.argv) > 3 else list(DRUMS) + list(CYMBALS)
    if index is None:
        import remotezip

        index = {}
        for i in range(1, 7):
            with remotezip.RemoteZip(URL.format(i)) as z:
                for zi in z.infolist():
                    index[zi.filename] = {"zip": i, "offset": zi.header_offset, "csize": zi.compress_size}
    rnd = random.Random(11)
    lib = json.loads((OUT / "index.json").read_text())["drums"] if (OUT / "index.json").exists() else {}
    seen: set = set()
    for kind, (pat, *_rest) in CYMBALS.items():
        if kind not in only:
            continue
        cands = {}
        for path in index:
            song, _, rest = path.partition("/")
            folder, _, fname = rest.rpartition("/")
            if folder == "Rhy/DK" and re.search(pat, fname, re.I) and not re.search(r"sample|trig|room|amb|fx", fname, re.I):
                cands.setdefault(song, []).append(path)
        songs = sorted(cands)
        rnd.shuffle(songs)
        d = OUT / kind
        d.mkdir(parents=True, exist_ok=True)
        lib[kind] = []
        for song in songs:
            if len(lib[kind]) >= per_drum:
                break
            path = sorted(cands[song])[0]
            dup = (index[path]["csize"], index[path].get("size"))
            if dup in seen:  # RawStems holds some identical files under several songs' names
                continue
            seen.add(dup)
            try:
                x = _decode(_member_prefix(index[path]))
            except Exception as e:  # network, truncated stream: next one
                print(f"  {path}: {e}", file=sys.stderr)
                continue
            if x is None:
                continue
            smp = cymbal_samples(x.astype(np.float64), kind)
            if "loud" not in smp:
                continue
            name = re.sub(r"[^A-Za-z0-9]+", "_", song)[:60]
            entry = {"name": name, "source": path} | describe(smp["loud"]) | {"flatness": round(flatness(smp["loud"][: int(0.08 * FS)]), 3),
                                                                             "length_s": round(len(smp["loud"]) / FS, 2)}
            for layer, sm in smp.items():
                sf.write(d / f"{name}_{layer}.flac", sm, FS, subtype="PCM_24")
                entry[layer] = f"{name}_{layer}.flac"
            lib[kind].append(entry)
            print(f"{kind:6s} {len(lib[kind]):2d}/{per_drum} centroid {entry['centroid_hz']:6.0f} Hz  flatness {entry['flatness']:.3f}  "
                  f"{entry['length_s']:.2f} s  {path}", flush=True)
    for drum, (lo, hi, length) in DRUMS.items():
        if drum not in only:
            continue
        group, pat, excl = PICK[drum]
        cands = {}
        for path in index:
            song, _, rest = path.partition("/")
            folder, _, fname = rest.rpartition("/")
            if folder == group and re.search(pat, fname, re.I) and not re.search(excl, fname, re.I):
                cands.setdefault(song, []).append(path)
        songs = sorted(cands)
        rnd.shuffle(songs)
        d = OUT / drum
        d.mkdir(parents=True, exist_ok=True)
        lib[drum] = []
        for song in songs:
            if len(lib[drum]) >= per_drum:
                break
            path = sorted(cands[song])[0]
            dup = (index[path]["csize"], index[path].get("size"))
            if dup in seen:  # RawStems holds some identical files under several songs' names
                continue
            seen.add(dup)
            try:
                x = _decode(_member_prefix(index[path]))
            except Exception as e:  # network, truncated stream: next one
                print(f"  {path}: {e}", file=sys.stderr)
                continue
            if x is None:
                continue
            smp = samples_of(x.astype(np.float64), lo, hi, length, drum in SINGLE)
            if "loud" not in smp:
                continue
            name = re.sub(r"[^A-Za-z0-9]+", "_", song)[:60]
            entry = {"name": name, "source": path} | describe(smp["loud"])
            for layer, s in smp.items():
                sf.write(d / f"{name}_{layer}.flac", s, FS, subtype="PCM_24")
                entry[layer] = f"{name}_{layer}.flac"
            lib[drum].append(entry)
            print(f"{drum:6s} {len(lib[drum]):2d}/{per_drum} {entry['pitch_hz']:6.1f} Hz  centroid {entry['centroid_hz']:6.0f} Hz  {path}", flush=True)
    (OUT / "index.json").write_text(json.dumps({"fs": FS, "drums": lib}, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
