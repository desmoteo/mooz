"""Build mooz/references/raw_multitracks.json: the average tone of raw,
unprocessed studio tracks per instrument, the target of the tone match's "reference" part.

Source: RawStems (https://huggingface.co/datasets/yongyizang/RawStems), the raw multitracks
of the Cambridge-MT "Mixing Secrets" library (https://www.cambridge-mt.com/ms/mtk/), used
under its terms; only averaged spectra are kept. The archives are 175 GB, so only the start
of each chosen track (about a minute) is fetched with HTTP range requests and decoded with
ffmpeg, one track per song and instrument.

  uv run --with remotezip --with requests python tools/build_raw_references.py [per_instrument]
"""

from __future__ import annotations

import json
import random
import re
import struct
import subprocess
import sys
import zlib
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mooz import eq  # noqa: E402
from mooz.profile import NFFT, _frames  # noqa: E402

URL = "https://huggingface.co/datasets/yongyizang/RawStems/resolve/main/RawStems_{}.zip"
OUT = Path(__file__).resolve().parents[1] / "mooz" / "references" / "raw_multitracks.json"
FS = 48000
FETCH = 8 << 20  # compressed bytes per track (~45-60 s of 24-bit mono FLAC)
# instrument -> (group folder, file-name pattern, exclude pattern)
PICK = {
    "kick": ("Rhy/DK", r"kick(in|out)?\s*\d?\.flac$|kick[ _-]?(in|out)\b", r"sample|trig|sub|amb|room|fx|squash"),
    "snare": ("Rhy/DK", r"snare(top|up)?\s*\d?\.flac$|snare[ _-]?top\b", r"sample|trig|bot|down|under|room|amb|fx|squash"),
    "toms": ("Rhy/DK", r"(floor|rack|hi|lo|low|mid)?\s*tom\s*\d?\.flac$", r"sample|trig|room|amb|fx"),
    "hihat": ("Rhy/DK", r"(hi[ _-]?hat|hat)s?(closed)?\s*\d?\.flac$", r"sample|trig|room|fx|open"),
    "cymbals": ("Rhy/DK", r"(ride|crash|cymbals?)\s*\d?\.flac$", r"sample|trig|room|fx"),
    "bass": ("Bass", r"\.flac$", r"synth|sub|fx|keys|moog"),
    "guitar": ("Gtr/EG", r"\.flac$", r"fx|room|amb|di\b|di\.flac|acoustic"),
    "vocals": ("Voc/LV", r"\.flac$", r"fx|room|amb|verb|delay|double|dbl|whisper|harm"),
}


def _member_prefix(info: dict) -> bytes:
    """The first FETCH compressed bytes of a zip member, inflated."""
    url = URL.format(info["zip"])
    h = requests.get(url, headers={"Range": f"bytes={info['offset']}-{info['offset'] + 29}"}, timeout=60).content
    name_len, extra_len = struct.unpack("<HH", h[26:30])
    start = info["offset"] + 30 + name_len + extra_len
    end = start + min(info["csize"], FETCH) - 1
    raw = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, timeout=300).content
    return zlib.decompressobj(-15).decompress(raw)


def _decode(flac: bytes) -> np.ndarray | None:
    p = subprocess.run(["ffmpeg", "-v", "quiet", "-i", "pipe:0", "-ac", "1", "-ar", str(FS), "-f", "f32le", "pipe:1"],
                       input=flac, capture_output=True)
    x = np.frombuffer(p.stdout, dtype=np.float32)
    return x if len(x) > FS * 10 else None


def tone(x: np.ndarray, grid: np.ndarray) -> np.ndarray | None:
    """As profile.measure: average spectrum where the track plays, relative to its level."""
    b = _frames(x, FS, grid)
    lev = 10 * np.log10(b.sum(axis=0) + 1e-20)
    if lev.max() < -80:
        return None
    plays = lev > np.percentile(lev, 90) - 20
    if plays.sum() < 20:
        return None
    spec = b[:, plays].mean(axis=1)
    return 10 * np.log10(spec + 1e-20) - 10 * np.log10(spec.sum() + 1e-20)


def main(per_instrument: int = 30) -> None:
    index = json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv) > 2 else None
    if index is None:
        import remotezip

        index = {}
        for i in range(1, 7):
            with remotezip.RemoteZip(URL.format(i)) as z:
                for zi in z.infolist():
                    index[zi.filename] = {"zip": i, "offset": zi.header_offset, "csize": zi.compress_size}
    grid = eq.log_grid(FS)
    rnd = random.Random(7)
    out = {"source": "RawStems (Cambridge-MT Mixing Secrets raw multitracks), first ~minute of each track",
           "fs": FS, "nfft": NFFT, "grid_hz": [round(float(g), 2) for g in grid], "instruments": {}}
    for inst, (group, pat, excl) in PICK.items():
        cands = {}
        for path in index:
            song, _, rest = path.partition("/")
            folder, _, fname = rest.rpartition("/")
            if folder == group and re.search(pat, fname, re.I) and not re.search(excl, fname, re.I):
                cands.setdefault(song, []).append(path)
        songs = sorted(cands)
        rnd.shuffle(songs)
        tones, used = [], []
        for song in songs:
            if len(tones) >= per_instrument:
                break
            path = sorted(cands[song])[0]
            x = None
            for attempt in range(3):
                try:
                    x = _decode(_member_prefix(index[path]))
                    break
                except (requests.RequestException, zlib.error) as e:
                    print(f"  {path}: {e}", file=sys.stderr)
            t = tone(x, grid) if x is not None else None
            if t is not None:
                tones.append(t)
                used.append(path)
                print(f"{inst:8s} {len(tones):3d}/{per_instrument} {path}", flush=True)
        if tones:
            out["instruments"][inst] = {"tracks": len(tones), "tone": [round(float(v), 2) for v in np.median(tones, axis=0)],
                                        "spread_db": [round(float(v), 2) for v in np.std(tones, axis=0)], "from": used}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print("written", OUT)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
