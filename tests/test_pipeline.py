"""End to end: process_file on a short synthetic song.

The separation models are replaced by fixture stems (written where the cache would be),
so the whole chain after the separation runs as it does on a real recording: trim, hum,
stem profiles, clean-up, studio chains, drum bus, balance, bus, loudness, files and reports.
The checks are invariants (finite audio, loudness target, true-peak ceiling, the files that
must exist), not golden values: they hold whatever the settings are tuned to.
"""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from rehearsal_master import config, loudness, pipeline, separate

FS = 48000
LEAD_S, MUSIC_S, TAIL_S = 5.0, 12.0, 1.5  # silence with hum, the song, silence
NAME = "260101_120000_TrLR Test Song.wav"  # a Zoom-style name: title and date come from it


def _bursts(n: int, times_s, shape: np.ndarray) -> np.ndarray:
    out = np.zeros(n)
    for t in times_s:
        i = int(t * FS)
        seg = shape[: max(0, n - i)]
        out[i : i + len(seg)] += seg
    return out


def _noise(rng, n: int, lo: float, hi: float | None = None) -> np.ndarray:
    kind, edges = ("high", lo) if hi is None else ("bandpass", [lo, hi])
    return signal.sosfilt(signal.butter(4, edges, kind, fs=FS, output="sos"), rng.normal(0, 1, n))


def _song(clean: bool = False) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """A band in 12 seconds: (recording, stems). The stems sum to the recording, hiss and
    mains hum aside (what a separation leaves in the residual); clean: exactly."""
    rng = np.random.default_rng(7)
    n = int((LEAD_S + MUSIC_S + TAIL_S) * FS)
    beats = LEAD_S + np.arange(0, MUSIC_S, 0.5)  # 120 BPM
    t1 = np.arange(FS) / FS  # one second, for the hit shapes

    kick = _bursts(n, beats, 0.5 * np.sin(2 * np.pi * 60 * t1) * np.exp(-t1 / 0.08))
    snare_hit = (0.3 * np.sin(2 * np.pi * 200 * t1) * np.exp(-t1 / 0.05)
                 + 0.25 * _noise(rng, FS, 2000.0) * np.exp(-t1 / 0.06))
    snare = _bursts(n, beats[1::2], snare_hit)
    toms = _bursts(n, LEAD_S + np.array([3.25, 3.5, 3.75, 7.25, 7.5, 7.75, 11.25, 11.5]),
                   0.35 * np.sin(2 * np.pi * 110 * t1) * np.exp(-t1 / 0.12))
    hihat = _bursts(n, LEAD_S + np.arange(0, MUSIC_S, 0.25), 0.08 * _noise(rng, FS, 7000.0) * np.exp(-t1 / 0.02))
    t2 = np.arange(2 * FS) / FS
    cymbals = _bursts(n, LEAD_S + np.array([0.0, 4.0, 8.0]), 0.1 * _noise(rng, 2 * FS, 3000.0, 12000.0) * np.exp(-t2 / 0.6))

    bass = np.zeros(n)
    for i, f in enumerate(np.tile([55.0, 73.4, 82.4, 65.4], 3)):  # a note per second
        note = sum(np.sin(2 * np.pi * f * k * t1) / k for k in range(1, 6)) * np.exp(-t1 / 0.7)
        note[: int(0.005 * FS)] *= np.linspace(0, 1, int(0.005 * FS))
        bass += _bursts(n, [LEAD_S + i], 0.2 * note)

    tm = np.arange(int(MUSIC_S * FS)) / FS
    chord = sum(np.sin(2 * np.pi * f * k * tm) / k for f in (196.0, 247.0, 294.0) for k in range(1, 9))
    guitar = np.zeros(n)
    guitar[int(LEAD_S * FS) : int(LEAD_S * FS) + len(tm)] = 0.1 * np.tanh(chord)  # a little distorted

    f0 = 220.0 * (1 + 0.01 * np.sin(2 * np.pi * 5.5 * tm))  # vibrato
    phase = 2 * np.pi * np.cumsum(f0) / FS
    voice = sum(np.sin(k * phase) / k ** 1.2 for k in range(1, 13))
    phrase = signal.sosfiltfilt(signal.butter(2, 8.0, "low", fs=FS, output="sos"), ((tm % 3.0) < 2.0).astype(float))
    vocals = np.zeros(n)
    vocals[int(LEAD_S * FS) : int(LEAD_S * FS) + len(tm)] = 0.1 * voice * phrase  # 2 s singing, 1 s rest
    room_ir = rng.normal(0, 1, int(0.3 * FS)) * np.exp(-np.arange(int(0.3 * FS)) / (0.07 * FS))
    vocal_room = 0.25 * signal.oaconvolve(vocals, room_ir / np.sqrt(np.sum(room_ir ** 2)))[:n]

    stereo = lambda x, left=1.0, right=1.0: np.column_stack([left * x, right * x])  # noqa: E731
    stems = {"kick": stereo(kick), "snare": stereo(snare), "toms": stereo(toms), "hihat": stereo(hihat, 0.8, 1.0),
             "cymbals": stereo(cymbals), "bass": stereo(bass), "guitar": stereo(guitar, 1.0, 0.7),
             "vocals": stereo(vocals), "vocal_room": stereo(vocal_room), "other": np.zeros((n, 2))}  # nobody on "other"
    mix = sum(stems.values())
    if not clean:
        t = np.arange(n) / FS
        hum = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((50.0, 2e-3), (100.0, 1e-3), (150.0, 1e-3)))
        mix = mix + stereo(hum) + 1e-4 * rng.normal(0, 1, (n, 2))
    return mix, stems


@pytest.fixture(scope="module")
def song():
    return _song()


def _fixture_separation(monkeypatch, mix, stems, highpass: dict | None = None):
    """In place of the models: the given stems, written where the cache would be. The real
    separation sees the recording after the rumble filter, so with highpass (the [highpass]
    settings) the stems get the same filter and still add up to what is separated."""
    def fixture_stems(x, fs, cfg, cache_dir, key):
        assert len(x) == len(mix) and fs == FS  # the whole recording goes to the separation
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, s in stems.items():
            if highpass and highpass["enabled"]:
                s = signal.sosfilt(signal.butter(highpass["order"], highpass["freq_hz"], "high", fs=fs, output="sos"), s, axis=0)
            sf.write(cache_dir / f"{name}.wav", s.astype(np.float32), fs, subtype="FLOAT")
        (cache_dir / "key.json").write_text(json.dumps(key))
        return separate._stems_in(cache_dir)

    monkeypatch.setattr(separate, "separate", fixture_stems)
    monkeypatch.setattr(separate, "available", lambda backend="demucs": True)


def _hermetic(tmp_path) -> dict:
    """Settings that keep a test on this machine: no MP3 (ffmpeg), no sample library (the user's cache)."""
    none = str(tmp_path / "no_samples")
    return {"output": {"mp3": False},
            "track_defaults": {"augment": {"library": {"dir": none}}, "cymbal_augment": {"dir": none}}}


@pytest.mark.parametrize("preset", [None, "albini"])
def test_process_file_runs_end_to_end_and_meets_loudness_and_peak(song, preset, tmp_path, monkeypatch):
    mix, stems = song
    path = tmp_path / NAME
    sf.write(path, mix.astype(np.float32), FS, subtype="FLOAT")
    cfg = config.for_file(config.load(None, preset, _hermetic(tmp_path)), path.name)
    _fixture_separation(monkeypatch, mix, stems, cfg["highpass"])

    rep = pipeline.process_file(path, cfg)

    assert (rep["title"], rep["date"]) == ("Test Song", "2026-01-01")
    start, end = rep["trim"]["start_s"], rep["trim"]["end_s"]
    assert abs(start - (LEAD_S - cfg["trim"]["pre_roll_s"])) < 0.2  # the lead-in kept, the silence before it cut
    assert LEAD_S + MUSIC_S <= end <= LEAD_S + MUSIC_S + TAIL_S
    assert rep["hum"]["base_hz"] == 50.0 and {h["k"] for h in rep["hum"]["harmonics"]} >= {1, 2, 3}

    y, fs = sf.read(rep["outputs"]["wav"], dtype="float64", always_2d=True)
    assert fs == FS and y.shape[1] == 2 and abs(len(y) - (end - start) * FS) <= 0.01 * FS
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01
    target, ceiling = cfg["loudness"]["target_lufs"], cfg["loudness"]["ceiling_dbtp"]
    assert rep["finalize"]["iterations"] < 5  # the loudness loop settled
    assert abs(loudness.integrated(y, fs) - target) < 0.1  # measured on the written file, fades included
    assert loudness.true_peak_db(y) <= ceiling + 0.05

    tracks = rep["tracks"]
    assert tracks["other"] == {"silent": True}  # an empty stem is passed through, not processed
    for name in ("vocals", "guitar", "bass", "kick", "snare", "hihat", "cymbals"):
        assert np.isfinite(tracks[name]["lufs_out"]), name
    assert set(tracks["drum_bus"]["stems"]) == {"kick", "snare", "toms", "hihat", "cymbals"}
    assert np.isfinite(tracks["remix"]["remix_lufs"])
    assert tracks["residual"]["level_db"] < -20  # the stems account for the recording (hum and hiss aside)
    blend = tracks["blend"]  # the original under the stems at its share of the loudness: -4.8 LU for 0.25
    share = cfg["mix"]["original_share"]
    assert blend["original_share"] == share and share > 0
    assert abs(blend["level_lu"] - 10 * np.log10(share / (1 - share))) < 0.01
    assert abs((blend["original_lufs"] - blend["processed_lufs"]) - blend["level_lu"]) < 0.1
    if cfg["mix"]["kit_gate_band_hz"] and cfg["tracks"]["hihat"]["kit_gate"]["enabled"]:
        # the lead-in before the first hit: the kit gate is closed there, and the original follows it
        assert blend["kit_gate"]["max_db"] == cfg["tracks"]["hihat"]["kit_gate"]["depth_db"]
        assert 0 < blend["kit_gate"]["closed_pct"] < 50
    for g, b in tracks["balance"].items():  # measured in the blend, before and after; the stems' gain bounded
        if "note" not in b:
            assert abs(b["stem_gain_db"]) <= pipeline.STEM_GAIN_MAX_DB and np.isfinite(b["after_lu"]), g

    out = path.parent / cfg["output"]["dir"]
    saved = json.loads((out / "reports" / f"{path.stem}.json").read_text())  # the report is valid JSON
    assert saved["processed"]["integrated_lufs"] == rep["processed"]["integrated_lufs"]
    for f in ("reports/{0}.png", "reports/{0}_tracks.png", "stages/{0}/5_mix.flac", "stages/{0}/4_drum_bus.flac",
              "stages/{0}/2_stems/vocals.flac", "stages/{0}/character.json"):
        assert (out / f.format(path.stem)).exists(), f
    # this run's files are gone, and the shared stems cache holds only what every preset can share
    assert not (out / ".work").exists()
    cache = path.parent / (cfg["output"]["stems_dir"] + separate.cache_name(cfg["stems"])) / path.stem
    assert {f.name for f in cache.iterdir()} <= {f"{n}.wav" for n in stems} | {"key.json", "_profile.json"}


def _everything_off(tmp_path) -> dict:
    """Overrides that switch every processor off: every table with an `enabled` key (but the
    separation itself), every track's EQ, ducks, sends and placement, fades, room reduction."""
    base = config.load(None)

    def off(d: dict) -> dict:
        o = {}
        for k, v in d.items():
            if isinstance(v, dict) and not k.startswith("_"):
                sub = off(v)
                if sub:
                    o[k] = sub
        if "enabled" in d:
            o["enabled"] = False
        return o

    over = off({k: v for k, v in base.items() if k not in ("tracks", "files", "plans")})
    over.pop("output", None)
    over["stems"] = {k: v for k, v in over.get("stems", {}).items() if k not in ("enabled", "plans")}
    over["stems"]["vocal_room"] = {"max_reduction_db": 0.0}
    over["trim"] = {"fade_in_s": 0.0, "fade_out_s": 0.0}  # the trim itself stays: the stems must line up with it
    over["loudness"] = {"target_lufs": -26.0}  # far below any limiting
    over["track_defaults"] = off(base["track_defaults"]) | {
        "eq": [], "dynamic_eq": [], "ducks": [], "sends": [], "width": 1.0, "balance": 0.0, "pan": 0.0, "gain_db": 0.0}
    over["tracks"] = {n: off(t) | {"eq": [], "dynamic_eq": [], "ducks": [], "sends": [], "width": 1.0, "balance": 0.0,
                                   "pan": 0.0, "gain_db": 0.0} for n, t in base["tracks"].items()}
    return over | _hermetic(tmp_path) | {"output": {"mp3": False, "report": False, "stages": "off"}}


def test_with_every_processor_off_the_master_is_the_recording(tmp_path, monkeypatch):
    """Nothing is lost: stems that sum to the recording, no processing, the original blended in,
    the balance and the bus chain off: the master is the trimmed recording, scaled."""
    mix, stems = _song(clean=True)
    path = tmp_path / NAME
    sf.write(path, mix.astype(np.float32), FS, subtype="FLOAT")
    _fixture_separation(monkeypatch, mix, stems)
    cfg = config.for_file(config.load(None, None, _everything_off(tmp_path)), path.name)
    assert cfg["mix"]["original_share"] > 0 and not cfg["balance"]["enabled"] and not cfg["glue"]["enabled"]

    rep = pipeline.process_file(path, cfg)

    y, fs = sf.read(rep["outputs"]["wav"], dtype="float64", always_2d=True)
    start = int(round(rep["trim"]["start_s"] * fs))
    x = mix[start : start + len(y)]
    gain = float(np.sum(x * y) / np.sum(y * y))
    err = 10 * np.log10(np.sum((x - gain * y) ** 2) / np.sum(x ** 2))
    assert err < -60, f"master differs from the recording by {err:.1f} dB"
    assert rep["finalize"]["limiter"]["max_gr_db"] == 0.0 and rep["tracks"]["blend"]["gain_db"] is not None
