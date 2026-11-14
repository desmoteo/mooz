"""Recordings with close tracks: what a take is, lining up, what stays out, the whole chain."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import test_pipeline as tp
from mooz import config, loudness, multitrack, pipeline, separate, takes

FS = tp.FS
DELAY = {"vocals": 480, "bass": 528}  # how much later the room hears them: 10 and 11 ms
POLARITY = {"vocals": -1, "bass": 1}


def _touch(folder: Path, *names: str) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        sf.write(folder / n, np.zeros((4800, 2 if "TrLR" in n or "TrMic" in n else 1), np.float32), FS, subtype="FLOAT")


def test_a_folder_per_take_gives_the_song_the_mic_as_the_room_and_the_inputs_as_close_tracks(tmp_path):
    a = tmp_path / "261105_225442 Miramare"
    _touch(a, "261105_225442_TrLR.WAV", "261105_225442_TrMic.WAV", "261105_225442_Tr2.WAV", "261105_225442_Tr1.WAV")
    _touch(tmp_path / "processed" / "x", "261105_225442_TrLR.WAV")  # our own output: not a take
    [t] = takes.discover([tmp_path], {"processed"})
    assert (t.name, t.title, t.date, t.root) == ("261105_225442 Miramare", "Miramare", "2026-11-05", tmp_path)
    assert t.room.name == "261105_225442_TrMic.WAV" and t.mixdown.name == "261105_225442_TrLR.WAV"
    assert list(t.close) == ["Tr1", "Tr2"]
    assert takes.discover([tmp_path], {"processed"}, "TrLR")[0].room.name == "261105_225442_TrLR.WAV"


def test_naming_one_file_brings_its_take_and_flat_folders_work(tmp_path):
    _touch(tmp_path, "261105_225442_TrLR.WAV", "261105_225442_Tr2.WAV", "261105_230000_TrLR.WAV")
    all_ = takes.discover([tmp_path], set())
    assert [t.name for t in all_] == ["261105_225442", "261105_230000"]
    assert all_[0].room.name == "261105_225442_TrLR.WAV" and all_[0].mixdown is None and list(all_[0].close) == ["Tr2"]
    assert all_[1].close == {}
    [t] = takes.discover([tmp_path / "261105_225442_Tr2.WAV"], set())
    assert t.room.name == "261105_225442_TrLR.WAV" and list(t.close) == ["Tr2"]


def test_a_stereo_file_named_after_its_song_stays_a_take_of_its_own(tmp_path):
    _touch(tmp_path, "260924_230957_TrLR Stanza OK.WAV", "song.wav", "song_AB.wav")
    found = takes.discover([tmp_path], set())
    assert [(t.name, t.title, t.close) for t in found] == [("260924_230957_TrLR Stanza OK", "Stanza OK", {}), ("song", "song", {})]


def test_two_takes_of_a_song_keep_their_own_names(tmp_path):
    _touch(tmp_path / "261105_230621 Vita", "261105_230621_TrMic.WAV", "261105_230621_Tr2.WAV")
    _touch(tmp_path / "261105_231255 Vita", "261105_231255_TrMic.WAV", "261105_231255_Tr2.WAV")
    assert [t.name for t in takes.discover([tmp_path], set())] == ["261105_230621 Vita", "261105_231255 Vita"]


def test_the_recorders_mixdown_is_recognised(tmp_path):
    rng = np.random.default_rng(1)
    mic, t2 = rng.normal(0, 0.1, (FS * 40, 2)), rng.normal(0, 0.1, (FS * 40, 1))
    folder = tmp_path / "261105_225442 Miramare"
    folder.mkdir()
    sf.write(folder / "261105_225442_TrMic.WAV", mic.astype(np.float32), FS, subtype="FLOAT")
    sf.write(folder / "261105_225442_Tr2.WAV", t2.astype(np.float32), FS, subtype="FLOAT")
    sf.write(folder / "261105_225442_TrLR.WAV", (1.413 * mic + t2).astype(np.float32), FS, subtype="FLOAT")
    f = takes.mixdown_formula(takes.discover([tmp_path], set())[0])
    assert f["gains"] == {"room": pytest.approx(1.413, abs=1e-3), "Tr2": pytest.approx(1.0, abs=1e-3)} and f["residual_db"] < -60


@pytest.mark.parametrize("delay,sign", [(480, -1), (528, 1), (-96, 1)])
def test_the_delay_and_polarity_to_the_room_are_found(delay, sign):
    rng = np.random.default_rng(3)
    close = rng.normal(0, 1, FS * 8).astype(np.float32)
    room = sign * multitrack.shifted(close, delay, len(close)) + 0.5 * rng.normal(0, 1, FS * 8).astype(np.float32)
    assert multitrack.delay_to_room(room, close, FS, 200.0, 5000.0, 250.0)[:2] == (delay, sign)
    assert multitrack.delay_to_room(room, close, FS, 200.0, 5000.0, 250.0)[2] > 10


def test_unrelated_signals_have_no_steady_delay():
    rng = np.random.default_rng(4)
    a, b = rng.normal(0, 1, FS * 60).astype(np.float32), rng.normal(0, 1, FS * 60).astype(np.float32)
    d, s, _ = multitrack.delay_to_room(a, b, FS, 200.0, 5000.0, 250.0)
    assert multitrack._steady(a, b, FS, 200.0, 5000.0, 250.0, d, s) < 30


def test_only_what_a_track_really_holds_is_used():
    cfg = config.load(None)["multitrack"]
    stems = cfg["stems"]
    # the band's mixer feed: voice and bass, the guitar is bleed
    feed = {"Tr2": {"vocals": -32.8, "bass": -35.1, "guitar": -54.8, "drums": -56.3, "other": -38.4}}
    assert multitrack.choose(feed, stems, cfg) == {"vocals": ["Tr2"], "bass": ["Tr2"]}
    # two guitars on two inputs both stay; the bleed of the bass amp in the singer's mic does not
    two = {"Tr1": {"guitar": -30.0, "bass": -50.0, "vocals": -60.0}, "Tr2": {"guitar": -28.0, "bass": -49.0, "vocals": -60.0},
           "Tr3": {"vocals": -25.0, "bass": -38.0, "guitar": -60.0}, "Tr4": {"bass": -24.0, "vocals": -60.0, "guitar": -60.0}}
    got = multitrack.choose(two, stems, cfg)
    assert got["guitar"] == ["Tr1", "Tr2"] and got["vocals"] == ["Tr3"] and got["bass"] == ["Tr4"]
    # a silent track holds nothing
    assert multitrack.choose({"Tr1": {"vocals": -np.inf, "bass": -np.inf}}, stems, cfg) == {}


def _take(tmp_path: Path, monkeypatch) -> tuple[takes.Take, dict, dict]:
    """A take on disk with a mixer feed on Tr2, and the separation replaced by its true stems."""
    mix, room = tp._song()
    n = len(mix)
    rng = np.random.default_rng(11)
    mono = lambda s: s.mean(axis=1)  # noqa: E731
    close = {s: POLARITY[s] * multitrack.shifted(mono(room[s]), -DELAY[s], n) for s in DELAY}
    # the voice's mic hears a voice with its own breath, not in the room's stem: the track is not a copy
    close["vocals"] = close["vocals"] + 0.01 * rng.normal(0, 1, n) * (np.abs(mono(room["vocals"])) > 0.005)
    close["guitar"] = 3e-3 * multitrack.shifted(mono(room["guitar"]), -200, n)  # bleed
    close["drums"] = 3e-3 * mono(room["kick"])
    close["other"] = np.zeros(n)
    folder = tmp_path / "260101_120000 Test Song"
    folder.mkdir()
    feed = sum(close.values())
    sf.write(folder / "260101_120000_TrMic.WAV", mix.astype(np.float32), FS, subtype="FLOAT")
    sf.write(folder / "260101_120000_Tr2.WAV", feed.astype(np.float32), FS, subtype="FLOAT")
    sf.write(folder / "260101_120000_TrLR.WAV", (1.413 * mix + feed[:, None]).astype(np.float32), FS, subtype="FLOAT")
    calls = []

    def fixture_stems(x, fs, cfg, cache_dir, key):
        calls.append(cache_dir.name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for name, s in (close if cache_dir.name.endswith(" Tr2") else room).items():
            sf.write(cache_dir / f"{name}.wav", np.asarray(s, np.float32), fs, subtype="FLOAT")
        (cache_dir / "key.json").write_text(json.dumps(key))
        return separate._stems_in(cache_dir)

    monkeypatch.setattr(separate, "separate", fixture_stems)
    monkeypatch.setattr(separate, "available", lambda backend="demucs": True)
    return takes.discover([tmp_path], {"processed"})[0], room, {"calls": calls, "mix": mix}


def test_close_tracks_are_lined_up_used_or_left_out_and_reach_the_master(tmp_path, monkeypatch):
    take, room, info = _take(tmp_path, monkeypatch)
    assert take.room.name.endswith("_TrMic.WAV") and list(take.close) == ["Tr2"]
    cfg = config.for_file(config.load(None, None, tp._hermetic(tmp_path)), take.name)

    rep = pipeline.process_file(take, cfg, ab=False)

    assert rep["title"] == "Test Song" and rep["date"] == "2026-01-01"
    mt = rep["multitrack"]
    assert mt["room_track"] == "260101_120000_TrMic.WAV"
    assert mt["recorder_mixdown"]["gains"]["room"] == pytest.approx(1.413, abs=1e-2)
    tr2 = mt["close"]["Tr2"]
    assert tr2["vocals"]["used"] and tr2["vocals"]["delay_ms"] == pytest.approx(10.0, abs=0.05) and tr2["vocals"]["polarity"] == -1
    assert tr2["bass"]["used"] and tr2["bass"]["delay_ms"] == pytest.approx(11.0, abs=0.05) and tr2["bass"]["polarity"] == 1
    assert not tr2["guitar"]["used"] and "delay_ms" not in tr2["guitar"]
    assert set(mt["merged"]) == {"vocals", "bass"}

    # the merged stem is the room's plus the lined-up close one, adding in phase
    cache = tmp_path / cfg["output"]["stems_dir"]
    merged_dir = next(p for p in Path(str(cache) + separate.cache_name(cfg["stems"]) + "_merged").glob(f"*/{take.name}"))
    room_v = sf.read(merged_dir / "_room_vocals.wav", dtype="float64", always_2d=True)[0]
    merged_v = sf.read(merged_dir / "vocals.wav", dtype="float64", always_2d=True)[0]
    assert np.sum((merged_v - room_v) * room_v) > 0
    # the close stem prevails, with a little of the room: the parts at close_share of their loudness, the whole
    # at level_lu over the room's own stem, whatever the share
    m = mt["merged"]["vocals"]
    k = 10 ** (m["room_gain_db"] / 20)
    close_part = merged_v - k * room_v
    share = cfg["multitrack"]["close_share"]["vocals"]
    assert share == 0.9 and m["close_share"] == share
    assert loudness.integrated(close_part, tp.FS) - loudness.integrated(k * room_v, tp.FS) == pytest.approx(10 * np.log10(share / (1 - share)), abs=0.2)
    assert loudness.integrated(merged_v, tp.FS) - loudness.integrated(room_v, tp.FS) == pytest.approx(cfg["multitrack"]["level_lu"], abs=0.1)
    # stems not merged are the room's own
    assert (merged_dir / "guitar.wav").is_symlink() and (merged_dir / "kick.wav").is_symlink()
    # the room's cache is untouched: the merge has a cache of its own
    room_cache = Path(str(cache) + separate.cache_name(cfg["stems"])) / take.name
    assert {f.name for f in room_cache.glob("_*")} <= {"_profile.json"}

    y, fs = sf.read(rep["outputs"]["wav"], dtype="float64", always_2d=True)
    assert np.isfinite(y).all() and abs(loudness.integrated(y, fs) - cfg["loudness"]["target_lufs"]) < 0.1
    assert loudness.true_peak_db(y) <= cfg["loudness"]["ceiling_dbtp"] + 0.05
    # the original is the room alone: the recorder's mix-down would hold the close track twice
    b = rep["tracks"]["blend"]
    assert abs((b["original_lufs"] - b["processed_lufs"]) - b["level_lu"]) < 0.1

    out = tmp_path / cfg["output"]["dir"] / "stages" / take.name
    for d in ("1_raw", "1_raw_Tr2", "1_merged"):
        assert (out / d).is_symlink() and (out / d).exists(), d
    assert (out / "1_raw_Tr2" / "vocals.wav").exists() and (out / "5_premix" / "original.flac").exists()
    assert (out / "5_premix" / "drum_bus.flac").exists() and (out / "5_premix" / "bass.flac").exists()

    # a second run does no new separation or merge
    info["calls"].clear()
    before = (merged_dir / "vocals.wav").stat().st_mtime
    monkeypatch.setattr(separate, "separate", lambda x, fs, cfg, cache_dir, key: separate._stems_in(cache_dir))
    pipeline.process_file(take, cfg)
    assert (merged_dir / "vocals.wav").stat().st_mtime == before


def test_ab_is_against_the_recorders_mixdown(tmp_path, monkeypatch):
    take, _, _ = _take(tmp_path, monkeypatch)
    over = tp._everything_off(tmp_path)
    cfg = config.for_file(config.load(None, None, over), take.name)
    rep = pipeline.process_file(take, cfg, ab=True)
    assert Path(rep["outputs"]["ab"]).exists() and Path(rep["outputs"]["ab"]).parent == tmp_path / cfg["output"]["dir"]


def test_the_premix_parts_add_up_to_the_mix(tmp_path, monkeypatch):
    """What is kept for listening is what is mixed: with the bus chain off the master is their sum."""
    take, _, _ = _take(tmp_path, monkeypatch)
    over = tp._everything_off(tmp_path)
    over["output"] = {"mp3": False, "report": False, "stages": "flac"}
    over["balance"] = {"enabled": True}
    over["rider"] = {"enabled": True}
    over["drum_bus"] |= {"enabled": True, "balance": {"enabled": True}}
    over["mix"] = {"original_share": 0.25, "kit_gate_band_hz": []}
    over["trim"] = {"fade_in_s": 0.0, "fade_out_s": 0.0}
    cfg = config.for_file(config.load(None, None, over), take.name)
    assert not cfg["glue"]["enabled"] and not cfg["multiband"]["enabled"]

    rep = pipeline.process_file(take, cfg)

    y, fs = sf.read(rep["outputs"]["wav"], dtype="float64", always_2d=True)
    parts = sorted((tmp_path / cfg["output"]["dir"] / "stages" / take.name / "5_premix").glob("*.flac"))
    assert {p.stem for p in parts} >= {"original", "vocals", "bass", "guitar", "drum_bus"}
    total = sum(sf.read(p, dtype="float64", always_2d=True)[0] for p in parts)[: len(y)]
    gain = float(np.sum(total * y) / np.sum(y * y))
    err = 10 * np.log10(np.sum((total - gain * y) ** 2) / np.sum(total ** 2))
    assert err < -55, f"the parts differ from the master by {err:.1f} dB"


def test_the_simple_bass_preset_changes_only_the_bass_chain():
    full, simple = config.load(None), config.load(None, "simple_bass")
    assert simple["output"]["dir"] != full["output"]["dir"]
    b = simple["tracks"]["bass"]
    for off in ("harmonic_clean", "resonance", "transient", "saturator", "exciter", "bass_enhancer"):
        assert full["tracks"]["bass"][off]["enabled"] and not b[off]["enabled"], off
    for on in ("expander", "tone_match", "compressor", "bass_fundamental"):
        assert b[on]["enabled"], on
    assert b["eq"] == full["tracks"]["bass"]["eq"] and b["ducks"] == full["tracks"]["bass"]["ducks"]
    for name in full["tracks"]:
        if name != "bass":
            assert simple["tracks"][name] == full["tracks"][name], name
    assert {k: v for k, v in simple.items() if k not in ("tracks", "output", "_layers", "_cli")} == \
           {k: v for k, v in full.items() if k not in ("tracks", "output", "_layers", "_cli")}


def test_a_close_stems_lows_are_turned_to_the_rooms_phase_and_keep_their_level():
    """A mixer's low cut turns the phase of the lowest notes: lined up in the body, the two parts
    cancel at the bottom. phase_to_room: they add there, and the close stem's spectrum is as it was."""
    from scipy import signal

    t1 = np.arange(2 * FS) / FS
    notes = [55.0, 61.7, 73.4, 82.4, 98.0, 110.0]
    room = np.concatenate([sum(np.sin(2 * np.pi * f * k * t1) / k for k in range(1, 7)) * np.exp(-t1 / 1.5) for f in notes * 2]) * 0.1
    close = signal.sosfilt(signal.butter(3, 110.0, "high", fs=FS, output="sos"), room)  # the same bass after a low cut

    def at(x, f):  # level at one of the notes' fundamentals (dB)
        fr, p = signal.welch(x, FS, nperseg=1 << 15)
        return 10 * np.log10(p[(fr > f - 3) & (fr < f + 3)].sum())

    turned, found = multitrack.phase_to_room(room, close, FS, 250.0)
    assert abs(found["phase_deg_and_coherence"]["63"][0]) > 100 and found["phase_deg_and_coherence"]["63"][1] > 0.9
    for f in notes[:4]:
        in_phase = 20 * np.log10(10 ** (at(room, f) / 20) + 10 ** (at(close, f) / 20))
        assert at(room + turned, f) > in_phase - 0.5, f  # they add
        assert abs(at(turned, f) - at(close, f)) < 0.3, f  # at the level the close stem had
    assert at(room + close, 61.7) < at(room + turned, 61.7) - 2.0  # as they came, the two took from each other
    k = slice(FS, -FS)
    high = signal.sosfiltfilt(signal.butter(4, 500.0, "high", fs=FS, output="sos"), np.column_stack([close, turned]), axis=0)[k]
    assert np.max(np.abs(high[:, 0] - high[:, 1])) < 1e-3 * np.max(np.abs(high[:, 0]))  # above the turn: sample for sample
    # nothing in common: nothing is turned
    noise = np.random.default_rng(3).normal(0, 0.1, len(room))
    same, _ = multitrack.phase_to_room(room, noise, FS, 250.0)
    assert np.max(np.abs(same[k] - noise[k])) < 0.02 * np.max(np.abs(noise))
