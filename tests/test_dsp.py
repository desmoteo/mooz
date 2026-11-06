import numpy as np
from scipy import signal

from rehearsal_master import dynamics, eq, loudness, stereo

FS = 48000
rng = np.random.default_rng(1)


def test_k_weighting_matches_bs1770_at_48k():
    sos = loudness.k_weighting_sos(48000)
    np.testing.assert_allclose(sos[0, :3], [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-9)
    np.testing.assert_allclose(sos[0, 4:], [-1.69065929318241, 0.73248077421585], atol=1e-9)
    np.testing.assert_allclose(sos[1, 4:], [-1.99004745483398, 0.99007225036621], atol=1e-9)


def test_loudness_of_1k_sine():
    # BS.1770: a 0 dBFS 997 Hz sine in one channel reads -3.01 LUFS
    t = np.arange(FS * 10) / FS
    x = np.column_stack([np.sin(2 * np.pi * 997 * t), np.zeros_like(t)])
    assert abs(loudness.integrated(x, FS) - (-3.01)) < 0.05


def test_release_hold_matches_recursion():
    g = np.maximum(0, rng.normal(0, 2, 5000)) * (rng.random(5000) < 0.02)
    r = np.exp(-1000 / (5 * FS))
    ref, prev = np.zeros_like(g), 0.0
    for i, v in enumerate(g):
        prev = max(v, r * prev)
        ref[i] = prev
    got = dynamics.release_hold(g, 5.0, FS)
    assert np.max(np.abs(np.where(ref > 1e-5, got - ref, 0))) < 1e-9


def test_band_split_reconstructs_exactly():
    x = rng.normal(0, 0.1, (FS, 2))
    assert np.max(np.abs(sum(dynamics.split_bands(x, FS, [120, 600, 4000])) - x)) < 1e-12


def test_limiter_holds_true_peak_ceiling():
    x = rng.normal(0, 0.1, (FS * 5, 2))
    x[rng.integers(0, len(x), 200)] *= 25
    y, _, _ = dynamics.limit(x, FS, -1.0, 2.0, 60.0)
    assert loudness.true_peak_db(y) <= -0.99


def test_mid_side_monos_bass_keeps_highs():
    t = np.arange(FS * 2) / FS
    mid = slice(FS // 2, -FS // 2)

    def side_ratio(freq):
        s = np.sin(2 * np.pi * freq * t)
        x = np.column_stack([s, 0.2 * s])  # lopsided
        y = stereo.mid_side(x, FS, 120.0, 1.0)
        return np.max(np.abs(y[mid, 0] - y[mid, 1])) / np.max(np.abs(x[mid, 0] - x[mid, 1]))

    assert side_ratio(50) < 0.05  # > 26 dB less side signal at 50 Hz
    assert abs(side_ratio(1000) - 1) < 0.01  # image untouched above the crossover


def test_fir_follows_requested_curve():
    grid = eq.log_grid(FS)
    curve = 3 * np.sin(np.log2(grid / 20))
    h = eq.design_fir(grid, curve, FS, 8191)
    inside = (grid > 60) & (grid < 16000)
    assert np.max(np.abs(eq.fir_response_db(h, grid, FS) - curve)[inside]) < 0.3


# --- per-track effects ----------------------------------------------------------------------

from scipy import signal as _sig  # noqa: E402

from rehearsal_master import config, effects  # noqa: E402

CFG = config.load(None)


def _db(a):
    return 20 * np.log10(np.sqrt(np.mean(a**2)) + 1e-20)


def test_neutral_track_is_identity():
    x = rng.normal(0, 0.1, (FS * 2, 2))
    y, _ = effects.process_track(x, FS, CFG["track_defaults"])
    assert np.array_equal(x, y)


def test_de_esser_cuts_sibilance_only():
    t = np.arange(FS * 6) / FS
    burst = ((t % 1) > 0.2) & ((t % 1) < 0.3)
    sib = _sig.sosfiltfilt(_sig.butter(2, [6000, 8000], "bandpass", fs=FS, output="sos"), rng.normal(0, 1, len(t)))
    vowel = 0.3 * np.sin(2 * np.pi * 220 * t) + 0.2 * np.sin(2 * np.pi * 440 * t)
    x = np.column_stack([vowel + 0.3 * sib * burst] * 2)
    y, _, _ = effects.dynamic_eq(x, FS, CFG["tracks"]["vocals"]["dynamic_eq"], np.ones(len(t), bool))
    hp = lambda a: _sig.sosfiltfilt(_sig.butter(4, 5000, "high", fs=FS, output="sos"), a[:, 0])  # noqa: E731
    s = slice(int(0.22 * FS), int(0.28 * FS))
    assert _db(hp(y)[s]) - _db(hp(x)[s]) < -4  # sibilance turned down
    q = slice(int(0.5 * FS), int(0.9 * FS))
    assert abs(_db(y[q, 0]) - _db(x[q, 0])) < 0.05  # vowels untouched


def test_resonance_suppressor_cuts_ringing_tone_not_noise():
    t = np.arange(FS * 6) / FS
    noise = rng.normal(0, 0.05, (len(t), 2))
    r = CFG["tracks"]["bass"]["resonance"]
    y, _, _ = effects.resonance_suppressor(noise, FS, r)
    assert abs(_db(y) - _db(noise)) < 0.1
    x = noise + 0.3 * np.sin(2 * np.pi * 110 * t)[:, None]
    y, _, _ = effects.resonance_suppressor(x, FS, r)
    f, p0 = _sig.welch(x[:, 0], FS, nperseg=16384)
    _, p1 = _sig.welch(y[:, 0], FS, nperseg=16384)
    tone, rest = np.argmin(np.abs(f - 110)), (f > 200) & (f < 350)
    # with make-up gain the overall level is kept; the ringing tone drops relative to the rest
    change = 10 * np.log10(p1[tone] / p0[tone]) - 10 * np.log10(p1[rest].sum() / p0[rest].sum())
    assert change < -3
    assert abs(_db(y) - _db(x)) < 0.1


def test_transient_shaper_boosts_attack_cuts_tail():
    n = np.arange(int(0.4 * FS))
    hit = np.sin(2 * np.pi * 200 * n / FS) * np.exp(-n / (0.15 * FS))
    x = np.column_stack([np.tile(np.pad(hit, (0, FS // 10)), 6)] * 2)
    y, _, _ = effects.transient_shaper(x, FS, 0.3, -0.4, 6.0)
    assert _db(y[:480]) - _db(x[:480]) > 2
    assert _db(y[9600:19200]) - _db(x[9600:19200]) < -2


def test_bass_enhancer_adds_harmonics_at_set_level():
    t = np.arange(FS * 4) / FS
    x = np.column_stack([0.5 * np.sin(2 * np.pi * 50 * t)] * 2)
    y = effects.bass_enhancer(x, FS, CFG["tracks"]["bass"]["bass_enhancer"])
    f, p0 = _sig.welch(x[:, 0], FS, nperseg=16384)
    _, p1 = _sig.welch(y[:, 0], FS, nperseg=16384)
    level = 10 * np.log10(p1[(f > 90) & (f < 400)].sum() / p0[(f > 45) & (f < 55)].sum())
    assert abs(level - CFG["tracks"]["bass"]["bass_enhancer"]["amount_db"]) < 1.5


def test_track_detail_traces_line_up_with_audio():
    x = rng.normal(0, 0.1, (FS * 4, 2))
    d = {}
    y, st = effects.process_track(x, FS, CFG["tracks"]["drums"], d)
    n = len(x) // int(0.1 * FS)
    assert len(d["traces"]["compressor"]) == n and len(d["traces"]["transient"]["boost"]) == n
    assert len(d["dyn_eq"]) == len(CFG["tracks"]["drums"]["dynamic_eq"])
    assert d["spec_in"].shape == d["spec_out"].shape == d["grid"].shape
    assert set(st) >= {"compressor", "dynamic_eq", "transient", "eq_db"}


def test_chunked_separation_reconstructs(tmp_path, monkeypatch):
    """Chunking + crossfades must not change a linear separator's output."""
    import soundfile as sf

    from rehearsal_master import separate

    monkeypatch.setattr(separate, "CHUNK_S", 1.0)
    monkeypatch.setattr(separate, "OVERLAP_S", 0.25)
    sr = 44100

    class Halves:  # stand-in model: splits audio into 0.3 / 0.7 of itself
        def separate(self, path):
            x = sf.read(path, dtype="float32", always_2d=True)[0]
            out = tmp_path / "out"
            out.mkdir(exist_ok=True)
            sf.write(out / "c_(Vocals)_m.wav", 0.3 * x, sr, subtype="FLOAT")
            sf.write(out / "c_(Instrumental)_m.wav", 0.7 * x, sr, subtype="FLOAT")
            return ["c_(Vocals)_m.wav", "c_(Instrumental)_m.wav"]

    x = rng.normal(0, 0.1, (int(3.6 * sr), 2)).astype(np.float32)
    src = tmp_path / "src.wav"
    sf.write(src, x, sr, subtype="FLOAT")
    (tmp_path / "out").mkdir()
    res = separate._separate_chunked(Halves(), src, tmp_path, sr)
    voc = sf.read(res["vocals"], dtype="float32", always_2d=True)[0]
    inst = sf.read(res["instrumental"], dtype="float32", always_2d=True)[0]
    assert len(voc) == len(x)
    np.testing.assert_allclose(voc, 0.3 * x, atol=1e-6)
    np.testing.assert_allclose(voc + inst, x, atol=1e-6)


def test_saturator_adds_even_harmonics_and_is_neutral_at_zero_mix():
    t = np.arange(FS * 2) / FS
    x = np.column_stack([0.3 * np.sin(2 * np.pi * 200 * t)] * 2)
    y0, _ = effects.saturator(x, FS, {"drive_db": 6.0, "even": 0.7, "mix": 0.0})
    np.testing.assert_allclose(y0, x, atol=1e-9)
    y, st = effects.saturator(x, FS, {"drive_db": 6.0, "even": 0.7, "mix": 0.5})
    f, p = _sig.welch(y[:, 0], FS, nperseg=8192)
    h1, h2 = p[np.argmin(np.abs(f - 200))], p[np.argmin(np.abs(f - 400))]
    assert 10 * np.log10(h2 / h1) > -40  # a real 2nd harmonic appears
    assert abs(_db(y) - _db(x)) < 1.5


def test_hit_gate_turns_down_the_gaps_only():
    n = np.arange(int(0.05 * FS))
    hit = np.sin(2 * np.pi * 300 * n / FS) * np.exp(-n / (0.01 * FS))
    x = np.tile(np.pad(hit, (0, FS // 2 - len(hit))), 8)
    x = np.column_stack([x + 0.003 * rng.normal(size=len(x))] * 2)  # hits + a buzz floor
    y, _, _ = effects.hit_gate(x, FS, {"fmin_hz": 150.0, "fmax_hz": 6000.0, "depth_db": 6.0, "release_ms": 100.0})
    hits = slice(FS // 2, FS // 2 + int(0.01 * FS))
    gaps = slice(FS // 2 + int(0.3 * FS), FS)
    assert abs(_db(y[hits]) - _db(x[hits])) < 1.0
    assert _db(y[gaps]) - _db(x[gaps]) < -4.5


def test_band_duck_only_dips_its_band():
    t = np.arange(FS * 2) / FS
    lo, hi = np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 5000 * t)
    x = np.column_stack([lo + hi] * 2) * 0.2
    trig = np.ones(len(t), np.float32)
    y, _, _ = effects.duck(x, FS, trig, 2000.0, 8000.0, 6.0, 100.0)
    f, p0 = _sig.welch(x[FS // 2 :, 0], FS, nperseg=8192)
    _, p1 = _sig.welch(y[FS // 2 :, 0], FS, nperseg=8192)
    k_lo, k_hi = np.argmin(np.abs(f - 100)), np.argmin(np.abs(f - 5000))
    assert abs(10 * np.log10(p1[k_lo] / p0[k_lo])) < 0.3
    assert 10 * np.log10(p1[k_hi] / p0[k_hi]) < -4


def test_pan_split_separates_two_positions_and_sums_back(tmp_path):
    import soundfile as sf

    from rehearsal_master import separate

    sr = 44100
    t = np.arange(sr * 4) / sr
    a = 0.3 * np.sin(2 * np.pi * 440 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))  # "guitar 1", hard left
    b = 0.3 * np.sin(2 * np.pi * 1250 * t)                                       # "guitar 2", centre
    x = np.column_stack([a + b, 0.15 * a + b]).astype(np.float32)
    src = tmp_path / "g.wav"
    sf.write(src, x, sr, subtype="FLOAT")
    out = separate._pan_split(src, tmp_path, sr)
    left = sf.read(out["left"], dtype="float64", always_2d=True)[0]
    right = sf.read(out["right"], dtype="float64", always_2d=True)[0]
    np.testing.assert_allclose(left + right, x, atol=1e-5)
    f, pl = _sig.welch(left[:, 0], sr, nperseg=8192)
    _, pr = _sig.welch(right[:, 0], sr, nperseg=8192)
    k440, k1250 = np.argmin(np.abs(f - 440)), np.argmin(np.abs(f - 1250))
    assert pl[k440] > 30 * pr[k440] and pr[k1250] > 30 * pl[k1250]


# --- clean-up --------------------------------------------------------------------------------

from rehearsal_master import cleanup  # noqa: E402


def test_hum_is_found_and_notched_without_touching_nearby_notes():
    t = np.arange(FS * 12) / FS
    hum = 0.01 * (np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + 0.5 * np.sin(2 * np.pi * 250 * t))
    note = 0.2 * np.sin(2 * np.pi * 146.8 * t) * (t > 6)  # D3 next to the 150 Hz harmonic, second half only
    x = np.column_stack([hum + note + 0.001 * rng.normal(size=len(t))] * 2)
    found = cleanup.detect_hum(x, FS, CFG["hum"], lead_in=(0, FS * 5))  # the first 5 s: hum only
    hz = [h["hz"] for h in found["harmonics"]]
    assert any(abs(f - 50) < 0.5 for f in hz) and any(abs(f - 150) < 0.5 for f in hz) and any(abs(f - 250) < 0.5 for f in hz)
    y = cleanup.remove_hum(x, FS, found, CFG["hum"])
    f, p0 = _sig.welch(x[: FS * 5, 0], FS, nperseg=FS * 4)
    _, p1 = _sig.welch(y[: FS * 5, 0], FS, nperseg=FS * 4)
    for h in (50, 150, 250):
        k = np.argmin(np.abs(f - h))
        assert 10 * np.log10(p1[k] / p0[k]) < -20
    late = slice(FS * 7, FS * 11)
    assert abs(_db(_sig.sosfiltfilt(_sig.butter(2, [140, 152], "bandpass", fs=FS, output="sos"), y[late, 0]))
               - _db(_sig.sosfiltfilt(_sig.butter(2, [140, 152], "bandpass", fs=FS, output="sos"), x[late, 0]))) < 1.5


def test_expander_cleans_gaps_keeps_notes():
    t = np.arange(FS * 4) / FS
    notes = (np.floor(t * 2) % 2 == 0).astype(float)  # 0.5 s on / 0.5 s off
    x = np.column_stack([0.3 * np.sin(2 * np.pi * 110 * t) * notes + 0.0005 * rng.normal(size=len(t))] * 2)
    y, st, _ = effects.expander(x, FS, CFG["tracks"]["bass"]["expander"])
    on = slice(int(2.0 * FS) + 200, int(2.4 * FS))
    gap = slice(int(2.7 * FS), int(2.95 * FS))
    assert abs(_db(y[on]) - _db(x[on])) < 0.3
    assert _db(y[gap]) - _db(x[gap]) < -6
    onset = slice(int(2.0 * FS), int(2.0 * FS) + 240)  # first 5 ms of a note
    assert abs(_db(y[onset]) - _db(x[onset])) < 1.0


def test_balance_brings_groups_to_targets():
    cfg = {"strength": 1.0, "max_change_db": 12.0, "ignore_below_db": 12.0, "drop_below_lu": -40.0,
           "groups": [{"name": "vocals", "stems": ["vocals"], "lu": -6.0},
                      {"name": "guitars", "stems": ["guitar_left", "guitar_right"], "lu": -8.0},
                      {"name": "bass", "stems": ["bass"], "lu": -7.0}]}
    lufs = {"vocals": -10.0, "guitar_left": -18.0, "guitar_right": -12.0, "bass": -40.0, "other": -60.0}
    total = 10 * np.log10(sum(10 ** (v / 10) for v in lufs.values()))
    gains, rep = cleanup.balance_gains(lufs, total, cfg)
    # the groups end up in the target pattern: vocals 2 LU above the guitars (they were 1.0 below)
    assert abs((rep["vocals"]["after_lu"] - rep["guitars"]["after_lu"]) - 2.0) < 0.2
    assert gains["guitar_left"] == gains["guitar_right"]  # a group keeps its internal balance
    assert rep["bass"]["note"].startswith("too quiet") and gains["bass"] == 0.0
    assert gains["other"] is None  # empty leftover dropped


def test_refined_stems_sum_to_the_mix():
    a = rng.normal(0, 0.1, (FS * 2, 2))
    b = rng.normal(0, 0.05, (FS * 2, 2))
    out = cleanup.refine_stems(a + b, {"a": a, "b": b}, FS, 1.5, -30.0)
    np.testing.assert_allclose(out["a"] + out["b"], a + b, atol=1e-6)


def test_hold_catches_single_sample_events():
    m = np.zeros(FS * 2, bool)
    m[FS + 123] = True  # one isolated sample, not on any 10 ms grid point
    h = effects._hold(m, FS, 0.01, 0.15)
    assert h[FS + 123] == 1 and h[FS + 123 + int(0.1 * FS)] == 1 and h[FS + 123 - int(0.005 * FS)] == 1
    assert h[FS + 123 + int(0.3 * FS)] == 0 and h[FS - int(0.1 * FS)] == 0


def test_buzz_is_learnt_then_removed_and_hats_are_kept(tmp_path):
    n = FS * 6
    t = np.arange(n) / FS
    buzz = _sig.sosfiltfilt(_sig.butter(4, [2000, 5000], "bandpass", fs=FS, output="sos"), rng.normal(size=n))
    buzz *= 0.05 * (1 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    hats = _sig.sosfiltfilt(_sig.butter(4, [7000, 11000], "bandpass", fs=FS, output="sos"), rng.normal(size=n))
    hats *= 0.08 * ((t % 0.25) < 0.04) * (t > 3)  # hi-hat bursts in the second half only
    x = np.column_stack([buzz + hats] * 2)
    cfg = dict(CFG["buzz"], min_learn_s=1.0)
    idle = t < 3  # first half: buzz alone
    np.save(tmp_path / "song.npy", cleanup.buzz_frames({"hihat": x}, idle, FS, cfg))
    w = cleanup.buzz_fingerprint(tmp_path, cfg)
    assert w is not None
    y, st = cleanup.remove_buzz(x, FS, w, cfg)
    band = lambda z, lo, hi: _db(_sig.sosfiltfilt(_sig.butter(4, [lo, hi], "bandpass", fs=FS, output="sos"), z[FS * 3 + 4800 : FS * 6 - 4800, 0]))  # noqa: E731
    assert band(y, 2500, 4500) - band(x, 2500, 4500) < -6  # the buzz goes
    assert abs(band(y, 7500, 10500) - band(x, 7500, 10500)) < 2.0  # the hats stay (nearly)
    protect = np.ones(n // 512 + 8)
    y2, _ = cleanup.remove_buzz(x, FS, w, cfg, protect)  # protected everywhere: nothing removed
    assert abs(_db(y2) - _db(x)) < 0.1


# --- stages, character, drum bus -------------------------------------------------------------

def test_stage_writer_keeps_relative_levels_without_clipping(tmp_path):
    import soundfile as sf

    from rehearsal_master import stages

    w = stages.StageWriter(tmp_path, "flac")
    loud = np.full((FS, 2), 2.0)  # above full scale, like the raw drum peaks
    quiet = np.full((FS, 2), 0.5)
    w.write("2_stems/a", loud, FS)
    w.write("2_stems/b", quiet, FS)
    a = sf.read(tmp_path / "2_stems/a.flac")[0]
    b = sf.read(tmp_path / "2_stems/b.flac")[0]
    assert np.abs(a).max() < 1.0
    assert abs(20 * np.log10(np.abs(a).max() / np.abs(b).max()) - 20 * np.log10(4)) < 0.01


def test_character_rules():
    from rehearsal_master import character

    base = {"contrast_lu": 5.0, "arc_lu": 0.0, "slope_lu_per_min": 0.0, "onset_rate_per_s": 3.5, "tempo_bpm": 120,
            "drums_playing_pct": 97, "guitar_flatness": 0.3}
    assert character._classify(base)[0] == "wall"
    assert character._classify(base | {"drums_playing_pct": 30})[0] == "ballad"
    assert character._classify(base | {"arc_lu": 6.0, "slope_lu_per_min": 2.0})[0] == "build"
    assert character._classify(base | {"contrast_lu": 9.0})[0] == "dynamic"
    assert character._classify(base | {"contrast_lu": 6.0, "drums_playing_pct": 70})[0] == "steady"


def test_drum_bus_balances_the_kit(tmp_path):
    import soundfile as sf

    from rehearsal_master import pipeline

    files, lufs = {}, {}
    for name, a in (("kick", 0.3), ("snare", 0.15), ("hihat", 0.02)):
        x = a * rng.normal(size=(FS * 4, 2))
        files[name] = tmp_path / f"{name}.wav"
        sf.write(files[name], x.astype(np.float32), FS, subtype="FLOAT")
        lufs[name] = loudness.integrated(x, FS)
    cfg = dict(CFG)
    cfg["drum_bus"] = dict(CFG["drum_bus"], compressor=dict(CFG["drum_bus"]["compressor"], enabled=False))
    bus, st = pipeline._drum_bus(files, lufs, FS, cfg)
    assert bus.shape == (FS * 4, 2) and set(st["stems"]) == {"kick", "snare", "hihat"}
    g = {k: v["gain_db"] for k, v in st["balance"].items()}
    assert g["snare"] - g["kick"] > 4  # the snare comes up against the kick, towards the kit targets
    assert g["hihat"] > g["kick"]  # the hats (their own group) come up too


# --- stem profiles (stage 0, per stem) ---------------------------------------------------------

def test_split_steps_run_on_cached_stems_not_in_the_separation(tmp_path):
    import soundfile as sf

    from rehearsal_master import separate

    step = {"split": "pan", "input": "guitar", "outputs": {"left": "gl", "right": "gr", "both": "guitar"}}
    steps = [{"model": "m.ckpt", "input": "mix", "outputs": {}}, step]
    cfg = {"plan": "p", "plans": {"p": {"steps": steps}}}
    assert separate.model_steps(cfg) == steps[:1]  # the cache does not depend on the split
    sr = 44100
    t = np.arange(sr * 4) / sr
    a, b = 0.3 * np.sin(2 * np.pi * 440 * t), 0.3 * np.sin(2 * np.pi * 1250 * t)
    x = np.column_stack([a + b, 0.15 * a + b]).astype(np.float32)
    sf.write(tmp_path / "guitar.wav", x, sr, subtype="FLOAT")
    stems = {"guitar": tmp_path / "guitar.wav"}
    pcfg = {"enabled": True, "input": "guitar", "outputs": step["outputs"]}
    assert separate.split(stems, pcfg | {"enabled": False}, tmp_path) == stems
    out = separate.split(stems, pcfg, tmp_path)
    assert set(out) == {"gl", "gr"} and (tmp_path / "guitar.wav").exists()
    total = sum(sf.read(f, dtype="float64", always_2d=True)[0] for f in out.values())
    np.testing.assert_allclose(total, x, atol=1e-5)


def test_profile_measures_floor_tone_and_punch(tmp_path):
    import soundfile as sf

    from rehearsal_master import profile

    n = FS * 20
    t = np.arange(n) / FS
    plays = (t % 4) < 3  # 3 s playing, 1 s rest
    hiss = 10 ** (-40 / 20) * rng.normal(0, 1, n)
    soft = 0.3 * np.sin(2 * np.pi * 200 * t) * plays + hiss   # a sustained note
    hits = np.zeros(n)
    hits[:: FS // 2] = 1.0
    sharp = signal_lfilter_decay(hits, 0.02) * rng.normal(0, 1, n) * plays + 1e-4 * rng.normal(0, 1, n)
    for name, x in (("soft", soft), ("sharp", sharp)):
        sf.write(tmp_path / f"{name}.wav", np.column_stack([x, x]).astype(np.float32), FS, subtype="FLOAT")
    p = profile.measure({k: tmp_path / f"{k}.wav" for k in ("soft", "sharp")})["stems"]
    assert 23 < p["soft"]["snr_db"] < 30  # the note at -13.5 dB RMS over a -40 dB hiss floor
    assert p["sharp"]["punch_db"] > p["soft"]["punch_db"] + 6
    assert p["sharp"]["hf_lf_db"] > p["soft"]["hf_lf_db"] + 20  # noise bursts vs a 200 Hz tone


def signal_lfilter_decay(x, tau_s):
    from scipy import signal as s

    a = np.exp(-1 / (tau_s * FS))
    return s.lfilter([1.0], [1.0, -a], x)


def test_tone_match_moves_towards_target_without_level_change():
    from rehearsal_master import profile

    grid = eq.log_grid(FS)
    measured = -0.0 * grid  # flat...
    measured = measured - 10 * np.log10(np.sum(10 ** (measured / 10)))
    target = measured + np.where(grid > 2000, -4.0, 0.0)  # ...the usual tone is darker
    cfg = {"strength": 1.0, "max_boost_db": 2.0, "max_cut_db": 3.0, "smooth_oct": 1.0, "fmin_hz": 40.0,
           "fmax_hz": 12000.0}
    g = profile.tone_match_curve(grid, list(measured), target, cfg)
    at = lambda f: g[np.argmin(np.abs(grid - f))]  # noqa: E731
    assert at(6000) - at(300) < -2.5  # the highs come down relative to the lows
    assert g.max() - g.min() <= 5.0 + 1e-9
    w = 10 ** (measured / 10)
    assert abs(10 * np.log10(np.sum(w * 10 ** (g / 10)) / np.sum(w))) < 1e-6  # level unchanged


def test_expander_follows_the_measured_floor_only_when_it_is_noise():
    from rehearsal_master import config, effects

    cfg = config.load(None)["tracks"]["bass"]
    x = 0.1 * rng.normal(0, 1, (FS * 4, 2))
    for snr, expected in ((33.0, 29.0), (60.0, 35.0), (12.0, 35.0)):  # hissy amp / clean / never pauses
        st: dict = {}
        effects.cleanup_track(x, FS, cfg | {"ducks": []}, None, st, None, {"snr_db": snr})
        assert st["expander"]["below_db"] == expected


# --- vocals recombined, tape plugin -------------------------------------------------------------

def test_vocal_room_recombination_is_exact_or_a_smooth_bounded_cut(tmp_path):
    import soundfile as sf

    n = FS * 6
    t = np.arange(n) / FS
    dry = (0.2 * np.sin(2 * np.pi * 220 * t) * ((t % 1) < 0.5))[:, None].repeat(2, 1)  # notes...
    room = 0.05 * rng.normal(0, 1, (n, 2)) * np.exp(-((t % 1) - 0.5).clip(0) * 4)[:, None]  # ...and their tails
    sf.write(tmp_path / "d.wav", dry.astype(np.float32), FS, subtype="FLOAT")
    sf.write(tmp_path / "r.wav", room.astype(np.float32), FS, subtype="FLOAT")
    cfg = {"max_reduction_db": 0.0, "smooth_ms": 50.0, "smooth_bins": 9}
    st = cleanup.recombine_room(tmp_path / "d.wav", tmp_path / "r.wav", tmp_path / "o.wav", cfg)
    y = sf.read(tmp_path / "o.wav", dtype="float64", always_2d=True)[0]
    np.testing.assert_allclose(y, dry + room, atol=1e-6)  # the vocal as recorded
    assert st["room_reduction_db"] == 0.0
    st = cleanup.recombine_room(tmp_path / "d.wav", tmp_path / "r.wav", tmp_path / "o.wav", cfg | {"max_reduction_db": 6.0})
    y = sf.read(tmp_path / "o.wav", dtype="float64", always_2d=True)[0]
    assert -6.0 <= st["room_reduction_db"] < -0.1
    tail = ((t % 1) > 0.6)
    e = lambda x: float(np.mean(x[tail] ** 2))  # noqa: E731
    assert e(y) < 0.6 * e(dry + room)  # the tails (room only) come down...
    note = ((t % 1) > 0.1) & ((t % 1) < 0.4)
    assert np.mean(y[note] ** 2) > 0.85 * np.mean((dry + room)[note] ** 2)  # ...the notes stay


def test_chowtape_keeps_level_and_alignment():
    import pytest

    from rehearsal_master import config, plugins

    cfg = config.load(None)["saturation"]["chowtape"]
    if not plugins.available(cfg["path"]):
        pytest.skip("CHOW Tape Model not installed")
    t = np.arange(FS * 2) / FS
    x = 0.1 * np.column_stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 330 * t)])
    y, st = plugins.chowtape(x, FS, cfg)
    assert abs(10 * np.log10(np.mean(y ** 2) / np.mean(x ** 2))) < 0.01
    c = _sig.correlate(y[:, 0], x[:, 0], mode="full", method="fft")
    assert abs(int(np.argmax(c)) - (len(x) - 1)) <= 3  # no delay to speak of
    assert st["harmonics_db"] < -10


# --- reinforcement: drum samples, bass fundamental, chest resonance ------------------------------

def test_drum_samples_go_on_real_hits_in_phase_and_not_on_bleed():
    from rehearsal_master import augment, config

    cfg = config.load(None)["tracks"]["kick"]["augment"]
    cfg = cfg | {"library": cfg["library"] | {"enabled": False}}  # the drummer's own samples only
    n = FS * 20
    t = np.arange(int(0.4 * FS)) / FS
    hit = np.sin(2 * np.pi * 60 * t * (1 + 0.5 * np.exp(-t / 0.02))) * np.exp(-t / 0.08)
    x = 1e-4 * rng.normal(0, 1, n)
    starts = np.arange(FS // 2, n - FS, FS)
    for i, s in enumerate(starts):
        x[s : s + len(hit)] += (0.5 + 0.3 * (i % 3) / 2) * hit
    bleed = starts[::4] + int(0.6 * FS)  # faint "hits" in between: another drum's bleed
    for s in bleed:
        x[s : s + len(hit)] += 0.02 * hit
    x = np.column_stack([x, x])
    y, st, samples = augment.drum_augment(x, FS, cfg, {})
    assert st["placed"] == len(starts) and st["samples"] >= 1
    add = y - x
    for s in bleed:
        assert np.abs(add[s + 200 : s + 2000]).max() < 1e-3 * cfg["mix"]
    s = starts[5]
    assert np.dot(add[s : s + 2000, 0], x[s : s + 2000, 0]) > 0.9 * np.linalg.norm(add[s : s + 2000, 0]) * np.linalg.norm(x[s : s + 2000, 0])


def test_bass_fundamental_is_raised_in_phase_and_leaves_harmonics():
    from rehearsal_master import augment, config

    cfg = config.load(None)["tracks"]["bass"]["bass_fundamental"] | {"restore": False, "fill": 0.7}
    t = np.arange(FS * 6) / FS
    f0 = np.where(t < 3, 55.0, 82.4)  # A1 then E2
    ph = 2 * np.pi * np.cumsum(f0) / FS
    fund = np.where(t < 3, 0.3, 0.05)  # the second note's fundamental is weak (a picked high note)
    x = fund * np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    y, st, _ = augment.bass_fundamental(np.column_stack([x, x]), FS, cfg)
    y = y[:, 0]
    def amp(sig, f, a, b):
        seg = sig[int(a * FS) : int(b * FS)]
        k = np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS)
        return 2 * abs(np.mean(seg * k))
    assert amp(y, 55, 1, 2.5) > amp(x, 55, 1, 2.5) * 10 ** (1.5 / 20)  # raised, so in phase (no cancelling)
    r = amp(y, 82.4, 4, 5.5) / amp(x, 82.4, 4, 5.5)
    assert 2.5 < r <= 10 ** (cfg["max_db"] / 20) * 1.02  # the weak one filled up, but never past max_db
    assert abs(amp(y, 110, 1, 2.5) / amp(x, 110, 1, 2.5) - 1) < 0.05  # harmonics untouched


def test_bass_fundamental_restore_brings_weak_low_notes_up_to_their_2nd_harmonic():
    from rehearsal_master import augment, config

    cfg = config.load(None)["tracks"]["bass"]["bass_fundamental"]
    assert cfg["restore"]
    t = np.arange(FS * 6) / FS
    f0 = np.where(t < 3, 41.2, 98.0)  # E1 (its fundamental lost by the room: 22 dB under the 2nd harmonic), then G2
    ph = 2 * np.pi * np.cumsum(f0) / FS
    fund = np.where(t < 3, 0.3 * 10 ** (-22 / 20), 0.3)
    x = fund * np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.15 * np.sin(3 * ph)
    y, st, _ = augment.bass_fundamental(np.column_stack([x, x]), FS, cfg)
    y = y[:, 0]
    def amp(sig, f, a, b):
        seg = sig[int(a * FS) : int(b * FS)]
        k = np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS)
        return 2 * abs(np.mean(seg * k))
    low = 20 * np.log10(amp(y, 41.2, 1, 2.5) / amp(y, 82.4, 1, 2.5))
    assert abs(low - cfg["restore_db"]) < 1.5  # the lost fundamental back, to restore_db under the 2nd harmonic
    high = amp(y, 98.0, 4, 5.5) / amp(x, 98.0, 4, 5.5)
    assert abs(20 * np.log10(high) - cfg["boost_db"]) < 0.7  # a healthy note: only the plain boost
    assert abs(amp(y, 196.0, 4, 5.5) / amp(x, 196.0, 4, 5.5) - 1) < 0.05  # harmonics untouched


def test_bass_fundamental_punch_raises_the_fundamental_at_note_starts_only():
    from rehearsal_master import augment, config

    cfg = config.load(None)["tracks"]["bass"]["bass_fundamental"] | {"attack_db": 3.0, "restore": False, "boost_db": 0.0}
    t = np.arange(FS * 3) / FS
    ph = 2 * np.pi * 55.0 * t
    x = np.column_stack([sum(0.3 / k * np.sin(k * ph) for k in range(1, 6))] * 2)
    pitch = (np.full(300, 55.0), np.arange(300) * 0.01, np.array([1.0, 2.0]))
    y, st, _ = augment.bass_fundamental(x, FS, cfg, pitch)
    def amp(sig, f, a, b):
        seg = sig[int(a * FS) : int(b * FS), 0]
        return 2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS)))
    assert abs(20 * np.log10(amp(y, 55.0, 1.0, 1.018) / amp(x, 55.0, 1.0, 1.018)) - 3.0) < 1.0  # the note's start
    assert abs(20 * np.log10(amp(y, 55.0, 1.3, 1.7) / amp(x, 55.0, 1.3, 1.7))) < 0.3  # its body as it was
    assert abs(amp(y, 110.0, 0.9, 1.1) / amp(x, 110.0, 0.9, 1.1) - 1) < 0.05  # the harmonics untouched


def test_harmonic_clean_keeps_the_harmonics_and_removes_what_lies_between():
    from rehearsal_master import augment, config

    cfg = config.load(None)["tracks"]["bass"]
    t = np.arange(FS * 6) / FS
    f0 = np.where(t < 3, 55.0, 73.4)
    ph = 2 * np.pi * np.cumsum(f0) / FS
    tone = sum(0.3 / k * np.sin(k * ph) for k in range(1, 30))
    buzz = 0.01 * np.random.default_rng(1).standard_normal(len(t))  # broadband: mostly between the harmonics
    x = np.column_stack([tone + buzz] * 2)
    pitch = augment.bass_pitch(x, FS, cfg["bass_fundamental"])
    y, st = augment.harmonic_clean(x, FS, cfg["harmonic_clean"] | {"fmax_hz": 5000.0}, *pitch)
    for lo, hi in ((150, 660), (660, 1700), (1700, 5000)):  # the buzz roughly halved in every band...
        sos = signal.butter(4, [lo, hi], "band", fs=FS, output="sos")
        left = np.sum(signal.sosfilt(sos, y[:, 0] - tone)[FS:-FS] ** 2) / np.sum(signal.sosfilt(sos, buzz)[FS:-FS] ** 2)
        assert left < 0.6
    def amp(sig, f, a, b):
        seg = sig[int(a * FS) : int(b * FS), 0]
        return 2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS)))
    for k in (1, 3, 5, 11, 17, 25):  # ...and the harmonics kept, low (by the pitch) and high (as spectral peaks)
        assert abs(amp(y, 55.0 * k, 1, 2.5) / amp(x, 55.0 * k, 1, 2.5) - 1) < 0.02
        assert abs(amp(y, 73.4 * k, 3.6, 5) / amp(x, 73.4 * k, 3.6, 5) - 1) < 0.02


def test_note_eq_keeps_each_notes_harmonics_and_tames_a_clanky_attack():
    from rehearsal_master import config, note_eq

    cfg = config.load(None)["tracks"]["bass"]
    rng2 = np.random.default_rng(2)
    t = np.arange(FS * 8) / FS
    # a note every 0.5 s, alternating A1 / D2 (12 harmonics: their space ends near 900 Hz), plucked
    f0 = np.where((t // 0.5) % 2 == 0, 55.0, 73.4)
    ph = 2 * np.pi * np.cumsum(f0) / FS
    pluck = np.exp(-3 * (t % 0.5))
    tone = sum(0.3 / k * np.sin(k * ph) for k in range(1, 13)) * pluck
    buzz = 0.003 * rng2.standard_normal(len(t))  # broadband: between the harmonics and above them
    # every note's attack: a short click at 1-1.4 kHz (over its harmonics, under attack.protect_hz), one in four 12 dB louder
    sos = signal.butter(2, [1000, 1400], "band", fs=FS, output="sos")
    click = np.zeros_like(t)
    for i, s in enumerate(np.arange(0.5, 7.5, 0.5)):
        a = int(s * FS)
        click[a:a + 480] += (0.4 if i % 4 == 0 else 0.1) * signal.sosfilt(sos, rng2.standard_normal(480)) * np.hanning(480)
    x = np.column_stack([tone + buzz + click] * 2)
    y, st, _ = note_eq.note_eq(x, FS, cfg["note_eq"] | {"enabled": True}, cfg["bass_fundamental"])
    assert st["pitched_pct"] > 80

    def amp(sig, f, a, b):
        seg = sig[int(a * FS) : int(b * FS), 0]
        return 2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS)))
    for k in (2, 3, 5, 8, 11):  # the notes' harmonics kept, mid-note (attacks aside)
        assert abs(amp(y, 55.0 * k, 2.1, 2.45) / amp(x, 55.0 * k, 2.1, 2.45) - 1) < 0.1
        assert abs(amp(y, 73.4 * k, 2.6, 2.95) / amp(x, 73.4 * k, 2.6, 2.95) - 1) < 0.1
    def left(f0c, a, b, lo, hi):  # what is left between (or above) the harmonics, mid-note
        i, n = int(a * FS), int(round((b - a) * FS))
        seg = lambda z: np.abs(np.fft.rfft(z[i:i + n, 0] * np.hanning(n))) ** 2  # noqa: E731
        f = np.fft.rfftfreq(n, 1 / FS)
        d = np.abs(f - f0c * np.round(f / f0c))
        m = (d > 0.3 * f0c) & (f >= lo) & (f < hi)
        return seg(y)[m].sum() / seg(x)[m].sum()
    for f0c, a in ((55.0, 2.12), (73.4, 2.62)):
        assert left(f0c, a, a + 0.3, 1500, 6000) < 0.1  # above the notes' space: about depth_db down
        # between the harmonics, above fmin_hz (250): a low A's are only 55 Hz apart, a few STFT bins (about -4.5 dB)
        assert left(f0c, a, a + 0.3, 300, 700) < (0.4 if f0c < 60 else 0.3)
        assert left(f0c, a, a + 0.3, 60, 200) > 0.9  # below fmin_hz: the room's low end is left alone
    # the clanky attacks come down towards the others, the others stay about as they were
    def burst(sig, s):
        b2 = signal.sosfilt(sos, sig[:, 0])[int(s * FS):int((s + 0.012) * FS)]
        return 10 * np.log10(np.sum(b2 ** 2))
    loud = np.mean([burst(y, s) - burst(x, s) for i, s in enumerate(np.arange(0.5, 7.5, 0.5)) if i % 4 == 0])
    soft = np.mean([burst(y, s) - burst(x, s) for i, s in enumerate(np.arange(0.5, 7.5, 0.5)) if i % 4])
    assert loud < -2 and abs(soft) < 2  # (the click lies between two of the attack EQ's band centres: part of the cut)


def test_upward_dynamic_eq_fills_thin_moments_only():
    from rehearsal_master import effects

    t = np.arange(FS * 6) / FS
    body = 0.2 * np.sin(2 * np.pi * 180 * t) * np.where((t % 2) < 1, 1.0, 0.25)  # thin every other second
    top = 0.2 * np.sin(2 * np.pi * 1500 * t)
    x = np.column_stack([body + top] * 2) * (t < 5)[:, None]  # and a silent last second
    band = {"mode": "boost", "freq_hz": 180.0, "q": 0.9, "threshold_db": 3.0, "ratio": 2.0, "max_boost_db": 3.0}
    y, st, _ = effects.dynamic_eq(x, FS, [band], loudness.active_mask(x, FS))
    def lvl(sig, a, b, f):
        seg = sig[int(a * FS) : int(b * FS), 0]
        return 20 * np.log10(2 * abs(np.mean(seg * np.exp(-2j * np.pi * f * np.arange(len(seg)) / FS))) + 1e-12)
    assert lvl(y, 1.3, 1.9, 180) - lvl(x, 1.3, 1.9, 180) > 1.0     # thin: raised
    assert abs(lvl(y, 2.3, 2.9, 180) - lvl(x, 2.3, 2.9, 180)) < 0.5  # normal: left alone
    assert np.abs(y[int(5.2 * FS):]).max() < 1e-9                    # silence stays silent


# --- measurement-driven mixing ----------------------------------------------------------------

def test_stereo_widens_gradually_between_50_and_400_hz():
    t = np.arange(FS * 4) / FS
    def side_kept(f):
        s = np.sin(2 * np.pi * f * t)
        y = stereo.mid_side(np.column_stack([s, -s]), FS, 120.0, 1.0, (50.0, 400.0))
        mid = slice(FS, 3 * FS)
        return np.sqrt(np.mean(y[mid, 0] ** 2) / np.mean(s[mid] ** 2))
    assert side_kept(30) < 0.05 and side_kept(1000) > 0.97
    assert 0.3 < side_kept(140) < 0.7  # half way (in log frequency)


def test_balance_targets_blend_with_published_priors():
    cfg = {"groups": [{"name": "vocals", "stems": ["v"], "lu": -9.0}, {"name": "drums", "stems": ["d"], "lu": -3.0}],
           "offsets": {}, "strength": 1.0, "max_change_db": 12.0, "ignore_below_db": 30.0, "drop_below_lu": -60.0,
           "priors": {"vocals": -3.0, "drums": -7.0}, "prior_weight": 0.5}
    _, rep = cleanup.balance_gains({"v": -20.0, "d": -20.0}, -10.0, cfg)
    assert rep["vocals"]["target_lu"] == -6.0 and rep["drums"]["target_lu"] == -5.0
    # as a pattern: the vocal ends 1 dB above the drums (targets -6 vs -5 would be -1; they were equal)
    assert abs((rep["vocals"]["after_lu"] - rep["drums"]["after_lu"]) - (-6.0 + 5.0)) < 0.05


def test_snare_gate_opens_on_hits_not_on_resonance_swells():
    from rehearsal_master import config, effects

    g = config.load(None)["tracks"]["snare"]["gate"]
    n = FS * 8
    t = np.arange(n) / FS
    # the body resonating along with a bass note (a slow swell, 20 dB under the hits)...
    swell = 0.05 * np.sin(2 * np.pi * 200 * t) * np.clip(np.sin(np.pi * t / 4), 0, None) ** 2
    buzz = 0.01 * rng.normal(0, 1, n) * np.clip(np.sin(np.pi * t / 4), 0, None) ** 2  # ...and its wires
    x = swell + buzz
    hits = np.arange(FS // 2, n - FS, FS)
    k = np.arange(int(0.15 * FS)) / FS
    for s in hits:  # real hits: sharp, at the same level
        x[s : s + len(k)] += 0.5 * np.sin(2 * np.pi * 220 * k) * np.exp(-k / 0.03) + 0.1 * rng.normal(0, 1, len(k)) * np.exp(-k / 0.02)
    x = np.column_stack([x, x])
    y, _, _ = effects.hit_gate(x, FS, g)
    hf = lambda a, b: float(np.mean(effects._band(y, FS, 1200, 12000)[a:b] ** 2) / np.mean(effects._band(x, FS, 1200, 12000)[a:b] ** 2))  # noqa: E731
    s = hits[1]  # the swell is at its loudest after it
    assert hf(s, s + int(0.03 * FS)) > 0.8                       # the hit's crack passes
    assert hf(s + int(0.5 * FS), s + int(0.9 * FS)) < 10 ** (-12 / 10)  # the buzz between hits goes well down


def test_library_samples_are_tuned_aligned_and_in_phase(tmp_path):
    import json

    import soundfile as sf

    from rehearsal_master import augment, config

    t = np.arange(int(0.4 * FS)) / FS
    def drum(f):
        return np.sin(2 * np.pi * f * t) * np.exp(-t / 0.08)
    (tmp_path / "kick").mkdir()
    one_shot = np.concatenate([np.zeros(int(0.002 * FS)), 0.5 * drum(66.0)])  # cut with 2 ms of pre-roll, as built
    sf.write(tmp_path / "kick" / "studio_loud.flac", one_shot.astype(np.float32), FS, subtype="PCM_24")
    (tmp_path / "index.json").write_text(json.dumps({"fs": FS, "drums": {"kick": [
        {"name": "studio", "pitch_hz": 66.0, "loud": "studio_loud.flac"}]}}))
    cfg = config.load(None)["tracks"]["kick"]["augment"]
    cfg = cfg | {"mix": 0.0, "library": cfg["library"] | {"dir": str(tmp_path), "mix": 0.5}}
    n = FS * 20
    x = 1e-4 * rng.normal(0, 1, n)
    starts = np.arange(FS // 2, n - FS, FS)
    for s in starts:
        x[s : s + len(t)] -= 0.6 * drum(60.0)  # the band's kick: 60 Hz, opposite polarity
    x = np.column_stack([x, x])
    augment._LIBRARY.clear()
    y, st, samples = augment.drum_augment(x, FS, cfg, {})
    assert st["library"]["placed"] == len(starts)
    assert abs(st["library"]["samples"][0]["tuned"] - 60.0 / 66.0) < 0.04  # tuned to the band's kick (150 ms pitch estimate)
    s = starts[4]
    add = (y - x)[s : s + 4000, 0]
    assert np.dot(add, x[s : s + 4000, 0]) > 0.8 * np.linalg.norm(add) * np.linalg.norm(x[s : s + 4000, 0])  # in phase


# --- space, rider, glue target ------------------------------------------------------------------

def test_sends_sit_at_their_measured_level_and_delay():
    from rehearsal_master import space

    t = np.arange(FS * 6) / FS
    x = (0.2 * np.sin(2 * np.pi * 440 * t) * ((t % 1) < 0.3))[:, None].repeat(2, 1)
    for c in ({"type": "plate", "level_lu": -18.0, "rt60_s": 1.0, "predelay_ms": 15.0},
              {"type": "slap", "level_lu": -22.0, "delay_ms": 95.0},
              {"type": "tape_echo", "level_lu": -15.0, "delay_ms": 140.0, "feedback": 0.4, "repeats": 3}):
        w, _ = space.send(x, FS, c)
        assert abs(loudness.integrated(w, FS) - loudness.integrated(x, FS) - c["level_lu"]) < 0.3
        if c["type"] != "plate":  # the first repeat comes delay_ms after the note
            first = np.flatnonzero(np.abs(w[:, 0]) > 0.05 * np.abs(w[:, 0]).max())[0]
            assert abs(first / FS * 1000 - c["delay_ms"]) < 8


def test_grit_adds_distortion_in_its_band_only():
    from rehearsal_master import space

    t = np.arange(FS * 3) / FS
    x = (0.3 * np.sin(2 * np.pi * 55 * t))[:, None].repeat(2, 1)
    y, _ = space.grit(x, FS, {"drive_db": 20.0, "band_hz": [800.0, 2000.0], "level_db": -10.0})
    f, py = _sig.welch((y - x)[:, 0], FS, nperseg=8192)
    band = (f > 800) & (f < 2000)
    assert py[band].sum() > 0.8 * py.sum()
    assert abs(10 * np.log10(np.mean((y - x) ** 2) / np.mean(x ** 2)) - (-10.0)) < 0.5


def test_vocal_rider_lifts_the_voice_where_the_band_gets_dense():
    from rehearsal_master import config, pipeline

    cfg = config.load(None)["rider"]
    t = np.arange(FS * 12) / FS
    v = 0.1 * np.sin(2 * np.pi * 2000 * t)[:, None].repeat(2, 1)
    dense = t > 6  # the guitars come in
    band = (0.05 + 0.25 * dense) * rng.normal(0, 1, len(t))
    band = _sig.sosfiltfilt(_sig.butter(2, [1000, 4000], "bandpass", fs=FS, output="sos"), band)[:, None].repeat(2, 1)
    out, st = pipeline._ride(v + band, v, FS, cfg)
    gain = lambda a, b: 20 * np.log10(np.sqrt(np.mean((out - band)[int(a * FS):int(b * FS)] ** 2) / np.mean(v[int(a * FS):int(b * FS)] ** 2)))  # noqa: E731
    assert gain(8, 11) > 1.5 and gain(8, 11) <= cfg["max_up_db"] + 0.1  # up where it is dense
    assert gain(1, 4) < 0.5                                              # not in the sparse part


def test_glue_target_finds_its_threshold():
    from rehearsal_master import config, dynamics

    g = config.load(None)["glue"]
    t = np.arange(FS * 10) / FS
    x = (0.3 * np.sin(2 * np.pi * 110 * t) * (1 + 0.8 * np.sin(2 * np.pi * 0.5 * t)))[:, None].repeat(2, 1)
    active = loudness.active_mask(x, FS)
    params = {k: g[k] for k in dynamics.BAND_PARAMS}
    lo, hi = 30.0, 99.5
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        _, _, st = dynamics.compress(x, FS, **(params | {"threshold_pct": mid}), active=active)
        lo, hi = (mid, hi) if st["avg_gr_db"] > g["target_gr_db"] else (lo, mid)
    _, _, st = dynamics.compress(x, FS, **(params | {"threshold_pct": 0.5 * (lo + hi)}), active=active)
    assert abs(st["avg_gr_db"] - g["target_gr_db"]) < 0.25


def test_chorus_send_is_wide_and_detuned_only_slightly():
    from rehearsal_master import space

    t = np.arange(FS * 4) / FS
    x = (0.2 * np.sin(2 * np.pi * 440 * t))[:, None].repeat(2, 1)
    w, _ = space.send(x, FS, {"type": "chorus", "level_lu": -16.0, "delay_ms": 12.0, "depth_ms": 1.5, "rate_hz": 0.35})
    assert abs(loudness.integrated(w, FS) - loudness.integrated(x, FS) + 16.0) < 0.5
    mid = slice(FS, 3 * FS)
    assert abs(np.corrcoef(w[mid, 0], w[mid, 1])[0, 1]) < 0.7  # left and right differ: width
    f, p = _sig.welch(w[mid, 0], FS, nperseg=1 << 15)
    assert abs(f[np.argmax(p)] - 440) < 3  # micro-pitch: a few cents, not a new note


def test_flattened_samples_level_out_the_hits(tmp_path):
    import json

    import soundfile as sf

    from rehearsal_master import augment, config

    t = np.arange(int(0.4 * FS)) / FS
    drum = lambda f: np.sin(2 * np.pi * f * t) * np.exp(-t / 0.08)  # noqa: E731
    (tmp_path / "kick").mkdir()
    sf.write(tmp_path / "kick" / "s_loud.flac", np.concatenate([np.zeros(96), 0.5 * drum(60.0)]).astype(np.float32), FS)
    (tmp_path / "index.json").write_text(json.dumps({"fs": FS, "drums": {"kick": [{"name": "s", "pitch_hz": 60.0, "loud": "s_loud.flac"}]}}))
    base = config.load(None)["tracks"]["kick"]["augment"]
    n = FS * 16
    x = 1e-4 * rng.normal(0, 1, n)
    starts = np.arange(FS // 2, n - FS, FS)
    vel = np.where(np.arange(len(starts)) % 2 == 0, 0.6, 0.3)  # loud, soft, loud, soft...
    for s, v in zip(starts, vel):
        x[s : s + len(t)] += v * drum(60.0)
    x = np.column_stack([x, x])
    spread = {}
    for fl in (0.0, 1.0):
        augment._LIBRARY.clear()
        cfg = base | {"mix": 0.0, "library": base["library"] | {"dir": str(tmp_path), "mix": 1.0, "flatten": fl}}
        y, _, _ = augment.drum_augment(x, FS, cfg, {})
        add = (y - x)[:, 0]
        lv = [np.sqrt(np.mean(add[s : s + 4000] ** 2)) for s in starts]
        spread[fl] = 20 * np.log10(max(lv) / min(lv))
    assert spread[0.0] > 5.0 and spread[1.0] < 1.0  # following the dynamics, or every hit alike


def test_presets_merge_nested_track_tables():
    from rehearsal_master import config

    c = config.load(None, "modern")["tracks"]["kick"]["augment"]
    assert c["enabled"] and c["library"]["enabled"] and c["library"]["mix"] == 0.55  # changed keys...
    assert c["library"]["drum"] == "kick" and c["clean_of"] == ["snare", "toms"]   # ...and the rest kept


def test_delay_send_follows_the_tempo_and_bounces():
    from rehearsal_master import space

    t = np.arange(FS * 4) / FS
    x = np.zeros((len(t), 2))
    x[FS // 2 : FS // 2 + 480] = 0.5  # one short click
    c = {"type": "delay", "level_lu": -16.0, "delay_beats": 0.75, "delay_ms": 300.0, "feedback": 0.5, "repeats": 3,
         "ping_pong": True, "hp_hz": 0.0, "lp_hz": 20000.0}
    w, st = space.send(x, FS, c, tempo_bpm=100.0)
    assert st["delay_ms"] == 450.0  # a dotted eighth at 100 BPM
    first = np.flatnonzero(np.abs(w).max(axis=1) > 1e-3 * np.abs(w).max())[0]
    assert abs((first - FS // 2) / FS * 1000 - 450.0) < 5
    a, b = FS // 2 + int(0.45 * FS), FS // 2 + int(0.9 * FS)  # first repeat left, second right
    assert np.abs(w[a : a + 480, 0]).max() > 2 * np.abs(w[a : a + 480, 1]).max()
    assert np.abs(w[b : b + 480, 1]).max() > 2 * np.abs(w[b : b + 480, 0]).max()


def test_dual_bass_chain_keeps_the_lows_clean_and_mono():
    from rehearsal_master import config, space

    cfg = config.load(None, "interpol")["tracks"]["bass"]["dual"]
    t = np.arange(FS * 4) / FS
    x = np.column_stack([0.3 * np.sin(2 * np.pi * 55 * t) + 0.1 * np.sin(2 * np.pi * 880 * t),
                         0.25 * np.sin(2 * np.pi * 55 * t) + 0.1 * np.sin(2 * np.pi * 880 * t)])
    off = cfg | {"drive_mix": 0.0, "chorus_mix": 0.0, "high_gain_db": 0.0, "low_compressor": cfg["low_compressor"] | {"enabled": False}}
    y0, _ = space.dual(x, FS, off)
    np.testing.assert_allclose(y0.mean(axis=1), x.mean(axis=1), atol=1e-9)  # nothing on: the bass as it was (mono lows)
    y, _ = space.dual(x, FS, cfg)
    lo = lambda s: _sig.sosfiltfilt(_sig.butter(4, 120, "low", fs=FS, output="sos"), s, axis=0)  # noqa: E731
    yl = lo(y)[FS : 3 * FS]
    assert np.abs(yl[:, 0] - yl[:, 1]).max() < 1e-2 * np.abs(yl).max()  # the lows stay mono (within -40 dB)...
    f, p = _sig.welch(y[FS : 3 * FS, 0], FS, nperseg=8192)
    _, p0 = _sig.welch(x[FS : 3 * FS, 0], FS, nperseg=8192)
    harm = (f > 1500) & (f < 6000)
    assert p[harm].sum() > 10 * p0[harm].sum()  # ...the upper part is driven (new harmonics)...
    hi = lambda s: _sig.sosfiltfilt(_sig.butter(4, 400, "high", fs=FS, output="sos"), s, axis=0)  # noqa: E731
    yh = hi(y)[FS : 3 * FS]
    assert np.corrcoef(yh[:, 0], yh[:, 1])[0, 1] < 0.95  # ...and chorused (wider)


def test_ensemble_steps_average_by_weight_and_plans_keep_their_own_cache():
    from rehearsal_master import config, separate

    one = {"model": "a.ckpt", "input": "mix", "outputs": {"vocals": "vocals"}}
    ens = {"input": "mix", "models": [{"model": "a.ckpt", "outputs": {"vocals": "vocals"}},
                                      {"model": "b.ckpt", "outputs": {"vocals": "vocals"}, "weight": 3.0,
                                       "weights": {"drums": 0.0}}]}
    assert [m["model"] for m in separate.members(one)] == ["a.ckpt"]
    m = separate.members(ens)
    assert m[0]["weight"] == 1.0 and m[1]["weight"] == 3.0 and m[1]["weights"] == {"drums": 0.0}
    a, b = np.ones((10, 2)), np.full((12, 2), 5.0)
    y = separate.combine([(1.0, a), (3.0, b)])
    assert y.shape == (10, 2) and np.allclose(y, 4.0)  # (1 * 1 + 3 * 5) / 4, cut to the shorter
    cfg = config.load(None)["stems"]
    assert separate.cache_name(dict(cfg, plan="cascade")) == ""  # the earlier stems stay where they were
    assert separate.cache_name(dict(cfg, plan="ensemble")) == "_ensemble"
    assert separate.cache_name(dict(cfg, backend="demucs")) == ""
    for plan in cfg["plans"]:  # every default plan is well-formed
        made = {"mix"}
        for step in separate.model_steps(dict(cfg, plan=plan)):
            if "combine" in step:
                assert set(step["inputs"]) <= made and step["combine"] == "max_spec"
                made.add(step["output"])
                continue
            assert step["input"] in made
            for mm in separate.members(step):
                assert mm["model"] and mm["outputs"]
                made |= set(mm["outputs"].values())


def test_max_spec_keeps_the_strongest_estimate_per_bin():
    from rehearsal_master import separate

    sr = 44100
    t = np.arange(sr * 3) / sr
    low, high = 0.5 * np.sin(2 * np.pi * 80 * t), 0.2 * np.sin(2 * np.pi * 2000 * t)
    a = np.column_stack([low, low])                        # one model kept only the fundamental,
    b = np.column_stack([0.5 * low + high, 0.5 * low + high])  # the other the partial (and less fundamental)
    y = separate.max_spec([a, b], block=sr)                 # (blocks shorter than the signal: the seams too)
    spec = lambda x, f: np.abs(np.fft.rfft(x[sr // 2: sr // 2 + sr, 0]))[f]  # 1 s: 1 Hz bins
    assert y.shape == a.shape
    assert abs(spec(y, 80) / spec(a, 80) - 1) < 0.05 and abs(spec(y, 2000) / spec(b, 2000) - 1) < 0.05


def test_cymbal_augment_tells_bells_crashes_and_rides_apart(tmp_path):
    import json

    import soundfile as sf

    from rehearsal_master import augment, config

    rng = np.random.default_rng(3)
    sr = FS
    def noise_hit(dur, tau, amp):
        t = np.arange(int(dur * sr)) / sr
        y = signal.sosfilt(signal.butter(4, [3000, 15000], "band", fs=sr, output="sos"), rng.standard_normal(len(t)))
        return amp * y * np.exp(-t / tau)
    def bell_hit(dur, amp):
        t = np.arange(int(dur * sr)) / sr
        return amp * sum(np.sin(2 * np.pi * f * t) for f in (2350.0, 4100.0, 5900.0, 7700.0)) / 4 * np.exp(-t / 0.5)
    x = np.zeros(sr * 12)
    plan = [(0.5, "crash"), (3.0, "ride"), (3.5, "ride"), (4.0, "ride"), (4.5, "bell"), (5.5, "ride"), (6.0, "ride"),
            (6.5, "bell"), (7.5, "crash"), (10.0, "ride"), (10.5, "ride")]
    for t0, k in plan:
        h = {"crash": noise_hit(2.0, 0.8, 0.6), "ride": noise_hit(0.45, 0.08, 0.25), "bell": bell_hit(1.0, 0.3)}[k]
        i = int(t0 * sr)
        x[i : i + len(h)] += h[: len(x) - i]
    x = np.column_stack([x, x])
    lib = tmp_path / "lib"
    index = {"fs": sr, "drums": {}}
    for k, h in (("crash", noise_hit(1.5, 0.8, 0.5)), ("ride", noise_hit(0.6, 0.1, 0.5)), ("bell", bell_hit(1.0, 0.5))):
        (lib / k).mkdir(parents=True)
        sf.write(lib / k / "a_loud.flac", h.astype(np.float32), sr, subtype="PCM_24")
        index["drums"][k] = [{"name": "a", "loud": "a_loud.flac", "pitch_hz": 100.0, "centroid_hz": 8000.0}]
    (lib / "index.json").write_text(json.dumps(index))
    cfg = config.load(None)["tracks"]["cymbals"]["cymbal_augment"] | {
        "dir": str(lib), "crash_pct": 40.0, "mix": {"crash": 0.5, "ride": 0.5, "bell": 0.5}, "min_flatness": {}}
    y, st, used = augment.cymbal_augment(x, FS, cfg)
    assert st["kinds"] == {"crash": 2, "ride": 7, "bell": 2}
    assert all(st[k]["placed"] == st["kinds"][k] for k in ("crash", "ride", "bell"))
    assert y.shape == x.shape and st["level_change_db"] > 0
