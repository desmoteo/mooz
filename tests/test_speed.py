"""The faster ways give what the plain ones gave (docs/SPEED.md)."""

import numpy as np
import pytest
from scipy import signal

from mooz import loudness


def _signal(n=300_000, ch=2, dtype=np.float64, seed=0):
    return (np.random.default_rng(seed).standard_normal((n, ch)) * 0.3).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("ch", [1, 2, 3])
def test_the_channel_helpers_are_the_expressions_they_stand_for(dtype, ch):
    x = _signal(ch=ch, dtype=dtype)
    for fast, plain in ((loudness.mono(x), x.mean(axis=1)), (loudness.peak(x), np.abs(x).max(axis=1)),
                        (loudness.power(x), np.mean(x * x, axis=1))):
        assert fast.dtype == plain.dtype and np.array_equal(fast, plain)


def _one_thread(monkeypatch):
    monkeypatch.setattr(loudness, "THREADS", 1)
    assert loudness.pool() is None


def test_oversampling_on_threads_gives_the_same_chunks_in_the_same_order(monkeypatch):
    x = _signal(700_000)
    assert loudness.THREADS > 1 or pytest.skip("one thread on this machine")
    many = [(s, e, a, up.copy()) for s, e, a, up in loudness.oversampled_chunks(x, 4)]
    _one_thread(monkeypatch)
    one = list(loudness.oversampled_chunks(x, 4))
    assert len(many) == len(one) == 6
    for (s, e, a, up), (s1, e1, a1, up1) in zip(many, one):
        assert (s, e, a) == (s1, e1, a1) and np.array_equal(up, up1)
        assert np.array_equal(up, signal.resample_poly(x[a : min(len(x), e + 64)], 4, 1, axis=0))


def test_the_true_peak_envelope_is_the_maximum_over_channels_and_phases(monkeypatch):
    x = _signal(400_000)
    env = loudness.true_peak_envelope(x)
    plain = np.empty(len(x))  # as it was written before the threads
    for s in range(0, len(x), 1 << 17):
        e, a, b = min(len(x), s + (1 << 17)), max(0, s - 64), min(len(x), s + (1 << 17) + 64)
        up = signal.resample_poly(x[a:b], 4, 1, axis=0)
        plain[s:e] = np.abs(up[(s - a) * 4 : (e - a) * 4]).reshape(e - s, -1).max(axis=1)
    np.maximum(plain, np.abs(x).max(axis=1), out=plain)
    assert np.array_equal(env, plain)
    _one_thread(monkeypatch)
    assert np.array_equal(loudness.true_peak_envelope(x), plain)


def test_the_compiled_envelope_follower_is_the_closed_form():
    from mooz import dynamics

    fs = 48000
    v = np.abs(_signal(200_000)[:, 0])
    v[50_000:60_000] = 0.0  # silence: the floor of 1e-6
    for release_ms in (5.0, 30.0, 300.0):
        c = -1000.0 / (release_ms * fs)
        ramp = c * np.arange(len(v))
        plain = np.exp(np.maximum.accumulate(np.log(np.maximum(v, 1e-6)) - ramp) + ramp)  # as it was written
        got = dynamics.release_hold(v, release_ms, fs)
        assert got.dtype == np.float64 and np.allclose(got, plain, rtol=1e-9, atol=0.0)
    rows = np.abs(_signal(5_000, ch=7)).T  # along an axis of a 2-D array, and float32 in: float64 out, as before
    two = dynamics.release_hold(rows.astype(np.float32), 80.0, 100.0, axis=1)
    assert two.shape == rows.shape and two.dtype == np.float64
    assert np.array_equal(two[3], dynamics.release_hold(rows[3].astype(np.float32), 80.0, 100.0))
    cols = dynamics.release_hold(rows.T.astype(np.float32), 80.0, 100.0, axis=0)
    assert np.array_equal(cols, two.T)


def test_stage_files_written_on_the_thread_are_the_files_written_at_once(tmp_path, monkeypatch):
    import soundfile as sf

    from mooz import stages

    x = _signal(96_000)
    w = stages.StageWriter(tmp_path / "s", "flac")
    w.write("2_stems/a", x, 48000)
    kept = x.copy()
    x *= 0.0  # the run goes on with its array: the file is of what it was when written
    w.write("5_mix", kept * 20.0, 48000)  # above full scale even with the offset: scaled down, not clipped
    assert w.written == ["2_stems/a", "5_mix"] and w.files["2_stems/a"] == tmp_path / "s/2_stems/a.flac"
    w.flush()
    sf.write(tmp_path / "plain.flac", kept * 10 ** (stages.GAIN_DB / 20), 48000, subtype="PCM_24", format="FLAC")
    assert np.array_equal(sf.read(w.files["2_stems/a"])[0], sf.read(tmp_path / "plain.flac")[0])
    assert np.abs(sf.read(w.files["5_mix"])[0]).max() < 1.0
    w.flush()  # nothing under way: returns

    def full(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(stages.sf, "write", full)
    w.write("4_drum_bus", kept, 48000)
    with pytest.raises(OSError, match="no space"):
        w.flush()  # what the encoding raised is raised where the run waits for it
    monkeypatch.undo()
    for i in range(5):  # more files than may wait: write() holds the run back, and all of them arrive
        w.write(f"5_premix/p{i}", kept, 48000)
    w.flush()
    assert all((tmp_path / f"s/5_premix/p{i}.flac").stat().st_size > 1000 for i in range(5))


def test_mp3_encoders_are_started_and_waited_for(tmp_path):
    import shutil
    import subprocess

    import soundfile as sf

    from mooz import export

    if shutil.which("ffmpeg") is None:
        pytest.skip("no ffmpeg")
    sf.write(tmp_path / "a.wav", _signal(48_000), 48000, subtype="PCM_24")
    started: list = []
    assert export.start_mp3(tmp_path / "a.wav", tmp_path / "a.mp3", "128k", {"title": "a"}, started) and len(started) == 1
    export.finish(started)
    assert started == [] and (tmp_path / "a.mp3").stat().st_size > 1000
    assert export.start_mp3(tmp_path / "missing.wav", tmp_path / "b.mp3", "128k", {}, started)
    with pytest.raises(subprocess.CalledProcessError):
        export.finish(started)


def test_the_launcher_asks_for_huge_pages_unless_the_variable_is_set():
    from mooz import cli

    assert cli._with_huge_pages({"A": "1"}) == {"A": "1", "GLIBC_TUNABLES": "glibc.malloc.hugetlb=1", "_MOOZ_TUNED": "1"}
    mine = cli._with_huge_pages({"GLIBC_TUNABLES": "glibc.malloc.arena_max=2"})
    assert mine["GLIBC_TUNABLES"] == "glibc.malloc.arena_max=2" and mine["_MOOZ_TUNED"] == "1"


def test_filters_on_a_thread_per_channel_give_the_same_samples(monkeypatch):
    x = _signal(300_000)
    sos = signal.butter(4, [150.0, 400.0], "bandpass", fs=48000, output="sos")
    plain = np.column_stack([signal.sosfiltfilt(sos, x[:, c]) for c in range(2)])
    assert loudness.THREADS > 1 or pytest.skip("one thread on this machine")
    assert np.array_equal(loudness.filtfilt(sos, x), plain)
    assert np.array_equal(loudness.filtfilt(sos, x[:, 0]), plain[:, 0])  # one channel: as ever
    _one_thread(monkeypatch)
    assert np.array_equal(loudness.filtfilt(sos, x), plain)


def test_filtering_the_mono_sum_is_the_mono_sum_of_the_filtered_to_rounding():
    x = _signal(300_000)
    for sos in (signal.butter(2, [40.0, 150.0], "bandpass", fs=48000, output="sos"), signal.butter(2, 2000.0, "high", fs=48000, output="sos")):
        after, before = loudness.mono(loudness.filtfilt(sos, x)), loudness.filtfilt(sos, loudness.mono(x))
        assert np.max(np.abs(after - before)) < 1e-12 * np.max(np.abs(after))


def test_compiled_smoothing_is_scipys_one_pole():
    from mooz import dynamics

    v = np.abs(_signal(200_000)[:, 0])
    for attack_ms in (0.5, 10.0, 30.0):
        a = np.exp(-1000.0 / (attack_ms * 48000))
        assert np.array_equal(dynamics.attack_smooth(v, attack_ms, 48000), signal.lfilter([1 - a], [1, -a], v))
    rows = np.abs(_signal(4_000, ch=5)).T
    a = np.exp(-1000.0 / (20.0 * 100.0))
    assert np.array_equal(dynamics.attack_smooth(rows, 20.0, 100.0, axis=1), signal.lfilter([1 - a], [1, -a], rows, axis=1))
    assert np.array_equal(dynamics.attack_smooth(rows.T, 20.0, 100.0, axis=0), signal.lfilter([1 - a], [1, -a], rows.T, axis=0))
    assert dynamics.attack_smooth(v.astype(np.float32), 5.0, 48000).dtype == np.float64


def test_a_threshold_search_on_the_level_alone_finds_what_the_compressor_gives():
    from mooz import dynamics

    x = _signal(300_000) * np.linspace(0.2, 1.0, 300_000)[:, None]
    active = loudness.active_mask(x, 48000)
    p = dict(ratio=2.0, knee_db=6.0, attack_ms=30.0, release_ms=200.0, rms_ms=50.0)
    level = dynamics.level_db(x, 48000, p["rms_ms"])
    for pct in (20.0, 55.5, 90.0):
        y, gr, st = dynamics.compress(x, 48000, threshold_pct=pct, **p, active=active)
        assert dynamics.average_reduction(level, 48000, threshold_pct=pct, **p, active=active) == st["avg_gr_db"]
        y2, gr2, st2 = dynamics.compress(x, 48000, threshold_pct=pct, **p, active=active, level=level)
        assert np.array_equal(y, y2) and np.array_equal(gr, gr2) and st == st2


def test_the_loudness_summary_filters_once_and_says_what_it_said(monkeypatch):
    x = _signal(400_000) * np.array([1.0, 0.6])
    plain = signal.sosfilt(loudness.k_weighting_sos(48000), x, axis=0)
    assert loudness.THREADS > 1 or pytest.skip("one thread on this machine")
    assert np.array_equal(loudness._k_weighted(x, 48000), plain)
    values, ld = loudness.summary(x, 48000)
    assert values["balance_db"] == round(loudness.channel_balance_db(x, 48000), 1) and abs(values["balance_db"] - 4.4) < 0.2
    assert ld.integrated == loudness.integrated(x, 48000) == loudness.measure(x, 48000).integrated
    _one_thread(monkeypatch)
    assert np.array_equal(loudness._k_weighted(x, 48000), plain) and loudness.summary(x, 48000)[0] == values
    assert loudness.summary(x[:, :1], 48000)[0]["balance_db"] == 0.0


def test_the_bass_pitch_is_kept_with_the_stem(tmp_path, monkeypatch):
    librosa = pytest.importorskip("librosa")
    from mooz import augment

    t = np.arange(48000 * 3) / 48000
    x = np.column_stack([np.sin(2 * np.pi * 55.0 * t)] * 2) * 0.3
    cfg = {"fmin_hz": 30.0, "fmax_hz": 200.0}
    plain = augment.pyin(x, 48000, cfg)  # nowhere to keep it
    assert abs(np.nanmedian(plain[0]) - 55.0) < 1.0
    monkeypatch.setattr(augment, "PITCH_DIR", tmp_path / "_pitch")
    first = augment.pyin(x, 48000, cfg)
    assert len(list((tmp_path / "_pitch").glob("*.npz"))) == 1
    monkeypatch.setattr(librosa, "pyin", lambda *a, **k: pytest.fail("tracked again"))
    again = augment.pyin(x, 48000, cfg)
    for a, b, c in zip(plain, first, again):
        assert np.array_equal(a, b, equal_nan=True) and np.array_equal(a, c, equal_nan=True)
    with pytest.raises(pytest.fail.Exception):  # other samples or other settings: another answer
        augment.pyin(x * 0.5, 48000, cfg)
    with pytest.raises(pytest.fail.Exception):
        augment.pyin(x, 48000, cfg | {"fmax_hz": 300.0})


def test_the_expressions_on_threads_are_numpys():
    from mooz import expr

    rng = np.random.default_rng(3)
    n = 200_000
    assert n > expr.SHORT
    g = rng.uniform(-30.0, 12.0, n)
    a, b = np.abs(rng.standard_normal(n)), np.abs(rng.standard_normal(n))
    x = _signal(n)
    eps = loudness.EPS
    for fast_, plain in ((expr.linear(g), 10 ** (g / 20)), (expr.db(a, eps), 20 * np.log10(a + eps)), (expr.power_db(a, eps), 10 * np.log10(a + eps)),
                         (expr.ratio_db(a, b, eps), 20 * np.log10((a + eps) / (b + eps))), (expr.gained(x, g), x * (10 ** (g / 20))[:, None]),
                         (expr.gained(x, -g), x * (10 ** (-g / 20))[:, None])):
        assert fast_.dtype == plain.dtype and fast_.shape == plain.shape and np.array_equal(fast_, plain)
    over = rng.uniform(-20.0, 20.0, n)
    for knee in (1e-6, 6.0, 12.0):
        slope = 1.0 - 1.0 / 3.0
        plain = np.where(over > knee / 2, slope * over, slope * (over + knee / 2) ** 2 / (2 * knee))
        assert np.array_equal(expr.knee(over, slope, knee), plain)
    few = slice(0, 100)  # short arrays go through numpy: the same values again
    assert np.array_equal(expr.linear(g[few]), 10 ** (g[few] / 20)) and np.array_equal(expr.gained(x[few], g[few]), x[few] * (10 ** (g[few] / 20))[:, None])


def _job(tmp_path, name, source=True):
    """A separation job with a plan of no steps: the process reads the source and answers, no model loaded."""
    import json

    import soundfile as sf

    out = tmp_path / name
    out.mkdir()
    if source:
        sf.write(out / "_input.wav", _signal(4_800).astype(np.float32), 48000, subtype="FLOAT")
    cfg = {"backend": "roformer", "plan": "none", "plans": {"none": {"steps": []}}, "device": "cpu", "keep_models": False}
    (out / "_job.json").write_text(json.dumps({"input": str(out / "_input.wav"), "out_dir": str(out), "cfg": cfg}))
    return out / "_job.json"


def test_one_separating_process_takes_one_source_after_another(tmp_path):
    """separate.py: the process stays between the sources of a run (its models stay loaded), a job that
    fails ends it and raises as a failed subprocess did, the next source starts another, release() ends it."""
    import subprocess

    from mooz import separate

    try:
        separate._ask(_job(tmp_path, "a"))
        first = separate._server["proc"]
        separate._ask(_job(tmp_path, "b"))
        assert separate._server["proc"] is first and first.poll() is None  # the same process, still there
        with pytest.raises(subprocess.CalledProcessError):
            separate._ask(_job(tmp_path, "c", source=False))  # no such file: the job fails
        assert separate._server is None and first.poll() is not None
        separate._ask(_job(tmp_path, "d"))
        second = separate._server["proc"]
        assert second is not first and second.poll() is None
        separate.release()
        assert separate._server is None and second.poll() == 0
        separate.release()  # nothing to end: nothing happens
    finally:
        separate.release()


def test_a_chain_says_how_far_it_is_after_each_processor_that_took_a_while(monkeypatch, capsys):
    """The web app's graph shows a chain's progress: pipeline._chain names the phase and the part of the
    chain it covers, effects.py ticks after each processor, and a tick is said only after work worth saying."""
    import json

    from mooz import effects, progress

    assert effects.CLEANUP_TICKS == _ticks_in(effects.cleanup_track) and effects.STUDIO_TICKS == _ticks_in(effects.studio_track)
    monkeypatch.setenv("REHEARSAL_PROGRESS", "1")
    clock = [100.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    progress.chain("bass", "chain", 0.3, 0.9, 4)
    progress.tick()  # a processor that is off: at once, no news
    clock[0] += 2.0
    progress.tick()
    clock[0] += 2.0
    progress.tick()
    clock[0] += 2.0
    progress.tick()
    progress.tick()  # one more than said: never past the phase's end
    said = [json.loads(line[2:]) for line in capsys.readouterr().out.splitlines() if line.startswith("@@")]
    assert [(e["state"], e["frac"]) for e in said] == [("chain", 0.3), ("chain", 0.6), ("chain", 0.75), ("chain", 0.9)]
    progress.chain("bass", "done", 1.0)  # a state with no phase: the ticks that follow belong to nobody
    clock[0] += 2.0
    progress.tick()
    assert len([line for line in capsys.readouterr().out.splitlines() if line.startswith("@@")]) == 1
    monkeypatch.delenv("REHEARSAL_PROGRESS")
    progress.chain("bass", "chain", 0.3, 0.9, 4)
    clock[0] += 2.0
    progress.tick()
    assert capsys.readouterr().out == ""  # nobody listening: nothing said, nothing kept


def _ticks_in(function) -> int:
    import inspect

    return inspect.getsource(function).count("progress.tick()")


def test_kept_stems_are_used_unless_the_run_is_told_to_separate_again_and_then_once(tmp_path, monkeypatch):
    """separate.py, --separate-again: a source's kept stems are left aside and it is separated once more; what
    the run has made itself it then uses (a second profile, the take's own turn after the stems made ahead)."""
    import json
    from pathlib import Path

    import soundfile as sf

    from mooz import separate

    x = _signal(4_800).astype(np.float32)  # (stereo)
    asked = []

    def stand_in(job):
        out = Path(json.loads(job.read_text())["out_dir"])
        asked.append(out.name)
        sf.write(out / "bass.wav", x * len(asked), 48000, subtype="FLOAT")

    monkeypatch.setattr(separate, "_ask", stand_in)
    monkeypatch.setattr(separate, "_made", set())
    cfg = {"backend": "demucs", "model": "m", "cache": True}
    one = lambda: float(sf.read(separate.separate(x, 48000, cfg, tmp_path / "take", {"file": "a"})["bass"])[0].max() / x.max())  # noqa: E731
    assert round(one()) == 1 and round(one()) == 1 and asked == ["take"]  # made, then kept
    monkeypatch.setattr(separate, "_made", set())  # another run
    assert round(one()) == 1 and asked == ["take"]
    monkeypatch.setattr(separate, "AGAIN", True)
    assert round(one()) == 2 and round(one()) == 2 and asked == ["take", "take"]  # once more, and that one is then used
    assert json.loads((tmp_path / "take" / "key.json").read_text()) == {"file": "a"}
