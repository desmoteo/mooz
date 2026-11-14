"""The bed under the hi-hat stem (bed.py): taken out between the hits, the hits left alone."""

import numpy as np
from scipy import signal

from mooz import bed, config

FS = 48000
CFG = config.load(None)["bed"]


def _noise(n, seed):
    return np.random.default_rng(seed).standard_normal(n)


def _bright_hits(n, every_s, decay_s, seed=1, start_s=0.0):
    """Hi-hat-like hits: noise above 3 kHz with an exponential decay."""
    x = np.zeros(n)
    env = np.exp(-np.arange(int(4 * decay_s * FS)) / (decay_s * FS))
    for s in np.arange(int(start_s * FS), n - len(env), int(every_s * FS)):
        x[s : s + len(env)] += env * _noise(len(env), seed + s % 977)
    return signal.sosfiltfilt(signal.butter(4, 3000.0, "high", fs=FS, output="sos"), x)


def _dark(n, seed=2):
    """Bed-like noise: 1.5-6 kHz, falling above."""
    return signal.sosfiltfilt(signal.butter(2, [1500.0, 6000.0], "band", fs=FS, output="sos"), _noise(n, seed))


def _level(x, lo, hi, frames, win_s=0.05):
    h = int(win_s * FS)
    b = signal.sosfiltfilt(signal.butter(4, [lo, hi], "band", fs=FS, output="sos"), x.mean(axis=1)) ** 2
    n = len(b) // h
    return 10 * np.log10(b[: n * h].reshape(n, h).mean(axis=1)[frames[:n]].mean() + 1e-20)


def _frames(n, every_s, offset_s, width_s, start_s=0.0):
    h = int(0.05 * FS)
    m = np.zeros(n // h, bool)
    for s in np.arange(int(start_s * FS), n, int(every_s * FS)):
        a = (s + int(offset_s * FS)) // h
        m[a : a + max(1, int(width_s / 0.05))] = True
    return m


def test_the_bed_goes_and_the_hits_stay():
    n = 60 * FS
    hits = _bright_hits(n, 0.4, 0.03)
    hits /= np.abs(hits).max()
    bed_ = _dark(n)
    bed_ *= 10 ** (-20 / 20) * np.abs(hits).max() / np.abs(bed_).max()
    bed_[: 20 * FS] = 0.0  # the band comes in at 20 s
    x = np.column_stack([hits + bed_] * 2)
    y, st = bed.remove_bed(x, FS, CFG)
    between = _frames(n, 0.4, 0.2, 0.15, start_s=20.0)  # 200-350 ms after each hit: bed only
    at = _frames(n, 0.4, 0.0, 0.025, start_s=20.0)      # the first 25 ms of each hit
    assert "note" not in st and st["hits"] > 100
    assert _level(y, 2000, 4000, between) - _level(x, 2000, 4000, between) < -6.0
    assert abs(_level(y, 2000, 4000, at) - _level(x, 2000, 4000, at)) < 0.5
    assert abs(_level(y, 9000, 16000, between) - _level(x, 9000, 16000, between)) < 0.3  # (the 8 kHz filter skirt reaches into the band)
    assert np.allclose(y[: 10 * FS], x[: 10 * FS], atol=1e-3)  # before the band comes in: hits alone, untouched


def test_an_open_hats_own_decay_is_not_a_bed():
    n = 40 * FS
    x = _bright_hits(n, 0.6, 0.4)  # long decays: between the hits there is only the hat itself
    x = np.column_stack([x / np.abs(x).max()] * 2)
    y, st = bed.remove_bed(x, FS, CFG)
    assert "note" in st and "bright" in st["note"]
    assert y is x


def test_the_snares_burst_is_keyed_on_the_snare_stem():
    n = 60 * FS
    hits = _bright_hits(n, 0.4, 0.03)
    hits /= np.abs(hits).max()
    bed_ = _dark(n) * 10 ** (-24 / 20)
    bed_ /= np.abs(_dark(n)).max()
    bed_ *= 10 ** (-24 / 20)
    snare = np.zeros(n)
    burst = np.zeros(n)
    env = np.exp(-np.arange(int(0.08 * FS)) / (0.02 * FS))
    body = np.sin(2 * np.pi * 200 * np.arange(len(env)) / FS) * env
    dark = _dark(len(env) * 4)[: len(env)] * env
    dark /= np.abs(dark).max()
    for s in np.arange(int(0.2 * FS), n - len(env), int(1.2 * FS)):  # 200 ms after a hat: no hat within 150 ms
        snare[s : s + len(env)] += body
        burst[s : s + len(env)] += dark * 10 ** (-14 / 20)
    x = np.column_stack([hits + bed_ + burst] * 2)
    key = np.column_stack([snare] * 2)
    y, st = bed.remove_bed(x, FS, CFG, key)
    assert "note" not in st and st["snare_alone_hits"] >= 10 and st["snare_burst_re_snare_db"] is not None
    at_snare = _frames(n, 1.2, 0.2, 0.05)
    assert _level(y, 2000, 4000, at_snare) - _level(x, 2000, 4000, at_snare) < -6.0
    at_hat = _frames(n, 0.4, 0.0, 0.025)
    assert abs(_level(y, 2000, 4000, at_hat) - _level(x, 2000, 4000, at_hat)) < 0.5
