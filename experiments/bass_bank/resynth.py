"""The bass rebuilt from its notes: what the recording lost (the low notes' fundamental, everything
above the harmonics the separation kept) synthesized along the pitch, what it has kept taken as it
is, and nothing else (the separation's noise between and above the harmonics is left out):

  body         the harmonics the stem has (up to meas_hz), each followed along the pitch (track_hz:
               its own amplitude and phase), rebalanced towards a studio bass (bank.target_profile;
               +-rebal_db), the rest of the stem left out
  fundamental  a clean tone at the studio bass's level re the 2nd harmonic (fund_margin_db under it,
               at most fund_max_db over the 2nd), following the 2nd harmonic's envelope, its phase locked
               to it (an octave down): no room noise amplified. It replaces the stem's own
  top          the harmonics above meas_hz up to top_hz, synthesized: the studio profile (beyond its
               10th harmonic, slope_db per octave down), a string's stretch (inharmonicity B), and a
               decay from the note's start that is faster the higher they are (decay_per_khz): a pluck
               is bright, then mellows. Their level set so that 1-3 kHz sits top_re_studio_db from a
               studio bass's long-term tone
  attacks      bank.py's attack track (attack_db)
  fill         where no pitch is found (a slide, a ghost note): bank.py's rebalanced output, crossfaded
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage, signal

import bank as bk

DEFAULTS = {
    "meas_hz": 700.0, "xfade_hz": 250.0, "track_hz": 15.0, "rebal_db": 6.0, "rebal_cut_db": 9.0,
    "fund_margin_db": 6.0, "fund_max_db": 6.0,
    "top_hz": 5000.0, "slope_db": 9.0, "stretch": 5e-5, "decay_per_khz": 4.0, "top_re_studio_db": -3.0,
    "attack_db": 0.0, "fill_db": 0.0,
}
STUDIO = Path(__file__).resolve().parents[2] / "rehearsal_master/references/raw_multitracks.json"


def _band_level(y: np.ndarray, fs: float, lo: float, hi: float) -> float:
    s = signal.butter(4, [lo, hi], "bandpass", fs=fs, output="sos")
    return float(np.mean(signal.sosfiltfilt(s, y) ** 2)) + 1e-30


def resynth(x: np.ndarray, fs: float, f0: np.ndarray, t_f: np.ndarray, onsets: np.ndarray, cfg: dict | None = None,
            bank_cfg: dict | None = None, parts: dict | None = None) -> tuple[np.ndarray, dict]:
    cfg = DEFAULTS | (cfg or {})
    prof = json.loads(bk.PROFILE.read_text())
    bparts: dict = {}
    bk.bank(x, fs, f0, t_f, onsets, (bank_cfg or {}) | {"attack_db": cfg["attack_db"]}, bparts)
    mono = bparts["input"]
    n = len(mono)
    t_n = np.arange(n) / fs
    step = int(fs // 100)
    t_g = np.arange(0, n, step) / fs
    ok = np.isfinite(f0)
    p = np.interp(t_g, t_f, np.where(ok, f0, 0.0))
    pitched = (np.interp(t_g, t_f, ok.astype(float)) > 0.99) & (p > 0)
    lf = np.interp(t_n, t_f[ok], np.log(f0[ok]))
    phi = 2 * np.pi * np.cumsum(np.exp(lf)) / fs
    # where a note is: 20 ms fades
    w_note = np.interp(t_n, t_g, ndimage.uniform_filter1d(pitched.astype(float), 3))
    lp = signal.butter(2, cfg["track_hz"], "low", fs=fs, output="sos")

    def track(k):  # the stem's k-th harmonic: complex envelope at the sample rate
        return 2 * (signal.sosfiltfilt(lp, mono * np.cos(k * phi)) - 1j * signal.sosfiltfilt(lp, mono * np.sin(k * phi)))

    # the note's level: its 2nd harmonic (frame rate); the studio profile is relative to it too (with the
    # mean of the 2nd-4th, the low notes' far too strong 4th harmonic raised every target: boom, box)
    amps = bk.harmonic_amps(mono, fs, f0, t_f, 2, step)[:, :len(t_g)]
    ref = 2 * amps[1]  # (harmonic_amps gives half the amplitude; track() the whole: both the same scale here)
    k_top = int(cfg["top_hz"] / np.nanmin(np.where(pitched, p, np.inf)))
    # the studio profile per frame, k = 1..k_top (dB re the 2nd), and re the 2nd-4th RMS
    tgt_db = np.full((k_top, len(t_g)), -200.0)
    for j in np.nonzero(pitched)[0]:
        t10 = bk.target_profile(p[j], prof, 10)
        kk = np.arange(1, k_top + 1)
        tgt_db[:, j] = np.concatenate([t10, t10[-1] - cfg["slope_db"] * np.log2(kk[10:] / 10.0)])
    T = 10 ** (tgt_db / 20) * ref  # each harmonic's target amplitude (frame rate)

    # time since the note started (an onset, or a new pitch) for the top's decay
    start = np.zeros(len(t_g), bool)
    start[np.clip(np.searchsorted(t_g, onsets), 0, len(t_g) - 1)] = True
    lp_ = np.log(np.where(pitched, p, 1.0))
    start[1:] |= (np.abs(np.diff(lp_)) > 0.5 * np.log(2) / 12) | (pitched[1:] & ~pitched[:-1])
    idx = np.where(start, np.arange(len(t_g)), 0)
    since = (np.arange(len(t_g)) - np.maximum.accumulate(idx)) * step / fs

    body = np.zeros(n)
    top = np.zeros(n)
    fund = np.zeros(n)
    # the fundamental: a clean tone, its phase locked to the 2nd harmonic's (halved), which is held through
    # the moments the 2nd harmonic is too weak to give one (its phase jumped there: clicks) and smoothed (5 Hz)
    c2 = track(2)
    a2f = np.abs(c2)[::step][:len(t_g)]
    th = np.unwrap(np.angle(c2[::step][:len(t_g)]))
    good = pitched & (a2f > 0.1 * np.median(a2f[pitched])) if pitched.any() else pitched
    th = np.interp(np.arange(len(t_g)), np.nonzero(good)[0], th[good]) if good.any() else np.zeros(len(t_g))
    th = ndimage.uniform_filter1d(th, 20)  # 0.2 s: a slowly turning offset, as a held note's phase is
    # its target: the studio level less fund_margin_db, never over fund_max_db above the 2nd harmonic
    w1 = np.minimum(T[0] * 10 ** (-cfg["fund_margin_db"] / 20), ref * 10 ** (cfg["fund_max_db"] / 20))
    fund = w_note * np.interp(t_n, t_g, np.nan_to_num(w1)) * np.cos(phi + np.interp(t_n, t_g, th) / 2)
    # the body: harmonics 2.. up to meas_hz as the stem has them, rebalanced; crossfading into the top
    g_lim = 10 ** (cfg["rebal_db"] / 20)
    rng = np.random.default_rng(0)
    p_frames = np.where(pitched, p, np.nan)
    for k in range(2, k_top + 1):
        fk = k * p_frames
        if not np.any(fk < cfg["top_hz"]):
            break
        meas_w = np.nan_to_num(np.clip((cfg["meas_hz"] + cfg["xfade_hz"] / 2 - fk) / cfg["xfade_hz"], 0, 1))
        on_k = np.nan_to_num((fk < cfg["top_hz"]).astype(float))
        if np.any(meas_w > 0):
            ck = c2 if k == 2 else track(k)
            ak = np.abs(ck)[::step][:len(t_g)]  # rebalanced on its own tracked amplitude
            gk = np.interp(t_n, t_g, np.nan_to_num(np.clip(T[k - 1] / (ak + 1e-12), 10 ** (-cfg["rebal_cut_db"] / 20), g_lim)))
            body += np.interp(t_n, t_g, meas_w) * gk * w_note * np.real(ck * np.exp(1j * k * phi))
            del ck
        syn_w = (1 - meas_w) * on_k
        if np.any(syn_w > 0):
            # bright at the note's start (a 5 ms rise: switched on at once it clicked), then mellowing
            decay = np.exp(-since * np.nan_to_num(fk) / 1000 * cfg["decay_per_khz"]) * np.clip(since / 0.005, 0, 1)
            a = np.interp(t_n, t_g, np.nan_to_num(T[k - 1]) * decay * syn_w)
            top += w_note * a * np.cos(k * np.sqrt(1 + cfg["stretch"] * k * k) * phi + rng.uniform(0, 2 * np.pi))
    del c2
    # the top's level: 1-3 kHz at top_re_studio_db from a studio bass's long-term tone (both re 100 Hz)
    studio = json.loads(STUDIO.read_text())
    tone = np.array(studio["instruments"]["bass"]["tone"])
    grid = np.array(studio["grid_hz"])
    ref_db = lambda fhz: float(np.interp(np.log(fhz), np.log(grid), tone))  # noqa: E731
    harm = fund + body
    # (the tone is per third-octave: 1-3 kHz spans 1.58 octaves, 80-125 Hz 0.64)
    want_hi = (10 * np.log10(_band_level(harm, fs, 80, 125)) + (ref_db(1732) - ref_db(100))
               + 10 * np.log10(np.log2(3) / np.log2(125 / 80)) + cfg["top_re_studio_db"])
    have_hi = 10 * np.log10(_band_level(top + body, fs, 1000, 3000))
    top_gain_db = float(np.clip(want_hi - have_hi, -24, 24))
    top *= 10 ** (top_gain_db / 20)
    fill = (1 - w_note) * bparts["rebalanced"] * 10 ** (cfg["fill_db"] / 20)
    out = fund + body + top + bparts["attacks"] + fill
    stats = {"pitched_pct": round(100 * float(pitched.mean()), 0), "harmonics": k_top, "top_gain_db": round(top_gain_db, 1),
             "layers_db_re_out": {k: round(10 * np.log10(np.mean(v ** 2) / np.mean(out ** 2) + 1e-30), 1)
                                  for k, v in (("fundamental", fund), ("body", body), ("top", top),
                                               ("attacks", bparts["attacks"]), ("fill", fill))}}
    if parts is not None:
        parts.update({"input": mono, "fundamental": fund, "body": body, "top": top, "attacks": bparts["attacks"],
                      "fill": fill, "resynth": out, "bank_rebalanced": bparts["rebalanced"] + bparts["attacks"]})
    return np.column_stack([out, out]), stats
