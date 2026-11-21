# The tape plugin of the bus (plugins.py): the one call for the whole take against the take in pieces of time on
# threads, each piece after a run-in and with the plugin's delay taken out. Run from the project root:
#   uv run python experiments/speed/tape.py <a stereo recording> [start_s end_s]
import sys, time, os, numpy as np, soundfile as sf, pedalboard
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from mooz import config
c = config.load(Path("rehearsal.toml"))["saturation"]["chowtape"]
x, fs = sf.read(sys.argv[1], dtype="float64", always_2d=True)
if len(sys.argv) > 3: x = x[int(float(sys.argv[2]) * fs):int(float(sys.argv[3]) * fs)]
def load():  # (on the main thread: the plugin asks for it)
    p = pedalboard.load_plugin(os.path.expanduser(c["path"]))
    for k, v in {"input_gain": c["input_gain_db"], "output_gain": 0.0, "dry_wet": 100.0, "tape_on_off": True, "tape_drive": c["drive"], "tape_saturation": c["saturation"],
                 "tape_bias": c["bias"], "tape_speed": c["speed_ips"], "loss_on_off": c["loss"], "tone_on_off": False, "compression_on_off": False, "wow_flutter_on_off": False,
                 "degrade_on_off": False, "chew_on_off": False, "input_filters_on_off": False, "oversampling_factor_render": float(c["oversampling"]), "oversampling_render_like_real_time": False}.items(): setattr(p, k, v)
    return p
xs = np.ascontiguousarray((x * (10 ** (c["level_db"] / 20) / np.sqrt(np.mean(x * x)))).T, dtype=np.float32); n = xs.shape[1]
t0 = time.perf_counter(); both = load().process(xs, fs); t_both = time.perf_counter() - t0
head = load().process(np.ascontiguousarray(xs[:, : 20 * fs]), fs, reset=False)
lag = max(range(0, 400), key=lambda d: float(np.dot(head[0, d : d + 10 * fs], both[0, : 10 * fs])))
print(f"one call {t_both:.2f} s for {n / fs:.0f} s | a call without reset comes out {lag} samples late")
def piece(args):
    pi, a0, b0, run = args
    s0 = max(0, a0 - run); seg = np.ascontiguousarray(xs[:, s0:min(n, b0 + lag)])
    seg = np.pad(seg, ((0, 0), (0, (b0 - s0) + lag - seg.shape[1])))
    return pi.process(seg, fs, reset=False)[:, (a0 - s0) + lag : (b0 - s0) + lag]
for k, run_s in ((6, 0.5), (6, 2.0), (6, 10.0), (3, 2.0)):
    cuts = np.linspace(0, n, k + 1).astype(int); ps = [load() for _ in range(k)]
    with ThreadPoolExecutor(k) as ex:
        t0 = time.perf_counter(); parts = list(ex.map(piece, [(ps[i], cuts[i], cuts[i + 1], int(run_s * fs)) for i in range(k)])); t_p = time.perf_counter() - t0
    d = np.concatenate(parts, axis=1) - both
    print(f"{k} pieces, {run_s:g} s of run-in, {k} threads: {t_p:.2f} s ({t_both / t_p:.1f}x) | {10 * np.log10(np.mean(d ** 2) / np.mean(both ** 2) + 1e-30):.0f} dB re the signal, largest sample {20 * np.log10(np.abs(d).max() + 1e-30):.0f} dBFS")
