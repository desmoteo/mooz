"""The prototypes on a band song's raw bass stem (60 s): pitch with/without octave correction,
clean-up, fundamental restored towards the 2nd harmonic. Listening files and measurements."""
import json, sys
from pathlib import Path
import numpy as np, soundfile as sf, librosa
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parent))
import proto
song = sys.argv[1] if len(sys.argv) > 1 else "260928_224856_TrLR Miramare"
OUT = Path("processed/experiments/bass_proto") / song; OUT.mkdir(parents=True, exist_ok=True)
fs = 48000; A, B = 60, 120
rep = json.load(open(f"processed/ensemble/reports/{song}.json")); t0 = rep["trim"]["start_s"]
x = sf.read(Path("processed/stems_ensemble") / song / "bass.wav", dtype="float64", always_2d=True,
            start=int((A + t0) * fs), stop=int((B + t0) * fs))[0]
def notes(f):
    ok = np.isfinite(f); u, c = np.unique(librosa.hz_to_note(f[ok]), return_counts=True)
    return f"voiced {ok.mean()*100:.0f}%, median {np.nanmedian(f):.0f} Hz, " + ", ".join(f"{a} {b*100/ok.sum():.0f}%" for a, b in sorted(zip(u, c), key=lambda z: -z[1])[:5])
f_raw, tf = proto.pitch(x, fs, octave_db=99.0)
f_cor, _ = proto.pitch(x, fs)
print("pitch, uncorrected:", notes(f_raw)); print("pitch, corrected:  ", notes(f_cor))
a, m = proto.harmonic_amps(x, fs, f_cor, tf)
low = m & (np.exp(np.interp(np.arange(len(x)) / fs, tf[np.isfinite(f_cor)], np.log(f_cor[np.isfinite(f_cor)]))) < 70)
for name, sel in (("all notes", m), ("notes below 70 Hz", low)):
    use = sel & (a[2] > np.percentile(a[2][m], 30))
    print(f"fundamental re 2nd harmonic, {name}: median {np.median(20*np.log10((a[1][use]+1e-9)/(a[2][use]+1e-9))):+.1f} dB")
cl = proto.clean(x, fs, f_cor, tf)
rem = x - cl
print(f"clean-up removed {10*np.log10((rem**2).sum()/(x**2).sum()):.1f} dB of the stem's energy; per band:")
for lo, hi in ((150, 600), (600, 2000), (2000, 5000)):
    sos = signal.butter(4, [lo, hi], "band", fs=fs, output="sos")
    e = lambda y: (signal.sosfilt(sos, y.mean(1)) ** 2).sum()
    print(f"   {lo}-{hi} Hz: {10*np.log10(e(cl)/e(x)):+.1f} dB")
sf.write(OUT / "1_raw.flac", (x * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
sf.write(OUT / "2_cleaned.flac", (cl * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
sf.write(OUT / "2_removed.flac", (rem * 10 ** (-12 / 20)).astype(np.float32), fs, subtype="PCM_24")
np.save(OUT / "pitch.npy", np.stack([tf, f_cor]))
print("written", OUT)
