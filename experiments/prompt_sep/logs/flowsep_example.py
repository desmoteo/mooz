import sys, os, numpy as np, soundfile as sf, torch, torchaudio
sys.argv = ["x"]; sys.path.insert(0, ".")
import flowsep_sep as F
model, stft, frames = F.load()
x, fs = sf.read("Hive/examples/acoustic_guitar.wav", dtype="float32", always_2d=True)
x = torchaudio.functional.resample(torch.from_numpy(x.T.copy()), fs, 16000).numpy().T
x = np.repeat(x[:, :1], 2, 1)
y = F.separate(model, stft, frames, x, "acoustic guitar", F.Path(os.environ["TMPDIR"]) / "fs")
ref, rfs = sf.read("Hive/examples/separated_audio.wav", dtype="float32")
print("mix rms", np.sqrt((x**2).mean()), "out rms", np.sqrt((y**2).mean()), "shipped rms", np.sqrt((ref**2).mean()))
sf.write(os.environ["TMPDIR"] + "/example_out.wav", y[:, 0], 16000)
def spec(a): 
    S = np.abs(torch.stft(torch.from_numpy(np.ascontiguousarray(a, np.float32)), 1024, 256, return_complex=True).numpy()); return np.log(S + 1e-5)
n = min(len(y), len(ref)); a, b, m = spec(y[:n, 0]), spec(ref[:n]), spec(x[:n, 0])
c = lambda p, q: np.corrcoef(p.ravel(), q.ravel())[0, 1]
print("log-spec corr: ours vs shipped", c(a, b), " ours vs mix", c(a, m), " shipped vs mix", c(b, m))
