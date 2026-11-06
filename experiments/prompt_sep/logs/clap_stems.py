import sys, numpy as np, soundfile as sf, torch
sys.path.insert(0, "."); from sam_audio.ranking.clap import get_model
from common import ROOT, LEVEL
LABELS = ["harmonica", "tambourine", "electric guitar", "singing", "drums", "bass guitar", "silence"]
song, a = sys.argv[1], int(sys.argv[2])
m = get_model(); torch.set_num_threads(4)
with torch.no_grad(): t = m.get_text_embedding([f"the sound of {x}" for x in LABELS], use_tensor=True)
def score(x):
    x = x.mean(1).astype(np.float32); segs = np.stack([x[i:i + 480000] for i in range(0, len(x) - 479999, 240000)])
    with torch.no_grad(): e = m.get_audio_embedding_from_data(x=torch.from_numpy(segs), use_tensor=True)
    return torch.softmax(100 * e @ t.T, -1).numpy()
rows = {}
for nm in ["vocals", "vocal_room", "guitar", "bass", "other", "cymbals", "hihat"]:
    rows[f"rof {nm}"] = sf.read(ROOT / f"processed/stems/{song}/{nm}.wav", start=a * 48000, frames=40 * 48000)[0]
d = ROOT / f"processed/experiments/prompt_sep/sam-large/{song}@{a}-{a + 40}s/1_stems"
for nm in ["vocals", "guitars", "bass", "other", "drums"]:
    rows[f"sam {nm}"] = sf.read(d / f"{nm}.flac")[0] / LEVEL
print(f"{'harmonica prob per 10 s window':<16}")
for k, x in rows.items():
    p = score(x); top = [LABELS[i] for i in p.mean(0).argsort()[::-1][:2]]
    print(f"  {k:<15} " + " ".join(f"{v:.2f}" for v in p[:, 0]) + f"   top: {top}")
