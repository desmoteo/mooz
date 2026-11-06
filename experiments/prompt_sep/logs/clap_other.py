import sys, glob, numpy as np, soundfile as sf, torch
sys.path.insert(0, "."); from sam_audio.ranking.clap import get_model
from common import OUT, LEVEL
LABELS = ["harmonica", "tambourine", "electric guitar", "singing", "drums", "bass guitar", "silence"]
m = get_model(); torch.set_num_threads(4)
with torch.no_grad(): t = m.get_text_embedding([f"the sound of {x}" for x in LABELS], use_tensor=True)
for f in sorted(glob.glob(str(OUT / "*" / "260928_225412_TrLR Abitudine Strato@190-230s" / "1_stems" / "other.flac"))):
    x = (sf.read(f)[0] / LEVEL).mean(1).astype(np.float32)
    segs = np.stack([x[i:i + 480000] for i in range(0, len(x) - 479999, 240000)])
    with torch.no_grad(): p = torch.softmax(100 * m.get_audio_embedding_from_data(x=torch.from_numpy(segs), use_tensor=True) @ t.T, -1).numpy()
    print(f"{f.split('/')[-4]:<22} harmonica " + " ".join(f"{v:.2f}" for v in p[:, 0]) + f"   top: {[LABELS[i] for i in p.mean(0).argsort()[::-1][:2]]}")
