"""Where do harmonica and tambourine play? LAION-CLAP zero-shot scores on 10 s windows
(every 5 s) of each prepared input, re a few band instruments. CPU only.

    .venv-sam/bin/python clap_scan.py
"""

import numpy as np
import soundfile as sf
import torch

from common import HERE
from sam_audio.ranking.clap import get_model

LABELS = ["harmonica", "tambourine", "electric guitar", "singing", "drums", "bass guitar"]
WIN, HOP = 480000, 240000


def main() -> None:
    torch.set_num_threads(4)
    model = get_model()
    with torch.no_grad():
        t = model.get_text_embedding([f"the sound of {x}" for x in LABELS], use_tensor=True)
    for song in sorted((HERE / "inputs").glob("*.wav")):
        x, fs = sf.read(song, dtype="float32")
        x = x.mean(1)
        starts = list(range(0, len(x) - WIN, HOP))
        segs = np.stack([x[s:s + WIN] for s in starts])
        with torch.no_grad():
            a = torch.cat([model.get_audio_embedding_from_data(x=torch.from_numpy(segs[i:i + 8]), use_tensor=True)
                           for i in range(0, len(segs), 8)])
        p = torch.softmax(100 * a @ t.T, -1).numpy()
        print(song.stem, flush=True)
        for k in ("harmonica", "tambourine"):
            j = LABELS.index(k)
            top = np.argsort(-p[:, j])[:4]
            print(f"  {k:<11} " + "  ".join(f"{starts[i] / fs:4.0f}s:{p[i, j]:.2f}" for i in top), flush=True)


if __name__ == "__main__":
    main()
