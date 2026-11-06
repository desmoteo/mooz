import sys, glob, yaml, torch, os
sys.path.insert(0, "Hive/models/flowsep")
from latent_diffusion.util import instantiate_from_config
snap = glob.glob("hf_cache/models--AlayaLab--FlowSep-hive/snapshots/*")[0]
cfg = yaml.load(open(f"{snap}/config.yaml"), Loader=yaml.FullLoader)
os.chdir(snap)  # vae.ckpt is referenced relative to the working directory
m = instantiate_from_config(cfg["model"])
sd = torch.load("flowsep_hive.ckpt", map_location="cpu", weights_only=False, mmap=True)["state_dict"]
print(m.load_state_dict(sd, strict=True))
import collections; c = collections.Counter()
for k, v in sd.items(): c[".".join(k.split(".")[:2])] += v.numel()
for k, n in c.most_common(8): print(f"{k:40s} {n/1e6:8.1f} M")
