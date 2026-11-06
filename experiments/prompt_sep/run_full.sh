#!/bin/sh
# Full songs: SAM Audio large, stage 1 (vocals, bass, guitars, drums), then DrumSep on its drums.
cd "$(dirname "$0")" && . ./env.sh
for song in inputs/*.wav; do
  systemd-run --user --scope --quiet -p MemoryMax=11G -p MemorySwapMax=0 \
    .venv-sam/bin/python sam_sep.py --model large --stage 1 --only vocals,bass,guitars,drums "$song" 2>&1 \
    | grep --line-buffered -v -i "warn\|pynvml\|Loading weights\|weight_norm\|WeightNorm"
  d="../../processed/experiments/prompt_sep/sam-large/$(basename "$song" .wav)"
  (cd ../.. && systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 \
    .venv/bin/python experiments/prompt_sep/drumsep_stage2.py "experiments/prompt_sep/$d" 2>&1 | grep --line-buffered -v "it/s")
done
echo "prompt_sep EXIT"
