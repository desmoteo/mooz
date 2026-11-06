#!/bin/sh
# Every setup on the 60 s excerpt (40-100 s of Stanza OK), one after the other.
#   sh experiments/bass_sr/run_all.sh > processed/logs/bass_sr.log 2>&1
cd "$(dirname "$0")/../.." || exit 1
. experiments/bass_sr/env.sh
run() { systemd-run --user --scope --quiet -p MemoryMax=12G -p MemorySwapMax=0 \
  experiments/bass_sr/.venv-audiosr/bin/python -W ignore experiments/bass_sr/run_sr.py "$@" 2>&1 \
  | grep --line-buffered -v -E "it/s\]|%\||expandable_segments|DDIM Sampling"; }
run --rates 1200,2000,4000,8000,orig
run --rates 2000 --guidance 6.0 --tag _g6
run --source resynth --rates 8000,4000
echo "bass_sr EXIT"
