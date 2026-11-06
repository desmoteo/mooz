#!/bin/sh
# Bass stems of the Stanza OK recording with the dockerized audio-separator (beveradb/audio-separator:gpu),
# one model at a time, only the bass, at 48 kHz (the pipeline's rate). Models are cached in the pipeline's
# own model folder; files are written as the calling user into prove/bass/.
#   sh prove/run_bass.sh > processed/logs/prove_bass.log 2>&1
cd "$(dirname "$0")" || exit 1
mkdir -p bass
IN="260924_230957_TrLR Stanza OK.WAV"
for model in BS-Roformer-SW.ckpt htdemucs_ft.yaml hdemucs_mmi.yaml kuielab_a_bass.onnx; do
  echo "== $model $(date +%T)"
  docker run --rm --gpus all --memory=8g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$PWD:/workdir" -v "$HOME/.cache/rehearsal-master/models:/models" -w /workdir \
    beveradb/audio-separator:gpu \
    -m "$model" --model_file_dir /models --output_dir bass --single_stem Bass \
    --sample_rate 48000 --output_format FLAC "$IN" 2>&1 \
    | grep -v -E "onnxruntime|libcudart|onnxruntime.ai|it/s\]|%\|" | grep -E "INFO - (cli|separator) - (Separation|Loading|Model)|ERROR|Error|rror:"
  echo "== $model exit $?"
done
ls -la bass
echo "prove_bass EXIT"
