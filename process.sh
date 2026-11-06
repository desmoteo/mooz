#!/bin/bash

mkdir -p processed

for file in *.[wW][aA][vV]; do
  [ -e "$file" ] || continue 

  echo "---------------------------------------------------"
  echo "Analyzing: $file"

  # PASS 1: Measure average loudness (LUFS)
  metrics=$(ffmpeg -hide_banner -i "$file" \
    -af "highpass=f=85,loudnorm=print_format=json" \
    -f null - 2>&1)

  measured_I=$(echo "$metrics" | grep -oP '"input_i" *: *"\K[^"]+')
  calc_thresh=$(awk "BEGIN {print $measured_I + 3.0}")

  echo "Average Loudness: $measured_I LUFS"
  echo "Calculated Threshold: $calc_thresh dB"

  # PASS 2: Apply EQ, Compression, and Tape Saturation
  sox -V1 "$file" "processed/$file" \
      gain -6 \
      highpass 85 \
      equalizer 550 3q -6 \
      compand 0.1,1.0 6:-90,-90,${calc_thresh},${calc_thresh},0,-6 \
      overdrive 5 10 \
      norm -1

  echo "Done polishing: $file"

done

echo "All files processed!"
