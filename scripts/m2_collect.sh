#!/bin/bash
# Merge all out/m2_*_s<seed>.npz files into merged/<model>_<system>.npz  (run after the jobs are done)
mkdir -p merged
for tag in $(ls out/m2_*.npz | sed -E 's/_s[0-9]+\.npz$//' | sort -u); do
  name=$(basename $tag | sed 's/^m2_//')
  python3 m2_merge.py "${tag}_s*.npz" -o merged/${name}.npz
done
