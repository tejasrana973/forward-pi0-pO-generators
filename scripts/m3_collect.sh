#!/bin/bash
# Merge out3/m3_*_s<seed>.npz into merged3/<model>_<target>_<system>.npz  (run after the jobs are done)
mkdir -p merged3
for tag in $(ls out3/m3_*.npz | sed -E 's/_s[0-9]+\.npz$//' | sort -u); do
  name=$(basename $tag | sed 's/^m3_//')
  python3 m3_merge.py "${tag}_s*.npz" -o merged3/${name}.npz
done
