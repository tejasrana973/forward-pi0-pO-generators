#!/bin/bash
# Merge out4/m4_*_s<seed>.npz into merged4/<model>_<target>.npz  (run after the jobs are done)
mkdir -p merged4
for tag in $(ls out4/m4_*.npz | sed -E 's/_s[0-9]+\.npz$//' | sort -u); do
  name=$(basename $tag | sed 's/^m4_//')
  python3 m4_merge.py "${tag}_s*.npz" -o merged4/${name}.npz
done
