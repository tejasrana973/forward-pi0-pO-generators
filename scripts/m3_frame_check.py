#!/usr/bin/env python3
"""
Frame check for p+O: lab-frame generation against CM-frame generation with the proton boost ('plus').
Both must give the same high-xi spectrum and event rates (statistical agreement).

Steps (run on KEKCC, about 3 minutes):
  for s in 1 2; do python3 m3_generate.py --model Sibyll23d --target O16 --frame lab --energy 1000 --boost none \
        --forward-only --nevents 70000 --seed $s --outdir fc3 > fc_lab_$s.log 2>&1; done
  for s in 3 4; do python3 m3_generate.py --model Sibyll23d --target O16 --frame cm --energy 43.3394149463 --boost plus \
        --forward-only --nevents 70000 --seed $s --outdir fc3 > fc_cm_$s.log 2>&1; done
  python3 m3_merge.py "fc3/m3_Sibyll23d_O16_lab1000_none-fo_s*.npz" -o fc3/lab.npz
  python3 m3_merge.py "fc3/m3_Sibyll23d_O16_cm43.3394_plus-fo_s*.npz" -o fc3/cm.npz
  python3 m3_frame_check.py fc3/lab.npz fc3/cm.npz > frame_check_po.log 2>&1
"""
import sys

import numpy as np

from m3_plot import load, spectrum


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    _, _, _, ha = spectrum(a)
    _, _, _, hb = spectrum(b)
    na, nb = a["meta"]["n_events"], b["meta"]["n_events"]
    ok = (ha > 20) & (hb > 20)
    chi2 = np.sum((ha[ok] - hb[ok] * na / nb) ** 2 / (ha[ok] + hb[ok] * (na / nb) ** 2))
    print("events: lab %d, cm %d" % (na, nb))
    print("spectrum: %d bins with > 20 entries, chi2/ndf = %.2f (about 1 expected)" % (ok.sum(), chi2 / ok.sum()))
    for k, t in enumerate(a["thr"]):
        if t in (0.1, 0.5, 0.7):
            pa, pb = a["evt_gt"][k] / na, b["evt_gt"][k] / nb
            print("P(max xi>%.1f): lab %.3e  cm %.3e  difference %.1f sigma" %
                  (t, pa, pb, (pa - pb) / np.hypot(np.sqrt(pa / na), np.sqrt(pb / nb))))


if __name__ == "__main__":
    main()
