#!/usr/bin/env python3
"""Merge m4_*.npz files (same model/target, different seeds).
Run: python3 m4_merge.py "out4/m4_Sibyll23e_O16_s*.npz" -o merged4/Sibyll23e_O16.npz"""
import argparse
import glob
import json
import os

import numpy as np

SUM = ["h_eta", "h_eta_nolep", "h_pt", "h_invpt", "h_nch", "h_sumpt"]
CNT = ["n_gen", "n_sel", "n_part", "n_lep"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args()
    files = sorted(set(f for p in a.files for f in glob.glob(p)))
    if not files:
        raise SystemExit("no input files found")
    tot, first, edges, metas = None, None, None, []
    for f in files:
        d = np.load(f, allow_pickle=False)
        m = json.loads(str(d["meta"]))
        if tot is None:
            tot = {k: d[k].copy() for k in SUM}
            edges = {k: d[k] for k in ["eta_edges", "pt_edges"]}
            first = m
        else:
            for k in ["model", "target"]:
                if m[k] != first[k]:
                    raise SystemExit("mismatch in %s for %s" % (k, f))
            for k in SUM:
                tot[k] += d[k]
        metas.append(m)
    meta = dict(first)
    for k in CNT:
        meta[k] = int(sum(m[k] for m in metas))
    meta.update(n_files=len(files), seeds=[m["seed"] for m in metas])
    os.makedirs(os.path.dirname(a.output) or ".", exist_ok=True)
    np.savez_compressed(a.output, **tot, **edges, meta=json.dumps(meta))
    print("merged %d files -> %s | %d events, acceptance %.4f" % (len(files), a.output, meta["n_gen"], meta["n_sel"] / meta["n_gen"]))


if __name__ == "__main__":
    main()
