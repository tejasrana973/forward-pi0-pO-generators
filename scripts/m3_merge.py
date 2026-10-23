#!/usr/bin/env python3
"""
Merge m3_*.npz files (same model/target/system, different seeds).
Run:  python3 m3_merge.py "out3/m3_Sibyll23d_O16_lab1000_none-fo_s*.npz" -o merged3/Sibyll23d_O16_lab1000_none-fo.npz
"""
import argparse
import glob
import json
import os

import numpy as np

KEYS_SUM = ["h2", "evt_gt", "pi0_gt", "pi0_gt_win", "pi0_gt_below", "pi0_gt_above"]
KEYS_SAME = ["model", "label", "target", "frame", "energy", "boost", "forward_only"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    files = sorted(set(f for pat in args.files for f in glob.glob(pat)))
    if not files:
        raise SystemExit("no input files found")
    tot, metas, first, edges = None, [], None, None
    for f in files:
        d = np.load(f, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        if tot is None:
            tot = {k: d[k].copy() for k in KEYS_SUM}
            edges = {k: d[k] for k in ["lxi_edges", "eta_edges", "thr"]}
            first = meta
        else:
            for k in KEYS_SAME:
                if meta[k] != first[k]:
                    raise SystemExit("mismatch in %s between %s and the first file" % (k, f))
            for k in KEYS_SUM:
                tot[k] += d[k]
        metas.append(meta)
    xs = {}
    for key in ["inelastic", "total", "non_diffractive"]:
        vals = [m["xsec_mb"][key] for m in metas if key in m.get("xsec_mb", {})]
        if vals:
            xs[key] = float(np.mean(vals))
    meta = dict(first)
    meta.update(n_events=sum(m["n_events"] for m in metas), n_pi0=sum(m["n_pi0"] for m in metas),
                n_files=len(files), xsec_mb=xs, seeds=[m["seed"] for m in metas])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, **tot, **edges, meta=json.dumps(meta))
    print("merged %d files -> %s | %d events, %d pi0" % (len(files), args.output, meta["n_events"], meta["n_pi0"]))


if __name__ == "__main__":
    main()
