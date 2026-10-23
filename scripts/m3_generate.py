#!/usr/bin/env python3
"""
M3 producer (p + nucleus): ONE process = ONE model, ONE target, ONE system, ONE seed.
Same idea as m2_generate.py, extended to nuclear targets. Output: small .npz histogram file.

Conventions (checked with chromo 0.11.0):
  * Projectile = proton, moving along +z.  Energies are per nucleon pair (sqrt(s_NN)).
  * --frame lab --energy 1000       proton of 1 TeV on a nucleus at rest (generated in the lab frame) -> Fig. 6
  * --frame cm  --energy 9620       sqrt(s_NN) = 9.62 TeV, generated in the nucleon-nucleon CM frame
  * "lab" frame of the CR collision = frame where a target nucleon is at rest, boost along +z (the
    projectile direction is known for p+A, so NO max-energy boost is needed; use --boost plus).
  * xi = E_pi0(lab) / E_beam(lab).  eta = pseudorapidity in the nucleon-nucleon CM frame.
  * --forward-only keeps only pi0 with pz* > 0 (proton direction in the CM frame), as in the paper.
  * --target p gives the pp reference with the same code (also allows --boost max as in M2).

Run (small test):
  python3 m3_generate.py --model Sibyll23d --target O16 --frame lab --energy 1000 --boost none \
          --forward-only --nevents 20000 --seed 1 --outdir test_out > t3.log 2>&1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

M_P = 0.938272      # GeV
M_PI0 = 0.1349768   # GeV
THR = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
LXI_EDGES = np.linspace(-9.0, 0.0, 226)           # log10(xi), 25 bins per decade
CHUNK = 2000
TARGETS = {"p": "p", "O16": (16, 8), "N14": (14, 7)}


def eta_window(rs, xi_cut=0.7, pt_max=1.5):
    return np.log(xi_cut * rs / np.sqrt(M_PI0**2 + pt_max**2)), np.log(rs / M_PI0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="O16", choices=list(TARGETS))
    ap.add_argument("--frame", required=True, choices=["lab", "cm"])
    ap.add_argument("--energy", required=True, type=float, help="lab: proton energy [GeV]; cm: sqrt(s_NN) [GeV]")
    ap.add_argument("--boost", default="plus", choices=["plus", "max", "none"],
                    help="cm only: plus (proton direction), max (pp only); use none with --frame lab")
    ap.add_argument("--forward-only", action="store_true")
    ap.add_argument("--nevents", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--checkpoint-seconds", type=float, default=0,
                    help="save the histograms every N seconds (overwrites the same file), so a process that hangs "
                         "or is killed loses at most N seconds of work; 0 = only save at the end")
    ap.add_argument("--outdir", default="out3")
    args = ap.parse_args()
    if args.boost == "max" and args.target != "p":
        raise SystemExit("--boost max only makes sense for pp (projectile direction is known for p+A)")

    import chromo
    import chromo.models as M
    from chromo.constants import GeV
    from chromo.kinematics import CenterOfMass, FixedTarget

    tgt = TARGETS[args.target]
    if args.frame == "lab":
        kin = FixedTarget(args.energy * GeV, "p", tgt)
        rs = float(np.sqrt(2.0 * M_P * (args.energy + M_P)))      # sqrt(s_NN)
        e_beam_lab = float(args.energy)
    else:
        kin = CenterOfMass(args.energy * GeV, "p", tgt)
        rs = args.energy
        e_beam_lab = (rs**2 - 2.0 * M_P**2) / (2.0 * M_P)
    gamma = rs / (2.0 * M_P)
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    eta_lo, eta_hi = eta_window(rs)
    eta_edges = np.unique(np.concatenate([np.arange(0.0, 16.01, 0.25), [eta_lo, eta_hi]]))

    t0 = time.time()
    gen = getattr(M, args.model)(kin, seed=args.seed)
    gen.set_stable(111)
    try:
        cs = gen.cross_section()
        xs = dict(inelastic=float(cs.inelastic), total=float(cs.total), non_diffractive=float(cs.non_diffractive))
    except Exception:
        xs = {}
    t_init = time.time() - t0

    h2 = np.zeros((len(LXI_EDGES) - 1, len(eta_edges) - 1), dtype=np.int64)
    evt_gt = np.zeros(len(THR), dtype=np.int64)
    pi0_gt = np.zeros(len(THR), dtype=np.int64)
    pi0_gt_win = np.zeros(len(THR), dtype=np.int64)
    pi0_gt_below = np.zeros(len(THR), dtype=np.int64)
    pi0_gt_above = np.zeros(len(THR), dtype=np.int64)
    n_events = 0
    n_pi0 = 0
    buf_xi, buf_eta, buf_evmax = [], [], []

    def flush():
        if not buf_xi:
            return
        xi = np.concatenate(buf_xi); eta = np.concatenate(buf_eta); evmax = np.array(buf_evmax)
        lxi = np.clip(np.log10(xi), LXI_EDGES[0], -1e-12)
        h, _, _ = np.histogram2d(lxi, np.clip(eta, 0, eta_edges[-1] - 1e-9), bins=[LXI_EDGES, eta_edges])
        h2[...] += h.astype(np.int64)
        win = (eta >= eta_lo) & (eta <= eta_hi)
        for k, thr in enumerate(THR):
            evt_gt[k] += int((evmax > thr).sum())
            above = xi > thr
            pi0_gt[k] += int(above.sum())
            pi0_gt_win[k] += int((above & win).sum())
            pi0_gt_below[k] += int((above & (eta < eta_lo)).sum())
            pi0_gt_above[k] += int((above & (eta > eta_hi)).sum())
        buf_xi.clear(); buf_eta.clear(); buf_evmax.clear()

    os.makedirs(args.outdir, exist_ok=True)
    boost_tag = args.boost if args.frame == "cm" else "none"
    tag = "%s_%s_%s%g_%s%s_s%d" % (args.model, args.target, args.frame, args.energy, boost_tag,
                                   "-fo" if args.forward_only else "", args.seed)
    path = os.path.join(args.outdir, "m3_%s.npz" % tag)
    t1 = time.time()

    def write_out():
        """Write the current histograms atomically (temp file, then rename)."""
        t_gen = time.time() - t1
        meta = dict(model=args.model, label=args.model, target=args.target, frame=args.frame, energy=args.energy,
                    boost=boost_tag, forward_only=bool(args.forward_only),
                    sqrt_s=rs, e_beam_lab=e_beam_lab, eta_lo=eta_lo, eta_hi=eta_hi, seed=args.seed,
                    n_events=n_events, n_pi0=n_pi0, xsec_mb=xs, t_init=t_init, t_gen=t_gen,
                    chromo=chromo.__version__)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, h2=h2, lxi_edges=LXI_EDGES, eta_edges=eta_edges, thr=THR, evt_gt=evt_gt,
                                pi0_gt=pi0_gt, pi0_gt_win=pi0_gt_win, pi0_gt_below=pi0_gt_below,
                                pi0_gt_above=pi0_gt_above, meta=json.dumps(meta))
        os.replace(tmp, path)
        return t_gen

    t_ck = time.time()
    for ev in gen(args.nevents):
        n_events += 1
        fs = ev.final_state()
        sel = fs.pid == 111
        if sel.any():
            en = fs.en[sel].astype(np.float64)
            pz = fs.pz[sel].astype(np.float64)
            pt = fs.pt[sel].astype(np.float64)
            if args.frame == "lab":
                e_lab = en
                pz_cm = gamma * (pz - beta * en)          # back to the nucleon-nucleon CM frame
                fwd = pz_cm > 0
            else:
                pz_cm = pz
                ep = gamma * (en + beta * pz)             # boost along +z (target nucleon at rest)
                if args.boost == "max":
                    em = gamma * (en - beta * pz)
                    if ep.max() >= em.max():
                        e_lab, fwd = ep, pz > 0
                    else:
                        e_lab, fwd = em, pz < 0
                else:
                    e_lab, fwd = ep, pz > 0
            if args.forward_only:
                en, pt, e_lab, pz_cm = en[fwd], pt[fwd], e_lab[fwd], pz_cm[fwd]
            if len(en) > 0:
                xi = e_lab / e_beam_lab
                eta = np.abs(np.arcsinh(pz_cm / np.maximum(pt, 1e-9)))
                buf_xi.append(xi); buf_eta.append(eta); buf_evmax.append(xi.max())
                n_pi0 += len(xi)
        if n_events % CHUNK == 0:
            flush()
        if n_events % 50 == 0 and args.checkpoint_seconds > 0 and time.time() - t_ck > args.checkpoint_seconds:
            flush(); write_out(); t_ck = time.time()
        if n_events % 50 == 0 and time.time() - t1 > args.max_seconds:   # check often: slow models do ~5 events/s
            print("stopping early after %d events (time limit)" % n_events, file=sys.stderr)
            break
    flush()
    t_gen = write_out()
    print("RESULT saved %s | %d events, %d pi0, %.1f ev/s, init %.1f s" %
          (path, n_events, n_pi0, n_events / max(t_gen, 1e-9), t_init))


if __name__ == "__main__":
    main()