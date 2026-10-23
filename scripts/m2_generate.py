#!/usr/bin/env python3
"""
M2 producer: ONE process = ONE model, ONE system, ONE seed.
Fills histograms of pi0 (stable) and saves them to a small .npz file (no event storage).

Systems (energies in GeV)
  --frame lab --energy 1000      1 TeV proton on a proton at rest (generated in the lab frame)  -> Fig. 2
  --frame cm  --energy 43.3394149463   same collision, generated in the CM frame, then boosted -> Fig. 3
  --frame cm  --energy 13600     13.6 TeV pp, generated in the CM frame, then boosted           -> Fig. 7
Boost choice for --frame cm (the paper's trick, event by event):
  --boost max    boost in the +z or -z direction, whichever gives the largest pi0 lab energy in the event
  --boost plus   always boost along +z (must reproduce the lab-frame spectrum: frame check)

What is stored (all for stable pi0 only):
  h2[xi_bin, |eta_cm|_bin]   pi0 counts, xi = E_lab / E_beam_lab in log10 bins, |eta| in CM frame
  evt_gt[thr]                events with at least one pi0 above xi = thr
  pi0_gt[thr], pi0_gt_win[thr]   pi0 above thr, and those inside the analytic eta window
  pi0_gt_below/above         pi0 above thr with |eta| below eta_lo / above eta_hi (leakage out of the window)
  max_pt_gt07                largest pT of any pi0 with xi > 0.7 (single-event outlier statistic, do not quote)
  meta                       model, energies, cross sections from the model, seed, timing

Run (small test):
  python3 m2_generate.py --model Sibyll23d --frame lab --energy 1000 --nevents 20000 --seed 1
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
LXI_EDGES = np.linspace(-9.0, 0.0, 226)           # log10(xi), 25 bins per decade (13.6 TeV reaches ~1e-9)
CHUNK = 5000                                      # events per histogram flush

# Pythia forward tune: Monash-based version, Table II (Appendix A) of Fieg, Kling, Schulz, Sjostrand,
# arXiv:2309.08604.  The main QCDCR-based tune needs the Christiansen-Skands baseline settings, not set here.
PYTHIA_FWD = [
    "SoftQCD:inelastic = on",
    "BeamRemnants:dampPopcorn = 0",
    "BeamRemnants:hardRemnantBaryon = on",
    "BeamRemnants:aRemnantBaryon = 0.68",
    "BeamRemnants:bRemnantBaryon = 1.22",
    "BeamRemnants:primordialKTsoft = 0.56",
    "BeamRemnants:primordialKThard = 1.8",
    "BeamRemnants:halfScaleForKT = 10",
    "BeamRemnants:halfMassForKT = 1",
    "BeamRemnants:primordialKTremnant = 0.56",
]


def eta_window(rs, xi_cut=0.7, pt_max=1.5):
    lo = np.log(xi_cut * rs / np.sqrt(M_PI0**2 + pt_max**2))
    hi = np.log(rs / M_PI0)
    return lo, hi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="chromo model name, e.g. Sibyll23d, EposLHC, QGSJetII04, Pythia8")
    ap.add_argument("--tune", default="default", choices=["default", "fwd"], help="Pythia8 only")
    ap.add_argument("--frame", required=True, choices=["lab", "cm"])
    ap.add_argument("--energy", required=True, type=float, help="lab: beam energy [GeV]; cm: sqrt(s) [GeV]")
    ap.add_argument("--boost", default="max", choices=["max", "plus", "none"], help="only for --frame cm (use none with --frame lab)")
    ap.add_argument("--forward-only", action="store_true",
                    help="keep only pi0 flying forward in the CM frame (along the beam direction chosen for the boost); "
                         "this gives the sharp low-xi edge at m_pi/sqrt(s) described in the paper")
    ap.add_argument("--nevents", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max-seconds", type=float, default=1e9, help="stop early (and still save) after this time")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    import chromo
    import chromo.models as M
    from chromo.constants import GeV
    from chromo.kinematics import CenterOfMass, FixedTarget

    if args.frame == "lab":
        kin = FixedTarget(args.energy * GeV, "p", "p")
        rs = float(kin.ecm)
    else:
        kin = CenterOfMass(args.energy * GeV, "p", "p")
        rs = args.energy
    gamma = rs / (2.0 * M_P)                          # Lorentz factor CM -> lab
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    e_beam_lab = (rs**2 - 2.0 * M_P**2) / (2.0 * M_P)  # total energy of the beam proton in the lab frame
    eta_lo, eta_hi = eta_window(rs)
    eta_edges = np.unique(np.concatenate([np.arange(0.0, 16.01, 0.25), [eta_lo, eta_hi]]))

    label = args.model + ("-fwd" if (args.model == "Pythia8" and args.tune == "fwd") else "")
    t0 = time.time()
    if args.model == "Pythia8" and args.tune == "fwd":
        gen = M.Pythia8(kin, seed=args.seed, config=PYTHIA_FWD)
    else:
        gen = getattr(M, args.model)(kin, seed=args.seed)
    gen.set_stable(111)                               # keep pi0 stable
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
    pi0_gt_below = np.zeros(len(THR), dtype=np.int64)   # above thr, but |eta| < eta_lo
    pi0_gt_above = np.zeros(len(THR), dtype=np.int64)   # above thr, but |eta| > eta_hi
    max_pt_gt07 = 0.0
    n_events = 0
    n_pi0 = 0

    buf_xi, buf_eta, buf_pt, buf_evmax = [], [], [], []

    def flush():
        nonlocal h2, evt_gt, pi0_gt, pi0_gt_win, pi0_gt_below, pi0_gt_above, max_pt_gt07
        if not buf_xi:
            return
        xi = np.concatenate(buf_xi); eta = np.concatenate(buf_eta); pt = np.concatenate(buf_pt)
        evmax = np.array(buf_evmax)
        lxi = np.clip(np.log10(xi), LXI_EDGES[0], -1e-12)
        h, _, _ = np.histogram2d(lxi, np.clip(eta, 0, eta_edges[-1] - 1e-9), bins=[LXI_EDGES, eta_edges])
        h2 += h.astype(np.int64)
        win = (eta >= eta_lo) & (eta <= eta_hi)
        for k, thr in enumerate(THR):
            evt_gt[k] += int((evmax > thr).sum())
            above = xi > thr
            pi0_gt[k] += int(above.sum())
            pi0_gt_win[k] += int((above & win).sum())
            pi0_gt_below[k] += int((above & (eta < eta_lo)).sum())
            pi0_gt_above[k] += int((above & (eta > eta_hi)).sum())
        m = xi > 0.7
        if m.any():
            max_pt_gt07 = max(max_pt_gt07, float(pt[m].max()))
        buf_xi.clear(); buf_eta.clear(); buf_pt.clear(); buf_evmax.clear()

    t1 = time.time()
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
                pz_cm = gamma * (pz - beta * en)      # boost back to the CM frame (only for eta)
                fwd = pz_cm > 0                       # flying forward = along the beam direction
            else:
                pz_cm = pz
                ep = gamma * (en + beta * pz)         # lab energy if the target is the -z proton
                if args.boost == "plus":
                    e_lab, fwd = ep, pz > 0
                else:
                    em = gamma * (en - beta * pz)     # lab energy for the opposite boost
                    if ep.max() >= em.max():
                        e_lab, fwd = ep, pz > 0
                    else:
                        e_lab, fwd = em, pz < 0
            if args.forward_only:                     # drop pi0 of the backward CM hemisphere
                en, pt, e_lab, pz_cm = en[fwd], pt[fwd], e_lab[fwd], pz_cm[fwd]
            if len(en) > 0:
                xi = e_lab / e_beam_lab
                eta = np.abs(np.arcsinh(pz_cm / np.maximum(pt, 1e-9)))
                buf_xi.append(xi); buf_eta.append(eta); buf_pt.append(pt); buf_evmax.append(xi.max())
                n_pi0 += len(xi)
        if n_events % CHUNK == 0:
            flush()
            if time.time() - t1 > args.max_seconds:
                print("stopping early after %d events (time limit)" % n_events, file=sys.stderr)
                break
    flush()
    t_gen = time.time() - t1

    meta = dict(model=args.model, label=label, tune=args.tune, frame=args.frame, energy=args.energy,
                boost=args.boost if args.frame == "cm" else "none", forward_only=bool(args.forward_only), sqrt_s=rs, e_beam_lab=e_beam_lab,
                eta_lo=eta_lo, eta_hi=eta_hi, seed=args.seed, n_events=n_events, n_pi0=n_pi0,
                xsec_mb=xs, t_init=t_init, t_gen=t_gen, chromo=chromo.__version__)
    os.makedirs(args.outdir, exist_ok=True)
    tag = "%s_%s%g_%s%s_s%d" % (label, args.frame, args.energy, meta["boost"],
                                "-fo" if args.forward_only else "", args.seed)
    path = os.path.join(args.outdir, "m2_%s.npz" % tag)
    np.savez_compressed(path, h2=h2, lxi_edges=LXI_EDGES, eta_edges=eta_edges, thr=THR,
                        evt_gt=evt_gt, pi0_gt=pi0_gt, pi0_gt_win=pi0_gt_win,
                        pi0_gt_below=pi0_gt_below, pi0_gt_above=pi0_gt_above,
                        max_pt_gt07=max_pt_gt07, meta=json.dumps(meta))
    print("RESULT saved %s | %d events, %d pi0, %.0f ev/s, init %.1f s" %
          (path, n_events, n_pi0, n_events / max(t_gen, 1e-9), t_init))


if __name__ == "__main__":
    main()