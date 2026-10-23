#!/usr/bin/env python3
"""
M4 producer: generator-level version of the ATLAS p+O charged-particle analysis
(arXiv:2604.05512, sqrt(s_NN) = 9.62 TeV).  ONE process = ONE model, ONE seed.

Particle-level definition used (from the paper):
  * primary charged particle = charged, lifetime > 300 ps, produced promptly or in decays of promptly produced
    particles with lifetime < 30 ps.  In chromo's final state, K0S, K0L, Lambda, Sigma, Xi, Omega stay undecayed,
    so their daughters are never counted (they are not primary, as in ATLAS); pi0, eta, rho, ... are decayed.
    Charged particles kept: |pid| in {211 pi, 321 K, 2212 p, 11 e, 13 mu}.
    Strange baryons Sigma+-, Xi-, Omega- are removed explicitly (paper: Rivet, at most 0.15 % effect).
  * fiducial region: pT > 0.5 GeV and |eta| < 2.5 (eta in the ATLAS LAB frame).
  * event selection: at least one primary charged particle in the fiducial region.
  * all distributions are per selected event.

Frames: chromo generates in the nucleon-nucleon CM frame with the proton along +z.
  ATLAS lab frame: oxygen (3.4 TeV/nucleon) along +z, proton (6.8 TeV) along -z.
  We flip z and boost with beta = tanh(y_cm), y_cm = -0.5*ln(6.8/3.4) = -0.3466 (the CM moves along the proton).
  Result: eta_lab > 0 is the oxygen-going side, eta_lab < 0 the proton-going side.

Stored (per file):
  eta histogram (100 bins, -2.5..2.5), also without e/mu (sensitivity check)
  pT histogram: counts and sum(1/pT) per bin (so 1/(2 pi pT) can be applied particle by particle)
  n_ch distribution (events per n_ch), sum of pT per n_ch
  n_gen (all generated inelastic events), n_sel (events passing the selection)

Run (small test):
  python3 m4_generate.py --model Sibyll23e --nevents 20000 --seed 1 --outdir test_out4 > t4.log 2>&1
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ETA_EDGES = np.linspace(-2.5, 2.5, 101)
PT_EDGES = np.concatenate([np.arange(0.5, 10.0, 0.05), np.arange(10.0, 50.0, 0.5), np.arange(50.0, 150.01, 2.5)])
NCH_MAX = 600
Y_CM = -0.5 * np.log(6.8 / 3.4)               # CM rapidity in the ATLAS lab frame
BETA = np.tanh(Y_CM)
GAMMA = 1.0 / np.sqrt(1.0 - BETA**2)
KEEP = np.array([211, 321, 2212, 11, 13])
LEPT = np.array([11, 13])
CHECKPOINT_STEP = 50


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--nevents", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--target", default="O16", choices=["O16", "N14"])
    ap.add_argument("--max-seconds", type=float, default=1e9)
    ap.add_argument("--checkpoint-seconds", type=float, default=0)
    ap.add_argument("--outdir", default="out4")
    args = ap.parse_args()

    import chromo
    import chromo.models as M
    from chromo.constants import GeV
    from chromo.kinematics import CenterOfMass

    tgt = (16, 8) if args.target == "O16" else (14, 7)
    kin = CenterOfMass(9620 * GeV, "p", tgt)
    t0 = time.time()
    gen = getattr(M, args.model)(kin, seed=args.seed)
    t_init = time.time() - t0

    n_eta = len(ETA_EDGES) - 1
    n_pt = len(PT_EDGES) - 1
    h_eta = np.zeros(n_eta); h_eta_nolep = np.zeros(n_eta)
    h_pt = np.zeros(n_pt); h_invpt = np.zeros(n_pt)
    h_nch = np.zeros(NCH_MAX + 1); h_sumpt = np.zeros(NCH_MAX + 1)
    cnt = dict(n_gen=0, n_sel=0, n_part=0, n_lep=0)

    os.makedirs(args.outdir, exist_ok=True)
    tag = "%s_%s_s%d" % (args.model, args.target, args.seed)
    path = os.path.join(args.outdir, "m4_%s.npz" % tag)
    t1 = time.time()

    def write_out():
        meta = dict(model=args.model, target=args.target, seed=args.seed, sqrt_s_NN=9620.0, y_cm=Y_CM,
                    t_init=t_init, t_gen=time.time() - t1, chromo=chromo.__version__, **cnt)
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, eta_edges=ETA_EDGES, pt_edges=PT_EDGES, h_eta=h_eta, h_eta_nolep=h_eta_nolep,
                                h_pt=h_pt, h_invpt=h_invpt, h_nch=h_nch, h_sumpt=h_sumpt, meta=json.dumps(meta))
        os.replace(tmp, path)

    t_ck = time.time()
    for ev in gen(args.nevents):
        cnt["n_gen"] += 1
        fs = ev.final_state()
        pid = np.abs(fs.pid)
        sel = np.isin(pid, KEEP) & (fs.charge != 0)
        if sel.any():
            pt = fs.pt[sel].astype(np.float64)
            pz = -fs.pz[sel].astype(np.float64)                 # flip z: proton -> -z (ATLAS axes)
            en = fs.en[sel].astype(np.float64)
            pzl = GAMMA * (pz + BETA * en)                       # boost to the lab frame
            eta = np.arcsinh(pzl / np.maximum(pt, 1e-9))
            fid = (pt > 0.5) & (np.abs(eta) < 2.5)
            n = int(fid.sum())
            if n > 0:
                cnt["n_sel"] += 1
                cnt["n_part"] += n
                e_f, p_f = eta[fid], pt[fid]
                lep = np.isin(pid[sel][fid], LEPT)
                cnt["n_lep"] += int(lep.sum())
                h_eta += np.histogram(e_f, bins=ETA_EDGES)[0]
                h_eta_nolep += np.histogram(e_f[~lep], bins=ETA_EDGES)[0]
                ip = np.clip(np.searchsorted(PT_EDGES, p_f, side="right") - 1, 0, n_pt - 1)
                inside = p_f < PT_EDGES[-1]
                np.add.at(h_pt, ip[inside], 1.0)
                np.add.at(h_invpt, ip[inside], 1.0 / p_f[inside])
                k = min(n, NCH_MAX)
                h_nch[k] += 1.0
                h_sumpt[k] += p_f.sum()
        if cnt["n_gen"] % CHECKPOINT_STEP == 0:
            if args.checkpoint_seconds > 0 and time.time() - t_ck > args.checkpoint_seconds:
                write_out(); t_ck = time.time()
            if time.time() - t1 > args.max_seconds:
                print("stopping early after %d events (time limit)" % cnt["n_gen"], file=sys.stderr)
                break
    write_out()
    print("RESULT saved %s | %d events, acceptance %.4f, %.1f ev/s, init %.1f s" %
          (path, cnt["n_gen"], cnt["n_sel"] / max(cnt["n_gen"], 1), cnt["n_gen"] / max(time.time() - t1, 1e-9), t_init))


if __name__ == "__main__":
    main()
