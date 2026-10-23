#!/usr/bin/env python3
"""
M0: check that chromo runs for pp (fixed target, 1 TeV) and p+O (9.62 TeV per nucleon pair),
that pi0 can be kept stable, and that energy-momentum is conserved.

IMPORTANT chromo rule: each Fortran generator can be initialised only ONCE per Python process.
So this script starts a fresh Python process for every (model, system) pair.
We will use the same pattern (one process per model) for all later production runs.

Run:  python3 m0_check_install.py > m0.log 2>&1            (all models, a few minutes)
      python3 m0_check_install.py --quick > m0.log 2>&1    (skip the slow QGSJet-III start-up)
"""
import argparse
import json
import subprocess
import sys
import time

import numpy as np

PI0 = 111
M_P = 0.938272  # GeV

PP_MODELS = ["EposLHC", "QGSJetII04", "Sibyll23d", "Pythia8"]
PO_MODELS = ["EposLHC", "EposLHCR", "QGSJetII04", "QGSJetIII",
             "Sibyll23d", "Sibyll23e", "Pythia8Angantyr"]


def worker(model, system, n_events, seed=1):
    """Runs INSIDE its own process: generate events, return simple numbers as a dict."""
    import chromo.models as M
    from chromo.constants import TeV
    from chromo.kinematics import CenterOfMass, FixedTarget

    if system == "pp":
        kin = FixedTarget(1 * TeV, "p", "p")           # 1 TeV proton on proton at rest
    else:
        kin = CenterOfMass(9.62 * TeV, "p", (16, 8))   # (A, Z) = oxygen-16, sqrt(s_NN) = 9.62 TeV

    t0 = time.time()
    gen = getattr(M, model)(kin, seed=seed)
    gen.set_stable(PI0)                                # keep pi0 in the final state (no decay)
    t_init = time.time() - t0

    t1 = time.time()
    n_pi0, sum_e, sum_pz, e_max = [], [], [], []
    for ev in gen(n_events):
        fs = ev.final_state()                          # stable particles only
        pi = fs.pid == PI0
        n_pi0.append(int(pi.sum()))
        sum_e.append(float(fs.en.sum()))
        sum_pz.append(float(fs.pz.sum()))
        e_max.append(float(fs.en[pi].max()) if pi.any() else 0.0)
    return dict(model=model, system=system, n_pi0=float(np.mean(n_pi0)),
                sum_e=float(np.mean(sum_e)), sum_pz=float(np.mean(sum_pz)),
                e_pi0_max=float(np.max(e_max)), t=t_init + (time.time() - t1))


def call_worker(model, system, n):
    """Start a fresh Python process for one (model, system) pair and read its result line."""
    cmd = [sys.executable, __file__, "--worker", model, system, "-n", str(n)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return json.loads(line[7:])
    print("FAILED:", model, system, "\n", p.stderr[-800:])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip QGSJetIII (slow start-up)")
    ap.add_argument("-n", type=int, default=300, help="events per model")
    ap.add_argument("--worker", nargs=2, metavar=("MODEL", "SYSTEM"), help="internal use")
    args = ap.parse_args()

    if args.worker:                                    # child process
        r = worker(args.worker[0], args.worker[1], args.n)
        print("RESULT " + json.dumps(r))
        return

    out = []
    e_exp = 1000.0 + M_P
    pz_exp = np.sqrt(1000.0**2 - M_P**2)
    out.append("=== TEST 1: p + p, E_lab = 1 TeV (sqrt(s) = 43.34 GeV) ===")
    out.append("expected: sum(E) = %.2f GeV, sum(pz) = %.2f GeV (to ~0.1 GeV)" % (e_exp, pz_exp))
    out.append("%-12s %8s %10s %10s %12s %7s" % ("model", "pi0/evt", "sum(E)", "sum(pz)", "max E_pi0", "t [s]"))
    for m in PP_MODELS:
        r = call_worker(m, "pp", args.n)
        if r:
            out.append("%-12s %8.2f %10.2f %10.2f %12.1f %7.1f" %
                       (m, r["n_pi0"], r["sum_e"], r["sum_pz"], r["e_pi0_max"], r["t"]))

    out.append("")
    out.append("=== TEST 2: p + 16O, sqrt(s_NN) = 9620 GeV (nucleon-nucleon CM frame) ===")
    out.append("%-16s %8s %7s" % ("model", "pi0/evt", "t [s]"))
    for m in PO_MODELS:
        if args.quick and m == "QGSJetIII":
            out.append("%-16s skipped (--quick)" % m)
            continue
        r = call_worker(m, "pO", args.n)
        if r:
            out.append("%-16s %8.1f %7.1f" % (m, r["n_pi0"], r["t"]))
    out.append("")
    out.append("M0 done.")
    print("\n".join(out))


if __name__ == "__main__":
    main()