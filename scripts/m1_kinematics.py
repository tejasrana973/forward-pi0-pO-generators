#!/usr/bin/env python3
"""
M1: pseudorapidity window of 'gamma-like' pi0 as a function of sqrt(s).

Idea (same logic as Leitgeb et al., arXiv:2509.04040):
  A pi0 is 'gamma-like' if it takes at least xi_cut = 70 % of the maximum possible
  energy sqrt(s)/2 in the nucleon-nucleon CM frame.  For such a pi0:
    * the largest possible rapidity is reached at pT -> 0 and E = sqrt(s)/2:
          eta_max = ln( sqrt(s) / m_pi0 )
    * the smallest rapidity is reached at E = xi_cut * sqrt(s)/2 and the largest pT (pT_max):
          eta_min = ln( xi_cut * sqrt(s) / sqrt(m_pi0^2 + pT_max^2) )
  (eta ~ y here, because E >> m.  Compare these formulas with Eqs. 3-5 of the paper and
   note any difference in your write-up.)

Part A: prints a table and makes a plot (eta window vs sqrt(s)).
Part B: cross-check with real generated events (Sibyll 2.3d, 1 TeV p on p at rest):
        what fraction of pi0 with E_cm >= xi_cut*sqrt(s)/2 and pT < pT_max lies inside the window?

Run:  python3 m1_kinematics.py > m1.log 2>&1        (A + B, about 1 min)
      python3 m1_kinematics.py --no-mc               (A only, instant)
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- plot style: Times New Roman (falls back to a Times-like font if TNR is missing) ----------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "stix"  # Times-like math glyphs to match
plt.rcParams["font.size"] = 14          # base size; axis labels, ticks, legend all scale from this
# plt.rcParams["axes.titlesize"] = 14     # plot title
# plt.rcParams["axes.labelsize"] = 14     # x/y axis labels
# plt.rcParams["xtick.labelsize"] = 12    # x-axis tick numbers
# plt.rcParams["ytick.labelsize"] = 12    # y-axis tick numbers
plt.rcParams["legend.fontsize"] = 13    # legend text
plt.rcParams["pdf.fonttype"] = 42       # keep text as real fonts in the PDF
# -------------------------------------------------------------------------------------------------------------------

M_PI0 = 0.1349768   # GeV
M_P = 0.938272      # GeV


def eta_max(sqrt_s):
    return np.log(sqrt_s / M_PI0)


def eta_min(sqrt_s, xi_cut=0.7, pt_max=1.5):
    return np.log(xi_cut * sqrt_s / np.sqrt(M_PI0**2 + pt_max**2))


# ----------------------------------------------------------------------------
def part_a(xi_cut, pt_max):
    points = [("1 TeV p on p at rest", 43.34),
              ("pp 7 TeV", 7000.0),
              ("pp 13 TeV", 13000.0),
              ("pp 13.6 TeV", 13600.0),
              ("pO 9.62 TeV (per nucleon pair)", 9620.0)]
    print("\n=== PART A: eta window for xi >= %.2f, pT < %.1f GeV ===" % (xi_cut, pt_max))
    print("%-34s %10s %9s %9s" % ("system", "sqrt(s)", "eta_min", "eta_max"))
    for name, rs in points:
        print("%-34s %10.1f %9.2f %9.2f" % (name, rs, eta_min(rs, xi_cut, pt_max), eta_max(rs)))
    print("Paper numbers to compare (1 TeV fixed target and 13.6 TeV):")
    print("   43.3 GeV : 3.0 < |eta| < 5.77        13.6 TeV : 8.75 < |eta| < 11.5")

    # sanity asserts against the paper's quoted numbers (rounded as in the paper)
    assert abs(eta_min(43.34, xi_cut, pt_max) - 3.0) < 0.05
    assert abs(eta_max(43.34) - 5.77) < 0.01
    assert abs(eta_min(13600.0, xi_cut, pt_max) - 8.75) < 0.05
    assert abs(eta_max(13600.0) - 11.5) < 0.05
    print("-> asserts passed: formulas reproduce the quoted windows.")

    # plot
    rs = np.logspace(1, 5, 400)              # 10 GeV ... 100 TeV
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.fill_between(rs, eta_min(rs, xi_cut, pt_max), eta_max(rs), alpha=0.25,
                    label=r"window for $\xi\geq%.1f$, $p_T<%.1f$ GeV" % (xi_cut, pt_max))
    ax.plot(rs, eta_max(rs), lw=1.5, label=r"$\eta_{\max}=\ln(\sqrt{s}/m_{\pi^0})$")
    ax.plot(rs, eta_min(rs, xi_cut, pt_max), lw=1.5, label=r"$\eta_{\min}$")
    for name, x, ls in [("1 TeV lab", 43.34, ":"), ("pO 9.62 TeV", 9620, "--"), ("13.6 TeV", 13600, "-.")]:
        ax.axvline(x, color="k", ls=ls, lw=0.8)
        ax.text(x * 1.05, 0.4, name, rotation=90, fontsize=12, va="bottom")
    ax.axhline(8.4, color="gray", ls="--", lw=0.8)
    ax.text(11, 8.55, r"LHCf acceptance edge $\eta\approx8.4$ (approx.)", fontsize=12, color="gray")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\sqrt{s}$ [GeV]")
    ax.set_ylabel(r"$\eta$ (CM frame)")
    ax.set_ylim(0, 13)
    ax.legend(fontsize=12, loc="upper left")
    fig.tight_layout()
    fig.savefig("m1_eta_window.png", dpi=300)
    fig.savefig("m1_eta_window.pdf")
    print("saved m1_eta_window.png / .pdf")


# ----------------------------------------------------------------------------
def part_b(xi_cut, pt_max, n_events):
    import chromo.models as M
    from chromo.constants import TeV
    from chromo.kinematics import FixedTarget

    kin = FixedTarget(1 * TeV, "p", "p")
    rs = kin.ecm
    e_lab = kin.elab + M_P
    pz_lab = np.sqrt(kin.elab**2 - M_P**2)
    beta = pz_lab / e_lab                    # boost velocity of the CM frame along +z
    gamma = 1.0 / np.sqrt(1.0 - beta**2)

    gen = M.Sibyll23d(kin, seed=7)
    gen.set_stable(111)                      # pi0 stays stable

    e_cm_all, pt_all, eta_all, y_all = [], [], [], []
    for ev in gen(n_events):
        fs = ev.final_state()
        pi = fs.pid == 111
        e, pz, pt = fs.en[pi], fs.pz[pi], fs.pt[pi]
        e_cm = gamma * (e - beta * pz)       # Lorentz boost lab -> CM
        pz_cm = gamma * (pz - beta * e)
        eta_cm = np.arcsinh(pz_cm / np.maximum(pt, 1e-9))
        y_cm = 0.5 * np.log((e_cm + pz_cm) / (e_cm - pz_cm))   # true rapidity
        e_cm_all.append(e_cm); pt_all.append(pt); eta_all.append(eta_cm); y_all.append(y_cm)
    e_cm = np.concatenate(e_cm_all); pt = np.concatenate(pt_all)
    eta = np.concatenate(eta_all); y = np.concatenate(y_all)

    xi = e_cm / (rs / 2.0)                   # energy fraction of the maximum sqrt(s)/2
    sel = (xi >= xi_cut) & (pt < pt_max)
    lo, hi = eta_min(rs, xi_cut, pt_max), eta_max(rs)
    inside = sel & (np.abs(eta) >= lo) & (np.abs(eta) <= hi)
    inside_y = sel & (np.abs(y) >= lo) & (np.abs(y) <= hi)
    print("\n=== PART B: Sibyll 2.3d cross-check, %d events, 1 TeV p on p ===" % n_events)
    print("pi0 total: %d" % len(xi))
    print("pi0 with xi >= %.2f and pT < %.1f GeV: %d" % (xi_cut, pt_max, sel.sum()))
    if sel.sum() > 0:
        print("inside the window, using pseudorapidity eta: %d / %d" % (inside.sum(), sel.sum()))
        print("inside the window, using true rapidity y   : %d / %d" % (inside_y.sum(), sel.sum()))
        print("(formulas are exact bounds on y; eta differs from y when pT is below ~m_pi0)")
    print("largest xi found: %.3f" % xi.max())
    print("fraction of events with a pi0 above xi_cut: see M2 (needs event-by-event bookkeeping)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xi-cut", type=float, default=0.7)
    ap.add_argument("--pt-max", type=float, default=1.5)
    ap.add_argument("--no-mc", action="store_true")
    ap.add_argument("-n", type=int, default=100000, help="events for part B")
    args = ap.parse_args()
    part_a(args.xi_cut, args.pt_max)
    if not args.no_mc:
        part_b(args.xi_cut, args.pt_max, args.n)


if __name__ == "__main__":
    main()