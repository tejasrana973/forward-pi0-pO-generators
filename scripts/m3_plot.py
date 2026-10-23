#!/usr/bin/env python3
"""
M3 plots and tables (p+O, p+N versus pp) from merged3/*.npz made by m3_merge.py.
Everything is dN/dln(xi) per inelastic event (no luminosity scaling: nuclear cross sections are model dependent).

  python3 m3_plot.py zoom  --system O16_lab1000_none-fo   --ref-system p_lab1000_none-fo   # Fig. 6 style (1 TeV)
  python3 m3_plot.py zoom  --system O16_cm9620_plus-fo    --ref-system p_cm9620_plus-fo    # 9.62 TeV
  python3 m3_plot.py eta   --system O16_cm9620_plus-fo --model Sibyll23d                  # eta decomposition
  python3 m3_plot.py nvo   --system lab1000_none-fo                                       # 14N / 16O
  python3 m3_plot.py nvo   --system cm9620_plus-fo
  python3 m3_plot.py table
"""
import argparse
import glob
import json
import os

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
plt.rcParams["legend.fontsize"] = 12    # legend text
plt.rcParams["pdf.fonttype"] = 42       # keep text as real fonts in the PDF
# -------------------------------------------------------------------------------------------------------------------

NICE = {"EposLHC": "EPOS-LHC", "EposLHCR": "EPOS LHC-R", "QGSJetII04": "QGSJET II-04", "QGSJetIII": "QGSJET III",
        "Sibyll23d": "Sibyll 2.3d", "Sibyll23e": "Sibyll 2.3e", "Pythia8": "Pythia 8 (Monash)",
        "Pythia8Angantyr": "Pythia 8 Angantyr"}
COLOR = {"EposLHC": "tab:green", "EposLHCR": "olive", "QGSJetII04": "tab:red", "QGSJetIII": "tab:brown",
         "Sibyll23d": "tab:blue", "Sibyll23e": "tab:cyan", "Pythia8": "tab:orange", "Pythia8Angantyr": "tab:orange"}
ORDER = ["EposLHC", "EposLHCR", "QGSJetII04", "QGSJetIII", "Sibyll23d", "Sibyll23e", "Pythia8", "Pythia8Angantyr"]
PP_OF = {"Pythia8Angantyr": "Pythia8"}      # pp reference model for each p+A model
DY_LHC = 0.5 * np.log(6.8 / 3.4)            # rapidity shift CM -> LHC lab frame for 6.8 TeV p + 3.4 TeV/nucleon O


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def load_system(merged, system):
    files = sorted(glob.glob(os.path.join(merged, "*_%s.npz" % system)))
    ds = [load(f) for f in files]
    ds.sort(key=lambda d: ORDER.index(d["meta"]["label"]) if d["meta"]["label"] in ORDER else 99)
    return ds


def spectrum(d, eta_range=None):
    h2 = d["h2"]
    if eta_range is not None:
        ec = d["eta_edges"]; lo, hi = eta_range
        h = h2[:, int(np.searchsorted(ec, lo - 1e-9)):int(np.searchsorted(ec, hi - 1e-9))].sum(axis=1)
    else:
        h = h2.sum(axis=1)
    e = d["lxi_edges"]
    dl = np.diff(e) * np.log(10.0)
    xc = 10 ** (0.5 * (e[:-1] + e[1:]))
    norm = 1.0 / (d["meta"]["n_events"] * dl)
    return xc, h * norm, np.sqrt(h) * norm, h


def tag(d):
    return NICE.get(d["meta"]["label"], d["meta"]["label"])


def save(fig, name):
    fig.savefig(name + ".png", dpi=200); fig.savefig(name + ".pdf")
    print("saved %s.png / .pdf" % name)


YL = r"$\mathrm{d}N/\mathrm{d}\ln\xi_{\pi^0}^{\mathrm{lab}}$ per event"
XL = r"$\xi_{\pi^0}^{\mathrm{lab}}=E_{\pi^0}/E_{\mathrm{beam}}$"


def ratio(d1, d2, zmask):
    """d1/d2 of the spectra in the bins selected by zmask (both need entries)."""
    x, y1, e1, h1 = spectrum(d1); _, y2, e2, h2 = spectrum(d2)
    ok = zmask & (h1 > 0) & (h2 > 0)
    r = y1[ok] / y2[ok]
    err = r * np.sqrt(1.0 / h1[ok] + 1.0 / h2[ok])
    return x[ok], r, err


# ----------------------------------------------------------------------------------------------------------
def cmd_zoom(a):
    ds = load_system(a.merged, a.system)
    refs = {d["meta"]["label"]: d for d in load_system(a.merged, a.ref_system)}
    if not ds:
        raise SystemExit("no files for system %s in %s" % (a.system, a.merged))
    refm = [d for d in ds if d["meta"]["label"] == a.ref]
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), gridspec_kw=dict(height_ratios=[3, 1.6, 1.6]), sharex=True)
    for d in ds:
        lab = d["meta"]["label"]; c = COLOR.get(lab)
        x, y, ey, h = spectrum(d)
        z = (x >= 0.1) & (h > 0)
        ax[0].errorbar(x[z], y[z], yerr=ey[z], fmt=".-", ms=3, lw=1, color=c, label=tag(d))
        if refm:
            xr, rr, er = ratio(d, refm[0], (x >= 0.1))
            ax[1].errorbar(xr, rr, yerr=er, fmt=".-", ms=3, lw=1, color=c)
        pp = refs.get(PP_OF.get(lab, lab))
        if pp is not None:
            xp, rp, ep = ratio(d, pp, (x >= 0.1))
            ax[2].errorbar(xp, rp, yerr=ep, fmt=".-", ms=3, lw=1, color=c)
    ax[0].set_yscale("log"); ax[0].set_xscale("log"); ax[0].set_xlim(0.1, 1.0)
    ax[0].axvline(0.7, color="k", ls=":", lw=0.8); ax[1].axvline(0.7, color="k", ls=":", lw=0.8); ax[2].axvline(0.7, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel(YL)
    ax[1].set_ylabel("ratio to %s" % NICE.get(a.ref, a.ref)); ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylim(0, a.ratio_max)
    tgt = ds[0]["meta"]["target"]; rs = ds[0]["meta"]["sqrt_s"]
    ax[2].set_ylabel("p+%s / pp" % tgt.replace("O16", "O").replace("N14", "N")); ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set_ylim(0, a.ratio_max); ax[2].set_xlabel(XL)
    ax[0].legend(loc="lower left")
    ax[0].set_title("p+%s, $\\sqrt{s_{NN}}=$ %.4g GeV, CM-forward $\\pi^0$" % (tgt, rs), fontsize=11)
    fig.tight_layout()
    save(fig, a.out or "m3_zoom_%s" % a.system)


def eta_groups(d):
    lo, hi = d["meta"]["eta_lo"], d["meta"]["eta_hi"]
    step = 1.0 if d["meta"]["sqrt_s"] < 100 else 2.0
    return list(np.arange(0.0, lo - 0.3, step)) + [lo, hi, 20.0]


def cmd_eta(a):
    ds = [d for d in load_system(a.merged, a.system) if d["meta"]["label"] == a.model]
    if not ds:
        raise SystemExit("model %s not found for system %s" % (a.model, a.system))
    d = ds[0]
    edges = eta_groups(d)
    fig, ax = plt.subplots(2, 1, figsize=(7, 8))
    cmap = plt.get_cmap("viridis"); ngr = len(edges) - 1
    fm = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
    for i in range(ngr):
        lo, hi = edges[i], edges[i + 1]
        is_win = abs(lo - d["meta"]["eta_lo"]) < 1e-6
        x, y, ey, h = spectrum(d, eta_range=(lo, hi))
        lab = r"$|\eta|>%s$" % fm(lo) if hi >= 20 else r"$%s<|\eta|<%s$" % (fm(lo), fm(hi))
        col = "darkturquoise" if is_win else cmap(0.85 * i / max(ngr - 1, 1)); lw = 2.4 if is_win else 1.3
        yy = np.where(h > 0, y, np.nan)
        ax[0].step(x, yy, where="mid", color=col, lw=lw, label=lab)
        ax[1].step(x[x >= 0.3], yy[x >= 0.3], where="mid", color=col, lw=lw)
    xt, yt, _, ht = spectrum(d)
    ax[0].step(xt, np.where(ht > 0, yt, np.nan), where="mid", color="k", lw=1, ls="--", label="all $\\eta$")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_yscale("log")
    ax[1].axvline(0.7, color="b", lw=1)
    top = ax[0].get_ylim()[1]; ax[0].set_ylim(ax[0].get_ylim()[0], top * 40)
    ax[0].set_xlim(1e-8 if d["meta"]["sqrt_s"] > 1000 else 1e-3, 1.0); ax[1].set_xlim(0.3, 1.0)
    ax[0].set_ylabel(YL); ax[1].set_ylabel(YL); ax[1].set_xlabel(XL)
    ax[0].legend(loc="upper left", fontsize=10, ncol=2)
    ax[0].set_title("%s, p+%s, $\\sqrt{s_{NN}}=$ %.4g GeV" % (tag(d), d["meta"]["target"], d["meta"]["sqrt_s"]), fontsize=11)
    fig.tight_layout()
    save(fig, a.out or "m3_eta_%s_%s" % (a.model, a.system))
    k = list(d["thr"]).index(0.7); n, nw = d["pi0_gt"][k], d["pi0_gt_win"][k]
    print("pi0 with xi>0.7: %d, inside window %.2f<|eta|<%.2f: %d (%.2f %%)" %
          (n, d["meta"]["eta_lo"], d["meta"]["eta_hi"], nw, 100.0 * nw / max(n, 1)))


def cmd_nvo(a):
    dn = {d["meta"]["label"]: d for d in load_system(a.merged, "N14_" + a.system)}
    do = {d["meta"]["label"]: d for d in load_system(a.merged, "O16_" + a.system)}
    common = [m for m in ORDER if m in dn and m in do]
    if not common:
        raise SystemExit("need N14_%s and O16_%s files for the same model" % (a.system, a.system))
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for m in common:
        x, y, _, h = spectrum(do[m])
        xr, r, er = ratio(dn[m], do[m], x >= 0.1)
        ax.errorbar(xr, r, yerr=er, fmt=".-", ms=3, lw=1, color=COLOR.get(m), label=NICE[m])
    ax.axhline(1, color="k", lw=0.6); ax.axvline(0.7, color="k", ls=":", lw=0.8)
    ax.set_xscale("log"); ax.set_xlim(0.1, 1.0); ax.set_ylim(0.5, 1.5)
    ax.set_xlabel(XL); ax.set_ylabel(r"p+$^{14}$N / p+$^{16}$O"); ax.legend(loc="lower left")
    fig.tight_layout()
    save(fig, a.out or "m3_nvo_%s" % a.system)


def pvals(d, thr_value):
    k = list(d["thr"]).index(thr_value)
    n = d["meta"]["n_events"]; p = d["evt_gt"][k] / n
    return p, np.sqrt(p * (1 - p) / n)


def cmd_table(a):
    allf = [load(f) for f in sorted(glob.glob(os.path.join(a.merged, "*.npz")))]
    key = lambda d: (d["meta"]["label"], d["meta"]["target"], d["meta"]["frame"], d["meta"]["energy"])
    by = {key(d): d for d in allf}
    print("%-16s %-4s %-14s %9s %8s %20s %8s %16s" %
          ("model", "tgt", "system", "events", "sig_inel", "P(max xi_pi0>0.7)", "N(pi0)", "in eta window"))
    for d in allf:
        m = d["meta"]; p, ep = pvals(d, 0.7); k = list(d["thr"]).index(0.7)
        npi, nw = int(d["pi0_gt"][k]), int(d["pi0_gt_win"][k]); fw = nw / max(npi, 1)
        print("%-16s %-4s %-14s %9d %8.2f %12.3e +-%.1e %8d %9.1f +-%.1f %%" %
              (m["label"], m["target"], "%s%g" % (m["frame"], m["energy"]), m["n_events"],
               m["xsec_mb"].get("inelastic", float("nan")), p, ep, npi, 100 * fw, 100 * np.sqrt(max(fw * (1 - fw), 0) / max(npi, 1))))
    for thr in (0.5, 0.7):
        print("\nRatios of P(event has a pi0 with xi > %.1f), with statistical errors:" % thr)
        print("%-16s %-14s %18s %18s" % ("model", "system", "p+O / pp", "p+N / p+O"))
        for (lab, tgt, fr, en), d in sorted(by.items()):
            if tgt != "O16":
                continue
            pO, ePO = pvals(d, thr)
            s1 = s2 = "-"
            pp = by.get((PP_OF.get(lab, lab), "p", fr, en))
            if pp is not None:
                p0, e0 = pvals(pp, thr); r = pO / p0
                s1 = "%.2f +- %.2f" % (r, r * np.hypot(ePO / pO, e0 / p0))
            nn = by.get((lab, "N14", fr, en))
            if nn is not None:
                pn, en_ = pvals(nn, thr); r = pn / pO
                s2 = "%.2f +- %.2f" % (r, r * np.hypot(en_ / pn, ePO / pO))
            print("%-16s %-14s %18s %18s" % (lab, "%s%g" % (fr, en), s1, s2))
    cm = [d for d in allf if d["meta"]["frame"] == "cm" and abs(d["meta"]["energy"] - 9620) < 1]
    if cm:
        lo, hi = cm[0]["meta"]["eta_lo"], cm[0]["meta"]["eta_hi"]
        print("\neta window for xi>0.7 at sqrt(s_NN)=9.62 TeV (NN-CM frame): %.2f < eta < %.2f" % (lo, hi))
        print("My estimate in the LHC lab frame (6.8 TeV p on 3.4 TeV/nucleon O, shift %.2f toward the proton): %.2f < eta < %.2f"
              % (DY_LHC, lo + DY_LHC, hi + DY_LHC))
        print("(to be re-checked in M4; LHCf acceptance starts at about eta = 8.4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("zoom", cmd_zoom), ("eta", cmd_eta), ("nvo", cmd_nvo), ("table", cmd_table)]:
        p = sub.add_parser(name)
        p.add_argument("--merged", default="merged3")
        p.add_argument("--system", default="O16_lab1000_none-fo")
        p.add_argument("--ref-system", default="p_lab1000_none-fo", help="pp reference system for the p+A/pp ratio")
        p.add_argument("--model", default="Sibyll23d")
        p.add_argument("--ref", default="Sibyll23d", help="reference model for the ratio panel")
        p.add_argument("--ratio-max", type=float, default=3.0)
        p.add_argument("--out", default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
