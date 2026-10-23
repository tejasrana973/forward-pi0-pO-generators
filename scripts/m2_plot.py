#!/usr/bin/env python3
"""
M2 plots and tables from merged .npz files (made by m2_merge.py).
Plotting needs only numpy + matplotlib, so you can run it on your laptop (with Times New Roman installed).

  python3 m2_plot.py fig2   --merged merged --system lab1000_none            # Fig. 2 style
  python3 m2_plot.py fig3   --merged merged --system cm43.3394_max   --model Sibyll23d   # Fig. 3 style
  python3 m2_plot.py fig3   --merged merged --system cm13600_max     --model Sibyll23d   # Fig. 7 style
  python3 m2_plot.py frames --merged merged                                  # lab vs CM+boost check
  python3 m2_plot.py table  --merged merged                                  # numbers for the note

Vertical scale: by default dN/dln(xi) for L = 2 fb^-1, using each model's own inelastic cross section
(as in Eq. 2 of the paper). Use --per-event for dN/dln(xi) per inelastic event instead.
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

NICE = {"EposLHC": "EPOS-LHC", "QGSJetII04": "QGSJET II-04", "Sibyll23d": "Sibyll 2.3d",
        "Pythia8": "Pythia 8 (Monash)", "Pythia8-fwd": "Pythia 8 (fwd tune)"}
COLOR = {"EposLHC": "tab:green", "QGSJetII04": "tab:red", "Sibyll23d": "tab:blue",
         "Pythia8": "tab:orange", "Pythia8-fwd": "tab:purple"}


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def load_system(merged, system):
    files = sorted(glob.glob(os.path.join(merged, "*_%s.npz" % system)))
    if not files:
        raise SystemExit("no files like %s/*_%s.npz" % (merged, system))
    return [load(f) for f in files]


def scale_factor(d, lumi):
    """Multiplier turning 'per inelastic event' into 'expected pi0 for luminosity lumi [fb^-1]'."""
    if lumi is None:
        return 1.0
    sig = d["meta"]["xsec_mb"].get("inelastic")
    if sig is None:
        raise SystemExit("no inelastic cross section stored for %s" % d["meta"]["label"])
    return sig * 1e12 * lumi          # 1 mb x 1 fb^-1 = 1e12 events


def spectrum(d, eta_range=None, lumi=None):
    """dN/dln(xi) with Poisson errors; optionally restricted to |eta| in eta_range."""
    h2 = d["h2"]
    if eta_range is not None:
        ec = d["eta_edges"]
        lo, hi = eta_range
        i0 = int(np.searchsorted(ec, lo - 1e-9)); i1 = int(np.searchsorted(ec, hi - 1e-9))
        h = h2[:, i0:i1].sum(axis=1)
    else:
        h = h2.sum(axis=1)
    e = d["lxi_edges"]
    dl = np.diff(e) * np.log(10.0)
    xc = 10 ** (0.5 * (e[:-1] + e[1:]))
    norm = scale_factor(d, lumi) / (d["meta"]["n_events"] * dl)
    return xc, h * norm, np.sqrt(h) * norm, h


def tag(d):
    return NICE.get(d["meta"]["label"], d["meta"]["label"])


def save(fig, name):
    fig.savefig(name + ".png", dpi=200)
    fig.savefig(name + ".pdf")
    print("saved %s.png / .pdf" % name)


def ylabel(lumi):
    if lumi is None:
        return r"$\mathrm{d}N/\mathrm{d}\ln\xi_{\pi^0}^{\mathrm{lab}}$ per inelastic event"
    return r"$\xi\,\mathrm{d}N/\mathrm{d}\xi$  ($L=%g\ \mathrm{fb^{-1}}$)" % lumi


# ----------------------------------------------------------------------------------------------------------
def cmd_fig2(a):
    ds = load_system(a.merged, a.system)
    order = ["EposLHC", "QGSJetII04", "Sibyll23d", "Pythia8", "Pythia8-fwd"]
    ds.sort(key=lambda d: order.index(d["meta"]["label"]) if d["meta"]["label"] in order else 99)
    ref = [d for d in ds if d["meta"]["label"] == a.ref]
    lumi = None if a.per_event else a.lumi
    fig, ax = plt.subplots(3, 1, figsize=(7, 10), gridspec_kw=dict(height_ratios=[3, 2.4, 1.4]))
    for d in ds:
        x, y, ey, h = spectrum(d, lumi=lumi)
        c = COLOR.get(d["meta"]["label"], None)
        ax[0].step(x, np.where(h > 0, y, np.nan), where="mid", color=c, label=tag(d))
        z = (x >= 0.1) & (h > 0)
        ax[1].errorbar(x[z], y[z], yerr=ey[z], fmt=".-", ms=3, lw=1, color=c, label=tag(d))
        if ref:
            xr, yr, eyr, hr = spectrum(ref[0], lumi=lumi)
            ok = z & (hr > 0)
            ratio = y[ok] / yr[ok]
            err = ratio * np.sqrt((ey[ok] / y[ok]) ** 2 + (eyr[ok] / yr[ok]) ** 2)
            ax[2].errorbar(x[ok], ratio, yerr=err, fmt=".-", ms=3, lw=1, color=c)
    for a_ in ax:
        a_.set_xscale("log")
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_xlim(1e-3, 1.0); ax[1].set_xlim(0.1, 1.0); ax[2].set_xlim(0.1, 1.0)
    ax[1].axvline(0.7, color="k", ls=":", lw=0.8)
    ax[0].set_ylabel(ylabel(lumi)); ax[1].set_ylabel(ylabel(lumi))
    ax[2].set_ylabel("ratio to %s" % NICE.get(a.ref, a.ref)); ax[2].axhline(1, color="k", lw=0.6)
    ax[2].set_xlabel(r"$\xi_{\pi^0}^{\mathrm{lab}}=E_{\pi^0}/E_{\mathrm{beam}}$")
    ax[2].set_ylim(0, 3)
    ax[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, a.out or "m2_fig2_%s" % a.system)


def eta_groups(d):
    """Lower edges in steps, then the analytic window [eta_lo, eta_hi], then everything above."""
    lo, hi = d["meta"]["eta_lo"], d["meta"]["eta_hi"]
    step = 1.0 if d["meta"]["sqrt_s"] < 100 else 2.0
    edges = list(np.arange(0.0, lo - 0.3, step)) + [lo, hi, 20.0]
    return edges


def cmd_fig3(a):
    ds = [d for d in load_system(a.merged, a.system) if d["meta"]["label"] == a.model]
    if not ds:
        raise SystemExit("model %s not found for system %s" % (a.model, a.system))
    d = ds[0]
    lumi = None if a.per_event else a.lumi
    edges = eta_groups(d)
    fig, ax = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw=dict(height_ratios=[1, 1]))
    cmap = plt.get_cmap("viridis")
    ngr = len(edges) - 1
    for i in range(ngr):
        lo, hi = edges[i], edges[i + 1]
        is_win = abs(lo - d["meta"]["eta_lo"]) < 1e-6
        x, y, ey, h = spectrum(d, eta_range=(lo, hi), lumi=lumi)
        fm = lambda v: ("%.2f" % v).rstrip("0").rstrip(".")
        if hi >= 20.0:
            lab = r"$|\eta|>%s$" % fm(lo)
        else:
            lab = r"$%s<|\eta|<%s$" % (fm(lo), fm(hi))
        col = "darkturquoise" if is_win else cmap(0.85 * i / max(ngr - 1, 1))
        lw = 2.4 if is_win else 1.3
        yy = np.where(h > 0, y, np.nan)
        ax[0].step(x, yy, where="mid", color=col, lw=lw, label=lab)
        z = x >= 0.3
        ax[1].step(x[z], yy[z], where="mid", color=col, lw=lw)
    xt, yt, _, ht = spectrum(d, lumi=lumi)
    ax[0].step(xt, np.where(ht > 0, yt, np.nan), where="mid", color="k", lw=1, ls="--", label="all $\\eta$")
    for a_ in ax:
        a_.set_xscale("log"); a_.set_yscale("log")
    ax[1].axvline(0.7, color="b", lw=1)
    ax[0].set_xlim(1e-8 if d["meta"]["sqrt_s"] > 1000 else 1e-3, 1.0); ax[1].set_xlim(0.3, 1.0)
    ax[0].set_ylabel(ylabel(lumi)); ax[1].set_ylabel(ylabel(lumi))
    ax[1].set_xlabel(r"$\xi_{\pi^0}^{\mathrm{lab}}=E_{\pi^0}/E_{\mathrm{beam}}$")
    top = ax[0].get_ylim()[1]
    ax[0].set_ylim(ax[0].get_ylim()[0], top * 40)   # head-room for the legend
    ax[0].legend(loc="upper left", fontsize=10, ncol=2)
    rs = d["meta"]["sqrt_s"]
    rs_txt = ("%.1f GeV" % rs) if rs < 100 else ("%.1f TeV" % (rs / 1000.0))
    hemi = ", CM-forward $\\pi^0$ only" if d["meta"].get("forward_only") else ", both hemispheres"
    ax[0].set_title("%s, $\\sqrt{s}=$ %s, boost: %s%s" % (tag(d), rs_txt, d["meta"]["boost"], hemi), fontsize=11)
    fig.tight_layout()
    save(fig, a.out or "m2_fig3_%s_%s" % (a.model, a.system))
    k = list(d["thr"]).index(0.7)
    n, nw = d["pi0_gt"][k], d["pi0_gt_win"][k]
    print("pi0 with xi>0.7: %d, inside window %.2f<|eta|<%.2f: %d (%.2f %%)" %
          (n, d["meta"]["eta_lo"], d["meta"]["eta_hi"], nw, 100.0 * nw / max(n, 1)))


def cmd_frames(a):
    pairs = {}
    for f in glob.glob(os.path.join(a.merged, "Sibyll23d_*.npz")):
        d = load(f)
        pairs[(d["meta"]["frame"], d["meta"]["boost"])] = d
    need = [("lab", "none"), ("cm", "plus"), ("cm", "max")]
    miss = [n for n in need if n not in pairs and not (n[0] == "cm" and any(k[0] == "cm" and k[1] == n[1] and abs(v["meta"]["energy"] - 43.3394) < 0.01 for k, v in pairs.items()))]
    lab = pairs.get(("lab", "none"))
    cm = {}
    for f in glob.glob(os.path.join(a.merged, "Sibyll23d_cm43.3394*.npz")):
        d = load(f); cm[d["meta"]["boost"]] = d
    if lab is None or "plus" not in cm:
        raise SystemExit("need Sibyll23d lab1000_none and cm43.3394_plus merged files")
    xl, yl, el, hl = spectrum(lab); xp, yp, ep, hp = spectrum(cm["plus"])
    ok = (hl > 20) & (hp > 20)
    chi2 = np.sum((hl[ok] - hp[ok] * 0 - hp[ok] * (lab["meta"]["n_events"] / cm["plus"]["meta"]["n_events"])) ** 2 /
                  (hl[ok] + hp[ok] * (lab["meta"]["n_events"] / cm["plus"]["meta"]["n_events"]) ** 2))
    print("frame check (lab vs CM + fixed boost), %d bins with >20 entries: chi2/ndf = %.2f  (about 1 expected)" %
          (ok.sum(), chi2 / ok.sum()))
    fig, ax = plt.subplots(2, 1, figsize=(7, 7), gridspec_kw=dict(height_ratios=[2, 1]), sharex=True)
    ax[0].step(xl[hl > 0], yl[hl > 0], where="mid", label="generated in lab frame")
    ax[0].step(xp[hp > 0], yp[hp > 0], where="mid", ls="--", label="CM frame + fixed boost")
    if "max" in cm:
        xm, ym, em, hm = spectrum(cm["max"])
        ax[0].step(xm[hm > 0], ym[hm > 0], where="mid", color="k", lw=1, label="CM frame + max-energy boost")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].legend(loc="lower left")
    ax[0].set_ylabel(ylabel(None))
    r = yp[ok] / yl[ok]
    ax[1].errorbar(xl[ok], r, yerr=r * np.sqrt(1 / hp[ok] + 1 / hl[ok]), fmt=".", ms=3)
    ax[1].axhline(1, color="k", lw=0.6); ax[1].set_ylim(0.8, 1.2)
    ax[1].set_ylabel("fixed boost / lab"); ax[1].set_xlabel(r"$\xi_{\pi^0}^{\mathrm{lab}}$")
    ax[0].set_title("Sibyll 2.3d, 1 TeV p on p", fontsize=12)
    fig.tight_layout()
    save(fig, a.out or "m2_frames_check")


def cmd_table(a):
    rows = []
    for f in sorted(glob.glob(os.path.join(a.merged, "*.npz"))):
        d = load(f)
        m = d["meta"]; k = list(d["thr"]).index(0.7)
        n = m["n_events"]; p = d["evt_gt"][k] / n
        npi = int(d["pi0_gt"][k]); nwin = int(d["pi0_gt_win"][k])
        nlo = int(d["pi0_gt_below"][k]); nhi = int(d["pi0_gt_above"][k])
        fw = nwin / max(npi, 1)
        fo = " fwd-only" if m.get("forward_only") else ""
        rows.append(dict(label=m["label"], system="%s %g GeV (%s)%s" % (m["frame"], m["energy"], m["boost"], fo), n=n,
                         sig=m["xsec_mb"].get("inelastic", float("nan")),
                         p=p, ep=np.sqrt(p * (1 - p) / n), npi=npi, fw=fw,
                         efw=np.sqrt(max(fw * (1 - fw), 0) / max(npi, 1)),
                         lo=nlo / max(npi, 1), hi=nhi / max(npi, 1), ev=(d["evt_gt"] / n)))
    print("%-12s %-30s %9s %8s %20s %8s %16s %8s %8s" %
          ("model", "system", "events", "sig_inel", "P(max xi_pi0>0.7)", "N(pi0)", "in eta window", "below", "above"))
    for r in rows:
        print("%-12s %-30s %9d %8.2f %12.3e +-%.1e %8d %9.2f +-%.2f %% %7.2f%% %7.2f%%" %
              (r["label"], r["system"], r["n"], r["sig"], r["p"], r["ep"], r["npi"], 100 * r["fw"], 100 * r["efw"],
               100 * r["lo"], 100 * r["hi"]))
    print("(N(pi0), window columns: pi0 with xi>0.7;  below = |eta| < eta_min,  above = |eta| > eta_max)")
    print("\nP(event has a pi0 above xi = threshold):")
    thr = load(sorted(glob.glob(os.path.join(a.merged, "*.npz")))[0])["thr"]
    print("%-12s %-30s " % ("model", "system") + " ".join("%9.1f" % t for t in thr))
    for r in rows:
        print("%-12s %-30s " % (r["label"], r["system"]) + " ".join("%9.2e" % v for v in r["ev"]))
    with open("m2_table.csv", "w") as f:
        f.write("model,system,events,sigma_inel_mb,P_evt_xi_gt_0.7,err,N_pi0_xi_gt_0.7,frac_in_window,err,frac_below,frac_above\n")
        for r in rows:
            f.write("%s,%s,%d,%.3f,%.4e,%.1e,%d,%.4f,%.4f,%.4f,%.4f\n" % (r["label"], r["system"], r["n"], r["sig"], r["p"], r["ep"], r["npi"], r["fw"], r["efw"], r["lo"], r["hi"]))
    with open("m2_table.tex", "w") as f:
        f.write("\\begin{tabular}{llrcc}\n\\hline\nModel & System & Events & $P(\\xi_{\\pi^0}^{\\max}>0.7)$ & In $\\eta$ window [\\%] \\\\\n\\hline\n")
        for r in rows:
            f.write("%s & %s & %.1e & $(%.2f\\pm%.2f)\\times10^{-3}$ & $%.1f\\pm%.1f$ \\\\\n" %
                    (NICE.get(r["label"], r["label"]), r["system"].replace("_", " "), r["n"], 1e3 * r["p"], 1e3 * r["ep"], 100 * r["fw"], 100 * r["efw"]))
        f.write("\\hline\n\\end{tabular}\n")
    print("\nsaved m2_table.csv and m2_table.tex")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("fig2", cmd_fig2), ("fig3", cmd_fig3), ("frames", cmd_frames), ("table", cmd_table)]:
        p = sub.add_parser(name)
        p.add_argument("--merged", default="merged")
        p.add_argument("--system", default="lab1000_none")
        p.add_argument("--model", default="Sibyll23d")
        p.add_argument("--ref", default="Sibyll23d", help="reference model for the ratio panel")
        p.add_argument("--lumi", type=float, default=2.0, help="fb^-1, as in the paper")
        p.add_argument("--per-event", action="store_true", help="per inelastic event instead of per luminosity")
        p.add_argument("--out", default=None)
        p.set_defaults(fn=fn)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()