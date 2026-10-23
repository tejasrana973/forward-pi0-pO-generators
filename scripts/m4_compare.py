#!/usr/bin/env python3
"""
M4 step 2: compare generator predictions (merged4/*_O16.npz) with the ATLAS p+O data (HEPData CSV tables).

  mkdir hepdata; tar -xzf HEPData-ins3140590-v1-csv.tar.gz -C hepdata
  python3 m4_compare.py --hepdata hepdata/HEPData-ins3140590-v1-csv > m4_compare.log 2>&1

HEPData tables (ins3140590 v1):
  Table 1  (1/Nev) dNev/dn_ch               n_ch bins 1..80 (width 1), then up to 200.5; sums to 1 (density per unit n_ch)
  Table 2  (1/Nev) 1/(2 pi pT) d2N/(deta dpT)   25 pT bins 0.5 .. 50 GeV
  Table 3  <pT> versus n_ch                 same n_ch bins as Table 1
  Table 4  (1/Nev) dN/deta                  100 eta bins of width 0.1, -2.5 .. 2.5 (eta > 0 = oxygen-going side)
Columns: x, x low, x high, y, stat+, stat-, sys+, sys-.   Data band = stat and sys added in quadrature.
No covariance matrix is available, so no chi2 is computed.  The summary reports the range of MC/data and the fraction of
bins in which the model lies inside the data band (model statistical errors are not included in that fraction).
"""
import argparse
import csv
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

NICE = {"Sibyll23e": "Sibyll 2.3e", "DpmjetIII191": "DPMJET-III 2019.1", "Pythia8Angantyr": "Pythia 8 Angantyr",
        "QGSJetII04": "QGSJET II-04", "QGSJetIII": "QGSJET III", "EposLHCR": "EPOS LHC-R"}
COLOR = {"Sibyll23e": "tab:green", "DpmjetIII191": "black", "Pythia8Angantyr": "tab:red",
         "QGSJetII04": "tab:purple", "QGSJetIII": "tab:orange", "EposLHCR": "deeppink"}
ORDER = ["Pythia8Angantyr", "EposLHCR", "DpmjetIII191", "QGSJetII04", "QGSJetIII", "Sibyll23e"]
MIN_ENTRIES = 10          # model bins with fewer entries are not shown / not used


def read_table(path):
    rows = [l for l in open(path) if not l.startswith("#:")]
    r = list(csv.reader(rows))
    d = np.array([[float(x) for x in row] for row in r[1:] if row], float)
    return dict(x=d[:, 0], lo=d[:, 1], hi=d[:, 2], y=d[:, 3], stat_p=d[:, 4], stat_m=np.abs(d[:, 5]),
                sys_p=d[:, 6], sys_m=np.abs(d[:, 7]))


def band(t):
    up = np.hypot(t["stat_p"], t["sys_p"]); dn = np.hypot(t["stat_m"], t["sys_m"])
    return t["y"] - dn, t["y"] + up


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


# ---------- model histograms rebinned to the data bins ----------
def m_eta(d, t):
    e = d["eta_edges"]; n = d["meta"]["n_sel"]
    i0 = np.rint((t["lo"] - e[0]) / 0.05).astype(int); i1 = np.rint((t["hi"] - e[0]) / 0.05).astype(int)
    c = np.array([d["h_eta"][a:b].sum() for a, b in zip(i0, i1)]); w = t["hi"] - t["lo"]
    return c / (n * w), np.sqrt(c) / (n * w), c


def m_pt(d, t):
    e = d["pt_edges"]; n = d["meta"]["n_sel"]
    i0 = np.array([np.argmin(np.abs(e - x)) for x in t["lo"]]); i1 = np.array([np.argmin(np.abs(e - x)) for x in t["hi"]])
    assert np.max(np.abs(e[i0] - t["lo"])) < 1e-6 and np.max(np.abs(e[i1] - t["hi"])) < 1e-6, "pT edges not on the fine grid"
    c = np.array([d["h_pt"][a:b].sum() for a, b in zip(i0, i1)])
    s = np.array([d["h_invpt"][a:b].sum() for a, b in zip(i0, i1)])
    w = t["hi"] - t["lo"]; den = n * 5.0 * w * 2.0 * np.pi
    y = s / den
    return y, np.where(c > 0, y / np.sqrt(np.maximum(c, 1)), 0.0), c


def m_nch(d, t):
    h = d["h_nch"]; n = d["meta"]["n_sel"]
    a = np.rint(t["lo"] + 0.5).astype(int); b = np.rint(t["hi"] + 0.5).astype(int)
    c = np.array([h[i:j].sum() for i, j in zip(a, b)]); w = t["hi"] - t["lo"]
    return c / (n * w), np.sqrt(c) / (n * w), c


def m_meanpt(d, t):
    h, s = d["h_nch"], d["h_sumpt"]; k = np.arange(len(h))
    a = np.rint(t["lo"] + 0.5).astype(int); b = np.rint(t["hi"] + 0.5).astype(int)
    num = np.array([s[i:j].sum() for i, j in zip(a, b)]); den = np.array([(k * h)[i:j].sum() for i, j in zip(a, b)])
    cnt = np.array([h[i:j].sum() for i, j in zip(a, b)])
    y = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    return y, np.zeros_like(y), cnt


def save(fig, name):
    fig.savefig(name + ".png", dpi=200); fig.savefig(name + ".pdf"); print("saved %s.png / .pdf" % name)


def compare(name, t, models, fn, xlabel, ylabel, xlog=False, ylog=False, rlim=(0.5, 1.5), out=None, summary=None,
            xlim=None, title=None):
    lo, hi = band(t)
    fig, ax = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw=dict(height_ratios=[3, 1.5]), sharex=True)
    xe = np.append(t["lo"], t["hi"][-1])                      # true bin edges for the band
    ext = lambda v: np.append(v, v[-1])
    ax[0].fill_between(xe, ext(lo), ext(hi), step="post", color="lightgray", label="ATLAS data (stat+sys)", zorder=1)
    ax[0].plot(t["x"], t["y"], "o", color="k", ms=2.5, zorder=3)
    ax[1].fill_between(xe, ext(lo / t["y"]), ext(hi / t["y"]), step="post", color="lightgray", zorder=1)
    ax[1].axhline(1, color="k", lw=0.8)
    for d in models:
        m = d["meta"]["model"]; y, ey, c = fn(d, t); ok = (c >= MIN_ENTRIES) & np.isfinite(y)
        col = COLOR[m]
        ax[0].plot(t["x"][ok], y[ok], "-", lw=1.3, color=col, label=NICE[m], zorder=2)
        r = y / t["y"]; er = ey / t["y"]
        ax[1].errorbar(t["x"][ok], r[ok], yerr=er[ok], fmt="-", lw=1.2, color=col, elinewidth=0.6)
        inside = ((y >= lo) & (y <= hi))[ok]
        if summary is not None and ok.any():
            summary.append((name, NICE[m], int(ok.sum()), np.nanmin(r[ok]), np.nanmax(r[ok]), np.nanmedian(r[ok]), inside.mean()))
    if xlog: ax[0].set_xscale("log")
    if ylog: ax[0].set_yscale("log")
    if xlim: ax[0].set_xlim(*xlim)
    ax[0].set_ylabel(ylabel); ax[0].legend(fontsize=9, ncol=2)
    ax[1].set_ylabel("MC / data"); ax[1].set_ylim(*rlim); ax[1].set_xlabel(xlabel)
    if title: ax[0].set_title(title, fontsize=11)
    fig.tight_layout(); save(fig, out or "m4_compare_%s" % name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hepdata", required=True, help="folder with Table1.csv ... Table4.csv")
    ap.add_argument("--merged", default="merged4")
    a = ap.parse_args()
    T = {i: read_table(os.path.join(a.hepdata, "Table%d.csv" % i)) for i in (1, 2, 3, 4)}
    models = [load(f) for f in sorted(glob.glob(os.path.join(a.merged, "*_O16.npz")))]
    models.sort(key=lambda d: ORDER.index(d["meta"]["model"]) if d["meta"]["model"] in ORDER else 99)
    if not models:
        raise SystemExit("no *_O16.npz files in %s" % a.merged)
    summary = []
    ttl = r"p+O, $\sqrt{s_{NN}}=9.62$ TeV, $p_T>0.5$ GeV, $|\eta|<2.5$"

    data_nch = (T[4]["y"] * 0.1).sum()
    print("Data: integral of dN/deta (Table 4) = %.2f charged particles per event; Table 2 integral gives %.2f; "
          "Table 1 normalisation = %.4f" % (data_nch, (2 * np.pi * T[2]["x"] * T[2]["y"] * (T[2]["hi"] - T[2]["lo"])).sum() * 5,
                                           (T[1]["y"] * (T[1]["hi"] - T[1]["lo"])).sum()))
    print("\n%-18s %10s %9s %9s %9s" % ("model", "events", "<n_ch>", "vs data", "A"))
    for d in models:
        m = d["meta"]; mean = m["n_part"] / m["n_sel"]
        print("%-18s %10d %9.2f %+8.1f%% %9.4f" % (m["model"], m["n_gen"], mean, 100 * (mean / data_nch - 1), m["n_sel"] / m["n_gen"]))

    compare("eta", T[4], models, m_eta, r"$\eta$  (ATLAS lab frame, $+$ = oxygen-going side)",
            r"$\frac{1}{N_{\mathrm{ev}}}\frac{\mathrm{d}N_{\mathrm{ch}}}{\mathrm{d}\eta}$", rlim=(0.5, 1.5),
            summary=summary, title=ttl)
    compare("pt", T[2], models, m_pt, r"$p_T$ [GeV]",
            r"$\frac{1}{N_{\mathrm{ev}}}\frac{1}{2\pi p_T}\frac{\mathrm{d}^2N_{\mathrm{ch}}}{\mathrm{d}\eta\,\mathrm{d}p_T}$ [GeV$^{-2}$]",
            xlog=True, ylog=True, rlim=(0.2, 3.0), summary=summary, title=ttl)
    compare("nch", T[1], models, m_nch, r"$n_{\mathrm{ch}}$",
            r"$\frac{1}{N_{\mathrm{ev}}}\frac{\mathrm{d}N_{\mathrm{ev}}}{\mathrm{d}n_{\mathrm{ch}}}$", xlog=True, ylog=True,
            rlim=(0.1, 5.0), summary=summary, title=ttl)
    compare("meanpt", T[3], models, m_meanpt, r"$n_{\mathrm{ch}}$", r"$\langle p_T\rangle$ [GeV]", xlog=True,
            rlim=(0.7, 1.3), summary=summary, title=ttl)

    print("\n=== MC / data summary (bins where the model has >= %d entries) ===" % MIN_ENTRIES)
    print("%-8s %-18s %6s %9s %9s %9s %14s" % ("observ.", "model", "bins", "min", "max", "median", "inside band"))
    for s in summary:
        print("%-8s %-18s %6d %9.2f %9.2f %9.2f %13.0f%%" % (s[0], s[1], s[2], s[3], s[4], s[5], 100 * s[6]))
    with open("m4_compare.csv", "w") as f:
        f.write("observable,model,bins,min_ratio,max_ratio,median_ratio,frac_inside_band\n")
        for s in summary:
            f.write("%s,%s,%d,%.4f,%.4f,%.4f,%.3f\n" % s)
    print("\nsaved m4_compare.csv")


if __name__ == "__main__":
    main()