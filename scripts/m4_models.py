#!/usr/bin/env python3
"""
M4 step 1 (no data needed): model predictions for the ATLAS p+O observables from merged4/*.npz, plus a check of
our fiducial-acceptance numbers against the values quoted in the ATLAS paper.

  python3 m4_models.py > m4_models.log 2>&1

Observables (all in the ATLAS fiducial region pT > 0.5 GeV, |eta| < 2.5, per selected event):
  eta:    (1/Nev) dN/deta                          eta in the ATLAS lab frame (+ = oxygen-going side)
  pT:     (1/Nev) (1/2 pi pT) d2N/(deta dpT)       averaged over |eta| < 2.5 (Delta eta = 5)
  n_ch:   (1/Nev) dNev/dn_ch
  <pT>(n_ch)
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

NICE = {"Sibyll23e": "Sibyll 2.3e", "DpmjetIII191": "DPMJET-III 2019.1", "Pythia8Angantyr": "Pythia 8 Angantyr",
        "QGSJetII04": "QGSJET II-04", "QGSJetIII": "QGSJET III", "EposLHCR": "EPOS LHC-R"}
COLOR = {"Sibyll23e": "tab:green", "DpmjetIII191": "black", "Pythia8Angantyr": "tab:red",
         "QGSJetII04": "tab:purple", "QGSJetIII": "tab:orange", "EposLHCR": "deeppink"}
ORDER = ["Pythia8Angantyr", "EposLHCR", "DpmjetIII191", "QGSJetII04", "QGSJetIII", "Sibyll23e"]
# fiducial acceptance A quoted in the ATLAS paper (fraction of inelastic events with >= 1 fiducial charged particle)
ATLAS_A = {"DpmjetIII191": 0.959, "EposLHCR": 0.842, "Pythia8Angantyr": 0.891, "QGSJetII04": 0.899, "Sibyll23e": 0.925}
PT_PLOT = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 70, 100, 150]


def load(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files if k != "meta"}
    out["meta"] = json.loads(str(d["meta"]))
    return out


def rebin_idx(fine_edges, new_edges):
    idx = np.searchsorted(fine_edges, new_edges)
    idx = np.clip(idx, 0, len(fine_edges) - 1)
    err = np.max(np.abs(fine_edges[idx] - np.asarray(new_edges)))
    if err > 1e-6:
        print("WARNING: requested edges not on the fine grid (max snap %.4f)" % err)
    return idx


def rebin(h, idx):
    return np.add.reduceat(np.concatenate([h, [0.0]]), idx[:-1])[: len(idx) - 1] if False else \
        np.array([h[idx[i]:idx[i + 1]].sum() for i in range(len(idx) - 1)])


def eta_dist(d, nolep=False):
    h = d["h_eta_nolep"] if nolep else d["h_eta"]
    e = d["eta_edges"]; idx = np.arange(0, len(e), 2)           # merge pairs: 0.1 wide bins
    c = rebin(h, idx); ee = e[idx]; w = np.diff(ee); n = d["meta"]["n_sel"]
    return 0.5 * (ee[1:] + ee[:-1]), c / (n * w), np.sqrt(c) / (n * w)


def pt_dist(d):
    e = d["pt_edges"]; idx = rebin_idx(e, PT_PLOT); ee = e[idx]
    c = rebin(d["h_pt"], idx); s = rebin(d["h_invpt"], idx); n = d["meta"]["n_sel"]
    w = np.diff(ee); den = n * 5.0 * w * 2 * np.pi          # Delta eta = 5
    y = s / den
    ey = np.where(c > 0, y / np.sqrt(np.maximum(c, 1)), 0.0)
    return np.sqrt(ee[1:] * ee[:-1]), y, ey, c


def nch_dist(d):
    h = d["h_nch"]; n = np.arange(len(h)); N = d["meta"]["n_sel"]
    return n[1:], h[1:] / N, np.sqrt(h[1:]) / N, h[1:]


def meanpt(d):
    h, s = d["h_nch"], d["h_sumpt"]; n = np.arange(len(h))
    ok = (n >= 1) & (h > 0)
    m = np.full(len(h), np.nan); m[ok] = s[ok] / (n[ok] * h[ok])
    return n, m, h


def tag(d):
    return NICE.get(d["meta"]["model"], d["meta"]["model"])


def save(fig, name):
    fig.savefig(name + ".png", dpi=200); fig.savefig(name + ".pdf"); print("saved %s.png / .pdf" % name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default="merged4")
    ap.add_argument("--ref", default="Sibyll23e")
    a = ap.parse_args()
    ds = [load(f) for f in sorted(glob.glob(os.path.join(a.merged, "*_O16.npz")))]
    ds.sort(key=lambda d: ORDER.index(d["meta"]["model"]) if d["meta"]["model"] in ORDER else 99)
    if not ds:
        raise SystemExit("no *_O16.npz files in %s" % a.merged)
    ref = [d for d in ds if d["meta"]["model"] == a.ref]

    print("=== Fiducial acceptance A = N(sel)/N(generated) vs the value quoted in the ATLAS paper ===")
    print("%-18s %10s %9s %8s %9s %8s" % ("model", "events", "A (ours)", "+-", "A (ATLAS)", "pull"))
    for d in ds:
        m = d["meta"]; A = m["n_sel"] / m["n_gen"]; eA = np.sqrt(A * (1 - A) / m["n_gen"])
        q = ATLAS_A.get(m["model"])
        print("%-18s %10d %9.4f %8.4f %9s %8s" % (m["model"], m["n_gen"], A, eA,
              "%.3f" % q if q else "-", "%.1f" % ((A - q) / eA) if q else "-"))
    print("(a pull of a few units can come from a different generator version or tune than ATLAS used; the quoted A has no uncertainty)")

    print("\n=== Mean charged multiplicity per selected event and lepton share (e, mu among primaries) ===")
    for d in ds:
        m = d["meta"]
        x, y, _ = eta_dist(d); _, y2, _ = eta_dist(d, nolep=True)
        print("%-18s <n_ch>=%6.2f  lepton share=%.3f %%   max |eta density change| without leptons: %.2f %%" %
              (m["model"], m["n_part"] / m["n_sel"], 100 * m["n_lep"] / m["n_part"], 100 * np.max(np.abs(y2 / y - 1))))

    # ---- eta ----
    fig, ax = plt.subplots(2, 1, figsize=(7, 7.5), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for d in ds:
        x, y, ey = eta_dist(d); c = COLOR[d["meta"]["model"]]
        ax[0].errorbar(x, y, yerr=ey, fmt=".-", ms=3, lw=1, color=c, label=tag(d))
        if ref:
            xr, yr, _ = eta_dist(ref[0]); ax[1].plot(x, y / yr, ".-", ms=3, lw=1, color=c)
    ax[0].set_ylabel(r"$\frac{1}{N_{\mathrm{ev}}}\frac{\mathrm{d}N_{\mathrm{ch}}}{\mathrm{d}\eta}$")
    ax[0].legend(loc="lower center", ncol=2, fontsize=10); ax[0].set_ylim(2, None)
    ax[1].set_ylabel("ratio to %s" % NICE.get(a.ref, a.ref)); ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_xlabel(r"$\eta$ (ATLAS lab frame, + = oxygen side)")
    ax[0].set_title(r"p+O, $\sqrt{s_{NN}}=9.62$ TeV, generator level, $p_T>0.5$ GeV", fontsize=11)
    fig.tight_layout(); save(fig, "m4_models_eta")

    # ---- pT ----
    fig, ax = plt.subplots(2, 1, figsize=(7, 7.5), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for d in ds:
        x, y, ey, c_ = pt_dist(d); col = COLOR[d["meta"]["model"]]
        ok = c_ >= 10                                        # show only bins with at least 10 entries
        ax[0].errorbar(x[ok], y[ok], yerr=ey[ok], fmt=".-", ms=3, lw=1, color=col, label=tag(d))
        if ref:
            xr, yr, _, cr = pt_dist(ref[0]); k = ok & (cr >= 10)
            ax[1].plot(x[k], y[k] / yr[k], ".-", ms=3, lw=1, color=col)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_ylabel(r"$\frac{1}{N_{\mathrm{ev}}}\frac{1}{2\pi p_T}\frac{\mathrm{d}^2N}{\mathrm{d}\eta\,\mathrm{d}p_T}$ [GeV$^{-2}$]")
    ax[0].legend(loc="lower left", fontsize=10)
    ax[1].set_ylabel("ratio to %s" % NICE.get(a.ref, a.ref)); ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_ylim(0, 3); ax[1].set_xlabel(r"$p_T$ [GeV]  (bins with $\geq 10$ entries)")
    fig.tight_layout(); save(fig, "m4_models_pt")

    # ---- n_ch ----
    fig, ax = plt.subplots(2, 1, figsize=(7, 7.5), gridspec_kw=dict(height_ratios=[3, 1.4]), sharex=True)
    for d in ds:
        x, y, ey, c_ = nch_dist(d); col = COLOR[d["meta"]["model"]]
        ok = (c_ >= 10) & (x <= 200)
        ax[0].plot(x[ok], y[ok], "-", lw=1.2, color=col, label=tag(d))
        if ref:
            xr, yr, _, cr = nch_dist(ref[0]); k = ok & (cr >= 10)
            ax[1].plot(x[k], y[k] / yr[k], "-", lw=1, color=col)
    ax[0].set_yscale("log"); ax[0].set_ylabel(r"$\frac{1}{N_{\mathrm{ev}}}\frac{\mathrm{d}N_{\mathrm{ev}}}{\mathrm{d}n_{\mathrm{ch}}}$")
    ax[0].legend(fontsize=10)
    ax[1].set_ylabel("ratio to %s" % NICE.get(a.ref, a.ref)); ax[1].axhline(1, color="k", lw=0.6)
    ax[1].set_ylim(0, 3); ax[1].set_xlabel(r"$n_{\mathrm{ch}}$ (bins with $\geq 10$ events)")
    fig.tight_layout(); save(fig, "m4_models_nch")

    # ---- <pT> vs n_ch ----
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for d in ds:
        n, m, h = meanpt(d); k = (h >= 50) & (n <= 200)
        ax.plot(n[k], m[k], "-", lw=1.2, color=COLOR[d["meta"]["model"]], label=tag(d))
    ax.set_xlabel(r"$n_{\mathrm{ch}}$ (bins with $\geq 50$ events)"); ax.set_ylabel(r"$\langle p_T\rangle$ [GeV]")
    ax.legend(fontsize=10); fig.tight_layout(); save(fig, "m4_models_meanpt")


if __name__ == "__main__":
    main()
