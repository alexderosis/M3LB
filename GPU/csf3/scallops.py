"""Scallop diagnostics on the INTERIOR of the ice face, for GPU/src/ice_melting.cu snapshots.

usage: python3 scallops.py <thickness.txt> [out.png]   (or several files: a time series)

With the ice standing clear of the lids its corners round off by ~100 cells, so
a running-mean detrend reads the rounded ends as "waves" and also removes any
wavelength longer than its window -- where scallops of 0.2-0.3 H would sit.
Here the ends are cut (the outer eighth each side) and only a QUADRATIC taper
is removed, so every wavelength from ~10 cells to the window length survives;
crests are maxima of the 9-cell-smoothed residual, and the spectrum names the
dominant wavelength.
"""
import sys
import numpy as np

files = [a for a in sys.argv[1:] if a.endswith(".txt")]
png = next((a for a in sys.argv[1:] if a.endswith(".png")), None)
rows = []
for path in files:
    h = np.loadtxt(path)
    if h.ndim == 1:
        h = h[:, None]
    H = h.shape[0]
    prof = h.mean(axis=1)
    lo, hi = H // 8, H - H // 8
    y = np.arange(lo, hi) + 1.0
    p = prof[lo:hi]
    c = np.polyfit(y, p, 2)
    res = p - np.polyval(c, y)
    k = 9
    sm = np.convolve(np.pad(res, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    crest = [i for i in range(1, len(sm) - 1) if sm[i] > sm[i - 1] and sm[i] >= sm[i + 1]]
    spec = np.abs(np.fft.rfft(res * np.hanning(len(res)))) ** 2
    freq = np.fft.rfftfreq(len(res))
    order = np.argsort(spec[1:])[::-1][:3] + 1
    lam = [round(1.0 / freq[j], 1) for j in order]
    tag = path.split("_")[-1].replace(".txt", "")
    print(f"{tag:>12}: mean h {prof.mean():7.2f}  ends {prof[:H//16].mean():6.2f}/{prof[-H//16:].mean():6.2f}"
          f"  interior residual rms {res.std():.3f} p-p {np.ptp(res):.2f}  crests {len(crest):2d}"
          f"  spacing {list(np.diff([int(y[i]) for i in crest]))}  dominant lambda {lam}")
    rows.append((tag, y, res, sm, crest, prof))

if png and rows:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=False)
    for tag, y, res, sm, crest, prof in rows:
        ax[0].plot(prof, np.arange(1, len(prof) + 1), lw=1, label=tag)
        ax[1].plot(sm, y, lw=1, label=tag)
    ax[0].set_xlabel("ice thickness h (cells)"); ax[0].set_ylabel("height y (cells)")
    ax[1].axvline(0, lw=0.5, color="k")
    ax[1].set_xlabel("interior h minus quadratic taper, 9-cell smoothed")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(png, dpi=100)
    print("wrote", png)
