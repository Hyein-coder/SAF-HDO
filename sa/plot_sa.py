"""plot_sa.py -- render tornado + spider charts from SA_driver.m output.

Reads sa_tornado.csv and sa_oat.csv (written by SA_driver.m) and produces
tornado.png and spider.png. Only needs numpy + matplotlib (both present);
no openpyxl/Excel dependency.

Run:  python plot_sa.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def read_csv(name):
    with open(os.path.join(HERE, name), newline="") as f:
        return list(csv.DictReader(f))


def tornado():
    rows = read_csv("sa_tornado.csv")
    base = float(rows[0]["base_msp"])
    names = [r["parameter"] for r in rows]
    lo = np.array([float(r["low_msp"]) for r in rows])
    hi = np.array([float(r["high_msp"]) for r in rows])
    span = np.abs(hi - lo)
    order = np.argsort(span)                      # smallest -> largest (drawn bottom-up)
    names = [names[i] for i in order]
    lo, hi = lo[order], hi[order]

    fig, ax = plt.subplots(figsize=(8, 0.6 * len(names) + 1.5))
    y = np.arange(len(names))
    for i in range(len(names)):
        left, right = min(lo[i], hi[i]), max(lo[i], hi[i])
        ax.barh(y[i], left - base, left=base, color="#4C72B0", height=0.6)
        ax.barh(y[i], right - base, left=base, color="#C44E52", height=0.6)
    ax.axvline(base, color="k", lw=1.2, ls="--", label=f"Base MSP = {base:.3f} $/kg")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlabel("Minimum selling price of SAF ($/kg)")
    ax.set_title("Tornado: MSP sensitivity to techno-economic parameters")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = os.path.join(HERE, "tornado.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


def spider():
    rows = read_csv("sa_oat.csv")
    params = {}
    for r in rows:
        params.setdefault(r["parameter"], []).append((float(r["value"]), float(r["msp"])))

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, pts in params.items():
        pts.sort()
        v = np.array([p[0] for p in pts]); m = np.array([p[1] for p in pts])
        # express each parameter as % change from its central grid point
        mid = v[len(v) // 2]
        pct = (v - mid) / mid * 100 if mid != 0 else v
        ax.plot(pct, m, marker="o", label=name)
    ax.set_xlabel("Change in parameter from base (%)")
    ax.set_ylabel("Minimum selling price of SAF ($/kg)")
    ax.set_title("One-at-a-time sensitivity of MSP")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(HERE, "spider.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    tornado()
    spider()
