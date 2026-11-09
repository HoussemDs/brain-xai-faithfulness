"""Merge the two region-masking halves, recompute BH-FDR over all regions per fill,
recompute method-vs-masking Spearman, and draw the revised region figure.

Usage:
    python code/merge_masking_halves.py \
        --half-a <kaggle output dir a>/results \
        --half-b <kaggle output dir b>/results \
        --out results_revised_null

Pure standard library except matplotlib (for the figure).
Permutation p-values for method-vs-masking Spearman are NOT spatially aware;
the 48 regions are spatially autocorrelated, so treat them as optimistic.
"""
import argparse
import csv
import math
import os
import random


def bh(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    prev = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        v = min(prev, pvals[i] * n / rank)
        q[i] = v
        prev = v
    return q


def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half-a", required=True)
    ap.add_argument("--half-b", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=10000)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # 1. Merge halves and recompute BH-FDR over all 48 regions per fill.
    rows = read(os.path.join(args.half_a, "region_masking_all_pvalues.csv")) + read(
        os.path.join(args.half_b, "region_masking_all_pvalues.csv"))
    for fill in sorted({r["fill"] for r in rows}):
        g = [r for r in rows if r["fill"] == fill]
        q = bh([float(r["raw_p"]) for r in g])
        for r, qq in zip(g, q):
            r["fdr_p"] = f"{qq:.6g}"
    out_regions = os.path.join(args.out, "region_masking_merged.csv")
    with open(out_regions, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for fill in sorted({r["fill"] for r in rows}):
        g = [r for r in rows if r["fill"] == fill]
        n_sig = sum(float(r["fdr_p"]) < 0.05 for r in g)
        print(f"fill={fill} regions={len(g)} survive_q05={n_sig} min_raw_p={min(float(r['raw_p']) for r in g):.4f}")

    # 2. Method-vs-masking Spearman on the zero-fill effect (48 regions).
    eff = {r["region"]: float(r["effect"]) for r in rows if r["fill"] == "zero"}
    scores = read(os.path.join(args.half_a, "method_region_scores.csv"))
    regs = [r["region"] for r in scores]
    y = [eff[r] for r in regs]
    methods = [k for k in scores[0] if k != "region"]
    rng = random.Random(0)
    corr_path = os.path.join(args.out, "method_vs_masking_correlation_merged.csv")
    with open(corr_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "spearman_attr_vs_masking", "perm_p_two_sided"])
        for m in methods:
            x = [float(r[m]) for r in scores]
            rho = spearman(x, y)
            perm = y[:]
            cnt = 0
            for _ in range(args.perms):
                rng.shuffle(perm)
                if abs(spearman(x, perm)) >= abs(rho):
                    cnt += 1
            p = (cnt + 1) / (args.perms + 1)
            w.writerow([m, f"{rho:.3f}", f"{p:.4f}"])
            print(f"{m:24s} rho={rho:+.3f} perm_p={p:.4f}")

    # 3. Revised region figure: effect vs -log10 FDR per fill; red = survives q=0.05.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fills = sorted({r["fill"] for r in rows})
    fig, axes = plt.subplots(1, len(fills), figsize=(5.2 * len(fills), 4.2), squeeze=False)
    for ax, fill in zip(axes[0], fills):
        g = [r for r in rows if r["fill"] == fill]
        xs = [float(r["effect"]) for r in g]
        ys = [-math.log10(max(float(r["fdr_p"]), 1e-12)) for r in g]
        colors = ["red" if float(r["fdr_p"]) < 0.05 else "grey" for r in g]
        ax.scatter(xs, ys, c=colors, s=18)
        ax.axhline(-math.log10(0.05), ls="--", color="k", lw=0.8)
        ax.set_title(f"{fill}-fill masking (BH over 48 regions)")
        ax.set_xlabel("masking effect (probability drop)")
        ax.set_ylabel("-log10 FDR p")
        for r, xx, yy in sorted(zip(g, xs, ys), key=lambda t: -t[2])[:3]:
            ax.annotate(r["region"][:22], (xx, yy), fontsize=6, xytext=(3, 3), textcoords="offset points")
    fig.tight_layout()
    fig_path = os.path.join(args.out, "region_masking_revised.png")
    fig.savefig(fig_path, dpi=160)
    print("wrote", out_regions, corr_path, fig_path)


if __name__ == "__main__":
    main()
