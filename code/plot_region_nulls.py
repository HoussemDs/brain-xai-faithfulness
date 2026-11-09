"""Two-row figure: registered null (200 scattered random regions) vs sensitivity null
(500 contiguous translated regions), each with mean and zero fill. Red = survives BH-FDR q=0.05
pooled over all 96 region-by-fill tests.

Usage:
    python code/plot_region_nulls.py --registered kaggle_output_prenatal_xai_96/results/region_masking_all_pvalues.csv \
        --sensitivity results_revised_null/region_masking_merged.csv --out paper/figures/10_region_nulls.png
"""
import argparse
import csv
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registered", required=True)
    ap.add_argument("--sensitivity", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    panels = [
        ("Registered null (200 random regions)", load(a.registered)),
        ("Sensitivity null (500 contiguous draws)", load(a.sensitivity)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
    for row, (title, rows) in enumerate(panels):
        for col, fill in enumerate(["mean", "zero"]):
            ax = axes[row][col]
            g = [r for r in rows if r["fill"] == fill]
            xs = [float(r["effect"]) for r in g]
            ys = [-math.log10(max(float(r["fdr_p"]), 1e-12)) for r in g]
            colors = ["red" if float(r["fdr_p"]) < 0.05 else "grey" for r in g]
            ax.scatter(xs, ys, c=colors, s=16)
            ax.axhline(-math.log10(0.05), ls="--", color="k", lw=0.8)
            ax.set_title(f"{title}\n{fill}-fill", fontsize=9)
            ax.set_xlabel("masking effect (probability drop)", fontsize=8)
            ax.set_ylabel("-log10 FDR p", fontsize=8)
            n_sig = sum(float(r["fdr_p"]) < 0.05 for r in g)
            ax.text(0.98, 0.02, f"survive q<0.05: {n_sig}/48 (pooled BH, 96 tests)", transform=ax.transAxes,
                    ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(a.out, dpi=160)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
