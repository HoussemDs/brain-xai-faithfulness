"""Recompute region-masking BH-FDR pooled over all region-by-fill tests (48 regions x 2 fills = 96).

This follows the pre-registered multiple-comparison scope (BH across all tests). The previous
per-fill values are kept in `fdr_p_perfill`; `fdr_p` is overwritten with the pooled values.

Usage:
    python code/pool_region_bh.py results/region_masking_all_pvalues.csv results_revised_null/region_masking_merged.csv
"""
import csv
import sys


def bh(pvals):
    n = len(pvals)
    order = sorted(range(n), key=lambda i: pvals[i])
    q = [0.0] * n
    running = 1.0
    for k in range(n - 1, -1, -1):
        i = order[k]
        running = min(running, pvals[i] * n / (k + 1))
        q[i] = running
    return q


def process(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for r in rows:
        if "fdr_p_perfill" not in r:
            r["fdr_p_perfill"] = r["fdr_p"]
    q = bh([float(r["raw_p"]) for r in rows])
    for r, qq in zip(rows, q):
        r["fdr_p"] = f"{qq:.6g}"
    out_fields = fields + ([] if "fdr_p_perfill" in fields else ["fdr_p_perfill"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(rows)
    survivors = [(r["region"], r["fill"], r["raw_p"], r["fdr_p"]) for r in rows if float(r["fdr_p"]) < 0.05]
    print(f"{path}: tests={len(rows)} survive_q05_pooled={len(survivors)}")
    for s in survivors:
        print("  ", s)


if __name__ == "__main__":
    for p in sys.argv[1:]:
        process(p)
