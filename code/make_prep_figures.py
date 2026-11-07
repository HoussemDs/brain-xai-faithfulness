"""Figures for the preprocessing report. Run from the project root:

    python code/make_prep_figures.py

Reads data/manifest.csv, full_ds/scans (raw NIfTI, local only) and
prep_output/preproc (preprocessed volumes, local only). Writes PNGs to
results/figures/.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST = list(csv.DictReader(open(ROOT / "data" / "manifest.csv")))
EXAMPLES = ["PS14_129", "PS14_105", "PS17_035", "PS15_085"]  # fixed, so reruns match


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def fig_cohort():
    ages = np.array([float(r["age"]) for r in MANIFEST])
    med = float(np.median(ages))
    older = ages > med  # proxy task: age dichotomized at the cohort median (older = 1)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(ages[~older], bins=12, alpha=0.7, label="younger (0)")
    ax[0].hist(ages[older], bins=12, alpha=0.7, label="older (1)")
    ax[0].axvline(med, color="k", ls=":", label=f"median {med:.2f} y")
    ax[0].set_xlabel("age (years)")
    ax[0].set_ylabel("children")
    ax[0].legend()
    ax[1].bar(["younger", "older"], [int(np.sum(~older)), int(np.sum(older))], color=["C0", "C1"])
    ax[1].set_ylabel("children")
    ax[1].set_title(f"n = {len(MANIFEST)} children, one scan each")
    save(fig, "01_cohort_age_group.png")


def fig_examples():
    fig, ax = plt.subplots(len(EXAMPLES), 2, figsize=(6, 3.2 * len(EXAMPLES)))
    for i, sid in enumerate(EXAMPLES):
        raw = np.asanyarray(nib.load(ROOT / "full_ds" / "scans" / f"{sid}.nii.gz").dataobj).astype(np.float32)
        raw_mid = raw[:, :, raw.shape[2] // 2].T[::-1]
        prep = np.load(ROOT / "prep_output" / "preproc" / f"{sid}.npy")
        prep_mid = prep[:, :, prep.shape[2] // 2].T[::-1]
        ax[i, 0].imshow(raw_mid, cmap="gray")
        ax[i, 1].imshow(prep_mid, cmap="gray")
        ax[i, 0].set_ylabel(sid)
        for a in ax[i]:
            a.set_xticks([])
            a.set_yticks([])
    ax[0, 0].set_title("native T1, axial")
    ax[0, 1].set_title("preprocessed, MNI 160 x 192 x 160")
    save(fig, "02_raw_vs_preprocessed.png")


def fig_intensity_and_mask():
    frac, mx = [], []
    for r in MANIFEST:
        v = np.load(ROOT / "prep_output" / "preproc" / f"{r['scan_id']}.npy")
        frac.append(float((v > 0).mean()))
        mx.append(float(v.max()))
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(frac, bins=20)
    ax[0].set_xlabel("fraction of voxels > 0 (brain plus margin)")
    ax[0].set_ylabel("scans")
    ax[1].hist(mx, bins=20)
    ax[1].set_xlabel("max intensity after normalization")
    save(fig, "03_mask_and_intensity.png")


def fig_features():
    feats = np.stack([np.load(ROOT / "prep_output" / "features" / f"{r['scan_id']}.npy") for r in MANIFEST])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    ax[0].hist(feats.ravel(), bins=80)
    ax[0].set_xlabel("feature value (all scans, all 9600 features)")
    ax[0].set_ylabel("count")
    ax[1].plot(np.linalg.norm(feats, axis=1), "o", ms=3)
    ax[1].set_xlabel("scan index (manifest order)")
    ax[1].set_ylabel("feature vector norm")
    save(fig, "04_features.png")


def fig_qc_sheet():
    """Mid-sagittal slice of every preprocessed scan, with the QC status from data/qc_status.csv."""
    status = {r["scan_id"]: r for r in csv.DictReader(open(ROOT / "data" / "qc_status.csv"))}
    n = len(MANIFEST)
    cols = 12
    rows_n = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows_n, cols, figsize=(cols * 1.6, rows_n * 1.7))
    for a in ax.flat:
        a.axis("off")
    for a, r in zip(ax.flat, MANIFEST):
        sid = r["scan_id"]
        v = np.load(ROOT / "prep_output" / "preproc" / f"{sid}.npy")
        a.imshow(v[v.shape[0] // 2].T[::-1], cmap="gray")
        failed = status[sid]["qc_status"] == "fail"
        a.set_title(sid + (" FAIL" if failed else ""), fontsize=7, color="red" if failed else "black")
    fig.suptitle("Mid-sagittal slice, preprocessed, all manifest scans (QC in data/qc_status.csv)", fontsize=9)
    save(fig, "05_qc_contact_sheet.png")


def fig_group_average():
    """Mean of all preprocessed volumes in MNI space. Sharp = registration is consistent
    across the cohort; a blurry smear would mean misalignment. Cohort-level QC."""
    vols = [np.load(ROOT / "prep_output" / "preproc" / f"{r['scan_id']}.npy") for r in MANIFEST]
    mean = np.mean(vols, axis=0)
    x, y, z = (s // 2 for s in mean.shape)
    views = [
        ("axial", mean[:, :, z].T[::-1]),
        ("coronal", mean[:, y, :].T[::-1]),
        ("sagittal", mean[x, :, :].T[::-1]),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    for a, (name, img) in zip(ax, views):
        a.imshow(img, cmap="gray")
        a.set_title(name)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(f"Group-average brain, mean of all {len(MANIFEST)} preprocessed volumes (MNI space)", fontsize=10)
    save(fig, "06_group_average_brain.png")


def fig_tissue_intensity():
    """Per-scan intensity histogram of brain voxels, averaged over the cohort. The two humps
    are gray and white matter. Split by mask-fraction group to show tissue contrast is
    consistent across both groups."""
    edges = np.linspace(0.05, 2.0, 120)
    centers = 0.5 * (edges[:-1] + edges[1:])
    low, high = [], []
    for r in MANIFEST:
        v = np.load(ROOT / "prep_output" / "preproc" / f"{r['scan_id']}.npy")
        nz = v[v > 0]
        h, _ = np.histogram(nz, bins=edges, density=True)
        (low if (v != 0).mean() < 0.6 else high).append(h)
    fig, ax = plt.subplots(figsize=(7, 4))
    for group, label, color in [(high, "higher mask-fraction group", "C0"),
                                (low, "lower mask-fraction group", "C1")]:
        m = np.mean(group, axis=0)
        ax.plot(centers, m, color=color, label=f"{label} (n={len(group)})")
        ax.fill_between(centers, m, alpha=0.15, color=color)
    ax.axvline(1.0, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("normalized T1 intensity (brain voxels)")
    ax.set_ylabel("density (cohort mean)")
    ax.set_title("Brain-voxel intensity distribution: gray-matter and white-matter peaks")
    ax.legend()
    save(fig, "07_tissue_intensity_hist.png")


if __name__ == "__main__":
    fig_cohort()
    fig_examples()
    fig_intensity_and_mask()
    fig_features()
    fig_qc_sheet()
    fig_group_average()
    fig_tissue_intensity()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
