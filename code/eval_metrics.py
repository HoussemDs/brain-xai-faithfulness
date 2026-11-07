"""Evaluation metrics for attribution maps and the statistics around them.

Covers: deletion AUC (faithfulness), trained vs randomized-weight similarity
(sanity check), bootstrap CIs resampled by child, permutation p-values, and
Benjamini-Hochberg FDR correction. All choices follow the pre-specified plan.
"""

import copy

import numpy as np
import torch
from scipy.stats import spearmanr


@torch.no_grad()
def deletion_auc(model, x, attribution, target, fractions=(0.05, 0.10, 0.20, 0.30), fill=0.0):
    """Zero out the top-k voxels by attribution and track the target probability.

    x: tensor (1, 1, D, H, W). attribution: numpy array (D, H, W).
    Returns (auc, curve) where curve is a list of (fraction, probability).
    The fraction 0 point (no deletion) is included in the curve.
    """
    model.eval()
    flat = attribution.ravel()
    order = np.argsort(-flat)
    n = flat.size
    shape = x.shape[2:]

    fractions = [0.0] + list(fractions)
    curve = []
    for frac in fractions:
        k = int(round(frac * n))
        occluded = x.clone()
        if k > 0:
            idx = np.unravel_index(order[:k], shape)
            occluded[0, 0, idx[0], idx[1], idx[2]] = fill
        prob = torch.softmax(model(occluded), dim=-1)[0, target].item()
        curve.append((frac, prob))

    # Trapezoidal area under the curve, written out to avoid numpy version differences.
    auc = 0.0
    for (f0, p0), (f1, p1) in zip(curve[:-1], curve[1:]):
        auc += (f1 - f0) * (p0 + p1) / 2.0
    auc /= fractions[-1] - fractions[0]
    return auc, curve


@torch.no_grad()
def insertion_auc(model, x, attribution, target, fractions=(0.05, 0.10, 0.20, 0.30), fill=0.0):
    """Insert the top-k voxels by attribution into a blank volume; track the probability.

    Mirror of deletion_auc. A faithful map raises the target probability quickly as
    its top voxels are added back, so a higher insertion AUC is better.
    """
    model.eval()
    flat = attribution.ravel()
    order = np.argsort(-flat)
    n = flat.size
    shape = x.shape[2:]

    fractions = [0.0] + list(fractions)
    curve = []
    for frac in fractions:
        k = int(round(frac * n))
        inserted = torch.full_like(x, fill)
        if k > 0:
            idx = np.unravel_index(order[:k], shape)
            inserted[0, 0, idx[0], idx[1], idx[2]] = x[0, 0, idx[0], idx[1], idx[2]]
        prob = torch.softmax(model(inserted), dim=-1)[0, target].item()
        curve.append((frac, prob))

    auc = 0.0
    for (f0, p0), (f1, p1) in zip(curve[:-1], curve[1:]):
        auc += (f1 - f0) * (p0 + p1) / 2.0
    auc /= fractions[-1] - fractions[0]
    return auc, curve


def randomize_model(model, seed=0):
    """Return a copy of the model with all weights re-initialized.

    Used for the model randomization test (Adebayo et al., 2018).
    Layers without reset_parameters are left as they are. BatchNorm running
    stats are reset as well, since BatchNorm.reset_parameters calls them.
    """
    rnd = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in rnd.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return rnd


def trained_vs_random_similarity(attr_trained, attr_random):
    """Spearman correlation between two attribution maps, over all voxels.

    Lower similarity under randomization is the sanity check passing.
    """
    r, _ = spearmanr(attr_trained.ravel(), attr_random.ravel())
    return float(r)


def bootstrap_ci_by_child(values, child_ids, n_boot=10000, alpha=0.05, seed=0):
    """Mean and percentile CI, resampling children (not scans) with replacement.

    values: 1D array, one entry per scan. child_ids: 1D array of the same length.
    Scans from the same child stay together in each resample.
    """
    values = np.asarray(values, dtype=float)
    child_ids = np.asarray(child_ids)
    children = np.unique(child_ids)
    groups = {c: values[child_ids == c] for c in children}

    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.choice(children, size=children.size, replace=True)
        means[i] = np.concatenate([groups[c] for c in picked]).mean()

    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(values.mean()), float(lo), float(hi)


def permutation_pvalue(group_a, group_b, n_perm=10000, seed=0):
    """Two-sided permutation test on the difference in means.

    Labels are shuffled across the pooled values. Returns (observed_diff, p).
    Report n_perm alongside the p-value in the results log.
    """
    group_a = np.asarray(group_a, dtype=float)
    group_b = np.asarray(group_b, dtype=float)
    observed = group_a.mean() - group_b.mean()

    pooled = np.concatenate([group_a, group_b])
    n_a = group_a.size
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        diff = pooled[:n_a].mean() - pooled[n_a:].mean()
        if abs(diff) >= abs(observed):
            count += 1
    # Add-one correction so p is never exactly zero.
    p = (count + 1) / (n_perm + 1)
    return float(observed), float(p)


def benjamini_hochberg(pvalues):
    """Return BH-adjusted p-values in the original order."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    # Enforce monotonicity from the largest p downward.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(ranked, 0.0, 1.0)
    return adjusted
