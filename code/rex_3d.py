"""3D ReX: causal occlusion responsibility maps for volumetric classifiers.

Each coarse box of the volume is occluded in turn. The responsibility of a box is
the drop in the target-class probability when that box is removed. The coarse
grid is upsampled to the input volume with trilinear interpolation.

Reference: Navaratnarajah et al., arXiv 2502.12181.
Grid and fill value are fixed in the pre-specified plan (5x6x5, zero fill).
"""

import numpy as np
import torch
import torch.nn.functional as F


def target_probability(model, x, target):
    # x: (B, 1, D, H, W). Returns (B,) probability of the target class.
    logits = model(x)
    return torch.softmax(logits, dim=-1)[:, target]


def box_slices(size, grid):
    # Split one axis into `grid` contiguous ranges, as even as possible.
    edges = np.linspace(0, size, grid + 1).round().astype(int)
    return [slice(edges[i], edges[i + 1]) for i in range(grid)]


@torch.no_grad()
def rex_responsibility(model, x, target, grid=(5, 6, 5), fill=0.0, batch_size=16):
    """Return a voxel-level responsibility map of shape (D, H, W).

    x: tensor (1, 1, D, H, W) on the model's device.
    """
    model.eval()
    _, _, D, H, W = x.shape
    p0 = target_probability(model, x, target).item()

    boxes = []
    for sd in box_slices(D, grid[0]):
        for sh in box_slices(H, grid[1]):
            for sw in box_slices(W, grid[2]):
                boxes.append((sd, sh, sw))

    drops = np.zeros(len(boxes), dtype=np.float32)
    for start in range(0, len(boxes), batch_size):
        chunk = boxes[start:start + batch_size]
        occluded = x.repeat(len(chunk), 1, 1, 1, 1)
        for b, (sd, sh, sw) in enumerate(chunk):
            occluded[b, :, sd, sh, sw] = fill
        probs = target_probability(model, occluded, target).cpu().numpy()
        drops[start:start + len(chunk)] = p0 - probs

    coarse = torch.from_numpy(drops).reshape(grid[0], grid[1], grid[2])
    coarse = coarse[None, None].float()
    # Box order is D-major, then H, then W, which matches reshape above.
    full = F.interpolate(coarse, size=(D, H, W), mode="trilinear", align_corners=False)
    return full[0, 0].cpu().numpy()
