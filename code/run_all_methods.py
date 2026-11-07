"""Run the pre-specified attribution methods on one preprocessed scan.

Methods (IDs from the plan, section 4.0). M1-M8 are the locked pre-specified set;
M9 is exploratory and logged as such in the decision log.
  M1 Grad-CAM, final conv layer      (captum LayerGradCam) -- the paper's method
  M2 Grad-CAM, earlier block         (captum LayerGradCam) -- resolution control
  M3 Saliency (plain gradient)       (captum Saliency)
  M4 Integrated Gradients            (captum IntegratedGradients)
  M5 SmoothGrad                      (captum NoiseTunnel, smoothgrad)
  M6 VarGrad                         (captum NoiseTunnel, vargrad)
  M7 LRP epsilon                     (zennit EpsilonPlusFlat)
  M8 Guided Backprop                 (captum) -- negative control (Adebayo et al.)
  M9 3D ReX, EXPLORATORY             (code/rex_3d.py) -- causal-occlusion cross-check

Usage:
  python code/run_all_methods.py --scan path/to/scan.npy --checkpoint path/to/sfcn.pt \
      --target 1 --layer <final_conv_module_name> --early_layer <earlier_block_name> \
      --out_dir results/scan01
  add --randomize to also compute maps on randomized weights (H4).

Expects a preprocessed volume saved as a float32 .npy of shape (D, H, W).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import (
    GuidedBackprop,
    IntegratedGradients,
    LayerAttribution,
    LayerGradCam,
    NoiseTunnel,
    Saliency,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_metrics import randomize_model  # noqa: E402
from rex_3d import rex_responsibility  # noqa: E402


def load_model(checkpoint, device):
    # The SFCN definition must come from the verified source (see plan, section 0,
    # "Pretrained SFCN weights and code: NOT YET VERIFIED"). Expected to live in
    # code/sfcn.py with a class SFCN(nn.Module) that outputs logits of shape (B, 2).
    from sfcn import SFCN  # noqa: E402

    model = SFCN()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state.get("state_dict", state))
    model.to(device).eval()
    return model


def get_layer(model, name):
    module = dict(model.named_modules()).get(name)
    if module is None:
        raise ValueError(f"layer '{name}' not found; list names with model.named_modules()")
    return module


def to_numpy_volume(attr):
    # Captum returns (1, C, D, H, W). Collapse channels with a sum of absolute values.
    arr = attr.detach().abs().sum(dim=1)[0]
    return arr.cpu().numpy().astype(np.float32)


def lrp_zennit(model, x, target):
    # Verify these zennit names against the installed version before the first run.
    from zennit.attribution import Gradient
    from zennit.composites import EpsilonPlusFlat

    composite = EpsilonPlusFlat()
    with Gradient(model=model, composite=composite) as attributor:
        # Seed relevance with a one-hot on the target class. Output shape is (1, 2).
        out = model(x)
        seed = torch.zeros_like(out)
        seed[0, target] = 1.0
        _, relevance = attributor(x, seed)
    return relevance


def compute_maps(model, x, target, layer_name, rex_grid, early_layer_name=None):
    timings = {}
    maps = {}

    def run(name, fn):
        start = time.perf_counter()
        maps[name] = fn()
        timings[name] = round(time.perf_counter() - start, 2)

    final_layer = get_layer(model, layer_name)

    def gradcam(layer):
        attr = LayerGradCam(model, layer).attribute(x, target=target)
        attr = LayerAttribution.interpolate(attr, x.shape[2:])
        return to_numpy_volume(attr)

    def saliency():
        return to_numpy_volume(Saliency(model).attribute(x, target=target, abs=False))

    def integrated_gradients():
        attr = IntegratedGradients(model).attribute(
            x, baselines=torch.zeros_like(x), target=target, n_steps=32
        )
        return to_numpy_volume(attr)

    def noise_tunnel(nt_type):
        attr = NoiseTunnel(Saliency(model)).attribute(
            x, nt_type=nt_type, nt_samples=20, stdevs=0.1, target=target
        )
        return to_numpy_volume(attr)

    def lrp():
        return to_numpy_volume(lrp_zennit(model, x, target))

    def rex():
        return rex_responsibility(model, x, target, grid=rex_grid)

    def guided_backprop():
        attr = GuidedBackprop(model).attribute(x, target=target)
        return to_numpy_volume(attr)

    run("M1_gradcam_final", lambda: gradcam(final_layer))
    if early_layer_name:
        run("M2_gradcam_early", lambda: gradcam(get_layer(model, early_layer_name)))
    run("M3_saliency", saliency)
    run("M4_integrated_gradients", integrated_gradients)
    run("M5_smoothgrad", lambda: noise_tunnel("smoothgrad"))
    run("M6_vargrad", lambda: noise_tunnel("vargrad"))
    run("M7_lrp", lrp)
    run("M8_guided_backprop", guided_backprop)
    run("M9_rex_exploratory", rex)
    return maps, timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--layer", required=True, help="final conv module name (M1)")
    parser.add_argument("--early_layer", default=None, help="earlier block module name (M2)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rex_grid", type=int, nargs=3, default=[5, 6, 5])
    parser.add_argument("--randomize", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    volume = np.load(args.scan).astype(np.float32)
    x = torch.from_numpy(volume)[None, None].to(device)
    x.requires_grad_(False)

    model = load_model(args.checkpoint, device)
    maps, timings = compute_maps(
        model, x, args.target, args.layer, tuple(args.rex_grid), args.early_layer
    )

    summary = {"timings_s": timings, "randomized": {}}
    for name, arr in maps.items():
        np.save(os.path.join(args.out_dir, f"{name}.npy"), arr)

    if args.randomize:
        from eval_metrics import trained_vs_random_similarity

        rnd_model = randomize_model(model, seed=0).to(device).eval()
        rnd_maps, rnd_timings = compute_maps(
            rnd_model, x, args.target, args.layer, tuple(args.rex_grid), args.early_layer
        )
        for name in maps:
            np.save(os.path.join(args.out_dir, f"{name}_randomized.npy"), rnd_maps[name])
            summary["randomized"][name] = {
                "spearman_trained_vs_random": trained_vs_random_similarity(
                    maps[name], rnd_maps[name]
                ),
                "timing_s": rnd_timings[name],
            }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
