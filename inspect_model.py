"""
Standalone script: load a trained VSSM checkpoint and inspect it.

Usage (run from the project root):
    python inspect_model.py --dataset aid
    python inspect_model.py --dataset aid --checkpoint path/to/model_best.pth.tar
    python inspect_model.py --dataset aid --device cuda

It:
  1. rebuilds VSSM from configs/<dataset>_scratch_config.py (same config used in training),
  2. loads the checkpoint state_dict,
  3. prints param count + recorded metrics (epoch / best F1) if present,
  4. runs one forward pass and prints SAD-CI gamma stats (overall + per stage).

NOTE:
  - The config in *_scratch_config.py must match the checkpoint architecture
    (especially sad_ci / forward_type / mlp_ratio), otherwise strict load fails.
  - 'v05' / 'v2' scans use triton CUDA kernels, so they need --device cuda
    (the default already auto-selects cuda when available).
"""
import os
import sys
import glob
import argparse
import importlib

import torch


def load_config(dataset):
    mod = importlib.import_module(f"configs.{dataset.lower()}_scratch_config")
    getter = getattr(mod, f"get_{dataset.lower()}_from_scratch_config")
    return getter()


def build_model(cfg):
    model_module = importlib.import_module("models.vmamba")
    ModelClass = getattr(model_module, cfg["model"]["name"])
    return ModelClass(**cfg["model"]["params"])


def find_checkpoint(cfg, explicit):
    if explicit:
        return explicit
    base = os.path.join(cfg["training"]["checkpoint_dir"], cfg["training"]["experiment_name"])
    cands = sorted(glob.glob(os.path.join(base, "*", "checkpoints", "model_best.pth.tar")))
    if not cands:
        cands = sorted(glob.glob(os.path.join(base, "*", "checkpoints", "last_checkpoint.pth.tar")))
    return cands[-1] if cands else None


def load_state_dict(ckpt_path, model, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state", ckpt))
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    try:
        model.load_state_dict(sd, strict=True)
        print("[ckpt] state_dict loaded (strict=True)")
    except RuntimeError as e:
        print(f"[warn] strict load failed ({e}). Falling back to strict=False.")
        model.load_state_dict(sd, strict=False)
    return ckpt


def gamma_stats(t):
    t = t.flatten()
    return {
        "mean": t.mean().item(),
        "std": t.std().item(),
        "min": t.min().item(),
        "max": t.max().item(),
        "lt025": (t < 0.25).float().mean().item(),
        "mid": ((t >= 0.25) & (t <= 0.75)).float().mean().item(),
        "gt075": (t > 0.75).float().mean().item(),
    }


def fmt(s):
    return (f"mean={s['mean']:.4f} std={s['std']:.4f} min={s['min']:.4f} max={s['max']:.4f} | "
            f"<0.25:{s['lt025'] * 100:5.1f}%  0.25-0.75:{s['mid'] * 100:5.1f}%  >0.75:{s['gt075'] * 100:5.1f}%")


def collect_gamma(model):
    """Return (overall_gamma_tensor_or_None, [per_stage_tensor_or_None, ...])."""
    per_stage = []
    layers = getattr(model, "layers", None)
    has_sadci = False

    if layers is not None:
        for layer in layers:
            g = []
            for m in layer.modules():
                if type(m).__name__ == "SADCI" and hasattr(m, "last_gamma"):
                    has_sadci = True
                    g.append(m.last_gamma.flatten())
            per_stage.append(torch.cat(g) if g else None)

    if not has_sadci:
        g = []
        for m in model.modules():
            if type(m).__name__ == "SADCI" and hasattr(m, "last_gamma"):
                has_sadci = True
                g.append(m.last_gamma.flatten())
        if g:
            return torch.cat(g), []
        return None, []

    valid = [t for t in per_stage if t is not None]
    return (torch.cat(valid) if valid else None), per_stage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aid", help="dataset name (aid / nwpu / ucmerced)")
    ap.add_argument("--checkpoint", default=None, help="path to .pth.tar; auto-detected if omitted")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for the forward pass; 'v05'/'v2' scans require CUDA (triton kernels)")
    ap.add_argument("--size", type=int, default=None, help="override input size (default: cfg img_size)")
    args = ap.parse_args()

    cfg = load_config(args.dataset)
    mp = cfg["model"]["params"]
    print(f"[config] dataset={args.dataset} experiment={cfg['training']['experiment_name']}")
    print(f"[config] forward_type={mp.get('forward_type')} sad_ci={mp.get('sad_ci')} "
          f"sad_ci_mlp_ratio={mp.get('sad_ci_mlp_ratio')}")

    ckpt_path = find_checkpoint(cfg, args.checkpoint)
    if not ckpt_path or not os.path.isfile(ckpt_path):
        print("[error] no checkpoint found. Pass --checkpoint explicitly.")
        sys.exit(1)
    print(f"[checkpoint] {ckpt_path}")

    model = build_model(cfg)
    model.to(args.device)
    ckpt = load_state_dict(ckpt_path, model, args.device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {type(model).__name__} | params: {n_params:,}")
    print(f"[ckpt meta] epoch={ckpt.get('epoch', '?')} "
          f"best_val_f1={ckpt.get('best_val_f1_score', ckpt.get('best_metric', '?'))}")

    size = args.size or cfg["data"].get("img_size", 224)
    dummy = torch.randn(1, 3, size, size, device=args.device)
    print(f"[forward] running one forward pass on random input (1,3,{size},{size}) ...")
    with torch.no_grad():
        model(dummy)

    overall, per_stage = collect_gamma(model)
    if overall is None:
        print("[gamma] model has no SAD-CI module (sad_ci is off in this config).")
    else:
        print("[gamma] overall :", fmt(gamma_stats(overall)))
        for i, t in enumerate(per_stage):
            if t is not None:
                print(f"[gamma] stage {i}  :", fmt(gamma_stats(t)))

    print("\nInterpretation:")
    print("  >0.75 ~100% -> gate collapses to VSS-only (ChannelMLP ignored)")
    print("  <0.25 ~100% -> gate collapses to ChannelMLP-only")
    print("  mean~0.5 & std~0 -> gate stuck at 0.5, not learning")


if __name__ == "__main__":
    main()
