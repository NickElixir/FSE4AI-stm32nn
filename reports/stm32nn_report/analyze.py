#!/usr/bin/env python3
"""Host scores for the CNN board run, plus the figures for the report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.analyze_cnn_quantization_sensitivity import load_model_and_meta
from scripts.train_cnn import CNNImageDataset
from scripts.train_mlp import LocalMNISTDataset

from stm32nn import check_export, compare, quantize_per_channel, round_fp16

OUT = Path(__file__).resolve().parent
FIG = OUT / "figures"
CKPT = ROOT / "models/PyTorch/cnn_mnist_2conv_c32_fc2_128_drop0.25_wd1e-05_1_1_1.pth"
BOARD = OUT / "cnn_fp16_wonly_board.json"
CUBE = ROOT / "reports/mcu_n657/cnn_mnist_2conv_c32_fc2_128_drop0.25_wd1e-05_fp32.json"


def subset_indices(n_pool: int) -> list[int]:
    rng = np.random.default_rng(0)
    idx = rng.choice(n_pool, size=200, replace=False)
    idx.sort()
    return [int(i) for i in idx]


def logits_of(model, loader) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for images, labels in loader:
            zs.append(model(images).float().cpu())
            ys.append(labels.cpu())
    return torch.cat(zs), torch.cat(ys)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    board = json.loads(BOARD.read_text())
    cube = json.loads(CUBE.read_text())
    device = torch.device("cpu")
    model, _ = load_model_and_meta(CKPT, device)
    base = LocalMNISTDataset(ROOT / "data/mnist", train=False)
    idx = subset_indices(len(base))
    order = [s["index"] for s in board["samples"]]
    if order != idx:
        raise SystemExit("board sample order is not the seed-0 subset")
    loader = DataLoader(Subset(CNNImageDataset(base, 1), idx), batch_size=64, shuffle=False)
    z_fp32, labels = logits_of(model, loader)
    conv_w = model.features[0].weight.detach().float().cpu().clone()
    rounded = {name: round_fp16(p.detach().float().cpu()) for name, p in model.named_parameters()}
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.copy_(rounded[name])
    z_fp16, _ = logits_of(model, loader)
    z_mcu = torch.tensor([s["logits_mcu"] for s in board["samples"]], dtype=torch.float32)
    z_cube = torch.tensor([s["logits_mcu"] for s in cube["samples"]], dtype=torch.float32)
    if [s["index"] for s in cube["samples"]] != idx:
        raise SystemExit("Cube.AI sample order is not the seed-0 subset")

    rows = {}
    for name, z in (("fp16_host", z_fp16), ("helium", z_mcu), ("cubeai_fp32", z_cube)):
        mse, disagree, cosine = compare(z, z_fp32)
        rows[name] = {
            "logit_mse": mse,
            "argmax_disagreement": disagree,
            "mean_cosine": cosine,
            "label_accuracy": float((z.argmax(-1) == labels).float().mean()),
        }
    per_image = ((z_mcu - z_fp32) ** 2).sum(-1).numpy()
    per_image_cube = ((z_cube - z_fp32) ** 2).sum(-1).numpy()
    latency = np.array([s["time_us"] for s in board["samples"]], dtype=float) / 1000.0

    weights = [p.detach().float().cpu() for p in model.parameters()]
    sample = torch.zeros(1, 1, 28, 28)
    issues_ok = check_export(weights, sample)
    issues_bad = check_export([torch.zeros(257, 8)], torch.tensor([200.0]))

    # One conv weight, per-channel int8, just to record the scale of that rounding.
    q = quantize_per_channel(conv_w, axis=0)
    q_err = float((q - conv_w).pow(2).mean())

    summary = {
        "n": 200,
        "seed": 0,
        "board_file_mse": board["logit_mse_vs_host_fp32"],
        "board_top1": board["top1_accuracy"],
        "board_disagreement": board["top1_disagreement_vs_host_fp32"],
        "latency_ms_mean": float(latency.mean()),
        "rows": rows,
        "export_issues_on_fp32_sample": issues_ok,
        "export_issues_on_bad_example": issues_bad,
        "first_conv_int8_weight_mse": q_err,
        "fp32_label_accuracy": float((z_fp32.argmax(-1) == labels).float().mean()),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(per_image, bins=30, color="#1f4e79", edgecolor="white")
    ax.set_xlabel(r"Per-image squared logit gap, Helium vs PC FP32")
    ax.set_ylabel("Images")
    ax.set_title("CNN, 200 MNIST images, seed 0")
    fig.tight_layout()
    fig.savefig(FIG / "helium_gap_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(np.log10(per_image_cube), bins=30, color="#8c3a3a", edgecolor="white")
    ax.set_xlabel(r"log10 of per-image squared logit gap, Cube.AI FP32 vs PC")
    ax.set_ylabel("Images")
    ax.set_title("Same 200 images")
    fig.tight_layout()
    fig.savefig(FIG / "cubeai_gap_hist.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    names = ["Host FP16\nrounding", "Helium\non STM32", "Cube.AI\nFP32"]
    vals = [rows["fp16_host"]["logit_mse"], rows["helium"]["logit_mse"], rows["cubeai_fp32"]["logit_mse"]]
    ax.bar(names, vals, color=["#4c78a8", "#54a24b", "#e45756"])
    ax.set_yscale("log")
    ax.set_ylabel("Mean squared logit gap vs PC FP32")
    ax.set_title("CNN on the same 200 images")
    fig.tight_layout()
    fig.savefig(FIG / "mse_bars.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(latency, bins=25, color="#2a6f6f", edgecolor="white")
    ax.set_xlabel("Kernel time (ms)")
    ax.set_ylabel("Images")
    ax.set_title("Helium CNN, firmware cycle counter")
    fig.tight_layout()
    fig.savefig(FIG / "latency_hist.png", dpi=140)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
