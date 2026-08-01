"""
plot_results.py — Generate ablation study figures
NTIRE 2026 CABR Challenge | Team CV SVNIT

Produces two publication-quality figures from the reported ablation numbers:
  1. results/ablation_metrics.png  — grouped bar chart (PSNR / SSIM / LPIPS)
  2. results/ablation_convergence.png — per-epoch PSNR curves

Usage:
    python ablation/plot_results.py

Requires: matplotlib, numpy  (pip install matplotlib numpy)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Reported best-checkpoint results (from Kaggle execution logs)
# ---------------------------------------------------------------------------
VARIANTS = [
    "E1: Full HAFT",
    "E2: −Refinement",
    "E3: −CoC Map",
    "E4: −Pos Map",
]
COLORS = ["#2B6CB0", "#D69E2E", "#2C7A7B", "#C53030"]

PSNR  = [31.13, 31.11, 31.09, 24.48]
SSIM  = [0.9292, 0.9309, 0.9297, 0.7497]
LPIPS = [0.1160, 0.1145, 0.1144, 0.2685]

# Per-epoch validation PSNR (10 epochs each)
EPOCH_PSNR = {
    "E1: Full HAFT":    [30.36, 30.77, 30.95, 31.02, 31.07, 31.03, 31.05, 31.11, 31.13, 31.13],
    "E2: −Refinement":  [30.33, 30.77, 30.84, 30.93, 31.05, 31.04, 31.05, 31.10, 31.11, 31.10],
    "E3: −CoC Map":     [30.23, 30.65, 30.85, 30.91, 30.93, 31.01, 31.07, 31.09, 31.09, 31.09],
    "E4: −Pos Map":     [19.29, 23.19, 23.93, 24.10, 24.22, 24.36, 24.40, 24.47, 24.47, 24.48],
}

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Grouped bar chart — PSNR / SSIM / LPIPS
# ---------------------------------------------------------------------------
def plot_bar_chart():
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="white")
    fig.subplots_adjust(wspace=0.35)

    metrics = [
        ("PSNR (dB) ↑", PSNR,  (22, 32.5)),
        ("SSIM ↑",       SSIM,  (0.65, 0.97)),
        ("LPIPS ↓",      LPIPS, (0.05, 0.32)),
    ]

    x = np.arange(len(VARIANTS))
    for ax, (title, vals, ylim) in zip(axes, metrics):
        bars = ax.bar(x, vals, color=COLORS, width=0.55, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8, color="#1A202C")
        ax.set_xticks(x)
        ax.set_xticklabels(VARIANTS, rotation=20, ha="right", fontsize=8.5, color="#4A5568")
        ax.set_ylim(*ylim)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#CBD5E0")
        ax.tick_params(colors="#718096", length=3)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (ylim[1] - ylim[0]) * 0.01,
                f"{v:.4f}" if v < 2 else f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#2D3748",
            )

    fig.savefig(OUT / "ablation_metrics.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: results/ablation_metrics.png")


# ---------------------------------------------------------------------------
# Figure 2: Per-epoch convergence curves
# ---------------------------------------------------------------------------
def plot_convergence():
    epochs = list(range(1, 11))
    styles = [
        dict(color="#2B6CB0", linestyle="-",  marker="o", markersize=4),
        dict(color="#D69E2E", linestyle="--", marker="s", markersize=4),
        dict(color="#2C7A7B", linestyle="-.", marker="^", markersize=4),
        dict(color="#C53030", linestyle=":",  marker="D", markersize=4),
    ]

    fig, ax = plt.subplots(figsize=(7, 4), facecolor="white")
    for (label, values), style in zip(EPOCH_PSNR.items(), styles):
        ax.plot(epochs, values, label=label, linewidth=1.8, **style)

    ax.set_xlabel("Epoch", fontsize=10, color="#4A5568")
    ax.set_ylabel("PSNR (dB)", fontsize=10, color="#4A5568")
    ax.set_xlim(1, 10)
    ax.set_ylim(17, 33)
    ax.set_xticks(epochs)
    ax.grid(color="#E2E8F0", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E0")
    ax.tick_params(colors="#718096", length=3)
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor="#CBD5E0", loc="lower right")

    fig.savefig(OUT / "ablation_convergence.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved: results/ablation_convergence.png")


if __name__ == "__main__":
    plot_bar_chart()
    plot_convergence()
