"""
run_ablation.py — HAFT Ablation Study Runner
NTIRE 2026 Controllable Aperture Bokeh Rendering (CABR) Challenge
Team: CV SVNIT

Runs 4 ablation experiments sequentially and saves a summary table.

Usage (from HAFT FACTSHEET/ directory):
    python ablation/run_ablation.py

Each experiment trains for the configured number of epochs and logs
PSNR / SSIM / LPIPS on the RealBokeh_3MP validation split.
Results are written to ablation/results/ablation_results.md.
"""

import os
import re
import sys
import glob
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment definitions
# Each entry maps to CLI flags consumed by train_haft_small.py (sys.argv checks)
#
#   --no_refinement  → disable depth-guided Refinement Head
#   --no_coc         → disable Circle-of-Confusion Map input
#   --no_pos         → disable Positional Map (sinusoidal spatial encoding)
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "id":    "E1",
        "name":  "Full HAFT (Ours)",
        "flags": [],                                          # all modules ON
    },
    {
        "id":    "E2",
        "name":  "w/o Refinement Head",
        "flags": ["--no_refinement"],                        # refinement OFF
    },
    {
        "id":    "E3",
        "name":  "w/o CoC Map",
        "flags": ["--no_coc"],                               # CoC map OFF
    },
    {
        "id":    "E4",
        "name":  "w/o Positional Map",
        "flags": ["--no_pos"],                               # positional map OFF
    },
]

# Where to write the markdown summary (relative to HAFT FACTSHEET/)
OUT_MD = Path("ablation/results/ablation_results.md")

# Regex to parse per-epoch metric lines logged by train_haft_small.py
# Example line: [E1_Full_HAFT] Ep  9/10 | Train Loss 0.0310 | Val PSNR 31.13 dB ...
METRIC_RE = re.compile(
    r"Val PSNR\s+([\d.]+)\s*dB.*?Val SSIM\s+([\d.]+).*?Val LPIPS\s+([\d.]+)"
)


def run_experiment(exp: dict) -> dict:
    """Train one ablation variant and return its best metrics."""
    print(f"\n{'=' * 60}")
    print(f"  {exp['id']} — {exp['name']}")
    print(f"  flags: {exp['flags'] or '(none — full model)'}")
    print(f"{'=' * 60}")

    # Fresh checkpoint for each variant (backbone weights are preserved by the
    # training script itself; only haft_large*.pth files are removed)
    for pth in glob.glob("checkpoints/haft_large*.pth"):
        os.remove(pth)

    cmd = [sys.executable, "train_haft_small.py"] + exp["flags"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    best = {"psnr": 0.0, "ssim": 0.0, "lpips": 1.0}

    with subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        encoding="utf-8",
    ) as proc:
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = METRIC_RE.search(line)
            if m:
                psnr, ssim, lpips = float(m.group(1)), float(m.group(2)), float(m.group(3))
                if psnr > best["psnr"]:
                    best = {"psnr": psnr, "ssim": ssim, "lpips": lpips}

    print(f"\n  → Best: PSNR {best['psnr']:.2f} dB | SSIM {best['ssim']:.4f} | LPIPS {best['lpips']:.4f}")
    return best


def write_markdown(results: list[dict]) -> None:
    """Write a clean markdown summary table."""
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)

    e1_psnr = results[0]["psnr"]  # delta reference

    lines = [
        "# Ablation Results — HAFT (NTIRE 2026 CABR)\n",
        "| # | Variant | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ | ΔPSNR |",
        "|---|---------|:-----------:|:------:|:-------:|:-----:|",
    ]
    for exp, res in zip(EXPERIMENTS, results):
        delta = res["psnr"] - e1_psnr
        delta_str = "—" if exp["id"] == "E1" else f"{delta:+.2f}"
        lines.append(
            f"| {exp['id']} | {exp['name']} "
            f"| {res['psnr']:.2f} | {res['ssim']:.4f} | {res['lpips']:.4f} | {delta_str} |"
        )

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nResults saved → {OUT_MD}")


def main():
    # Must be run from HAFT FACTSHEET/ so relative paths resolve correctly
    here = Path(__file__).parent
    os.chdir(here.parent)  # cd to HAFT FACTSHEET/

    results = []
    for exp in EXPERIMENTS:
        results.append(run_experiment(exp))

    print("\n" + "=" * 60)
    print("  ABLATION COMPLETE")
    print("=" * 60)
    for exp, res in zip(EXPERIMENTS, results):
        print(f"  {exp['id']:3s} {exp['name']:30s} | PSNR {res['psnr']:.2f} | SSIM {res['ssim']:.4f} | LPIPS {res['lpips']:.4f}")

    write_markdown(results)


if __name__ == "__main__":
    main()
