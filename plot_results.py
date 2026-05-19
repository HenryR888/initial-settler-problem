"""
Here we plot fancy training figures for our ippo-cpr rollout: 

Loads per-seed result files saved by ippo-cpr.py and plots mean ± 1 std
with a rolling-average smoother, in the style of NeurIPS / AAMAS figures.

Usage (from initial_settler_problem/):
    python plot_results.py # saves to results/cpr/curves.pdf
    python plot_results.py --out fig.png # PNG instead
    python plot_results.py --smooth 10 # wider smoother
"""
import argparse
import glob
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

# ── Global style ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "legend.framealpha": 0.9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Experiment conditions ──────────────────────────────────────────────────────
CONDITIONS = {
    "no-punishment": {"color": "#e74c3c", "label": "No Punishment"},
    "punishment":    {"color": "#2980b9", "label": "Punishment"},
}

# ── Panels: (metric_key, y-axis label, y-limits or None) ──────────────────────
PANELS = [
    ("mean_patch_level",     "Mean Patch Level",    (0.0, 1.05)),
    ("patch_survival_rate",  "Patch Survival Rate", (0.0, 1.05)),
    ("harvest_rate",         "Harvest Rate",        (0.0, 1.05)),
    ("episode_total_reward", "Episode Total Reward", None),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    """1-D rolling average; output length = len(x) - w + 1."""
    if w <= 1 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def load_runs(results_dir: str, tag: str) -> list[dict]:
    """Return a list of saved result dicts matching seed*_{tag}.pkl."""
    paths = sorted(glob.glob(os.path.join(results_dir, f"seed*_{tag}.pkl")))
    runs = []
    for p in paths:
        with open(p, "rb") as f:
            runs.append(pickle.load(f))
    return runs


def env_steps_axis(n_points: int, cfg: dict, window: int) -> np.ndarray:
    """x-axis in environment steps, trimmed to match rolling-mean output."""
    step_size = cfg["NUM_STEPS"] * cfg["NUM_ENVS"]
    raw = np.arange(1, n_points + 1) * step_size
    return raw[window - 1:]   # align with trailing-window mean


def plot_condition(ax, runs: list, key: str, color: str, label: str, window: int):
    """Plot mean ± 1 std across seeds onto ax."""
    smoothed = []
    for run in runs:
        raw = np.array(run["metrics"][key])
        smoothed.append(rolling_mean(raw, window))

    min_len = min(len(c) for c in smoothed)
    arr = np.stack([c[:min_len] for c in smoothed])   # (seeds, T)
    mean, std = arr.mean(0), arr.std(0)
    x = env_steps_axis(len(runs[0]["metrics"][key]), runs[0]["config"], window)[:min_len]

    ax.plot(x, mean, color=color, label=label, linewidth=2.0, zorder=3)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, zorder=2)


# ── Main figure ───────────────────────────────────────────────────────────────

def make_figure(results_dir: str, out_path: str, window: int = 5):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    axes = axes.flatten()

    for ax, (key, ylabel, ylim) in zip(axes, PANELS):
        plotted_any = False
        for tag, style in CONDITIONS.items():
            runs = load_runs(results_dir, tag)
            if not runs:
                continue
            if key not in runs[0]["metrics"]:
                continue
            plot_condition(ax, runs, key, style["color"], style["label"], window)
            plotted_any = True

        ax.set_xlabel("Environment Steps", labelpad=6)
        ax.set_ylabel(ylabel, labelpad=6)
        if ylim:
            ax.set_ylim(*ylim)
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M")
        )
        if plotted_any:
            ax.legend(loc="best")

    # Number of seeds found (for subtitle)
    n_seeds = {
        tag: len(load_runs(results_dir, tag))
        for tag in CONDITIONS
    }
    seed_str = "  |  ".join(
        f"{CONDITIONS[t]['label']}: {n} seed{'s' if n != 1 else ''}"
        for t, n in n_seeds.items() if n > 0
    )

    fig.suptitle(
        f"ISP Results for No Punishment Baseline\n{seed_str}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results/cpr")
    parser.add_argument("--out", default="./results/cpr/curves.pdf")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Rolling-average window width (default 5)")
    args = parser.parse_args()
    make_figure(args.results_dir, args.out, args.smooth)
