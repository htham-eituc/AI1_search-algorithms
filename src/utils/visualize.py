"""
visualize.py
============
Reusable, notebook-friendly visualization functions for comparing
metaheuristic algorithm benchmarks.

All public functions follow a consistent signature:
    plot_*(data, ..., ax=None, save_path=None) -> matplotlib.axes.Axes

Pass ax= to embed into an existing subplot grid.
Pass save_path="file.png" to auto-save.

Dependencies: matplotlib, numpy, pandas, scipy (optional, for smoothing)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

# Colorblind-friendly palette (IBM design)
PALETTE = [
    "#648FFF", "#785EF0", "#DC267F", "#FE6100",
    "#FFB000", "#009E73", "#56B4E9", "#0072B2",
    "#D55E00", "#CC79A7",
]

LINESTYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

def _get_ax(ax, figsize=(10, 5)):
    """Return existing ax or create a new one."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _save_or_show(fig, save_path):
    """Tight-layout then save/show."""
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")


def _smooth(curve: np.ndarray, window: int = 10) -> np.ndarray:
    """Simple moving-average smoothing for noisy convergence curves."""
    if window <= 1 or len(curve) < window:
        return curve
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


# ---------------------------------------------------------------------------
# 1.  Convergence curve  (best fitness vs iteration)
# ---------------------------------------------------------------------------

def plot_convergence(
    data: dict,
    *,
    algorithms: Optional[Sequence[str]] = None,
    dim: Optional[int] = None,
    problem: str = "",
    log_scale: bool = True,
    smooth_window: int = 1,
    ax=None,
    save_path=None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Plot convergence curves for one or more algorithms.

    Parameters
    ----------
    data : dict
        Keys are algorithm names. Values are dicts with at least:
            "convergence_curve" : np.ndarray  shape (max_iter,)
        Typically loaded from a .pkl convergence file.
    algorithms : list[str], optional
        Subset of algorithms to plot. Default: all keys in `data`.
    dim : int, optional
        If data contains multiple dims, filter to this one.
    problem : str
        Used in the title.
    log_scale : bool
        Use log y-axis (recommended for fitness values spanning many orders).
    smooth_window : int
        Moving-average window size (1 = no smoothing).
    """
    ax = _get_ax(ax, figsize=(10, 5))

    algos = algorithms or list(data.keys())

    for i, algo in enumerate(algos):
        entry = data[algo]
        # entry may be a list of runs (robustness) or a single dict
        if isinstance(entry, list):
            curves = np.array([r["convergence_curve"] for r in entry
                               if dim is None or r.get("dimensions") == dim])
            if curves.size == 0:
                continue
            mean_c = np.mean(curves, axis=0)
            std_c  = np.std(curves, axis=0)
            curve  = _smooth(mean_c, smooth_window)
            std_sm = _smooth(std_c, smooth_window)
            iters  = np.arange(len(curve))
            color  = PALETTE[i % len(PALETTE)]
            ls     = LINESTYLES[i % len(LINESTYLES)]
            ax.plot(iters, curve, color=color, ls=ls, lw=2, label=algo)
            ax.fill_between(iters, curve - std_sm, curve + std_sm,
                            color=color, alpha=0.15)
        else:
            if dim is not None and entry.get("dimensions") != dim:
                continue
            curve = _smooth(np.array(entry["convergence_curve"]), smooth_window)
            iters = np.arange(len(curve))
            ax.plot(iters, curve,
                    color=PALETTE[i % len(PALETTE)],
                    ls=LINESTYLES[i % len(LINESTYLES)],
                    lw=2, label=algo)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness (log)" if log_scale else "Best Fitness", fontsize=12)
    dim_str = f" — {dim}D" if dim else ""
    ax.set_title(title or f"Convergence Curve: {problem}{dim_str}", fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 2.  Robustness box-plot  (mean ± std over 30 runs)
# ---------------------------------------------------------------------------

def plot_robustness_boxplot(
    summary_df: pd.DataFrame,
    *,
    dim: Optional[int] = None,
    problem: str = "Rosenbrock",
    log_scale: bool = True,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Box-plot of best_fitness distribution over N runs.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Loaded from robustness CSV/PKL. Expected columns:
        algorithm, dimensions, mean_fitness, std_fitness,
        min_fitness, max_fitness, median_fitness
    """
    ax = _get_ax(ax, figsize=(11, 5))

    if dim is not None:
        df = summary_df[summary_df["dimensions"] == dim].copy()
    else:
        df = summary_df.copy()

    df = df.sort_values("mean_fitness").reset_index(drop=True)
    x  = np.arange(len(df))

    # Draw box manually from pre-aggregated stats (no raw runs needed)
    for i, row in df.iterrows():
        color = PALETTE[i % len(PALETTE)]
        # IQR approximation: median ± 0.7*std  (normal assumption)
        q1   = row["median_fitness"] - 0.674 * row["std_fitness"]
        q3   = row["median_fitness"] + 0.674 * row["std_fitness"]
        whi  = row["max_fitness"]
        wlo  = row["min_fitness"]
        # whisker line
        ax.vlines(i, wlo, whi, color=color, lw=1.5, zorder=2)
        # box (IQR)
        ax.bar(i, q3 - q1, bottom=q1, width=0.5,
               color=color, alpha=0.5, zorder=3)
        # median line
        ax.hlines(row["median_fitness"], i - 0.25, i + 0.25,
                  color=color, lw=2.5, zorder=4)
        # mean marker
        ax.scatter(i, row["mean_fitness"], color=color, marker="D",
                   s=50, zorder=5, edgecolors="white", linewidths=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(df["algorithm"], rotation=30, ha="right", fontsize=11)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel("Best Fitness", fontsize=12)
    dim_str = f" ({dim}D)" if dim else ""
    ax.set_title(f"Robustness — {problem}{dim_str}  [◆=mean, bar=IQR, whisker=min/max]",
                 fontsize=12)
    ax.grid(True, axis="y", alpha=0.35)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 3.  Robustness bar chart  (mean ± std, sorted)
# ---------------------------------------------------------------------------

def plot_robustness_bar(
    summary_df: pd.DataFrame,
    *,
    dim: Optional[int] = None,
    problem: str = "Rosenbrock",
    log_scale: bool = False,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Horizontal bar chart of mean best_fitness ± std over N runs.
    Easier to read than boxplot when comparing many algorithms.
    """
    ax = _get_ax(ax, figsize=(9, 5))

    df = summary_df.copy()
    if dim is not None:
        df = df[df["dimensions"] == dim]
    df = df.sort_values("mean_fitness", ascending=True).reset_index(drop=True)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    y_pos  = np.arange(len(df))

    ax.barh(y_pos, df["mean_fitness"],
            xerr=df["std_fitness"],
            color=colors, alpha=0.8,
            error_kw={"elinewidth": 1.5, "capsize": 4, "ecolor": "black"},
            height=0.6)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["algorithm"], fontsize=11)
    ax.set_xlabel("Mean Best Fitness ± Std", fontsize=12)
    if log_scale:
        ax.set_xscale("log")
    dim_str = f" ({dim}D)" if dim else ""
    ax.set_title(f"Robustness: Mean ± Std — {problem}{dim_str}", fontsize=13)
    ax.grid(True, axis="x", alpha=0.35)
    ax.invert_yaxis()   # best (lowest) at top

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 4.  Scalability plot  (fitness vs dimension)
# ---------------------------------------------------------------------------

def plot_scalability(
    data: dict,
    *,
    algorithms: Optional[Sequence[str]] = None,
    problem: str = "",
    metric: str = "mean_fitness",
    log_scale: bool = True,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Plot algorithm performance (mean best fitness) across problem dimensions.

    Parameters
    ----------
    data : dict  OR  pd.DataFrame
        If dict: keys are algo names, values are lists of dicts
                 with "dimensions" and "best_fitness" keys.
        If DataFrame: expects columns algorithm, dimensions, mean_fitness.
    metric : str
        Column/key name for fitness metric (default "mean_fitness").
    """
    ax = _get_ax(ax, figsize=(9, 5))

    if isinstance(data, pd.DataFrame):
        df = data
        algos = algorithms or df["algorithm"].unique().tolist()
        for i, algo in enumerate(algos):
            sub = df[df["algorithm"] == algo].sort_values("dimensions")
            ax.plot(sub["dimensions"], sub[metric],
                    marker="o", color=PALETTE[i % len(PALETTE)],
                    ls=LINESTYLES[i % len(LINESTYLES)],
                    lw=2, label=algo)
    else:
        algos = algorithms or list(data.keys())
        for i, algo in enumerate(algos):
            runs = data[algo] if isinstance(data[algo], list) else [data[algo]]
            dims_seen = sorted(set(r["dimensions"] for r in runs))
            means = [np.mean([r["best_fitness"] for r in runs
                              if r["dimensions"] == d]) for d in dims_seen]
            ax.plot(dims_seen, means,
                    marker="o", color=PALETTE[i % len(PALETTE)],
                    ls=LINESTYLES[i % len(LINESTYLES)],
                    lw=2, label=algo)

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Dimensions", fontsize=12)
    ax.set_ylabel("Mean Best Fitness", fontsize=12)
    ax.set_title(f"Scalability — {problem}", fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, which="both", alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 5.  Time complexity bar chart
# ---------------------------------------------------------------------------

def plot_time_comparison(
    summary_df: pd.DataFrame,
    *,
    dim: Optional[int] = None,
    problem: str = "",
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Horizontal bar chart of mean execution time per algorithm.
    """
    ax = _get_ax(ax, figsize=(9, 5))

    df = summary_df.copy()
    if dim is not None:
        df = df[df["dimensions"] == dim]
    df = df.sort_values("mean_time", ascending=True).reset_index(drop=True)

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(df))]
    y_pos  = np.arange(len(df))

    ax.barh(y_pos, df["mean_time"], color=colors, alpha=0.8, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["algorithm"], fontsize=11)
    ax.set_xlabel("Mean Execution Time (s)", fontsize=12)
    dim_str = f" ({dim}D)" if dim else ""
    ax.set_title(f"Computation Time — {problem}{dim_str}", fontsize=13)
    ax.grid(True, axis="x", alpha=0.35)
    ax.invert_yaxis()

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 6.  Diversity curve
# ---------------------------------------------------------------------------

def plot_diversity(
    data: dict,
    *,
    algorithms: Optional[Sequence[str]] = None,
    dim: Optional[int] = None,
    problem: str = "",
    smooth_window: int = 10,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Plot population diversity (exploration) over iterations.
    Useful for comparing exploration vs exploitation behaviour.

    data format: same as plot_convergence (dict of algo → result dict or list)
    """
    ax = _get_ax(ax, figsize=(10, 5))
    algos = algorithms or list(data.keys())

    for i, algo in enumerate(algos):
        entry = data[algo]
        entries = entry if isinstance(entry, list) else [entry]
        if dim is not None:
            entries = [e for e in entries if e.get("dimensions") == dim]
        if not entries:
            continue

        curves = np.array([e["diversity_curve"] for e in entries
                           if "diversity_curve" in e])
        if curves.size == 0:
            continue

        mean_d = _smooth(np.mean(curves, axis=0), smooth_window)
        iters  = np.arange(len(mean_d))
        color  = PALETTE[i % len(PALETTE)]
        ax.plot(iters, mean_d, color=color,
                ls=LINESTYLES[i % len(LINESTYLES)], lw=2, label=algo)

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Population Diversity", fontsize=12)
    dim_str = f" — {dim}D" if dim else ""
    ax.set_title(f"Diversity (Exploration) — {problem}{dim_str}", fontsize=13)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 7.  Multi-panel comparison dashboard
# ---------------------------------------------------------------------------

def plot_dashboard(
    convergence_data: dict,
    summary_df: pd.DataFrame,
    *,
    dim: Optional[int] = None,
    problem: str = "",
    smooth_window: int = 5,
    save_path=None,
) -> plt.Figure:
    """
    2×2 dashboard: Convergence | Robustness bar | Scalability | Time
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Algorithm Comparison — {problem}" +
                 (f" ({dim}D)" if dim else ""), fontsize=15, y=1.01)

    plot_convergence(convergence_data, dim=dim, problem=problem,
                     smooth_window=smooth_window, ax=axes[0, 0])

    plot_robustness_bar(summary_df, dim=dim, problem=problem, ax=axes[0, 1])

    plot_scalability(summary_df, problem=problem, ax=axes[1, 0])

    plot_time_comparison(summary_df, dim=dim, problem=problem, ax=axes[1, 1])

    _save_or_show(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# 8.  Convergence per-run spaghetti (for robustness PKL)
# ---------------------------------------------------------------------------

def plot_convergence_runs(
    raw_runs: list[dict],
    *,
    algorithm: str = "",
    dim: Optional[int] = None,
    problem: str = "",
    max_runs_shown: int = 15,
    log_scale: bool = True,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Plot individual convergence curves for all 30 runs of ONE algorithm,
    plus the median curve highlighted in bold.

    raw_runs : list of dicts from robustness PKL["raw"]
               filtered to a single algorithm before passing in.
    """
    ax = _get_ax(ax, figsize=(10, 5))

    runs = raw_runs
    if dim is not None:
        runs = [r for r in runs if r.get("dimensions") == dim]
    if not runs:
        print(f"No runs found for {algorithm} dim={dim}.")
        return ax

    curves = np.array([r["convergence_curve"] for r in runs
                       if r.get("convergence_curve") is not None])
    if curves.size == 0:
        print("No convergence curves in this data.")
        return ax

    iters   = np.arange(curves.shape[1])
    n_show  = min(max_runs_shown, len(curves))
    median  = np.median(curves, axis=0)
    color   = PALETTE[0]

    for curve in curves[:n_show]:
        ax.plot(iters, curve, color=color, alpha=0.2, lw=0.9)

    ax.plot(iters, median, color="#DC267F", lw=2.5,
            label=f"Median ({len(curves)} runs)")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Convergence Spread — {algorithm} | {problem}"
                 + (f" {dim}D" if dim else ""), fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 9.  2D Contour landscape (static)
# ---------------------------------------------------------------------------

def plot_contour_2d(
    func,
    bounds: tuple,
    *,
    final_positions: Optional[dict] = None,
    resolution: int = 300,
    log_scale: bool = True,
    title: str = "",
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Draw a filled contour of a 2D objective function landscape.

    Parameters
    ----------
    func : callable
        Objective function accepting (N, 2) array, returning (N,) fitness.
    bounds : tuple  (low, high)  e.g. (-5.12, 5.12)
    final_positions : dict  {algo_name: np.array shape (2,)}
        Optional: scatter each algorithm's final best position on the plot.
    log_scale : bool
        Apply log10 to fitness values before plotting (shows valleys better).
    """
    ax = _get_ax(ax, figsize=(7, 6))
    low, high = bounds

    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    Z = func(grid).reshape(resolution, resolution)

    if log_scale:
        Z_plot = np.log10(np.clip(Z, 1e-10, None))
        cbar_label = "log₁₀(fitness)"
    else:
        Z_plot = Z
        cbar_label = "fitness"

    cf = ax.contourf(X, Y, Z_plot, levels=50, cmap="viridis")
    ax.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.3, alpha=0.4)
    plt.colorbar(cf, ax=ax, label=cbar_label, fraction=0.046, pad=0.04)

    if final_positions:
        for i, (algo, pos) in enumerate(final_positions.items()):
            pos = np.array(pos)
            ax.scatter(pos[0], pos[1],
                       color=PALETTE[i % len(PALETTE)],
                       marker="*", s=220, zorder=5,
                       edgecolors="white", linewidths=0.8, label=algo)
        ax.legend(fontsize=9, loc="upper right",
                  framealpha=0.7, markerscale=0.9)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(title or "Function Landscape (2D)", fontsize=13)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 10.  2D Contour + trajectory overlay  (HC / SA path lines)
# ---------------------------------------------------------------------------

def plot_trajectory_2d(
    func,
    bounds: tuple,
    trajectories: dict,
    *,
    resolution: int = 300,
    log_scale: bool = True,
    title: str = "",
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Draw contour landscape with agent trajectory lines overlaid.

    Parameters
    ----------  
    trajectories : dict  {algo_name: list_of_positions}
        Each value is a list / array of shape (T, 2) — the path walked.
        For HC/SA extracted from population_history[:, best_idx, :].
    """
    ax = plot_contour_2d(func, bounds, log_scale=log_scale,
                         title=title or "Agent Trajectories (2D)", ax=ax)

    for i, (algo, path) in enumerate(trajectories.items()):
        path = np.array(path)
        # Guard: skip if trajectory is empty or not a 2-column array
        if path.ndim != 2 or path.shape[0] == 0 or path.shape[1] < 2:
            print(f"[plot_trajectory_2d] WARNING: '{algo}' has no valid trajectory "
                  f"(shape={path.shape}). Re-run dim2_experiment.py to regenerate.")
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.plot(path[:, 0], path[:, 1],
                color=color, lw=1.5, alpha=0.85,
                ls=LINESTYLES[i % len(LINESTYLES)], label=algo)
        ax.scatter(path[0, 0], path[0, 1],
                   color=color, marker="o", s=80, zorder=6,
                   edgecolors="white", linewidths=0.8)
        ax.scatter(path[-1, 0], path[-1, 1],
                   color=color, marker="X", s=120, zorder=6,
                   edgecolors="white", linewidths=0.8)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.7)
    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 11.  Animated GIF — population movement
# ---------------------------------------------------------------------------

from pathlib import Path
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt


def make_population_gif(
    func,
    bounds: tuple,
    population_history: list,
    algo_name: str,
    save_path: Union[str, Path],
    *,
    best_history: Optional[list] = None,
    n_frames: int = 60,
    fps: int = 10,
    resolution: int = 200,
    log_scale: bool = True,
    dot_size: int = 30,
) -> Path:
    """
    Create an animated GIF showing population movement on a 2D contour.
    """

    import matplotlib.animation as animation

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ===== Safety checks =====
    total_iters = len(population_history)

    if total_iters == 0:
        raise ValueError("population_history is empty")

    # Prevent frame overflow
    n_frames = min(n_frames, total_iters)
    frame_indices = np.linspace(0, total_iters - 1, n_frames, dtype=int)

    low, high = bounds

    # ===== Precompute contour =====
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)

    X, Y = np.meshgrid(x, y)

    grid = np.column_stack([X.ravel(), Y.ravel()])
    Z = func(grid).reshape(resolution, resolution)

    if log_scale:
        Z_plot = np.log10(np.clip(Z, 1e-10, None))
    else:
        Z_plot = Z

    # ===== Figure =====
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout(pad=1.5)

    ax.contourf(X, Y, Z_plot, levels=50, cmap="viridis")
    ax.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.25, alpha=0.35)

    ax.set_xlim(low, high)
    ax.set_ylim(low, high)

    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)

    scat = ax.scatter(
        [],
        [],
        c="#FE6100",
        s=dot_size,
        zorder=5,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.85,
    )

    best_scat = ax.scatter(
        [],
        [],
        c="#FFB000",
        s=120,
        marker="*",
        zorder=6,
        edgecolors="white",
        linewidths=0.8,
    )

    title_txt = ax.set_title("", fontsize=11)

    iter_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        fontsize=9,
        color="white",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4),
    )

    # ===== Animation update =====
    def _update(frame_num):

        idx = min(frame_indices[frame_num], total_iters - 1)

        pop = np.asarray(population_history[idx])

        if pop.size > 0:
            scat.set_offsets(pop[:, :2])

        if best_history is not None and idx < len(best_history):
            bp = np.asarray(best_history[idx])
            best_scat.set_offsets(bp[:2].reshape(1, 2))

        iter_text.set_text(f"iter {idx + 1}/{total_iters}")
        title_txt.set_text(f"{algo_name} — frame {frame_num + 1}/{n_frames}")

        return scat, best_scat, iter_text, title_txt

    # ===== Create animation =====
    ani = animation.FuncAnimation(
        fig,
        _update,
        frames=n_frames,
        interval=1000 // fps,
        blit=True,
    )

    # ===== Save GIF =====
    ani.save(str(save_path), writer="pillow", fps=fps)

    plt.close(fig)

    print(f"GIF saved → {save_path}")

    return save_path


# ---------------------------------------------------------------------------
# 12.  Animated GIF — single-agent trajectory  (HC / SA)
# ---------------------------------------------------------------------------

def make_trajectory_gif(
    func,
    bounds: tuple,
    trajectory: list,
    algo_name: str,
    save_path: Union[str, Path],
    *,
    n_frames: int = 60,
    fps: int = 10,
    resolution: int = 200,
    log_scale: bool = True,
) -> Path:
    """
    Create an animated GIF showing a single agent's path on a 2D contour.

    Parameters
    ----------
    trajectory : list of np.array shape (2,)
        Sequence of best positions — one per recorded step.
        For HC/SA: extracted as best_position per iteration from population_history
        (population_history[i][argmin(fitness_at_i)]).
    """
    import matplotlib.animation as animation

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    low, high = bounds
    traj = np.array(trajectory)

    # Guard: validate trajectory shape before attempting to animate
    if traj.ndim != 2 or traj.shape[0] == 0 or traj.shape[1] < 2:
        print(f"[make_trajectory_gif] WARNING: trajectory for '{algo_name}' is invalid "
              f"(shape={traj.shape}). Re-run dim2_experiment.py to regenerate PKL.")
        return save_path
    total_steps = len(traj)
    frame_indices = np.linspace(0, total_steps - 1, n_frames, dtype=int)

    # Pre-render contour
    x = np.linspace(low, high, resolution)
    y = np.linspace(low, high, resolution)
    X, Y = np.meshgrid(x, y)
    grid = np.column_stack([X.ravel(), Y.ravel()])
    Z = func(grid).reshape(resolution, resolution)
    Z_plot = np.log10(np.clip(Z, 1e-10, None)) if log_scale else Z

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.tight_layout(pad=1.5)
    ax.contourf(X, Y, Z_plot, levels=50, cmap="viridis")
    ax.contour(X, Y, Z_plot, levels=20, colors="white", linewidths=0.25, alpha=0.35)
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("x", fontsize=10)
    ax.set_ylabel("y", fontsize=10)
    ax.set_title(f"{algo_name} — Trajectory", fontsize=11)

    # Start marker
    ax.scatter(traj[0, 0], traj[0, 1], c="cyan", s=100,
               marker="o", zorder=7, edgecolors="white", linewidths=0.8,
               label="start")

    line,  = ax.plot([], [], color="#FE6100", lw=1.5, alpha=0.8)
    point  = ax.scatter([], [], c="#FFB000", s=140, marker="*",
                        zorder=8, edgecolors="white", linewidths=0.8)
    iter_text = ax.text(0.02, 0.96, "", transform=ax.transAxes,
                        fontsize=9, color="white", verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.4))

    def _update(frame_num):
        end_idx = frame_indices[frame_num] + 1
        path_so_far = traj[:end_idx]
        line.set_data(path_so_far[:, 0], path_so_far[:, 1])
        point.set_offsets(traj[frame_indices[frame_num]].reshape(1, 2))
        iter_text.set_text(f"step {frame_indices[frame_num] + 1}/{total_steps}")
        return line, point, iter_text

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000 // fps, blit=True
    )
    ani.save(str(save_path), writer="pillow", fps=fps)
    plt.close(fig)
    print(f"GIF saved → {save_path}")
    return save_path


# ---------------------------------------------------------------------------
# 13.  Parameter sensitivity curves
# ---------------------------------------------------------------------------

def plot_parameter_sensitivity(
    results_dict: dict,
    param_name: str,
    param_values: list,
    *,
    algo_name: str = "",
    problem: str = "",
    log_scale: bool = True,
    smooth_window: int = 5,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Plot convergence curves for different values of one hyperparameter.

    Parameters
    ----------
    results_dict : dict  {param_value: convergence_curve np.ndarray}
    param_name   : str   e.g. "step_size", "mutation_rate", "F"
    param_values : list  ordered list of the param values used (for legend sort)
    """
    ax = _get_ax(ax, figsize=(9, 5))

    cmap   = plt.get_cmap("plasma")
    colors = [cmap(i / max(len(param_values) - 1, 1))
              for i in range(len(param_values))]

    for i, val in enumerate(param_values):
        if val not in results_dict:
            continue
        curve = _smooth(np.array(results_dict[val]), smooth_window)
        iters = np.arange(len(curve))
        ax.plot(iters, curve, color=colors[i], lw=2,
                label=f"{param_name}={val}")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Parameter Sensitivity: {algo_name} — {param_name}\n"
                 f"Problem: {problem}", fontsize=12)
    ax.legend(fontsize=10, ncol=1)
    ax.grid(True, which="both", alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 14.  Time-to-threshold heatmap
# ---------------------------------------------------------------------------

def plot_time_to_threshold(
    convergence_data: dict,
    thresholds: list,
    *,
    dim: Optional[int] = None,
    problem: str = "",
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Heatmap showing at which iteration each algorithm crossed each fitness threshold.
    Cell is white/marked if the threshold was never reached.

    Parameters
    ----------
    convergence_data : dict  {algo: result_dict}  (from flatten_for_dim)
    thresholds : list of float  e.g. [1000, 100, 10, 1, 0.1]
    """
    ax = _get_ax(ax, figsize=(10, 5))

    algos = list(convergence_data.keys())
    matrix = np.full((len(algos), len(thresholds)), np.nan)

    for i, algo in enumerate(algos):
        entry = convergence_data[algo]
        curve = np.array(entry.get("convergence_curve", []))
        if curve.size == 0:
            continue
        for j, thresh in enumerate(thresholds):
            hits = np.where(curve <= thresh)[0]
            if len(hits) > 0:
                matrix[i, j] = hits[0]   # first iteration below threshold

    # Normalise for colour (ignore NaN)
    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1
    matrix_norm = matrix / vmax

    im = ax.imshow(matrix_norm, cmap="RdYlGn_r", aspect="auto",
                   vmin=0, vmax=1)

    # Annotate cells
    for i in range(len(algos)):
        for j in range(len(thresholds)):
            val = matrix[i, j]
            if np.isnan(val):
                txt = "✗"
                color = "white"
            else:
                txt = f"{int(val)}"
                color = "black" if matrix_norm[i, j] < 0.7 else "white"
            ax.text(j, i, txt, ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f"≤{t}" for t in thresholds], fontsize=10)
    ax.set_yticks(range(len(algos)))
    ax.set_yticklabels(algos, fontsize=11)
    ax.set_xlabel("Fitness Threshold", fontsize=12)
    ax.set_title(f"Iterations to Reach Threshold — {problem}"
                 + (f" ({dim}D)" if dim else ""), fontsize=13)

    plt.colorbar(im, ax=ax, label="Relative iteration (green=faster)",
                 fraction=0.046, pad=0.04)
    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# 15.  Best vs average quality grouped bar
# ---------------------------------------------------------------------------

import pandas as pd
import numpy as np
from typing import Optional

def table_best_vs_avg(
    convergence_data: dict,
    *,
    dim: Optional[int] = None,
    problem: str = "",
    sort_by: str = "best",
    display_table: bool = True,
):
    """
    Create a table comparing Best Fitness and Final Average Fitness.

    convergence_data : dict
        {algo: result_dict} from flatten_for_dim()
    """

    rows = []

    for algo, entry in convergence_data.items():

        best = entry.get("best_fitness", np.nan)

        avg_curve = entry.get("average_fitness_curve")
        avg = (
            avg_curve[-1]
            if avg_curve is not None and len(avg_curve) > 0
            else np.nan
        )

        rows.append({
            "Algorithm": algo,
            "Best Fitness": best,
            "Avg Fitness (final)": avg,
        })

    df = pd.DataFrame(rows)

    # Sort
    if sort_by == "best":
        df = df.sort_values("Best Fitness")
    elif sort_by == "avg":
        df = df.sort_values("Avg Fitness (final)")

    df.reset_index(drop=True, inplace=True)

    # Friendly formatting
    df["Best Fitness"] = df["Best Fitness"].map(lambda x: f"{x:.6g}")
    df["Avg Fitness (final)"] = df["Avg Fitness (final)"].map(lambda x: f"{x:.6g}")

    if display_table:
        title = f"Best vs Average Solution Quality"
        if problem:
            title += f" — {problem}"
        if dim:
            title += f" ({dim}D)"

        print(title)

    return df

def plot_best_vs_avg(
    convergence_data: dict,
    *,
    dim: Optional[int] = None,
    problem: str = "",
    log_scale: bool = True,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Grouped bar chart comparing best_fitness and final average_fitness per algo.
    """

    ax = _get_ax(ax, figsize=(11, 5))

    algos = list(convergence_data.keys())
    best_vals = []
    avg_vals = []

    for algo in algos:
        entry = convergence_data[algo]
        best_vals.append(entry.get("best_fitness", np.nan))

        avg_curve = entry.get("average_fitness_curve")
        avg_vals.append(
            avg_curve[-1] if avg_curve is not None and len(avg_curve) > 0 else np.nan
        )

    x = np.arange(len(algos))
    width = 0.38

    bars1 = ax.bar(
        x - width / 2,
        best_vals,
        width,
        label="Best Fitness",
        color="#648FFF",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )

    bars2 = ax.bar(
        x + width / 2,
        avg_vals,
        width,
        label="Avg Fitness (final)",
        color="#FE6100",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )

    # --- Friendly number formatting ---
    def format_value(v):
        if v >= 1:
            return f"{v:.2f}"
        elif v >= 0.01:
            return f"{v:.3f}"
        elif v >= 0.0001:
            return f"{v:.5f}"
        else:
            return f"{v:.2e}"

    # Value labels
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        if np.isnan(h) or h <= 0:
            continue

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.05,
            format_value(h),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=30,
        )

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=11)

    ax.set_ylabel("Fitness", fontsize=12)

    dim_str = f" ({dim}D)" if dim else ""

    ax.set_title(
        f"Best vs Average Solution Quality — {problem}{dim_str}",
        fontsize=13,
        pad=15,   # ← add padding from title
    )

    # Move legend outside
    ax.legend(
        fontsize=11,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )

    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    _save_or_show(ax.get_figure(), save_path)

    return ax


# ---------------------------------------------------------------------------
# 16.  Scalability drop chart  (30D → 100D side-by-side + % annotation)
# ---------------------------------------------------------------------------

def plot_scalability_drop(
    data_low: dict,
    data_high: dict,
    *,
    dim_low: int = 30,
    dim_high: int = 100,
    problem: str = "",
    log_scale: bool = True,
    ax=None,
    save_path=None,
) -> plt.Axes:
    """
    Grouped bar chart comparing best_fitness at two dimensionalities,
    with % degradation annotated above each pair.

    Parameters
    ----------
    data_low  : dict  {algo: result_dict}  at lower dimension
    data_high : dict  {algo: result_dict}  at higher dimension
    """
    ax = _get_ax(ax, figsize=(12, 5))

    algos = [a for a in data_low if a in data_high]
    low_vals  = [data_low[a].get("best_fitness", np.nan)  for a in algos]
    high_vals = [data_high[a].get("best_fitness", np.nan) for a in algos]

    x     = np.arange(len(algos))
    width = 0.38

    bars1 = ax.bar(x - width / 2, low_vals,  width,
                   label=f"{dim_low}D",  color="#648FFF",
                   alpha=0.85, edgecolor="white", linewidth=0.7)
    bars2 = ax.bar(x + width / 2, high_vals, width,
                   label=f"{dim_high}D", color="#DC267F",
                   alpha=0.85, edgecolor="white", linewidth=0.7)

    # Annotate % drop above each pair
    for i, (lo, hi) in enumerate(zip(low_vals, high_vals)):
        if np.isnan(lo) or np.isnan(hi) or lo == 0:
            continue
        pct = (hi - lo) / abs(lo) * 100
        y_top = max(lo, hi)
        sign  = "+" if pct >= 0 else ""
        ax.text(x[i], y_top * (1.15 if not log_scale else 2),
                f"{sign}{pct:.0f}%", ha="center", va="bottom",
                fontsize=8, fontweight="bold", color="#333333")

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Best Fitness", fontsize=12)
    ax.set_title(f"Scalability: {dim_low}D → {dim_high}D — {problem}", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    _save_or_show(ax.get_figure(), save_path)
    return ax


# ---------------------------------------------------------------------------
# Convenience I/O helpers
# ---------------------------------------------------------------------------

def load_pkl(path: Union[str, Path]) -> dict:
    """Load a .pkl result file produced by robustness or convergence experiments."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_summary_csv(path: Union[str, Path]) -> pd.DataFrame:
    """Load a robustness summary CSV into a DataFrame."""
    df = pd.read_csv(path)
    # Coerce numeric columns
    numeric = ["dimensions", "n_runs", "mean_fitness", "std_fitness",
               "min_fitness", "max_fitness", "median_fitness", "mean_time"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Quick demo (run this file directly to see all plots)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("Generating synthetic demo data …")

    # Simulate convergence data for 3 algorithms over 200 iterations
    np.random.seed(42)
    demo_conv = {}
    for name, decay in [("HC", 0.97), ("GA", 0.93), ("PSO", 0.90)]:
        curve = 1000.0 * np.cumprod(np.random.uniform(decay - 0.02, decay + 0.02, 200))
        demo_conv[name] = {
            "convergence_curve": curve,
            "diversity_curve": np.random.rand(200) * np.linspace(1, 0.1, 200),
            "dimensions": 10,
        }

    # Simulate summary DataFrame
    demo_summary = pd.DataFrame([
        {"algorithm": "HC",  "dimensions": 10, "mean_fitness": 150.0,  "std_fitness": 30.0,
         "min_fitness": 90.0,  "max_fitness": 220.0, "median_fitness": 145.0, "mean_time": 0.5},
        {"algorithm": "GA",  "dimensions": 10, "mean_fitness": 45.0,   "std_fitness": 12.0,
         "min_fitness": 20.0,  "max_fitness": 80.0,  "median_fitness": 43.0,  "mean_time": 1.2},
        {"algorithm": "PSO", "dimensions": 10, "mean_fitness": 12.0,   "std_fitness": 5.0,
         "min_fitness": 5.0,   "max_fitness": 25.0,  "median_fitness": 11.5,  "mean_time": 0.9},
    ])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_convergence(demo_conv, problem="Demo", dim=10, ax=axes[0])
    plot_robustness_bar(demo_summary, problem="Demo", dim=10, ax=axes[1])
    plot_diversity(demo_conv, problem="Demo", dim=10, ax=axes[2])
    plt.tight_layout()
    plt.show()
    print("Demo complete.")