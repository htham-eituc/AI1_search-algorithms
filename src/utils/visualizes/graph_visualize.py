"""
graph_visualize.py  (v4)
========================
Visualisation helpers for TSP experiments — sparse / clustered / euclidean.

Handles four result statuses produced by tsp_experiments.py:
  "ok"      — algorithm finished, finite distance
  "inf"     — algorithm finished but returned inf/NaN
  "timeout" — exceeded wall-clock limit
  "error"   — unhandled exception

New in v4
---------
  plot_failure_stats(case)          — bar chart of ok/inf/timeout/error counts
  plot_failure_heatmap(case)        — algorithm × n_cities failure grid
  All quality plots filter to "ok" rows only and annotate failure counts.

Quick reference
---------------
    from graph_visualize import TSPVisualizer
    viz = TSPVisualizer("results/tsp_results.pkl")

    viz.plot_all_paths("sparse", "test_0")
    viz.plot_failure_stats("sparse")
    viz.plot_failure_heatmap("sparse")
    viz.plot_large_test_dashboard("sparse")   # includes failure panel
    viz.plot_scalability("clustered")
    viz.plot_complexity("euclidean")
    viz.plot_solution_quality("sparse")
    viz.plot_convergence("euclidean", "test_1")
    viz.plot_cross_case_comparison()
    viz.summary_table("sparse")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection


# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────

ALGO_COLORS = {
    "CA_TSP":           "#E63946",
    "GA_TSP":           "#2196F3",
    "SA_TSP":           "#FF9800",
    "ACO_TSP":          "#4CAF50",
    "HillClimbing_TSP": "#9C27B0",
    "A_STAR_TSP":       "#00BCD4",
}
ALGO_MARKERS = {
    "CA_TSP": "o", "GA_TSP": "s", "SA_TSP": "^",
    "ACO_TSP": "D", "HillClimbing_TSP": "P", "A_STAR_TSP": "*",
}
ALGO_SHORT = {
    "CA_TSP": "CA", "GA_TSP": "GA", "SA_TSP": "SA",
    "ACO_TSP": "ACO", "HillClimbing_TSP": "HC", "A_STAR_TSP": "A*",
}
CASE_COLOR = {
    "sparse": "#607D8B", "clustered": "#8BC34A", "euclidean": "#FF5722",
}
CASE_LABEL = {
    "sparse": "Sparse", "clustered": "Clustered", "euclidean": "Euclidean",
}

# Status colours used in failure charts
STATUS_COLORS = {
    "ok":      "#4CAF50",
    "inf":     "#FF9800",
    "timeout": "#F44336",
    "error":   "#9C27B0",
}

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F8F9FA",
    "axes.grid":         True,
    "grid.color":        "white",
    "grid.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mds_coords(dm: np.ndarray) -> np.ndarray:
    """Classical MDS: reconstruct 2-D positions from a distance matrix."""
    finite_max = dm[np.isfinite(dm)].max() if np.any(np.isfinite(dm)) else 1.0
    D  = np.where(np.isinf(dm), finite_max * 10, dm).astype(float)
    D2 = D ** 2
    row_mean = D2.mean(axis=1, keepdims=True)
    col_mean = D2.mean(axis=0, keepdims=True)
    B  = -0.5 * (D2 - row_mean - col_mean + D2.mean())
    eigvals, eigvecs = np.linalg.eigh(B)
    idx     = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[idx][:2], 0.0)
    eigvecs = eigvecs[:, idx][:, :2]
    return (eigvecs * np.sqrt(eigvals)).astype(float)


def _extract_coords(tdata: dict) -> np.ndarray:
    """
    TSPProblem stores no x/y data — coordinates are always MDS-reconstructed
    from the distance matrix during the experiment run and saved into tdata.
    If somehow missing, recompute from the distance matrix here.
    """
    coords = tdata.get("coordinates")
    if coords is not None:
        return np.asarray(coords)
    # recompute on the fly from the saved distance matrix
    dm = tdata.get("dist_matrix")
    if dm is not None:
        return _mds_coords(np.asarray(dm))
    return None


def _draw_sparse_edges(ax, coords: np.ndarray, dm: np.ndarray,
                       alpha: float = 0.15) -> None:
    n    = dm.shape[0]
    segs = [[coords[i], coords[j]]
            for i in range(n) for j in range(i + 1, n)
            if np.isfinite(dm[i, j])]
    ax.add_collection(
        LineCollection(segs, colors="#888888", linewidths=0.4, alpha=alpha)
    )


def _draw_graph_nodes(ax, coords: np.ndarray,
                      node_color: str = "white",
                      border_color: str = "#222222",
                      node_size: float = 220,
                      font_size: float = 6.5) -> None:
    """
    Draw every city as a labelled circle — the standard graph-node look.
    Always called AFTER edges so nodes sit on top.
    """
    ax.scatter(coords[:, 0], coords[:, 1],
               s=node_size, c=node_color, edgecolors=border_color,
               linewidths=1.2, zorder=5)
    for i, (x, y) in enumerate(coords):
        ax.annotate(str(i), (x, y),
                    fontsize=font_size, ha="center", va="center",
                    fontweight="bold", zorder=6)


def _draw_tour_edges(ax, coords: np.ndarray,
                     tour: list[int],
                     color: str,
                     lw: float = 1.8,
                     alpha: float = 0.85,
                     directed: bool = True) -> None:
    """
    Draw the tour as explicit graph edges (node_u → node_v).
    Each of the n edges is drawn individually so overlaps are visible.
    directed=True adds a small midpoint arrow showing tour direction.
    """
    if coords is None or tour is None or len(tour) < 2:
        return

    n = len(tour)
    for step in range(n):
        u = tour[step]
        v = tour[(step + 1) % n]          # last node wraps back to first
        x0, y0 = coords[u]
        x1, y1 = coords[v]

        ax.plot([x0, x1], [y0, y1],
                color=color, linewidth=lw, alpha=alpha,
                solid_capstyle="round", zorder=3)

        if directed:
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            dx = (x1 - x0) * 0.01
            dy = (y1 - y0) * 0.01
            ax.annotate("",
                        xy=(mx + dx, my + dy),
                        xytext=(mx - dx, my - dy),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=color,
                                        lw=lw * 0.6,
                                        mutation_scale=9),
                        zorder=4)


def _build_df(results: dict) -> pd.DataFrame:
    rows = []
    for case_name, tests in results.items():
        for test_id, tdata in tests.items():
            for aname, ares in tdata["algorithms"].items():
                status = ares.get("status", "ok")
                bf     = ares.get("best_fitness")
                # treat inf/NaN best_fitness as NaN so they're excluded from
                # numeric plots automatically
                if isinstance(bf, float) and (bf != bf or bf == float("inf")):
                    bf = np.nan
                rows.append({
                    "case":           case_name,
                    "test_id":        test_id,
                    "n_cities":       tdata["n_cities"],
                    "algorithm":      aname,
                    "status":         status,
                    "best_fitness":   bf if status == "ok" else np.nan,
                    "execution_time": (ares.get("execution_time")
                                       or ares.get("wall_time")),
                    "peak_memory_kb": ares.get("peak_memory_kb"),
                    "nodes_expanded": ares.get("nodes_expanded"),
                    "time_complexity":  ares.get("time_complexity"),
                    "space_complexity": ares.get("space_complexity"),
                    "convergence":      ares.get("convergence"),
                })
    return pd.DataFrame(rows)


def _ok(df: pd.DataFrame) -> pd.DataFrame:
    """Return only rows with status=='ok' and finite best_fitness."""
    return df[(df["status"] == "ok") & df["best_fitness"].notna()]


def _failed(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows that are NOT 'ok'."""
    return df[df["status"] != "ok"]


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class TSPVisualizer:

    def __init__(self, pkl_path: str | Path = "results/tsp_results.pkl") -> None:
        with open(pkl_path, "rb") as fh:
            self.results: dict = pickle.load(fh)
        self.df = _build_df(self.results)
        print(f"Loaded: {list(self.results.keys())}")
        print(f"DataFrame: {len(self.df)} rows  "
              f"| status counts:\n{self.df['status'].value_counts().to_string()}")

    def cases(self) -> list[str]:
        return list(self.results.keys())

    def _tdata(self, case: str, test_id: str) -> dict:
        return self.results[case][test_id]

    def _cdf(self, case: str, large_only: bool = False) -> pd.DataFrame:
        df = self.df[self.df["case"] == case].copy()
        if large_only:
            df = df[df["test_id"] != "test_0"]
        return df

    # ── 1. test_0 graph visualisation ────────────────────────────────────────

    def plot_all_paths(
        self,
        case: str,
        test_id: str = "test_0",
        figsize: tuple = (20, 10),
        directed: bool = True,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Draw the TSP result as a proper graph:
          • Nodes  — circles with city index labels
          • Edges  — one line per (tour[i] → tour[i+1]) step, closing the loop
          • For sparse: faint background edges show available connections
          • Failed tasks show just the node graph with a status stamp

        directed=True adds midpoint arrows to show tour direction.
        """
        tdata  = self._tdata(case, test_id)
        coords = _extract_coords(tdata)
        dm     = tdata["dist_matrix"]
        algos  = list(tdata["algorithms"].keys())
        n      = tdata["n_cities"]

        n_panels = len(algos) + 1        # +1 for best-tour overlay
        ncols    = 4
        nrows    = (n_panels + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0], figsize[1] * nrows / 2),
        )
        axes = np.array(axes).flatten()

        # ── helper: draw background graph for one axis ────────────────────────
        def _bg(ax):
            """Draw background: available edges (sparse) or nothing (others)."""
            if case == "sparse":
                _draw_sparse_edges(ax, coords, dm, alpha=0.18)

        # ── helper: draw failed-task panel ────────────────────────────────────
        def _draw_failed(ax, st: str):
            _bg(ax)
            # draw all nodes without a tour
            _draw_graph_nodes(ax, coords, node_color="#EEEEEE",
                              border_color="#AAAAAA", node_size=180)
            ax.text(0.5, 0.5, st.upper(),
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=18, color="red", fontweight="bold", alpha=0.6,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec="red", alpha=0.5))

        # ── best-tour panel ───────────────────────────────────────────────────
        ok_algos = [a for a in algos
                    if tdata["algorithms"][a].get("status") == "ok"]
        best_algo = (min(ok_algos,
                         key=lambda a: tdata["algorithms"][a]["best_fitness"])
                     if ok_algos else algos[0])
        best_res  = tdata["algorithms"][best_algo]
        ax0       = axes[0]

        _bg(ax0)
        if best_res.get("status") == "ok":
            _draw_tour_edges(ax0, coords, best_res["best_solution"],
                             color=ALGO_COLORS.get(best_algo, "#333"),
                             lw=2.0, directed=directed)
            _draw_graph_nodes(ax0, coords)
        else:
            _draw_failed(ax0, best_res.get("status", "?"))

        st   = best_res.get("status", "ok")
        bf   = best_res.get("best_fitness")
        bf_s = f"{bf:.1f}" if st == "ok" and bf is not None else st.upper()
        ax0.set_title(f"★ Best  [{ALGO_SHORT.get(best_algo, best_algo)}]\n"
                      f"dist = {bf_s}",
                      fontsize=10, fontweight="bold")
        ax0.set_aspect("equal", "box")
        ax0.axis("off")

        # ── per-algorithm panels ──────────────────────────────────────────────
        for idx, aname in enumerate(algos):
            ax    = axes[idx + 1]
            ares  = tdata["algorithms"][aname]
            color = ALGO_COLORS.get(aname, "gray")
            st    = ares.get("status", "ok")
            bf    = ares.get("best_fitness")
            t     = ares.get("execution_time") or 0.0
            bf_s  = f"{bf:.1f}" if st == "ok" and bf is not None else st.upper()

            _bg(ax)
            if st == "ok":
                _draw_tour_edges(ax, coords, ares["best_solution"],
                                 color=color, lw=1.7, directed=directed)
                _draw_graph_nodes(ax, coords)
            else:
                _draw_failed(ax, st)

            ax.set_title(
                f"{ALGO_SHORT.get(aname, aname)}\n"
                f"dist = {bf_s}   t = {t:.2f}s",
                fontsize=9,
                color="#CC0000" if st != "ok" else "black",
            )
            ax.set_aspect("equal", "box")
            ax.axis("off")

        for ax in axes[n_panels:]:
            ax.set_visible(False)

        coords_note = "" if tdata.get("coordinates") is not None else "  (layout via MDS)"
        fig.suptitle(
            f"TSP Graph  ·  {CASE_LABEL.get(case, case)}  ·  "
            f"{test_id}  ·  {n} cities{coords_note}",
            fontsize=13, fontweight="bold", y=1.01,
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 2. Failure statistics ─────────────────────────────────────────────────

    def plot_failure_stats(
        self,
        case: str,
        figsize: tuple = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Left  — stacked bar: ok / inf / timeout / error count per algorithm.
        Right — success rate (%) per algorithm, coloured by algorithm.
        """
        df = self._cdf(case, large_only=True)
        algo_order = sorted(df["algorithm"].unique())
        statuses   = ["ok", "inf", "timeout", "error"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # stacked bar
        bottoms = np.zeros(len(algo_order))
        for st in statuses:
            counts = [
                (df[(df["algorithm"] == a) & (df["status"] == st)]).shape[0]
                for a in algo_order
            ]
            ax1.bar([ALGO_SHORT.get(a, a) for a in algo_order],
                    counts, bottom=bottoms,
                    color=STATUS_COLORS[st], label=st, alpha=0.9)
            bottoms += np.array(counts)

        ax1.set_title("Result Status per Algorithm", fontsize=11)
        ax1.set_xlabel("Algorithm"); ax1.set_ylabel("Number of Tests")
        ax1.legend(fontsize=8)
        ax1.tick_params(axis="x", rotation=30)

        # success rate
        total = df.groupby("algorithm").size()
        ok    = df[df["status"] == "ok"].groupby("algorithm").size()
        rate  = (ok / total * 100).reindex(algo_order).fillna(0)

        bars = ax2.bar(
            [ALGO_SHORT.get(a, a) for a in algo_order],
            rate.values,
            color=[ALGO_COLORS.get(a, "gray") for a in algo_order],
            alpha=0.85,
        )
        for bar, val in zip(bars, rate.values):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

        ax2.set_ylim(0, 115)
        ax2.set_title("Success Rate (%)", fontsize=11)
        ax2.set_xlabel("Algorithm"); ax2.set_ylabel("% of tests solved")
        ax2.axhline(100, color="gray", linewidth=0.8, linestyle="--")
        ax2.tick_params(axis="x", rotation=30)

        fig.suptitle(
            f"Failure Statistics  ·  {CASE_LABEL.get(case, case)}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 3. Failure heatmap (algo × problem size) ──────────────────────────────

    def plot_failure_heatmap(
        self,
        case: str,
        figsize: tuple = (12, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Heatmap: rows = algorithm, columns = n_cities bins,
        cell value = failure rate (0–1).  Shows WHERE (by size) each
        algorithm starts to break down.
        """
        df = self._cdf(case, large_only=True).copy()
        df["failed"] = (df["status"] != "ok").astype(int)

        # bin n_cities into ~5 groups
        n_bins = min(5, df["n_cities"].nunique())
        df["size_bin"] = pd.cut(df["n_cities"], bins=n_bins)

        pivot = (
            df.groupby(["algorithm", "size_bin"])["failed"]
            .mean()
            .unstack("size_bin")
            .fillna(0)
        )

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=1)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [str(c) for c in pivot.columns], rotation=30, ha="right", fontsize=8
        )
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(
            [ALGO_SHORT.get(a, a) for a in pivot.index], fontsize=9
        )
        ax.set_xlabel("Problem Size (n_cities bins)")
        ax.set_ylabel("Algorithm")

        # annotate cells
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        fontsize=8,
                        color="white" if val > 0.6 else "black")

        plt.colorbar(im, ax=ax, label="Failure rate")
        ax.set_title(
            f"Failure Rate by Algorithm × Problem Size  ·  "
            f"{CASE_LABEL.get(case, case)}",
            fontsize=11, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 4. Scalability ────────────────────────────────────────────────────────

    def plot_scalability(
        self,
        case: str,
        figsize: tuple = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        df = _ok(self._cdf(case, large_only=True))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for aname, grp in df.groupby("algorithm"):
            grp = grp.sort_values("n_cities")
            c   = ALGO_COLORS.get(aname, "gray")
            mk  = ALGO_MARKERS.get(aname, "o")
            lb  = ALGO_SHORT.get(aname, aname)
            ax1.plot(grp["n_cities"], grp["execution_time"],
                     marker=mk, color=c, linewidth=2, markersize=6, label=lb)
            ax2.plot(grp["n_cities"], grp["best_fitness"],
                     marker=mk, color=c, linewidth=2, markersize=6, label=lb)

        # annotate failure counts per algorithm on ax2
        fail_df  = _failed(self._cdf(case, large_only=True))
        fail_cnt = fail_df.groupby("algorithm").size()
        for aname, cnt in fail_cnt.items():
            ax2.annotate(f"✗{cnt}", xy=(0.02, 0.98),
                         xycoords="axes fraction",
                         fontsize=7, color="red", va="top")

        ax1.set_title("Execution Time vs Cities", fontsize=11)
        ax1.set_xlabel("Cities"); ax1.set_ylabel("Time (s)")
        ax1.legend(fontsize=8)
        ax2.set_title("Solution Quality vs Cities\n(ok results only)", fontsize=11)
        ax2.set_xlabel("Cities"); ax2.set_ylabel("Tour Distance")
        ax2.legend(fontsize=8)

        fig.suptitle(f"Scalability  ·  {CASE_LABEL.get(case, case)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 5. Complexity ─────────────────────────────────────────────────────────

    def plot_complexity(
        self,
        case: str,
        figsize: tuple = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        df = self._cdf(case, large_only=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        algo_order = sorted(df["algorithm"].unique())

        # mean time — include timeout tasks (they hit the cap, that IS the time)
        means = df.groupby("algorithm")["execution_time"].mean()
        for aname in algo_order:
            c  = ALGO_COLORS.get(aname, "gray")
            lb = ALGO_SHORT.get(aname, aname)
            ax1.bar(lb, means.get(aname, 0), color=c, alpha=0.85, edgecolor="white")

        ax1.set_title("Mean Execution Time (all tasks)", fontsize=11)
        ax1.set_xlabel("Algorithm"); ax1.set_ylabel("Mean Time (s)")
        ax1.tick_params(axis="x", rotation=30)

        has_mem = False
        for aname, grp in _ok(df).groupby("algorithm"):
            grp = grp.sort_values("n_cities")
            mem = grp["peak_memory_kb"].dropna()
            if not mem.empty:
                has_mem = True
                ax2.plot(grp.loc[mem.index, "n_cities"], mem,
                         marker=ALGO_MARKERS.get(aname, "o"),
                         color=ALGO_COLORS.get(aname, "gray"),
                         linewidth=2, markersize=5,
                         label=ALGO_SHORT.get(aname, aname))
        if not has_mem:
            ax2.text(0.5, 0.5, "No memory data",
                     ha="center", va="center",
                     transform=ax2.transAxes, color="gray")
        ax2.set_title("Peak Memory vs Cities (ok only)", fontsize=11)
        ax2.set_xlabel("Cities"); ax2.set_ylabel("Peak Memory (KB)")
        ax2.legend(fontsize=8)

        fig.suptitle(f"Complexity  ·  {CASE_LABEL.get(case, case)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 6. Solution quality ───────────────────────────────────────────────────

    def plot_solution_quality(
        self,
        case: str,
        figsize: tuple = (14, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        df_ok   = _ok(self._cdf(case, large_only=True))
        df_fail = _failed(self._cdf(case, large_only=True))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        algo_order = sorted(self._cdf(case, large_only=True)["algorithm"].unique())

        data_boxes = [df_ok[df_ok["algorithm"] == a]["best_fitness"].values
                      for a in algo_order]
        bp = ax1.boxplot(data_boxes, patch_artist=True,
                         medianprops={"color": "black", "linewidth": 2})
        for patch, aname in zip(bp["boxes"], algo_order):
            patch.set_facecolor(ALGO_COLORS.get(aname, "gray"))
            patch.set_alpha(0.8)

        # annotate failure count above each box
        fail_cnt = df_fail.groupby("algorithm").size()
        for i, aname in enumerate(algo_order):
            cnt = fail_cnt.get(aname, 0)
            if cnt:
                ax1.text(i + 1, ax1.get_ylim()[1],
                         f"✗{cnt}", ha="center", va="bottom",
                         fontsize=8, color="red")

        ax1.set_xticks(range(1, len(algo_order) + 1))
        ax1.set_xticklabels([ALGO_SHORT.get(a, a) for a in algo_order],
                            rotation=30)
        ax1.set_title("Tour Distance Distribution\n(ok results only)", fontsize=11)
        ax1.set_ylabel("Tour Distance")

        stats = (df_ok.groupby("algorithm")["best_fitness"]
                 .agg(mean="mean", best="min")
                 .reindex(algo_order))
        x     = np.arange(len(algo_order))
        width = 0.35
        for i, aname in enumerate(algo_order):
            c = ALGO_COLORS.get(aname, "gray")
            ax2.bar(x[i] - width/2, stats.loc[aname, "mean"] if aname in stats.index else 0,
                    width, color=c, alpha=0.55)
            ax2.bar(x[i] + width/2, stats.loc[aname, "best"] if aname in stats.index else 0,
                    width, color=c, alpha=1.0)
        ax2.set_xticks(x)
        ax2.set_xticklabels([ALGO_SHORT.get(a, a) for a in algo_order], rotation=30)
        ax2.set_title("Mean vs Best Distance", fontsize=11)
        ax2.set_ylabel("Tour Distance")
        ax2.legend(
            handles=[mpatches.Patch(facecolor="gray", alpha=0.55, label="Mean"),
                     mpatches.Patch(facecolor="gray", alpha=1.0,  label="Best")],
            fontsize=8,
        )

        fig.suptitle(f"Solution Quality  ·  {CASE_LABEL.get(case, case)}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 7. Convergence ────────────────────────────────────────────────────────

    def plot_convergence(
        self,
        case: str,
        test_id: str = "test_1",
        figsize: tuple = (12, 5),
        save_path: str | None = None,
    ) -> plt.Figure:
        tdata = self._tdata(case, test_id)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for aname, ares in tdata["algorithms"].items():
            color = ALGO_COLORS.get(aname, "gray")
            label = ALGO_SHORT.get(aname, aname)
            st    = ares.get("status", "ok")
            curve = ares.get("convergence")
            final = ares.get("best_fitness") or 0

            if st != "ok":
                for ax in (ax1, ax2):
                    ax.axhline(0, color=color, linewidth=1, linestyle=":",
                               label=f"{label} [{st}]", alpha=0.5)
                continue

            if curve is not None and len(curve) > 1:
                iters = np.arange(len(curve))
                ax1.plot(iters, curve, color=color, linewidth=2, label=label)
                ax2.semilogy(iters, [max(v, 1e-9) for v in curve],
                             color=color, linewidth=2, label=label)
            else:
                for ax in (ax1, ax2):
                    ax.axhline(max(final, 1e-9), color=color, linewidth=1.5,
                               linestyle="--", label=f"{label} (no curve)")

        for ax in (ax1, ax2):
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Best Tour Distance")
            ax.legend(fontsize=8)

        ax1.set_title("Convergence (linear)", fontsize=11)
        ax2.set_title("Convergence (log scale)", fontsize=11)
        fig.suptitle(
            f"Convergence  ·  {CASE_LABEL.get(case, case)}  ·  {test_id}",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 8. 5-panel dashboard (adds failure panel) ─────────────────────────────

    def plot_large_test_dashboard(
        self,
        case: str,
        figsize: tuple = (20, 12),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        2×3 dashboard:
          [Time vs n]  [Quality vs n]  [Failure stacked bar]
          [Memory]     [Box-plot]      [Success rate]
        """
        df_all = self._cdf(case, large_only=True)
        df     = _ok(df_all)

        fig = plt.figure(figsize=figsize)
        gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.32)
        ax_time  = fig.add_subplot(gs[0, 0])
        ax_qual  = fig.add_subplot(gs[0, 1])
        ax_fail  = fig.add_subplot(gs[0, 2])
        ax_mem   = fig.add_subplot(gs[1, 0])
        ax_box   = fig.add_subplot(gs[1, 1])
        ax_rate  = fig.add_subplot(gs[1, 2])

        algo_order = sorted(df_all["algorithm"].unique())

        # time vs n
        for aname, grp in df_all.groupby("algorithm"):
            grp = grp.sort_values("n_cities")
            ax_time.plot(grp["n_cities"], grp["execution_time"],
                         marker=ALGO_MARKERS.get(aname, "o"),
                         color=ALGO_COLORS.get(aname, "gray"),
                         linewidth=2, markersize=5,
                         label=ALGO_SHORT.get(aname, aname))
        ax_time.set_title("Time vs Cities", fontsize=10)
        ax_time.set_xlabel("Cities"); ax_time.set_ylabel("Time (s)")
        ax_time.legend(fontsize=7)

        # quality vs n (ok only)
        for aname, grp in df.groupby("algorithm"):
            grp = grp.sort_values("n_cities")
            ax_qual.plot(grp["n_cities"], grp["best_fitness"],
                         marker=ALGO_MARKERS.get(aname, "o"),
                         color=ALGO_COLORS.get(aname, "gray"),
                         linewidth=2, markersize=5,
                         label=ALGO_SHORT.get(aname, aname))
        ax_qual.set_title("Quality vs Cities (ok only)", fontsize=10)
        ax_qual.set_xlabel("Cities"); ax_qual.set_ylabel("Tour Distance")
        ax_qual.legend(fontsize=7)

        # failure stacked bar
        statuses = ["ok", "inf", "timeout", "error"]
        bottoms  = np.zeros(len(algo_order))
        for st in statuses:
            counts = [df_all[(df_all["algorithm"] == a)
                             & (df_all["status"] == st)].shape[0]
                      for a in algo_order]
            ax_fail.bar([ALGO_SHORT.get(a, a) for a in algo_order],
                        counts, bottom=bottoms,
                        color=STATUS_COLORS[st], label=st, alpha=0.9)
            bottoms += np.array(counts)
        ax_fail.set_title("Result Status", fontsize=10)
        ax_fail.set_xlabel("Algorithm"); ax_fail.set_ylabel("Count")
        ax_fail.legend(fontsize=7); ax_fail.tick_params(axis="x", rotation=30)

        # memory vs n (ok only)
        has_mem = False
        for aname, grp in df.groupby("algorithm"):
            grp = grp.sort_values("n_cities")
            mem = grp["peak_memory_kb"].dropna()
            if not mem.empty:
                has_mem = True
                ax_mem.plot(grp.loc[mem.index, "n_cities"], mem,
                            marker=ALGO_MARKERS.get(aname, "o"),
                            color=ALGO_COLORS.get(aname, "gray"),
                            linewidth=2, markersize=5,
                            label=ALGO_SHORT.get(aname, aname))
        if not has_mem:
            ax_mem.text(0.5, 0.5, "No memory data",
                        ha="center", va="center",
                        transform=ax_mem.transAxes, color="gray")
        ax_mem.set_title("Memory vs Cities (ok only)", fontsize=10)
        ax_mem.set_xlabel("Cities"); ax_mem.set_ylabel("Peak Memory (KB)")
        ax_mem.legend(fontsize=7)

        # box-plot (ok only)
        data_boxes = [df[df["algorithm"] == a]["best_fitness"].dropna().values
                      for a in algo_order]
        bp = ax_box.boxplot(data_boxes, patch_artist=True,
                            medianprops={"color": "black", "linewidth": 2})
        for patch, aname in zip(bp["boxes"], algo_order):
            patch.set_facecolor(ALGO_COLORS.get(aname, "gray"))
            patch.set_alpha(0.8)
        ax_box.set_xticks(range(1, len(algo_order) + 1))
        ax_box.set_xticklabels([ALGO_SHORT.get(a, a) for a in algo_order],
                               rotation=30, fontsize=8)
        ax_box.set_title("Distance Distribution (ok only)", fontsize=10)
        ax_box.set_ylabel("Tour Distance")

        # success rate
        total = df_all.groupby("algorithm").size()
        ok_c  = df_all[df_all["status"] == "ok"].groupby("algorithm").size()
        rate  = (ok_c / total * 100).reindex(algo_order).fillna(0)
        bars  = ax_rate.bar(
            [ALGO_SHORT.get(a, a) for a in algo_order],
            rate.values,
            color=[ALGO_COLORS.get(a, "gray") for a in algo_order],
            alpha=0.85,
        )
        for bar, val in zip(bars, rate.values):
            ax_rate.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 1,
                         f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
        ax_rate.set_ylim(0, 115)
        ax_rate.set_title("Success Rate (%)", fontsize=10)
        ax_rate.set_xlabel("Algorithm"); ax_rate.set_ylabel("% solved")
        ax_rate.axhline(100, color="gray", linewidth=0.8, linestyle="--")
        ax_rate.tick_params(axis="x", rotation=30)

        case_color = CASE_COLOR.get(case, "#333")
        fig.suptitle(f"TSP Dashboard  ·  {CASE_LABEL.get(case, case)}",
                     fontsize=14, fontweight="bold", y=1.01, color=case_color)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 9. Cross-case comparison ──────────────────────────────────────────────

    def plot_cross_case_comparison(
        self,
        figsize: tuple = (18, 14),
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        3 rows (one per case) × 3 columns:
          Time vs n  |  Quality vs n  |  Success rate bar
        """
        cases = [c for c in ("sparse", "clustered", "euclidean")
                 if c in self.results]

        fig, axes = plt.subplots(len(cases), 3, figsize=figsize)
        if len(cases) == 1:
            axes = axes[np.newaxis, :]

        for row, case in enumerate(cases):
            df_all = self._cdf(case, large_only=True)
            df     = _ok(df_all)
            ax_t, ax_q, ax_r = axes[row]
            case_c = CASE_COLOR.get(case, "#333")

            for aname, grp in df_all.groupby("algorithm"):
                grp = grp.sort_values("n_cities")
                c   = ALGO_COLORS.get(aname, "gray")
                mk  = ALGO_MARKERS.get(aname, "o")
                lb  = ALGO_SHORT.get(aname, aname)
                ax_t.plot(grp["n_cities"], grp["execution_time"],
                          marker=mk, color=c, linewidth=1.8, markersize=4,
                          label=lb)

            for aname, grp in df.groupby("algorithm"):
                grp = grp.sort_values("n_cities")
                ax_q.plot(grp["n_cities"], grp["best_fitness"],
                          marker=ALGO_MARKERS.get(aname, "o"),
                          color=ALGO_COLORS.get(aname, "gray"),
                          linewidth=1.8, markersize=4,
                          label=ALGO_SHORT.get(aname, aname))

            algo_order = sorted(df_all["algorithm"].unique())
            total = df_all.groupby("algorithm").size()
            ok_c  = df_all[df_all["status"] == "ok"].groupby("algorithm").size()
            rate  = (ok_c / total * 100).reindex(algo_order).fillna(0)
            bars  = ax_r.bar(
                [ALGO_SHORT.get(a, a) for a in algo_order],
                rate.values,
                color=[ALGO_COLORS.get(a, "gray") for a in algo_order],
                alpha=0.85,
            )
            for bar, val in zip(bars, rate.values):
                if val < 100:
                    ax_r.text(bar.get_x() + bar.get_width() / 2,
                              bar.get_height() + 1,
                              f"{val:.0f}%", ha="center", va="bottom", fontsize=7,
                              color="red")
            ax_r.set_ylim(0, 115)
            ax_r.axhline(100, color="gray", linewidth=0.8, linestyle="--")
            ax_r.tick_params(axis="x", rotation=30)

            label = CASE_LABEL.get(case, case)
            ax_t.set_title(f"{label} — Time vs Cities",    fontsize=10,
                           color=case_c, fontweight="bold")
            ax_q.set_title(f"{label} — Quality vs Cities", fontsize=10,
                           color=case_c, fontweight="bold")
            ax_r.set_title(f"{label} — Success Rate",      fontsize=10,
                           color=case_c, fontweight="bold")
            ax_t.set_ylabel("Time (s)", fontsize=9)
            ax_q.set_ylabel("Tour Distance", fontsize=9)
            ax_r.set_ylabel("% solved", fontsize=9)

            if row == len(cases) - 1:
                for ax in (ax_t, ax_q, ax_r):
                    ax.set_xlabel("Cities" if ax != ax_r else "Algorithm",
                                  fontsize=9)
            ax_t.legend(fontsize=6); ax_q.legend(fontsize=6)

        fig.suptitle(
            "Cross-Case Comparison  ·  Sparse / Clustered / Euclidean",
            fontsize=14, fontweight="bold",
        )
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ── 10. Summary table ─────────────────────────────────────────────────────

    def summary_table(self, case: str, large_only: bool = True) -> pd.DataFrame:
        """
        Tidy table with quality metrics (ok rows only) + failure counts.
        """
        df_all = self._cdf(case, large_only=large_only)
        df_ok  = _ok(df_all)

        quality = (
            df_ok.groupby("algorithm")["best_fitness"]
            .agg(mean_dist="mean", std_dist="std", min_dist="min")
        )
        timing = df_all.groupby("algorithm")["execution_time"].mean().rename("mean_time_s")
        memory = df_ok.groupby("algorithm")["peak_memory_kb"].mean().rename("mean_mem_kb")
        n_ok      = df_ok.groupby("algorithm").size().rename("n_ok")
        n_timeout = (df_all[df_all["status"] == "timeout"]
                     .groupby("algorithm").size().rename("n_timeout"))
        n_inf     = (df_all[df_all["status"] == "inf"]
                     .groupby("algorithm").size().rename("n_inf"))
        n_error   = (df_all[df_all["status"] == "error"]
                     .groupby("algorithm").size().rename("n_error"))

        tbl = (pd.concat([quality, timing, memory,
                           n_ok, n_timeout, n_inf, n_error], axis=1)
               .fillna(0)
               .sort_values("mean_dist"))
        tbl.index = [ALGO_SHORT.get(i, i) for i in tbl.index]
        return tbl.round(2)
