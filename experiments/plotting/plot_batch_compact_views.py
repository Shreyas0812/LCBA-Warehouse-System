"""
plot_batch_compact_views.py
===========================
Compact alternatives to many batch slice plots.

This script summarizes a batch run with a small number of figures:
1. Heatmaps per metric, one subplot per algorithm.
2. Critical-load curve: max task count that still achieves a completion
   threshold, plotted against comm_range.
3. Tradeoff scatter: completion rate vs wall time.
4. Completion-vs-load degradation curves, one subplot per comm_range.
5. Completed-only makespan-vs-load curves, one subplot per comm_range.

Outputs are written directly to:
    results/experiments/<map_name>/<timestamp>/batch_results/compact_views/

Usage:
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --all --save
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --heatmaps --save
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --capacity --save
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --tradeoff --save
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --degradation --save
    python experiments/plotting/plot_batch_compact_views.py --csv path/to/summary.csv --makespan-curves --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALG_ORDER = ["gcbba", "cbba", "dmchba", "sga"]
ALG_LABELS = {"gcbba": "LCBA", "cbba": "CBBA", "dmchba": "DMCHBA", "sga": "SGA"}
ALG_COLORS = {"gcbba": "#1f77b4", "cbba": "#ff7f0e", "dmchba": "#2ca02c", "sga": "#d62728"}

MIN_SEEDS_DEFAULT = 2

HEATMAP_METRICS = [
    ("completion_rate", "Completion Rate"),
    ("timeout_rate", "Timeout Rate"),
    ("makespan_completed", "Makespan (completed runs only)"),
    ("wall_time_seconds", "Wall Time (s)"),
]


def infer_map_name(csv_path: str) -> str:
    parts = os.path.normpath(csv_path).split(os.sep)
    for i, part in enumerate(parts):
        if part == "experiments" and i + 2 < len(parts):
            return parts[i + 1]
    return "unknown_map"


def infer_run_timestamp(csv_path: str) -> str:
    parts = os.path.normpath(csv_path).split(os.sep)
    for i, part in enumerate(parts):
        if part == "experiments" and i + 2 < len(parts):
            return parts[i + 2]
    return os.path.basename(os.path.dirname(csv_path))


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_output_root(csv_path: str) -> str:
    return os.path.join(
        os.path.dirname(csv_path),
        "batch_results",
        "compact_views",
    )


def load_batch(csv_path: str, include_wall_clock_timeouts: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["experiment_type"] == "batch"].copy()
    if df.empty:
        return df

    denom = df["num_tasks_total"].replace(0, pd.NA)
    df["completion_rate"] = df["num_tasks_completed"] / denom
    df["timeout_rate"] = df["hit_wall_clock_ceiling"].astype(float)
    df["makespan_completed"] = df["makespan"].where(df["all_tasks_completed"], pd.NA)

    if not include_wall_clock_timeouts:
        df = df[~df["hit_wall_clock_ceiling"]].copy()

    return df


def mean_with_min_seeds(df: pd.DataFrame, keys: list, metric: str, min_seeds: int) -> pd.DataFrame:
    grp = df.groupby(keys)
    counts = grp[metric].count().reset_index(name="_n")
    means = grp[metric].mean().reset_index()
    out = means.merge(counts, on=keys)
    return out[out["_n"] >= min_seeds].drop(columns="_n")


def write_run_manifest(output_dir: str, command_text: str, metadata: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "run_manifest.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Batch Plot Run Manifest\n")
        f.write("=======================\n")
        f.write(f"command: {command_text}\n")
        for key in sorted(metadata.keys()):
            f.write(f"{key}: {metadata[key]}\n")


def save_or_show(fig, out_path: str, save: bool):
    fig.tight_layout()
    if save:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_heatmaps(df: pd.DataFrame, metric: str, metric_label: str, out_dir: str, map_name: str, save: bool, min_seeds: int):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "initial_tasks", "comm_range"],
        metric,
        min_seeds,
    )

    algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    if not algs:
        print(f"No data for heatmap metric: {metric}")
        return

    # Keep a fixed axis grid across all algorithm subplots in this figure,
    # even when some cells are missing after seed filtering.
    task_axis = sorted(df["initial_tasks"].dropna().unique())
    comm_axis = sorted(df["comm_range"].dropna().unique())
    if not task_axis or not comm_axis:
        print(f"No axis values available for heatmap metric: {metric}")
        return

    # Use one color scale across all methods for fair visual comparison.
    if metric in ("completion_rate", "timeout_rate"):
        global_vmin, global_vmax = 0.0, 1.0
    else:
        metric_vals = agg[metric].dropna()
        if metric_vals.empty:
            print(f"No finite data for heatmap metric: {metric}")
            return
        global_vmin = float(metric_vals.min())
        global_vmax = float(metric_vals.max())
        if global_vmin == global_vmax:
            pad = max(1e-9, abs(global_vmin) * 0.01)
            global_vmin -= pad
            global_vmax += pad

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    fig.suptitle(
        f"{metric_label} Heatmaps - {map_name}\n(gray cells: missing data)",
        fontsize=13,
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#e6e6e6")

    for idx in range(4):
        ax = axes[idx // 2][idx % 2]
        if idx >= len(algs):
            ax.axis("off")
            continue

        alg = algs[idx]
        sub = agg[agg["allocation_method"] == alg]
        piv = (
            sub.pivot(index="initial_tasks", columns="comm_range", values=metric)
            .reindex(index=task_axis, columns=comm_axis)
            .sort_index()
            .sort_index(axis=1)
        )
        if piv.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ALG_LABELS.get(alg, alg))
            ax.axis("off")
            continue

        im = ax.imshow(
            np.ma.masked_invalid(piv.values),
            aspect="auto",
            origin="lower",
            vmin=global_vmin,
            vmax=global_vmax,
            cmap=cmap,
        )

        # Annotate each heatmap cell with the metric value.
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.iloc[r, c]
                if pd.isna(v):
                    continue

                if metric in ("completion_rate", "timeout_rate"):
                    label = f"{float(v):.2f}"
                elif metric == "wall_time_seconds":
                    label = f"{float(v):.1f}"
                else:
                    label = f"{float(v):.0f}"

                rgba = im.cmap(im.norm(float(v)))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"

                ax.text(c, r, label, ha="center", va="center", color=text_color, fontsize=7)

        ax.set_title(ALG_LABELS.get(alg, alg), fontsize=10)
        ax.set_xlabel("comm range", fontsize=9)
        ax.set_ylabel("initial tasks", fontsize=9)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([str(c) for c in piv.columns], fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([str(r) for r in piv.index], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = os.path.join(out_dir, f"compact_heatmap_{metric}.png")
    save_or_show(fig, out, save)


def plot_capacity_curve(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int, completion_threshold: float):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "initial_tasks"],
        "completion_rate",
        min_seeds,
    )
    if agg.empty:
        print("No data for capacity curve.")
        return

    rows = []
    for (alg, cr), sub in agg.groupby(["allocation_method", "comm_range"]):
        ok = sub[sub["completion_rate"] >= completion_threshold]
        max_tc = ok["initial_tasks"].max() if not ok.empty else 0
        rows.append({"allocation_method": alg, "comm_range": cr, "max_completed_task_count": max_tc})

    cap = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    present_algs = [a for a in ALG_ORDER if a in cap["allocation_method"].unique()]
    # Draw LCBA last so it stays visible when curves overlap.
    draw_order = [a for a in present_algs if a != "gcbba"] + (["gcbba"] if "gcbba" in present_algs else [])

    for alg in draw_order:
        sub = cap[cap["allocation_method"] == alg].sort_values("comm_range")
        is_lcba = alg == "gcbba"
        ax.plot(
            sub["comm_range"],
            sub["max_completed_task_count"],
            marker="o",
            label=ALG_LABELS.get(alg, alg),
            color=ALG_COLORS.get(alg),
            linewidth=2.8 if is_lcba else 2.2,
            markersize=10 if is_lcba else 8,
            markeredgecolor="black" if is_lcba else None,
            markeredgewidth=1.3 if is_lcba else 0.0,
            zorder=5 if is_lcba else 3,
        )

    ax.set_title(f"Critical Load vs Comm Range - {map_name}", fontsize=12)
    ax.set_xlabel("comm range")
    ax.set_ylabel(f"max task count with completion >= {completion_threshold:.2f}")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out = os.path.join(out_dir, "compact_capacity_curve.png")
    save_or_show(fig, out, save)


def plot_tradeoff(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    comp = mean_with_min_seeds(
        df,
        ["allocation_method", "initial_tasks", "comm_range"],
        "completion_rate",
        min_seeds,
    )
    wall = mean_with_min_seeds(
        df,
        ["allocation_method", "initial_tasks", "comm_range"],
        "wall_time_seconds",
        min_seeds,
    )
    merged = comp.merge(
        wall,
        on=["allocation_method", "initial_tasks", "comm_range"],
        suffixes=("_completion", "_wall"),
    )
    if merged.empty:
        print("No data for tradeoff plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in [a for a in ALG_ORDER if a in merged["allocation_method"].unique()]:
        sub = merged[merged["allocation_method"] == alg]
        ax.scatter(
            sub["wall_time_seconds"],
            sub["completion_rate"],
            s=45,
            alpha=0.8,
            label=ALG_LABELS.get(alg, alg),
            color=ALG_COLORS.get(alg),
        )

    ax.set_title(f"Completion vs Wall Time Tradeoff - {map_name}", fontsize=12)
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("completion rate")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()

    out = os.path.join(out_dir, "compact_tradeoff.png")
    save_or_show(fig, out, save)


def _subplot_grid(n_panels: int, ncols: int = 3):
    ncols = min(ncols, max(1, n_panels))
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    return fig, axes, nrows, ncols


def _line_style_for_alg(alg: str) -> dict:
    is_lcba = alg == "gcbba"
    return {
        "linewidth": 2.8 if is_lcba else 2.1,
        "markersize": 9 if is_lcba else 6,
        "markeredgecolor": "black" if is_lcba else None,
        "markeredgewidth": 1.2 if is_lcba else 0.0,
        "zorder": 5 if is_lcba else 3,
    }


def plot_completion_degradation(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "initial_tasks"],
        "completion_rate",
        min_seeds,
    )
    if agg.empty:
        print("No data for completion degradation curves.")
        return

    comm_ranges = sorted(agg["comm_range"].unique())
    fig, axes, nrows, ncols = _subplot_grid(len(comm_ranges), ncols=3)
    fig.suptitle(f"Completion vs Load by Comm Range - {map_name}", fontsize=13)

    present_algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    draw_order = [a for a in present_algs if a != "gcbba"] + (["gcbba"] if "gcbba" in present_algs else [])

    for idx, cr in enumerate(comm_ranges):
        ax = axes[idx // ncols][idx % ncols]
        sub_cr = agg[agg["comm_range"] == cr]

        for alg in draw_order:
            sub = sub_cr[sub_cr["allocation_method"] == alg].sort_values("initial_tasks")
            if sub.empty:
                continue
            ax.plot(
                sub["initial_tasks"],
                sub["completion_rate"],
                marker="o",
                label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg),
                **_line_style_for_alg(alg),
            )

        ax.set_title(f"comm range = {cr}", fontsize=10)
        ax.set_xlabel("initial tasks")
        ax.set_ylabel("completion rate")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(comm_ranges), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    out = os.path.join(out_dir, "compact_completion_degradation_by_comm_range.png")
    save_or_show(fig, out, save)


def plot_makespan_curves(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "initial_tasks"],
        "makespan_completed",
        min_seeds,
    )
    if agg.empty:
        print("No completed-run data for makespan curves.")
        return

    comm_ranges = sorted(df["comm_range"].unique())
    fig, axes, nrows, ncols = _subplot_grid(len(comm_ranges), ncols=3)
    fig.suptitle(f"Completed-Only Makespan vs Load by Comm Range - {map_name}", fontsize=13)

    present_algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    draw_order = [a for a in present_algs if a != "gcbba"] + (["gcbba"] if "gcbba" in present_algs else [])

    for idx, cr in enumerate(comm_ranges):
        ax = axes[idx // ncols][idx % ncols]
        sub_cr = agg[agg["comm_range"] == cr]

        if sub_cr.empty:
            ax.text(0.5, 0.5, "No completed runs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"comm range = {cr}", fontsize=10)
            ax.set_xlabel("initial tasks")
            ax.set_ylabel("makespan (timesteps)")
            ax.grid(axis="y", alpha=0.3)
            continue

        for alg in draw_order:
            sub = sub_cr[sub_cr["allocation_method"] == alg].sort_values("initial_tasks")
            if sub.empty:
                continue
            ax.plot(
                sub["initial_tasks"],
                sub["makespan_completed"],
                marker="o",
                label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg),
                **_line_style_for_alg(alg),
            )

        ax.set_title(f"comm range = {cr}", fontsize=10)
        ax.set_xlabel("initial tasks")
        ax.set_ylabel("makespan (timesteps)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(comm_ranges), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.text(
        0.01,
        0.01,
        "Note: makespan is reported only for runs that completed all tasks.",
        fontsize=8,
        color="gray",
        va="bottom",
    )

    out = os.path.join(out_dir, "compact_makespan_completed_by_comm_range.png")
    save_or_show(fig, out, save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to summary.csv")
    parser.add_argument("--save", action="store_true", help="Save PNGs instead of showing")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: results/experiments/<map>/<timestamp>/batch_results/compact_views)")
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS_DEFAULT)
    parser.add_argument("--include-wall-clock-timeouts", action="store_true")
    parser.add_argument("--completion-threshold", type=float, default=0.95)

    parser.add_argument("--all", action="store_true", help="Generate all compact views")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--tradeoff", action="store_true")
    parser.add_argument("--degradation", action="store_true", help="Completion-vs-load curves by comm range")
    parser.add_argument("--makespan-curves", action="store_true", help="Completed-only makespan-vs-load curves by comm range")

    args = parser.parse_args()

    df = load_batch(args.csv, include_wall_clock_timeouts=args.include_wall_clock_timeouts)
    if df.empty:
        print("No batch rows found after filtering.")
        return

    map_name = infer_map_name(args.csv)
    ts = infer_run_timestamp(args.csv)
    out_dir = args.output_dir or default_output_root(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    write_run_manifest(
        out_dir,
        "python " + " ".join(os.sys.argv),
        {
            "csv": args.csv,
            "map_name": map_name,
            "timestamp": ts,
            "min_seeds": args.min_seeds,
            "include_wall_clock_timeouts": args.include_wall_clock_timeouts,
            "completion_threshold": args.completion_threshold,
            "mode": "compact",
            "save": args.save,
        },
    )

    do_all = args.all or not (
        args.heatmaps or args.capacity or args.tradeoff or args.degradation or args.makespan_curves
    )

    if do_all or args.heatmaps:
        for metric, label in HEATMAP_METRICS:
            plot_heatmaps(df, metric, label, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.capacity:
        plot_capacity_curve(df, out_dir, map_name, args.save, args.min_seeds, args.completion_threshold)

    if do_all or args.tradeoff:
        plot_tradeoff(df, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.degradation:
        plot_completion_degradation(df, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.makespan_curves:
        plot_makespan_curves(df, out_dir, map_name, args.save, args.min_seeds)


if __name__ == "__main__":
    main()