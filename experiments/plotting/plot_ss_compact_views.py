"""
plot_ss_compact_views.py
========================
Publication-ready compact views for steady-state experiments.

This is a compact-focused variant of plot_ss_results.py, optimized for
publication figures. Generates the same plot types with consistent styling.

Structure:
1. Heatmaps per metric, one subplot per algorithm.
   - Primary axes: task_arrival_rate (X) × comm_range (Y)
2. Stability curve: max arrival rate achieving steady throughput,
   vs comm_range (one line per algorithm).
3. Tradeoff scatter: throughput (X) vs avg task wait time (Y).
4. Throughput degradation curves, grouped by comm_range.
5. Wait-time degradation curves, grouped by comm_range.
6. Completion-progress curves: steady-state tasks completed and total steps
    vs comm_range, split by stop reason.

Outputs are written to:
    results/experiments/<map_name>/<timestamp>/ss_results/compact_views/

Usage:
    python experiments/plotting/plot_ss_compact_views.py --csv path/to/summary.csv --all --save
    python experiments/plotting/plot_ss_compact_views.py --csv path/to/summary.csv --heatmaps --save
    python experiments/plotting/plot_ss_compact_views.py --csv path/to/summary.csv --capacity --save
    python experiments/plotting/plot_ss_compact_views.py --csv path/to/summary.csv --tradeoff --save
    python experiments/plotting/plot_ss_compact_views.py --csv path/to/summary.csv --degradation --save
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

ALG_ORDER = ["gcbba", "cbba", "dmchba", "sga"]
ALG_LABELS = {"gcbba": "LCBA", "cbba": "CBBA", "dmchba": "DMCHBA", "sga": "SGA"}
ALG_COLORS = {"gcbba": "#1f77b4", "cbba": "#ff7f0e", "dmchba": "#2ca02c", "sga": "#d62728"}

MIN_SEEDS_DEFAULT = 2

# Select 4 key metrics for publication (not all 6)
HEATMAP_METRICS = [
    ("throughput", "Throughput (tasks/step)"),
    ("wait_time", "Avg Task Wait Time (steps)"),
    ("completion_rate", "Completion Rate"),
    ("queue_depth", "Avg Queue Depth"),
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
        "ss_results",
        "compact_views",
    )


def load_ss(csv_path: str, include_wall_clock_timeouts: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["experiment_type"] == "steady_state"].copy()
    if df.empty:
        return df

    # Compute steady-state specific metrics
    df["throughput"] = df["steady_state_tasks_completed"] / df["total_steps"].replace(0, pd.NA)
    df["wait_time"] = df["avg_task_wait_time"]
    df["completion_rate"] = df["steady_state_tasks_completed"] / df["total_tasks_injected"].replace(0, pd.NA)
    df["queue_depth"] = df["avg_queue_depth"]
    df["idle_ratio"] = df["avg_idle_ratio"]
    df["allocation_time_ms"] = df["avg_allocation_time_ms"]

    if not include_wall_clock_timeouts:
        df = df[~df["hit_wall_clock_ceiling"]].copy()

    return df


def filter_method(df: pd.DataFrame, allocation_method: str | None) -> pd.DataFrame:
    if allocation_method:
        return df[df["allocation_method"] == allocation_method].copy()
    return df


def mean_with_min_seeds(df: pd.DataFrame, keys: list, metric: str, min_seeds: int) -> pd.DataFrame:
    grp = df.groupby(keys)
    counts = grp[metric].count().reset_index(name="_n")
    means = grp[metric].mean().reset_index()
    out = means.merge(counts, on=keys)
    return out[out["_n"] >= min_seeds].drop(columns="_n")


def write_run_manifest(output_dir: str, command_text: str, metadata: dict) -> None:
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "run_manifest_compact.txt")
    with open(manifest_path, "w", encoding="utf-8") as f:
        f.write("Steady-State Compact Plot Run Manifest\n")
        f.write("========================================\n")
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
        ["allocation_method", "task_arrival_rate", "comm_range"],
        metric,
        min_seeds,
    )

    algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    if not algs:
        print(f"No data for heatmap metric: {metric}")
        return

    # Use one color scale across all methods for fair visual comparison.
    if metric == "completion_rate":
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
    fig.suptitle(f"{metric_label} Heatmaps - {map_name}", fontsize=13)

    for idx in range(4):
        ax = axes[idx // 2][idx % 2]
        if idx >= len(algs):
            ax.axis("off")
            continue

        alg = algs[idx]
        sub = agg[agg["allocation_method"] == alg]
        piv = sub.pivot(index="task_arrival_rate", columns="comm_range", values=metric).sort_index().sort_index(axis=1)
        if piv.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(ALG_LABELS.get(alg, alg))
            ax.axis("off")
            continue

        im = ax.imshow(
            piv.values,
            aspect="auto",
            origin="lower",
            vmin=global_vmin,
            vmax=global_vmax,
        )

        # Annotate each heatmap cell with the metric value.
        for r in range(piv.shape[0]):
            for c in range(piv.shape[1]):
                v = piv.iloc[r, c]
                if pd.isna(v):
                    continue

                if metric in ("completion_rate",):
                    label = f"{float(v):.2f}"
                elif metric in ("throughput",):
                    label = f"{float(v):.3f}"
                else:
                    label = f"{float(v):.1f}"

                rgba = im.cmap(im.norm(float(v)))
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text_color = "black" if luminance > 0.6 else "white"

                ax.text(c, r, label, ha="center", va="center", color=text_color, fontsize=7)

        ax.set_title(ALG_LABELS.get(alg, alg), fontsize=10)
        ax.set_xlabel("comm range", fontsize=9)
        ax.set_ylabel("arrival rate", fontsize=9)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:.4g}" for c in piv.columns], fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"{r:.4g}" for r in piv.index], fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    out = os.path.join(out_dir, f"compact_heatmap_{metric}.png")
    save_or_show(fig, out, save)


def plot_capacity_curve(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int, throughput_threshold: float):
    """Plot max arrival rate achieving stable throughput at each comm_range."""
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "task_arrival_rate"],
        "throughput",
        min_seeds,
    )
    if agg.empty:
        print("No data for capacity curve.")
        return

    rows = []
    for (alg, cr), sub in agg.groupby(["allocation_method", "comm_range"]):
        max_throughput = sub["throughput"].max()
        threshold_val = max_throughput * throughput_threshold
        ok = sub[sub["throughput"] >= threshold_val]
        max_ar = ok["task_arrival_rate"].max() if not ok.empty else 0
        rows.append({"allocation_method": alg, "comm_range": cr, "max_arrival_rate": max_ar})

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
            sub["max_arrival_rate"],
            marker="o",
            label=ALG_LABELS.get(alg, alg),
            color=ALG_COLORS.get(alg),
            linewidth=2.8 if is_lcba else 2.2,
            markersize=10 if is_lcba else 8,
            markeredgecolor="black" if is_lcba else None,
            markeredgewidth=1.3 if is_lcba else 0.0,
            zorder=5 if is_lcba else 3,
        )

    ax.set_title(f"Max Arrival Rate (Stable Throughput) vs Comm Range - {map_name}", fontsize=12)
    ax.set_xlabel("comm range")
    ax.set_ylabel(f"max arrival rate (throughput >= {throughput_threshold:.0%} peak)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    out = os.path.join(out_dir, "compact_capacity_curve.png")
    save_or_show(fig, out, save)


def plot_tradeoff(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    """Plot throughput (X) vs avg wait time (Y)."""
    tput = mean_with_min_seeds(
        df,
        ["allocation_method", "task_arrival_rate", "comm_range"],
        "throughput",
        min_seeds,
    )
    wait = mean_with_min_seeds(
        df,
        ["allocation_method", "task_arrival_rate", "comm_range"],
        "wait_time",
        min_seeds,
    )
    merged = tput.merge(
        wait,
        on=["allocation_method", "task_arrival_rate", "comm_range"],
        suffixes=("_throughput", "_wait"),
    )
    if merged.empty:
        print("No data for tradeoff plot.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in [a for a in ALG_ORDER if a in merged["allocation_method"].unique()]:
        sub = merged[merged["allocation_method"] == alg]
        ax.scatter(
            sub["throughput"],
            sub["wait_time"],
            s=45,
            alpha=0.8,
            label=ALG_LABELS.get(alg, alg),
            color=ALG_COLORS.get(alg),
        )

    ax.set_title(f"Throughput vs Task Wait Time - {map_name}", fontsize=12)
    ax.set_xlabel("throughput (tasks/step)")
    ax.set_ylabel("avg task wait time (steps)")
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


def plot_throughput_degradation(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "task_arrival_rate"],
        "throughput",
        min_seeds,
    )
    if agg.empty:
        print("No data for throughput degradation curves.")
        return

    comm_ranges = sorted(agg["comm_range"].unique())
    fig, axes, nrows, ncols = _subplot_grid(len(comm_ranges), ncols=3)
    fig.suptitle(f"Throughput vs Arrival Rate by Comm Range - {map_name}", fontsize=13)

    present_algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    draw_order = [a for a in present_algs if a != "gcbba"] + (["gcbba"] if "gcbba" in present_algs else [])

    for idx, cr in enumerate(comm_ranges):
        ax = axes[idx // ncols][idx % ncols]
        sub_cr = agg[agg["comm_range"] == cr]

        for alg in draw_order:
            sub = sub_cr[sub_cr["allocation_method"] == alg].sort_values("task_arrival_rate")
            if sub.empty:
                continue
            ax.plot(
                sub["task_arrival_rate"],
                sub["throughput"],
                marker="o",
                label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg),
                **_line_style_for_alg(alg),
            )

        ax.set_title(f"comm range = {cr}", fontsize=10)
        ax.set_xlabel("arrival rate")
        ax.set_ylabel("throughput (tasks/step)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(comm_ranges), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    out = os.path.join(out_dir, "compact_throughput_degradation_by_comm_range.png")
    save_or_show(fig, out, save)


def plot_wait_time_degradation(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    agg = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "task_arrival_rate"],
        "wait_time",
        min_seeds,
    )
    if agg.empty:
        print("No data for wait time degradation curves.")
        return

    comm_ranges = sorted(df["comm_range"].unique())
    fig, axes, nrows, ncols = _subplot_grid(len(comm_ranges), ncols=3)
    fig.suptitle(f"Wait Time vs Arrival Rate by Comm Range - {map_name}", fontsize=13)

    present_algs = [a for a in ALG_ORDER if a in agg["allocation_method"].unique()]
    draw_order = [a for a in present_algs if a != "gcbba"] + (["gcbba"] if "gcbba" in present_algs else [])

    for idx, cr in enumerate(comm_ranges):
        ax = axes[idx // ncols][idx % ncols]
        sub_cr = agg[agg["comm_range"] == cr]

        if sub_cr.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"comm range = {cr}", fontsize=10)
            ax.set_xlabel("arrival rate")
            ax.set_ylabel("wait time (steps)")
            ax.grid(axis="y", alpha=0.3)
            continue

        for alg in draw_order:
            sub = sub_cr[sub_cr["allocation_method"] == alg].sort_values("task_arrival_rate")
            if sub.empty:
                continue
            ax.plot(
                sub["task_arrival_rate"],
                sub["wait_time"],
                marker="o",
                label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg),
                **_line_style_for_alg(alg),
            )

        ax.set_title(f"comm range = {cr}", fontsize=10)
        ax.set_xlabel("arrival rate")
        ax.set_ylabel("wait time (steps)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    for idx in range(len(comm_ranges), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    out = os.path.join(out_dir, "compact_wait_time_degradation_by_comm_range.png")
    save_or_show(fig, out, save)


def plot_completion_progress(df: pd.DataFrame, out_dir: str, map_name: str, save: bool, min_seeds: int):
    """Show how much work gets done before a run stops, split by stop reason."""
    if "stop_reason" not in df.columns:
        print("No data for completion-progress plot.")
        return

    tasks = mean_with_min_seeds(
        df,
        ["allocation_method", "comm_range", "stop_reason"],
        "steady_state_tasks_completed",
        min_seeds,
    )
    if tasks.empty:
        print("No data for completion-progress plot.")
        return

    methods = [m for m in ALG_ORDER if m in tasks["allocation_method"].unique()]
    if not methods:
        print("No allocation methods available for completion-progress plot.")
        return

    stop_order = [s for s in ["all_tasks_completed", "timestep_ceiling", "wall_clock_ceiling", "saturation_ceiling"] if s in tasks["stop_reason"].unique()]
    if not stop_order:
        stop_order = sorted(tasks["stop_reason"].unique())

    for stop_reason in stop_order:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sub_stop = tasks[tasks["stop_reason"] == stop_reason]
        if sub_stop.empty:
            continue

        for alg in methods:
            sub_alg = sub_stop[sub_stop["allocation_method"] == alg].sort_values("comm_range")
            if sub_alg.empty:
                continue
            ax.plot(
                sub_alg["comm_range"],
                sub_alg["steady_state_tasks_completed"],
                marker="o",
                linestyle="-",
                linewidth=2,
                label=ALG_LABELS.get(alg, alg),
                color=ALG_COLORS.get(alg),
            )

        ax.set_title(f"Completed tasks before stop: {stop_reason.replace('_', ' ')} - {map_name}")
        ax.set_xlabel("comm range")
        ax.set_ylabel("steady-state tasks completed")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)

        out = os.path.join(out_dir, f"compact_completion_progress_{stop_reason}.png")
        save_or_show(fig, out, save)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to summary.csv")
    parser.add_argument("--save", action="store_true", help="Save PNGs instead of showing")
    parser.add_argument("--output-dir", default=None, help="Output folder (default: results/experiments/<map>/<timestamp>/ss_results/compact_views)")
    parser.add_argument("--min-seeds", type=int, default=MIN_SEEDS_DEFAULT)
    parser.add_argument("--include-wall-clock-timeouts", action="store_true")
    parser.add_argument("--throughput-threshold", type=float, default=0.9, help="Fraction of peak throughput to define stability (capacity curve)")
    parser.add_argument("--allocation-method", default=None, choices=["gcbba", "cbba", "dmchba", "sga"], help="Optional filter to plot a single allocation method")

    parser.add_argument("--all", action="store_true", help="Generate all compact views")
    parser.add_argument("--heatmaps", action="store_true")
    parser.add_argument("--capacity", action="store_true")
    parser.add_argument("--tradeoff", action="store_true")
    parser.add_argument("--degradation", action="store_true", help="Throughput and wait-time degradation curves by comm range")
    parser.add_argument("--completion-progress", action="store_true", help="Completed-work and run-length proxy by stop reason")

    args = parser.parse_args()

    df = load_ss(args.csv, include_wall_clock_timeouts=args.include_wall_clock_timeouts)
    df = filter_method(df, args.allocation_method)
    if df.empty:
        print("No steady-state rows found after filtering.")
        return

    map_name = infer_map_name(args.csv)
    ts = infer_run_timestamp(args.csv)
    out_dir = args.output_dir or default_output_root(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    write_run_manifest(
        out_dir,
        "python " + " ".join(__import__("sys").argv),
        {
            "csv": args.csv,
            "map_name": map_name,
            "timestamp": ts,
            "min_seeds": args.min_seeds,
            "include_wall_clock_timeouts": args.include_wall_clock_timeouts,
            "throughput_threshold": args.throughput_threshold,
            "allocation_method": args.allocation_method,
            "mode": "compact",
            "save": args.save,
        },
    )

    do_all = args.all or not (
        args.heatmaps or args.capacity or args.tradeoff or args.degradation
    )

    if do_all or args.heatmaps:
        for metric, label in HEATMAP_METRICS:
            plot_heatmaps(df, metric, label, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.capacity:
        plot_capacity_curve(df, out_dir, map_name, args.save, args.min_seeds, args.throughput_threshold)

    if do_all or args.tradeoff:
        plot_tradeoff(df, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.degradation:
        plot_throughput_degradation(df, out_dir, map_name, args.save, args.min_seeds)
        plot_wait_time_degradation(df, out_dir, map_name, args.save, args.min_seeds)

    if do_all or args.completion_progress:
        plot_completion_progress(df, out_dir, map_name, args.save, args.min_seeds)


if __name__ == "__main__":
    main()
