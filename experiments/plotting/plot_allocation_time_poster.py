"""
Generate a 3-panel allocation time vs comm_range figure for the poster.
One subplot per map (warehouse_small, crossdock, warehouse_large), shared y-axis.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────
ALG_ORDER  = ["gcbba", "cbba", "dmchba", "sga"]
ALG_LABELS = {"gcbba": "LCBA", "cbba": "CBBA", "dmchba": "DMCHBA", "sga": "SGA"}
ALG_COLORS = {"gcbba": "#1f77b4", "cbba": "#ff7f0e", "dmchba": "#2ca02c", "sga": "#d62728"}
ALG_STYLES = {
    "gcbba":  dict(lw=2.5, marker="o", ms=7, zorder=5),
    "cbba":   dict(lw=2.0, marker="s", ms=6, zorder=4),
    "dmchba": dict(lw=2.0, marker="^", ms=6, zorder=3),
    "sga":    dict(lw=1.5, marker="D", ms=5, zorder=2),
}

MAP_LABELS = {
    "gridworld_warehouse_small": "Warehouse Small\n(30\u00d730, 6 agents)",
    "gridworld_crossdock":       "Crossdock\n(44\u00d728, 12 agents)",
    "gridworld_warehouse_large": "Warehouse Large\n(60\u00d760, 18 agents)",
}

SS_EXPERIMENT_TYPE = "steady_state"


# ── helpers ────────────────────────────────────────────────────────────────
def infer_map(csv_path: str) -> str:
    for key in MAP_LABELS:
        if key in csv_path.replace("\\", "/"):
            return key
    raise ValueError(f"Cannot infer map name from path: {csv_path}")


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # keep steady-state rows that have a valid allocation time
    df = df[df["experiment_type"] == SS_EXPERIMENT_TYPE].copy()
    df = df[df["avg_allocation_time_ms"].notna()].copy()
    df["allocation_method"] = df["allocation_method"].str.lower().str.strip()
    return df


def mean_with_min_seeds(df: pd.DataFrame, min_seeds: int = 2) -> pd.DataFrame:
    """
    Group by (allocation_method, comm_range), keep groups with >= min_seeds rows,
    return mean avg_allocation_time_ms per group.
    """
    grp = (
        df.groupby(["allocation_method", "comm_range"])
        .agg(
            mean_alloc_time=("avg_allocation_time_ms", "mean"),
            n=("avg_allocation_time_ms", "count"),
        )
        .reset_index()
    )
    return grp[grp["n"] >= min_seeds]


# ── main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True, help="One summary.csv per map")
    parser.add_argument("--out", default="Latex/figures/poster_allocation_time_multimap.png")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--min-seeds", type=int, default=2)
    args = parser.parse_args()

    datasets = []
    for csv_path in args.csv:
        map_key = infer_map(csv_path)
        df = load_data(csv_path)
        agg = mean_with_min_seeds(df, args.min_seeds)
        datasets.append((map_key, agg))

    # sort by MAP_LABELS order
    order = list(MAP_LABELS.keys())
    datasets.sort(key=lambda x: order.index(x[0]) if x[0] in order else 99)

    n_maps = len(datasets)
    fig, axes = plt.subplots(
        1, n_maps,
        figsize=(5.5 * n_maps, 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if n_maps == 1:
        axes = [axes]

    for col, (map_key, agg) in enumerate(datasets):
        ax = axes[col]
        for alg in ALG_ORDER:
            sub = agg[agg["allocation_method"] == alg].sort_values("comm_range")
            if sub.empty:
                continue
            style = ALG_STYLES[alg]
            ax.plot(
                sub["comm_range"],
                sub["mean_alloc_time"],
                color=ALG_COLORS[alg],
                label=ALG_LABELS[alg],
                **style,
            )

        ax.set_yscale("log")
        ax.set_xlabel("Comm Range (grid units)", fontsize=11)
        if n_maps > 1:
            ax.set_title(MAP_LABELS.get(map_key, map_key), fontsize=11, pad=6)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=10)

        if col == 0:
            ax.set_ylabel("Mean Allocation Time (ms)", fontsize=11)

        # legend only on rightmost panel
        if col == n_maps - 1:
            ax.legend(fontsize=10, loc="upper right", framealpha=0.85)

    if n_maps > 1:
        fig.suptitle(
            "Allocation Time vs. Communication Range",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )

    if args.save:
        out_path = args.out
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
