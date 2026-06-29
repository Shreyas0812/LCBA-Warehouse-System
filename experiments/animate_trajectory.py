#!/usr/bin/env python3
"""Render a warehouse trajectory CSV into an animated MP4/GIF.

Pure matplotlib — no ROS required. Draws the static warehouse backdrop
(obstacles, induct / eject / charging stations) from the gridworld YAML config
and animates the agents moving along their recorded trajectories.

Usage
-----
    python experiments/animate_trajectory.py \
        --config config/gridworld_warehouse_small.yaml \
        --traj results/demo_warehouse_small/trajectories.csv \
        --out results/demo_warehouse_small/lcba_demo \
        --fps 15 --format both

Outputs `<out>.mp4` and/or `<out>.gif`. Colors mirror the RViz config so the
animation matches the ROS visualizer aesthetic.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Rectangle

# Colors (RGBA 0-1), taken from the gridworld_rviz visualization config so the
# matplotlib render matches the RViz look.
C_BG = "#0f1115"
C_GRID = "#1d2230"
C_OBSTACLE = "#c0392b"   # red walls/columns
C_INDUCT = "#3aa0ff"     # light blue docking bays
C_EJECT = "#ff8c1a"      # orange storage bins
C_CHARGER = "#19e68a"    # green-teal chargers
C_AGENT = "#ff3df2"      # magenta robots
C_TEXT = "#e6e8ec"


def _params(config_path):
    with open(config_path) as f:
        data = yaml.safe_load(f)
    # YAML is nested under <node_name>.ros__parameters
    node = next(iter(data.values()))
    return node["ros__parameters"]


def _chunks(flat, n):
    return [flat[i : i + n] for i in range(0, len(flat), n)]


def load_world(config_path):
    p = _params(config_path)
    world = {
        "w": int(p["grid_width"]),
        "h": int(p["grid_height"]),
        "obstacles": _chunks(p.get("obstacle_regions", []), 6),
        "induct": [(c[0], c[1]) for c in _chunks(p.get("induct_stations", []), 4)],
        "eject": [(c[0], c[1]) for c in _chunks(p.get("eject_stations", []), 4)],
        "charger": [(c[0], c[1]) for c in _chunks(p.get("charging_stations", []), 4)],
    }
    return world


def load_trajectories(traj_path):
    # timestep -> {agent_id: (x, y)}
    frames = defaultdict(dict)
    agents = set()
    with open(traj_path) as f:
        for row in csv.DictReader(f):
            aid = int(row["agent_id"])
            t = int(row["timestep"])
            frames[t][aid] = (int(row["x"]), int(row["y"]))
            agents.add(aid)
    max_t = max(frames) if frames else 0
    # Forward-fill so every agent has a position at every timestep.
    last = {}
    filled = []
    for t in range(max_t + 1):
        for aid, pos in frames.get(t, {}).items():
            last[aid] = pos
        filled.append({aid: last[aid] for aid in last})
    return filled, sorted(agents)


def draw_static(ax, world):
    ax.set_facecolor(C_BG)
    w, h = world["w"], world["h"]
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(-0.5, h - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(C_GRID)

    # faint grid
    for x in range(w + 1):
        ax.axvline(x - 0.5, color=C_GRID, lw=0.3, zorder=0)
    for y in range(h + 1):
        ax.axhline(y - 0.5, color=C_GRID, lw=0.3, zorder=0)

    # obstacles (rectangular regions: sx,sy,sz,ex,ey,ez)
    for sx, sy, _sz, ex, ey, _ez in world["obstacles"]:
        ax.add_patch(Rectangle((sx - 0.5, sy - 0.5), (ex - sx) + 1, (ey - sy) + 1,
                               color=C_OBSTACLE, alpha=0.85, zorder=1, lw=0))

    def cells(pts, color, size=0.8):
        for (x, y) in pts:
            off = (1 - size) / 2
            ax.add_patch(Rectangle((x - 0.5 + off, y - 0.5 + off), size, size,
                                   color=color, zorder=2, lw=0))

    cells(world["induct"], C_INDUCT, 0.9)
    cells(world["eject"], C_EJECT, 0.7)
    cells(world["charger"], C_CHARGER, 0.8)


def make_legend(ax):
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_INDUCT, markersize=9, label="Induct"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_EJECT, markersize=9, label="Eject"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_CHARGER, markersize=9, label="Charger"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_AGENT, markersize=9, label="Robot"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_OBSTACLE, markersize=9, label="Obstacle"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                    ncol=5, frameon=False, fontsize=9, labelcolor=C_TEXT,
                    handletextpad=0.3, columnspacing=1.2)
    return leg


def animate(world, frames, agent_ids, out, fps, fmt, title, trail, step, start, end):
    end = len(frames) if end is None else min(end + 1, len(frames))
    sel = list(range(start, end, max(1, step)))
    n = len(sel)

    fig, ax = plt.subplots(figsize=(7.2, 7.6))
    fig.patch.set_facecolor(C_BG)
    draw_static(ax, world)
    make_legend(ax)

    title_txt = fig.suptitle("", color=C_TEXT, fontsize=13, fontweight="bold", y=0.97)
    sub_txt = ax.set_title("", color=C_TEXT, fontsize=10, pad=8)

    scat = ax.scatter([], [], s=110, c=C_AGENT, edgecolors="white",
                      linewidths=0.8, zorder=5)
    trails = {aid: ax.plot([], [], color=C_AGENT, alpha=0.35, lw=1.4, zorder=3)[0]
              for aid in agent_ids}
    labels = {aid: ax.text(0, 0, str(aid), color="white", fontsize=7,
                           ha="center", va="center", zorder=6) for aid in agent_ids}
    hist = defaultdict(list)

    def update(i):
        t = sel[i]
        frame = frames[t]
        xs, ys = [], []
        for aid in agent_ids:
            if aid in frame:
                x, y = frame[aid]
                xs.append(x); ys.append(y)
                hist[aid].append((x, y))
                if len(hist[aid]) > trail:
                    hist[aid] = hist[aid][-trail:]
                hx, hy = zip(*hist[aid])
                trails[aid].set_data(hx, hy)
                labels[aid].set_position((x, y))
        scat.set_offsets(np.c_[xs, ys])
        title_txt.set_text(title)
        sub_txt.set_text(f"LCBA  ·  {len(agent_ids)} robots  ·  timestep {t}")
        return [scat, *trails.values(), *labels.values(), title_txt, sub_txt]

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=False)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    if fmt in ("mp4", "both"):
        path = out + ".mp4"
        anim.save(path, writer=FFMpegWriter(fps=fps, bitrate=2400),
                  savefig_kwargs={"facecolor": C_BG})
        print(f"wrote {path}  ({n} frames @ {fps}fps)")
    if fmt in ("gif", "both"):
        path = out + ".gif"
        anim.save(path, writer=PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": C_BG})
        print(f"wrote {path}  ({n} frames @ {fps}fps)")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="gridworld YAML config")
    ap.add_argument("--traj", required=True, help="trajectories.csv")
    ap.add_argument("--out", required=True, help="output path stem (no extension)")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--format", choices=["mp4", "gif", "both"], default="both")
    ap.add_argument("--title", default="LCBA Warehouse — Multi-Robot Task Allocation")
    ap.add_argument("--trail", type=int, default=12, help="trail length (timesteps)")
    ap.add_argument("--step", type=int, default=1, help="use every Nth timestep")
    ap.add_argument("--start", type=int, default=0, help="first timestep to render")
    ap.add_argument("--end", type=int, default=None, help="last timestep to render")
    args = ap.parse_args()

    world = load_world(args.config)
    frames, agent_ids = load_trajectories(args.traj)
    print(f"world {world['w']}x{world['h']} · {len(agent_ids)} agents · {len(frames)} timesteps")
    animate(world, frames, agent_ids, args.out, args.fps, args.format,
            args.title, args.trail, args.step, args.start, args.end)


if __name__ == "__main__":
    main()
