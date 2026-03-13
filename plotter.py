"""Post-simulation plotting — generates publication-quality figures.

Reads the CSV logs produced by DataLogger and creates multi-panel
plots of consciousness metrics, energy, emotions, and network stats
over time.

Usage:
    python -m genesis.analytics.plotter logs/
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


def _load_csv(path: Path) -> dict[str, np.ndarray]:
    """Load a CSV file into a dict of column arrays."""
    import csv
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    columns: dict[str, list] = {k: [] for k in rows[0]}
    for row in rows:
        for k, v in row.items():
            try:
                columns[k].append(float(v))
            except (ValueError, TypeError):
                columns[k].append(v)
    result = {}
    for k, vals in columns.items():
        if all(isinstance(v, float) for v in vals):
            result[k] = np.array(vals, dtype=np.float64)
        else:
            result[k] = np.array(vals)
    return result


def plot_agent(data: dict[str, np.ndarray], agent_id: int,
               output_dir: Path) -> None:
    """Generate a multi-panel figure for one agent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = data.get("tick")
    if ticks is None or len(ticks) == 0:
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Agent {agent_id} — Consciousness Emergence Timeline",
                 fontsize=14, fontweight="bold")

    # (0,0) Energy & Integrity
    ax = axes[0, 0]
    ax.plot(ticks, data["energy"], label="Energy", color="green")
    ax.plot(ticks, data["integrity"], label="Integrity", color="blue")
    ax.set_ylabel("Level")
    ax.set_title("Homeostasis")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (0,1) Phi & Complexity
    ax = axes[0, 1]
    ax.plot(ticks, data["phi"], label="Φ", color="purple")
    ax.plot(ticks, data["complexity"], label="Complexity", color="orange")
    ax.set_ylabel("Score")
    ax.set_title("Consciousness Metrics")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # (1,0) Composite Consciousness
    ax = axes[1, 0]
    ax.fill_between(ticks, data["composite_consciousness"],
                    alpha=0.4, color="crimson")
    ax.plot(ticks, data["composite_consciousness"], color="crimson")
    ax.set_ylabel("Score")
    ax.set_title("Composite Consciousness Score")
    ax.grid(alpha=0.3)

    # (1,1) Self-model accuracy
    ax = axes[1, 1]
    ax.plot(ticks, data["self_model_accuracy"], color="teal")
    if "has_ego" in data:
        ego_ticks = ticks[data["has_ego"] > 0.5]
        if len(ego_ticks) > 0:
            ax.axvline(ego_ticks[0], color="red", linestyle="--",
                       label=f"Ego emerges @ tick {int(ego_ticks[0])}")
            ax.legend(fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Self-Model Accuracy")
    ax.grid(alpha=0.3)

    # (2,0) Prediction error
    ax = axes[2, 0]
    ax.plot(ticks, data["prediction_error"], color="brown")
    ax.set_ylabel("Error")
    ax.set_xlabel("Tick")
    ax.set_title("Prediction Error")
    ax.grid(alpha=0.3)

    # (2,1) Pain & Pleasure
    ax = axes[2, 1]
    ax.plot(ticks, data["pain"], label="Pain", color="red")
    ax.plot(ticks, data["pleasure"], label="Pleasure", color="gold")
    ax.set_ylabel("Cumulative")
    ax.set_xlabel("Tick")
    ax.set_title("Pain / Pleasure")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / f"agent_{agent_id}.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_comparison(all_data: dict[int, dict[str, np.ndarray]],
                    output_dir: Path) -> None:
    """Generate an overlay comparison plot for all agents."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Agent Comparison", fontsize=14, fontweight="bold")

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red",
              "tab:purple", "tab:brown"]

    for aid, data in all_data.items():
        ticks = data.get("tick")
        if ticks is None:
            continue
        c = colors[aid % len(colors)]
        axes[0, 0].plot(ticks, data["phi"], label=f"Agent {aid}", color=c)
        axes[0, 1].plot(ticks, data["composite_consciousness"],
                        label=f"Agent {aid}", color=c)
        axes[1, 0].plot(ticks, data["energy"], label=f"Agent {aid}", color=c)
        axes[1, 1].plot(ticks, data["self_model_accuracy"],
                        label=f"Agent {aid}", color=c)

    axes[0, 0].set_title("Φ (Phi)")
    axes[0, 1].set_title("Composite Consciousness")
    axes[1, 0].set_title("Energy")
    axes[1, 1].set_title("Self-Model Accuracy")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1, 0].set_xlabel("Tick")
    axes[1, 1].set_xlabel("Tick")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = output_dir / "comparison.png"
    fig.savefig(str(out), dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def main(log_dir: str) -> None:
    """Entry point for the plotter."""
    log_path = Path(log_dir)
    if not log_path.exists():
        print(f"  Error: {log_dir} does not exist.")
        sys.exit(1)

    csvs = sorted(log_path.glob("agent_*.csv"))
    if not csvs:
        print(f"  No agent_*.csv files found in {log_dir}")
        sys.exit(1)

    print(f"  Plotting {len(csvs)} agent logs from {log_dir}/ ...")

    all_data: dict[int, dict[str, np.ndarray]] = {}
    for csv_file in csvs:
        aid = int(csv_file.stem.split("_")[1])
        data = _load_csv(csv_file)
        if data:
            all_data[aid] = data
            plot_agent(data, aid, log_path)

    if len(all_data) > 1:
        plot_comparison(all_data, log_path)

    print("  Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Project Genesis simulation logs")
    parser.add_argument("log_dir", help="Directory containing agent_*.csv files")
    args = parser.parse_args()
    main(args.log_dir)
