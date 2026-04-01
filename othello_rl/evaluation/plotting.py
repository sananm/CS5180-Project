from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt


def plot_training_curves(csv_path, output_path, metric="win_rate") -> None:
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise KeyError(f"Metric {metric!r} not present in {csv_path}")
        for row in reader:
            groups[row["opponent"]].append(row)

    if not groups:
        raise ValueError(f"No training-curve rows found in {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for opponent, rows in sorted(groups.items()):
        parsed_rows = sorted(rows, key=lambda row: int(row["iteration"]))
        iterations = [int(row["iteration"]) for row in parsed_rows]
        values = [float(row[metric]) for row in parsed_rows]
        ax.plot(iterations, values, marker="o", label=opponent)

        if metric == "win_rate" and all(row.get("ci_low") and row.get("ci_high") for row in parsed_rows):
            ci_low = [float(row["ci_low"]) for row in parsed_rows]
            ci_high = [float(row["ci_high"]) for row in parsed_rows]
            ax.fill_between(iterations, ci_low, ci_high, alpha=0.2)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title("Training Curves")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
