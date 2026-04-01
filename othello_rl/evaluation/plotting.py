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


def plot_loss_and_entropy(csv_path, output_path) -> None:
    """Plot 2-panel figure: policy/value loss (left), entropy (right).
    
    Args:
        csv_path: Path to CSV with columns: iteration, algorithm, policy_loss, value_loss, entropy, ...
        output_path: Path to save the figure.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    
    # Read and group by algorithm
    data = defaultdict(list)
    with csv_path.open(newline='') as f:
        for row in csv.DictReader(f):
            data[row['algorithm']].append(row)
    
    if not data:
        raise ValueError(f"No data rows in {csv_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = {'AlphaZero': 'tab:blue', 'PPO': 'tab:orange'}
    
    for algo, rows in sorted(data.items()):
        rows = sorted(rows, key=lambda r: int(r['iteration']))
        iters = [int(r['iteration']) for r in rows]
        pol_loss = [float(r['policy_loss']) for r in rows]
        val_loss = [float(r['value_loss']) for r in rows]
        entropy = [float(r['entropy']) for r in rows]
        
        color = colors.get(algo, None)
        
        # Loss panel
        ax1.plot(iters, pol_loss, label=f'{algo} policy', color=color, linestyle='-')
        ax1.plot(iters, val_loss, label=f'{algo} value', color=color, linestyle='--', alpha=0.7)
        
        # Entropy panel
        ax2.plot(iters, entropy, label=algo, color=color)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curves')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Policy Entropy (nats)')
    ax2.set_title('Policy Entropy Over Training')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_ablation_results(csv_path, output_path, title="Ablation Study", x_label="Parameter", x_key="num_sims") -> None:
    """Plot ablation results as grouped bar chart with error bars.

    Args:
        csv_path: Path to CSV with columns: {x_key}, vs_random_wr, vs_random_ci_low,
                  vs_random_ci_high, vs_minimax4_wr, vs_minimax4_ci_low, vs_minimax4_ci_high
        output_path: Path to save the figure.
        title: Plot title.
        x_label: Label for x-axis.
        x_key: CSV column to use for x-axis values.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)

    rows = []
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {csv_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_vals = [row[x_key] for row in rows]
    x_pos = range(len(x_vals))
    width = 0.35

    random_wr = [float(row['vs_random_wr']) for row in rows]
    random_err_low = [float(row['vs_random_wr']) - float(row['vs_random_ci_low']) for row in rows]
    random_err_high = [float(row['vs_random_ci_high']) - float(row['vs_random_wr']) for row in rows]

    mm4_wr = [float(row['vs_minimax4_wr']) for row in rows]
    mm4_err_low = [float(row['vs_minimax4_wr']) - float(row['vs_minimax4_ci_low']) for row in rows]
    mm4_err_high = [float(row['vs_minimax4_ci_high']) - float(row['vs_minimax4_wr']) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar([p - width/2 for p in x_pos], random_wr, width, label='vs Random',
           yerr=[random_err_low, random_err_high], capsize=5, color='steelblue')
    ax.bar([p + width/2 for p in x_pos], mm4_wr, width, label='vs Minimax-4',
           yerr=[mm4_err_low, mm4_err_high], capsize=5, color='darkorange')

    ax.set_xlabel(x_label)
    ax.set_ylabel('Win Rate')
    ax.set_title(title)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_vals)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_color_asymmetry(csv_path, output_path) -> None:
    """Plot color asymmetry as grouped bar chart.
    
    Args:
        csv_path: Path to CSV with columns: agent, color, games, wins, draws, losses, win_rate, ci_low, ci_high
        output_path: Path to save the figure.
    """
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    
    rows = []
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        raise ValueError(f"No data rows in {csv_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Group by agent
    agents = []
    black_wr = []
    black_err = []
    white_wr = []
    white_err = []
    
    i = 0
    while i < len(rows):
        # Expect pairs: black then white for each agent
        if rows[i]['color'] == 'black':
            agents.append(rows[i]['agent'])
            black_wr.append(float(rows[i]['win_rate']))
            black_err.append([
                float(rows[i]['win_rate']) - float(rows[i]['ci_low']),
                float(rows[i]['ci_high']) - float(rows[i]['win_rate'])
            ])
        if i + 1 < len(rows) and rows[i + 1]['color'] == 'white':
            white_wr.append(float(rows[i + 1]['win_rate']))
            white_err.append([
                float(rows[i + 1]['win_rate']) - float(rows[i + 1]['ci_low']),
                float(rows[i + 1]['ci_high']) - float(rows[i + 1]['win_rate'])
            ])
            i += 2
        else:
            i += 1
    
    x_pos = range(len(agents))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.bar([p - width/2 for p in x_pos], black_wr, width, label='As Black',
           yerr=list(zip(*black_err)), capsize=5, color='#333333')
    ax.bar([p + width/2 for p in x_pos], white_wr, width, label='As White',
           yerr=list(zip(*white_err)), capsize=5, color='#cccccc', edgecolor='black')
    
    ax.set_xlabel('Agent')
    ax.set_ylabel('Win Rate')
    ax.set_title('Win Rate by Color')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(agents)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
