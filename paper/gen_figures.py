"""gen_figures.py — Generate all four publication-quality PDF figures for the paper.

Reads Phase 6 result CSVs and writes PDF figures to paper/figures/.
Run from the project root: python paper/gen_figures.py

Figures produced:
  paper/figures/training_az.pdf    — AlphaZero win rate vs iteration
  paper/figures/training_ppo.pdf   — PPO win rate vs iteration (pool-5)
  paper/figures/ablation_mcts.pdf  — AlphaZero win rate vs MCTS sim count
  paper/figures/ablation_pool.pdf  — PPO win rate vs pool size

Requirements: matplotlib, numpy (standard in PyTorch environment).
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Publication style settings (AAAI single column: 3.3 in wide)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})
COLUMN_WIDTH = 3.3   # AAAI single-column inches
FIG_HEIGHT = 2.4

# ---------------------------------------------------------------------------
# Paths — relative to project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"

# ---------------------------------------------------------------------------
# Guard: results/ must exist (this script is meant to run on GCP or after
# scp-ing the results/ directory locally).
# ---------------------------------------------------------------------------
if not RESULTS_DIR.exists():
    print(
        "ERROR: results/ not found. Run this script on GCP after Phase 6 "
        "experiment scripts complete, then scp the paper/figures/ directory "
        "locally."
    )
    sys.exit(1)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: read training-curve CSV (columns: iteration, agent_name, opponent,
#         games, wins, draws, losses, win_rate, ci_low, ci_high, ...)
# ---------------------------------------------------------------------------
def _read_training_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Return rows grouped by opponent name, sorted by iteration."""
    groups: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            groups[row["opponent"]].append(row)
    for opponent in groups:
        groups[opponent].sort(key=lambda r: int(r["iteration"]))
    return groups


# ---------------------------------------------------------------------------
# Figure 1: AlphaZero training curves
# ---------------------------------------------------------------------------
def make_training_az() -> Path:
    csv_path = RESULTS_DIR / "alphazero_training" / "training_curves.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Skipping training_az.pdf.")
        return None

    groups = _read_training_csv(csv_path)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT))

    label_map = {"random": "vs Random", "minimax4": "vs Minimax-4"}
    colors = {"random": "tab:blue", "minimax4": "tab:orange"}

    for opponent, rows in sorted(groups.items()):
        iters = [int(r["iteration"]) for r in rows]
        wr = [float(r["win_rate"]) for r in rows]
        label = label_map.get(opponent, opponent)
        color = colors.get(opponent, None)
        ax.plot(iters, wr, marker="o", markersize=2, label=label, color=color)
        if all(r.get("ci_low") and r.get("ci_high") for r in rows):
            ci_low = [float(r["ci_low"]) for r in rows]
            ci_high = [float(r["ci_high"]) for r in rows]
            ax.fill_between(iters, ci_low, ci_high, alpha=0.15, color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("Self-Play Iteration")
    ax.set_ylabel("Win Rate")
    ax.set_title("AlphaZero Training Progress")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "training_az.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 2: PPO training curves (use pool-5 as representative run)
# ---------------------------------------------------------------------------
def make_training_ppo() -> Path:
    # Use pool-5 as the representative PPO training curve
    csv_path = RESULTS_DIR / "ppo_training" / "ppo_pool5" / "training_curves.csv"
    if not csv_path.exists():
        # Fallback to pool-1
        csv_path = RESULTS_DIR / "ppo_training" / "ppo_pool1" / "training_curves.csv"
    if not csv_path.exists():
        print(f"ERROR: ppo training_curves.csv not found. Skipping training_ppo.pdf.")
        return None

    groups = _read_training_csv(csv_path)

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT))

    label_map = {"random": "vs Random", "minimax4": "vs Minimax-4"}
    colors = {"random": "tab:blue", "minimax4": "tab:orange"}

    for opponent, rows in sorted(groups.items()):
        iters = [int(r["iteration"]) for r in rows]
        wr = [float(r["win_rate"]) for r in rows]
        label = label_map.get(opponent, opponent)
        color = colors.get(opponent, None)
        ax.plot(iters, wr, marker="o", markersize=2, label=label, color=color)
        if all(r.get("ci_low") and r.get("ci_high") for r in rows):
            ci_low = [float(r["ci_low"]) for r in rows]
            ci_high = [float(r["ci_high"]) for r in rows]
            ax.fill_between(iters, ci_low, ci_high, alpha=0.15, color=color)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("PPO Update Step")
    ax.set_ylabel("Win Rate")
    ax.set_title("PPO Training Progress (Pool-5)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "training_ppo.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 3: AlphaZero MCTS sim-count ablation
# ---------------------------------------------------------------------------
def make_ablation_mcts() -> Path:
    csv_path = RESULTS_DIR / "ablation_mcts" / "ablation_results.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Skipping ablation_mcts.pdf.")
        return None

    rows = []
    with csv_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print(f"ERROR: No data in {csv_path}. Skipping ablation_mcts.pdf.")
        return None

    x_vals = [str(r["num_sims"]) for r in rows]
    x_pos = list(range(len(x_vals)))
    width = 0.35

    random_wr = [float(r["vs_random_wr"]) for r in rows]
    random_err = [
        [max(0.0, float(r["vs_random_wr"]) - float(r["vs_random_ci_low"])) for r in rows],
        [max(0.0, float(r["vs_random_ci_high"]) - float(r["vs_random_wr"])) for r in rows],
    ]
    mm4_wr = [float(r["vs_minimax4_wr"]) for r in rows]
    mm4_err = [
        [max(0.0, float(r["vs_minimax4_wr"]) - float(r["vs_minimax4_ci_low"])) for r in rows],
        [max(0.0, float(r["vs_minimax4_ci_high"]) - float(r["vs_minimax4_wr"])) for r in rows],
    ]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT))
    ax.bar(
        [p - width / 2 for p in x_pos], random_wr, width,
        label="vs Random", yerr=random_err, capsize=3,
        color="tab:blue", error_kw={"linewidth": 0.8},
    )
    ax.bar(
        [p + width / 2 for p in x_pos], mm4_wr, width,
        label="vs Minimax-4", yerr=mm4_err, capsize=3,
        color="tab:orange", error_kw={"linewidth": 0.8},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals)
    ax.set_xlabel("MCTS Simulations per Move")
    ax.set_ylabel("Win Rate")
    ax.set_title("AlphaZero: MCTS Sim Count Ablation")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_mcts.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Figure 4: PPO opponent pool size ablation
# ---------------------------------------------------------------------------
def make_ablation_pool() -> Path:
    # Aggregate final-checkpoint win rates across pool sizes
    pool_dirs = [
        ("1", RESULTS_DIR / "ppo_training" / "ppo_pool1" / "training_curves.csv"),
        ("5", RESULTS_DIR / "ppo_training" / "ppo_pool5" / "training_curves.csv"),
        ("20", RESULTS_DIR / "ppo_training" / "ppo_pool20" / "training_curves.csv"),
    ]

    rows_out = []
    for pool_size, csv_path in pool_dirs:
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found; skipping pool size {pool_size}.")
            continue
        groups = _read_training_csv(csv_path)
        entry = {"pool_size": pool_size}
        for opponent in ("random", "minimax4"):
            opp_rows = groups.get(opponent, [])
            if opp_rows:
                last = opp_rows[-1]
                entry[f"vs_{opponent}_wr"] = float(last["win_rate"])
                entry[f"vs_{opponent}_ci_low"] = float(last["ci_low"])
                entry[f"vs_{opponent}_ci_high"] = float(last["ci_high"])
            else:
                entry[f"vs_{opponent}_wr"] = 0.0
                entry[f"vs_{opponent}_ci_low"] = 0.0
                entry[f"vs_{opponent}_ci_high"] = 0.0
        rows_out.append(entry)

    if not rows_out:
        print("ERROR: No PPO pool ablation data found. Skipping ablation_pool.pdf.")
        return None

    x_vals = [r["pool_size"] for r in rows_out]
    x_pos = list(range(len(x_vals)))
    width = 0.35

    random_wr = [r["vs_random_wr"] for r in rows_out]
    random_err = [
        [r["vs_random_wr"] - r["vs_random_ci_low"] for r in rows_out],
        [r["vs_random_ci_high"] - r["vs_random_wr"] for r in rows_out],
    ]
    mm4_wr = [r["vs_minimax4_wr"] for r in rows_out]
    mm4_err = [
        [r["vs_minimax4_wr"] - r["vs_minimax4_ci_low"] for r in rows_out],
        [r["vs_minimax4_ci_high"] - r["vs_minimax4_wr"] for r in rows_out],
    ]

    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, FIG_HEIGHT))
    ax.bar(
        [p - width / 2 for p in x_pos], random_wr, width,
        label="vs Random", yerr=random_err, capsize=3,
        color="tab:blue", error_kw={"linewidth": 0.8},
    )
    ax.bar(
        [p + width / 2 for p in x_pos], mm4_wr, width,
        label="vs Minimax-4", yerr=mm4_err, capsize=3,
        color="tab:orange", error_kw={"linewidth": 0.8},
    )
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_vals)
    ax.set_xlabel("Opponent Pool Size")
    ax.set_ylabel("Win Rate (final checkpoint)")
    ax.set_title("PPO: Opponent Pool Size Ablation")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = FIGURES_DIR / "ablation_pool.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=600)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating paper figures...")
    make_training_az()
    make_training_ppo()
    make_ablation_mcts()
    make_ablation_pool()
    print("Done. Check paper/figures/ for output PDFs.")
