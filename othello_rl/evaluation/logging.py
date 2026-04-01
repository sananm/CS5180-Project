from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

from othello_rl.evaluation.stats import build_win_rate_row


CSV_COLUMNS = [
    "iteration",
    "agent_name",
    "opponent",
    "games",
    "wins",
    "draws",
    "losses",
    "win_rate",
    "ci_low",
    "ci_high",
    "checkpoint_path",
    "notes",
]


@dataclass
class TrainingCurveRow:
    iteration: int
    agent_name: str
    opponent: str
    games: int
    wins: int
    draws: int
    losses: int
    win_rate: float
    ci_low: float
    ci_high: float
    checkpoint_path: str
    notes: str


class TrainingCurveLogger:
    def __init__(self, csv_path):
        self.csv_path = Path(csv_path)

    def log_row(self, row: TrainingCurveRow) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        should_write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0

        with self.csv_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
            if should_write_header:
                writer.writeheader()
            writer.writerow(asdict(row))

    def log_from_counts(
        self,
        iteration,
        agent_name,
        opponent,
        wins,
        draws,
        losses,
        checkpoint_path=None,
        notes="",
    ) -> TrainingCurveRow:
        win_rate_row = build_win_rate_row(agent_name, opponent, wins, draws, losses)
        row = TrainingCurveRow(
            iteration=int(iteration),
            agent_name=agent_name,
            opponent=opponent,
            games=int(win_rate_row["games"]),
            wins=int(wins),
            draws=int(draws),
            losses=int(losses),
            win_rate=float(win_rate_row["win_rate"]),
            ci_low=float(win_rate_row["ci_low"]),
            ci_high=float(win_rate_row["ci_high"]),
            checkpoint_path="" if checkpoint_path is None else str(checkpoint_path),
            notes=notes,
        )
        self.log_row(row)
        return row
