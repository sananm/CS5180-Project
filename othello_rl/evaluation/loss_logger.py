"""Per-iteration loss and entropy logging for training curves."""
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path


LOSS_CSV_COLUMNS = [
    "iteration", "algorithm", "policy_loss", "value_loss",
    "entropy", "total_loss", "wall_clock_seconds"
]


@dataclass
class LossRow:
    """A single row of loss metrics for one training iteration."""
    iteration: int
    algorithm: str
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float
    wall_clock_seconds: float


class LossLogger:
    """Logs per-iteration loss metrics to CSV.
    
    Example:
        logger = LossLogger("logs/loss.csv")
        logger.log(LossRow(1, "PPO", 0.5, 0.3, 1.2, 0.8, 10.5))
    """
    
    def __init__(self, csv_path):
        """Initialize the logger with a path to the CSV file.
        
        Args:
            csv_path: Path to the CSV file (created if not exists).
        """
        self.csv_path = Path(csv_path)
    
    def log(self, row: LossRow) -> None:
        """Append a row of loss metrics to the CSV file.
        
        Creates the file with header on first write.
        
        Args:
            row: LossRow dataclass with metrics for one iteration.
        """
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOSS_CSV_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(row))
