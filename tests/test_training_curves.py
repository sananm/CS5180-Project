import csv

from othello_rl.evaluation.logging import TrainingCurveLogger
from othello_rl.evaluation.plotting import plot_training_curves


def test_training_curve_logger_writes_header_once(tmp_path):
    csv_path = tmp_path / "curves" / "training.csv"
    logger = TrainingCurveLogger(csv_path)

    logger.log_from_counts(1, "ppo", "random", 8, 1, 1)
    logger.log_from_counts(2, "ppo", "random", 9, 0, 1)

    lines = csv_path.read_text().splitlines()
    assert lines[0] == "iteration,agent_name,opponent,games,wins,draws,losses,win_rate,ci_low,ci_high,checkpoint_path,notes"
    assert sum(line.startswith("iteration,agent_name") for line in lines) == 1


def test_log_from_counts_computes_win_rate_and_ci_fields(tmp_path):
    csv_path = tmp_path / "curves.csv"
    logger = TrainingCurveLogger(csv_path)

    row = logger.log_from_counts(3, "ppo", "minimax4", 6, 1, 1, checkpoint_path="ckpt.pt", notes="smoke")

    assert row.games == 8
    assert row.win_rate == 6.5 / 8.0
    assert 0.0 <= row.ci_low <= row.ci_high <= 1.0

    with csv_path.open(newline="") as handle:
        stored = list(csv.DictReader(handle))

    assert stored[0]["checkpoint_path"] == "ckpt.pt"
    assert stored[0]["notes"] == "smoke"


def test_plot_training_curves_writes_non_empty_file(tmp_path):
    csv_path = tmp_path / "curves.csv"
    png_path = tmp_path / "plots" / "curves.png"
    logger = TrainingCurveLogger(csv_path)

    logger.log_from_counts(1, "ppo", "random", 8, 1, 1)
    logger.log_from_counts(2, "ppo", "random", 9, 0, 1)

    plot_training_curves(csv_path, png_path)

    assert png_path.exists()
    assert png_path.stat().st_size > 0
