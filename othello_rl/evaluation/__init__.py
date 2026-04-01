from __future__ import annotations

from importlib import import_module


__all__ = [
    "Arena",
    "GameRecord",
    "MatchResult",
    "build_win_rate_row",
    "compute_elo_ratings",
    "compute_score",
    "expected_score",
    "Tournament",
    "TournamentResult",
    "TournamentRow",
    "TrainingCurveLogger",
    "TrainingCurveRow",
    "plot_training_curves",
    "update_elo",
    "wilson_interval",
    # New Phase 6 exports
    "binomial_test",
    "SignificanceResult",
    "disaggregate_by_color",
    "ColorAnalysis",
    "ColorStats",
    "LossLogger",
    "LossRow",
    "LOSS_CSV_COLUMNS",
    # Compute timing exports
    "timed_call",
    "timed_az_iteration",
    "timed_ppo_iteration",
    "TimedResult",
]

_EXPORTS = {
    "Arena": "othello_rl.evaluation.arena",
    "GameRecord": "othello_rl.evaluation.arena",
    "MatchResult": "othello_rl.evaluation.arena",
    "Tournament": "othello_rl.evaluation.tournament",
    "TournamentResult": "othello_rl.evaluation.tournament",
    "TournamentRow": "othello_rl.evaluation.tournament",
    "wilson_interval": "othello_rl.evaluation.stats",
    "compute_score": "othello_rl.evaluation.stats",
    "build_win_rate_row": "othello_rl.evaluation.stats",
    "expected_score": "othello_rl.evaluation.elo",
    "update_elo": "othello_rl.evaluation.elo",
    "compute_elo_ratings": "othello_rl.evaluation.elo",
    "TrainingCurveLogger": "othello_rl.evaluation.logging",
    "TrainingCurveRow": "othello_rl.evaluation.logging",
    "plot_training_curves": "othello_rl.evaluation.plotting",
    # New Phase 6 modules
    "binomial_test": "othello_rl.evaluation.significance",
    "SignificanceResult": "othello_rl.evaluation.significance",
    "disaggregate_by_color": "othello_rl.evaluation.color_analysis",
    "ColorAnalysis": "othello_rl.evaluation.color_analysis",
    "ColorStats": "othello_rl.evaluation.color_analysis",
    "LossLogger": "othello_rl.evaluation.loss_logger",
    "LossRow": "othello_rl.evaluation.loss_logger",
    "LOSS_CSV_COLUMNS": "othello_rl.evaluation.loss_logger",
    # Compute timing exports
    "timed_call": "othello_rl.evaluation.compute_timing",
    "timed_az_iteration": "othello_rl.evaluation.compute_timing",
    "timed_ppo_iteration": "othello_rl.evaluation.compute_timing",
    "TimedResult": "othello_rl.evaluation.compute_timing",
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(_EXPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
