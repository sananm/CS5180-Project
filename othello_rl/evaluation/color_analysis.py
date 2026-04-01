"""Color asymmetry analysis for game records."""
from __future__ import annotations

from dataclasses import dataclass
from othello_rl.evaluation.arena import GameRecord
from othello_rl.evaluation.stats import wilson_interval


@dataclass
class ColorStats:
    """Win statistics when playing as a specific color."""
    games: int
    wins: int
    draws: int
    losses: int
    score: float  # wins + 0.5 * draws
    win_rate: float
    ci_low: float
    ci_high: float


@dataclass
class ColorAnalysis:
    """Color-disaggregated analysis for one agent."""
    agent_name: str
    as_black: ColorStats
    as_white: ColorStats


def disaggregate_by_color(records: list[GameRecord], agent_name: str) -> ColorAnalysis:
    """Compute win rates when agent plays as black vs white.
    
    Args:
        records: List of GameRecord objects from a match or tournament.
        agent_name: Name of the agent to analyze.
    
    Returns:
        ColorAnalysis with as_black and as_white statistics.
    """
    as_black = [r for r in records if r.black_agent == agent_name]
    as_white = [r for r in records if r.white_agent == agent_name]

    def compute_stats(games: list[GameRecord], is_black: bool) -> ColorStats:
        n = len(games)
        if n == 0:
            return ColorStats(0, 0, 0, 0, 0.0, 0.0, 0.0, 1.0)
        wins = sum(1 for g in games if g.winner == agent_name)
        draws = sum(1 for g in games if g.is_draw)
        losses = n - wins - draws
        score = wins + 0.5 * draws
        wr = score / n
        ci = wilson_interval(score, n)
        return ColorStats(n, wins, draws, losses, score, wr, ci[0], ci[1])

    return ColorAnalysis(
        agent_name=agent_name,
        as_black=compute_stats(as_black, True),
        as_white=compute_stats(as_white, False),
    )
