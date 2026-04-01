"""Statistical significance testing for head-to-head matchups."""
from __future__ import annotations

from dataclasses import dataclass
from scipy.stats import binomtest


@dataclass
class SignificanceResult:
    """Result of a binomial significance test."""
    wins_a: int
    wins_b: int
    draws: int
    decisive_games: int
    p_value: float
    ci_low: float
    ci_high: float
    significant_at_05: bool


def binomial_test(wins_a: int, wins_b: int, draws: int = 0) -> SignificanceResult:
    """Test if agent A is significantly different from 50% win rate.
    
    Uses exact binomial test on decisive games (excluding draws).
    
    Args:
        wins_a: Number of wins for agent A.
        wins_b: Number of wins for agent B.
        draws: Number of draws (excluded from significance calculation).
    
    Returns:
        SignificanceResult with p-value, confidence interval, and significance flag.
    """
    decisive = wins_a + wins_b
    if decisive == 0:
        return SignificanceResult(
            wins_a=wins_a, wins_b=wins_b, draws=draws,
            decisive_games=0, p_value=1.0, ci_low=0.0, ci_high=1.0,
            significant_at_05=False
        )
    result = binomtest(wins_a, decisive, p=0.5, alternative='two-sided')
    ci = result.proportion_ci(confidence_level=0.95, method='wilson')
    return SignificanceResult(
        wins_a=wins_a, wins_b=wins_b, draws=draws,
        decisive_games=decisive,
        p_value=result.pvalue,
        ci_low=ci.low,
        ci_high=ci.high,
        significant_at_05=result.pvalue < 0.05
    )
