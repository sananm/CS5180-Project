from __future__ import annotations


def wilson_interval(successes: float, total: int, z: float = 1.96) -> tuple[float, float]:
    """Compute a Wilson score interval for fractional binomial successes."""
    if total <= 0:
        return (0.0, 1.0)

    total_f = float(total)
    phat = float(successes) / total_f
    z_sq = z * z
    denominator = 1.0 + (z_sq / total_f)
    center = (phat + (z_sq / (2.0 * total_f))) / denominator
    margin = (
        z
        * (((phat * (1.0 - phat)) + (z_sq / (4.0 * total_f))) / total_f) ** 0.5
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_score(wins: int, draws: int, losses: int) -> float:
    _ = losses
    return float(wins + 0.5 * draws)


def build_win_rate_row(
    agent_a: str,
    agent_b: str,
    wins_a: int,
    draws: int,
    losses_a: int,
) -> dict[str, float | int | str]:
    total = wins_a + draws + losses_a
    score = compute_score(wins_a, draws, losses_a)
    win_rate = score / total if total else 0.0
    ci_low, ci_high = wilson_interval(score, total)
    return {
        "agent_a": agent_a,
        "agent_b": agent_b,
        "wins": wins_a,
        "draws": draws,
        "losses": losses_a,
        "games": total,
        "score": score,
        "win_rate": win_rate,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
