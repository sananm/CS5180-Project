import pytest

from othello_rl.evaluation.arena import GameRecord
from othello_rl.evaluation.elo import compute_elo_ratings, update_elo
from othello_rl.evaluation.stats import build_win_rate_row, wilson_interval


def test_wilson_interval_bounds():
    lo, hi = wilson_interval(7.5, 10)
    assert 0.0 <= lo <= hi <= 1.0


def test_wilson_interval_monotonicity():
    low_lo, low_hi = wilson_interval(2.5, 10)
    high_lo, high_hi = wilson_interval(7.5, 10)
    assert high_lo > low_lo
    assert high_hi > low_hi


def test_build_win_rate_row_contains_ci_fields():
    row = build_win_rate_row("minimax4", "random", 8, 1, 1)

    assert row["games"] == 10
    assert "ci_low" in row
    assert "ci_high" in row


def test_elo_update_is_zero_sum():
    rating_a, rating_b = update_elo(1500.0, 1500.0, 1.0)
    assert pytest.approx(rating_a + rating_b) == 3000.0


def test_compute_elo_ratings_orders_stronger_agent_above_weaker_agent():
    game_records = [
        GameRecord(i, "strong", "weak", "strong", False, 40 + i)
        for i in range(5)
    ] + [
        GameRecord(5 + i, "weak", "strong", "strong", False, 45 + i)
        for i in range(5)
    ]

    ratings = compute_elo_ratings(game_records)

    assert ratings["strong"] > ratings["weak"]
