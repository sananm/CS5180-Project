from __future__ import annotations

from othello_rl.config.default import ELO_INITIAL_RATING, ELO_K_FACTOR
from othello_rl.evaluation.arena import GameRecord


def expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def update_elo(
    rating_a: float,
    rating_b: float,
    score_a: float,
    k_factor: float = ELO_K_FACTOR,
) -> tuple[float, float]:
    expected_a = expected_score(rating_a, rating_b)
    delta = k_factor * (score_a - expected_a)
    return (rating_a + delta, rating_b - delta)


def compute_elo_ratings(
    game_records: list[GameRecord],
    initial_rating: float = ELO_INITIAL_RATING,
    k_factor: float = ELO_K_FACTOR,
) -> dict[str, float]:
    ratings: dict[str, float] = {}

    for record in game_records:
        ratings.setdefault(record.black_agent, float(initial_rating))
        ratings.setdefault(record.white_agent, float(initial_rating))

        if record.is_draw or record.winner is None:
            score_black = 0.5
        elif record.winner == record.black_agent:
            score_black = 1.0
        elif record.winner == record.white_agent:
            score_black = 0.0
        else:
            raise ValueError(f"Winner {record.winner!r} does not match game record")

        ratings[record.black_agent], ratings[record.white_agent] = update_elo(
            ratings[record.black_agent],
            ratings[record.white_agent],
            score_black,
            k_factor=k_factor,
        )

    return dict(sorted(ratings.items(), key=lambda item: (-item[1], item[0])))
