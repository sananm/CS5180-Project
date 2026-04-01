import numpy as np
import pytest

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.evaluation.tournament import Tournament


class FirstLegalAgent(BaseAgent):
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_action(self, board: np.ndarray, valid_moves: np.ndarray) -> int:
        return int(np.flatnonzero(valid_moves)[0])


def test_round_robin_returns_one_row_per_unordered_pair():
    agents = {
        "alpha": FirstLegalAgent("alpha"),
        "beta": FirstLegalAgent("beta"),
        "gamma": FirstLegalAgent("gamma"),
    }

    result = Tournament(games_per_matchup=2).run_round_robin(agents)

    assert len(result.rows) == 3
    assert {frozenset((row.agent_a, row.agent_b)) for row in result.rows} == {
        frozenset(("alpha", "beta")),
        frozenset(("alpha", "gamma")),
        frozenset(("beta", "gamma")),
    }


def test_each_row_totals_requested_games_per_matchup():
    agents = {
        "alpha": FirstLegalAgent("alpha"),
        "beta": FirstLegalAgent("beta"),
        "gamma": FirstLegalAgent("gamma"),
    }

    result = Tournament(games_per_matchup=4).run_round_robin(agents)

    assert all(row.games_played == 4 for row in result.rows)
    assert all((row.wins_a + row.wins_b + row.draws) == 4 for row in result.rows)


def test_color_balancing_in_game_records():
    agents = {
        "alpha": FirstLegalAgent("alpha"),
        "beta": FirstLegalAgent("beta"),
    }

    result = Tournament(games_per_matchup=4).run_round_robin(agents)
    pair_records = [
        record
        for record in result.game_records
        if {record.black_agent, record.white_agent} == {"alpha", "beta"}
    ]

    assert len(pair_records) == 4
    assert sum(record.black_agent == "alpha" for record in pair_records) == 2
    assert sum(record.white_agent == "alpha" for record in pair_records) == 2
    assert sum(record.black_agent == "beta" for record in pair_records) == 2
    assert sum(record.white_agent == "beta" for record in pair_records) == 2


def test_tournament_rejects_odd_games_per_matchup():
    with pytest.raises(ValueError):
        Tournament(games_per_matchup=3)
