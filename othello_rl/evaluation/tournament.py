from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.config.default import TOURNAMENT_GAMES_PER_MATCHUP
from othello_rl.evaluation.arena import Arena, GameRecord


@dataclass
class TournamentRow:
    agent_a: str
    agent_b: str
    wins_a: int
    wins_b: int
    draws: int
    games_played: int


@dataclass
class TournamentResult:
    rows: list[TournamentRow]
    game_records: list[GameRecord]


class Tournament:
    """Balanced round-robin tournaments across a fixed agent set."""

    def __init__(
        self,
        arena: Arena | None = None,
        games_per_matchup: int = TOURNAMENT_GAMES_PER_MATCHUP,
    ):
        if games_per_matchup % 2 != 0:
            raise ValueError("games_per_matchup must be even for balanced colors")
        self.arena = arena or Arena()
        self.games_per_matchup = games_per_matchup

    def run_round_robin(self, agents: dict[str, BaseAgent]) -> TournamentResult:
        rows: list[TournamentRow] = []
        game_records: list[GameRecord] = []

        for (name_a, agent_a), (name_b, agent_b) in combinations(agents.items(), 2):
            match = self.arena.play_match(
                agent_a,
                agent_b,
                num_games=self.games_per_matchup,
                agent_a_label=name_a,
                agent_b_label=name_b,
            )
            rows.append(
                TournamentRow(
                    agent_a=name_a,
                    agent_b=name_b,
                    wins_a=match.wins_a,
                    wins_b=match.wins_b,
                    draws=match.draws,
                    games_played=match.games_played,
                )
            )
            game_records.extend(match.records)

        return TournamentResult(rows=rows, game_records=game_records)
