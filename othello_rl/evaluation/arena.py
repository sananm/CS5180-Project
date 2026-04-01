from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from othello_rl.agents.base_agent import BaseAgent
from othello_rl.game.othello_env import OthelloEnv
from othello_rl.config.default import ARENA_MAX_MOVES


@dataclass
class GameRecord:
    game_index: int
    black_agent: str
    white_agent: str
    winner: str | None
    is_draw: bool
    move_count: int


@dataclass
class MatchResult:
    agent_a: str
    agent_b: str
    wins_a: int
    wins_b: int
    draws: int
    games_played: int
    records: list[GameRecord]


class Arena:
    """Runs complete Othello games between two agents."""

    def __init__(
        self,
        env_factory: Callable[[], OthelloEnv] | None = None,
        max_moves: int = ARENA_MAX_MOVES,
    ):
        self.env_factory = env_factory or OthelloEnv
        self.max_moves = max_moves

    def play_game(
        self,
        agent_black: BaseAgent,
        agent_white: BaseAgent,
        game_index: int = 0,
        black_label: str | None = None,
        white_label: str | None = None,
    ) -> GameRecord:
        env = self.env_factory()
        env.reset()

        black_label = black_label or agent_black.name
        white_label = white_label or agent_white.name

        if hasattr(agent_black, "reset"):
            agent_black.reset()
        if hasattr(agent_white, "reset"):
            agent_white.reset()

        move_count = 0
        while move_count < self.max_moves:
            current_agent = agent_black if env.player == 1 else agent_white
            canonical = env.game.getCanonicalForm(env.board, env.player)
            valid_moves = env.get_valid_actions()
            action = current_agent.get_action(canonical, valid_moves)
            _, reward, done, _ = env.step(action)
            move_count += 1

            if done:
                winner = None
                is_draw = reward == 0.0
                if not is_draw:
                    winner = black_label if current_agent is agent_black else white_label
                return GameRecord(
                    game_index=game_index,
                    black_agent=black_label,
                    white_agent=white_label,
                    winner=winner,
                    is_draw=is_draw,
                    move_count=move_count,
                )

        raise RuntimeError(f"Game exceeded {self.max_moves} moves without terminating")

    def play_match(
        self,
        agent_a: BaseAgent,
        agent_b: BaseAgent,
        num_games: int,
        agent_a_label: str | None = None,
        agent_b_label: str | None = None,
    ) -> MatchResult:
        agent_a_label = agent_a_label or agent_a.name
        agent_b_label = agent_b_label or agent_b.name

        records: list[GameRecord] = []
        wins_a = wins_b = draws = 0

        for game_index in range(num_games):
            if game_index % 2 == 0:
                record = self.play_game(
                    agent_a,
                    agent_b,
                    game_index=game_index,
                    black_label=agent_a_label,
                    white_label=agent_b_label,
                )
            else:
                record = self.play_game(
                    agent_b,
                    agent_a,
                    game_index=game_index,
                    black_label=agent_b_label,
                    white_label=agent_a_label,
                )

            records.append(record)

            if record.is_draw:
                draws += 1
            elif record.winner == agent_a_label:
                wins_a += 1
            elif record.winner == agent_b_label:
                wins_b += 1
            else:
                raise RuntimeError(f"Unexpected winner label: {record.winner}")

        return MatchResult(
            agent_a=agent_a_label,
            agent_b=agent_b_label,
            wins_a=wins_a,
            wins_b=wins_b,
            draws=draws,
            games_played=num_games,
            records=records,
        )
