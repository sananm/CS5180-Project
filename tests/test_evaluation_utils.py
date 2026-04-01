"""Tests for Phase 6 evaluation utilities."""
import tempfile
import os
import pytest
from othello_rl.evaluation.significance import binomial_test, SignificanceResult
from othello_rl.evaluation.color_analysis import (
    disaggregate_by_color, ColorAnalysis, ColorStats
)
from othello_rl.evaluation.loss_logger import LossLogger, LossRow, LOSS_CSV_COLUMNS
from othello_rl.evaluation.arena import GameRecord


class TestBinomialTest:
    """Tests for statistical significance testing."""

    def test_significant_result(self):
        """Strong win dominance should be significant."""
        result = binomial_test(wins_a=70, wins_b=30, draws=0)
        assert isinstance(result, SignificanceResult)
        assert result.decisive_games == 100
        assert result.p_value < 0.05
        assert result.significant_at_05 == True

    def test_not_significant_close(self):
        """50-50 split should not be significant."""
        result = binomial_test(wins_a=50, wins_b=50, draws=0)
        assert result.p_value > 0.9  # Should be ~1.0
        assert result.significant_at_05 == False

    def test_with_draws(self):
        """Draws should be excluded from decisive games."""
        result = binomial_test(wins_a=40, wins_b=40, draws=20)
        assert result.decisive_games == 80
        assert result.draws == 20

    def test_zero_decisive(self):
        """All draws should return p=1.0."""
        result = binomial_test(wins_a=0, wins_b=0, draws=10)
        assert result.p_value == 1.0
        assert result.significant_at_05 == False

    def test_ci_bounds(self):
        """Confidence interval should be in [0, 1]."""
        result = binomial_test(wins_a=60, wins_b=40, draws=0)
        assert 0 <= result.ci_low <= result.ci_high <= 1


class TestColorAnalysis:
    """Tests for color asymmetry disaggregation."""

    def test_basic_disaggregation(self):
        """Should correctly split wins by color."""
        records = [
            GameRecord(0, 'AlphaZero', 'PPO', 'AlphaZero', False, 30),
            GameRecord(1, 'PPO', 'AlphaZero', 'AlphaZero', False, 32),
            GameRecord(2, 'AlphaZero', 'PPO', 'PPO', False, 28),
            GameRecord(3, 'PPO', 'AlphaZero', 'PPO', False, 35),
        ]
        analysis = disaggregate_by_color(records, 'AlphaZero')
        
        assert analysis.agent_name == 'AlphaZero'
        assert analysis.as_black.games == 2
        assert analysis.as_black.wins == 1
        assert analysis.as_white.games == 2
        assert analysis.as_white.wins == 1

    def test_draws_handled(self):
        """Draws should be counted correctly."""
        records = [
            GameRecord(0, 'A', 'B', None, True, 60),  # Draw
            GameRecord(1, 'B', 'A', 'A', False, 30),  # A wins as white
        ]
        analysis = disaggregate_by_color(records, 'A')
        
        assert analysis.as_black.draws == 1
        assert analysis.as_black.wins == 0
        assert analysis.as_white.wins == 1

    def test_empty_records(self):
        """Empty records should return zero stats."""
        analysis = disaggregate_by_color([], 'A')
        assert analysis.as_black.games == 0
        assert analysis.as_white.games == 0

    def test_win_rate_calculation(self):
        """Win rate should be score / games."""
        records = [
            GameRecord(0, 'A', 'B', 'A', False, 30),
            GameRecord(1, 'A', 'B', None, True, 30),  # Draw
        ]
        analysis = disaggregate_by_color(records, 'A')
        # 1 win + 0.5 draw = 1.5 score / 2 games = 0.75
        assert analysis.as_black.win_rate == 0.75


class TestLossLogger:
    """Tests for loss/entropy CSV logging."""

    def test_creates_file_with_header(self):
        """First log should create file with header."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'loss.csv')
            logger = LossLogger(path)
            logger.log(LossRow(1, 'PPO', 0.5, 0.3, 1.2, 0.8, 10.5))
            
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2  # Header + 1 row
            assert 'iteration' in lines[0]
            assert 'algorithm' in lines[0]
            assert 'entropy' in lines[0]

    def test_appends_without_header(self):
        """Subsequent logs should append without header."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'loss.csv')
            logger = LossLogger(path)
            logger.log(LossRow(1, 'AZ', 0.5, 0.3, 1.2, 0.8, 10.5))
            logger.log(LossRow(2, 'AZ', 0.4, 0.2, 1.1, 0.6, 11.0))
            
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 3  # Header + 2 rows

    def test_creates_parent_dirs(self):
        """Should create parent directories if needed."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'nested', 'dir', 'loss.csv')
            logger = LossLogger(path)
            logger.log(LossRow(1, 'PPO', 0.5, 0.3, 1.2, 0.8, 10.5))
            assert os.path.exists(path)

    def test_csv_columns_match(self):
        """CSV columns should match LOSS_CSV_COLUMNS constant."""
        expected = ['iteration', 'algorithm', 'policy_loss', 'value_loss',
                    'entropy', 'total_loss', 'wall_clock_seconds']
        assert LOSS_CSV_COLUMNS == expected
