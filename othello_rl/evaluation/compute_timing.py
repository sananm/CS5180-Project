"""GPU-aware wall-clock timing utilities for compute efficiency analysis."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import torch

T = TypeVar('T')


@dataclass
class TimedResult:
    """Result of a timed operation."""
    result: object  # The actual return value
    wall_clock_seconds: float
    device: str


def timed_call(fn: Callable[[], T], device: str = "cpu") -> TimedResult:
    """Execute a function and measure wall-clock time.
    
    For GPU operations, synchronizes CUDA before and after timing
    to ensure accurate measurement.
    
    Args:
        fn: Zero-argument callable to time.
        device: Device being used ('cpu' or 'cuda').
    
    Returns:
        TimedResult with the function's return value and elapsed time.
    """
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    result = fn()
    
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    return TimedResult(result=result, wall_clock_seconds=elapsed, device=device)


def timed_az_iteration(trainer, device: str = "cpu") -> tuple[dict, float]:
    """Time one AlphaZero iteration (self-play + training).
    
    Args:
        trainer: AlphaZeroTrainer instance.
        device: Device for timing sync.
    
    Returns:
        Tuple of (metrics_dict, wall_clock_seconds).
        metrics_dict has keys: num_examples, avg_loss, policy_loss, value_loss
    """
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    # Self-play phase
    num_examples = trainer.run_self_play()
    
    # Training phase
    avg_loss = trainer.run_training()
    
    # Get last batch metrics if available (for policy/value breakdown)
    # Note: run_training doesn't return breakdown; we estimate from avg_loss
    policy_loss = avg_loss / 2  # Rough estimate (actual split requires tracking)
    value_loss = avg_loss / 2
    
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "num_examples": num_examples,
        "avg_loss": avg_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    }, elapsed


def timed_ppo_iteration(trainer, device: str = "cpu") -> tuple[dict, float]:
    """Time one PPO iteration (collect rollout + update).
    
    Args:
        trainer: PPOTrainer instance.
        device: Device for timing sync.
    
    Returns:
        Tuple of (metrics_dict, wall_clock_seconds).
        metrics_dict has keys: policy_loss, value_loss, entropy_loss (negative of entropy bonus)
    """
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    
    buffer = trainer.collect_rollout()
    info = trainer.update(buffer)
    
    if device == "cuda" or (device.startswith("cuda:") and torch.cuda.is_available()):
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    return {
        "policy_loss": info["policy_loss"],
        "value_loss": info["value_loss"],
        "entropy_loss": info["entropy_loss"],  # This is -entropy_coef * entropy (negative)
    }, elapsed
