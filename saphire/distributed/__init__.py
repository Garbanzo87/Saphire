"""Distributed rollouts, remote environment/reward server, and external RL trainer integrations (verl, OpenRLHF)."""
from .rollouts import RolloutEngine, evaluate_distributed

__all__ = ["RolloutEngine", "evaluate_distributed"]
