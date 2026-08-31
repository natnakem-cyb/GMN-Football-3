"""
GMN-Football Python Training Package
Provides Gymnasium and PettingZoo environments, action mapping, and training pipelines.
"""

from training.gmn_gym import GMNFootballEnv
from training.gmn_pettingzoo import GMNMultiAgentEnv

__all__ = ["GMNFootballEnv", "GMNMultiAgentEnv"]
