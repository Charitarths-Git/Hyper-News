"""
Legacy compatibility wrapper.

Keep older imports working while routing all bandit logic through the
maintained implementation in backend.bandit.
"""

from bandit import LinUCBBandit, NeuralContextualBandit

__all__ = ["LinUCBBandit", "NeuralContextualBandit"]
